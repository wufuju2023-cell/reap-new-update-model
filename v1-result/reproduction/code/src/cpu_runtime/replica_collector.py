"""Collector routing to two independent GPU processes, fixed release by default.

An explicit owning-learner publication provider enables latest-at-reservation
mode: the pool contract stays fixed while each attempt pins its own release.
Only a newly admitted actor on an idle endpoint loads the newer parameters.

The router only holds its lock for local admission/receipts. Each process keeps
its existing serial actor, model, private session RNG and tokenwise inference.
The trusted search adapter must pass the supplied endpoint URL AND release to run_collector (or
its isolated CPU-container wrapper). This module never launches a GPU process,
changes release heads, retries a mutation, or declares a proof accepted.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
import threading
from urllib.parse import urlsplit

from . import matchmaker as mm
from .collector_batch import _known_search, RETIRE_SNAPSHOT, validate_retirement, run_collector_batch
from .http_clients import GpuHttpClient
from .transport_budget import DEFAULT_BUDGET
from .verified_dataset_store import safe_directory


class ReplicaRoutingBlocked(mm.MatchmakerBlocked):
    pass


def _identifier(value, label):
    mm.require(type(value) is str and mm.ID.fullmatch(value), "invalid "+label)


def validate_endpoints(endpoints):
    mm.require(type(endpoints) is list and len(endpoints) == 2, "exactly two independent endpoints required")
    for item in endpoints:
        mm._fields(item, "replica_id base_url deployment_id model_release_sha256", "replica endpoint")
        for key in ("replica_id", "deployment_id"):
            _identifier(item[key], key)
        mm._sha(item["model_release_sha256"], "fixed release")
        mm.require(type(item["base_url"]) is str, "endpoint URL required")
        url = urlsplit(item["base_url"])
        mm.require(url.scheme in ("http", "https") and url.hostname and url.port is not None
            and url.username is None and url.password is None and not url.query and not url.fragment
            and url.path == "" and item["base_url"] == f"{url.scheme}://{url.netloc}",
            "endpoint must be an explicit port URL without path/credentials")
    for key in ("replica_id", "base_url", "deployment_id"):
        mm.require(len({item[key] for item in endpoints}) == 2, "replicas require distinct "+key)
    mm.require(len({item["model_release_sha256"] for item in endpoints}) == 1,
               "this fixed collector pool requires one immutable release")
    return deepcopy(endpoints)


def validate_intent(intent):
    mm._fields(intent, "schema_version proposal attempt_source intent_sha256", "reserved intent")
    mm.require(intent["schema_version"] == "reap.matchmaker.intent.v1", "reserved Matchmaker intent required")
    base = {key: value for key, value in intent.items() if key != "intent_sha256"}
    mm.require(intent["intent_sha256"] == mm.content_sha256(base), "intent content pin mismatch")
    mm.require(type(intent["proposal"]) is dict, "proposal required")
    _identifier(intent["proposal"].get("attempt_id"), "attempt id")
    mm._sha(intent["proposal"].get("model_release_sha256"), "attempt release")


def _audit_staging(directory, expected):
    names = set(os.listdir(directory.path))
    mm.require(names - {name for name in names if mm.STAGING.fullmatch(name)} == set(expected),
               "incomplete/extra router records; reconcile without retry")
    committed = [directory.read(name) for name in expected]
    for name in names:
        if mm.STAGING.fullmatch(name):
            mm.require(directory.read(name) in committed, "partial/orphan router staging; no automatic repair")


class ReplicaCollector:
    def __init__(self, root, endpoints, search_callback, *, http_timeout=DEFAULT_BUDGET.client, client_factory=GpuHttpClient,
                 publication_provider=None):
        self.endpoints = validate_endpoints(endpoints)
        mm.require(callable(search_callback), "search callback required")
        self.publication_provider = publication_provider
        self._release_cache = {}
        self._publication_identity = None
        if publication_provider is not None:
            from .released_attempts import ReleasedAttemptProvider
            mm.require(isinstance(publication_provider, ReleasedAttemptProvider), "owning-learner release provider required")
            learner = publication_provider.learner
            self._publication_identity = {"provider":deepcopy(publication_provider.identity),
                "contract_sha256":mm.content_sha256(learner.run["contract"])}
            # Endpoint bootstrap metadata may name a compatible source release,
            # but never becomes a publication of this new learner run.
            baseline, _ = learner.store.load_release(self.endpoints[0]["model_release_sha256"])
            mm.require(mm.canonical_bytes(baseline["contract"]) == mm.canonical_bytes(learner.run["contract"]),
                       "endpoint bootstrap contract differs from learner contract")
        self.callback = search_callback
        self.clients = {item["replica_id"]: client_factory(item["base_url"], timeout_seconds=http_timeout)
                        for item in self.endpoints}
        self.lock = threading.RLock(); self.stack = ExitStack(); self.lease = None
        self.blocked = False; self.closed = False; self.routes = {}; self.active = {}; self.retiring = set()
        self.identity = {"schema_version":"reap.replica-collector.pool.v1", "endpoints":self.endpoints,
                         "per_endpoint_capacity":1, "release_policy":"fixed_for_pool_and_attempt",
                         "retry_or_failover":False}
        if publication_provider is not None:
            self.identity = {**self.identity, "schema_version":"reap.replica-collector.pool.v2",
                "release_policy":"latest-published-at-reservation;fixed-for-attempt",
                "publication":deepcopy(self._publication_identity)}
        try:
            path = Path(root).absolute()
            parent = self.stack.enter_context(safe_directory(path.parent))
            try:
                self.root = self.stack.enter_context(parent.child(path.name, create=True)); fresh = True
            except FileExistsError:
                self.root = self.stack.enter_context(parent.child(path.name)); fresh = False
            self.lease = mm._Lease(self.root)
            if fresh:
                self.attempts = self.stack.enter_context(self.root.child("attempts", create=True))
                mm._publish(self.root, "pool.json", self.identity); parent.sync(); self.root.sync()
            else:
                mm.require(mm.canonical_bytes(mm._read(self.root,"pool.json")) == mm.canonical_bytes(self.identity),
                           "pool/deployment identity changed; no route migration")
                names = set(os.listdir(self.root.path))
                stages = {name for name in names if mm.STAGING.fullmatch(name)}
                mm.require(names - stages == {"pool.json", "attempts", ".writer.lock"},
                           "unexpected pool record; no automatic recovery")
                for name in stages:
                    mm.require(self.root.read(name) == self.root.read("pool.json"), "orphan pool publication")
                self.attempts = self.stack.enter_context(self.root.child("attempts"))
                self._restore()
        except BaseException:
            self.close(); raise

    def _restore(self):
        for aid in sorted(os.listdir(self.attempts.path)):
            _identifier(aid,"saved attempt")
            with self.attempts.child(aid) as directory:
                _audit_staging(directory, {"route.json","search.json","retirement-intent.json","retirement.json"})
                route = mm._read(directory,"route.json")
                mm._fields(route,"schema_version endpoint intent","route")
                mm.require(route["schema_version"] == "reap.replica-collector.route.v1", "route schema differs")
                validate_intent(route["intent"])
                mm.require(route["intent"]["proposal"]["attempt_id"] == aid and self._endpoint_registered(route["endpoint"]),
                           "saved route identity differs")
                mm.require(route["endpoint"]["model_release_sha256"] == route["intent"]["proposal"]["model_release_sha256"],
                           "saved route release differs")
                self._validate_release(route["endpoint"]["model_release_sha256"])
                response = mm._read(directory,"search.json"); _known_search(route["intent"], response)
                expected = self._retirement_intent(route,response)
                mm.require(mm.canonical_bytes(mm._read(directory,"retirement-intent.json")) == mm.canonical_bytes(expected),
                           "retirement request changed")
                validate_retirement(mm._read(directory,"retirement.json"),route["intent"])
                self.routes[aid] = route

    def _endpoint_registered(self, endpoint):
        if self.publication_provider is None:
            return endpoint in self.endpoints
        without_release = lambda value: {k:v for k,v in value.items() if k != "model_release_sha256"}
        return (type(endpoint) is dict and set(endpoint) == set(self.endpoints[0])
                and any(without_release(endpoint) == without_release(item) for item in self.endpoints))

    def _validate_release(self, pin):
        """Read immutable CAS once per version, then its small durable confirmation.

        Reservation freshness is provided by the same owning provider under its
        learner lock. A previously reserved R1 remains valid after R2 publishes.
        This method never trains, publishes, selects a new head or touches actors.
        """
        if self.publication_provider is None:
            mm.require(pin == self.endpoints[0]["model_release_sha256"],
                       "new release requires an explicitly new pool, never mutate old actors")
            return
        learner = self.publication_provider.learner
        from gpu_runtime.learner import _read_control
        from gpu_runtime.release_head import _validate_confirmation
        release = self._release_cache.get(pin)
        if release is None:
            release, _ = learner.store.load_release(pin)
            mm.require(release["run_sha256"] == learner.run_sha,
                       "release must be published by the owning learner run, not bootstrap or another run")
            mm.require(mm.content_sha256(release["contract"]) == self._publication_identity["contract_sha256"],
                       "new release changes the fixed pool contract")
            self._release_cache[pin] = deepcopy(release)
        step = release["source"]["learner_step"]
        mm._int(step, 1, 10**10, "released learner step")
        confirmation = _read_control(learner.control/"release-head"/f"published-{step:08d}.json")
        intent = {k:confirmation[k] for k in ("schema_version","run_sha256","step","checkpoint_sha256","previous_head_sha256")}
        _validate_confirmation(learner.store, learner.run_sha, intent, confirmation, verify_release=False)
        mm.require(confirmation["run_sha256"] == learner.run_sha and confirmation["step"] == step
            and confirmation["checkpoint_sha256"] == release["source"]["checkpoint_sha256"]
            and confirmation["model_release_sha256"] == pin, "release lacks its exact committed publication confirmation")

    def _write(self, aid, name, value):
        try:
            mm.require(len(mm.canonical_bytes(value)) <= 1024*1024, "router record too large")
            with self.attempts.child(aid) as directory:
                mm._publish(directory,name,value)
        except BaseException:
            self.blocked = True; raise

    def _fail(self, aid, phase, error):
        with self.lock:
            self.blocked = True
            # The original route alone already blocks recovery if this write fails.
            self._write(aid,"unknown.json",{"phase":phase,"exception_type":type(error).__name__,
                                           "mutation_retry_allowed":False})

    def search(self, intent):
        validate_intent(intent); aid = intent["proposal"]["attempt_id"]
        # Large CAS validation is outside the router admission lock. Existing
        # actors can search/retire while the next release is validated.
        self._validate_release(intent["proposal"]["model_release_sha256"])
        with self.lock:
            mm.require(not self.closed and not self.blocked, "router closed/unknown; no new dispatch")
            mm.require(aid not in self.routes, "attempt already routed; never repeat search")
            occupied = set(self.active.values())
            endpoint = next((e for e in self.endpoints if e["replica_id"] not in occupied), None)
            mm.require(endpoint is not None, "both replica slots resident; wait for confirmed retirement")
            endpoint = deepcopy(endpoint)
            endpoint["model_release_sha256"] = intent["proposal"]["model_release_sha256"]
            route = {"schema_version":"reap.replica-collector.route.v1",
                     "endpoint":deepcopy(endpoint),"intent":deepcopy(intent)}
            try:
                with self.attempts.child(aid,create=True) as directory:
                    mm._publish(directory,"route.json",route); self.attempts.sync()
            except BaseException:
                self.blocked = True; raise
            self.routes[aid] = route; self.active[aid] = endpoint["replica_id"]
        try:
            response = deepcopy(self.callback(deepcopy(intent),deepcopy(endpoint)))
            _known_search(intent,response)
            with self.lock:
                self._write(aid,"search.json",response)
            return response
        except Exception as exc:
            self._fail(aid,"search",exc); raise

    @staticmethod
    def _retirement_intent(route, response):
        return {"schema_version":"reap.replica-collector.retirement-intent.v1",
                "route_sha256":mm.content_sha256(route),"search_sha256":mm.content_sha256(response),
                "request":{"session_id":route["intent"]["proposal"]["attempt_id"],
                           "name":RETIRE_SNAPSHOT,"expected_policy_version":0}, "mutation_retry_allowed":False}

    def retire(self, intent, response):
        validate_intent(intent); aid = intent["proposal"]["attempt_id"]
        with self.lock:
            mm.require(not self.closed and aid in self.active and aid not in self.retiring, "no active single-submit route")
            route = self.routes[aid]
            mm.require(mm.canonical_bytes(route["intent"]) == mm.canonical_bytes(intent), "retire intent changed")
            _known_search(intent,response)
            with self.attempts.child(aid) as directory:
                mm.require(mm.canonical_bytes(mm._read(directory,"search.json")) == mm.canonical_bytes(response),
                           "retire search receipt changed")
            self.retiring.add(aid)
            self._write(aid,"retirement-intent.json",self._retirement_intent(route,response))
            client = self.clients[route["endpoint"]["replica_id"]]
        try:
            receipt = client.retire_session(aid,RETIRE_SNAPSHOT,expected_policy_version=0)
            validate_retirement(receipt,intent)
            with self.lock:
                self._write(aid,"retirement.json",receipt)
                del self.active[aid]
            return receipt
        except Exception as exc:
            self._fail(aid,"retire",exc); raise

    def close(self):
        with self.lock:
            self.closed = True
            try:
                if self.lease is not None:
                    self.lease.close(); self.lease = None
            finally:
                self.stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def run_replica_collector_batch(scheduler, output, *, router, prepare_source, verify,
                                validation_workers=1, max_pending_validation=2):
    """Use the existing admission, retirement, bounded validation and proof gates."""
    mm.require(isinstance(router,ReplicaCollector), "explicit replica router required")
    return run_collector_batch(scheduler,output,prepare_source=prepare_source,
        provide_release=lambda:router.endpoints[0]["model_release_sha256"],search=router.search,
        retire=router.retire,verify=verify,search_workers=2,validation_workers=validation_workers,
        max_pending_validation=max_pending_validation,
        reserve_attempt=router.publication_provider)
