"""Bounded callback executor for the finite Matchmaker.

Callbacks are trusted local adapters, not evidence of mathematical verification.
Two search-stage slots may feed one verifier. Validation capacity includes its
running job; a completed search remains in its search-stage slot while handoff
is blocked. GPU operations still belong to the runtime's serial actor.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import os
import threading
import time

from . import matchmaker as mm
from .verified_dataset_store import safe_directory

RETIRE_SNAPSHOT = "retired-collector"


class CollectorBatchBlocked(mm.MatchmakerBlocked):
    pass


def _same(left, right):
    return mm.canonical_bytes(left) == mm.canonical_bytes(right)


def _known_search(intent, response):
    mm._fields(response, "schema_version attempt_id intent_sha256 outcome artifacts_complete artifacts_sha256 "
               "terminal_receipt_sha256 actual_steps policy_version raw", "collector search receipt")
    proposal = intent["proposal"]
    mm.require(response["schema_version"] == "reap.collector-batch.search.v1", "invalid search receipt schema")
    mm.require(response["attempt_id"] == proposal["attempt_id"] and response["intent_sha256"] == intent["intent_sha256"],
               "search receipt identity mismatch")
    mm.require(response["outcome"] in ("solved_candidate", "exhausted") and response["artifacts_complete"] is True,
               "search terminal/artifact completeness is unknown")
    mm._sha(response["artifacts_sha256"], "search artifacts")
    mm._sha(response["terminal_receipt_sha256"], "terminal receipt")
    mm._int(response["actual_steps"], 0, proposal["budget_steps"], "search actual_steps")
    mm.require(type(response["policy_version"]) is int and response["policy_version"] == 0,
               "collector must retain fixed local policy version zero")
    mm.require(type(response["raw"]) is dict, "raw collector receipt must be an object")
    mm.canonical_bytes(response)


def retirement_intent(intent, search_receipt):
    return {"schema_version": "reap.collector-batch.retirement-intent.v1",
        "attempt_id": intent["proposal"]["attempt_id"], "intent_sha256": intent["intent_sha256"],
        "search_receipt_sha256": mm.content_sha256(search_receipt),
        "request": {"session_id": intent["proposal"]["attempt_id"], "snapshot": RETIRE_SNAPSHOT,
                    "expected_policy_version": 0}, "mutation_retry_allowed": False}


def validate_retirement(receipt, intent):
    mm._fields(receipt, "schema_version session_id policy_version snapshot snapshot_sha256 status mutation_retry_allowed "
               "tombstone_scope", "collector retirement receipt")
    mm.require(receipt["schema_version"] == "reap.gpu.retirement.v1" and receipt["status"] == "released"
        and receipt["session_id"] == intent["proposal"]["attempt_id"] and receipt["snapshot"] == RETIRE_SNAPSHOT
        and type(receipt["policy_version"]) is int and receipt["policy_version"] == 0
        and receipt["mutation_retry_allowed"] is False and receipt["tombstone_scope"] == "current_runtime",
        "retirement release is unconfirmed or belongs to a different attempt")
    mm._sha(receipt["snapshot_sha256"], "retirement snapshot")


class _Output:
    def __init__(self, path, identity):
        self.stack = ExitStack(); self.lease = None
        try:
            path = Path(path).absolute()
            parent = self.stack.enter_context(safe_directory(path.parent))
            try:
                self.root = self.stack.enter_context(parent.child(path.name, create=True))
                fresh = True
            except FileExistsError:
                self.root = self.stack.enter_context(parent.child(path.name))
                fresh = False
            self.lease = mm._Lease(self.root)
            if fresh:
                self.attempts = self.stack.enter_context(self.root.child("attempts", create=True))
                mm._publish(self.root, "batch.json", identity)
                parent.sync(); self.root.sync()
            else:
                mm.require(_same(mm._read(self.root, "batch.json"), identity), "batch run/config identity mismatch")
                self.attempts = self.stack.enter_context(self.root.child("attempts"))
                mm.require(not any(name.startswith("dispatch-failure-") for name in os.listdir(self.root.path)),
                           "previous dispatch/preparation failure requires reconciliation")
            self.identity = identity
        except BaseException:
            self.close(); raise

    def close(self):
        try:
            if self.lease:
                self.lease.close()
        finally:
            self.stack.close()

    def create_attempt(self, intent):
        with self.attempts.child(intent["proposal"]["attempt_id"], create=True) as directory:
            mm._publish(directory, "intent.json", intent)
            self.attempts.sync()

    def write(self, aid, name, value):
        mm.require(len(mm.canonical_bytes(value)) <= 1024*1024, "oversized callback receipt; store large evidence separately")
        with self.attempts.child(aid) as directory:
            mm._publish(directory, name, value)

    def audit_completed(self, state):
        mm.require(set(os.listdir(self.attempts.path)) == set(state["attempts"]), "missing/extra attempt output directory")
        for aid, item in state["attempts"].items():
            mm.require(item["result"] is not None, "pending attempt cannot be automatically restarted")
            with self.attempts.child(aid) as directory:
                mm.require(_same(mm._read(directory, "intent.json"), item["intent"]), "saved intent mismatch")
                if "publication_selection" in self.identity:
                    from .released_attempts import validate_selection
                    validate_selection(mm._read(directory, "publication-selection.json"), item["intent"],
                                       self.identity["publication_selection"]["learner_run_sha256"])
                search = mm._read(directory, "search-response.json"); _known_search(item["intent"], search)
                mm.require(_same(mm._read(directory, "retirement-intent.json"), retirement_intent(item["intent"], search)),
                           "saved retirement intent mismatch")
                retirement = mm._read(directory, "retirement-response.json"); validate_retirement(retirement, item["intent"])
                mm.require(_same(mm._read(directory, "retirement-receipt.json"), {
                    "intent_sha256": mm.content_sha256(retirement_intent(item["intent"], search)),
                    "receipt_sha256": mm.content_sha256(retirement)}), "saved retirement receipt mismatch")
                result = mm._read(directory, "terminal-result.json")
                mm.require(_same(result, item["result"]), "saved terminal result differs from scheduler")
                if result["outcome"] == "accepted":
                    mm.require(search["outcome"] == "solved_candidate" and
                               _same(mm._read(directory, "validation-response.json"), result),
                               "accepted result lacks matching independent validation response")
                else:
                    mm.require(_same(result, _exhausted(item["intent"], search)) and search["outcome"] == "exhausted",
                               "exhausted result/search mismatch")
                # Do not ignore incomplete or altered publication residue.
                for name in os.listdir(directory.path):
                    if mm.STAGING.fullmatch(name):
                        raw = directory.read(name)
                        mm.require(any(raw == directory.read(candidate) for candidate in os.listdir(directory.path)
                                       if not mm.STAGING.fullmatch(candidate)), "orphan publication requires reconciliation")


def _exhausted(intent, search):
    return {"schema_version": "reap.matchmaker.result.v1", "attempt_id": intent["proposal"]["attempt_id"],
        "intent_sha256": intent["intent_sha256"], "outcome": "exhausted",
        "evidence": {"terminal_receipt_sha256": search["terminal_receipt_sha256"], "actual_steps": search["actual_steps"]}}


def run_collector_batch(scheduler, output, *, prepare_source, provide_release, search, retire, verify,
                        search_workers=2, validation_workers=1, max_pending_validation=2, reserve_attempt=None):
    """Run a finite campaign; each mutation callback is submitted at most once.

    prepare_source(proposal) -> compiled descriptor accepted by Matchmaker.
    provide_release() -> one explicit immutable model-release SHA for a new job.
    reserve_attempt -> optional same-process ReleasedAttemptProvider; supersedes
        provide_release and binds a durable selection sidecar before search.
    search(intent) -> normalized terminal receipt (see _known_search).
    retire(intent, search_receipt) -> actual released runtime retirement.v1.
    verify(intent, search_receipt) -> complete accepted/unknown Matchmaker result.

    Callback exceptions are diagnostic types only, never exception text that may
    contain credentials. Unknowns block new dispatch but existing work drains.
    Restart with pending or unknown attempts never invokes any callback. Complete
    past attempts are audited and skipped, including proof validation/retirement.
    Adapters must impose their own safe I/O deadlines and report unknown on an
    uncertain timeout; this executor never force-kills a thread mid-mutation.
    """
    mm._int(search_workers, 1, 2, "search_workers")
    mm.require(type(validation_workers) is int and validation_workers == 1, "exactly one validation worker required")
    mm._int(max_pending_validation, 1, 2, "max_pending_validation")
    initial = scheduler.status()
    if initial["pending_attempts"] or initial["blocked"]:
        raise CollectorBatchBlocked("existing pending/unknown attempt; reconcile original receipts, never restart search")
    identity = {"schema_version": "reap.collector-batch.run.v1", "matchmaker_run_sha256": scheduler.run_sha256,
        "search_workers": search_workers, "validation_workers": validation_workers,
        "max_pending_validation": max_pending_validation,
        "pending_definition": "queued_plus_running_validation; blocked_handoff_retains_search_stage_slot"}
    if reserve_attempt is not None:
        from .released_attempts import ReleasedAttemptProvider
        mm.require(isinstance(reserve_attempt, ReleasedAttemptProvider), "explicit owning-learner release provider required")
        identity["publication_selection"] = deepcopy(reserve_attempt.identity)
    store = _Output(output, identity)
    stop = threading.Event(); metrics_lock = threading.Lock()
    metrics = {"max_search_stage_occupancy": 0, "max_validation_stage_occupancy": 0,
        "max_search_callbacks_active": 0, "max_validation_callbacks_active": 0, "handoff_backpressure_observed": False}
    active = {"search": 0, "validation": 0}; failures = []; accepted = []; exhausted = []; unknown = []
    started = []; searches = {}; validations = {}; queue = deque(); begin = time.perf_counter()

    def measured(kind, callback, *args):
        with metrics_lock:
            active[kind] += 1
            key = "max_"+kind+"_callbacks_active"
            metrics[key] = max(metrics[key], active[kind])
        try:
            return callback(*deepcopy(args))
        finally:
            with metrics_lock:
                active[kind] -= 1

    def diagnostic(intent, phase, exception):
        stop.set()
        aid = intent["proposal"]["attempt_id"]
        record = {"schema_version": "reap.collector-batch.diagnostic.v1", "attempt_id": aid,
            "intent_sha256": intent["intent_sha256"], "phase": phase,
            "exception_type": type(exception).__name__, "mutation_retry_allowed": False}
        # If this write also fails, the original scheduler intent still blocks
        # restart. Do not manufacture a receipt pointing to absent diagnostics.
        store.write(aid, "diagnostic.json", record)
        if phase.endswith("record") or (phase == "dispatch" and isinstance(exception, OSError)):
            reason = "storage_unknown"
        elif phase in {"search", "retire"} and isinstance(exception, (TimeoutError, ConnectionError)):
            reason = "transport_unknown"
        else:
            reason = "process_unknown"
        return {"schema_version": "reap.matchmaker.result.v1", "attempt_id": aid,
            "intent_sha256": intent["intent_sha256"], "outcome": "unknown",
            "evidence": {"diagnostic_sha256": mm.content_sha256(record), "reason": reason}}

    def execute_search(intent):
        aid = intent["proposal"]["attempt_id"]; phase = "search"
        try:
            response = deepcopy(measured("search", search, intent))
            phase = "search_record"; store.write(aid, "search-response.json", response)
            phase = "search_validation"; _known_search(intent, response)
            phase = "retirement_intent_record"
            retiring = retirement_intent(intent, response); store.write(aid, "retirement-intent.json", retiring)
            phase = "retire"; retired = deepcopy(retire(deepcopy(intent), deepcopy(response)))
            phase = "retirement_response_record"; store.write(aid, "retirement-response.json", retired)
            phase = "retirement_validation"; validate_retirement(retired, intent)
            phase = "retirement_receipt_record"
            store.write(aid, "retirement-receipt.json", {"intent_sha256": mm.content_sha256(retiring),
                                                        "receipt_sha256": mm.content_sha256(retired)})
            return {"search": response, "unknown": None}
        except Exception as exc:
            return {"search": None, "unknown": diagnostic(intent, phase, exc)}

    def execute_verification(intent, response):
        aid = intent["proposal"]["attempt_id"]; phase = "verify"
        try:
            result = deepcopy(measured("validation", verify, intent, response))
            phase = "validation_record"; store.write(aid, "validation-response.json", result)
            phase = "validation_result"
            mm.require(type(result) is dict and result.get("outcome") in ("accepted", "unknown"),
                       "verifier must return accepted or unknown, not search exhaustion")
            if result["outcome"] == "accepted":
                mm.require(type(result.get("evidence")) is dict and
                           result["evidence"].get("actual_steps") == response["actual_steps"],
                           "verified result/search step accounting mismatch")
            if result["outcome"] == "unknown":
                stop.set()
            return result
        except Exception as exc:
            return diagnostic(intent, phase, exc)

    def settle(intent, result):
        aid = intent["proposal"]["attempt_id"]
        try:
            # Validate before committing the local terminal claim, then let the
            # scheduler validate again while appending its sole event ledger.
            scheduler._validate_result(result)
        except Exception as exc:
            result = diagnostic(intent, "terminal_validation", exc)
        if result["outcome"] == "unknown":
            stop.set()
        store.write(aid, "terminal-result.json", result)
        scheduler.record_result(aid, result)
        {"accepted": accepted, "exhausted": exhausted, "unknown": unknown}[result["outcome"]].append(aid)

    def failed(intent, phase, exc):
        stop.set()
        failures.append({"attempt_id": intent["proposal"]["attempt_id"] if intent else None,
                         "phase": phase, "exception_type": type(exc).__name__, "mutation_retry_allowed": False})

    try:
        store.audit_completed(initial)
        with ThreadPoolExecutor(max_workers=search_workers, thread_name_prefix="collector-search") as search_pool, \
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="collector-verify") as validation_pool:
            while True:
                for future, intent in list(validations.items()):
                    if future.done():
                        del validations[future]
                        try:
                            settle(intent, future.result())
                        except Exception as exc:
                            failed(intent, "validation_settlement", exc)

                for future, intent in list(searches.items()):
                    if not future.done():
                        continue
                    try:
                        result = future.result()
                        if result["unknown"] is not None:
                            del searches[future]; settle(intent, result["unknown"])
                        elif result["search"]["outcome"] == "exhausted":
                            del searches[future]; settle(intent, _exhausted(intent, result["search"]))
                        elif len(queue)+len(validations) < max_pending_validation:
                            del searches[future]; queue.append((intent, result["search"]))
                        else:
                            metrics["handoff_backpressure_observed"] = True
                    except Exception as exc:
                        searches.pop(future, None); failed(intent, "search_settlement", exc)

                metrics["max_validation_stage_occupancy"] = max(metrics["max_validation_stage_occupancy"], len(queue)+len(validations))
                if queue and not validations:
                    intent, response = queue.popleft()
                    validations[validation_pool.submit(execute_verification, intent, response)] = intent

                while not stop.is_set() and len(searches) < search_workers:
                    intent = None
                    try:
                        selection = None
                        if reserve_attempt is None:
                            release = provide_release()
                            proposal = scheduler.plan_next(release)
                            if proposal is None:
                                break
                            source = prepare_source(deepcopy(proposal))
                            if stop.is_set():
                                break
                            intent = scheduler.reserve(proposal, source)
                        else:
                            reserved = reserve_attempt(scheduler, prepare_source, stop.is_set)
                            if reserved is None:
                                break
                            intent, selection = reserved
                        store.create_attempt(intent)
                        if selection is not None:
                            store.write(intent["proposal"]["attempt_id"], "publication-selection.json", selection)
                        started.append(intent["proposal"]["attempt_id"])
                        searches[search_pool.submit(execute_search, intent)] = intent
                        metrics["max_search_stage_occupancy"] = max(metrics["max_search_stage_occupancy"], len(searches))
                    except Exception as exc:
                        failed(intent, "dispatch", exc)
                        if intent is not None:
                            try:
                                settle(intent, diagnostic(intent, "dispatch", exc))
                            except Exception:
                                pass  # Durable pending intent is the fail-closed recovery barrier.
                        try:
                            mm._publish(store.root, f"dispatch-failure-{len(started):08d}.json", failures[-1])
                        except Exception:
                            pass
                        break

                if not searches and not validations and not queue:
                    break
                unfinished = [future for future in (*searches, *validations) if not future.done()]
                if unfinished:
                    wait(unfinished, timeout=0.05, return_when=FIRST_COMPLETED)
                # A full handoff queue always has a running verifier; there is
                # no busy retry of search, retirement, or validation callbacks.

        final = scheduler.status()
        return {"schema_version": "reap.collector-batch.result.v1", "matchmaker_run_sha256": scheduler.run_sha256,
            "status": "blocked" if stop.is_set() or final["blocked"] or final["pending_attempts"] else "completed",
            "started": started, "accepted": accepted, "exhausted": exhausted, "unknown": unknown,
            "skipped_completed": list(initial["attempts"]), "failures": failures,
            "scheduler": final, "metrics": {**metrics, "elapsed_seconds": time.perf_counter()-begin,
                "measurement": "host callback/stage concurrency, not GPU kernel parallelism or utilization"},
            "callbacks_are_trusted_adapters": True, "independent_GPU_or_Lean_verification_performed_by_executor": False}
    finally:
        store.close()
