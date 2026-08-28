"""Finite, durable scheduling of compiled closed propositions.

This module performs no search, verification, training, or remote mutation.
Accepted results must come from a separate trusted verifier. Content pins bind
its claims, but this scheduler does not itself establish their mathematical truth.
One operator owns the directory; never modify committed files or the lease.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import uuid

from . import verified_dataset_store as safe

WRAPPER_VERSION = "reap.closed-prop-wrapper.v1"
SHA = re.compile(r"[a-f0-9]{64}\Z")
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}\Z")
NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9']*(?:\.[A-Za-z_][A-Za-z_0-9']*)+\Z")
STAGING = re.compile(r"\.staged-[a-f0-9]{32}\Z")


class MatchmakerError(ValueError):
    pass


class MatchmakerBlocked(MatchmakerError):
    """Uncertain operations must be reconciled; never automatically resubmit."""


def require(condition, message):
    if not condition:
        raise MatchmakerError(message)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf8")


def content_sha256(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical_bytes(value)).hexdigest()


def _fields(value, keys, label):
    require(type(value) is dict and set(value) == set(keys.split()), f"invalid {label} fields")


def _sha(value, label):
    require(type(value) is str and SHA.fullmatch(value), f"invalid {label} SHA256")


def _int(value, low, high, label):
    require(type(value) is int and low <= value <= high, f"invalid {label} integer")


def problem_pin(environment_sha256, declarations_sha256, closed_prop_name):
    _sha(environment_sha256, "environment"); _sha(declarations_sha256, "declarations")
    require(type(closed_prop_name) is str and NAME.fullmatch(closed_prop_name), "qualified closed Prop name required")
    return content_sha256({"schema_version": "reap.closed-prop.identity.v1",
        "environment_sha256": environment_sha256, "declarations_sha256": declarations_sha256,
        "closed_prop_name": closed_prop_name})


def attempted_prop_pin(problem_sha256, polarity, wrapper_version=WRAPPER_VERSION):
    _sha(problem_sha256, "problem")
    require(polarity in ("prove", "disprove") and wrapper_version == WRAPPER_VERSION, "unsupported attempted proposition")
    return content_sha256({"schema_version": "reap.closed-prop.attempt.v1", "problem_sha256": problem_sha256,
        "polarity": polarity, "wrapper_version": wrapper_version})


def validate_curriculum(value):
    _fields(value, "schema_version environment_sha256 wrapper_version problems", "curriculum")
    require(value["schema_version"] == "reap.matchmaker.curriculum.v1" and value["wrapper_version"] == WRAPPER_VERSION,
            "unsupported curriculum schema/wrapper")
    _sha(value["environment_sha256"], "environment")
    require(type(value["problems"]) is list and 1 <= len(value["problems"]) <= 4, "curriculum requires 1..4 fixed problems")
    seen_ids, seen_pins, targets = set(), set(), {}
    for item in value["problems"]:
        _fields(item, "problem_id family_id is_target problem_sha256 declarations_sha256 closed_prop_name "
                "compilation_receipt_sha256 parent_problem_sha256 transformation_sha256", "problem")
        for key in ("problem_id", "family_id"):
            require(type(item[key]) is str and ID.fullmatch(item[key]), f"invalid {key}")
        require(type(item["is_target"]) is bool, "is_target must be bool")
        for key in ("problem_sha256", "declarations_sha256", "compilation_receipt_sha256"):
            _sha(item[key], key)
        require(item["problem_sha256"] == problem_pin(value["environment_sha256"], item["declarations_sha256"],
                                                      item["closed_prop_name"]), "problem identity pin mismatch")
        require(item["problem_id"] not in seen_ids and item["problem_sha256"] not in seen_pins, "duplicate problem")
        seen_ids.add(item["problem_id"]); seen_pins.add(item["problem_sha256"])
        if item["is_target"]:
            require(item["parent_problem_sha256"] is None and item["transformation_sha256"] is None,
                    "target must not claim a transformation")
            require(item["family_id"] not in targets, "one target per family required")
            targets[item["family_id"]] = item["problem_sha256"]
        else:
            _sha(item["parent_problem_sha256"], "parent"); _sha(item["transformation_sha256"], "transformation")
    for item in value["problems"]:
        require(item["family_id"] in targets, "family target missing")
        if not item["is_target"]:
            require(item["parent_problem_sha256"] == targets[item["family_id"]], "variant parent must be its family target")
    canonical_bytes(value)


def validate_config(value):
    _fields(value, "history_window trust_count mastery_count base_steps budget_multiplier cap_steps max_attempts "
            "total_steps seed max_inflight max_wait_dispatches", "configuration")
    for key in ("history_window", "trust_count", "mastery_count"):
        _int(value[key], 1, 64, key)
    require(value["trust_count"] <= value["history_window"] and value["mastery_count"] <= value["history_window"],
            "history must fit trust/mastery windows")
    _int(value["base_steps"], 1, 1000000, "base_steps")
    _int(value["cap_steps"], value["base_steps"], 1000000, "cap_steps")
    _int(value["budget_multiplier"], 1, 16, "budget_multiplier")
    _int(value["max_attempts"], 1, 10000, "max_attempts")
    _int(value["total_steps"], 1, 10000000000, "total_steps")
    _int(value["seed"], 0, 2**63-1, "seed")
    _int(value["max_inflight"], 1, 4, "max_inflight")
    _int(value["max_wait_dispatches"], 1, 10000, "max_wait_dispatches")


def _read(directory, name):
    raw = directory.read(name)
    require(len(raw) <= 1024*1024, "oversized scheduler receipt")
    try:
        value = json.loads(raw)
        require(canonical_bytes(value) == raw, "noncanonical/duplicate/nonfinite JSON rejected")
    except (ValueError, UnicodeError) as exc:
        raise MatchmakerError("invalid canonical scheduler JSON") from exc
    return value


def _publish(directory, name, value):
    """Publish a complete regular file with atomic no-replace hardlink.

    Staging files remain immutable. Any failure is surfaced, including a failure
    to fsync after publication; callers stop rather than retry an unknown write.
    This avoids renameat2 directory support assumptions on NFS.
    """
    staging = ".staged-" + uuid.uuid4().hex
    directory.write_new(staging, canonical_bytes(value))
    if os.name == "nt":
        os.link(directory.path/staging, directory.path/name)
    else:
        os.link(staging, name, src_dir_fd=directory.handle, dst_dir_fd=directory.handle, follow_symlinks=False)
    directory.sync()


class _Lease:
    def __init__(self, directory):
        try:
            directory.write_new(".writer.lock", b"0")
            directory.sync()
        except FileExistsError:
            pass
        if os.name == "nt":
            import ctypes
            import msvcrt
            handle = safe._kernel.CreateFileW(str(directory.path/".writer.lock"), 0xC0000000, 3,
                                              None, 3, 0x00200000, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            info = safe._FileInfo()
            if (not safe._kernel.GetFileInformationByHandle(handle, ctypes.byref(info))
                    or info.attributes & (0x400 | 0x10)):
                safe._kernel.CloseHandle(handle)
                raise MatchmakerError("invalid writer lease file")
            self.fd = msvcrt.open_osfhandle(handle, os.O_BINARY | os.O_RDWR)
        else:
            self.fd = os.open(".writer.lock", os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory.handle)
        try:
            require(stat.S_ISREG(os.fstat(self.fd).st_mode) and os.fstat(self.fd).st_size == 1, "invalid writer lease")
            if os.name == "nt":
                msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException as exc:
            os.close(self.fd)
            raise MatchmakerBlocked("writer lease unavailable; no work dispatched") from exc

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class Matchmaker:
    """Single-writer, finite scheduler. Use as a context manager or close().

    Reservations are never refunded: total_steps bounds the sum of all admitted
    search budgets, including failures/unknowns. Recent history follows result
    receipt order, not wall-clock guesses. The explicit release applies only to
    the newly proposed attempt. Existing attempts retain their original pins.
    """
    @classmethod
    def create(cls, root, curriculum, config):
        validate_curriculum(curriculum); validate_config(config)
        obj = cls()
        try:
            root = Path(root).absolute()
            parent = obj._stack.enter_context(safe.safe_directory(root.parent))
            obj._directory = obj._stack.enter_context(parent.child(root.name, create=True))
            obj._lease = _Lease(obj._directory)
            obj._events = obj._stack.enter_context(obj._directory.child("events", create=True))
            obj._directory.sync(); parent.sync()
            run = {"schema_version": "reap.matchmaker.run.v1", "curriculum": deepcopy(curriculum), "config": deepcopy(config)}
            _publish(obj._directory, "run.json", run)
            obj._initialize(run)
            return obj
        except BaseException:
            obj.close(); raise

    @classmethod
    def open(cls, root, expected_run_sha256):
        _sha(expected_run_sha256, "expected run")
        obj = cls()
        try:
            obj._directory = obj._stack.enter_context(safe.safe_directory(Path(root)))
            obj._lease = _Lease(obj._directory)
            names = set(os.listdir(obj._directory.path))
            require(all(name in {".writer.lock", "run.json", "events"} or STAGING.fullmatch(name) for name in names),
                    "unexpected scheduler root artifact")
            run = _read(obj._directory, "run.json")
            _fields(run, "schema_version curriculum config", "run")
            require(run["schema_version"] == "reap.matchmaker.run.v1", "unsupported run schema")
            validate_curriculum(run["curriculum"]); validate_config(run["config"])
            require(content_sha256(run) == expected_run_sha256, "run pin mismatch")
            for name in names:
                if STAGING.fullmatch(name):
                    require(canonical_bytes(_read(obj._directory, name)) == canonical_bytes(run),
                            "orphan/different run staging requires reconciliation")
            obj._initialize(run)
            obj._events = obj._stack.enter_context(obj._directory.child("events"))
            files = []
            for name in os.listdir(obj._events.path):
                if STAGING.fullmatch(name):
                    # A retained staging inode must have a matching committed
                    # event. Never ignore an orphan: it could be an interrupted
                    # publication or lost commit, not permission to re-dispatch.
                    staged = _read(obj._events, name)
                    require(type(staged) is dict and type(staged.get("index")) is int,
                            "invalid/orphan staging requires reconciliation")
                    _int(staged["index"], 1, 2*obj._config["max_attempts"], "staged event index")
                    committed = _read(obj._events, f"{staged['index']:08d}.json")
                    require(canonical_bytes(staged) == canonical_bytes(committed),
                            "staging/committed event mismatch requires reconciliation")
                else:
                    require(re.fullmatch(r"[0-9]{8}\.json", name), "unexpected event artifact")
                    files.append(name)
            require(len(files) <= 2*obj._config["max_attempts"], "too many scheduler events")
            for index, name in enumerate(sorted(files), 1):
                require(name == f"{index:08d}.json", "missing/nonsequential scheduler event")
                event = _read(obj._events, name)
                obj._validate_event(event)
                obj._apply(event)
            obj._recovered_pending = {aid for aid, item in obj._attempts.items() if item["result"] is None}
            return obj
        except BaseException:
            obj.close(); raise

    def __init__(self):
        self._stack = ExitStack(); self._lease = None
        self._lock = threading.RLock(); self._closed = False; self._poisoned = False

    def _initialize(self, run):
        self._run = deepcopy(run); self.run_sha256 = content_sha256(run)
        self._config = self._run["config"]
        self._problems = {p["problem_id"]: p for p in self._run["curriculum"]["problems"]}
        self._attempts = {}; self._history = {pid: [] for pid in self._problems}
        self._last_dispatch = {pid: 0 for pid in self._problems}; self._families = {}
        self._disproved = set(); self._reserved = 0; self._event_count = 0
        self._head = self.run_sha256; self._recovered_pending = set()

    def _live(self):
        require(not self._closed, "matchmaker is closed")
        if self._poisoned:
            raise MatchmakerBlocked("local publication outcome unknown; reopen for read-only reconstruction before proceeding")

    def _blocked(self):
        if self._recovered_pending:
            raise MatchmakerBlocked("recovered intent lacks result; reconcile original attempt, do not resend")
        if any(item["result"] and item["result"]["outcome"] == "unknown" for item in self._attempts.values()):
            raise MatchmakerBlocked("unknown attempt outcome blocks new dispatch; not a mathematical failure")

    def _budget(self, pid):
        budget = self._config["base_steps"]
        for result in self._history[pid][-self._config["history_window"]:]:
            if result["outcome"] == "exhausted":
                budget = min(self._config["cap_steps"], budget*self._config["budget_multiplier"])
        return budget

    def _priority(self, pid):
        history = self._history[pid][-self._config["history_window"]:]
        if len(history) < self._config["trust_count"]:
            return 0, "untrusted"
        success = [item["outcome"] == "accepted" for item in history]
        if all(success[-self._config["mastery_count"]:]) and len(success) >= self._config["mastery_count"]:
            return 1, "mastered"
        if any(success) and not all(success):
            return 0, "mixed"
        return 1, "consistent"

    def _plan(self, release):
        self._blocked(); _sha(release, "model release")
        pending = {item["intent"]["proposal"]["problem_id"] for item in self._attempts.values() if item["result"] is None}
        sequence = len(self._attempts)+1
        if len(pending) >= self._config["max_inflight"] or sequence > self._config["max_attempts"]:
            return None
        candidates = []
        for pid, problem in self._problems.items():
            if pid in pending or pid in self._disproved or problem["family_id"] in self._families:
                continue
            budget = self._budget(pid)
            if self._reserved + budget > self._config["total_steps"]:
                continue
            priority, reason = self._priority(pid)
            overdue = (sequence-1-self._last_dispatch[pid]) >= self._config["max_wait_dispatches"]
            tie = content_sha256({"seed": self._config["seed"], "problem_id": pid})
            candidates.append(((0 if overdue else 1, self._last_dispatch[pid] if overdue else priority,
                                self._last_dispatch[pid], tie, pid), pid, budget, "fairness" if overdue else reason))
        if not candidates:
            return None
        _, pid, budget, reason = min(candidates)
        problem = self._problems[pid]
        # A pure seeded decision; no shared or session RNG is consumed.
        polarity = ("prove", "disprove")[int(content_sha256({"seed": self._config["seed"],
                         "problem_id": pid, "sequence": sequence}), 16) % 2]
        return {"schema_version": "reap.matchmaker.proposal.v1", "run_sha256": self.run_sha256,
            "state_sha256": self._head, "attempt_id": f"mm-{self.run_sha256[:20]}-{sequence:05d}",
            "attempt_index": sequence, "problem_id": pid, "family_id": problem["family_id"],
            "problem_sha256": problem["problem_sha256"], "polarity": polarity,
            "attempted_prop_sha256": attempted_prop_pin(problem["problem_sha256"], polarity),
            "model_release_sha256": release, "budget_steps": budget, "priority_reason": reason}

    def plan_next(self, model_release_sha256):
        with self._lock:
            self._live()
            return deepcopy(self._plan(model_release_sha256))

    def reserve(self, proposal, attempt_source):
        """Return an intent only after durable reservation. No external work here."""
        with self._lock:
            self._live()
            require(type(proposal) is dict and "model_release_sha256" in proposal, "proposal required")
            require(canonical_bytes(proposal) == canonical_bytes(self._plan(proposal["model_release_sha256"])),
                    "stale or altered proposal")
            require(proposal is not None, "no available work")
            self._validate_source(proposal, attempt_source)
            base = {"schema_version": "reap.matchmaker.intent.v1", "proposal": deepcopy(proposal),
                    "attempt_source": deepcopy(attempt_source)}
            intent = {**base, "intent_sha256": content_sha256(base)}
            self._commit("intent", intent)
            return deepcopy(intent)

    def _validate_source(self, proposal, source):
        _fields(source, "execution_source_sha256 compilation_receipt_sha256 attempted_prop_sha256 budget_steps", "attempt source")
        for key in ("execution_source_sha256", "compilation_receipt_sha256", "attempted_prop_sha256"):
            _sha(source[key], key)
        _int(source["budget_steps"], 1, 1000000, "source budget")
        require(source["budget_steps"] == proposal["budget_steps"] and
                source["attempted_prop_sha256"] == proposal["attempted_prop_sha256"], "attempt source differs from proposal")

    def _validate_result(self, result):
        _fields(result, "schema_version attempt_id intent_sha256 outcome evidence", "result")
        require(result["schema_version"] == "reap.matchmaker.result.v1", "unsupported result schema")
        require(type(result["attempt_id"]) is str and result["attempt_id"] in self._attempts, "unknown attempt ID")
        item = self._attempts[result["attempt_id"]]; intent = item["intent"]; proposal = intent["proposal"]
        require(result["intent_sha256"] == intent["intent_sha256"], "result intent pin mismatch")
        evidence = result["evidence"]
        if result["outcome"] == "accepted":
            _fields(evidence, "verdict problem_sha256 attempted_prop_sha256 execution_source_sha256 model_release_sha256 "
                    "proof_sha256 verification_receipt_sha256 dataset_sha256 dataset_receipt_sha256 actual_steps", "accepted evidence")
            require(evidence["verdict"] == {"prove": "proved", "disprove": "disproved"}[proposal["polarity"]],
                    "accepted verdict differs from attempted polarity")
            for key in ("problem_sha256", "attempted_prop_sha256", "model_release_sha256"):
                require(evidence[key] == proposal[key], f"accepted {key} mismatch")
            require(evidence["execution_source_sha256"] == intent["attempt_source"]["execution_source_sha256"],
                    "accepted execution source mismatch")
            for key in ("proof_sha256", "verification_receipt_sha256", "dataset_sha256", "dataset_receipt_sha256"):
                _sha(evidence[key], key)
            _int(evidence["actual_steps"], 0, proposal["budget_steps"], "actual_steps")
        elif result["outcome"] == "exhausted":
            _fields(evidence, "terminal_receipt_sha256 actual_steps", "exhausted evidence")
            _sha(evidence["terminal_receipt_sha256"], "terminal receipt")
            _int(evidence["actual_steps"], 0, proposal["budget_steps"], "actual_steps")
        elif result["outcome"] == "unknown":
            _fields(evidence, "diagnostic_sha256 reason", "unknown evidence")
            _sha(evidence["diagnostic_sha256"], "diagnostic")
            require(evidence["reason"] in ("transport_unknown", "process_unknown", "storage_unknown"), "unknown reason required")
        else:
            raise MatchmakerError("only accepted/exhausted/unknown terminal results are allowed")
        canonical_bytes(result)

    def record_result(self, attempt_id, result):
        """Record a trusted terminal verdict; identical retries are read-only.

        This also reconciles an original pending intent after restart. An already
        recorded unknown outcome cannot be overwritten or turned into a retry.
        """
        with self._lock:
            self._live(); self._validate_result(result)
            require(attempt_id == result["attempt_id"], "result ID mismatch")
            item = self._attempts[attempt_id]
            if item["result"] is not None:
                require(canonical_bytes(item["result"]) == canonical_bytes(result), "immutable result already exists")
                return item["result_event_sha256"]
            receipt = self._commit("result", deepcopy(result))
            self._recovered_pending.discard(attempt_id)
            return receipt

    def _validate_event(self, event):
        _fields(event, "schema_version index previous_event_sha256 kind payload", "event")
        require(event["schema_version"] == "reap.matchmaker.event.v1", "unsupported event schema")
        _int(event["index"], 1, 2*self._config["max_attempts"], "event index")
        require(event["index"] == self._event_count+1 and event["previous_event_sha256"] == self._head, "broken event chain")
        if event["kind"] == "intent":
            intent = event["payload"]
            _fields(intent, "schema_version proposal attempt_source intent_sha256", "intent")
            require(intent["schema_version"] == "reap.matchmaker.intent.v1", "unsupported intent schema")
            base = {k: v for k, v in intent.items() if k != "intent_sha256"}
            require(content_sha256(base) == intent["intent_sha256"], "intent pin mismatch")
            proposal = intent["proposal"]
            require(type(proposal) is dict and "model_release_sha256" in proposal, "invalid proposal")
            require(canonical_bytes(proposal) == canonical_bytes(self._plan(proposal["model_release_sha256"])),
                    "intent violates deterministic scheduling/reservations")
            self._validate_source(proposal, intent["attempt_source"])
        elif event["kind"] == "result":
            self._validate_result(event["payload"])
            require(self._attempts[event["payload"]["attempt_id"]]["result"] is None, "duplicate result event")
        else:
            raise MatchmakerError("invalid event kind")

    def _commit(self, kind, payload):
        event = {"schema_version": "reap.matchmaker.event.v1", "index": self._event_count+1,
                 "previous_event_sha256": self._head, "kind": kind, "payload": payload}
        self._validate_event(event)
        try:
            _publish(self._events, f"{event['index']:08d}.json", event)
        except BaseException:
            self._poisoned = True
            raise
        self._apply(event)
        return self._head

    def _apply(self, event):
        payload = deepcopy(event["payload"]); pin = content_sha256(event)
        if event["kind"] == "intent":
            proposal = payload["proposal"]
            self._attempts[proposal["attempt_id"]] = {"intent": payload, "result": None, "result_event_sha256": None}
            self._last_dispatch[proposal["problem_id"]] = proposal["attempt_index"]
            self._reserved += proposal["budget_steps"]
        else:
            item = self._attempts[payload["attempt_id"]]
            item["result"] = payload; item["result_event_sha256"] = pin
            problem = self._problems[item["intent"]["proposal"]["problem_id"]]
            if payload["outcome"] != "unknown":
                self._history[problem["problem_id"]].append(payload)
            if payload["outcome"] == "accepted":
                if payload["evidence"]["verdict"] == "disproved":
                    self._disproved.add(problem["problem_id"])
                if problem["is_target"]:
                    self._families[problem["family_id"]] = {"verdict": payload["evidence"]["verdict"],
                                                            "attempt_id": payload["attempt_id"]}
        self._head = pin; self._event_count += 1

    def status(self):
        with self._lock:
            self._live()
            pending = [aid for aid, item in self._attempts.items() if item["result"] is None]
            unknown = [aid for aid, item in self._attempts.items() if item["result"] and item["result"]["outcome"] == "unknown"]
            return deepcopy({"run_sha256": self.run_sha256, "head_sha256": self._head,
                "event_count": self._event_count, "attempt_count": len(self._attempts), "reserved_steps": self._reserved,
                "remaining_steps": self._config["total_steps"]-self._reserved, "pending_attempts": pending,
                "recovered_pending": sorted(self._recovered_pending), "unknown_attempts": unknown,
                "blocked": bool(unknown or self._recovered_pending), "terminal_families": self._families,
                "problems": {pid: {"known_results": len(history),
                    "exhausted": sum(r["outcome"] == "exhausted" for r in history),
                    "accepted": sum(r["outcome"] == "accepted" for r in history),
                    "disproved": pid in self._disproved, "next_budget_steps": self._budget(pid)}
                    for pid, history in self._history.items()}, "attempts": self._attempts})

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    if self._lease is not None:
                        self._lease.close()
                finally:
                    self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
