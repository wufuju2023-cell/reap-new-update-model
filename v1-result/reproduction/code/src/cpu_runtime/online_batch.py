"""Bounded online V1 batches with conservative, durable resume.

Verified proofs go to solutions.jsonl; known exhausted searches go to a separate
terminal-unsolved.jsonl and do not prevent later theorems from starting.
An interrupted/unknown mutation is never retried, renamed, or overwritten.
There is no total TTT deadline. Local completion is not an independent GPU audit.
"""
from __future__ import annotations

import argparse
from .transport_budget import DEFAULT_BUDGET
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit
import uuid

from .online_ttt import run_online, validate_initialization
from .http_clients import GpuHttpClient

PROFILE = "online-search-visit-backup-v1"
SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}")
EXPERIENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
COMPLETED = {"passed_execution", "solved_without_online_update"}
RETIREMENT_SNAPSHOT = "retired-online-ttt"


class BatchBlocked(RuntimeError):
    """Manual review is required before any further remote mutation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BatchBlocked(message + "; manual intervention required; do not retry the original session")


def encoded(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def no_links(path: Path) -> None:
    for part in (path, *path.parents):
        require(not part.is_symlink() and not (hasattr(part, "is_junction") and part.is_junction()), "symlink/junction rejected")


def read_json(path: Path) -> dict[str, Any]:
    no_links(path)
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=unique_keys,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (OSError, ValueError) as exc:
        raise BatchBlocked(f"incomplete/invalid evidence {path.name}; manual intervention required") from exc
    require(isinstance(value, dict), "evidence must be a JSON object")
    return value


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def sync_directory(path: Path) -> None:
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def publish(path: Path, content: bytes, *, replace: bool = False) -> None:
    """Fsync bytes before atomic publication; caller holds the sole writer lock."""
    no_links(path)
    require(replace or not path.exists(), "refusing to overwrite existing evidence")
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    require(replace or not path.exists(), "evidence appeared during publication")
    os.replace(temporary, path)
    sync_directory(path.parent)


@contextmanager
def batch_lock(output: Path):
    path = output / ".online-batch.lock"
    no_links(path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
                os.fsync(fd)
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise BatchBlocked("another batch writer holds the lock; no work started") from exc
        else:
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise BatchBlocked("another batch writer holds the lock; no work started") from exc
        locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def load_manifest(path: Path, project: Path) -> tuple[list[dict[str, str]], str]:
    no_links(path)
    raw = path.read_bytes()
    require(len(raw) <= 16 * 1024 * 1024, "batch manifest is too large")
    entries, seen = [], set()
    for line in raw.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=unique_keys)
        except ValueError as exc:
            raise BatchBlocked("invalid manifest JSONL; no work started") from exc
        experience_fields = {"experience_id", "experience_weights_sha256", "experience_snapshot_sha256"}
        require(isinstance(row, dict) and {"session_id", "theorem_file"} <= set(row)
                and set(row) <= {"session_id", "theorem_file", "experience_policy"} | experience_fields
                and (not (set(row) & experience_fields) or experience_fields <= set(row)),
                "manifest requires session_id/theorem_file and either all or none of the pinned experience fields")
        initialization = {}
        if "experience_id" in row:
            require(isinstance(row["experience_id"], str) and EXPERIENCE.fullmatch(row["experience_id"]) is not None,
                    "experience_id must name one explicit immutable release; omit for fresh initialization")
            for field in ("experience_weights_sha256", "experience_snapshot_sha256"):
                require(isinstance(row[field], str) and re.fullmatch(r"[0-9a-f]{64}", row[field]) is not None,
                        f"{field} must pin the release content")
            initialization = {field: row[field] for field in experience_fields}
        sid, name = row["session_id"], row["theorem_file"]
        require(isinstance(sid, str) and SESSION.fullmatch(sid) is not None and sid not in seen, "invalid/duplicate session_id")
        require(isinstance(name, str) and bool(name), "theorem_file must be nonempty")
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and relative.as_posix() == name and "\\" not in name and ":" not in name
                and all(part not in (".", "..") for part in relative.parts), "theorem_file must stay inside project")
        source = project.joinpath(*relative.parts)
        no_links(source)
        require(source.is_file() and source.resolve().is_relative_to(project), "theorem source missing/outside project")
        theorem_sha = digest(source.read_bytes())
        if "experience_policy" in row:
            from .experience_policy import validate_policy
            try:
                selected = validate_policy(row["experience_policy"], session_id=sid, theorem_sha256=theorem_sha)
            except (ValueError, TypeError, KeyError) as exc:
                raise BatchBlocked("invalid pinned experience policy; no work started") from exc
            require(encoded(selected) == encoded(initialization), "manifest pins differ from selected policy")
            initialization["experience_policy"] = row["experience_policy"]
        entries.append({"session_id": sid, "theorem_file": name, "theorem_sha256": theorem_sha, **initialization})
        seen.add(sid)
    require(0 < len(entries) <= 10000, "batch requires 1..10000 manifest entries")
    return entries, digest(raw)


def load_solutions(path: Path) -> tuple[bytes, dict[str, dict[str, Any]]]:
    no_links(path)
    if not path.exists():
        return b"", {}
    raw = path.read_bytes()
    require(not raw or raw.endswith(b"\n"), f"partial final {path.name} line; do not truncate automatically")
    records = {}
    for line in raw.splitlines():
        require(bool(line.strip()), f"blank {path.name} record")
        try:
            record = json.loads(line, object_pairs_hook=unique_keys)
        except ValueError as exc:
            raise BatchBlocked(f"invalid {path.name}; manual intervention required") from exc
        require(isinstance(record, dict) and isinstance(record.get("session_id"), str)
                and record["session_id"] not in records, f"duplicate/invalid {path.name} record")
        records[record["session_id"]] = record
    return raw, records


def completed_record(output: Path, entry: dict, identity: dict) -> dict:
    sid, config = entry["session_id"], identity["config"]
    directory = output / sid
    no_links(directory)
    require(directory.is_dir(), f"completed session output missing: {sid}")
    intent = read_json(output / ".batch-intents" / (sid + ".json"))
    require(intent == {"entry": entry, "batch_identity_sha256": digest(encoded(identity))}, f"attempt identity mismatch: {sid}")
    source = Path(config["project_dir"]) / entry["theorem_file"]
    no_links(source)
    require(digest(source.read_bytes()) == entry["theorem_sha256"], f"theorem source changed: {sid}")
    hashes = {}

    def artifact(name):
        path = directory / name
        result = read_json(path)
        hashes[name] = digest(path.read_bytes())
        return result

    session = artifact("session.json")
    expected = {"session_id": sid, "theorem_file": str(source), "profile": PROFILE, "tree_id": sid + ".tree0",
        "theorem_sha256": entry["theorem_sha256"], "gamma": config["gamma"], "max_updates": config["max_updates"],
        "total_deadline_seconds": None,
        "policy_base_url": config["gpu_base_url"] + f"/sessions/{sid}/policy/v1",
        "value_base_url": config["gpu_base_url"] + f"/sessions/{sid}/value/v1"}
    require(all(key in session and session[key] == value for key, value in expected.items()), f"saved session/config mismatch: {sid}")
    experience_id = entry.get("experience_id")
    require(session.get("experience_id") == experience_id and session.get("experience_candidate", False) is False,
            f"saved session initialization mismatch: {sid}")
    for field in ("experience_weights_sha256", "experience_snapshot_sha256"):
        require(session.get(field) == entry.get(field), f"saved session release pin mismatch: {sid}")
    report, lean = artifact("online-result.json"), artifact("result.json")
    require(report.get("experience_id") == experience_id, f"result initialization mismatch: {sid}")
    require(report.get("session_id") == sid and report.get("tree_id") == sid + ".tree0"
            and report.get("profile") == PROFILE and type(report.get("returncode")) is int
            and "error" in report and report["error"] is None
            and "total_deadline_seconds" in report and report["total_deadline_seconds"] is None,
            f"session has incomplete/failed/unknown execution: {sid}")
    require(lean.get("schema_version") == "reap.training.result.v1" and lean.get("session_id") == sid
            and type(lean.get("solved")) is bool and "error" in lean and lean["error"] is None,
            f"final Lean result invalid: {sid}")
    solved = lean["solved"]
    if solved:
        require(report.get("status") in COMPLETED and report["returncode"] == 0 and report.get("root_verified") is True
                and lean.get("status") == "solved" and isinstance(lean.get("proof_script"), str)
                and bool(lean["proof_script"].strip()), f"final Lean proof invalid: {sid}")
    else:
        # RolloutSink writes exhausted then throws the expected tactic error.
        # A generic lean_failed alone is never a safe terminal classification.
        require(report.get("status") == "lean_failed" and report["returncode"] == 1
                and report.get("root_verified") is False and lean.get("status") == "exhausted"
                and "proof_script" in lean and lean["proof_script"] is None, f"not a known exhausted search: {sid}")
        progress_path = directory / "progress.jsonl"
        no_links(progress_path)
        progress_hash, last_line = hashlib.sha256(), b""
        try:
            with progress_path.open("rb") as stream:
                for line in stream:
                    require(line.endswith(b"\n"), f"partial final progress record: {sid}")
                    progress_hash.update(line)
                    last_line = line
            progress = json.loads(last_line, object_pairs_hook=unique_keys)
        except (OSError, ValueError) as exc:
            raise BatchBlocked(f"missing/invalid exhausted progress: {sid}; manual intervention required") from exc
        require(isinstance(progress, dict) and progress.get("schema_version") == "reap.training.progress.v1"
                and progress.get("session_id") == sid and progress.get("done") is True
                and progress.get("solved") is False and progress.get("status") == "exhausted",
                f"missing terminal exhausted progress: {sid}")
        hashes["progress.jsonl"] = progress_hash.hexdigest()
    updates, receipts = report.get("optimizer_updates"), report.get("learn_receipts")
    require(type(updates) is int and 0 <= updates <= config["max_updates"] and isinstance(receipts, list)
            and len(receipts) == updates and report.get("policy_version") == updates, f"update/version accounting mismatch: {sid}")
    exercised = report.get("online_update_consumed_by_later_generation")
    generations = report.get("post_update_generations")
    require(type(exercised) is bool and isinstance(generations, list) and exercised == bool(updates and generations)
            and (report["status"] == "passed_execution") == (solved and exercised), f"online gate mismatch: {sid}")
    require("training_enabled" not in report or report["training_enabled"] is (config["max_updates"] > 0),
            f"training mode mismatch: {sid}")
    require(all(g.get("tree_id") == sid + ".tree0" and type(g.get("policy_version")) is int
                and 0 < g["policy_version"] <= updates for g in generations), f"generation identity mismatch: {sid}")
    created = artifact("create-receipt.json")
    require(created.get("session_id") == sid and created.get("policy_version") == 0, f"create receipt mismatch: {sid}")
    try:
        validate_initialization(created, sid, theorem_id=entry["theorem_sha256"] if experience_id else None,
            experience_id=experience_id, weights_sha256=entry.get("experience_weights_sha256"),
            snapshot_sha256=entry.get("experience_snapshot_sha256"))
        if "experience_policy" in entry:
            from .experience_policy import validate_policy, validate_execution_contract, digest as policy_digest
            policy = artifact("experience-policy.json")
            require(encoded(policy) == encoded(entry["experience_policy"]), f"saved experience policy changed: {sid}")
            validate_policy(policy, session_id=sid, theorem_sha256=entry["theorem_sha256"], gamma=config["gamma"])
            validate_execution_contract(created, policy)
            expected_contract = policy["selection"]["expected_initialization_contract_sha256"]
            for saved in (session, report):
                require(saved.get("experience_policy_sha256") == policy_digest(policy)
                    and saved.get("expected_initialization_contract_sha256") == expected_contract,
                    f"saved policy/contract pins changed: {sid}")
            require(created.get("theorem_id") == entry["theorem_sha256"], f"policy theorem identity changed: {sid}")
    except ValueError as exc:
        raise BatchBlocked(f"create receipt initialization mismatch: {sid}: {exc}; "
                           "manual intervention required; do not retry the original session") from exc
    for label in ("before", "after"):
        snapshot = artifact(f"snapshot-{label}.json")
        require(snapshot == {"session_id": sid, "snapshot": f"{label}-online-ttt"}, f"snapshot receipt mismatch: {sid}")
    checkpoints = directory / "checkpoints"
    no_links(checkpoints)
    require(checkpoints.is_dir(), f"checkpoint directory missing: {sid}")
    requests = sorted(checkpoints.glob("learn-*.request.json"))
    saved_receipts = sorted(checkpoints.glob("learn-*.receipt.json"))
    require(len(requests) == len(saved_receipts) == updates, f"unresolved or extra remote mutation: {sid}")
    for index, (request_path, receipt_path, receipt) in enumerate(zip(requests, saved_receipts, receipts)):
        require(request_path.name.replace(".request.json", ".receipt.json") == receipt_path.name, f"unpaired learn mutation: {sid}")
        event = artifact(request_path.relative_to(directory).as_posix())
        saved = artifact(receipt_path.relative_to(directory).as_posix())
        require(saved == receipt and receipt.get("event_id") == event.get("event_id")
                and receipt.get("applied") is True and receipt.get("idempotent") is False
                and receipt.get("policy_version") == index + 1 and event.get("policy_version") == index
                and event.get("session_id") == sid and event.get("tree_id") == sid + ".tree0"
                and event.get("gamma") == config["gamma"], f"learn receipt/event mismatch: {sid}")
    return {"schema_version": "reap.online-batch.solution.v1" if solved else "reap.online-batch.unsolved.v1",
            **entry, "profile": PROFILE,
            "batch_identity_sha256": digest(encoded(identity)), "status": report["status"] if solved else "search_exhausted",
            "root_verified": solved, "online_execution_gate_passed": solved and exercised,
            "policy_version": updates, "proof_script": lean["proof_script"], "artifact_sha256": hashes,
            "independent_GPU_tensor_and_wire_audit_passed": False}


def retirement_intent(record: dict, identity: dict) -> dict:
    result = {"schema_version": "reap.online-batch.retirement-intent.v1", "session_id": record["session_id"],
            "policy_version": record["policy_version"], "snapshot": RETIREMENT_SNAPSHOT,
            "terminal_record_sha256": digest(encoded(record)), "batch_identity_sha256": digest(encoded(identity))}
    if identity["config"].get("reuse_final_snapshot", False):
        result.update(schema_version="reap.online-batch.retirement-intent.v2",
                      snapshot="after-online-ttt", reuse_snapshot=True)
    return result


def validate_retirement_receipt(receipt: dict, intent: dict) -> None:
    reuse = intent.get("reuse_snapshot") is True
    schema = "reap.gpu.retirement.v2" if reuse else "reap.gpu.retirement.v1"
    require(isinstance(receipt, dict) and receipt.get("schema_version") == schema
            and receipt.get("session_id") == intent["session_id"]
            and type(receipt.get("policy_version")) is int and receipt["policy_version"] == intent["policy_version"]
            and receipt.get("snapshot") == intent["snapshot"] and receipt.get("status") == "released"
            and receipt.get("mutation_retry_allowed") is False and receipt.get("tombstone_scope") == "current_runtime"
            and isinstance(receipt.get("snapshot_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", receipt["snapshot_sha256"]) is not None,
            f"missing/mismatched released retirement receipt: {intent['session_id']}")
    if reuse:
        require(set(receipt) == {"schema_version", "session_id", "policy_version", "snapshot", "snapshot_sha256",
                "status", "mutation_retry_allowed", "tombstone_scope", "snapshot_mode", "snapshot_revision"}
                and receipt["snapshot_mode"] == "reuse_verified"
                and type(receipt["snapshot_revision"]) is int and receipt["snapshot_revision"] > 0,
                "retirement did not explicitly verify same-runtime snapshot reuse")


def retirement_paths(output: Path, sid: str) -> tuple[Path, Path, Path]:
    directory = output / ".batch-retirements"
    paths = tuple(directory / (sid + suffix) for suffix in (".intent.json", ".response.json", ".receipt.json"))
    for path in paths:
        no_links(path)
    return paths


def retirement_already_recorded(output: Path, record: dict, identity: dict) -> bool:
    """Preflight only: any uncertain earlier submission blocks all new work."""
    intent_path, response_path, receipt_path = retirement_paths(output, record["session_id"])
    if not intent_path.exists():
        require(not response_path.exists() and not receipt_path.exists(), "retirement response/receipt has no durable intent")
        return False
    intent = retirement_intent(record, identity)
    require(read_json(intent_path) == intent, "retirement intent/terminal identity mismatch")
    require(response_path.exists() and receipt_path.exists(), "retirement outcome is unknown; inspect existing job/receipt, never resubmit")
    receipt = read_json(receipt_path)
    require(set(receipt) == {"intent_sha256", "receipt"} and receipt["intent_sha256"] == digest(encoded(intent)),
            "retirement receipt/intent binding mismatch")
    validate_retirement_receipt(receipt["receipt"], intent)
    require(read_json(response_path) == receipt["receipt"], "saved retirement response/receipt mismatch")
    return True


def retire_completed_record(output: Path, record: dict, identity: dict, client: GpuHttpClient) -> None:
    intent = retirement_intent(record, identity)
    intent_path, response_path, receipt_path = retirement_paths(output, record["session_id"])
    intent_path.parent.mkdir(exist_ok=True)
    # The exclusive intent is the dispatch boundary. Never repeat a submission,
    # including if the previous call returned an invalid response or lost ACK.
    publish(intent_path, encoded(intent))
    options = {"reuse_snapshot": True} if intent.get("reuse_snapshot") else {}
    receipt = client.retire_session(record["session_id"], intent["snapshot"],
                                   expected_policy_version=record["policy_version"], **options)
    publish(response_path, encoded(receipt))
    validate_retirement_receipt(receipt, intent)
    publish(receipt_path, encoded({"intent_sha256": digest(encoded(intent)), "receipt": receipt}))


def run_batch(*, manifest: Path, project_dir: Path, output_root: Path, gpu_base_url: str,
              gamma: float, max_updates: int = 5, concurrency: int = 2,
              http_timeout: float = DEFAULT_BUDGET.client, barrier_timeout: int = DEFAULT_BUDGET.barrier, lean_bin: str = "lake",
              executor: Callable[..., dict] | None = None, retire_completed: bool = False,
              retirement_client: GpuHttpClient | None = None, reuse_final_snapshot: bool = False) -> dict:
    require(type(concurrency) is int and 1 <= concurrency <= 32, "concurrency must be 1..32")
    require(type(retire_completed) is bool, "retire_completed must be a boolean")
    require(type(reuse_final_snapshot) is bool and (not reuse_final_snapshot or retire_completed),
            "reuse_final_snapshot requires explicit retire_completed")
    require(type(max_updates) is int and max_updates >= 0 and type(barrier_timeout) is int and barrier_timeout > 0,
            "update limit must be nonnegative and barrier limit must be positive")
    require(type(gamma) in (int, float) and math.isfinite(gamma) and 0 < gamma < 1
            and type(http_timeout) in (int, float) and math.isfinite(http_timeout) and http_timeout > 0, "invalid gamma/HTTP timeout")
    url = urlsplit(gpu_base_url)
    require(url.scheme in ("http", "https") and bool(url.hostname) and not url.username and not url.password
            and url.path in ("", "/") and not url.query and not url.fragment, "GPU URL must be an origin without credentials")
    gpu_base_url = gpu_base_url.rstrip("/")
    project_dir, output_root, manifest = Path(project_dir), Path(output_root), Path(manifest)
    for path in (project_dir, output_root, manifest):
        no_links(path)
    project_dir, output_root = project_dir.resolve(), output_root.resolve()
    entries, manifest_sha = load_manifest(manifest, project_dir)
    for entry in entries:
        if "experience_policy" in entry:
            from .experience_policy import validate_policy
            validate_policy(entry["experience_policy"], session_id=entry["session_id"],
                            theorem_sha256=entry["theorem_sha256"], gamma=gamma)
    config = {"project_dir": str(project_dir), "gpu_base_url": gpu_base_url, "gamma": gamma,
              "max_updates": max_updates, "http_timeout": http_timeout, "barrier_timeout": barrier_timeout,
              "lean_bin": lean_bin, "total_deadline_seconds": None,
              "retire_completed": retire_completed,
              "reuse_final_snapshot": reuse_final_snapshot,
              "implementation_sha256": {name: digest(Path(__file__).with_name(name).read_bytes())
                                         for name in ("online_batch.py", "online_ttt.py", "http_clients.py")}}
    if any("experience_policy" in entry for entry in entries):
        config["implementation_sha256"]["experience_policy.py"] = digest(Path(__file__).with_name("experience_policy.py").read_bytes())
    identity = {"schema_version": "reap.online-batch.identity.v1", "profile": PROFILE,
                "manifest_sha256": manifest_sha, "entries": entries, "config": config}
    output_root.mkdir(parents=True, exist_ok=True)
    execute = executor or run_online
    with batch_lock(output_root):
        identity_path = output_root / "batch-config.json"
        if identity_path.exists():
            require(read_json(identity_path) == identity, "batch profile/config/source/manifest changed")
        else:
            require(set(p.name for p in output_root.iterdir()) == {".online-batch.lock"}, "nonempty output has no batch identity")
            publish(identity_path, encoded(identity))
        intents = output_root / ".batch-intents"
        no_links(intents)
        intents.mkdir(exist_ok=True)
        result_paths = {True: output_root / "solutions.jsonl", False: output_root / "terminal-unsolved.jsonl"}
        prefixes, saved = {}, {}
        for solved, path in result_paths.items():
            prefixes[solved], records = load_solutions(path)
            require(not set(saved).intersection(records), "session recorded as both solved and unsolved")
            require(all(record.get("root_verified") is solved for record in records.values()),
                    f"record placed in wrong terminal file: {path.name}")
            saved.update(records)
        require(set(saved).issubset({e["session_id"] for e in entries}), "terminal records contain sessions outside manifest")
        pending, recovered, skipped = [], [], []
        retired, retirement_skipped, retirement_pending = [], [], []
        for entry in entries:
            sid = entry["session_id"]
            no_links(intents / (sid + ".json"))
            no_links(output_root / sid)
            attempted = (intents / (sid + ".json")).exists() or (output_root / sid).exists() or sid in saved
            if retire_completed:
                attempted = attempted or any(path.exists() for path in retirement_paths(output_root, sid))
            if attempted:
                record = completed_record(output_root, entry, identity)
                if retire_completed:
                    if retirement_already_recorded(output_root, record, identity):
                        retirement_skipped.append(sid)
                    else:
                        retirement_pending.append(record)
                if sid in saved:
                    require(saved[sid] == record, f"saved solution/config/source/artifacts changed: {sid}")
                    skipped.append(sid)
                else:
                    recovered.append(record)
            else:
                pending.append(entry)

        def append(record):
            solved = record["root_verified"]
            path, prefix = result_paths[solved], prefixes[solved]
            # Logical append via atomic replacement: the old prefix is retained
            # byte-for-byte, and only a complete fsynced line becomes visible.
            require((path.read_bytes() if path.exists() else b"") == prefix,
                    "terminal records changed during append; stop without overwriting")
            new = prefix + encoded(record)
            publish(path, new, replace=path.exists())
            prefixes[solved] = new

        for record in recovered:
            append(record)
        retire_client = retirement_client or (GpuHttpClient(gpu_base_url, timeout_seconds=http_timeout) if retire_completed else None)
        for record in retirement_pending:
            try:
                retire_completed_record(output_root, record, identity, retire_client)
                retired.append(record["session_id"])
            except Exception as exc:
                raise BatchBlocked("retirement outcome unresolved; inspect original request; do not retry") from exc
        completed, exhausted, failures, cursor = [], [], [], 0
        scheduling_started = time.monotonic()
        scheduling_guard = threading.Lock()
        scheduling = {"started": 0, "finished": 0, "active": 0, "max_active": 0,
                      "session_execution_seconds_total": 0.0}

        def tracked_execute(**kwargs):
            started = time.monotonic()
            with scheduling_guard:
                scheduling["started"] += 1
                scheduling["active"] += 1
                scheduling["max_active"] = max(scheduling["max_active"], scheduling["active"])
            try:
                return execute(**kwargs)
            finally:
                with scheduling_guard:
                    scheduling["finished"] += 1
                    scheduling["active"] -= 1
                    scheduling["session_execution_seconds_total"] += time.monotonic() - started

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            running = {}

            def submit(entry):
                sid = entry["session_id"]
                source = project_dir / entry["theorem_file"]
                no_links(source)
                require(digest(source.read_bytes()) == entry["theorem_sha256"], f"theorem source changed before execution: {sid}")
                publish(intents / (sid + ".json"), encoded({"entry": entry, "batch_identity_sha256": digest(encoded(identity))}))
                future = pool.submit(tracked_execute, session_id=sid, project_dir=project_dir,
                    theorem_file=entry["theorem_file"], output_root=output_root, gpu_base_url=gpu_base_url,
                    gamma=gamma, max_updates=max_updates, http_timeout=http_timeout,
                    barrier_timeout=barrier_timeout, lean_bin=lean_bin,
                    **({field: entry[field] for field in
                        ("experience_id", "experience_weights_sha256", "experience_snapshot_sha256")}
                       if "experience_id" in entry else {}),
                    **({"experience_policy": entry["experience_policy"]} if "experience_policy" in entry else {}))
                running[future] = entry

            while cursor < len(pending) and len(running) < concurrency:
                submit(pending[cursor])
                cursor += 1
            while running:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    entry = running.pop(future)
                    try:
                        returned = future.result()
                        record = completed_record(output_root, entry, identity)
                        require(returned == read_json(output_root / entry["session_id"] / "online-result.json"), "executor return differs from saved result")
                        append(record)
                        (completed if record["root_verified"] else exhausted).append(entry["session_id"])
                        if retire_completed:
                            retire_completed_record(output_root, record, identity, retire_client)
                            retired.append(entry["session_id"])
                    except Exception as exc:
                        failures.append({"session_id": entry["session_id"], "error_type": type(exc).__name__,
                                         "mutation_retry_allowed": False, "manual_intervention_required": True})
                while not failures and cursor < len(pending) and len(running) < concurrency:
                    submit(pending[cursor])
                    cursor += 1
        return {"schema_version": "reap.online-batch.result.v1", "profile": PROFILE,
            "status": "manual_intervention_required" if failures else "completed", "concurrency": concurrency,
            "completed": sorted(completed), "exhausted": sorted(exhausted), "skipped": skipped,
            "skipped_exhausted": [sid for sid in skipped if not saved[sid]["root_verified"]],
            "retired": sorted(retired), "retirement_skipped": retirement_skipped,
            "recovered_without_GPU_calls": [r["session_id"] for r in recovered if r["session_id"] not in retired],
            "recovered_without_search_rerun": [r["session_id"] for r in recovered],
            "not_started": [e["session_id"] for e in pending[cursor:]], "failures": failures,
            "scheduling": {**scheduling, "elapsed_seconds": time.monotonic() - scheduling_started,
                "measurement": "host_wall_time_including_Lean_HTTP_and_snapshots_not_GPU_utilization"},
            "total_deadline_seconds": None, "independent_GPU_tensor_and_wire_audit_passed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-base-url", required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=5)
    parser.add_argument("--http-timeout-seconds", type=float, default=DEFAULT_BUDGET.client)
    parser.add_argument("--barrier-timeout-seconds", type=int, default=DEFAULT_BUDGET.barrier)
    parser.add_argument("--lean-bin", default="lake")
    parser.add_argument("--retire-completed", action="store_true",
                        help="after verified terminal artifacts, persist one retirement intent and require a released receipt before refill")
    parser.add_argument("--reuse-final-snapshot", action="store_true",
                        help="requires --retire-completed; reuse unchanged same-runtime after-online-ttt snapshot")
    args = parser.parse_args()
    try:
        result = run_batch(manifest=args.manifest, project_dir=args.project_dir, output_root=args.output_dir,
            gpu_base_url=args.gpu_base_url, gamma=args.gamma, max_updates=args.max_updates,
            concurrency=args.concurrency, http_timeout=args.http_timeout_seconds,
            barrier_timeout=args.barrier_timeout_seconds, lean_bin=args.lean_bin, retire_completed=args.retire_completed,
            reuse_final_snapshot=args.reuse_final_snapshot)
    except (BatchBlocked, OSError, ValueError) as exc:
        print(json.dumps({"status": "manual_intervention_required", "error_type": type(exc).__name__,
                          "message": str(exc), "mutation_retry_allowed": False}))
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
