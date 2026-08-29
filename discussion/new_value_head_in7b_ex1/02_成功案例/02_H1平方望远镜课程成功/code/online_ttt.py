"""Coordinate in-search updates through the Lean observer's checkpoint barrier.

One process and one MCTS tree remain alive while LEARN runs.  No uncertain
mutation is retried, and no ACK permits continued search after a failed update.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any, Callable

from .batch_solver import SessionSpec, SESSION_RE, _finalize_solver_status
from .http_clients import GpuHttpClient
from .transport_budget import DEFAULT_BUDGET
from .search_deadline import DeadlineExpired, DeadlineProxy, SearchDeadline

OBJECTIVE = "search_visit_backup"
VALUE_SEMANTICS = "reap.search_backup_discounted_return.v1"
CATEGORICAL_VALUE_SEMANTICS = "new_value_head.distance_categorical_64_saturated.v1"
MAX_LEARN_BODY_BYTES = 8192  # Current authenticated browser transport contract.
MAX_REQUEST_BODY_BYTES = 256 * 1024  # Explicit upgraded transport, never a model context limit.
EXPERIENCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXPERIENCE_RESETS = ["optimizer", "rng", "buffer", "policy_version", "event_receipts"]


def validate_experience_reference(experience_id: str | None, weights_sha256: str | None,
                                  snapshot_sha256: str | None) -> None:
    if experience_id is not None and (not isinstance(experience_id, str)
                                      or EXPERIENCE_ID_RE.fullmatch(experience_id) is None):
        raise ValueError("experience_id must name one explicit immutable release")
    if weights_sha256 is not None or snapshot_sha256 is not None:
        if experience_id is None or not all(isinstance(value, str) and SHA256_RE.fullmatch(value)
                                            for value in (weights_sha256, snapshot_sha256)):
            raise ValueError("pinned initialization requires experience_id and both SHA256 digests")


def validate_initialization(created: dict[str, Any], session_id: str, *,
                            theorem_id: str | None = None, experience_id: str | None = None,
                            weights_sha256: str | None = None, snapshot_sha256: str | None = None) -> None:
    """Verify declared provenance and fresh state; this is not a tensor audit.

    The same gate is used before any Lean process and when reviewing saved
    batch completion. A bad create receipt is preserved; no mutation is retried
    or automatically deleted. Legacy fresh receipts remain accepted.
    """
    validate_experience_reference(experience_id, weights_sha256, snapshot_sha256)
    if not isinstance(created, dict):
        raise ValueError("created session receipt must be an object")
    if theorem_id is not None and created.get("theorem_id") != theorem_id:
        raise ValueError("created session theorem identity mismatch")
    lineage = created.get("lineage", {})
    if not isinstance(lineage, dict) or lineage.get("experience_id") != experience_id:
        raise ValueError("created session experience lineage mismatch")
    if experience_id is None:
        return
    source = lineage.get("source")
    identifier = lambda value: isinstance(value, str) and EXPERIENCE_ID_RE.fullmatch(value) is not None
    digest = lambda value: isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
    if (set(lineage) != {"experience_id", "weights_sha256", "source", "reset"}
            or not identifier(theorem_id) or not digest(lineage.get("weights_sha256"))
            or lineage.get("reset") != EXPERIENCE_RESETS or not isinstance(source, dict)
            or set(source) != {"session_id", "theorem_id", "policy_version", "snapshot",
                               "snapshot_sha256", "parent_experience_id"}
            or not identifier(source["session_id"]) or source["session_id"] == session_id
            or not identifier(source["theorem_id"]) or source["theorem_id"] == theorem_id
            or type(source["policy_version"]) is not int or source["policy_version"] <= 0
            or not identifier(source["snapshot"]) or not digest(source["snapshot_sha256"])
            or not (source["parent_experience_id"] is None or identifier(source["parent_experience_id"]))):
        raise ValueError("inherited release/reset/source/destination identity mismatch")
    if weights_sha256 is not None and (lineage["weights_sha256"] != weights_sha256
                                       or source["snapshot_sha256"] != snapshot_sha256):
        raise ValueError("created experience content differs from pinned release; no Lean search started")
    optimizer = created.get("optimizer_metadata")
    if (created.get("schema_version") != "reap.gpu.session.v1" or created.get("session_id") != session_id
            or type(created.get("policy_version")) is not int or created["policy_version"] != 0
            or created.get("completed") is not False or not isinstance(optimizer, dict)
            or type(optimizer.get("steps")) is not int or optimizer["steps"] != 0
            or created.get("buffer_metadata") != {"events": {}, "pending_event_ids": [], "consumed_event_ids": []}
            or created.get("event_receipts") != {}):
        raise ValueError("inherited session must have fresh optimizer/version/buffer/event state")


def validate_created(created: dict[str, Any], session_id: str, gamma: float,
                     *, theorem_id: str | None = None, experience_id: str | None = None,
                     weights_sha256: str | None = None, snapshot_sha256: str | None = None) -> None:
    validate_initialization(created, session_id, theorem_id=theorem_id, experience_id=experience_id,
                            weights_sha256=weights_sha256, snapshot_sha256=snapshot_sha256)
    metadata = created.get("value_metadata", {})
    contract = created.get("initialization_contract", {})
    search = contract.get("search_config", {}) if isinstance(contract, dict) else {}
    categorical = contract.get("backend") == "real-search-categorical"
    expected_semantics = CATEGORICAL_VALUE_SEMANTICS if categorical else VALUE_SEMANTICS
    categorical_matches_contract = (not categorical or (
        isinstance(search, dict)
        and metadata.get("head") == "categorical-64"
        and metadata.get("max_distance") == 64
        and metadata.get("categorical_value") == search.get("categorical_value")
        and search.get("value_semantics") == expected_semantics
        and search.get("objective") == OBJECTIVE
        and search.get("gamma") == gamma))
    if (created.get("session_id") != session_id or created.get("policy_version") != 0
            or metadata.get("objective") != OBJECTIVE or metadata.get("gamma") != gamma
            or metadata.get("value_semantics") != expected_semantics
            or not categorical_matches_contract):
        raise ValueError("created session does not match strict objective/session/gamma/value semantics")


def validate_learn_receipt(receipt: dict[str, Any], event: dict[str, Any], version: int) -> None:
    if (receipt.get("event_id") != event["event_id"]
            or receipt.get("policy_version") != version + 1
            or receipt.get("applied") is not True or receipt.get("idempotent") is not False):
        raise ValueError("learn receipt identity/version/applied mismatch")
    detail = receipt.get("detail", {})
    if detail.get("objective") != OBJECTIVE or detail.get("optimizer_steps") != version + 1:
        raise ValueError("learn receipt objective/optimizer step mismatch")
    for flag in ("finite_loss", "finite_gradients", "finite_parameters", "finite_optimizer_state", "base_parameters_frozen"):
        if detail.get(flag) is not True:
            raise ValueError(f"learn receipt missing successful {flag} gate")
    for key in ("loss", "policy_loss", "kl", "value_loss", "grad_norm", "value_target"):
        value = detail.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"learn receipt nonfinite/missing {key}")
    trace = detail.get("search_trace", {})
    for key in ("tree_id", "step", "node_index", "policy_version", "gamma", "terminal_verified"):
        if trace.get(key) != event[key]:
            raise ValueError(f"learn receipt source mismatch: {key}")
    if detail.get("prompt_sha256") != hashlib.sha256(event["prompt"].encode()).hexdigest():
        raise ValueError("learn receipt prompt hash mismatch")
    diffs = detail.get("parameter_diffs", {})
    for group in ("adapter", "value_head", "optimizer"):
        item = diffs.get(group, {})
        if (item.get("before_sha256") == item.get("after_sha256")
                or item.get("changed_tensors", 0) + item.get("added_tensors", 0) <= 0):
            raise ValueError(f"learn receipt has no changed {group} update")


def validate_body_limit(value: int) -> None:
    if type(value) is not int or not MAX_LEARN_BODY_BYTES <= value <= MAX_REQUEST_BODY_BYTES:
        raise ValueError("max_request_body_bytes must be an integer from 8192 to 262144")


def validate_training_snapshot_receipt(receipt: dict[str, Any], session_id: str, name: str) -> None:
    """The snapshot endpoint currently acknowledges only its stable identity pair."""
    if receipt != {"session_id": session_id, "snapshot": name}:
        raise ValueError("training snapshot receipt identity/schema mismatch")


def validate_learn_body(version: int, event: dict[str, Any], *, max_request_body_bytes: int = MAX_LEARN_BODY_BYTES) -> None:
    validate_body_limit(max_request_body_bytes)
    # Match GpuHttpClient's actual encoding, including spaces and UTF-8.
    body = json.dumps({"expected_policy_version": version, "event": event},
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(event["candidates"]) > 64:
        raise ValueError("candidate count exceeds 64; never truncate the visit distribution")
    if len(body) > max_request_body_bytes:
        raise ValueError(f"learn body exceeds configured transport {max_request_body_bytes}-byte limit; never truncate")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()


def atomic_new_json(path: Path, value: Any) -> None:
    """Publish a new ACK/result in an owned directory; never replace evidence."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError(path)
    os.rename(temporary, path)


class JsonlTail:
    """Consume only complete flushed records, retaining a partial final line."""
    def __init__(self, path: Path):
        self.path, self.offset, self.pending = path, 0, b""

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("rb") as handle:
            if self.path.stat().st_size < self.offset:
                raise ValueError("observer file was truncated")
            handle.seek(self.offset)
            data = handle.read()
        self.offset += len(data)
        lines = (self.pending + data).split(b"\n")
        self.pending = lines.pop()
        return [json.loads(line) for line in lines if line.strip()]


class OnlineCoordinator:
    def __init__(self, session_id: str, tree_id: str, ack_dir: Path,
                 learn: Callable[[int, dict[str, Any]], dict[str, Any]], *,
                 gamma: float, max_updates: int = 5, min_checkpoint_interval: int = 1,
                 training_snapshot: Callable[[str], dict[str, Any]] | None = None,
                 training_snapshot_interval: int | None = None,
                 max_request_body_bytes: int = MAX_LEARN_BODY_BYTES,
                 deadline: SearchDeadline | None = None):
        if not SESSION_RE.fullmatch(session_id) or not SESSION_RE.fullmatch(tree_id):
            raise ValueError("invalid session/tree id")
        if not math.isfinite(gamma) or not 0 < gamma < 1 or type(max_updates) is not int or max_updates < 0:
            raise ValueError("invalid gamma/update budget")
        if type(min_checkpoint_interval) is not int or min_checkpoint_interval < 1:
            raise ValueError("min_checkpoint_interval must be a positive integer")
        if training_snapshot_interval is not None and (
                type(training_snapshot_interval) is not int or training_snapshot_interval < 1):
            raise ValueError("training_snapshot_interval must be a positive integer or None")
        if training_snapshot_interval is not None and training_snapshot is None:
            raise ValueError("training_snapshot_interval requires a snapshot callback")
        validate_body_limit(max_request_body_bytes)
        self.session_id, self.tree_id = session_id, tree_id
        self.ack_dir, self.learn, self.gamma = ack_dir, learn, gamma
        self.max_updates, self.version, self.next_sequence = max_updates, 0, 0
        self.min_checkpoint_interval = min_checkpoint_interval
        self.training_snapshot = training_snapshot
        self.training_snapshot_interval = training_snapshot_interval
        self.training_snapshot_dir = ack_dir / "training-snapshots"
        self.pending_training_snapshot: dict[str, Any] | None = None
        self.max_request_body_bytes = max_request_body_bytes
        self.last_update_checkpoint: int | None = None
        self.last_step = -1
        self.deadline = deadline
        self.deadline_stopped = False
        self.deadline_refresh_skipped = False
        self.generated: dict[tuple[int, int, int, int], dict[str, Any]] = {}
        self.edge_sources: dict[tuple[int, str], dict[str, Any]] = {}
        self.backed_up_nodes: set[int] = set()
        self.consumed_stats: dict[int, tuple[Any, ...]] = {}
        self.selection_value_refresh = False
        self.pending_refresh: dict[str, Any] | None = None
        self.selection_refreshes: list[dict[str, Any]] = []
        self.receipts: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.post_update_generations: list[dict[str, Any]] = []
        self.ack_dir.mkdir(parents=True, exist_ok=False)

    def _snapshot_after_confirmed_learn(self, *, step: int, event: dict[str, Any]) -> None:
        """Durably witness periodic mutable training state before releasing Lean.

        The GPU snapshot contains only same-session mutable training state
        (trainable parameters, optimizer/RNG and session metadata); it is not a
        claim that the live Lean tree can be resumed.
        """
        interval = self.training_snapshot_interval
        if interval is None or len(self.receipts) % interval:
            return
        assert self.training_snapshot is not None
        name = f"online-ttt-v{self.version}"
        intent = {
            "schema_version": "reap.online-ttt.training-snapshot-intent.v1",
            "session_id": self.session_id,
            "tree_id": self.tree_id,
            "checkpoint_step": step,
            "learn_count": len(self.receipts),
            "policy_version": self.version,
            "learn_event_id": event["event_id"],
            "snapshot": name,
            "snapshot_scope": "same-session-mutable-training-state-only",
            "lean_tree_recovery": False,
            "mutation_retry_allowed": False,
        }
        self.training_snapshot_dir.mkdir(exist_ok=True)
        intent_path = self.training_snapshot_dir / f"snapshot-v{self.version}.intent.json"
        receipt_path = self.training_snapshot_dir / f"snapshot-v{self.version}.receipt.json"
        # An intent with no receipt is deliberately terminal for this coordinator:
        # the server may already have persisted the snapshot, so never retry it.
        atomic_new_json(intent_path, intent)
        self.pending_training_snapshot = intent
        response = self.training_snapshot(name)
        validate_training_snapshot_receipt(response, self.session_id, name)
        atomic_new_json(receipt_path, {
            "schema_version": "reap.online-ttt.training-snapshot-receipt.v1",
            "session_id": self.session_id,
            "tree_id": self.tree_id,
            "checkpoint_step": step,
            "learn_count": len(self.receipts),
            "policy_version": self.version,
            "learn_event_id": event["event_id"],
            "snapshot": name,
            "response": response,
        })
        self.pending_training_snapshot = None

    def _validate(self, record: dict[str, Any]) -> None:
        if record.get("schema_version") != "reap.training.observer.v1":
            raise ValueError("unknown observer schema")
        if record.get("session_id") != self.session_id or record.get("tree_id") != self.tree_id:
            raise ValueError("observer session/tree mismatch")
        if record.get("sequence") != self.next_sequence:
            raise ValueError("observer sequence gap or replay")
        self.next_sequence += 1
        if record.get("policy_version") != self.version:
            raise ValueError("observer policy version mismatch")

    def _target(self, record: dict[str, Any]) -> tuple[dict[str, Any], tuple[Any, ...]] | None:
        gamma = record.get("gamma")
        if not isinstance(gamma, (int, float)) or isinstance(gamma, bool) or not math.isclose(gamma, self.gamma, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Lean and trainer gamma mismatch")
        if record.get("root_is_solved"):
            # This is only a search flag; final root replay has not happened.
            return None
        nodes = record["tree"]["nodes"]
        for node_index in sorted(self.backed_up_nodes, reverse=True):
            node = nodes[node_index]
            data = node["data"]
            count, total = data.get("numVisit"), data.get("valueSum")
            if data.get("toPlay") != "OR" or data.get("isSolved"):
                continue
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                continue
            if not isinstance(total, (int, float)) or not math.isfinite(total) or total / count > -1:
                continue  # unvisited/default/sentinel is not a value target
            candidates, prompts = [], set()
            for child in node.get("children", []):
                edge = child["edge"]
                if edge.get("isFocus"):
                    continue
                tactic = edge.get("tacticStr")
                source = self.edge_sources.get((node_index, tactic))
                if source is None:
                    raise ValueError("tree candidate has no generation provenance")
                if source["child_index"] != child["childIndex"]:
                    raise ValueError("tree edge child differs from its original eval provenance")
                prompts.add(source["prompt"])
                candidates.append({"tactic": tactic, "visits": edge["numVisit"],
                    "raw_logprob": source["raw_logprob"],
                    "behavior_version": source["policy_version"]})
            if not candidates or sum(item["visits"] for item in candidates) == 0:
                continue
            if len(prompts) != 1:
                raise ValueError("one search target mixes inference prompts")
            stats = (count, total, tuple((c["tactic"], c["visits"]) for c in candidates))
            if self.consumed_stats.get(node_index) == stats:
                continue
            event = {"kind": "search_visit_backup", "event_id": f"{self.session_id}.s{record['step']}.n{node_index}",
                "session_id": self.session_id, "tree_id": self.tree_id,
                "step": record["step"], "node_index": node_index,
                "policy_version": self.version, "prompt": next(iter(prompts)),
                "gamma": self.gamma, "candidates": candidates,
                "backup": {"value_sum": total, "visits": count, "kind": "OR", "valid": True},
                "reward": 0, "terminal_verified": False}
            if not SESSION_RE.fullmatch(event["event_id"]):
                raise ValueError("event id exceeds protocol limit")
            return event, stats
        return None

    def accept(self, record: dict[str, Any]) -> None:
        if self.pending_training_snapshot is not None:
            raise RuntimeError("training snapshot outcome is unknown; do not continue or retrain")
        self._validate(record)
        kind = record.get("kind")
        if self.pending_refresh is not None and kind not in ("checkpoint_ack", "selection_value_refresh"):
            raise ValueError("updated model requires selection-value-refresh before continuing search")
        if kind == "checkpoint_ack" and self.pending_refresh is not None:
            pending = self.pending_refresh
            if record.get("step") != pending["step"] or pending["acknowledged"]:
                raise ValueError("refresh barrier ACK is missing, duplicate or mismatched")
            pending["acknowledged"] = True
        elif kind == "selection_value_refresh":
            self._accept_selection_refresh(record)
        if kind == "generation":
            identity = tuple(record[k] for k in ("step", "node_index", "generation_index", "candidate_index"))
            if identity in self.generated:
                raise ValueError("duplicate generation identity")
            self.generated[identity] = record
            if self.version > 0:
                self.post_update_generations.append({k: record[k] for k in
                    ("sequence", "step", "node_index", "policy_version", "tree_id")})
        elif kind == "eval":
            identity = tuple(record[k] for k in ("step", "node_index", "generation_index", "candidate_index"))
            source = self.generated.pop(identity, None)
            if source is None or source["tactic"] != record["tactic"]:
                raise ValueError("eval does not match an observed generation")
            if record["disposition"] == "created":
                key = (record["node_index"], record["tactic"])
                if key in self.edge_sources or record.get("child_index") is None:
                    raise ValueError("invalid created edge provenance")
                self.edge_sources[key] = {**source, "child_index": record["child_index"]}
            # A merged duplicate/alias increments the original core edge prior,
            # not its visit count; keep that edge's FIRST creator provenance.
            elif record["disposition"] not in ("merged", "eval_rejected", "ancestor_rejected"):
                raise ValueError("unknown eval disposition")
        elif kind == "backup":
            self.backed_up_nodes.add(record["node_index"])
        elif kind == "checkpoint":
            step = record["step"]
            if step != self.last_step + 1:
                raise ValueError("checkpoint step gap/replay")
            self.last_step = step
            ack = {"session_id": self.session_id, "step": step,
                   "status": "continue", "policy_version": self.version}
            path = self.ack_dir / f"checkpoint-{step:06d}.ack.json"
            try:
                if self.deadline is not None:
                    self.deadline.check()
                target = self._target(record)
                # Checkpoints are numbered from zero. Interval N first permits
                # learning at N-1, then at least N checkpoints after the last
                # committed update. Missing targets do not consume a slot.
                # Still validate each current checkpoint above; never accumulate
                # skipped trees or fabricate a cross-checkpoint training event.
                interval_ready = (step + 1 >= self.min_checkpoint_interval and
                    (self.last_update_checkpoint is None or
                     step - self.last_update_checkpoint >= self.min_checkpoint_interval))
                if target is not None and len(self.receipts) < self.max_updates and interval_ready:
                    event, stats = target
                    validate_learn_body(self.version, event, max_request_body_bytes=self.max_request_body_bytes)
                    def submit():
                        self.requests.append(event)
                        # Persist intent before mutation, after admission. A
                        # refused admission creates no ambiguous pending intent.
                        atomic_new_json(self.ack_dir / f"learn-{step:06d}.request.json", event)
                        return self.learn(self.version, event)
                    receipt = self.deadline.call("learn", submit) if self.deadline is not None else submit()
                    validate_learn_receipt(receipt, event, self.version)
                    atomic_new_json(self.ack_dir / f"learn-{step:06d}.receipt.json", receipt)
                    self.version += 1
                    self.receipts.append(receipt)
                    self.last_update_checkpoint = step
                    self.consumed_stats[event["node_index"]] = stats
                    ack["policy_version"] = self.version
                    self._snapshot_after_confirmed_learn(step=step, event=event)
                    if self.selection_value_refresh:
                        self.pending_refresh = {"step": step, "version": self.version,
                            "nodes": record["tree"]["nodes"], "acknowledged": False}
                if self.deadline is not None:
                    self.deadline.check()
            except DeadlineExpired as exc:
                # This ACK stops Lean at a completed iteration. Any admitted
                # learn has already returned and its receipt was preserved.
                ack.update(status="error", error=str(exc), stop_reason="total_deadline")
                self.deadline_stopped = True
                self.deadline_refresh_skipped = self.pending_refresh is not None
                self.pending_refresh = None
            except BaseException as exc:
                ack.update(status="error", error=f"{type(exc).__name__}: {exc}")
                atomic_new_json(path, ack)
                raise
            atomic_new_json(path, ack)
            self.backed_up_nodes.clear()

    def _accept_selection_refresh(self, record: dict[str, Any]) -> None:
        pending = self.pending_refresh
        if not self.selection_value_refresh or pending is None or not pending["acknowledged"]:
            raise ValueError("unexpected selection-value-refresh or missing ACK")
        if (record.get("step") != pending["step"] or
                type(record.get("selection_value_version")) is not int or
                record["selection_value_version"] != pending["version"] or
                record.get("scope") != "selection-only-not-training-Q"):
            raise ValueError("selection-value-refresh version/step/scope mismatch")
        tree = pending["nodes"]
        expected = {i: node for i, node in enumerate(tree)
                    if not node["data"]["isSolved"] and node["data"]["numVisit"] > 0}
        changes = record.get("nodes")
        if not isinstance(changes, list):
            raise ValueError("refresh nodes must be an array")
        values = {}
        for change in changes:
            index, value = change.get("node_index"), change.get("value")
            if (type(index) is not int or index not in expected or index in values or
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError("invalid/duplicate/proved/unvisited refresh node")
            node = expected[index]
            kind = node["data"]["toPlay"]
            if change.get("node_kind") != kind or kind not in ("OR", "AND") or (kind == "OR" and value > 0):
                raise ValueError("refresh node kind/value mismatch")
            values[index] = value
        if values.keys() != expected.keys():
            raise ValueError("refresh omitted a visited unsolved node")
        for index, node in expected.items():
            if node["data"]["toPlay"] == "AND":
                child_values = [values[child["childIndex"]] for child in node["children"]
                                if child["childIndex"] in expected]
                if values[index] != min([1.0, *child_values]):
                    raise ValueError("refresh AND minimum differs from refreshed unsolved children")
        self.selection_refreshes.append({"step": pending["step"], "policy_version": pending["version"],
            "sequence": record["sequence"], "node_count": len(values), "scope": record["scope"]})
        self.pending_refresh = None

    def finish(self) -> None:
        if self.pending_refresh is not None:
            raise ValueError("Lean ended before required selection-value-refresh; check CPU overlay capability")


def _stop_owned_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run_online(*, session_id: str, project_dir: Path, theorem_file: str,
               output_root: Path, gpu_base_url: str, gamma: float,
               max_updates: int = 5, min_checkpoint_interval: int = 1,
               training_snapshot_interval: int | None = None,
               max_request_body_bytes: int = MAX_LEARN_BODY_BYTES,
               http_timeout: float = DEFAULT_BUDGET.client,
               total_deadline_seconds: float | None = None,
               barrier_timeout: int = DEFAULT_BUDGET.barrier, lean_bin: str = "lake",
               client: GpuHttpClient | None = None,
               experience_id: str | None = None, experience_candidate: bool = False,
               defer_experience_seal: bool = False,
               experience_weights_sha256: str | None = None, experience_snapshot_sha256: str | None = None,
               experience_policy: dict | None = None,
               selection_value_refresh: bool = False,
               command: list[str] | None = None) -> dict[str, Any]:
    deadline = SearchDeadline(total_deadline_seconds) if total_deadline_seconds is not None else None
    if type(selection_value_refresh) is not bool:
        raise ValueError("selection_value_refresh must be explicitly boolean")
    if type(defer_experience_seal) is not bool or (defer_experience_seal and not experience_candidate):
        raise ValueError('deferred experience seal requires an explicit experience candidate')
    if type(min_checkpoint_interval) is not int or min_checkpoint_interval < 1:
        raise ValueError("min_checkpoint_interval must be a positive integer")
    if training_snapshot_interval is not None and (
            type(training_snapshot_interval) is not int or training_snapshot_interval < 1):
        raise ValueError("training_snapshot_interval must be a positive integer or None")
    validate_body_limit(max_request_body_bytes)
    # Preserve legacy default records; an explicitly spaced run records its
    # schedule in both the immutable session specification and final report.
    schedule_fields = ({"min_checkpoint_interval": min_checkpoint_interval}
                       if min_checkpoint_interval != 1 else {})
    if training_snapshot_interval is not None:
        schedule_fields["training_snapshot_interval"] = training_snapshot_interval
    if defer_experience_seal:
        schedule_fields['defer_experience_seal'] = True
    if max_request_body_bytes != MAX_LEARN_BODY_BYTES:
        schedule_fields["max_request_body_bytes"] = max_request_body_bytes
    if not SESSION_RE.fullmatch(session_id) or len(session_id) > 40:
        raise ValueError("online session id must be valid and at most 40 characters")
    validate_experience_reference(experience_id, experience_weights_sha256, experience_snapshot_sha256)
    pinned = {"experience_weights_sha256": experience_weights_sha256,
              "experience_snapshot_sha256": experience_snapshot_sha256}
    if not any(value is not None for value in pinned.values()):
        pinned = {}
    project_dir, output_root = project_dir.resolve(), output_root.resolve()
    theorem_path = Path(theorem_file)
    if not theorem_path.is_absolute():
        theorem_path = project_dir / theorem_path
    theorem_sha = hashlib.sha256(theorem_path.read_bytes()).hexdigest()
    policy_fields = {}
    if experience_policy is not None:
        from .experience_policy import validate_policy, digest as policy_digest
        selected = validate_policy(experience_policy, session_id=session_id, theorem_sha256=theorem_sha, gamma=gamma)
        supplied = {"experience_id": experience_id, "experience_weights_sha256": experience_weights_sha256,
                    "experience_snapshot_sha256": experience_snapshot_sha256}
        if any(value is not None for value in supplied.values()) and supplied != {key: selected.get(key) for key in supplied}:
            raise ValueError("explicit experience pins differ from selected policy")
        experience_id = selected.get("experience_id")
        experience_weights_sha256 = selected.get("experience_weights_sha256")
        experience_snapshot_sha256 = selected.get("experience_snapshot_sha256")
        pinned = {key: value for key, value in selected.items() if key != "experience_id"}
        pinned["expected_initialization_contract_sha256"] = experience_policy["selection"]["expected_initialization_contract_sha256"]
        policy_fields = {"experience_policy_sha256": policy_digest(experience_policy),
                         "expected_initialization_contract_sha256": pinned["expected_initialization_contract_sha256"]}
    directory = output_root / session_id
    directory.mkdir(parents=True, exist_ok=False)
    if experience_policy is not None:
        atomic_new_json(directory / "experience-policy.json", experience_policy)
    tree_id = session_id + ".tree0"
    spec = SessionSpec(session_id, str(theorem_path),
        f"{gpu_base_url}/sessions/{session_id}/policy/v1",
        f"{gpu_base_url}/sessions/{session_id}/value/v1")
    initialization = ({"experience_id": experience_id, "experience_candidate": experience_candidate}
                      if experience_id is not None or experience_candidate or experience_policy is not None else {})
    atomic_new_json(directory / "session.json", {**asdict(spec), **initialization, **pinned, **policy_fields, **schedule_fields,
        "profile": "online-search-visit-backup-v1", "tree_id": tree_id,
        "theorem_sha256": theorem_sha, "gamma": gamma,
        "max_updates": max_updates, "total_deadline_seconds": total_deadline_seconds,
        "selection_value_refresh": selection_value_refresh})
    client = client or GpuHttpClient(gpu_base_url, timeout_seconds=http_timeout)

    def learn(version: int, event: dict[str, Any]) -> dict[str, Any]:
        return client._request("POST", f"/sessions/{session_id}/learn/v1",
                               {"expected_policy_version": version, "event": event})

    def training_snapshot(name: str) -> dict[str, Any]:
        return client.snapshot(session_id, name)

    coordinator = OnlineCoordinator(session_id, tree_id, directory / "checkpoints",
                                    learn, gamma=gamma, max_updates=max_updates,
                                    min_checkpoint_interval=min_checkpoint_interval,
                                    training_snapshot=training_snapshot,
                                    training_snapshot_interval=training_snapshot_interval,
                                    max_request_body_bytes=max_request_body_bytes,
                                    **({"deadline": deadline} if deadline is not None else {}))
    coordinator.selection_value_refresh = selection_value_refresh
    created = (client.create_session(session_id, theorem_id=theorem_sha, experience_id=experience_id, **pinned)
               if initialization else client.create_session(session_id))
    atomic_new_json(directory / "create-receipt.json", created)
    validate_created(created, session_id, gamma,
                     theorem_id=theorem_sha if initialization else None, experience_id=experience_id,
                     weights_sha256=experience_weights_sha256, snapshot_sha256=experience_snapshot_sha256)
    if defer_experience_seal:
        from gpu_runtime.success_finalize_objective import CONTRACT
        if created.get('value_metadata', {}).get('success_finalization') != CONTRACT:
            raise ValueError('deferred seal requires explicit successful-finalization backend capability')
    if experience_policy is not None:
        from .experience_policy import validate_execution_contract
        validate_execution_contract(created, experience_policy)
    atomic_new_json(directory / "snapshot-before.json", client.snapshot(session_id, "before-online-ttt"))
    env = os.environ.copy()
    env.update({"REAP_SESSION_ID": session_id, "REAP_SESSION_DIR": str(directory),
        "REAP_TREE_ID": tree_id, "REAP_POLICY_VERSION": "0",
        "REAP_POLICY_ENDPOINT": spec.policy_base_url,
        "REAP_VALUE_ENDPOINT": spec.value_base_url, "REAP_PS_ENDPOINT": "",
        "REAP_OBSERVER_PATH": str(directory / "observer.jsonl"),
        "REAP_CHECKPOINT_DIR": str(coordinator.ack_dir),
        "REAP_CHECKPOINT_TIMEOUT_SECONDS": str(barrier_timeout),
        "REAP_SELECTION_VALUE_REFRESH": "selection-value-refresh" if selection_value_refresh else ""})
    options: dict[str, Any] = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    proxy = None
    if deadline is not None:
        proxy = DeadlineProxy(deadline, gpu_base_url, session_id, http_timeout)
        env["REAP_POLICY_ENDPOINT"] = f"{proxy.url}/sessions/{session_id}/policy/v1"
        env["REAP_VALUE_ENDPOINT"] = f"{proxy.url}/sessions/{session_id}/value/v1"
        atomic_new_json(directory / "deadline.json", {
            "total_deadline_seconds": total_deadline_seconds,
            "semantics": "stop-new-requests-drain-admitted-v1",
            "policy_endpoint": env["REAP_POLICY_ENDPOINT"],
            "value_endpoint": env["REAP_VALUE_ENDPOINT"],
            "setup_and_final_snapshot_in_budget": False})
        deadline.start()
    start = time.monotonic()
    error = None
    tail = JsonlTail(directory / "observer.jsonl")
    with (proxy if proxy is not None else nullcontext()), (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command or [lean_bin, "env", "lean", str(theorem_path)],
            cwd=project_dir, env=env, stdout=stdout, stderr=stderr, **options)
        atomic_new_json(directory / "process.json", {"pid": process.pid,
            "command": command or [lean_bin, "env", "lean", str(theorem_path)],
            "tree_id": tree_id, "started_at_epoch": time.time()})
        try:
            while True:
                for event in tail.read():
                    coordinator.accept(event)
                if process.poll() is not None:
                    for event in tail.read():
                        coordinator.accept(event)
                    if tail.pending.strip():
                        raise ValueError("incomplete final observer record")
                    # A refresh denied by the shared budget is deliberately
                    # incomplete, not evidence that refreshed search occurred.
                    if not (deadline is not None and (coordinator.deadline_stopped or
                            any(not item['admitted'] for item in deadline.report()['requests']))):
                        coordinator.finish()
                    break
                time.sleep(0.02)
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc),
                     "mutation_retry_allowed": False}
            _stop_owned_process(process)
    budget_evidence = deadline.report() if deadline is not None else None
    if budget_evidence is not None and budget_evidence["unknown_or_pending"] and error is None:
        error = {"type": "UnknownSearchRequestOutcome", "message": "deadline gate recorded an uncertain request",
                 "mutation_retry_allowed": False}
    solved, status = _finalize_solver_status(directory / "result.json", process.returncode, False)
    budget_reached = deadline is not None and deadline.elapsed() >= deadline.seconds
    budget_stopped = budget_evidence is not None and (coordinator.deadline_stopped or
                     any(not item['admitted'] for item in budget_evidence['requests']))
    exercised = bool(coordinator.receipts and coordinator.post_update_generations)
    report = {"session_id": session_id, "tree_id": tree_id, "lean_pid": process.pid,
        "profile": "online-search-visit-backup-v1", "returncode": process.returncode,
        "root_verified": bool(solved and error is None),
        "online_update_consumed_by_later_generation": exercised,
        "status": "passed_execution" if solved and exercised and error is None else
                  ("solved_without_online_update" if solved and error is None else
                   ("total_deadline_reached" if budget_stopped and error is None else status)),
        "policy_version": coordinator.version, "optimizer_updates": len(coordinator.receipts),
        "training_enabled": max_updates > 0,
        "selection_value_refresh": selection_value_refresh,
        "selection_refreshes": coordinator.selection_refreshes,
        "experience_id": experience_id, **policy_fields, **schedule_fields,
        "checkpoint_count": coordinator.last_step + 1,
        "elapsed_seconds": time.monotonic() - start,
        "post_update_generations": coordinator.post_update_generations,
        "learn_receipts": coordinator.receipts, "error": error,
        "total_deadline_seconds": total_deadline_seconds,
        **({"deadline": budget_evidence,
            "deadline_reached": budget_reached,
            "deadline_checkpoint_stop": coordinator.deadline_stopped,
            "drain_overshoot_seconds": max(0.0, deadline.elapsed() - deadline.seconds),
            "selection_refresh_incomplete": coordinator.deadline_refresh_skipped or coordinator.pending_refresh is not None}
           if deadline is not None else {}),
        "note": "Execution gate only; independently audit wire versions, parameter changes and final proof."}
    # Snapshot only after all known mutations returned; an unknown outcome is
    # not repaired by further mutations or by starting the experiment again.
    if error is None:
        atomic_new_json(directory / "snapshot-after.json", client.snapshot(session_id, "after-online-ttt"))
        if experience_candidate and solved and coordinator.receipts and not defer_experience_seal:
            atomic_new_json(directory / "experience-candidate.json",
                client.snapshot(session_id, "experience-candidate", for_experience=True))
    atomic_new_json(directory / "online-result.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--theorem-file", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-base-url", required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--max-updates", type=int, default=5)
    parser.add_argument("--min-checkpoint-interval", type=int, default=1,
                        help="minimum checkpoints between committed updates; N first permits step N-1 (default 1)")
    parser.add_argument("--training-snapshot-interval", type=int,
                        help="opt in: snapshot mutable same-session training state after every N confirmed learns")
    parser.add_argument("--max-request-body-bytes", type=int, default=MAX_LEARN_BODY_BYTES,
                        help="explicit matching bridge limit, 8192..262144; does not enlarge the 4096-token training context")
    parser.add_argument("--experience-id", help="explicit immutable parameter release; never selects latest")
    parser.add_argument("--experience-weights-sha256", help="pin immutable inherited parameter content")
    parser.add_argument("--experience-snapshot-sha256", help="pin the accepted source snapshot")
    parser.add_argument("--experience-policy-record", type=Path,
                        help="explicit fresh/chain/bank catalog and selection; validates the actual initialization contract")
    parser.add_argument("--experience-candidate", action="store_true",
                        help="seal a solved, trained session for later independent acceptance")
    parser.add_argument('--defer-experience-seal', action='store_true',
                        help='opt-in terminal-learning capability: keep the candidate unsealed until independent replay/update')
    parser.add_argument("--selection-value-refresh", action="store_true",
                        help="requires explicit CPU 0004 overlay; refresh selection caches after each update")
    parser.add_argument("--http-timeout-seconds", type=float, default=DEFAULT_BUDGET.client)
    parser.add_argument("--total-deadline-seconds", type=float,
                        help="opt in: stop new search/learn requests after this budget, drain admitted work")
    parser.add_argument("--barrier-timeout-seconds", type=int, default=DEFAULT_BUDGET.barrier)
    parser.add_argument("--lean-bin", default="lake")
    args = parser.parse_args()
    if not math.isfinite(args.gamma) or not 0 < args.gamma < 1:
        parser.error("gamma must be finite and strictly between zero and one")
    if args.max_updates < 0 or args.barrier_timeout_seconds < 1:
        parser.error("update budget must be nonnegative and barrier timeout positive")
    if args.min_checkpoint_interval < 1:
        parser.error("minimum checkpoint interval must be positive")
    if args.training_snapshot_interval is not None and args.training_snapshot_interval < 1:
        parser.error("training snapshot interval must be positive")
    if not MAX_LEARN_BODY_BYTES <= args.max_request_body_bytes <= MAX_REQUEST_BODY_BYTES:
        parser.error("request body limit must be from 8192 to 262144 bytes")
    policy = None
    if args.experience_policy_record is not None:
        from .experience_policy import read_json
        policy = read_json(args.experience_policy_record)
    report = run_online(session_id=args.session_id, project_dir=args.project_dir,
        theorem_file=args.theorem_file, output_root=args.output_dir,
        gpu_base_url=args.gpu_base_url.rstrip("/"), gamma=args.gamma,
        max_updates=args.max_updates, min_checkpoint_interval=args.min_checkpoint_interval,
        training_snapshot_interval=args.training_snapshot_interval,
        max_request_body_bytes=args.max_request_body_bytes,
        http_timeout=args.http_timeout_seconds,
        total_deadline_seconds=args.total_deadline_seconds,
        barrier_timeout=args.barrier_timeout_seconds, lean_bin=args.lean_bin,
        experience_id=args.experience_id, experience_candidate=args.experience_candidate,
        defer_experience_seal=args.defer_experience_seal,
        experience_weights_sha256=args.experience_weights_sha256,
        experience_snapshot_sha256=args.experience_snapshot_sha256,
        experience_policy=policy,
        selection_value_refresh=args.selection_value_refresh)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if (report["status"] == "passed_execution" or
                 (args.max_updates == 0 and report["root_verified"])) else 2


if __name__ == "__main__":
    raise SystemExit(main())
