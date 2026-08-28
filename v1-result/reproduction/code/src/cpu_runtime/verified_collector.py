"""Fixed-release inference collector for the independent verified replay profile.

No learning, implicit release selection, old sigmoid target conversion, or
automatic mutation retry. Exported proofs remain candidates until independent
Lean verification and the existing verified-trajectory replay gate pass.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import textwrap
import time
from typing import Any

from .batch_solver import SessionSpec, SESSION_RE
from .http_clients import GpuHttpClient
from .online_ttt import JsonlTail, atomic_new_json, json_bytes, _stop_owned_process
from .transport_budget import DEFAULT_BUDGET
from .verified_trajectory import plan_success_path, write_new

PROFILE = "verified-release-collector-v1"
OBJECTIVE = "verified_success_replay"
MIXED_OBJECTIVE = "verified_replay_mathlib_sft"
VALUE_SEMANTICS = "verified_negative_longest_branch_categorical.v1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
THEOREM = re.compile(r"[A-Za-z_][A-Za-z0-9_'.]*(?:\.[A-Za-z_][A-Za-z0-9_']*)*\Z")
RESETS = ["optimizer", "rng", "buffer", "policy_version", "event_receipts"]
KINDS = {"selection", "generation", "eval", "backup", "checkpoint", "checkpoint_ack"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: Any) -> bool:
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def gamma_option(value: Any) -> int:
    require(type(value) in (int, float) and math.isfinite(value) and 0 < value < 1,
            "puct_value_gamma must be finite and strictly between zero and one")
    scaled = round(value * 1000)
    require(1 <= scaled <= 999 and math.isclose(value, scaled/1000, rel_tol=0, abs_tol=1e-15),
            "Reap puct_value_gamma must be exactly representable in thousandths")
    return scaled


def validate_created(created: dict, session_id: str, *, theorem_sha256: str,
                     model_release_sha256: str, learner_objective: str = OBJECTIVE) -> int:
    """Declared role, provenance and reset gate; not a tensor equality audit."""
    require(digest(theorem_sha256) and digest(model_release_sha256), "explicit SHA256 pins required")
    require(isinstance(created, dict), "created actor receipt must be an object")
    lineage, metadata = created.get("lineage"), created.get("value_metadata")
    require(created.get("schema_version") == "reap.gpu.session.v1"
            and created.get("session_id") == session_id and created.get("theorem_id") == theorem_sha256
            and created.get("role") == "actor" and created.get("completed") is False
            and type(created.get("policy_version")) is int and created["policy_version"] == 0,
            "created actor identity/role/local version mismatch")
    require(isinstance(lineage, dict)
            and set(lineage) == {"model_release_sha256", "weights_sha256", "source", "reset"}
            and lineage["model_release_sha256"] == model_release_sha256
            and digest(lineage["weights_sha256"]) and lineage["reset"] == RESETS,
            "created actor release/reset lineage mismatch")
    source = lineage["source"]
    require(isinstance(source, dict)
            and set(source) == {"source_kind", "learner_id", "learner_step", "checkpoint_sha256"}
            and source["source_kind"] == "learner_checkpoint"
            and isinstance(source["learner_id"], str) and SESSION_RE.fullmatch(source["learner_id"])
            and source["learner_id"] != session_id
            and type(source["learner_step"]) is int and source["learner_step"] >= 1
            and digest(source["checkpoint_sha256"]), "invalid learner checkpoint source")
    require(isinstance(created.get("optimizer_metadata"), dict)
            and type(created["optimizer_metadata"].get("steps")) is int
            and created["optimizer_metadata"]["steps"] == 0
            and created.get("buffer_metadata") == {"events": {}, "pending_event_ids": [], "consumed_event_ids": []}
            and created.get("event_receipts") == {}, "actor private training state must start empty")
    require(learner_objective in (OBJECTIVE, MIXED_OBJECTIVE), "unknown collector learner objective")
    require(isinstance(metadata, dict) and metadata.get("objective") == learner_objective
            and metadata.get("value_semantics") == VALUE_SEMANTICS,
            "created actor must use verified replay categorical semantics")
    expected_return = "negative_integer_longest_generated_action_branch"
    if learner_objective == MIXED_OBJECTIVE:
        expected_return = "negative_integer_verified_action_longest_branch; human_profile_linear_remaining_actions"
        require(metadata.get("max_batch_samples") == 10 and type(metadata["max_batch_samples"]) is int
                and json_bytes(metadata.get("mixture")) == json_bytes({
                    "source_counts": {"replay": 9, "mathlib_sft": 1}, "sample_weight": 0.1,
                    "sampling_unit": "verified_action_row", "ratio_scope": "every_complete_batch",
                    "source_profiles": {"replay": "verified-generated-action-negative-longest-branch-v1",
                        "mathlib_sft": "mathlib_sft_linear_negative_remaining_actions_v1"},
                    "source_losses": {"replay": ["policy", "value"], "mathlib_sft": ["policy", "value"]}}),
                "mixed actor requires the explicit 9/1 training contract")
    support = metadata.get("support", {})
    maximum = support.get("distance_max")
    require(type(support.get("distance_min")) is int and support["distance_min"] == 1
            and type(maximum) is int and 2 <= maximum <= 4096
            and support.get("return") == expected_return
            and support.get("overflow") == "reject", "invalid categorical distance support")
    return maximum


class CollectorCoordinator:
    """A strict, fixed-v0 observer consumer with an initial identity barrier."""
    def __init__(self, session_id: str, tree_id: str, ack_dir: Path, *,
                 model_release_sha256: str, puct_value_gamma: float, max_distance: int):
        require(SESSION_RE.fullmatch(session_id) is not None and SESSION_RE.fullmatch(tree_id) is not None,
                "invalid collector session/tree identity")
        require(digest(model_release_sha256), "explicit model release SHA256 required")
        gamma_option(puct_value_gamma)
        require(type(max_distance) is int and 2 <= max_distance <= 4096, "invalid distance support")
        self.sid, self.tid, self.acks = session_id, tree_id, ack_dir
        self.release, self.gamma, self.maximum = model_release_sha256, puct_value_gamma, max_distance
        self.sequence, self.last_step, self.pending_ack, self.ready = 0, -1, None, False
        self.last_checkpoint = None
        self.generated: dict[tuple, dict] = {}
        self.events: list[dict] = []
        ack_dir.mkdir(parents=True, exist_ok=False)

    def accept(self, event: dict) -> None:
        require(isinstance(event, dict) and event.get("schema_version") == "reap.training.observer.v1"
                and event.get("session_id") == self.sid and event.get("tree_id") == self.tid,
                "collector observer identity/schema mismatch")
        require(type(event.get("sequence")) is int and event["sequence"] == self.sequence,
                "collector observer sequence gap/replay")
        require(type(event.get("policy_version")) is int and event["policy_version"] == 0,
                "fixed release actor policy version changed")
        kind = event.get("kind")
        if not self.ready:
            require(kind == "collector_contract" and event.get("profile") == PROFILE
                    and event.get("model_release_sha256") == self.release
                    and event.get("return_discount") == 1 and type(event.get("return_discount")) is int
                    and event.get("puct_value_gamma") == self.gamma
                    and event.get("max_distance") == self.maximum
                    and event.get("value_adapter") == "positive-distance-negated-once"
                    and event.get("training_enabled") is False, "missing or incompatible Lean collector contract")
            atomic_new_json(self.acks / "collector-ready.ack.json", {"session_id": self.sid,
                "tree_id": self.tid, "status": "continue", "policy_version": 0,
                "model_release_sha256": self.release})
            self.ready = True
        else:
            require(kind in KINDS, "unexpected collector event; updates/refresh are forbidden")
            if self.pending_ack is not None:
                require(kind == "checkpoint_ack" and event.get("step") == self.pending_ack,
                        "search continued before exact fixed-version ACK")
                self.pending_ack = None
            else:
                require(kind != "checkpoint_ack", "unexpected/duplicate collector ACK")
            if kind == "generation":
                value = event.get("search_value")
                require(type(value) in (int, float) and math.isfinite(value) and -self.maximum <= value <= -1,
                        "collector search value must be a negative distance in support")
                key = tuple(event.get(k) for k in ("step", "node_index", "generation_index", "candidate_index"))
                require(all(type(v) is int and v >= 0 for v in key) and key not in self.generated,
                        "invalid/duplicate generation identity")
                self.generated[key] = event
            elif kind == "eval":
                key = tuple(event.get(k) for k in ("step", "node_index", "generation_index", "candidate_index"))
                previous = self.generated.pop(key, None)
                require(previous is not None and previous.get("tactic") == event.get("tactic")
                        and event.get("disposition") in {"created", "merged", "eval_rejected", "ancestor_rejected"},
                        "collector eval does not match generation")
            elif kind == "checkpoint":
                step = event.get("step")
                require(type(step) is int and step == self.last_step + 1
                        and event.get("gamma") == self.gamma and not self.generated,
                        "collector checkpoint step/gamma/incomplete generation mismatch")
                require(type(event.get("root_is_solved")) is bool and isinstance(event.get("tree"), dict),
                        "invalid checkpoint tree/root")
                atomic_new_json(self.acks / f"checkpoint-{step:06d}.ack.json",
                    {"session_id": self.sid, "step": step, "status": "continue", "policy_version": 0})
                self.last_step, self.pending_ack, self.last_checkpoint = step, step, event
        self.sequence += 1
        self.events.append(event)

    def finish(self) -> None:
        require(self.ready and self.last_checkpoint is not None and self.pending_ack is None
                and not self.generated and self.events[-1]["kind"] == "checkpoint_ack",
                "collector ended without complete final checkpoint/ACK")


def classify_terminal(directory: Path, coordinator: CollectorCoordinator, returncode: int) -> tuple[str, dict]:
    result = json.loads((directory / "result.json").read_bytes())
    tree = json.loads((directory / "raw_tree.json").read_bytes())
    frame = coordinator.last_checkpoint
    require(result.get("schema_version") == "reap.training.result.v1"
            and result.get("session_id") == coordinator.sid and isinstance(frame, dict)
            and frame["tree"].get("nodes") == tree.get("nodes"), "terminal result/tree identity mismatch")
    if (returncode == 0 and result.get("solved") is True and result.get("status") == "solved"
            and result.get("error") is None and frame["root_is_solved"] is True
            and tree.get("solution") == 0 and type(tree.get("solution")) is int):
        require(isinstance(result.get("proof_script"), str) and result["proof_script"].strip(),
                "solved result has no proof script")
        return "solved_pending_independent_verification", result
    if (returncode == 1 and result.get("solved") is False and result.get("status") == "exhausted"
            and result.get("error") is None and result.get("proof_script") is None
            and frame["root_is_solved"] is False and tree.get("solution") is None):
        return "exhausted", result
    raise ValueError("collector has no consistent known terminal outcome")


def run_collector(*, session_id: str, project_dir: Path, theorem_file: str, theorem: str,
                  output_root: Path, gpu_base_url: str, model_release_sha256: str,
                  puct_value_gamma: float, return_discount: int = 1,
                  http_timeout: float = DEFAULT_BUDGET.client,
                  barrier_timeout: int = DEFAULT_BUDGET.barrier, lean_bin: str = "lake",
                  client: GpuHttpClient | None = None, command: list[str] | None = None,
                  learner_objective: str = OBJECTIVE) -> dict:
    require(isinstance(session_id, str) and SESSION_RE.fullmatch(session_id) is not None and len(session_id) <= 40,
            "collector session id must be valid and at most 40 characters")
    require(digest(model_release_sha256), "explicit model_release_sha256 required")
    require(learner_objective in (OBJECTIVE, MIXED_OBJECTIVE), "unknown collector learner objective")
    require(type(return_discount) is int and return_discount == 1, "verified return_discount must remain 1")
    option = gamma_option(puct_value_gamma)
    require(isinstance(theorem, str) and THEOREM.fullmatch(theorem) is not None, "invalid theorem declaration name")
    require(type(barrier_timeout) is int and barrier_timeout > 0, "barrier timeout must be positive")
    project_dir, output_root = project_dir.resolve(), output_root.resolve()
    path = Path(theorem_file)
    if not path.is_absolute():
        path = project_dir / path
    source = path.read_bytes()
    text = source.decode("utf8")
    require(text.count("  reapTrainingMCTS") == 1, "collector requires one explicit source tactic marker")
    theorem_sha = hashlib.sha256(source).hexdigest()
    directory = output_root / session_id
    directory.mkdir(parents=True, exist_ok=False)
    tid = session_id + ".tree0"
    spec = SessionSpec(session_id, str(path), f"{gpu_base_url}/sessions/{session_id}/policy/v1",
                       f"{gpu_base_url}/sessions/{session_id}/value/v1")
    session = {**asdict(spec), "profile": PROFILE, "tree_id": tid, "theorem": theorem,
        "theorem_sha256": theorem_sha, "model_release_sha256": model_release_sha256,
        "return_discount": 1, "puct_value_gamma": puct_value_gamma, "training_enabled": False,
        "policy_version": 0, "total_deadline_seconds": None}
    # The Lean fixed-v0 collector/value adapter is unchanged. Only an explicit
    # caller choice permits a mixed release; old default records stay identical.
    if learner_objective != OBJECTIVE:
        session["learner_objective"] = learner_objective
    atomic_new_json(directory / "session-intent.json", session)
    write_new(directory / "source.lean", source)
    run_source = "import Reap.VerifiedCollector\n" + text.replace("  reapTrainingMCTS",
        f"  set_option reap.visit_discount {option} in\n    reapVerifiedCollectorMCTS")
    run_path = directory / "run.lean"
    write_new(run_path, run_source.encode())
    atomic_new_json(directory / "create-intent.json", {"session_id": session_id,
        "theorem_id": theorem_sha, "model_release_sha256": model_release_sha256,
        "mutation_retry_allowed": False})
    client = client or GpuHttpClient(gpu_base_url, timeout_seconds=http_timeout)
    process, coordinator, error, status = None, None, None, "failed_unknown"
    started = time.monotonic()
    try:
        created = client.create_session(session_id, theorem_id=theorem_sha,
                                        model_release_sha256=model_release_sha256)
        atomic_new_json(directory / "create-receipt.json", created)
        maximum = validate_created(created, session_id, theorem_sha256=theorem_sha,
                                   model_release_sha256=model_release_sha256, learner_objective=learner_objective)
        # Keep the real actor provenance inside the existing 17-file replay
        # bundle, whose session.json is hashed. Do not relabel this as old TTT.
        session.update(role="actor", lineage=created["lineage"], initialization_receipt=created,
            initialization_receipt_sha256=hashlib.sha256(json_bytes(created)).hexdigest())
        atomic_new_json(directory / "session.json", session)
        coordinator = CollectorCoordinator(session_id, tid, directory / "checkpoints",
            model_release_sha256=model_release_sha256, puct_value_gamma=puct_value_gamma, max_distance=maximum)
        env = os.environ.copy()
        env.update(REAP_SESSION_ID=session_id, REAP_SESSION_DIR=str(directory), REAP_TREE_ID=tid,
            REAP_POLICY_VERSION="0", REAP_POLICY_ENDPOINT=spec.policy_base_url,
            REAP_VALUE_ENDPOINT=spec.value_base_url, REAP_PS_ENDPOINT="", REAP_SELECTION_VALUE_REFRESH="",
            REAP_OBSERVER_PATH=str(directory / "observer.jsonl"), REAP_CHECKPOINT_DIR=str(coordinator.acks),
            REAP_CHECKPOINT_TIMEOUT_SECONDS=str(barrier_timeout),
            REAP_COLLECTOR_RELEASE_SHA256=model_release_sha256,
            REAP_COLLECTOR_PUCT_GAMMA_OPTION=str(option), REAP_COLLECTOR_MAX_DISTANCE=str(maximum),
            REAP_COLLECTOR_RETURN_DISCOUNT="1")
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        tail = JsonlTail(directory / "observer.jsonl")
        command = command or [lean_bin, "env", "lean", str(run_path)]
        with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(command, cwd=project_dir, env=env, stdout=stdout, stderr=stderr, **options)
            atomic_new_json(directory / "process.json", {"pid": process.pid, "command": command,
                "tree_id": tid, "started_at_epoch": time.time(), "run_source_sha256": hashlib.sha256(run_source.encode()).hexdigest()})
            while True:
                for event in tail.read():
                    coordinator.accept(event)
                if process.poll() is not None:
                    for event in tail.read():
                        coordinator.accept(event)
                    require(not tail.pending.strip(), "incomplete final observer record")
                    coordinator.finish()
                    break
                time.sleep(0.02)
        status, result = classify_terminal(directory, coordinator, process.returncode)
        require(path.read_bytes() == source, "theorem source changed during attempt")
        if status == "solved_pending_independent_verification":
            tree = json.loads((directory / "raw_tree.json").read_bytes())
            candidate = plan_success_path(tree, coordinator.events, session)
            require(result["proof_script"] == candidate["proof_script"], "selected proof differs from result")
            proof = text.replace("  reapTrainingMCTS", textwrap.indent(candidate["proof_script"], "  "))
            proof += "\n#print axioms " + theorem + "\n"
            write_new(directory / "proof.lean", proof.encode())
            atomic_new_json(directory / "proof-candidate.json", {"proof_from_session": session_id,
                "source_theorem_sha256": theorem_sha, "generated_proof_sha256": hashlib.sha256(proof.encode()).hexdigest(),
                "model_release_sha256": model_release_sha256, "independent_verified": False,
                "candidate_root_return": candidate["root_return"], "candidate_rows": len(candidate["rows"])})
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "mutation_retry_allowed": False}
        status = "failed_unknown"
        if process is not None:
            _stop_owned_process(process)
    report = {"schema_version": "reap.verified-collector.result.v1", **session,
        "status": status, "root_verified": status == "solved_pending_independent_verification",
        "independent_verified": False, "optimizer_updates": 0, "error": error,
        "returncode": None if process is None else process.returncode,
        "lean_pid": None if process is None else process.pid,
        "checkpoint_count": 0 if coordinator is None else coordinator.last_step + 1,
        "elapsed_seconds": time.monotonic() - started, "mutation_retry_allowed": False,
        "note": "No learn/snapshot/retire automatically submitted. Proof requires independent Lean acceptance and replay."}
    atomic_new_json(directory / "collector-result.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("session-id", "theorem-file", "theorem", "gpu-base-url", "model-release-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--puct-value-gamma", type=float, required=True)
    parser.add_argument("--return-discount", type=int, default=1)
    parser.add_argument("--learner-objective", choices=(OBJECTIVE, MIXED_OBJECTIVE), default=OBJECTIVE)
    parser.add_argument("--barrier-timeout-seconds", type=int, default=DEFAULT_BUDGET.barrier)
    args = parser.parse_args(argv)
    result = run_collector(session_id=args.session_id, project_dir=args.project_dir,
        theorem_file=args.theorem_file, theorem=args.theorem, output_root=args.output_dir,
        gpu_base_url=args.gpu_base_url.rstrip("/"), model_release_sha256=args.model_release_sha256,
        puct_value_gamma=args.puct_value_gamma, return_discount=args.return_discount,
        barrier_timeout=args.barrier_timeout_seconds, learner_objective=args.learner_objective)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "solved_pending_independent_verification" else 2


if __name__ == "__main__":
    raise SystemExit(main())
