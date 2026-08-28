#!/usr/bin/env python3
"""One fixed-contract KL-guard case on AMD 7B, replaying a pinned v0 event.

Run accept and reject in separate processes with different, new --output-dir
values. No model download, experience inheritance, new Lean search, threshold
adaptation, or full-base hash occurs here. A native AdamW post-step hook only
observes private tensor fingerprints; it never replaces step or KL computation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import time

from gpu_runtime import GpuRuntime
from gpu_runtime.identifiers import validate_identifier
from gpu_runtime.search_backend import KLGuardExceeded, KL_REDUCTION, RealSearchBackend
from gpu_runtime.search_objective import prepare_search_event, validate_gamma

try:
    from .smoke_gpu import equal_tree, require, tensor_delta
    from .smoke_search_gpu import capture, phase
except ImportError:
    from smoke_gpu import equal_tree, require, tensor_delta
    from smoke_search_gpu import capture, phase

TARGET, CONTROL = "kl-replay-target", "kl-replay-control"
MODEL_REVISION = "fe76f68d9a88f342cb7b546307c20292fea9cced"


def positive_threshold(value) -> float:
    if isinstance(value, bool):
        raise ValueError("KL threshold must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("KL threshold must be finite and positive")
    return number


def encoded(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_bytes(path: Path, raw: bytes) -> dict:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return {"file": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read_event(path: Path, expected_sha256: str, gamma: float) -> tuple[bytes, dict, dict]:
    """Preserve all training fields and original versions; change only identity."""
    require(bool(re.fullmatch(r"[0-9a-f]{64}", expected_sha256)), "lowercase event SHA256 required")
    raw = path.read_bytes()
    require(len(raw) <= 2 * 1024 * 1024, "event exceeds 2 MiB")
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "source event SHA256 mismatch")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate event JSON key")
            result[key] = value
        return result

    source = json.loads(raw, object_pairs_hook=pairs,
                        parse_constant=lambda _: require(False, "nonfinite event JSON"))
    require(isinstance(source, dict), "source event must be an object, not a wire wrapper")
    sid = validate_identifier(source.get("session_id"), kind="source session_id")
    require(type(source.get("policy_version")) is int and source["policy_version"] == 0,
            "fresh replay requires an original policy_version=0 event; never relabel trained versions")
    prepare_search_event(source, session_id=sid, policy_version=0, gamma=gamma)
    event = deepcopy(source)
    event.update(session_id=TARGET, tree_id=TARGET + ".replay-tree", event_id=TARGET + ".replay-0")
    changed = {key: {"source": source[key], "replay": event[key]}
               for key in ("session_id", "tree_id", "event_id")}
    preserved = {key: value for key, value in source.items() if key not in changed}
    require(preserved == {key: value for key, value in event.items() if key not in changed},
            "replay modified training data")
    provenance = {"source_file_sha256": expected_sha256, "identity_rebindings": changed,
        "preserved_training_fields_sha256": hashlib.sha256(encoded(preserved)).hexdigest(),
        "candidate_behavior_versions_preserved": True,
        "source_version_zero_is_not_a_same_weights_or_on_policy_claim": True,
        "execution": "offline_replay_of_previously_recorded_search_feedback_NOT_new_Lean_search"}
    return raw, event, provenance


def host_rng(backend) -> dict:
    return {"cpu": backend.torch.get_rng_state().clone(),
            "device": (backend.torch.cuda.get_rng_state(backend.device).clone()
                       if backend.device.type == "cuda" else None)}


def save_capture(root: Path, name: str, backend, state: dict) -> dict:
    path = root / (name + ".pt")
    with path.open("xb") as handle:
        backend.torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"file": path.name, "sha256": digest.hexdigest(), "bytes": path.stat().st_size,
            "decoded_state_sha256": backend._tensor_manifest(state)["sha256"]}


def audit(runtime, backend, event: dict, *, case: str, output_dir: Path) -> dict:
    """Actual runtime transaction. CPU fixtures test orchestration, not the GPU gate."""
    require(case in ("accept", "reject"), "case must be accept or reject")
    maximum = positive_threshold(backend.max_post_update_kl)
    contract = deepcopy(backend._search_config())
    require(contract.get("kl_guard", {}).get("maximum") == maximum, "guard must be fixed in contract")
    prepare_search_event(event, session_id=TARGET, policy_version=0, gamma=backend.gamma)
    for sid in (TARGET, CONTROL):
        runtime.create_session(sid)
    before = {sid: capture(runtime, backend, sid) for sid in (TARGET, CONTROL)}
    artifacts = {"before_" + sid: save_capture(output_dir, "before-" + sid, backend, state)
                 for sid, state in before.items()}
    before_fingerprints = runtime.actor.submit(lambda: backend.session_fingerprints(TARGET))
    rng_before = runtime.actor.submit(lambda: host_rng(backend))
    observations = []

    def observe_step(optimizer, _args, _kwargs):
        # Native post-hook: called by the actual optimizer after its real step.
        # Do not call runtime.actor.submit from inside its actor worker.
        current = backend.session_fingerprints(TARGET)
        state = runtime.sessions.get(TARGET).snapshot()
        allowed = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        observations.append({"phase": "actual_AdamW_step_returned_before_KL_and_commit",
            "fingerprints": current,
            "parameter_deltas": backend._fingerprint_changes(before_fingerprints, current),
            "optimizer_state_entries": len(optimizer.state),
            "optimizer_steps": backend.sessions[TARGET].optimizer_steps,
            "policy_version": state["policy_version"],
            "pending_event_ids": state["buffer_metadata"]["pending_event_ids"],
            "committed_event_ids": sorted(state["event_receipts"]),
            "trainable_model_parameters_within_target_optimizer": all(
                not parameter.requires_grad or id(parameter) in allowed for parameter in backend.model.parameters())})

    optimizer = runtime.actor.submit(lambda: backend.sessions[TARGET].optimizer)
    require(isinstance(optimizer, backend.torch.optim.AdamW), "probe requires actual AdamW")
    handle = runtime.actor.submit(lambda: optimizer.register_step_post_hook(observe_step))
    receipt, rejection, unexpected = None, None, None
    phase("one-real-runtime-learn-" + case)
    started = time.perf_counter()
    try:
        receipt = runtime.learn(TARGET, expected_policy_version=0, event=event)
    except KLGuardExceeded as exc:
        rejection = {"type": type(exc).__name__, "detail": deepcopy(exc.detail)}
    except Exception as exc:
        unexpected = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        runtime.actor.submit(handle.remove)
    learn_seconds = time.perf_counter() - started
    rng_after = runtime.actor.submit(lambda: host_rng(backend))
    after = {sid: capture(runtime, backend, sid) for sid in (TARGET, CONTROL)}
    for sid, state in after.items():
        artifacts["after_" + sid] = save_capture(output_dir, "after-" + sid, backend, state)
    artifacts["host_rng"] = save_capture(output_dir, "host-rng-before-after", backend,
                                         {"before": rng_before, "after": rng_after})
    artifacts["tentative_observations"] = write_bytes(output_dir / "tentative-observations.json", encoded(observations))
    deltas = {group: tensor_delta(backend.torch, before[TARGET]["backend"][group], after[TARGET]["backend"][group])
              for group in ("adapter", "value_head", "optimizer")}
    guard = (receipt.get("detail", {}).get("kl_guard") if receipt else
             rejection["detail"] if rejection else None)
    tentative = observations[0] if len(observations) == 1 else {}
    tentative_deltas = tentative.get("parameter_deltas", {})
    target_before, target_after = before[TARGET], after[TARGET]
    fresh = all(state["backend"]["optimizer_steps"] == 0 and state["backend"]["examples_seen"] == 0
                and not state["backend"]["optimizer"]["state"] and state["metadata"]["policy_version"] == 0
                and not state["metadata"]["lineage"] and not state["metadata"]["event_receipts"]
                and not state["metadata"]["buffer_metadata"]["events"] for state in before.values())
    numeric_guard = (isinstance(guard, dict) and type(guard.get("post_update_kl")) in (int, float)
                     and math.isfinite(guard["post_update_kl"]) and guard["post_update_kl"] >= 0)
    gates = {"fresh_sessions_without_experience": fresh,
        "fixed_contract_unchanged": contract == backend._search_config(),
        "exactly_one_actual_optimizer_step": len(observations) == 1,
        "tentative_adapter_changed": tentative_deltas.get("adapter", {}).get("changed_tensors", 0) > 0,
        "tentative_value_head_changed": tentative_deltas.get("value_head", {}).get("changed_tensors", 0) > 0,
        "tentative_Adam_moments_created": tentative_deltas.get("optimizer", {}).get("added_tensors", 0) > 0,
        "tentative_not_committed": (tentative.get("optimizer_steps") == 0 and tentative.get("policy_version") == 0
            and tentative.get("pending_event_ids") == [event["event_id"]] and tentative.get("committed_event_ids") == []),
        "optimizer_scope_correct": tentative.get("trainable_model_parameters_within_target_optimizer") is True,
        "actual_finite_post_step_guard": bool(numeric_guard and guard["maximum"] == maximum
            and guard["timing"] == "after_optimizer_step_before_commit" and guard["reduction"] == KL_REDUCTION),
        "control_exactly_unchanged": equal_tree(backend.torch, before[CONTROL], after[CONTROL]),
        "host_RNG_exactly_unchanged": equal_tree(backend.torch, rng_before, rng_after),
        "no_unexpected_exception": unexpected is None}
    if case == "accept":
        gates.update({"accepted_version_one": bool(receipt and receipt.get("applied") is True
            and receipt.get("idempotent") is False and receipt.get("policy_version") == 1
            and target_after["metadata"]["policy_version"] == 1 and rejection is None),
            "accepted_within_threshold": bool(numeric_guard and guard["accepted"] is True and guard["post_update_kl"] <= maximum),
            "adapter_and_head_committed": all(deltas[key]["changed_tensors"] > 0 for key in ("adapter", "value_head")),
            "one_Adam_step_committed": target_after["backend"]["optimizer_steps"] == 1
                and target_after["backend"]["examples_seen"] == 1 and deltas["optimizer"]["added_tensors"] > 0,
            "one_receipt_and_consumed_event": list(target_after["metadata"]["event_receipts"]) == [event["event_id"]]
                and target_after["metadata"]["event_receipts"][event["event_id"]]["response"] == receipt
                and target_after["metadata"]["buffer_metadata"]["pending_event_ids"] == []
                and target_after["metadata"]["buffer_metadata"]["consumed_event_ids"] == [event["event_id"]]})
    else:
        gates.update({"rejected_exact_KL_exception": rejection is not None and receipt is None,
            "rejected_above_threshold": bool(numeric_guard and guard["accepted"] is False and guard["post_update_kl"] > maximum),
            "rollback_exact_full_target": equal_tree(backend.torch, target_before, target_after),
            "version_zero_no_committed_receipt": target_after["metadata"]["policy_version"] == 0
                and target_after["metadata"]["event_receipts"] == {} and target_after["backend"]["optimizer_steps"] == 0})
    return {"ok": all(gates.values()), "case": case, "gates": gates, "training_contract": contract,
        "contract_sha256": hashlib.sha256(encoded(contract)).hexdigest(), "learn_receipt": receipt,
        "guard_rejection": rejection, "unexpected_exception": unexpected, "post_step_guard": guard,
        "committed_parameter_deltas": deltas, "artifacts": artifacts, "learn_wall_seconds": learn_seconds,
        "timing_scope": "runtime_learn_plus_private_state_post_step_observer; excludes_model_load_and_disk_artifacts; NOT_kernel_time",
        "rollback_scope": "adapter_value_head_Adam_private_RNG_counters_full_logical_metadata_buffer_and_receipts",
        "Adam_scope": "fresh_empty_Adam_to_real_tentative_moments; populated_preexisting_moments_NOT_tested_here",
        "optimizer_step_observer": "native_read_only_register_step_post_hook; no_mocked_step_or_KL"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--event-file", type=Path, required=True)
    parser.add_argument("--event-sha256", required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--case", choices=("accept", "reject"), required=True)
    parser.add_argument("--max-post-update-kl", type=positive_threshold, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new exclusive runroot; one case per process")
    args = parser.parse_args(argv)
    gamma = validate_gamma(args.gamma)
    raw, event, provenance = read_event(args.event_file, args.event_sha256, gamma)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"schema_version": "reap.kl-guard-gpu-replay.v1", "ok": False, "case": args.case,
        "real_7B_GPU_gate_passed": False, "new_Lean_search_verified": False,
        "new_proof_verified": False, "performance_improvement_claimed": False,
        "full_frozen_base_content_hash_checked": False, "experience_inherited": False,
        "input_provenance": provenance, "maximum": args.max_post_update_kl}
    source_root = Path(__file__).resolve().parents[2]
    names = ("containers/gpu/smoke_kl_guard.py", "containers/gpu/smoke_search_gpu.py", "containers/gpu/smoke_gpu.py",
             "gpu_runtime/search_backend.py", "gpu_runtime/real_backend.py", "gpu_runtime/runtime.py",
             "gpu_runtime/search_objective.py", "gpu_runtime/session_store.py", "gpu_runtime/actor.py")
    intent = {"case": args.case, "maximum": args.max_post_update_kl, "gamma": gamma,
              "one_process_one_fixed_contract": True, "model_path": str(args.model_path.resolve()),
              "input_provenance": provenance,
              "source_sha256": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in names}}
    write_bytes(args.output_dir / "intent.json", encoded(intent))
    write_bytes(args.output_dir / "source-event.json", raw)
    write_bytes(args.output_dir / "replay-event.json", encoded(event))
    try:
        import torch
        require(bool(torch.version.hip) and torch.cuda.is_available(), "usable AMD ROCm/HIP required")
        require(args.model_path.is_dir(), "already downloaded model directory required")
        lock_raw = (args.model_path / "reap-model-lock.json").read_bytes()
        lock = json.loads(lock_raw)
        require(lock.get("schema_version") == "reap.model-lock.v2" and lock.get("repo") == "FrenzyMath/REAL-Prover"
                and lock.get("revision") == MODEL_REVISION and lock.get("hidden_size") == 3584,
                "fixed REAL-Prover model lock required")
        report["model_lock"] = {"sha256": hashlib.sha256(lock_raw).hexdigest(), "revision": MODEL_REVISION,
                                "meaning": "recorded_identity_only; model_files_not_rehashed_by_this_probe"}
        phase("load-fresh-model-one-fixed-" + args.case + "-contract")
        backend = RealSearchBackend(str(args.model_path), gamma=gamma, max_post_update_kl=args.max_post_update_kl)
        require(backend.hidden_size == 3584 and backend.device.type == "cuda", "actual AMD 7B backend required")
        require(all(parameter.device == backend.device for parameter in backend.model.parameters()), "model device mismatch")
        report.update(torch_version=torch.__version__, torch_hip=torch.version.hip,
            device=torch.cuda.get_device_name(backend.device),
            dependencies={name: importlib.metadata.version(name) for name in ("transformers", "peft")})
        with GpuRuntime(backend=backend, snapshot_root=args.output_dir / "snapshots", max_resident_sessions=2) as runtime:
            report.update(audit(runtime, backend, event, case=args.case, output_dir=args.output_dir))
        report["real_7B_GPU_gate_passed"] = report["ok"]
    except Exception as exc:
        report.update(ok=False, error_type=type(exc).__name__, error=str(exc))
    report["total_wall_seconds"] = time.perf_counter() - started
    report["source_sha256"] = intent["source_sha256"]
    write_bytes(args.output_dir / "report.json", encoded(report))
    print(json.dumps({"ok": report["ok"], "case": args.case, "report": str(args.output_dir / "report.json")}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
