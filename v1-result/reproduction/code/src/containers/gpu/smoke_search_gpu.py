#!/usr/bin/env python3
"""Strict search-objective ROCm gate using synthetic data, NOT Lean online TTT.

Two sessions coexist, A performs exactly one optimizer update, and B remains
untouched.  This tests actual weights only when run on the required AMD GPU.
It neither starts an instance nor downloads a model nor depends on cpu_runtime.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path
import sys
import time

from gpu_runtime import GpuRuntime
from gpu_runtime.search_backend import RealSearchBackend, SNAPSHOT_SCHEMA
from gpu_runtime.search_objective import VALUE_SEMANTICS, validate_gamma

try:
    from .smoke_gpu import (PROMPT, base_fingerprint, equal_tree, finite_tree,
                            require, require_update)
except ImportError:  # Direct script execution inside the GPU image/upload.
    from smoke_gpu import (PROMPT, base_fingerprint, equal_tree, finite_tree,
                           require, require_update)


def synthetic_event(session_id: str, gamma: float) -> dict:
    """Positive visit counts and a legal backup; no claim of Lean verification."""
    return {"kind": "search_visit_backup", "event_id": session_id + ".synthetic.0",
            "session_id": session_id, "tree_id": session_id + ".synthetic-tree",
            "step": 0, "node_index": 0, "policy_version": 0, "prompt": PROMPT,
            "gamma": gamma,
            "candidates": [
                {"tactic": "exact True.intro", "visits": 3,
                 "raw_logprob": -2.0, "behavior_version": 0},
                {"tactic": "constructor", "visits": 1,
                 "raw_logprob": -1.0, "behavior_version": 0}],
            "backup": {"value_sum": -12.0, "visits": 4, "kind": "OR", "valid": True},
            "reward": 0, "terminal_verified": False,
            "source": "synthetic_GPU_numerics_NOT_Lean_observer_or_verifier"}


def capture(runtime: GpuRuntime, backend: RealSearchBackend, session_id: str) -> dict:
    raw = runtime.inspect_backend(session_id)
    require(raw.get("schema_version") == SNAPSHOT_SCHEMA, "wrong strict snapshot schema")
    require(raw.get("session_id") == session_id, "snapshot belongs to another session")
    require(raw.get("search_config") == backend._search_config(), "snapshot search configuration mismatch")
    require(raw.get("encoding") == "torch-save-base64", "unexpected snapshot encoding")
    state = backend.torch.load(io.BytesIO(base64.b64decode(raw["payload"], validate=True)),
                               map_location="cpu", weights_only=True)
    require(isinstance(state, dict), "decoded backend state must be an object")
    for key in ("adapter", "value_head", "optimizer", "optimizer_steps", "examples_seen"):
        require(key in state, f"missing mutable snapshot state: {key}")
    finite_tree(backend.torch, state, session_id)
    metadata = runtime.actor.submit(lambda: runtime.sessions.get(session_id).snapshot())
    return {"backend": state, "metadata": metadata}


def value_probe(runtime: GpuRuntime, backend: RealSearchBackend, session_id: str, version: int) -> dict:
    result = runtime.value(session_id, {"model": "REAL-Prover",
        "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 1})
    score = json.loads(result["choices"][0]["message"]["content"])["score"]
    maximum = backend._search_config()["max_distance"]
    require(isinstance(score, (int, float)) and math.isfinite(score) and 1 <= score <= maximum,
            "value endpoint did not return a bounded nonterminal distance")
    require(result["policy_version"] == version, "value response policy version mismatch")
    return {"score": score, "policy_version": result["policy_version"],
            "transport_semantics": "distance_1_plus_log_gamma_discounted_return"}


def phase(name: str) -> None:
    print(json.dumps({"phase": name, "time": time.time()}), file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/opt/models/REAL-Prover")
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--snapshot-root", type=Path,
                        default=Path("/workspace/out/search-smoke-snapshots"))
    parser.add_argument("--output", type=Path, help="new report file; an existing file is never overwritten")
    args = parser.parse_args()
    gamma = validate_gamma(args.gamma)
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    import torch
    require(bool(torch.version.hip), "ROCm/HIP is required; CPU/CUDA-only is not this gate")
    require(torch.cuda.is_available(), "no usable AMD ROCm GPU")
    require(Path(args.model_path).is_dir(), "an already-downloaded local model directory is required")
    args.snapshot_root.mkdir(parents=True, exist_ok=False)
    phase("load-existing-model")
    backend = RealSearchBackend(args.model_path, gamma=gamma)
    require(backend.hidden_size == 3584, "this gate requires the configured REAL-Prover 7B hidden size")
    session_a, session_b = "search-smoke-a", "search-smoke-b"
    with GpuRuntime(backend=backend, snapshot_root=args.snapshot_root) as runtime:
        created = {sid: runtime.create_session(sid) for sid in (session_a, session_b)}
        for sid, metadata in created.items():
            require(metadata["value_metadata"]["value_semantics"] == VALUE_SEMANTICS,
                    f"wrong value semantics: {sid}")
            require(metadata["value_metadata"]["gamma"] == gamma, f"wrong gamma: {sid}")
        equivalence = {}
        for sid in (session_a, session_b):
            error = runtime.actor.submit(lambda sid=sid: backend.initial_equivalence_error(sid, PROMPT))
            require(math.isfinite(error) and error <= 1e-6, f"fresh LoRA differs from frozen base: {sid}")
            equivalence[sid] = error
        phase("full-base-fingerprint-before")
        before_base = runtime.actor.submit(lambda: base_fingerprint(backend, 8 * 1024 * 1024))
        before_values = {sid: value_probe(runtime, backend, sid, 0) for sid in (session_a, session_b)}
        snapshot = runtime.snapshot(session_a, "before-one-update")
        before_a, before_b = capture(runtime, backend, session_a), capture(runtime, backend, session_b)
        phase("one-synthetic-search-update")
        event = synthetic_event(session_a, gamma)
        receipt = runtime.learn(session_a, expected_policy_version=0, event=event)
        require(receipt["applied"] is True and receipt["idempotent"] is False
                and receipt["policy_version"] == 1, "one update did not commit version 0 to 1")
        finite_tree(torch, receipt, "search-receipt")
        detail = receipt["detail"]
        require(detail["objective"] == "search_visit_backup" and detail["optimizer_steps"] == 1,
                "wrong objective or optimizer count")
        for flag in ("finite_loss", "finite_gradients", "finite_parameters", "finite_optimizer_state"):
            require(detail.get(flag) is True, f"missing numeric gate: {flag}")
        require(detail["grad_norm"] > 0, "no nonzero gradient")
        expected_target = max(backend.value_floor, gamma ** 2)
        require(math.isclose(detail["value_target"], expected_target, rel_tol=1e-12), "wrong backup target")
        require([c["target_probability"] for c in detail["candidate_targets"]] == [0.75, 0.25],
                "policy targets are not normalized visits")
        after_a = capture(runtime, backend, session_a)
        deltas = require_update(torch, before_a, after_a)
        require(equal_tree(torch, before_b, capture(runtime, backend, session_b)), "A update mutated B")
        # Idempotency must not execute a second optimizer step.
        duplicate = runtime.learn(session_a, expected_policy_version=0, event=event)
        require(duplicate["applied"] is False and duplicate["idempotent"] is True
                and duplicate["policy_version"] == 1, "same-event replay was not idempotent")
        require(equal_tree(torch, after_a, capture(runtime, backend, session_a)), "idempotent replay mutated A")
        after_value = value_probe(runtime, backend, session_a, 1)
        phase("restore-and-isolation")
        restored = runtime.restore(session_a, "before-one-update")
        require(restored["policy_version"] == 0, "restore did not recover version 0")
        require(equal_tree(torch, before_a, capture(runtime, backend, session_a)),
                "restore failed to recover exact adapter/head/optimizer/RNG/counters/metadata")
        require(equal_tree(torch, before_b, capture(runtime, backend, session_b)), "A restore mutated B")
        restored_value = value_probe(runtime, backend, session_a, 0)
        require(restored_value == before_values[session_a], "restored value response differs")
        phase("full-base-fingerprint-after")
        after_base = runtime.actor.submit(lambda: base_fingerprint(backend, 8 * 1024 * 1024))
        require(before_base == after_base, "frozen base full fingerprint changed")
        report = {"ok": True, "gate": "strict_search_objective_real_ROCm_GPU_synthetic_precheck",
            "ttt_verified": False, "lean_feedback_verified": False, "same_tree_online_verified": False,
            "training_data": "one_synthetic_search_visit_backup_event_NOT_Lean_feedback",
            "torch_hip": torch.version.hip, "device": torch.cuda.get_device_name(0),
            "model_path": str(Path(args.model_path).resolve()), "hidden_size": backend.hidden_size,
            "gamma": gamma, "training_config": backend._search_config(), "coexisting_sessions": 2,
            "optimizer_updates_executed": 1, "fresh_lora_max_logit_delta": equivalence,
            "synthetic_event": event, "learn_receipt": receipt, "parameter_deltas": deltas,
            "value_before": before_values, "value_after_update": after_value,
            "value_after_restore": restored_value, "idempotent_replay_preserved_state": True,
            "isolation": {"A_update_and_restore_preserved_B_exactly": True,
                "scope": "decoded_adapter_value_optimizer_RNG_counters_and_full_session_metadata",
                "limitation": "B_has_no_training_steps_so_its_optimizer_momentum_is_empty"},
            "snapshot": str(snapshot), "restore_exact": True, "restored_policy_version": 0,
            "frozen_base_before": before_base, "frozen_base_after": after_base,
            "wall_seconds": time.perf_counter() - started}
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
        print(encoded, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
