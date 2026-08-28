#!/usr/bin/env python3
"""Real GPU gate for admitted historical successful trajectories; no search rerun."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time

from gpu_runtime import GpuRuntime
from gpu_runtime.verified_backend import VerifiedReplayBackend, SNAPSHOT_SCHEMA
from gpu_runtime.verified_objective import OBJECTIVE_KIND
from cpu_runtime.verified_trajectory import load_verified_dataset
from containers.gpu.smoke_gpu import require, equal_tree, finite_tree, base_fingerprint
from containers.gpu.smoke_kl_guard import save_capture, host_rng, write_bytes, encoded
from containers.gpu.smoke_search_gpu import phase


def capture(runtime, backend, sid):
    raw = runtime.inspect_backend(sid)
    require(raw["schema_version"] == SNAPSHOT_SCHEMA and raw["session_id"] == sid, "wrong replay snapshot")
    require(raw["verified_config"] == backend._config(), "wrong replay contract")
    state = backend.torch.load(io.BytesIO(base64.b64decode(raw["payload"], validate=True)),
                               map_location="cpu", weights_only=True)
    finite_tree(backend.torch, state, sid)
    return {"backend": state, "metadata": runtime.sessions.get(sid).snapshot()}


def audit(runtime, backend, datasets, output):
    refs = [{"dataset_sha256": digest, "row": index} for digest, dataset in datasets.items()
            for index in range(len(dataset["rows"]))]
    require(1 <= len(refs) <= 32, "one gate batch must contain 1..32 verified rows")
    event = {"kind": OBJECTIVE_KIND, "event_id": "verified-replay-0", "session_id": "verified-learner",
             "policy_version": 0, "samples": refs}
    write_bytes(output / "event.json", encoded(event))
    for sid in ("verified-learner", "verified-control"):
        runtime.create_session(sid)
    phase("full-frozen-base-before")
    base_before = runtime.actor.submit(lambda: base_fingerprint(backend, 8*1024*1024))
    before = {sid: capture(runtime, backend, sid) for sid in ("verified-learner", "verified-control")}
    artifacts = {sid+"-before": save_capture(output, sid+"-before", backend, state) for sid, state in before.items()}
    runtime.snapshot("verified-learner", "before")
    host_before = runtime.actor.submit(lambda: host_rng(backend))
    phase("one-real-successful-replay-update")
    started = time.perf_counter()
    receipt = runtime.learn("verified-learner", expected_policy_version=0, event=event)
    seconds = time.perf_counter()-started
    write_bytes(output / "learn-receipt.json", encoded(receipt))
    host_after = runtime.actor.submit(lambda: host_rng(backend))
    after = {sid: capture(runtime, backend, sid) for sid in before}
    artifacts.update({sid+"-after": save_capture(output, sid+"-after", backend, state) for sid, state in after.items()})
    duplicate = runtime.learn("verified-learner", expected_policy_version=0, event=event)
    duplicate_unchanged = equal_tree(backend.torch, after["verified-learner"], capture(runtime, backend, "verified-learner"))
    prompt = next(iter(datasets.values()))["rows"][0]["prompt"]
    response = runtime.value("verified-learner", {"model": "REAL-Prover", "messages": [{"role": "user", "content": prompt}]})
    distance = json.loads(response["choices"][0]["message"]["content"])["score"]
    runtime.snapshot("verified-learner", "after")
    phase("full-frozen-base-after")
    base_after = runtime.actor.submit(lambda: base_fingerprint(backend, 8*1024*1024))
    runtime.restore("verified-learner", "before")
    restored = capture(runtime, backend, "verified-learner")
    restored_exact = equal_tree(backend.torch, before["verified-learner"], restored)
    runtime.restore("verified-learner", "after")
    final_exact = equal_tree(backend.torch, after["verified-learner"], capture(runtime, backend, "verified-learner"))
    detail = receipt["detail"]
    gates = {
        "one_committed_update": receipt["applied"] is True and receipt["policy_version"] == 1,
        "actual_verified_objective": detail["objective"] == OBJECTIVE_KIND,
        "all_rows_used_once": [(x["dataset_sha256"], x["row"]) for x in detail["samples"]]
                              == [(x["dataset_sha256"], x["row"]) for x in refs],
        "actual_targets_preserved": all(x["return"] == datasets[x["dataset_sha256"]]["rows"][x["row"]]["return"]
                                        and x["value_class"] == -x["return"]-1 for x in detail["samples"]),
        "finite_joint_training": all(detail[k] is True for k in ("finite_loss", "finite_gradients", "finite_parameters", "finite_optimizer_state")),
        "adapter_changed": detail["parameter_diffs"]["adapter"]["changed_tensors"] > 0,
        "categorical_head_changed": detail["parameter_diffs"]["value_head"]["changed_tensors"] > 0,
        "Adam_created": detail["parameter_diffs"]["optimizer"]["added_tensors"] > 0,
        "counts_exact": after["verified-learner"]["backend"]["optimizer_steps"] == 1
                        and after["verified-learner"]["backend"]["examples_seen"] == len(refs),
        "control_exact": equal_tree(backend.torch, before["verified-control"], after["verified-control"]),
        "host_RNG_exact": equal_tree(backend.torch, host_before, host_after),
        "frozen_base_full_hash_equal": base_before == base_after,
        "duplicate_no_update": duplicate["idempotent"] is True and duplicate["applied"] is False and duplicate_unchanged,
        "value_expected_distance": response["policy_version"] == 1 and 1 <= distance <= backend.max_distance,
        "restore_before_exact": restored_exact, "restore_after_exact": final_exact,
    }
    if backend.max_post_update_kl is not None:
        gates["post_update_KL_accepted"] = detail["kl_guard"]["accepted"] is True
    return {"ok": all(gates.values()), "gates": gates, "artifacts": artifacts,
            "rows": len(refs), "dataset_pins": list(datasets), "training_config": backend._config(),
            "learn_receipt": receipt, "base_before": base_before, "base_after": base_after,
            "value_distance": distance, "learn_wall_seconds": seconds,
            "timing_scope": "runtime.learn including transactional state copy, not GPU kernel time"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-sha256", action="append", required=True)
    parser.add_argument("--max-distance", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from gpu_runtime.verified_objective import SHA256, validate_support
    validate_support(args.max_distance)
    require(len(set(args.dataset_sha256)) == len(args.dataset_sha256), "duplicate dataset pin")
    require(all(SHA256.fullmatch(x) for x in args.dataset_sha256), "invalid dataset pin")
    datasets = {digest: load_verified_dataset(args.dataset_root/digest, expected_sha256=digest)
                for digest in args.dataset_sha256}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "reap.verified-replay.gpu-gate.v1", "ok": False,
              "real_7B_GPU_gate_passed": False, "new_Lean_search": False,
              "SFT_mixture_or_curriculum_tested": False, "performance_improvement_claimed": False}
    try:
        import torch
        require(bool(torch.version.hip) and torch.cuda.is_available(), "actual ROCm GPU required")
        phase("load-existing-7B-verified-replay")
        backend = VerifiedReplayBackend(args.model_path, dataset_root=args.dataset_root,
            max_distance=args.max_distance, max_post_update_kl=100.0)
        with GpuRuntime(backend=backend, snapshot_root=args.output_dir/"snapshots") as runtime:
            report.update(audit(runtime, backend, datasets, args.output_dir))
            report["real_7B_GPU_gate_passed"] = report["ok"]
            report["actor"] = runtime.actor.metrics()
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    write_bytes(args.output_dir/"report.json", encoded(report))
    print(json.dumps({"ok": report["ok"], "report": str(args.output_dir/"report.json")}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
