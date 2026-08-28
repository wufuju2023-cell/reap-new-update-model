#!/usr/bin/env python3
"""Verify an already accepted release on real AMD 7B, without downloading.

Uses one synthetic search update after inheritance. This is a mechanism gate,
not a proof experiment or evidence of performance improvement. The release
must first be published from a newly sealed, independently checked Lean run.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import time

from gpu_runtime import GpuRuntime
from gpu_runtime.search_backend import RealSearchBackend
from containers.gpu.smoke_gpu import base_fingerprint, require
from containers.gpu.smoke_search_gpu import synthetic_event, value_probe, phase


def audit(runtime, experience_id: str, session_id: str, theorem_id: str, *, event_factory=synthetic_event):
    backend = runtime.backend
    metadata, weights = runtime.experiences.load(experience_id)
    control_id = session_id + "-control"
    runtime.create_session(control_id)
    control = backend.session_fingerprints(control_id)
    created = runtime.create_session(session_id, theorem_id=theorem_id, experience_id=experience_id)
    require(created["policy_version"] == 0 and created["optimizer_metadata"]["steps"] == 0,
            "destination counters were inherited")
    initial = backend.session_fingerprints(session_id)
    payload = backend.torch.load(io.BytesIO(base64.b64decode(weights["payload"], validate=True)),
                                 map_location="cpu", weights_only=True)
    for key in ("adapter", "value_head"):
        require(initial[key] == backend._tensor_manifest(payload[key]), f"initial {key} differs from release")
    require(not backend.sessions[session_id].optimizer.state, "optimizer is not fresh")
    require(backend.sessions[session_id].rng_seed == backend._new_rng(session_id)[0],
            "destination RNG seed does not belong to the destination")
    require(not created["event_receipts"] and not created["buffer_metadata"]["events"], "training history copied")
    runtime.snapshot(session_id, "inherited-initial")
    value = value_probe(runtime, backend, session_id, 0)
    event = event_factory(session_id, backend.gamma)
    learned = runtime.learn(session_id, expected_policy_version=0, event=event)
    require(learned["policy_version"] == 1, "first destination update did not produce v1")
    updated = backend.session_fingerprints(session_id)
    require(updated["adapter"] != initial["adapter"], "adapter did not update")
    require(updated["value_head"] != initial["value_head"], "value head did not update")
    require(backend.session_fingerprints(control_id) == control, "concurrent session contaminated")
    require(runtime.experiences.load(experience_id) == (metadata, weights), "source release changed")
    runtime.restore(session_id, "inherited-initial")
    require(backend.session_fingerprints(session_id) == initial, "inherited-session restore mismatch")
    return {"experience_id": experience_id, "lineage": created["lineage"],
            "initial_fingerprints": initial, "updated_fingerprints": updated,
            "value_probe": value, "learn_receipt": learned,
            "gates": {"initial_parameters_equal_release": True, "fresh_training_state": True,
                      "first_update_v1": True, "control_unchanged": True,
                      "release_unchanged": True, "restore_exact": True},
            "synthetic_update": True, "lean_feedback_ttt_verified": False,
            "performance_improvement_demonstrated": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--experience-root", type=Path, required=True)
    p.add_argument("--experience-id", required=True)
    p.add_argument("--session-id", required=True)
    p.add_argument("--theorem-id", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    import torch
    require(bool(torch.version.hip) and torch.cuda.is_available(), "usable AMD ROCm GPU required")
    require(Path(args.model_path).is_dir(), "reuse a verified local model; no download is performed")
    require(args.experience_root.is_dir(), "accepted experience store must already exist")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    phase("load_verified_model")
    backend = RealSearchBackend(args.model_path, gamma=args.gamma)
    phase("base_fingerprint_before")
    base_before = base_fingerprint(backend, 1024 * 1024)
    with GpuRuntime(backend=backend, snapshot_root=args.output_dir / "snapshots",
                    experience_root=args.experience_root) as runtime:
        phase("inherit_update_restore")
        result = audit(runtime, args.experience_id, args.session_id, args.theorem_id)
        phase("base_fingerprint_after")
        base_after = base_fingerprint(backend, 1024 * 1024)
        require(base_before == base_after, "frozen base changed")
    result.update(schema_version="reap.experience.gpu-mechanism.v1", ok=True,
                  gpu_name=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  hip=torch.version.hip, base_before=base_before, base_after=base_after,
                  elapsed_seconds=time.monotonic() - start)
    with (args.output_dir / "report.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"ok": True, "report": str(args.output_dir / "report.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
