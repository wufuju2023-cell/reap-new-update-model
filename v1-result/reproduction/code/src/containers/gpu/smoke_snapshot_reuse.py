#!/usr/bin/env python3
"""Restore a pinned trained snapshot and verify real resident retirement; no learn/search."""
from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import time
from unittest.mock import patch

from gpu_runtime import GpuRuntime
from gpu_runtime.identifiers import validate_identifier
from gpu_runtime.snapshot_store import SnapshotStore
from gpu_runtime.verified_backend import VerifiedReplayBackend, SNAPSHOT_SCHEMA
from containers.gpu.smoke_gpu import require, equal_tree, finite_tree
from containers.gpu.smoke_kl_guard import write_bytes, encoded
from containers.gpu.smoke_policy_scoring import CudaMeter, read_json


FILES = ("manifest.json", "session.json", "backend.json")
SOURCE_FILES = ("containers/gpu/smoke_snapshot_reuse.py", "gpu_runtime/runtime.py",
    "gpu_runtime/snapshot_store.py", "gpu_runtime/session_store.py", "gpu_runtime/actor.py",
    "gpu_runtime/real_backend.py", "gpu_runtime/search_backend.py", "gpu_runtime/verified_backend.py",
    "gpu_runtime/verified_objective.py", "containers/gpu/smoke_gpu.py",
    "containers/gpu/smoke_kl_guard.py", "containers/gpu/smoke_policy_scoring.py")


def file_pin(path):
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def no_links(path):
    require(not any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction())
                    for p in (path, *path.parents)), "linked input/output paths are forbidden")


def snapshot_pins(path):
    no_links(path)
    require(path.is_dir() and {p.name for p in path.iterdir()} == set(FILES), "unexpected snapshot files")
    for filename in FILES:
        no_links(path / filename)
        require((path / filename).is_file(), "snapshot member must be a regular file")
    return {filename: file_pin(path / filename) for filename in FILES}


def prepare_source(source, output, manifest_sha256, *, session_id, expected_policy_version):
    """Copy without replacing/renaming any source member; validate before model load."""
    require(bool(re.fullmatch(r"[0-9a-f]{64}", manifest_sha256)), "explicit source manifest SHA256 required")
    validate_identifier(session_id, kind="session_id")
    require(type(expected_policy_version) is int and expected_policy_version >= 1, "trained source version required")
    source, output = Path(source).absolute(), Path(output).absolute()
    no_links(output)
    before = snapshot_pins(source)
    require(source.parent.name == session_id and source.name != "reuse-final", "source session/name mismatch")
    require(before["manifest.json"]["sha256"] == manifest_sha256, "source manifest pin mismatch")
    logical = SnapshotStore(source.parent.parent).verify_for_reuse(session_id, source.name,
        expected_manifest_sha256=manifest_sha256)
    require(logical["session_id"] == session_id and logical["policy_version"] == expected_policy_version
            and not logical["completed"] and not logical["buffer_metadata"].get("pending_event_ids"),
            "source must be the expected nonsealed, completed-update session")
    copied = output / "snapshots" / session_id / source.name
    copied.mkdir(parents=True, exist_ok=False)
    for filename in FILES:
        with (source / filename).open("rb") as src, (copied / filename).open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
    require(snapshot_pins(source) == before == snapshot_pins(copied), "source changed while copying")
    SnapshotStore(output / "snapshots").verify_for_reuse(session_id, source.name,
        expected_manifest_sha256=manifest_sha256)
    descriptor = {"schema_version": "reap.snapshot-reuse.source.v1", "source": str(source),
        "copied_snapshot": str(copied), "session_id": session_id, "snapshot": source.name,
        "expected_policy_version": expected_policy_version, "files": before}
    write_bytes(output / "source-pins.json", encoded(descriptor))
    return descriptor


def decode(backend, raw):
    require(raw.get("schema_version") == SNAPSHOT_SCHEMA and raw.get("verified_config") == backend._config(),
            "source verified-replay objective/model/support/optimizer contract mismatch")
    payload = backend.torch.load(io.BytesIO(base64.b64decode(raw["payload"], validate=True)),
                                 map_location="cpu", weights_only=True)
    finite_tree(backend.torch, payload, "restored snapshot")
    return payload


def audit(runtime, backend, source, output, *, meter):
    """Uses the actual backend/runtime; meter alone may be fake in CPU tests."""
    sid, original_name, version = source["session_id"], source["snapshot"], source["expected_policy_version"]
    require(runtime.max_resident_sessions == 1, "probe requires resident cap 1")
    source_logical, source_raw = runtime.snapshots.load(sid, original_name)
    source_payload = decode(backend, source_raw)
    require(source_payload["optimizer_steps"] == version and source_payload["examples_seen"] > 0
            and bool(source_payload["optimizer"]["state"]), "trained Adam source state required")
    del source_raw
    runtime.create_session(sid, theorem_id=source_logical["theorem_id"])
    restored = runtime.restore(sid, original_name)
    write_bytes(output / "restore-receipt.json", encoded(restored))
    restored_payload = decode(backend, runtime.inspect_backend(sid))
    require(equal_tree(backend.torch, source_payload, restored_payload) and restored == source_logical,
            "restored mutable tensors/RNG/counters/logical state differ from source")
    fingerprints = {"source": backend._tensor_manifest(source_payload),
                    "restored": backend._tensor_manifest(restored_payload)}
    write_bytes(output / "fingerprints.json", encoded(fingerprints))
    del restored_payload
    final_name = "reuse-final"
    # No inspect/fingerprint/backend operation is allowed between these two calls.
    final_path = runtime.snapshot(sid, final_name)
    final_before = snapshot_pins(final_path)  # Filesystem reads do not touch session state.
    intent = {"schema_version": "reap.snapshot-reuse.retirement-intent.v1", "session_id": sid,
        "name": final_name, "expected_policy_version": version, "reuse_snapshot": True,
        "source_manifest_sha256": source["files"]["manifest.json"]["sha256"],
        "final_manifest_sha256": final_before["manifest.json"]["sha256"], "mutation_retry_allowed": False}
    write_bytes(output / "retirement-intent.json", encoded(intent))
    meter.synchronize()
    memory_before = meter.memory()
    started = time.perf_counter()
    with patch.object(backend, "export_session", wraps=backend.export_session) as export:
        receipt = runtime.retire_session(sid, final_name, expected_policy_version=version, reuse_snapshot=True)
        export_calls = export.call_count
    meter.synchronize()
    retirement_seconds = time.perf_counter() - started
    write_bytes(output / "retirement-receipt.json", encoded(receipt))
    gc.collect()
    meter.synchronize()
    memory_after = meter.memory()
    # Only CPU/disk inspection after deletion. Never re-create the retired ID.
    final_logical, final_raw = runtime.snapshots.load(sid, final_name)
    final_payload = decode(backend, final_raw)
    del final_raw
    require(sid not in backend.sessions and sid not in runtime.sessions._states, "retired state still resident")
    readback = runtime.retirement_receipt(sid)
    write_bytes(output / "retirement-readback.json", encoded(readback))
    replacement = "reuse-capacity-check"
    require(replacement != sid, "replacement identity collision")
    runtime.create_session(replacement)
    capacity_reused = replacement in backend.sessions and replacement in runtime.sessions._states
    runtime.delete_session(replacement)
    source_after = snapshot_pins(Path(source["source"]))
    copied_after = snapshot_pins(Path(source["copied_snapshot"]))
    final_after = snapshot_pins(final_path)
    write_bytes(output / "final-pins.json", encoded({"before": final_before, "after": final_after}))
    gates = {
        "source_and_copy_all_bytes_unchanged": source_after == copied_after == source["files"],
        "final_all_bytes_unchanged": final_before == final_after,
        "final_payload_and_logical_exact": equal_tree(backend.torch, source_payload, final_payload)
                                            and final_logical == source_logical,
        "reuse_export_call_count_zero": export_calls == 0,
        "strict_v2_released_receipt": receipt == readback and receipt["schema_version"] == "reap.gpu.retirement.v2"
            and receipt["session_id"] == sid and receipt["policy_version"] == version
            and receipt["snapshot"] == final_name and receipt["snapshot_mode"] == "reuse_verified"
            and type(receipt["snapshot_revision"]) is int and receipt["snapshot_revision"] > 0
            and receipt["snapshot_sha256"] == final_before["manifest.json"]["sha256"]
            and receipt["status"] == "released" and receipt["mutation_retry_allowed"] is False,
        "only_source_and_final_snapshots": {p.name for p in final_path.parent.iterdir()} == {original_name, final_name},
        "resident_capacity_released": capacity_reused and not backend.sessions and not runtime.sessions._states,
    }
    return {"ok": all(gates.values()), "gates": gates, "reuse_export_calls": export_calls,
        "source": source, "verified_config": backend._config(), "retirement_receipt": receipt,
        "memory_before_retirement": memory_before, "memory_after_retirement": memory_after,
        "retirement_wall_seconds": retirement_seconds,
        "timing_scope": "one runtime retire including full file verification and synchronized GPU completion; no speed baseline",
        "memory_scope": "allocator measurements after restored state and after deletion/GC; not kernel utilization or throughput"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--session-id", default="verified-learner")
    parser.add_argument("--expected-policy-version", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    no_links(args.output_dir.absolute())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "reap.snapshot-reuse.gpu-probe.v1", "ok": False,
        "real_7B_GPU_gate_passed": False, "no_training": True, "no_generation_or_Lean": True,
        "performance_improvement_claimed": False, "mutation_retry_allowed": False}
    try:
        source = prepare_source(args.source_snapshot, args.output_dir, args.source_manifest_sha256,
            session_id=args.session_id, expected_policy_version=args.expected_policy_version)
        require(args.model_path.is_dir(), "existing local model required; never download")
        lock, lock_sha = read_json(args.model_path / "reap-model-lock.json")
        require(lock.get("schema_version") == "reap.model-lock.v2" and lock.get("repo") == "FrenzyMath/REAL-Prover"
                and lock.get("revision") == "fe76f68d9a88f342cb7b546307c20292fea9cced"
                and lock.get("hidden_size") == 3584, "pinned REAL-Prover lock mismatch")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        root = Path(__file__).resolve().parents[2]
        before = {name: file_pin(root / name) for name in SOURCE_FILES}
        write_bytes(args.output_dir / "source-code-pins.json", encoded(before))
        import torch
        require(bool(torch.version.hip) and torch.cuda.is_available(), "actual AMD HIP GPU required")
        backend = VerifiedReplayBackend(str(args.model_path), dataset_root=args.dataset_root,
            max_distance=64, max_post_update_kl=100.0, device=f"cuda:{torch.cuda.current_device()}")
        require(backend.device.type == "cuda" and backend.hidden_size == 3584
                and all(p.device == backend.device for p in backend.model.parameters()), "actual 7B device mismatch")
        report.update(model_lock_sha256=lock_sha, model_revision=lock["revision"],
            model_lock_is_prior_verification_only=True, gpu_name=torch.cuda.get_device_name(backend.device),
            device=str(backend.device), hip=torch.version.hip, torch_version=torch.__version__,
            dependencies={name: importlib.metadata.version(name) for name in ("transformers", "peft")})
        with GpuRuntime(backend=backend, snapshot_root=args.output_dir / "snapshots", max_resident_sessions=1) as runtime:
            report.update(audit(runtime, backend, source, args.output_dir, meter=CudaMeter(torch, backend.device)))
            report["actor"] = runtime.actor.metrics()
        after = {name: file_pin(root / name) for name in SOURCE_FILES}
        report["source_code_unchanged"] = before == after
        report["source_code_pins"] = after
        report["ok"] = report["ok"] and before == after
        report["real_7B_GPU_gate_passed"] = report["ok"]
    except Exception as exc:
        report.update(ok=False, error={"type": type(exc).__name__, "message": str(exc)})
    write_bytes(args.output_dir / "report.json", encoded(report))
    print(json.dumps({"ok": report["ok"], "report": str(args.output_dir / "report.json")}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
