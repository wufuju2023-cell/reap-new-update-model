#!/usr/bin/env python3
"""Read-only CPU audit of fixed-release actors from immutable snapshots.

No model is created, no GPU method is called, and no training is performed.
The operator must prevent concurrent writers to the supplied evidence files.
The final report contains hashes and boolean gates, never tensor payloads.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

# Also supports direct ``python tools/audit_learner_actor_snapshots.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from containers.gpu.smoke_gpu import equal_tree, finite_tree, tensor_leaves
from gpu_runtime.identifiers import validate_identifier


def require(condition, message):
    if not condition:
        raise ValueError(message)


def no_links(path):
    for part in (path, *path.parents):
        require(not part.is_symlink() and not (hasattr(part, "is_junction") and part.is_junction()),
                "evidence paths must not contain links")


def decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def nonfinite(_):
        raise ValueError("nonfinite JSON number")
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    # JSON can encode 1e999 without using NaN/Infinity constants.
    def check(item):
        if isinstance(item, float):
            require(math.isfinite(item), "nonfinite JSON number")
        elif isinstance(item, dict):
            for child in item.values(): check(child)
        elif isinstance(item, list):
            for child in item: check(child)
    check(value)
    return value


def read(path, *, limit=None):
    path = Path(path).absolute(); no_links(path)
    require(path.is_file(), "required evidence file missing")
    with path.open("rb") as stream:
        raw = stream.read() if limit is None else stream.read(limit+1)
    require(limit is None or len(raw) <= limit, "evidence file exceeds its limit")
    return raw, {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def hash_file(path):
    no_links(path)
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk); count += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": count}


def require_sha(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "invalid SHA256 pin")


def load_snapshot(torch, path):
    path = Path(path).absolute(); no_links(path)
    require(path.is_dir() and {p.name for p in path.iterdir()} == {"manifest.json", "session.json", "backend.json"},
            "snapshot must contain exactly three files")
    sid, name = path.parent.name, path.name
    validate_identifier(sid, kind="session_id"); validate_identifier(name, kind="snapshot name")
    raw, manifest_pin = read(path/"manifest.json", limit=65536)
    manifest = decode(raw)
    require(isinstance(manifest, dict) and set(manifest) == {"schema_version", "session_id", "snapshot", "files"}
            and manifest["schema_version"] == "reap.gpu.snapshot.v1"
            and manifest["session_id"] == sid and manifest["snapshot"] == name,
            "snapshot manifest identity/schema mismatch")
    files = manifest["files"]
    require(isinstance(files, dict) and set(files) == {"session.json", "backend.json"}, "snapshot files mismatch")
    states, pins = {}, {"manifest.json": manifest_pin}
    for filename in ("session.json", "backend.json"):
        expected = files[filename]
        require(isinstance(expected, dict) and set(expected) == {"sha256", "bytes"}
                and type(expected["bytes"]) is int and expected["bytes"] > 0, "invalid file manifest")
        require_sha(expected["sha256"])
        raw, pin = read(path/filename)
        require(pin == expected, "snapshot file hash/length mismatch")
        states[filename] = decode(raw); pins[filename] = pin
    logical, backend = states["session.json"], states["backend.json"]
    require(isinstance(logical, dict) and logical.get("schema_version") == "reap.gpu.session.v1"
            and logical.get("session_id") == sid and logical.get("role") == "actor"
            and logical.get("completed") is False and type(logical.get("policy_version")) is int
            and logical["policy_version"] == 0, "snapshot is not a fixed actor at local v0")
    require_sha(logical.get("theorem_id"))
    require(isinstance(backend, dict) and set(backend) == {"schema_version", "encoding", "payload", "session_id", "verified_config"}
            and backend["schema_version"] == "reap.gpu.verified-replay-backend.v1"
            and backend["encoding"] == "torch-save-base64" and backend["session_id"] == sid,
            "backend identity/schema mismatch")
    require(isinstance(backend["verified_config"], dict)
            and backend["verified_config"].get("objective") == "verified_success_replay", "backend objective mismatch")
    encoded = backend.pop("payload")
    require(isinstance(encoded, str) and bool(encoded), "backend payload missing")
    binary = base64.b64decode(encoded, validate=True)
    require(base64.b64encode(binary).decode("ascii") == encoded, "backend base64 is not canonical")
    del encoded, raw, states
    state = torch.load(io.BytesIO(binary), map_location="cpu", weights_only=True)
    del binary
    require(isinstance(state, dict) and set(state) == {"adapter", "value_head", "optimizer", "optimizer_steps", "examples_seen", "rng"},
            "backend private state fields mismatch")
    finite_tree(torch, state, "actor")
    leaves = tensor_leaves(torch, state)
    require(bool(leaves) and all(t.device.type == "cpu" for t in leaves.values()), "snapshot tensor decoding must use CPU only")
    for key in ("adapter", "value_head"):
        require(isinstance(state[key], dict) and bool(state[key])
                and all(torch.is_tensor(t) for t in state[key].values()), "missing actor parameter tensors")
    require(isinstance(state["rng"], dict) and set(state["rng"]) == {"seed", "device_type", "cpu", "device"},
            "actor RNG state missing")
    return {"path": path, "logical": logical, "envelope": backend, "state": state, "pins": pins,
            "tensor_count": len(leaves)}


def audit(*, actor_before, actor_after, new_actor, report, continuation):
    import torch
    report_path, continuation_path = Path(report).absolute(), Path(continuation).absolute()
    raw, report_pin = read(report_path, limit=4*1024*1024); source_report = decode(raw)
    raw, continuation_pin = read(continuation_path, limit=65536); final = decode(raw)
    require(isinstance(source_report, dict) and source_report.get("schema_version") == "reap.learner-loop.gpu-gate.v1"
            and source_report.get("ok") is True and source_report.get("error") is None,
            "worker report has an error or did not pass")
    require(isinstance(source_report.get("gates"), dict) and bool(source_report["gates"])
            and all(v is True for v in source_report["gates"].values()), "worker gate is false/nonboolean")
    release1 = source_report["release1"]["model_release_sha256"]
    release2 = source_report["release2"]["model_release_sha256"]
    checkpoint = source_report["second"]["checkpoint_sha256"]
    for pin in (release1, release2, checkpoint): require_sha(pin)
    require(release1 != release2, "worker releases did not advance")
    require(isinstance(final, dict) and final.get("checkpoint_sha256") == checkpoint
            and final.get("release_sha256") == release2 and final.get("goal_complete") is False,
            "final continuation is missing or differs from committed report")
    old = load_snapshot(torch, actor_before)
    after = load_snapshot(torch, actor_after)
    new = load_snapshot(torch, new_actor)
    actors = source_report["actors"]
    require(isinstance(actors, list) and len(actors) == 2, "report actor identity missing")
    require(old["logical"]["session_id"] == after["logical"]["session_id"] == actors[0]["session_id"]
            and new["logical"]["session_id"] == actors[1]["session_id"]
            and old["logical"]["session_id"] != new["logical"]["session_id"], "actor IDs differ from worker report")
    for snap, declared, release in ((old, actors[0], release1), (after, actors[0], release1), (new, actors[1], release2)):
        require(snap["logical"]["theorem_id"] == declared["theorem_id"]
                and snap["logical"]["lineage"] == declared["lineage"]
                and snap["logical"]["lineage"]["model_release_sha256"] == release, "actor theorem/release lineage differs")
    new_logical, new_state = new["logical"], new["state"]
    optimizer = new_state["optimizer"]
    require(isinstance(optimizer, dict) and set(optimizer) == {"state", "param_groups"}
            and isinstance(optimizer["param_groups"], list) and bool(optimizer["param_groups"]), "optimizer structure missing")
    gates = {
        "worker_report_ok_without_error": True,
        "final_continuation_bound_to_report": True,
        "actor_ids_theorems_releases_bound_to_report": True,
        "old_actor_complete_logical_state_equal": equal_tree(torch, old["logical"], after["logical"]),
        "old_actor_backend_envelope_equal": equal_tree(torch, old["envelope"], after["envelope"]),
        "old_actor_all_decoded_backend_state_equal": equal_tree(torch, old["state"], after["state"]),
        "new_actor_empty_AdamW_state": new_logical.get("optimizer_metadata", {}).get("kind") == "AdamW"
            and optimizer["state"] == {},
        "new_actor_zero_optimizer_steps": type(new_state["optimizer_steps"]) is int and new_state["optimizer_steps"] == 0
            and type(new_logical.get("optimizer_metadata", {}).get("steps")) is int and new_logical["optimizer_metadata"]["steps"] == 0,
        "new_actor_zero_examples_seen": type(new_state["examples_seen"]) is int and new_state["examples_seen"] == 0,
        "new_actor_empty_event_receipts": new_logical.get("event_receipts") == {},
        "new_actor_empty_buffer": equal_tree(torch, new_logical.get("buffer_metadata"),
            {"events": {}, "pending_event_ids": [], "consumed_event_ids": []}),
    }
    for snap in (old, after, new):
        require(all(hash_file(snap["path"]/name) == pin for name, pin in snap["pins"].items()),
                "snapshot changed during audit")
    require(hash_file(report_path) == report_pin and hash_file(continuation_path) == continuation_pin,
            "worker evidence changed during audit")
    gates["all_input_bytes_unchanged_during_audit"] = True
    return {"schema_version": "reap.learner-actor-snapshot-audit.v1", "ok": all(gates.values()), "gates": gates,
        "CPU_only": True, "training_performed": False, "payloads_or_tensor_values_disclosed": False,
        "scope": "the supplied before/after snapshot interval only; no later actor mutation is covered",
        "report_pin": report_pin, "continuation_pin": continuation_pin,
        "sources": {label: {"session_id": snap["logical"]["session_id"], "snapshot": snap["path"].name,
            "pins": snap["pins"], "decoded_tensor_count": snap["tensor_count"]}
            for label, snap in (("actor_before", old), ("actor_after", after), ("new_actor", new))}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("actor-before", "actor-after", "new-actor", "report", "continuation", "output"):
        parser.add_argument("--"+flag, required=True, type=Path)
    args = vars(parser.parse_args(argv)); output = args.pop("output").absolute()
    no_links(output)
    for key in ("actor_before", "actor_after", "new_actor"):
        directory = args[key].absolute(); no_links(directory)
        require(not output.is_relative_to(directory), "audit output cannot be inside an input snapshot")
    # Reserve output before reading; reruns must use a fresh audit artifact.
    with output.open("xb") as stream:
        try:
            result = audit(**args)
        except Exception as exc:
            result = {"schema_version": "reap.learner-actor-snapshot-audit.v1", "ok": False,
                "error_type": type(exc).__name__, "error": "audit failed; no partial acceptance",
                "CPU_only": True, "training_performed": False}
        result["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        stream.write((json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode())
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"ok": result["ok"], "output": str(output)}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
