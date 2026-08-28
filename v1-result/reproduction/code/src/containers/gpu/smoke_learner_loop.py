#!/usr/bin/env python3
"""Bounded live learner/release/collector gate; commands are durable, never retried.

The CPU operator supplies an independently Lean-verified new dataset after the
first collector finishes. This worker cannot manufacture collector success.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time
from http.server import ThreadingHTTPServer

from gpu_runtime import GpuRuntime
from gpu_runtime.learner import LearnerCoordinator, describe_dataset
from gpu_runtime.server import RuntimeHandler
from gpu_runtime.verified_backend import VerifiedReplayBackend
from cpu_runtime.verified_trajectory import load_verified_dataset
from containers.gpu.smoke_gpu import require, equal_tree, base_fingerprint
from containers.gpu.smoke_kl_guard import write_bytes, encoded
from containers.gpu.smoke_verified_replay import capture


def save(root, name, value):
    write_bytes(root/name, encoded(value))


def wait_command(root, name, deadline):
    path = root/name
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("operator phase incomplete; saved checkpoints remain; no mutation retried")
        time.sleep(0.5)
    raw = path.read_bytes()
    value = json.loads(raw)
    require(isinstance(value, dict), "command must be an object")
    save(root, name+".consumed.json", {"sha256": hashlib.sha256(raw).hexdigest(), "command": value})
    return value


def fingerprint(runtime, backend, sid):
    return runtime.actor.submit(lambda: backend.session_fingerprints(sid))


def require_amd():
    import torch
    require(torch.version.hip and torch.cuda.is_available(), "actual AMD GPU required")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-sha256", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8760)
    parser.add_argument("--operator-budget-seconds", type=int, default=7200)
    parser.add_argument("--resume-checkpoint", help="explicit committed step-one learner checkpoint; never repeat its update")
    parser.add_argument("--learner-store", type=Path, help="existing canonical store when resuming")
    args = parser.parse_args(argv)
    require(60 <= args.operator_budget_seconds <= 10800, "explicit bounded operator window required")
    require(bool(args.resume_checkpoint) == bool(args.learner_store), "resume checkpoint and canonical store must be specified together")
    datasets = {pin: load_verified_dataset(args.dataset_root/pin, expected_sha256=pin) for pin in args.dataset_sha256}
    require(len(datasets) == len(args.dataset_sha256), "duplicate seed dataset")
    rows = sum(len(d["rows"]) for d in datasets.values())
    require(1 <= rows <= 32, "gate seeds must fit one batch")
    root = args.output_dir; root.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "reap.learner-loop.gpu-gate.v1", "ok": False,
        "performance_improvement_claimed": False, "SFT_mixture_or_curriculum_tested": False}
    server = worker = learner = runtime = None
    try:
        require_amd()
        backend = VerifiedReplayBackend(args.model_path, dataset_root=args.dataset_root,
            max_distance=64, max_post_update_kl=100.0)
        runtime = GpuRuntime(backend=backend, snapshot_root=root/"snapshots",
            learner_release_root=args.learner_store or root/"learner-store", max_resident_sessions=3)
        implementation = {str(p).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in (Path("gpu_runtime"), Path("cpu_runtime")) for p in sorted(directory.glob("*.py"))}
        if args.resume_checkpoint:
            checkpoint = runtime.learner_releases.load_checkpoint(args.resume_checkpoint)
            require(checkpoint["manifest"]["step"] == 1
                and [d["dataset_sha256"] for d in checkpoint["run"]["catalog"]] == list(datasets),
                "resume gate requires the exact seed catalog and committed step one")
            learner = LearnerCoordinator.restore(runtime, checkpoint_sha256=args.resume_checkpoint,
                journal_root=root/"resumed-journal")
            first = {"checkpoint_sha256": args.resume_checkpoint, "step": 1,
                "runtime_receipt": checkpoint["data_receipt"]["runtime_receipt"]}
            first_seconds = None
            report["first_update_executed_this_process"] = False
            save(root, "resume-implementation.json", {"actual_sources": implementation,
                "original_run_sha256": checkpoint["manifest"]["run_sha256"], "checkpoint_sha256": args.resume_checkpoint})
        else:
            learner = LearnerCoordinator(runtime, learner_id="central-verified", dataset_pins=list(datasets),
                journal_root=root/"journal", batch_size=rows, implementation=implementation)
            report["first_update_executed_this_process"] = True
        report["base_before"] = runtime.actor.submit(lambda: base_fingerprint(backend, 8*1024*1024))
        before = fingerprint(runtime, backend, learner.learner_id)
        if not args.resume_checkpoint:
            started = time.perf_counter(); first = learner.train_next(); first_seconds = time.perf_counter()-started
        release1 = learner.publish()
        weights1 = fingerprint(runtime, backend, learner.learner_id)
        save(root, "learner-step1-fingerprints.json", weights1)
        report.update(first=first, release1=release1, first_step_with_checkpoint_seconds=first_seconds)
        RuntimeHandler.runtime, RuntimeHandler.backend_name = runtime, "verified-replay"
        server = ThreadingHTTPServer(("127.0.0.1", args.port), RuntimeHandler)
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        save(root, "ready.json", {"port": args.port, "release": release1, "learner_step": 1,
            "backend": "verified-replay", "instance_stop_requested": False})
        print(json.dumps({"phase": "ready-for-fixed-release-collector", "release": release1["model_release_sha256"]}), flush=True)
        deadline = time.monotonic()+args.operator_budget_seconds
        command = wait_command(root, "append-and-train.json", deadline)
        require(set(command) == {"dataset_sha256", "actor_session_id"}, "append command fields mismatch")
        descriptor = describe_dataset(backend, command["dataset_sha256"])
        actor = runtime.sessions.get(command["actor_session_id"]).snapshot()
        require(actor.get("role") == "actor" and actor["policy_version"] == 0
            and actor["lineage"]["model_release_sha256"] == release1["model_release_sha256"], "collector actor release/version mismatch")
        require(descriptor["source_session_id"] == actor["session_id"]
            and descriptor["source_model_release_sha256"] == release1["model_release_sha256"]
            and descriptor["theorem_sha256"] == actor["theorem_id"], "verified dataset actor/release lineage differs")
        actor_before = fingerprint(runtime, backend, actor["session_id"])
        actor_metadata_before = actor
        require(all(actor_before[k] == weights1[k] for k in ("adapter", "value_head")), "actor did not inherit learner release tensors")
        started = time.perf_counter()
        second = learner.train_next(append_dataset_pins=[command["dataset_sha256"]])
        second_seconds = time.perf_counter()-started
        release2 = learner.publish()
        actor_after = fingerprint(runtime, backend, actor["session_id"])
        learner_after = capture(runtime, backend, learner.learner_id)
        weights2 = fingerprint(runtime, backend, learner.learner_id)
        save(root, "learner-step2-fingerprints.json", weights2)
        require(any(s["dataset_sha256"] == command["dataset_sha256"] for s in second["runtime_receipt"]["detail"]["samples"]), "new actor data was not trained")
        learner.close(); runtime.delete_session(learner.learner_id)
        learner = LearnerCoordinator.restore(runtime, checkpoint_sha256=second["checkpoint_sha256"], journal_root=root/"restored-journal")
        restore_exact = equal_tree(backend.torch, learner_after, capture(runtime, backend, learner.learner_id))
        require(restore_exact, "learner full state restore mismatch")
        save(root, "ready2.json", {"release": release2, "learner_step": 2, "new_dataset": descriptor,
            "full_restore_exact": restore_exact})
        print(json.dumps({"phase": "ready-for-second-release-actor", "release": release2["model_release_sha256"]}), flush=True)
        finish = wait_command(root, "finish.json", deadline)
        require(set(finish) == {"actor_session_id"}, "finish command fields mismatch")
        actor2 = runtime.sessions.get(finish["actor_session_id"]).snapshot()
        require(actor2.get("role") == "actor" and actor2["policy_version"] == 0
            and actor2["lineage"]["model_release_sha256"] == release2["model_release_sha256"], "second actor release/version mismatch")
        actor2_weights = fingerprint(runtime, backend, actor2["session_id"])
        actor2_state = capture(runtime, backend, actor2["session_id"])
        actor1_final = fingerprint(runtime, backend, actor["session_id"])
        report["base_after"] = runtime.actor.submit(lambda: base_fingerprint(backend, 8*1024*1024))
        first_changes = first["runtime_receipt"]["detail"]["parameter_diffs"]
        gates = {"first_committed_receipt_changed_adapter": first_changes["adapter"]["changed_tensors"] > 0,
            "first_committed_receipt_changed_head": first_changes["value_head"]["changed_tensors"] > 0,
            "new_verified_actor_dataset_admitted": True,
            "new_actor_rows_in_second_update": True,
            "second_update_changes_adapter": weights1["adapter"] != weights2["adapter"],
            "second_update_changes_head": weights1["value_head"] != weights2["value_head"],
            "old_actor_all_state_unchanged": actor_before == actor_after,
            "old_actor_unchanged_after_restore_and_new_actor": actor1_final == actor_before
                and runtime.sessions.get(actor["session_id"]).snapshot() == actor_metadata_before,
            "distinct_release_versions": release1["model_release_sha256"] != release2["model_release_sha256"],
            "release1_still_exact": runtime.learner_releases.load_release(release1["model_release_sha256"])[0] == release1,
            "new_actor_inherits_release2_tensors": all(actor2_weights[k] == weights2[k] for k in ("adapter", "value_head")),
            "new_actor_private_optimizer": actor2_state["backend"]["optimizer_steps"] == 0
                and actor2_state["backend"]["examples_seen"] == 0
                and actor2_state["backend"]["optimizer"]["state"] == {}
                and actor2["event_receipts"] == {}
                and actor2["buffer_metadata"] == {"events": {}, "pending_event_ids": [], "consumed_event_ids": []},
            "learner_full_restore_exact": restore_exact,
            "frozen_base_full_hash_exact": report["base_before"] == report["base_after"]}
        report.update(ok=all(gates.values()), gates=gates, second=second, release2=release2,
            second_step_with_checkpoint_seconds=second_seconds, new_dataset=descriptor,
            actors=[actor, actor2], actor1_before=actor_before, actor1_after=actor_after,
            actor2_fingerprints=actor2_weights, actor_metrics=runtime.actor.metrics())
        save(root, "final-continuation.json", {"checkpoint_sha256": second["checkpoint_sha256"],
            "release_sha256": release2["model_release_sha256"], "goal_complete": False,
            "instance_stop_requested": False, "next": "Mathlib SFT mixture and bounded target variants"})
    except BaseException as exc:
        report["ok"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "mutation_retry_allowed": False}
    finally:
        if server:
            server.shutdown(); server.server_close(); worker.join(5)
        if learner:
            learner.close()
        if runtime:
            runtime.close()
    save(root, "report.json", report)
    print(json.dumps({"ok": report["ok"], "report": str(root/"report.json")}), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
