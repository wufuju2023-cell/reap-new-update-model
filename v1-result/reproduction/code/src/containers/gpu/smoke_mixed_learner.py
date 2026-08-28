#!/usr/bin/env python3
"""Two fixed 9/1 learner steps on AMD, using existing verified evidence only.

No Lean search, dataset append, adaptive threshold, model download, mutation
retry or instance control. A new output root is mandatory. Large state stays
in that remote persistent root; the JSON report is not a local tensor backup.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

from gpu_runtime import GpuRuntime
from gpu_runtime.mixed_backend import MixedReplayBackend
from gpu_runtime.mixed_objective import SOURCE_COUNTS, SAMPLE_WEIGHT, next_mixed_batch, make_mixed_sampler
from gpu_runtime.verified_objective import SHA256, validate_support
from cpu_runtime.verified_trajectory import load_verified_dataset
from cpu_runtime.verified_dataset_store import read_bundle, safe_directory
from cpu_runtime.mathlib_trajectory import load_mathlib_dataset, BUNDLE_FILES as MATHLIB_FILES
from containers.gpu.smoke_gpu import require, equal_tree, finite_tree, base_fingerprint
from containers.gpu.smoke_kl_guard import encoded, write_bytes, save_capture, host_rng

LEARNER = "mixed-central"
ACTOR1, ACTOR2 = "mixed-release-actor1", "mixed-release-actor2"


def save(root, name, value):
    return write_bytes(root/name, encoded(value))


def require_amd():
    import torch
    require(bool(torch.version.hip) and torch.cuda.is_available(), "actual AMD GPU required")


def capture(runtime, backend, sid):
    raw = runtime.inspect_backend(sid)
    require(raw["schema_version"] == MixedReplayBackend.SNAPSHOT_SCHEMA
        and raw["session_id"] == sid and raw["mixed_config"] == backend._config(), "mixed capture contract mismatch")
    state = backend.torch.load(io.BytesIO(base64.b64decode(raw["payload"], validate=True)),
        map_location="cpu", weights_only=True)
    finite_tree(backend.torch, state, sid)
    return {"backend": state, "metadata": runtime.sessions.get(sid).snapshot()}


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = set((root/"gpu_runtime").glob("*.py"))
    paths.update(root/"cpu_runtime"/name for name in (
        "__init__.py", "verified_trajectory.py", "verified_dataset_store.py", "mathlib_trajectory.py"))
    paths.update(root/"containers/gpu"/name for name in (
        "smoke_mixed_learner.py", "smoke_gpu.py", "smoke_kl_guard.py", "smoke_search_gpu.py"))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def load_inputs(replay_root, replay_pins, mathlib_root, mathlib_pins, *, seed, max_distance):
    pins = {"replay": replay_pins, "mathlib_sft": mathlib_pins}
    seen = set()
    for values in pins.values():
        require(isinstance(values, list) and bool(values), "both explicit pin catalogs required")
        for pin in values:
            require(isinstance(pin, str) and bool(SHA256.fullmatch(pin)) and pin not in seen,
                "duplicate/cross-source/invalid dataset pin")
            seen.add(pin)
    datasets, manifest = {s: {} for s in pins}, {s: {} for s in pins}
    for source, values in pins.items():
        root = Path(replay_root if source == "replay" else mathlib_root)
        for pin in values:
            directory = root/pin
            loader = load_verified_dataset if source == "replay" else load_mathlib_dataset
            dataset = loader(directory, expected_sha256=pin)
            if source == "replay":
                raw = read_bundle(directory)
            else:
                with safe_directory(directory) as reader:
                    raw = {name: reader.read(name) for name in sorted(MATHLIB_FILES)}
            require(hashlib.sha256(raw["dataset.json"]).hexdigest() == pin,
                "dataset changed while collecting its file manifest")
            # Bind the exact object used for later row checks to these bytes.
            require(dataset == json.loads(raw["dataset.json"]), "loaded dataset differs from manifested dataset")
            datasets[source][pin] = dataset
            manifest[source][pin] = {name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
                for name, value in sorted(raw.items())}
    config, initial = make_mixed_sampler(replay_pins=replay_pins, mathlib_sft_pins=mathlib_pins,
        seed=seed, max_distance=max_distance,
        load_replay=lambda pin: datasets["replay"][pin], load_mathlib_sft=lambda pin: datasets["mathlib_sft"][pin])
    return datasets, manifest, config, initial


def validate_step(receipt, refs, datasets, *, expected_step):
    require(receipt["applied"] is True and receipt["idempotent"] is False
        and receipt["policy_version"] == expected_step, "one new committed mixed update required")
    detail = receipt["detail"]
    require(detail["objective"] == MixedReplayBackend.OBJECTIVE_KIND
        and detail["source_counts"] == SOURCE_COUNTS and detail["sample_weight"] == SAMPLE_WEIGHT,
        "mixed receipt objective/ratio/weight differs")
    require([{key: row[key] for key in ("source", "dataset_sha256", "row")} for row in detail["samples"]] == refs,
        "actual receipt rows differ from deterministic sampler")
    for row in detail["samples"]:
        dataset = datasets[row["source"]][row["dataset_sha256"]]
        original = dataset["rows"][row["row"]]
        require(type(row["return"]) is int and row["return"] == original["return"]
            and row["distance"] == -row["return"] and row["value_class"] == -row["return"]-1,
            "mixed label was changed")
        if row["source"] == "replay":
            require(row["source_policy_version"] == original["policy_version"]
                and row["source_session_id"] == dataset["session_id"]
                and row["source_tree_id"] == dataset["tree_id"]
                and row["node_index"] == original["node_index"]
                and row["source_theorem_sha256"] == dataset["theorem_sha256"], "replay provenance differs")
        else:
            require(row["source_policy_version"] is None and row["mathlib_source"] == dataset["source"]
                and all(key not in row for key in ("source_session_id", "source_tree_id", "node_index")),
                "human source was relabeled as an actor")
    config = detail["training_config"]
    require(set(detail["source_losses"]) == set(SOURCE_COUNTS), "source loss groups differ")
    for source, count in SOURCE_COUNTS.items():
        losses = detail["source_losses"][source]
        require(losses["rows"] == count and losses["weight_sum"] == count*SAMPLE_WEIGHT, "source loss weighting differs")
        require(all(type(losses[key]) in (int, float) and math.isfinite(losses[key])
            for key in ("policy_loss", "value_loss", "kl", "loss")), "nonfinite source loss")
        require(math.isclose(losses["loss"], losses["policy_loss"]+config["kl_beta"]*losses["kl"]
            +config["value_coefficient"]*losses["value_loss"], rel_tol=1e-10, abs_tol=1e-10), "source total loss differs")
    for key in ("policy_loss", "value_loss", "kl", "loss"):
        require(math.isclose(sum(v[key] for v in detail["source_losses"].values()), detail[key],
            rel_tol=1e-10, abs_tol=1e-10), "source contributions do not sum to full batch loss")
    require(detail["optimizer_steps"] == expected_step and detail["examples_seen"] == 10*expected_step,
        "mixed optimizer/example counters differ")
    require(all(detail[key] is True for key in ("finite_loss", "finite_gradients", "finite_parameters", "finite_optimizer_state")),
        "mixed finite training gate failed")
    guard = detail.get("kl_guard", {})
    measured = guard.get("post_update_kl")
    require(guard.get("accepted") is True and type(measured) in (int, float) and math.isfinite(measured)
        and 0 <= measured <= 100.0 and guard.get("maximum") == config["max_post_update_kl"] == 100.0
        and guard.get("timing") == "after_optimizer_step_before_commit"
        and guard.get("reduction") == config["kl_reduction"]
        and guard.get("scope") == "current_verified_batch_prefixes_only",
        "actual fixed KL guard measurement/contract required")
    return {"source_counts": detail["source_counts"], "sample_weight": detail["sample_weight"],
        "source_losses": detail["source_losses"], "samples": detail["samples"]}


def fresh_actor(state, release, theorem_sha256):
    logical, private = state["metadata"], state["backend"]
    seed = int.from_bytes(hashlib.sha256(logical["session_id"].encode()).digest()[:8], "big") % 2**63
    return (logical["role"] == "actor" and logical["policy_version"] == 0
        and logical["completed"] is False and logical["theorem_id"] == theorem_sha256
        and logical["optimizer_metadata"]["steps"] == 0
        and logical["lineage"]["model_release_sha256"] == release["model_release_sha256"]
        and logical["lineage"]["source"] == release["source"]
        and logical["lineage"]["weights_sha256"] == release["weights_sha256"]
        and logical["lineage"]["reset"] == release["reset"]
        and private["optimizer_steps"] == 0 and private["examples_seen"] == 0
        and private["optimizer"]["state"] == {} and private["rng"]["seed"] == seed
        and logical["event_receipts"] == {}
        and logical["buffer_metadata"] == {"events": {}, "pending_event_ids": [], "consumed_event_ids": []})


def audit(runtime, backend, datasets, config, initial_state, output, implementation):
    from gpu_runtime.mixed_learner import MixedLearnerCoordinator
    learner = None
    try:
        host_before = runtime.actor.submit(lambda: host_rng(backend))
        base_before = runtime.actor.submit(lambda: base_fingerprint(backend, 8*1024*1024))
        learner = MixedLearnerCoordinator(runtime, learner_id=LEARNER,
            replay_pins=list(datasets["replay"]), mathlib_sft_pins=list(datasets["mathlib_sft"]),
            sampler_seed=config["seed"], journal_root=output/"journal", implementation=implementation)
        require(learner.run["initialization"] == {"kind": "base"}
            and learner.run["sampler"]["config"] == config and learner.sampler_state == initial_state,
            "mixed coordinator configuration differs from pinned preflight")
        save(output, "run.json", learner.run)
        before = capture(runtime, backend, LEARNER)
        refs1, expected1 = next_mixed_batch(config, initial_state)
        started = time.perf_counter(); first = learner.train_next(); seconds1 = time.perf_counter()-started
        checked1 = validate_step(first["runtime_receipt"], refs1, datasets, expected_step=1)
        first_state = capture(runtime, backend, LEARNER)
        require(learner.sampler_state == expected1, "first sampler commit differs")
        release1 = learner.publish()
        publish1_exact = equal_tree(backend.torch, first_state, capture(runtime, backend, LEARNER))
        save(output, "step1.json", {"step": first, "release": release1, "sampler_after": expected1,
            "train_with_checkpoint_seconds": seconds1})
        theorem = next(iter(datasets["replay"].values()))["theorem_sha256"]
        runtime.create_session(ACTOR1, role="actor", theorem_id=theorem,
            model_release_sha256=release1["model_release_sha256"])
        actor1 = capture(runtime, backend, ACTOR1)
        artifacts = {"actor1_before": save_capture(output, "actor1-before", backend, actor1)}
        learner.close(); runtime.delete_session(LEARNER)
        learner = MixedLearnerCoordinator.restore(runtime, checkpoint_sha256=first["checkpoint_sha256"],
            journal_root=output/"restored-journal")
        restore_exact = equal_tree(backend.torch, first_state, capture(runtime, backend, LEARNER))
        sampler_restore_exact = learner.sampler_state == expected1 and learner.run["sampler"]["config"] == config
        require(restore_exact and sampler_restore_exact, "complete CP1 restore failed; do not train again")
        refs2, expected2 = next_mixed_batch(config, expected1)
        started = time.perf_counter(); second = learner.train_next(); seconds2 = time.perf_counter()-started
        checked2 = validate_step(second["runtime_receipt"], refs2, datasets, expected_step=2)
        second_state = capture(runtime, backend, LEARNER)
        require(learner.sampler_state == expected2, "second sampler commit differs")
        release2 = learner.publish()
        publish2_exact = equal_tree(backend.torch, second_state, capture(runtime, backend, LEARNER))
        runtime.create_session(ACTOR2, role="actor", theorem_id=theorem,
            model_release_sha256=release2["model_release_sha256"])
        actor2 = capture(runtime, backend, ACTOR2)
        actor1_final = capture(runtime, backend, ACTOR1)
        artifacts.update({"actor1_final": save_capture(output, "actor1-final", backend, actor1_final),
            "actor2_initial": save_capture(output, "actor2-initial", backend, actor2)})
        base_after = runtime.actor.submit(lambda: base_fingerprint(backend, 8*1024*1024))
        host_after = runtime.actor.submit(lambda: host_rng(backend))
        checkpoint1 = runtime.learner_releases.load_checkpoint(first["checkpoint_sha256"])
        checkpoint2 = runtime.learner_releases.load_checkpoint(second["checkpoint_sha256"])
        same_weights = lambda x, y: all(equal_tree(backend.torch, x["backend"][k], y["backend"][k])
            for k in ("adapter", "value_head"))
        changed = lambda x, y, key: not equal_tree(backend.torch, x["backend"][key], y["backend"][key])
        gates = {"first_receipt_9_1_verified": True, "second_receipt_9_1_verified": True,
            "first_adapter_and_head_changed": all(changed(before, first_state, k) for k in ("adapter", "value_head")),
            "second_adapter_and_head_changed": all(changed(first_state, second_state, k) for k in ("adapter", "value_head")),
            "CP1_complete_private_restore_exact": restore_exact, "sampler_restore_exact": sampler_restore_exact,
            "two_cursor_commits_exact": checkpoint1["sampler_state"] == expected1 and checkpoint2["sampler_state"] == expected2,
            "publishing_does_not_mutate_learner": publish1_exact and publish2_exact,
            "actor1_exact_R1_weights": same_weights(actor1, first_state),
            "actor2_exact_R2_weights": same_weights(actor2, second_state),
            "new_actor_private_states_fresh": fresh_actor(actor1, release1, theorem) and fresh_actor(actor2, release2, theorem),
            "actor_seeds_independent": actor1["backend"]["rng"]["seed"] != actor2["backend"]["rng"]["seed"]
                and actor1["backend"]["rng"]["seed"] != first_state["backend"]["rng"]["seed"],
            "old_actor_complete_state_unchanged": equal_tree(backend.torch, actor1, actor1_final),
            "R1_immutable_and_R2_distinct": runtime.learner_releases.load_release(release1["model_release_sha256"])[0] == release1
                and release1["model_release_sha256"] != release2["model_release_sha256"],
            "full_frozen_base_exact": base_before == base_after,
            "host_RNG_exact": equal_tree(backend.torch, host_before, host_after)}
        save(output, "training-continuation.json", {"checkpoint_sha256": second["checkpoint_sha256"],
            "release_sha256": release2["model_release_sha256"], "sampler_state": expected2,
            "training_gates_passed": all(gates.values()), "input_integrity_check_pending": True,
            "goal_complete": False, "instance_stop_requested": False})
        sampled = [*refs1, *refs2]
        return {"ok": all(gates.values()), "gates": gates, "first": first, "second": second,
            "release1": release1, "release2": release2, "checked_batches": [checked1, checked2],
            "sampler_initial": initial_state, "sampler_after_first": expected1, "sampler_after_second": expected2,
            "training_config": backend._config(), "base_before": base_before, "base_after": base_after,
            "train_with_checkpoint_seconds": [seconds1, seconds2],
            "timing_scope": "each train_next including checkpoint; excludes model preparation, publish, restore and actor checks",
            "sampled_unique_rows": {s: len({(r["dataset_sha256"], r["row"]) for r in sampled if r["source"] == s}) for s in SOURCE_COUNTS},
            "catalog_rows": {s: sum(len(d["rows"]) for d in datasets[s].values()) for s in SOURCE_COUNTS},
            "artifacts": artifacts, "large_states": "remote persistent output/store only; report is not a local tensor backup",
            "actor_metrics": runtime.actor.metrics()}
    finally:
        if learner is not None:
            learner.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--replay-dataset-root", type=Path, required=True)
    parser.add_argument("--replay-dataset-sha256", action="append", required=True)
    parser.add_argument("--mathlib-dataset-root", type=Path, required=True)
    parser.add_argument("--mathlib-dataset-sha256", action="append", required=True)
    parser.add_argument("--sampler-seed", type=int, required=True)
    parser.add_argument("--max-distance", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_support(args.max_distance)
    require(bool(SHA256.fullmatch(args.expected_base_sha256)), "explicit base identity pin required")
    datasets, inputs_before, config, initial = load_inputs(args.replay_dataset_root, args.replay_dataset_sha256,
        args.mathlib_dataset_root, args.mathlib_dataset_sha256, seed=args.sampler_seed, max_distance=args.max_distance)
    root = args.output_dir; root.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "containers.gpu.smoke_mixed_learner", *(sys.argv[1:] if argv is None else argv)]
    before = source_hashes()
    save(root, "input-manifest.json", {"datasets": inputs_before, "sampler_config": config,
        "source_sha256": before, "command": command, "command_sha256": hashlib.sha256(encoded(command)).hexdigest()})
    report = {"schema_version": "reap.mixed-learner.gpu-gate.v1", "ok": False,
        "real_7B_GPU_gate_passed": False, "new_Lean_search": False, "actor_search_performed": False,
        "curriculum_tested": False, "performance_improvement_claimed": False,
        "mutation_retry_allowed": False, "instance_stop_requested": False}
    runtime = None
    try:
        require_amd()
        backend = MixedReplayBackend(args.model_path, dataset_root=args.replay_dataset_root,
            mathlib_dataset_root=args.mathlib_dataset_root, max_distance=args.max_distance, max_post_update_kl=100.0)
        require(backend.hidden_size == 3584 and backend.experience_contract()["base_sha256"] == args.expected_base_sha256,
            "actual loaded 7B base/tokenizer identity differs from the explicit pin")
        runtime = GpuRuntime(backend=backend, snapshot_root=root/"snapshots",
            learner_release_root=root/"learner-store", max_resident_sessions=3)
        report.update(audit(runtime, backend, datasets, config, initial, root, before))
        _, inputs_after, after_config, _ = load_inputs(args.replay_dataset_root, args.replay_dataset_sha256,
            args.mathlib_dataset_root, args.mathlib_dataset_sha256, seed=args.sampler_seed, max_distance=args.max_distance)
        after = source_hashes()
        require(inputs_before == inputs_after and config == after_config and before == after,
            "inputs or executable source changed during the GPU gate")
        report["input_files_unchanged"] = True; report["source_sha256_after"] = after
        report["real_7B_GPU_gate_passed"] = report["ok"]
    except BaseException as exc:
        report["ok"] = report["real_7B_GPU_gate_passed"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "mutation_retry_allowed": False}
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as exc:
                report["ok"] = report["real_7B_GPU_gate_passed"] = False
                report["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
    if "second" in report and "release2" in report:
        save(root, "final-continuation.json", {"checkpoint_sha256": report["second"]["checkpoint_sha256"],
            "release_sha256": report["release2"]["model_release_sha256"],
            "sampler_state": report["sampler_after_second"], "stage_passed": report["ok"],
            "goal_complete": False, "instance_stop_requested": False})
    save(root, "report.json", report)
    print(json.dumps({"ok": report["ok"], "report": str(root/"report.json")}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
