#!/usr/bin/env python3
"""Two explicit v3 updates: inherit v2 parameters, restore CP1, append verified data.

No search, download, threshold adjustment, mutation retry, or instance control.
The appended actor data must originate from the INITIAL release, not this probe's
first release. Run only with an exclusive output and an operator-owned single
writer CAS root. Existing v2 objects are read-only; new v3 objects append there.
Accept only report.ok plus the actual external worker exit=0; report publication
or fsync can itself have an unknown outcome and must never trigger a retry.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
import uuid

from gpu_runtime import GpuRuntime
from gpu_runtime.continual_mixed_learner import ContinualMixedLearner
from gpu_runtime.continual_mixed_store import ContinualMixedStore, sampling_transition
from gpu_runtime.learner_release_store import canonical_bytes, content_sha256
from gpu_runtime.mixed_backend import MixedReplayBackend
from gpu_runtime.mixed_objective import make_mixed_sampler, next_mixed_batch, SOURCE_COUNTS, SAMPLE_WEIGHT
from gpu_runtime.verified_objective import SHA256
from cpu_runtime import verified_dataset_store as safe
from containers.gpu import smoke_mixed_learner as mixed
from containers.gpu.smoke_gpu import require, equal_tree, base_fingerprint, finite_tree
from containers.gpu.smoke_kl_guard import encoded, save_capture, host_rng

LEARNER = 'continual-mixed-central'
ACTOR1, ACTOR2 = 'continual-release-actor1', 'continual-release-actor2'


def save(root, name, value):
    """Complete fsynced file, then exclusive hardlink publication (NFS-safe).

    Retain staging evidence. A post-publication fsync error propagates, with no
    blind retry or overwrite; the caller must also inspect the process exit.
    """
    require(Path(name).name == name, 'ordinary artifact filename required')
    raw = encoded(value)
    with safe.safe_directory(root) as directory:
        staging = '.stage-' + uuid.uuid4().hex
        directory.write_new(staging, raw)
        if os.name == 'nt':
            os.link(directory.path/staging, directory.path/name)
        else:
            os.link(staging, name, src_dir_fd=directory.handle, dst_dir_fd=directory.handle,
                    follow_symlinks=False)
        directory.sync()
    return {'file': name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def inventory(root, names=None):
    """Stream existing files; repeated hardlinks hash once per inventory only."""
    root = Path(root)
    with safe.safe_directory(root):
        if names is None:
            names = []
            for path in sorted(root.rglob('*')):
                info = path.lstat()
                require(not path.is_symlink() and not getattr(path, 'is_junction', lambda: False)(),
                        'linked source-store entry rejected')
                require(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode), 'nonregular source-store entry')
                if stat.S_ISREG(info.st_mode):
                    names.append(path.relative_to(root).as_posix())
        result, cache = {}, {}
        for name in sorted(names):
            relative = Path(name)
            require(not relative.is_absolute() and '..' not in relative.parts, 'invalid inventory path')
            with safe.safe_directory(root/relative.parent) as directory:
                fd = (safe._win_open(directory.path/relative.name, directory=False) if os.name == 'nt'
                      else os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                   dir_fd=directory.handle))
                with os.fdopen(fd, 'rb') as handle:
                    before = os.fstat(handle.fileno())
                    require(stat.S_ISREG(before.st_mode), 'source-store file must be regular')
                    key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                    if key not in cache:
                        digest = hashlib.sha256()
                        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
                            digest.update(chunk)
                        cache[key] = digest.hexdigest()
                    after = os.fstat(handle.fileno())
                    require(key == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                            'source-store file changed while hashing')
                    result[name] = {'sha256': cache[key], 'bytes': before.st_size}
        return result


def backend_options(source):
    contract = source['contract']; config = contract['mixed_config']
    require(contract['backend'] == 'mixed-replay' and config['lora']['dropout'] == 0.0,
            'unsupported source contract')
    maximum = config['max_post_update_kl']
    require(type(maximum) in (int, float) and math.isfinite(maximum) and maximum > 0,
            'source must already contain a finite positive KL guard; no threshold override')
    return dict(max_distance=config['support']['distance_max'],
        max_sequence_tokens=config['max_sequence_tokens'], max_post_update_kl=maximum,
        expected_hidden_size=config['hidden_size'], lora_rank=config['lora']['rank'],
        lora_alpha=config['lora']['alpha'], **{key: config[key] for key in (
            'learning_rate', 'value_learning_rate', 'value_coefficient', 'kl_beta', 'max_grad_norm')})


def plan_batches(datasets, initial_replay, human, append_pin, *, seed, max_distance):
    require(append_pin not in initial_replay and append_pin not in human, 'append pin must be new')
    def sampler(pins):
        return make_mixed_sampler(replay_pins=pins, mathlib_sft_pins=human,
            seed=seed, max_distance=max_distance, load_replay=lambda p: datasets['replay'][p],
            load_mathlib_sft=lambda p: datasets['mathlib_sft'][p])
    config, initial = sampler(initial_replay)
    expanded, _ = sampler(initial_replay + [append_pin])
    refs1, after1 = next_mixed_batch(config, initial)
    effective = {**after1, 'config_sha256': content_sha256(expanded)}
    refs2, after2 = next_mixed_batch(expanded, effective)
    require(any(r['source'] == 'replay' and r['dataset_sha256'] == append_pin for r in refs2),
            'new pin absent from planned second batch; reject before any learn')
    return dict(initial_replay=initial_replay, human=human, append_pin=append_pin,
        config=config, expanded_config=expanded, initial=initial,
        refs1=refs1, after1=after1, effective_before2=effective, refs2=refs2, after2=after2)


def prepare_inputs(args, source):
    options = backend_options(source)
    datasets, manifest, _, _ = mixed.load_inputs(args.replay_dataset_root,
        [*args.replay_dataset_sha256, args.append_replay_dataset_sha256],
        args.mathlib_dataset_root, args.mathlib_dataset_sha256,
        seed=args.sampler_seed, max_distance=options['max_distance'])
    raw = safe.read_bundle(args.replay_dataset_root/args.append_replay_dataset_sha256)
    session = json.loads(raw['session.json'])
    require(session.get('model_release_sha256') == source['model_release_sha256']
        and session.get('lineage', {}).get('model_release_sha256') == source['model_release_sha256']
        and session.get('role') == 'actor' and type(session.get('policy_version')) is int
        and session['policy_version'] == 0 and session.get('training_enabled') is False,
        'new verified data must preserve its actual initial-release actor lineage')
    plan = plan_batches(datasets, args.replay_dataset_sha256, args.mathlib_dataset_sha256,
        args.append_replay_dataset_sha256, seed=args.sampler_seed, max_distance=options['max_distance'])
    return datasets, manifest, plan


def validate_step(receipt, refs, datasets, expected_step, expected_config):
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
    require(canonical_bytes(config) == canonical_bytes(expected_config), "receipt changed source training contract")
    maximum = expected_config["max_post_update_kl"]
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
        and 0 <= measured <= maximum and guard.get("maximum") == maximum
        and guard.get("timing") == "after_optimizer_step_before_commit"
        and guard.get("reduction") == config["kl_reduction"]
        and guard.get("scope") == "current_verified_batch_prefixes_only",
        "actual source-contract KL guard measurement required")
    return {"source_counts": detail["source_counts"], "sample_weight": detail["sample_weight"],
        "source_losses": detail["source_losses"], "samples": detail["samples"]}


def audit(runtime, backend, datasets, plan, source, weights, output, implementation):
    learner = None
    try:
        require(canonical_bytes(backend.experience_contract()) == canonical_bytes(source['contract']),
                'loaded backend differs from complete source contract')
        # Repeat preflight at audit entry: invalid sampling cannot create a learner.
        require(plan == plan_batches(datasets, plan['initial_replay'], plan['human'], plan['append_pin'],
            seed=plan['config']['seed'], max_distance=backend.max_distance), 'sampling plan changed')
        host_before = runtime.actor.submit(lambda: host_rng(backend))
        base_before = runtime.actor.submit(lambda: base_fingerprint(backend, 8*1024*1024))
        learner = ContinualMixedLearner(runtime, learner_id=LEARNER,
            replay_pins=plan['initial_replay'], mathlib_sft_pins=plan['human'],
            sampler_seed=plan['config']['seed'], journal_root=output/'journal',
            initial_model_release_sha256=source['model_release_sha256'], implementation=implementation)
        require(learner.run['initialization'] == {'kind': 'learner_release', 'release_sha256': source['model_release_sha256']}
            and learner.run['sampler']['config'] == plan['config'] and learner.sampler_state == plan['initial'],
            'actual initial run differs from admitted plan')
        save(output, 'run.json', learner.run)
        before = mixed.capture(runtime, backend, LEARNER)
        source_weights = backend.torch.load(io.BytesIO(base64.b64decode(weights['payload'], validate=True)),
                                           map_location='cpu', weights_only=True)
        finite_tree(backend.torch, source_weights, 'source weights')
        same_weights = lambda x, y: all(equal_tree(backend.torch, x[k], y[k]) for k in ('adapter', 'value_head'))
        fresh = deepcopy(before); fresh['metadata']['role'] = 'actor'
        require(before['metadata']['role'] == 'learner' and before['metadata']['theorem_id'] is None
            and mixed.fresh_actor(fresh, source, None) and same_weights(before['backend'], source_weights),
            'learner must inherit source parameters with fresh local0/private state')
        started = time.perf_counter(); first = learner.train_next(); seconds1 = time.perf_counter()-started
        checked1 = validate_step(first['runtime_receipt'], plan['refs1'], datasets, 1, backend._config())
        state1 = mixed.capture(runtime, backend, LEARNER)
        require(learner.sampler_state == plan['after1'], 'first cursor mismatch')
        release1 = learner.publish()
        publish1 = equal_tree(backend.torch, state1, mixed.capture(runtime, backend, LEARNER))
        save(output, 'step1.json', {'step': first, 'release': release1, 'train_with_checkpoint_seconds': seconds1})
        theorem = datasets['replay'][plan['initial_replay'][0]]['theorem_sha256']
        runtime.create_session(ACTOR1, role='actor', theorem_id=theorem, model_release_sha256=release1['model_release_sha256'])
        actor1 = mixed.capture(runtime, backend, ACTOR1)
        artifacts = {'actor1_before': save_capture(output, 'actor1-before', backend, actor1)}
        learner.close(); runtime.delete_session(LEARNER)
        learner = ContinualMixedLearner.restore(runtime, checkpoint_sha256=first['checkpoint_sha256'],
                                                journal_root=output/'restored-journal')
        restored = mixed.capture(runtime, backend, LEARNER)
        require(equal_tree(backend.torch, state1, restored) and learner.sampler_state == plan['after1'],
                'CP1 complete private restore failed; no second learn')
        catalog1 = deepcopy(learner.catalog)
        started = time.perf_counter(); second = learner.train_next(append_dataset_pins=[plan['append_pin']])
        seconds2 = time.perf_counter()-started
        checked2 = validate_step(second['runtime_receipt'], plan['refs2'], datasets, 2, backend._config())
        state2 = mixed.capture(runtime, backend, LEARNER)
        cp1 = learner.store.load_checkpoint(first['checkpoint_sha256'])
        cp2 = learner.store.load_checkpoint(second['checkpoint_sha256'])
        config2, transition, refs2, after2 = sampling_transition(learner.run, catalog1, learner.catalog, plan['after1'])
        require(cp2['data_receipt']['sampler_transition'] == transition
            and transition['added_dataset_pins'] == [plan['append_pin']]
            and transition['effective_sampler_before'] == plan['effective_before2']
            and cp2['data_receipt']['batch_refs'] == refs2 == plan['refs2']
            and cp2['data_receipt']['sampler_config'] == config2 == plan['expanded_config']
            and learner.sampler_state == after2 == plan['after2'], 'append/cursor/data receipt mismatch')
        release2 = learner.publish()
        publish2 = equal_tree(backend.torch, state2, mixed.capture(runtime, backend, LEARNER))
        runtime.create_session(ACTOR2, role='actor', theorem_id=theorem, model_release_sha256=release2['model_release_sha256'])
        actor2 = mixed.capture(runtime, backend, ACTOR2)
        actor1_after = mixed.capture(runtime, backend, ACTOR1)
        artifacts.update(actor1_final=save_capture(output, 'actor1-final', backend, actor1_after),
                         actor2_initial=save_capture(output, 'actor2-initial', backend, actor2))
        base_after = runtime.actor.submit(lambda: base_fingerprint(backend, 8*1024*1024))
        host_after = runtime.actor.submit(lambda: host_rng(backend))
        gates = {'source_parameters_and_fresh_learner_state': True,
            'two_exact_source_contract_9_1_receipts': True, 'CP1_complete_private_restore_exact': True,
            'new_verified_pin_sampled_step2_with_cursors_preserved': True,
            'catalog_append_only': learner.catalog == catalog1 + [learner.catalog[-1]]
                and learner.catalog[-1]['dataset_sha256'] == plan['append_pin'],
            'checkpoint_parent_and_cursors_exact': cp1['sampler_state'] == plan['after1']
                and cp2['sampler_state'] == plan['after2']
                and cp2['manifest']['parent_checkpoint_sha256'] == first['checkpoint_sha256'],
            'both_steps_change_adapter_and_head': all(not equal_tree(backend.torch, x['backend'][k], y['backend'][k])
                for x, y in ((before, state1), (state1, state2)) for k in ('adapter', 'value_head')),
            'publishing_does_not_mutate_learner': publish1 and publish2,
            'actor1_R1_and_actor2_R2_exact': same_weights(actor1['backend'], state1['backend'])
                and same_weights(actor2['backend'], state2['backend']),
            'actor_states_fresh': mixed.fresh_actor(actor1, release1, theorem) and mixed.fresh_actor(actor2, release2, theorem),
            'actor_seeds_independent': len({x['backend']['rng']['seed'] for x in (actor1, actor2, state1)}) == 3,
            'old_actor_complete_state_unchanged': equal_tree(backend.torch, actor1, actor1_after),
            'source_release_unchanged': runtime.learner_releases.load_release(source['model_release_sha256']) == (source, weights),
            'new_releases_distinct_and_R1_immutable': release1['model_release_sha256'] != release2['model_release_sha256']
                and runtime.learner_releases.load_release(release1['model_release_sha256'])[0] == release1,
            'full_frozen_base_exact': base_before == base_after,
            'host_RNG_exact': equal_tree(backend.torch, host_before, host_after)}
        save(output, 'training-continuation.json', {'checkpoint_sha256': second['checkpoint_sha256'],
            'release_sha256': release2['model_release_sha256'], 'training_gates_passed': all(gates.values()),
            'input_integrity_pending': True, 'goal_complete': False})
        return dict(ok=all(gates.values()), gates=gates, first=first, second=second, release1=release1, release2=release2,
            run_scope=deepcopy(learner.run['scope']), specialist_scope_tested=False,
            checked_batches=[checked1, checked2], source_release=source, sampling_plan=plan,
            sampler_after_second=plan['after2'], append_transition=transition, artifacts=artifacts,
            full_base_before=base_before, full_base_after=base_after,
            contract_base_tokenizer_sha256=source['contract']['base_sha256'],
            train_with_checkpoint_seconds=[seconds1, seconds2],
            timing_scope='train_next including checkpoint; excludes loading, publish, restore, integrity and actor checks',
            appended_source='actor generated with initial source release; not generated with this probe release1',
            actor_metrics=runtime.actor.metrics())
    finally:
        if learner is not None:
            learner.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-path', 'expected-base-sha256', 'initial-model-release-sha256', 'append-replay-dataset-sha256'):
        parser.add_argument('--'+name, required=True)
    for name in ('source-release-root', 'replay-dataset-root', 'mathlib-dataset-root', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('replay-dataset-sha256', 'mathlib-dataset-sha256'):
        parser.add_argument('--'+name, action='append', required=True)
    parser.add_argument('--sampler-seed', type=int, required=True)
    return parser


def fail(report, key, exc):
    report['ok'] = report['real_7B_GPU_gate_passed'] = False
    report[key] = {'type': type(exc).__name__, 'message': str(exc), 'mutation_retry_allowed': False}


def finalize(root, report):
    """All audit/cleanup checks precede the last, atomic report publication."""
    try:
        if 'second' in report:
            save(root, 'final-continuation.json', {'checkpoint_sha256': report['second']['checkpoint_sha256'],
                'release_sha256': report['release2']['model_release_sha256'],
                'sampler_state': report['sampler_after_second'], 'stage_passed': report['ok'], 'goal_complete': False})
        report['worker_result'] = save(root, 'worker-result.json', {'ok': report['ok'],
            'planned_exit_code': 0 if report['ok'] else 1, 'report_publication_pending': True,
            'external_process_exit_still_required': True, 'mutation_retry_allowed': False})
    except BaseException as exc:
        fail(report, 'finalization_error', exc)
    save(root, 'report.json', report)
    return 0 if report['ok'] else 1


def main(argv=None):
    args = build_parser().parse_args(argv)
    for pin in (args.expected_base_sha256, args.initial_model_release_sha256, args.append_replay_dataset_sha256):
        require(bool(SHA256.fullmatch(pin)), 'explicit lowercase SHA256 pin required')
    source_root, output = args.source_release_root.absolute(), args.output_dir.absolute()
    require(source_root != output and source_root not in output.parents and output not in source_root.parents,
            'output and source-store roots must be separate')
    # Read-only source and strict data checks occur before model allocation/learn.
    old_files = inventory(source_root)
    store = ContinualMixedStore(source_root)
    source, weights = store.load_release(args.initial_model_release_sha256)
    require(store.load_checkpoint(source['source']['checkpoint_sha256'])['run']['schema_version'] == 'reap.learner.run.v2',
            'this probe explicitly requires the old v2 source release')
    require(source['contract']['base_sha256'] == args.expected_base_sha256, 'source base/tokenizer pin differs')
    datasets, input_before, plan = prepare_inputs(args, source)
    output.mkdir(parents=True, exist_ok=False)
    source_before = mixed.source_hashes()
    source_before['containers/gpu/smoke_continual_mixed_learner.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(output, 'input-manifest.json', {'source_release': source, 'old_store_files': old_files,
        'datasets': input_before, 'sampling_plan': plan, 'code': source_before,
        'command': [sys.executable, '-m', 'containers.gpu.smoke_continual_mixed_learner', *(sys.argv[1:] if argv is None else argv)]})
    report = dict(schema_version='reap.continual-mixed-learner.gpu-gate.v1', ok=False,
        real_7B_GPU_gate_passed=False, new_Lean_search=False, actor_search_performed=False,
        run_scope=None, specialist_scope_tested=False,
        curriculum_tested=False, performance_improvement_claimed=False,
        mutation_retry_allowed=False, instance_stop_requested=False)
    runtime = None
    try:
        mixed.require_amd()
        backend = MixedReplayBackend(args.model_path, dataset_root=args.replay_dataset_root,
            mathlib_dataset_root=args.mathlib_dataset_root, **backend_options(source))
        require(backend.hidden_size == 3584 and backend.device.type == 'cuda'
            and all(t.device == backend.device for t in backend.model.parameters()), 'actual device/7B shape mismatch')
        require(canonical_bytes(backend.experience_contract()) == canonical_bytes(source['contract']), 'full source contract mismatch')
        runtime = GpuRuntime(backend=backend, snapshot_root=output/'snapshots', learner_release_root=source_root,
            learner_profile='continual-mixed-v3', max_resident_sessions=3)
        report.update(audit(runtime, backend, datasets, plan, source, weights, output, source_before))
        _, inputs_after, plan_after = prepare_inputs(args, source)
        source_after = mixed.source_hashes()
        source_after['containers/gpu/smoke_continual_mixed_learner.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        require(inputs_after == input_before and plan_after == plan and source_after == source_before,
                'data, sampling plan or source code changed')
        require(inventory(source_root, old_files) == old_files, 'old source-store files changed')
        report.update(input_files_unchanged=True, old_source_files_unchanged=True, source_sha256_after=source_after)
        report['real_7B_GPU_gate_passed'] = report['ok']
    except BaseException as exc:
        fail(report, 'error', exc)
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as exc:
                fail(report, 'cleanup_error', exc)
    return finalize(output, report)


if __name__ == '__main__':
    raise SystemExit(main())
