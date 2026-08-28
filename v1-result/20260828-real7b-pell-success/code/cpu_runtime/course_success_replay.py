"""Adapt an already checked course proof to the existing independent replay.

The caller supplies Production's privately bound checked_proof function. This
module does not infer acceptance from an isSolved flag or an arbitrary JSON.
Run export in the pinned offline Lean environment; no GPU API is called here.
"""
from __future__ import annotations

import json
from pathlib import Path

from .verified_trajectory import encode, export_verified, require, sha, write_new


def export_course_success(*, family: Path, lesson: dict, cpu_image: str, checked_proof,
                          output: Path, lean_project: Path, replay_module: Path, timeout: float = 180):
    accepted, online = checked_proof(family, lesson['session_id'], cpu_image)
    proof_dir = family / 'proof-check' / lesson['session_id']
    accepted_raw = (proof_dir / 'accepted.json').read_bytes()
    require(json.loads(accepted_raw) == accepted, 'checked acceptance differs from persisted bytes')
    require(online.get('error') is None and online.get('root_verified') is True,
            'known successful original execution required')
    # A separately named envelope maps facts already checked by checked_proof.
    # The original accepted.json, container receipt and logs are never rewritten.
    envelope = {
        'proof_from_session': lesson['session_id'],
        'source_theorem_sha256': accepted['theorem_sha256'],
        'generated_proof_sha256': accepted['proof_sha256'],
        'returncode': accepted['lean_returncode'], 'network': accepted['network'],
        'image': accepted['image'].removeprefix('sha256:'),
        'container_exit_code': accepted['container_exit_code'], 'container_image': accepted['image'],
        'stdout': (proof_dir / 'stdout.log').read_bytes().decode('utf-8'),
        'stderr': (proof_dir / 'stderr.log').read_bytes().decode('utf-8'),
        'course_acceptance': {'schema_version': 'reap.course-proof-replay-binding.v1',
            'accepted_sha256': sha(accepted_raw), 'accepted_utf8': accepted_raw.decode('utf-8'),
            'accepted': accepted},
    }
    output = Path(output).absolute()
    inputs = output.with_name(output.name + '-course-receipt')
    require(not output.exists() and not inputs.exists(), 'replay/envelope exists; never retry an uncertain export')
    inputs.mkdir(parents=True)
    write_new(inputs / 'receipt.json', encode(envelope))
    write_new(inputs / 'proof.stdout', envelope['stdout'].encode())
    write_new(inputs / 'proof.stderr', envelope['stderr'].encode())
    dataset = export_verified(session_dir=family / 'outputs' / lesson['session_id'],
        source=family / 'inputs' / lesson['input_file'], proof=proof_dir / 'proof.lean',
        proof_receipt=inputs / 'receipt.json', theorem=lesson['theorem'], output=output,
        lean_project=lean_project, replay_module=replay_module, timeout=timeout)
    require(checked_proof(family, lesson['session_id'], cpu_image) == (accepted, online)
            and (proof_dir / 'accepted.json').read_bytes() == accepted_raw,
            'course evidence changed during independent replay')
    return {'dataset_sha256': sha((output / 'dataset.json').read_bytes()),
            'course_acceptance_sha256': sha(accepted_raw), 'rows': len(dataset['rows']),
            'session_id': dataset['session_id'], 'final_policy_version': dataset['final_policy_version']}


def validate_success_receipt(receipt, event, *, expected_rows, expected_training_config):
    """Admission after one submitted LEARN; a bad/unknown response is not retried."""
    import math
    from gpu_runtime.success_finalize_objective import OBJECTIVE_KIND
    require(receipt.get('event_id') == event['event_id'] and receipt.get('applied') is True
            and receipt.get('idempotent') is False and type(receipt.get('policy_version')) is int
            and receipt['policy_version'] == event['policy_version'] + 1, 'terminal learn identity/version mismatch')
    detail = receipt.get('detail', {})
    require(detail.get('objective') == OBJECTIVE_KIND and detail.get('source') == event
            and detail.get('optimizer_steps') == receipt['policy_version'], 'terminal learn source/steps mismatch')
    require(detail.get('training_config') == expected_training_config, 'terminal training config mismatch')
    require(detail.get('online_update_consumed_by_later_generation') is False,
            'terminal update must not claim online continuation')
    for flag in ('finite_loss', 'finite_gradients', 'finite_parameters', 'finite_optimizer_state', 'base_parameters_frozen'):
        require(detail.get(flag) is True, 'terminal learn failed ' + flag)
    for key in ('loss', 'policy_loss', 'kl', 'value_loss', 'grad_norm'):
        require(type(detail.get(key)) in (int, float) and math.isfinite(detail[key]), 'nonfinite terminal ' + key)
    samples = detail.get('samples')
    require(isinstance(samples, list) and len(samples) == len(expected_rows), 'terminal receipt row count differs')
    for actual, expected in zip(samples, expected_rows):
        require(isinstance(actual, dict) and set(actual) == set(expected) | {'target_tokens'}
                and {k: v for k, v in actual.items() if k != 'target_tokens'} == expected
                and type(actual['target_tokens']) is int and actual['target_tokens'] > 0,
                'terminal receipt row provenance/targets differ')
    for group in ('adapter', 'value_head', 'optimizer'):
        diff = detail.get('parameter_diffs', {}).get(group, {})
        require(diff.get('before_sha256') != diff.get('after_sha256')
                and diff.get('changed_tensors', 0) + diff.get('added_tensors', 0) > 0,
                'terminal update did not change ' + group)
