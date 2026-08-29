#!/usr/bin/env python3
"""Fail-closed, offline recovery for one acknowledged terminal success learn.

This entry never constructs an HTTP client.  It only validates the immutable
course intent, synchronous response, replay dataset and an operator-captured
actor-ledger receipt before optionally writing the missing phase completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
_SOURCE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_SOURCE_ROOT))
from course_driver import Driver, pin, read, require, safe, sha, write_new
from success_finalization import checked_replay, receipt_rows
from cpu_runtime.course_success_replay import validate_success_receipt as _SCALAR_VALIDATE


VALUE_TRAINING_FIELDS = {
    'clipped_distance', 'distance', 'loss_kind', 'lower_bin', 'lower_weight',
    'prediction', 'saturated', 'support_max', 'support_min', 'target',
    'target_nonzero', 'upper_bin', 'upper_weight',
}


def _number(value, name):
    require(type(value) in (int, float) and math.isfinite(value), 'invalid categorical ' + name)
    return float(value)


def _categorical_audit(audit, expected, config):
    require(type(audit) is dict and set(audit) == VALUE_TRAINING_FIELDS,
            'categorical value_training schema differs')
    support = config['categorical_value']['support']
    lo, hi = support['distance_min'], support['distance_max']
    require(type(lo) is int and type(hi) is int and (lo, hi) == (1, 64),
            'categorical support contract differs')
    distance = float(-expected['return'])
    clipped = min(float(hi), max(float(lo), distance))
    lower, upper = math.floor(clipped), math.ceil(clipped)
    upper_weight = clipped - lower
    lower_weight = 1.0 - upper_weight
    nonzero = [{'bin': lower, 'weight': lower_weight}]
    if upper != lower and upper_weight:
        nonzero.append({'bin': upper, 'weight': upper_weight})
    require(_number(audit['distance'], 'distance') == distance
            and _number(audit['clipped_distance'], 'clipped_distance') == clipped
            and _number(audit['target'], 'target') == clipped,
            'categorical distance/target differs from verified return')
    require(audit['loss_kind'] == 'two_hot_categorical_cross_entropy'
            and audit['support_min'] == lo and audit['support_max'] == hi
            and audit['lower_bin'] == lower and audit['upper_bin'] == upper
            and _number(audit['lower_weight'], 'lower_weight') == lower_weight
            and _number(audit['upper_weight'], 'upper_weight') == upper_weight
            and audit['target_nonzero'] == nonzero,
            'categorical two-hot projection differs')
    prediction = _number(audit['prediction'], 'prediction')
    require(lo <= prediction <= hi and audit['saturated'] is (distance >= hi),
            'categorical prediction/saturation differs')
    return clipped


def validate_success_receipt(receipt, event, *, expected_rows, expected_training_config):
    """Strict superset of the scalar receipt contract for categorical heads."""
    categorical = (expected_training_config.get('head') == 'categorical-64'
                   and type(expected_training_config.get('categorical_value')) is dict)
    if not categorical:
        _SCALAR_VALIDATE(receipt, event, expected_rows=expected_rows,
                         expected_training_config=expected_training_config)
        return
    # Reuse every non-row gate from the established validator with a projected
    # scalar-shaped receipt, then validate the categorical extension exactly.
    projected = {**receipt, 'detail': {**receipt.get('detail', {})}}
    samples = receipt.get('detail', {}).get('samples')
    require(type(samples) is list and len(samples) == len(expected_rows),
            'terminal receipt row count differs')
    scalar_samples = []
    for actual, expected in zip(samples, expected_rows):
        require(type(actual) is dict and set(actual) == set(expected) | {'target_tokens', 'value_training'},
                'terminal categorical row schema differs')
        categorical_target = _categorical_audit(actual['value_training'], expected, expected_training_config)
        # Categorical training reports clipped distance, whereas the scalar
        # contract reports gamma-decoded value. All remaining provenance is exact.
        expected_categorical = {**expected, 'value_target': categorical_target}
        require({k: v for k, v in actual.items() if k not in ('target_tokens', 'value_training')}
                == expected_categorical,
                'terminal categorical row provenance/targets differ')
        require(type(actual['target_tokens']) is int and actual['target_tokens'] > 0,
                'terminal categorical target token count differs')
        scalar_samples.append({**expected, 'target_tokens': actual['target_tokens']})
    projected['detail']['samples'] = scalar_samples
    _SCALAR_VALIDATE(projected, event, expected_rows=expected_rows,
                     expected_training_config=expected_training_config)


def recover(family, index, actor_ledger, *, commit=False):
    family, actor_ledger = safe(family), safe(actor_ledger)
    driver = Driver(family)
    require(type(index) is int and 0 <= index < len(driver.plan['lessons']), 'invalid lesson index')
    lesson = driver.plan['lessons'][index]
    require(driver.completion(lesson, 'online')['root_verified'] is True
            and driver.completion(lesson, 'verify') is not None,
            'verified successful original course required')
    operation = driver.operation(lesson, 'success-learn')
    require(operation.is_dir() and not (operation / 'complete.json').exists(),
            'recovery requires exactly one unresolved success-learn phase')
    intent, response = read(operation / 'intent.json'), read(operation / 'response.json')
    require(intent == {'schema_version': 'reap.teacher-course.intent.v1',
        'plan_sha256': driver.plan_sha, 'phase': 'success-learn',
        'session_id': lesson['session_id'], 'mutation_retry_allowed': False,
        'event': intent.get('event')}, 'success-learn intent identity differs')
    event, prepared = checked_replay(driver.adapter, driver.plan, lesson, driver.family)
    require(intent['event'] == event, 'success-learn intent event differs from trusted replay')
    validate_success_receipt(response, event, expected_rows=receipt_rows(prepared),
        expected_training_config=driver.plan['runtime']['initialization_contract']['search_config'])
    ledger = read(actor_ledger)
    require(ledger == {'schema_version': 'reap.course-success-recovery-ledger.v1',
        'plan_sha256': driver.plan_sha, 'session_id': lesson['session_id'],
        'event_id': event['event_id'], 'response_sha256': sha((operation / 'response.json').read_bytes()),
        'policy_version': response['policy_version'], 'applied': True,
        'unknown_or_pending': False, 'actor': ledger.get('actor')},
        'actor ledger binding differs')
    actor = ledger['actor']
    require(type(actor) is dict and set(actor) == {'submitted', 'started', 'completed', 'failed', 'active', 'queued'}
            and all(type(actor[k]) is int and actor[k] >= 0 for k in actor)
            and actor['submitted'] == actor['started'] == actor['completed']
            and actor['failed'] == actor['active'] == actor['queued'] == 0,
            'actor ledger is not fully applied/settled')
    evidence = [pin(operation / 'response.json'), pin(actor_ledger)]
    dataset = driver.family / 'success-replay-store' / event['dataset_sha256']
    evidence.extend(pin(p) for p in driver.tree_files(dataset))
    result = {'policy_version': response['policy_version'], 'terminal_updates': 1,
              'recovered_without_network_or_mutation': True}
    report = {'status': 'validated_dry_run' if not commit else 'recovered_complete',
              'session_id': lesson['session_id'], 'policy_version': response['policy_version'],
              'intent_sha256': sha((operation / 'intent.json').read_bytes()),
              'response_sha256': sha((operation / 'response.json').read_bytes()),
              'dataset_sha256': event['dataset_sha256'], 'actor_ledger_sha256': sha(actor_ledger.read_bytes()),
              'network_calls': 0, 'mutation_calls': 0}
    if commit:
        write_new(operation / 'complete.json', {'intent_sha256': report['intent_sha256'],
                                                'result': result, 'evidence': evidence})
        require(driver.completion(lesson, 'success-learn') == result, 'recovered completion failed reload')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--actor-ledger', type=Path, required=True)
    parser.add_argument('--commit', action='store_true')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(recover(args.family, args.index, args.actor_ledger, commit=args.commit),
                         sort_keys=True, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'blocked', 'reason': str(exc),
                          'mutation_retry_allowed': False}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
