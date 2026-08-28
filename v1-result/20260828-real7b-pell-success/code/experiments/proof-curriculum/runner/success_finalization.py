"""Append-only successful replay admission, separate from online evidence."""
from __future__ import annotations

import hashlib
from pathlib import Path

from course_driver import read, require, safe, sha, write_new


def enabled(plan):
    return plan['runtime'].get('success_finalization', False) is True


def marker_path(family, lesson):
    return family / 'success-replay-imports' / (lesson['session_id'] + '.json')


def checked_replay(adapter, plan, lesson, family):
    from gpu_runtime.success_finalize_objective import prepare_success_event
    marker = read(marker_path(family, lesson))
    event = expected_event(adapter, plan, lesson, family, marker.get('dataset_sha256'))
    require(marker == {'schema_version': 'reap.course-success-import.v1',
        'plan_sha256': sha((family / 'plan.json').read_bytes()), 'session_id': lesson['session_id'],
        'dataset_sha256': event['dataset_sha256'], 'course_acceptance_sha256': event['course_acceptance_sha256']},
        'successful replay import binding differs')
    search = plan['runtime']['initialization_contract']['search_config']
    prepared = prepare_success_event(event, session_id=lesson['session_id'], policy_version=event['policy_version'],
        dataset_root=family / 'success-replay-store', gamma=search['gamma'], value_floor=search['value_floor'])
    return event, prepared


def expected_event(adapter, plan, lesson, family, dataset_sha256):
    _, checked = adapter.proof_functions(lesson)
    _, online = checked(family, lesson['session_id'], plan['runtime']['cpu_image'])
    event = {'kind': 'verified_success_discounted_v1', 'event_id': lesson['session_id'] + '.success',
        'session_id': lesson['session_id'], 'tree_id': lesson['session_id'] + '.tree0',
        'theorem_id': lesson['execution_source_sha256'], 'policy_version': online['policy_version'],
        'dataset_sha256': dataset_sha256,
        'course_acceptance_sha256': sha((family / 'proof-check' / lesson['session_id'] / 'accepted.json').read_bytes())}
    return event


def ready(adapter, plan, lesson, family):
    if not marker_path(family, lesson).exists():
        return False
    checked_replay(adapter, plan, lesson, family)
    return True


def receipt_rows(prepared):
    return [{'row': r['row'], 'node_index': r['node_index'], 'generation_sequence': r['generation_sequence'],
        'eval_sequence': r['eval_sequence'], 'source_policy_version': r['policy_version'], 'return': r['return'],
        'value_target': r['value_target'], 'prompt_sha256': hashlib.sha256(r['prompt'].encode()).hexdigest(),
        'tactic_sha256': hashlib.sha256(r['tactic'].encode()).hexdigest()} for r in prepared['samples']]


def import_replay(driver, index, bundle):
    from cpu_runtime.verified_dataset_store import install_verified_dataset
    from gpu_runtime.success_finalize_objective import prepare_success_event
    require(enabled(driver.plan), 'successful finalization not explicitly enabled')
    require(driver.state()['next_index'] == index, 'only current lesson may import successful replay')
    lesson = driver.plan['lessons'][index]
    require(driver.completion(lesson, 'verify') and not driver.completion(lesson, 'success-learn'),
            'independently checked proof before any terminal mutation required')
    marker = marker_path(driver.family, lesson)
    require(not marker.exists(), 'successful replay import already exists; never replace')
    digest = sha(safe(Path(bundle) / 'dataset.json').read_bytes())
    event = expected_event(driver.adapter, driver.plan, lesson, driver.family, digest)
    search = driver.plan['runtime']['initialization_contract']['search_config']
    options = dict(session_id=lesson['session_id'], policy_version=event['policy_version'],
        dataset_root=driver.family / 'success-replay-store', gamma=search['gamma'], value_floor=search['value_floor'])
    candidate = prepare_success_event(event, dataset_directory=safe(bundle), **options)
    # Store performs full load/copy/reload, rejects links and publishes once.
    install_verified_dataset(safe(bundle), driver.family / 'success-replay-store', expected_sha256=digest)
    require(prepare_success_event(event, **options) == candidate, 'successful replay changed during import')
    driver.validate_files()
    require(expected_event(driver.adapter, driver.plan, lesson, driver.family, digest) == event,
            'original course evidence changed during import')
    marker.parent.mkdir(exist_ok=True)
    write_new(marker, {'schema_version': 'reap.course-success-import.v1', 'plan_sha256': driver.plan_sha,
        'session_id': lesson['session_id'], 'dataset_sha256': digest,
        'course_acceptance_sha256': event['course_acceptance_sha256']})
    checked_replay(driver.adapter, driver.plan, lesson, driver.family)
    return {'status': 'imported_success_replay', 'dataset_sha256': digest, 'gpu_called': False,
            'deployment_required': 'install this exact digest bundle in the GPU trusted success dataset root before run'}


def replay_export(*, source_root, export_dir, output, lean_project):
    from production import Production
    from verification_exchange import validate_export
    from cpu_runtime.course_success_replay import export_course_success
    adapter = Production(source_root)
    _, plan, lesson = validate_export(adapter, safe(export_dir))
    require(enabled(plan), 'successful finalization not explicitly enabled')
    _, checked = adapter.proof_functions(lesson)
    return export_course_success(family=safe(export_dir), lesson=lesson, cpu_image=plan['runtime']['cpu_image'],
        checked_proof=checked, output=safe(output), lean_project=safe(lean_project),
        replay_module=adapter.root / 'containers/cpu/verified-replay/VerifiedReplay.lean')
