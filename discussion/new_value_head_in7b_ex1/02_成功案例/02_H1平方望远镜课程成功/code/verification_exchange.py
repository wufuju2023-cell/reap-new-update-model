"""Operator-carried, request-bound offline proof exchange; no remote authentication.

The only verifier is the existing Podman helper plus checked_proof. Export and
import do not call GPU APIs, retry search, or rewrite existing proof evidence.
"""
from __future__ import annotations

import os
from pathlib import Path
import uuid

from course_driver import canonical, pin, read, require, safe, sha, sync_dir, write_new
from production import Production

SCHEMA = 'reap.teacher-course.verification-request.v1'


def proof_inputs(adapter, lesson, root):
    prepare = adapter.prepare_proof_function(lesson)
    return prepare(root, lesson['session_id'], lesson['input_file'], lesson['theorem'])


def verifier_pins(adapter):
    paths = {'extract_online_proof.py': adapter.root / 'v1-result/reproduction/scripts/extract_online_proof.py',
             'cross_experience.py': adapter.root / 'v1-result/reproduction/scripts/cross_experience.py',
             'production.py': Path(__file__).with_name('production.py'),
             'verification_exchange.py': Path(__file__)}
    paths.update({name: adapter.root / name for name in ('cpu_runtime/course_success_replay.py',
        'cpu_runtime/verified_trajectory.py', 'containers/cpu/verified-replay/VerifiedReplay.lean')})
    return {key: sha(safe(value).read_bytes()) for key, value in paths.items()}


def lesson_for(plan, index):
    require(type(index) is int and 0 <= index < len(plan['lessons']), 'invalid lesson index')
    return plan['lessons'][index]


def request_path(family, lesson):
    return family / 'verification-requests' / (lesson['session_id'] + '.json')


def imported_path(family, lesson):
    return family / 'verification-imports' / (lesson['session_id'] + '.json')


def make_request(adapter, plan, lesson, family):
    proof, _, bindings = proof_inputs(adapter, lesson, family)
    require(bindings['source']['sha256'] == lesson['execution_source_sha256'], 'request source differs from plan')
    request = {'schema_version': SCHEMA, 'plan_sha256': sha((family / 'plan.json').read_bytes()),
            'family_id': plan['family_id'], 'index': lesson['index'], 'session_id': lesson['session_id'],
            'input_file': lesson['input_file'], 'theorem': lesson['theorem'],
            'cpu_image': plan['runtime']['cpu_image'], 'input_files': bindings,
            'proof_sha256': sha(proof), 'verifier_sha256': verifier_pins(adapter),
            'trust_scope': 'operator-carried local evidence; not cryptographic remote attestation'}
    if plan['runtime'].get('success_finalization') is True:
        request['success_inputs'] = {name: {'path': f"outputs/{lesson['session_id']}/{name}",
            'sha256': sha((family / 'outputs' / lesson['session_id'] / name).read_bytes())}
            for name in ('raw_tree.json', 'observer.jsonl')}
    return request


def expected_paths(request):
    sid, name = request['session_id'], request['input_file']
    require(isinstance(name, str) and Path(name).name == name and '/' not in name and '\\' not in name,
            'unsafe verification input name')
    return {'source': 'inputs/' + name, 'session': f'outputs/{sid}/session.json',
            'result': f'outputs/{sid}/result.json', 'online_result': f'outputs/{sid}/online-result.json'}


def validate_export(adapter, directory):
    directory = safe(directory)
    request, plan = read(directory / 'request.json'), read(directory / 'plan.json')
    require(request.get('schema_version') == SCHEMA, 'unknown verification request')
    adapter.cross.identifier(request['session_id'])
    require(sha((directory / 'plan.json').read_bytes()) == request['plan_sha256'], 'export plan pin differs')
    require(plan['runtime'].get('verification_mode') == 'external_receipt', 'external receipt plan required')
    lesson = lesson_for(plan, request['index'])
    require(request['session_id'] == lesson['session_id'] and request['input_file'] == lesson['input_file']
            and request['theorem'] == lesson['theorem'] and request['family_id'] == plan['family_id']
            and request['cpu_image'] == plan['runtime']['cpu_image'], 'export maps a different lesson/image')
    expected = expected_paths(request)
    require(set(request['input_files']) == set(expected), 'exact four proof inputs required')
    for key, relative in expected.items():
        item = request['input_files'][key]
        require(item['path'] == relative and sha(safe(directory / relative).read_bytes()) == item['sha256'],
                'export input pin differs: ' + key)
    require(request['verifier_sha256'] == verifier_pins(adapter), 'local verifier implementation differs from request')
    if plan['runtime'].get('success_finalization') is True:
        extra = request.get('success_inputs', {})
        require(set(extra) == {'raw_tree.json', 'observer.jsonl'}, 'successful replay requires frozen tree and observer')
        for name, item in extra.items():
            require(item['path'] == f"outputs/{lesson['session_id']}/{name}"
                    and sha(safe(directory / item['path']).read_bytes()) == item['sha256'], 'successful replay input changed')
    proof, _, bindings = proof_inputs(adapter, lesson, directory)
    require(bindings == request['input_files'] and sha(proof) == request['proof_sha256']
            and bindings['source']['sha256'] == lesson['execution_source_sha256'], 'recomputed proof/request differs')
    require(read(directory / 'export-ready.json') == {'request_sha256': sha((directory / 'request.json').read_bytes())},
            'incomplete or changed export')
    return request, plan, lesson


def active_lesson(driver, index):
    require(driver.plan['runtime'].get('verification_mode') == 'external_receipt', 'external receipt mode required')
    lesson = lesson_for(driver.plan, index)
    state = driver.state()
    require(state['next_index'] == index, 'only current sequential lesson may exchange verification')
    online = driver.completion(lesson, 'online')
    require(online and online['root_verified'] is True and not driver.completion(lesson, 'verify'),
            'known solved online output, before verify intent, required')
    return lesson


def export_request(driver, index, output):
    lesson = active_lesson(driver, index)
    request = make_request(driver.adapter, driver.plan, lesson, driver.family)
    original = request_path(driver.family, lesson)
    original.parent.mkdir(exist_ok=True)
    if original.exists():
        require(read(original) == request, 'original verification request changed; do not reidentify')
    else:
        write_new(original, request)
    output = safe(output)
    require(not output.is_relative_to(driver.family), 'export must be separate from authoritative family')
    output.mkdir(parents=True, exist_ok=False)
    relatives = [*expected_paths(request).values(), *(x['path'] for x in request.get('success_inputs', {}).values())]
    for relative in relatives:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_new(destination, safe(driver.family / relative).read_bytes())
    write_new(output / 'plan.json', (driver.family / 'plan.json').read_bytes())
    write_new(output / 'request.json', original.read_bytes())
    write_new(output / 'export-ready.json', {'request_sha256': sha(original.read_bytes())})
    validate_export(driver.adapter, output)
    driver.validate_files()
    return {'status': 'exported_for_independent_lean', 'output': str(output),
            'request_sha256': sha(original.read_bytes()), 'session_id': lesson['session_id'], 'gpu_called': False}


def verify_export(*, source_root, export_dir, adapter=None, runner=None):
    """On a prepared CPU host: invoke actual offline Podman unless TEST injects runner."""
    adapter = adapter or Production(source_root)
    directory = safe(export_dir)
    request, plan, lesson = validate_export(adapter, directory)
    safe(directory / 'proof-check' / lesson['session_id'])
    verify, checked = adapter.proof_functions(lesson)
    options = {} if runner is None else {'runner': runner}
    verify(directory, lesson['session_id'], lesson['input_file'], lesson['theorem'], request['cpu_image'], **options)
    accepted, _ = checked(directory, lesson['session_id'], request['cpu_image'])
    validate_export(adapter, directory)
    return {'status': 'independently_verified_export', 'session_id': lesson['session_id'],
            'request_sha256': sha((directory / 'request.json').read_bytes()), 'proof_sha256': accepted['proof_sha256']}


def import_result(driver, index, export_dir):
    lesson = active_lesson(driver, index)
    source = safe(export_dir)
    request, _, exported_lesson = validate_export(driver.adapter, source)
    original = request_path(driver.family, lesson)
    require(original.is_file(), 'export request must already exist in the authoritative family')
    require(request == read(original) == make_request(driver.adapter, driver.plan, lesson, driver.family)
            and exported_lesson == lesson, 'returned receipt belongs to a different original request')
    _, checked = driver.adapter.proof_functions(lesson)
    accepted, _ = checked(source, lesson['session_id'], request['cpu_image'])
    proof_dir = source / 'proof-check' / lesson['session_id']
    names = set(accepted['files']) | {'accepted.json', 'receipt.json'}
    require({p.name for p in proof_dir.iterdir()} == names and all(safe(proof_dir / n).is_file() for n in names),
            'unexpected proof package members')
    destination = safe(driver.family / 'proof-check' / lesson['session_id'])
    require(not destination.exists() and not imported_path(driver.family, lesson).exists(),
            'existing proof/import evidence is never overwritten')
    # Copy into an isolated local verification root, then run every checker on
    # those exact copied bytes before publishing the proof directory atomically.
    staging = driver.family / ('.verification-import-' + uuid.uuid4().hex)
    staging.mkdir()
    for relative in (*expected_paths(request).values(), *(x['path'] for x in request.get('success_inputs', {}).values()),
                     'plan.json', 'request.json', 'export-ready.json'):
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        write_new(target, safe(source / relative).read_bytes())
    staged_proof = staging / 'proof-check' / lesson['session_id']
    staged_proof.mkdir(parents=True)
    for name in sorted(names):
        write_new(staged_proof / name, safe(proof_dir / name).read_bytes())
    copied_request, _, _ = validate_export(driver.adapter, staging)
    require(copied_request == read(original), 'request changed during import')
    staged_accepted, _ = checked(staging, lesson['session_id'], request['cpu_image'])
    require(staged_accepted == accepted, 'proof package changed during import')
    driver.validate_files()
    require(make_request(driver.adapter, driver.plan, lesson, driver.family) == request, 'remote inputs changed during import')
    destination.parent.mkdir(exist_ok=True)
    require(not destination.exists(), 'proof destination appeared during import')
    os.rename(staged_proof, destination)  # under the authoritative family writer lock
    sync_dir(destination.parent)
    marker = imported_path(driver.family, lesson)
    marker.parent.mkdir(exist_ok=True)
    write_new(marker, {'schema_version': 'reap.teacher-course.verification-import.v1',
        'request_sha256': sha(original.read_bytes()), 'accepted_sha256': sha((destination / 'accepted.json').read_bytes()),
        'session_id': lesson['session_id'], 'plan_sha256': driver.plan_sha})
    check_imported(driver.adapter, driver.plan, lesson, driver.family)
    return {'status': 'imported_independent_lean', 'session_id': lesson['session_id'],
            'request_sha256': sha(original.read_bytes()), 'gpu_called': False}


def check_imported(adapter, plan, lesson, family):
    marker = read(imported_path(family, lesson))
    request_file = request_path(family, lesson)
    request = read(request_file)
    require(request == make_request(adapter, plan, lesson, family), 'imported verification inputs/request changed')
    require(marker == {'schema_version': 'reap.teacher-course.verification-import.v1',
        'request_sha256': sha(request_file.read_bytes()),
        'accepted_sha256': sha((family / 'proof-check' / lesson['session_id'] / 'accepted.json').read_bytes()),
        'session_id': lesson['session_id'], 'plan_sha256': sha((family / 'plan.json').read_bytes())}, 'import marker differs')
    _, checked = adapter.proof_functions(lesson)
    checked(family, lesson['session_id'], plan['runtime']['cpu_image'])


def import_ready(adapter, plan, lesson, family):
    directory = family / 'proof-check' / lesson['session_id']
    marker = imported_path(family, lesson)
    if not directory.exists() and not marker.exists():
        return False
    require(directory.is_dir() and marker.is_file(), 'partial/unregistered proof import; preserve evidence for review')
    check_imported(adapter, plan, lesson, family)
    return True
