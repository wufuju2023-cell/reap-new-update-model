#!/usr/bin/env python3
"""Finite single-family teacher curriculum, append-only intents, explicit execution.

Local operator files are trusted assertions, not authenticated remote evidence.
No central/mixed learning, implicit retry, automatic recovery, or scheduling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid
from urllib.parse import urlsplit

sys.dont_write_bytecode = True
from production import Production

SCHEMA = 'reap.teacher-course.plan.v1'
RELATION = 'operator_reviewed_teaching_dependencies_not_logical_implication_certificates'
PINS = ('experience_id', 'experience_weights_sha256', 'experience_snapshot_sha256')
RUNTIME_FIELDS = {'project_dir', 'environment_sha256', 'cpu_image', 'base_sha256', 'gpu_base_url',
                  'http_timeout_seconds', 'barrier_timeout_seconds', 'max_request_body_bytes',
                  'scoring_mode', 'model_context_tokens', 'lean_bin', 'initialization_contract'}
RUNTIME_OPTIONAL_FIELDS = {'verification_mode', 'success_finalization', 'training_snapshot_interval',
                           'total_deadline_seconds', 'selection_value_refresh'}
APPROVAL_CHECKS = ['pinned_cpu_environment', 'tokenwise_scoring', 'context_4096',
                   'checkpoint_interval_supported', 'bridge_body_limit_and_long_timeouts',
                   'real_search_backend_and_base_identity']


class Blocked(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise Blocked(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)()
                    for p in (path, *path.parents)), 'symlink/junction refused: ' + str(path))
    return path


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(safe(path).read_bytes(), object_pairs_hook=unique,
                       parse_constant=lambda _: require(False, 'nonfinite JSON'))
    require(type(value) is dict, 'JSON object required')
    return value


def sync_dir(path):
    if os.name != 'nt':
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_new(path, value):
    """Flush bytes then no-replace hard-link publication, including on POSIX."""
    path = safe(path)
    temporary = path.with_name('.staged-' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(value if isinstance(value, bytes) else canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path, follow_symlinks=False)
        sync_dir(path.parent)
    finally:
        temporary.unlink()  # only our own exact staging file, never old evidence


def pin(path):
    path = safe(path)
    return {'path': str(path), 'sha256': sha(path.read_bytes())}


def validate_pin(item):
    require(pin(item['path']) == item, 'pinned source/evidence changed: ' + item['path'])


def digest_value(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'lowercase SHA256 required')


def validate_runtime(config):
    require(RUNTIME_FIELDS <= set(config) <= RUNTIME_FIELDS | RUNTIME_OPTIONAL_FIELDS,
            'explicit runtime fields required: ' + ', '.join(sorted(RUNTIME_FIELDS)))
    if 'total_deadline_seconds' in config:
        from cpu_runtime.search_deadline import SearchDeadline
        try:
            SearchDeadline(config['total_deadline_seconds'])
        except ValueError as exc:
            raise Blocked(str(exc)) from exc
    require(config.get('verification_mode', 'local_container') in ('local_container', 'external_receipt'),
            'unknown verification mode')
    if 'success_finalization' in config:
        from gpu_runtime.success_finalize_objective import CONTRACT
        require(config['success_finalization'] is True
                and config['initialization_contract']['search_config'].get('success_finalization') == CONTRACT,
                'success finalization requires explicit backend capability and one full-trajectory update budget')
    if 'training_snapshot_interval' in config:
        require(type(config['training_snapshot_interval']) is int and config['training_snapshot_interval'] > 0,
                'training_snapshot_interval must be an explicit positive integer')
    if 'selection_value_refresh' in config:
        require(type(config['selection_value_refresh']) is bool,
                'selection_value_refresh must be explicitly boolean')
    for key in ('environment_sha256', 'base_sha256'):
        digest_value(config[key])
    require(isinstance(config['cpu_image'], str)
            and re.fullmatch(r'(?:sha256:)?[0-9a-f]{64}', config['cpu_image']), 'immutable CPU image required')
    require(Path(config['project_dir']).is_absolute() and safe(config['project_dir']).is_dir(),
            'absolute existing pinned CPU project directory required')
    require(config['scoring_mode'] == 'tokenwise' and type(config['model_context_tokens']) is int
            and config['model_context_tokens'] == 4096, 'tokenwise and 4096 context required')
    require(type(config['max_request_body_bytes']) is int
            and 8192 <= config['max_request_body_bytes'] <= 262144, 'invalid body limit')
    timeout = config['http_timeout_seconds']
    barrier = config['barrier_timeout_seconds']
    require(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
            and type(barrier) is int and barrier > timeout, 'barrier must exceed explicit client timeout')
    url = urlsplit(config['gpu_base_url'])
    require(url.scheme in ('http', 'https') and url.hostname and not url.username and not url.password
            and url.path == '' and not url.query and not url.fragment, 'GPU URL must be a credential-free origin')
    require(isinstance(config['lean_bin'], str) and config['lean_bin'], 'Lean binary required')


def membership(record_path, family_id):
    """Explicit record input; no built-in problem names or experiment paths."""
    record_path = safe(record_path)
    record = read(record_path)
    schema = record.get('schema_version')
    if schema == 'reap.teacher-course.members.v1':
        require(record.get('family_id') == family_id and record.get('approved_for_execution') is True,
                'exact explicitly approved family required')
        require(record.get('relation_kind') == RELATION
                and record.get('teacher_reference_imported_by_student') is False, 'teacher relation/visibility mismatch')
        source, target = record_path.parent / record['source'], record['target']
        members = [item['declaration'] for item in record['lessons']] + [target]
    elif schema == 'reap.frontiermath.student-declarations.v1':
        # Legacy reader only. No implicit selection or dispatch of official problems.
        require(record.get('teacher_reference_imported') is False, 'teacher proof must remain separate')
        source = record_path.parent / record['source']
        target = record['primary_original_target']
        members = [item['declaration'] for item in record['lessons']] + [target]
        require(all(item.get('teacher_proof_visible_to_student') is False for item in record['lessons']),
                'teacher proof visibility changed')
    elif schema in ('reap.codex-five.course-plan.v1', 'reap.codex-five.course-plan.v2'):
        require(record.get('relation_kind') == RELATION
                and record.get('teacher_reference_imported_by_student') is False, 'unknown teacher relation')
        matches = [p for p in record['problems'] if p['id'] == family_id]
        require(len(matches) == 1, 'family not explicitly in reviewed course record')
        problem = matches[0]
        source, target = record_path.parent / record['source'], problem['target']
        stages = problem['stages']
        require([s['stage'] for s in stages] == list(range(1, len(stages) + 1)), 'nonsequential teacher stages')
        require(problem['final_stage']['declaration'] == target, 'final target mismatch')
        members = [d for stage in stages for d in stage['declarations']] + [target]
    else:
        raise Blocked('unknown explicit course schema')
    require(safe(source).parent == safe(record_path.parent), 'course source must be an immediate sibling')
    require(sha(source.read_bytes()) == record['source_sha256'], 'teacher source hash mismatch')
    require(0 < len(members) <= 64 and len(set(members)) == len(members), 'bounded unique course members required')
    return record_path, source, target, members


def prepare(*, source_root, store, family_id, course_record, budget_file, runtime_path,
            budget='main', exhausted_policy='stop', seed_path=None, visibility_plan=None, premise_plan=None,
            session_nonce=None):
    root, store = safe(source_root), safe(store)
    require(re.fullmatch(r'[a-z][a-z0-9-]{0,39}', family_id), 'invalid family ID')
    require(not store.is_relative_to(root) and not root.is_relative_to(store), 'store must be isolated from source project')
    runtime = read(runtime_path)
    adapter = Production(root)
    validate_runtime(runtime)
    require(exhausted_policy in ('stop', 'continue'), 'explicit known-exhaustion policy required')
    adapter.policy.validate_contract(runtime['initialization_contract'])
    require(runtime['initialization_contract']['base_sha256'] == runtime['base_sha256']
            and runtime['initialization_contract']['search_config']['gamma'] == 0.99,
            'runtime initialization contract/base/gamma mismatch')
    course_path, declarations, target, members = membership(course_record, family_id)
    visibility_pin, definitions = None, {}
    if visibility_plan is not None:
        visibility = read(visibility_plan)
        require(visibility.get('schema_version') == 'reap.teacher-course.visibility.v1'
                and visibility.get('source_sha256') == sha(declarations.read_bytes())
                and visibility.get('course_record_sha256') == sha(course_path.read_bytes()),
                'visibility source/course identity mismatch')
        definitions = visibility.get('definitions')
        require(isinstance(definitions, dict) and all(name in definitions for name in members),
                'explicit definition visibility required for every selected member')
        # The shared wrapper checks names and limits here; Lean later checks
        # definition kinds, reachability and complete-proposition equivalence.
        for name in members:
            adapter.closed.prepare(environment_sha256=runtime['environment_sha256'],
                declarations=declarations.read_bytes(), closed_prop_name=name, polarity='prove',
                budget_steps=1, unfold_closed_prop=True, unfold_definitions=definitions[name])
        visibility_pin = pin(visibility_plan)
    premise_pin, hints = None, {}
    if premise_plan is not None:
        premises = read(premise_plan)
        require(premises.get('schema_version') == 'reap.teacher-course.premise-hints.v1'
                and premises.get('source_sha256') == sha(declarations.read_bytes())
                and premises.get('course_record_sha256') == sha(course_path.read_bytes())
                and premises.get('environment_sha256') == runtime['environment_sha256']
                and premises.get('cpu_image') == runtime['cpu_image'],
                'premise source/course/environment/image identity mismatch')
        review = premises.get('review', {})
        require(isinstance(review, dict) and isinstance(review.get('reviewer'), str)
                and review['reviewer'].strip() and review.get('basic_library_only') is True
                and review.get('excludes_complete_goal') is True,
                'explicit basic-lemma semantic review required')
        hints = premises.get('lessons')
        require(type(hints) is dict and set(hints) == set(members),
                'explicit premise list required for exactly the selected members')
        for name in members:
            # Validate before creating a family. Lean checks actual imported
            # modules/types in the same prefix used by preflight and search.
            adapter.closed.prepare(environment_sha256=runtime['environment_sha256'],
                declarations=declarations.read_bytes(), closed_prop_name=name, polarity='prove',
                budget_steps=1, unfold_closed_prop=True, unfold_definitions=definitions.get(name),
                premise_hints=hints[name])
        premise_pin = pin(premise_plan)
    budget_path = safe(budget_file)
    budgets = read(budget_path)
    require(budgets['schema_version'] in ('reap.proof-curriculum.budgets.v1', 'reap.frontiermath.budgets.v1',
                                         'reap.proof-curriculum.budgets.v2', 'reap.proof-curriculum.budgets.v3')
            and budgets['gamma'] == 0.99,
            'unknown budget contract')
    overrides = budgets.get('lesson_profiles', {})
    require(type(overrides) is dict and set(overrides) <= set(members),
            'lesson profiles must name selected declarations only')
    require(not overrides or budgets['schema_version'] == 'reap.proof-curriculum.budgets.v3',
            'lesson profiles require budget schema v3')
    require(isinstance(budget, str) and budget in budgets['profiles'], 'unknown default budget profile')
    profile_names = {name: overrides.get(name, budget) for name in members}
    require(all(isinstance(name, str) and name in budgets['profiles'] for name in profile_names.values()),
            'unknown lesson budget profile')
    fields = {'search_steps', 'num_samples', 'max_tokens', 'max_updates', 'min_checkpoint_interval'}
    if budgets['schema_version'] in ('reap.proof-curriculum.budgets.v2', 'reap.proof-curriculum.budgets.v3'):
        fields.add('max_nodes')
    # Validate every selected profile before creating any family files.
    for profile_name in set(profile_names.values()):
        chosen = budgets['profiles'][profile_name]
        require(type(chosen) is dict and set(chosen) == fields
                and all(type(v) is int and v > 0 for v in chosen.values()), 'explicit positive budget fields required')
        require(chosen['max_updates'] * chosen['min_checkpoint_interval'] <= chosen['search_steps'], 'inconsistent budget')
        require('max_nodes' not in chosen or chosen['search_steps'] <= chosen['max_nodes'] <= 1000000,
                'node budget must cover the step budget and be at most 1000000')
    raw = declarations.read_bytes()
    require(not re.search(r'\b(sorry|admit|axiom)\b', raw.decode('utf-8')), 'unproved declarations refused')
    seed, seed_pin = None, None
    if seed_path:
        seed = read(seed_path)
        adapter.policy._metadata(seed)
        seed_pin = pin(seed_path)
    store.mkdir(parents=True, exist_ok=True)
    family = store / family_id
    family.mkdir(exist_ok=False)  # one authoritative directory per family in this store
    sync_dir(store)
    with adapter.batch.batch_lock(family):
        (family / 'inputs').mkdir()
        (family / 'ops').mkdir()
        (family / 'outputs').mkdir()
        (family / 'outputs/.batch-intents').mkdir()
        require(session_nonce is None or re.fullmatch(r'[0-9a-f]{12}', session_nonce),
                'explicit matched-pair session nonce must be 12 lowercase hex characters')
        nonce = uuid.uuid4().hex[:12] if session_nonce is None else session_nonce
        lessons, files = [], []
        for index, declaration in enumerate(members):
            chosen = budgets['profiles'][profile_names[declaration]]
            prepared = adapter.closed.prepare(environment_sha256=runtime['environment_sha256'], declarations=raw,
                closed_prop_name=declaration, polarity='prove', budget_steps=chosen['search_steps'],
                num_samples=chosen['num_samples'], max_tokens=chosen['max_tokens'], unfold_closed_prop=True,
                max_nodes=chosen.get('max_nodes'),
                unfold_definitions=definitions.get(declaration),
                **({'premise_hints': hints[declaration]} if premise_pin else {}))
            input_file = f'lesson-{index + 1:02d}.lean'
            write_new(family / 'inputs' / input_file, prepared['execution_source'])
            files.append(pin(family / 'inputs' / input_file))
            lessons.append({'index': index, 'declaration': declaration, 'kind': 'target' if declaration == target else 'course',
                'session_id': f'course-{nonce}-{index + 1:02d}', 'experience_id': f'exp-{nonce}-{index + 1:02d}',
                'input_file': input_file, 'theorem': prepared['theorem'], 'problem': prepared['problem'],
                'execution_source_sha256': prepared['execution_source_sha256'],
                'preflight_source_sha256': prepared['preflight_source_sha256'], 'budget': dict(chosen)})
            if overrides:
                lessons[-1]['budget_name'] = profile_names[declaration]
            if visibility_pin:
                lessons[-1]['unfold_definitions'] = list(definitions[declaration])
            if premise_pin:
                lessons[-1]['premise_hints'] = hints[declaration]
        pins = None if seed is None else {'experience_id': seed['experience_id'],
            'experience_weights_sha256': seed['weights_sha256'],
            'experience_snapshot_sha256': seed['source']['snapshot_sha256']}
        if seed:
            require(seed['source']['session_id'] not in [l['session_id'] for l in lessons]
                    and seed['source']['theorem_id'] not in [l['execution_source_sha256'] for l in lessons],
                    'source and destination identity must differ')
        source_files = [pin(p) for p in adapter.source_files()]
        # Environment SHA is an operator declaration; manifests are additionally
        # bound when present. run also requires an explicit readiness approval.
        environment_files = [pin(Path(runtime['project_dir']) / name) for name in ('lean-toolchain', 'lake-manifest.json')
                             if (Path(runtime['project_dir']) / name).is_file()]
        plan = {'schema_version': SCHEMA, 'mode': 'online-search-visit-backup-v1', 'family_id': family_id,
                'store': str(store), 'family_dir': str(family), 'source_root': str(root),
                'relation_kind': RELATION, 'original_target': target, 'course_record': pin(course_path),
                'declarations': pin(declarations), 'budgets': pin(budget_path), 'budget_name': budget,
                'runtime_input': pin(runtime_path), 'runtime': runtime, 'environment_files': environment_files,
                'implementation_files': source_files, 'prepared_files': files, 'lessons': lessons,
                'initial_source': pins, 'initial_source_metadata': seed, 'initial_source_file': seed_pin,
                'known_exhaustion_policy': exhausted_policy, 'independent_tensor_audit_claimed': False}
        if session_nonce is not None:
            plan['session_nonce'] = {'value': nonce, 'purpose': 'explicit-fresh-matched-pair-api-seed-key'}
        if visibility_pin:
            plan['visibility_input'] = visibility_pin
        if premise_pin:
            plan['premise_input'] = premise_pin
        write_new(family / 'plan.json', plan)
        write_new(family / 'plan.sha256', (sha((family / 'plan.json').read_bytes()) + '\n').encode())
    return family


class Driver:
    def __init__(self, family, *, adapter=None):
        self.family = safe(family)
        self.plan = read(self.family / 'plan.json')
        self.plan_sha = sha((self.family / 'plan.json').read_bytes())
        require((self.family / 'plan.sha256').read_text().strip() == self.plan_sha, 'plan changed')
        require(self.plan['schema_version'] == SCHEMA and self.plan['family_dir'] == str(self.family),
                'wrong or relocated plan; do not reidentify an in-flight family')
        require(self.plan['mode'] == 'online-search-visit-backup-v1' and self.plan['relation_kind'] == RELATION,
                'online teacher course mode required')
        self.validate_files()
        self.adapter = adapter or Production(self.plan['source_root'])

    def validate_files(self):
        plan = self.plan
        require(sha((self.family / 'plan.json').read_bytes()) == self.plan_sha, 'plan changed during operation')
        items = [plan[k] for k in ('course_record', 'declarations', 'budgets', 'runtime_input')]
        items += plan['implementation_files'] + plan['environment_files'] + plan['prepared_files']
        if plan['initial_source_file']:
            items.append(plan['initial_source_file'])
        if plan.get('visibility_input'):
            items.append(plan['visibility_input'])
        if plan.get('premise_input'):
            items.append(plan['premise_input'])
        for item in items:
            validate_pin(item)

    def operation(self, lesson, phase):
        return self.family / 'ops' / f"{lesson['index']:02d}-{phase}"

    def completion(self, lesson, phase):
        directory = self.operation(lesson, phase)
        if not directory.exists():
            return None
        intent = read(directory / 'intent.json')
        require(intent['plan_sha256'] == self.plan_sha and intent['session_id'] == lesson['session_id']
                and intent['phase'] == phase, 'operation identity mismatch')
        require((directory / 'complete.json').is_file(),
                f"unresolved {phase}: {lesson['session_id']}; preserve intent, inspect read-only, never resubmit")
        complete = read(directory / 'complete.json')
        require(complete['intent_sha256'] == sha((directory / 'intent.json').read_bytes()), 'intent changed')
        for item in complete['evidence']:
            validate_pin(item)
        return complete['result']

    def perform(self, lesson, phase, payload, action, collect):
        directory = self.operation(lesson, phase)
        require(not directory.exists(), 'existing operation may never be reissued')
        self.validate_files()
        directory.mkdir()
        sync_dir(directory.parent)
        intent = {'schema_version': 'reap.teacher-course.intent.v1', 'plan_sha256': self.plan_sha,
                  'phase': phase, 'session_id': lesson['session_id'], 'mutation_retry_allowed': False, **payload}
        write_new(directory / 'intent.json', intent)
        # Includes BaseException/process death: absent complete always blocks.
        result = action(intent, directory)
        self.validate_files()
        evidence = [pin(p) for p in collect(directory)]
        write_new(directory / 'complete.json', {'intent_sha256': sha((directory / 'intent.json').read_bytes()),
                                               'result': result, 'evidence': evidence})
        return result

    @staticmethod
    def tree_files(path):
        safe(path)
        result = []
        for p in sorted(path.rglob('*')):
            safe(p)
            if p.is_file():
                result.append(p)
        return result

    def state(self):
        """Read only: no locks/files/network; never reconcile a missing ACK."""
        self.validate_files()
        source = self.plan['initial_source']
        rows, next_index, stopped = [], None, False
        previous_terminal = True
        success_mode = self.plan['runtime'].get('success_finalization') is True
        for lesson in self.plan['lessons']:
            row = {'index': lesson['index'], 'declaration': lesson['declaration'], 'session_id': lesson['session_id'],
                   'status': 'not_started', 'requested_source': None, 'inherited_source': None}
            phases = {phase: self.completion(lesson, phase)
                      for phase in ('preflight', 'online', 'verify', 'success-learn', 'seal', 'publish', 'retire')}
            if not success_mode:
                require(not phases['success-learn'] and not phases['seal'], 'terminal phases require explicit opt-in')
            require(not phases['success-learn'] or phases['verify'], 'terminal learning before independent verification')
            require(not phases['seal'] or phases['success-learn'], 'seal before successful terminal update')
            if phases['online']:
                require(previous_terminal and not stopped, 'later lesson started before predecessor completed')
                require(phases['preflight'] is not None, 'online without preflight')
                intent = read(self.operation(lesson, 'online') / 'intent.json')
                require(intent['source_pins'] == source, 'lesson inherited wrong predecessor')
                row.update(requested_source=source, inherited_source=source, status='searched')
                if phases['online']['root_verified']:
                    row['status'] = 'awaiting_independent_lean'
                    if (self.plan['runtime'].get('verification_mode', 'local_container') == 'external_receipt'
                            and not phases['verify']):
                        from verification_exchange import import_ready
                        if not import_ready(self.adapter, self.plan, lesson, self.family):
                            row['status'] = 'awaiting_external_verification'
                    if phases['verify']:
                        row['status'] = 'proved_zero_updates' if phases['online']['policy_version'] == 0 else 'proved_trained'
                        if success_mode:
                            row['status'] = 'awaiting_success_replay' if not phases['success-learn'] else 'proved_success_trained'
                            if phases['success-learn']:
                                require(phases['success-learn']['policy_version'] == phases['online']['policy_version'] + 1,
                                        'terminal update version did not accumulate on online updates')
                            if phases['publish']:
                                require(phases['seal'], 'publication before terminal seal')
                                source = phases['publish']['pins']
                                row['status'] = 'published'
                        elif phases['online']['policy_version'] == 0:
                            require(phases['publish'] is None, 'zero-update proof cannot publish a version')
                        elif phases['publish']:
                            source = phases['publish']['pins']
                            row['status'] = 'published'
                else:
                    require(not any(phases[k] for k in ('verify', 'success-learn', 'seal', 'publish')),
                            'exhausted lesson cannot verify/train/publish')
                    row['status'] = 'search_exhausted'
                ready = (not phases['online']['root_verified'] or (phases['verify'] is not None
                         and ((not success_mode and phases['online']['policy_version'] == 0) or phases['publish'] is not None)))
                require(not phases['retire'] or ready, 'retirement precedes required proof/publication gates')
                previous_terminal = bool(ready and phases['retire'])
                if previous_terminal:
                    row['retired'] = True
                    if not phases['online']['root_verified'] and self.plan['known_exhaustion_policy'] == 'stop':
                        stopped = True
            else:
                require(not any(phases[k] for k in ('verify', 'success-learn', 'seal', 'publish', 'retire')), 'phase without online')
                previous_terminal = False
            if next_index is None and not previous_terminal and not stopped:
                next_index = lesson['index']
            rows.append(row)
        done = next_index is None and not stopped
        proved = {'proved_zero_updates', 'proved_trained', 'proved_success_trained', 'awaiting_success_replay', 'published'}
        status = 'stopped_exhausted' if stopped else ('completed' if done else 'ready')
        if next_index is not None and rows[next_index]['status'] == 'awaiting_external_verification':
            status = 'awaiting_external_verification'
        if next_index is not None and rows[next_index]['status'] == 'awaiting_success_replay':
            status = 'awaiting_success_replay'
        return {'family_id': self.plan['family_id'], 'status': status,
                'next_index': next_index, 'confirmed_source': source, 'lessons': rows,
                'independently_proved_count': sum(row['status'] in proved for row in rows),
                'original_target_proved': any(row['declaration'] == self.plan['original_target']
                                              and row['status'] in proved for row in rows),
                'published_count': sum(row['status'] == 'published' for row in rows),
                'exhausted_count': sum(row['status'] == 'search_exhausted' for row in rows),
                'scope': 'local evidence and declared lineage; no independent tensor audit claim'}

    def approval(self, path):
        value = read(path)
        require(value.get('schema_version') == 'reap.teacher-course.runtime-approval.v1'
                and value.get('plan_sha256') == self.plan_sha and value.get('ready') is True
                and isinstance(value.get('operator'), str) and value['operator'].strip()
                and value.get('checks') == APPROVAL_CHECKS,
                'explicit plan-bound operator runtime readiness approval required; not a machine attestation')
        return pin(path)

    def preflight(self):
        with self.adapter.batch.batch_lock(self.family):
            self.state()
            for lesson in self.plan['lessons']:
                if self.completion(lesson, 'preflight'):
                    continue
                self.perform(lesson, 'preflight', {},
                    lambda intent, directory: self.adapter.preflight(self.plan, lesson, directory / 'compile'),
                    lambda directory: self.tree_files(directory / 'compile'))
            return self.state()

    def run(self, *, approval_path, authorize_remote=False, max_lessons=1):
        require(authorize_remote is True, 'run requires explicit --authorize-remote')
        require(type(max_lessons) is int and 1 <= max_lessons <= len(self.plan['lessons']), 'finite lesson bound required')
        approval = self.approval(approval_path)
        with self.adapter.batch.batch_lock(self.family):
            for _ in range(max_lessons):
                state = self.state()
                if state['status'] in ('completed', 'stopped_exhausted'):
                    return state
                lesson = self.plan['lessons'][state['next_index']]
                require(self.completion(lesson, 'preflight'), 'run preflight in the pinned CPU environment first')
                # Source for this lesson is captured BEFORE its own publication.
                source = state['lessons'][lesson['index']]['requested_source']
                if not self.completion(lesson, 'online'):
                    source = state['confirmed_source']

                    def execute(intent, directory):
                        validate_pin(approval)
                        entry, identity = self.adapter.batch_binding(self.plan, lesson, self.family, source)
                        binding = {'entry': entry, 'batch_identity_sha256': self.adapter.batch.digest(self.adapter.batch.encoded(identity))}
                        write_new(self.family / 'outputs/.batch-intents' / (lesson['session_id'] + '.json'), binding)
                        self.adapter.execute(self.plan, lesson, self.family, source)
                        record = self.adapter.checked_online(self.plan, lesson, self.family, source)
                        return {'root_verified': record['root_verified'], 'policy_version': record['policy_version'],
                                'status': record['status'], 'source_pins': source}

                    self.perform(lesson, 'online', {'source_pins': source, 'runtime_approval': approval}, execute,
                        lambda directory: self.tree_files(self.family / 'outputs' / lesson['session_id'])
                          + [self.family / 'outputs/.batch-intents' / (lesson['session_id'] + '.json')])
                online = self.completion(lesson, 'online')
                final_version = online['policy_version']
                if online['root_verified']:
                    if not self.completion(lesson, 'verify'):
                        if self.plan['runtime'].get('verification_mode', 'local_container') == 'external_receipt':
                            from verification_exchange import import_ready
                            if not import_ready(self.adapter, self.plan, lesson, self.family):
                                return self.state()  # no verify intent, no remote retry
                        self.perform(lesson, 'verify', {},
                            lambda intent, directory: self.adapter.verify(self.plan, lesson, self.family),
                            lambda directory: self.verification_evidence(lesson))
                    if self.plan['runtime'].get('success_finalization') is True:
                        from success_finalization import ready, checked_replay, receipt_rows
                        if not self.completion(lesson, 'success-learn'):
                            if not ready(self.adapter, self.plan, lesson, self.family):
                                return self.state()
                            event, prepared = checked_replay(self.adapter, self.plan, lesson, self.family)
                            def learn_success(intent, directory):
                                response = self.adapter.client(self.plan)._request('POST',
                                    f"/sessions/{lesson['session_id']}/learn/v1",
                                    {'expected_policy_version': event['policy_version'], 'event': event})
                                write_new(directory / 'response.json', response)
                                self.adapter.success.validate_success_receipt(response, event,
                                    expected_rows=receipt_rows(prepared),
                                    expected_training_config=self.plan['runtime']['initialization_contract']['search_config'])
                                return {'policy_version': response['policy_version'], 'terminal_updates': 1}
                            self.perform(lesson, 'success-learn', {'event': event}, learn_success,
                                lambda d: [d / 'response.json', *self.tree_files(self.family / 'success-replay-store' / event['dataset_sha256']),
                                           self.family / 'success-replay-imports' / (lesson['session_id'] + '.json')])
                        final_version = self.completion(lesson, 'success-learn')['policy_version']
                        if not self.completion(lesson, 'seal'):
                            def seal(intent, directory):
                                response = self.adapter.client(self.plan).snapshot(lesson['session_id'],
                                    'experience-candidate', for_experience=True)
                                write_new(directory / 'response.json', response)
                                require(response.get('source', {}).get('policy_version') == final_version,
                                        'terminal seal policy version mismatch')
                                return {'policy_version': final_version}
                            self.perform(lesson, 'seal', {'policy_version': final_version}, seal,
                                lambda d: [d / 'response.json'])
                    if final_version > 0 and not self.completion(lesson, 'publish'):
                        acceptance = self.adapter.acceptance(self.plan, lesson, self.family, source)

                        def publish(intent, directory):
                            response = self.adapter.publish(self.plan, intent)
                            write_new(directory / 'response.json', response)
                            pins = self.adapter.checked_publication(response, intent)
                            return {'pins': pins}

                        self.perform(lesson, 'publish', {'experience_id': lesson['experience_id'], 'acceptance': acceptance},
                                     publish, lambda directory: [directory / 'response.json'])
                if not self.completion(lesson, 'retire'):
                    def retire(intent, directory):
                        response = self.adapter.retire(self.plan, intent)
                        write_new(directory / 'response.json', response)
                        self.adapter.checked_retirement(response, intent)
                        return {'released': True, 'policy_version': final_version}

                    self.perform(lesson, 'retire', {'policy_version': final_version,
                        'snapshot': self.adapter.batch.RETIREMENT_SNAPSHOT}, retire, lambda directory: [directory / 'response.json'])
            return self.state()

    def inspect(self, *, index, phase, experience_store=None, allow_remote_read=False):
        """Read original operation only. Even a match NEVER unlocks or writes."""
        require(type(index) is int and 0 <= index < len(self.plan['lessons']), 'invalid lesson index')
        require(phase in ('online', 'success-learn', 'seal', 'publish', 'retire'), 'only original external operations can be inspected')
        lesson = self.plan['lessons'][index]
        intent = read(self.operation(lesson, phase) / 'intent.json')
        require(intent['plan_sha256'] == self.plan_sha and intent['session_id'] == lesson['session_id']
                and intent['phase'] == phase, 'inspect original intent identity mismatch')
        result = {'phase': phase, 'intent': intent, 'family_remains_blocked': True, 'mutation_retry_allowed': False}
        if phase == 'publish' and experience_store is not None:
            result['matching_release'] = self.adapter.inspect_publication(self.plan, safe(experience_store), intent)
        elif phase == 'retire' and allow_remote_read:
            result['matching_retirement'] = self.adapter.inspect_retirement(self.plan, intent)
        else:
            result['limitation'] = ('No create/learn/publication GET API. Review original bridge UUID/job and '
                'trusted stored receipts; publication can read an actual local ExperienceStore, retirement supports GET. '
                'Automatic adoption and same-tree crash resume are intentionally unavailable.')
        return result

    def export_verification(self, *, index, output):
        from verification_exchange import export_request
        with self.adapter.batch.batch_lock(self.family):
            self.state()
            return export_request(self, index, output)

    def verification_evidence(self, lesson):
        result = self.tree_files(self.family / 'proof-check' / lesson['session_id'])
        if self.plan['runtime'].get('verification_mode', 'local_container') == 'external_receipt':
            from verification_exchange import request_path, imported_path
            result += [request_path(self.family, lesson), imported_path(self.family, lesson)]
        return result

    def import_verification(self, *, index, export_dir):
        from verification_exchange import import_result
        with self.adapter.batch.batch_lock(self.family):
            self.state()
            return import_result(self, index, export_dir)

    def import_success_replay(self, *, index, bundle):
        from success_finalization import import_replay
        with self.adapter.batch.batch_lock(self.family):
            return import_replay(self, index, bundle)

    def replay_success(self, *, index, output, lean_project):
        from cpu_runtime.course_success_replay import export_course_success
        with self.adapter.batch.batch_lock(self.family):
            require(self.plan['runtime'].get('success_finalization') is True, 'successful finalization not enabled')
            require(self.state()['next_index'] == index, 'only current lesson may replay')
            lesson = self.plan['lessons'][index]
            require(self.completion(lesson, 'verify'), 'independent proof acceptance required before replay')
            _, checked = self.adapter.proof_functions(lesson)
            return export_course_success(family=self.family, lesson=lesson,
                cpu_image=self.plan['runtime']['cpu_image'], checked_proof=checked, output=safe(output),
                lean_project=safe(lean_project), replay_module=self.adapter.root / 'containers/cpu/verified-replay/VerifiedReplay.lean')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source-root', type=Path, required=True)
    prep.add_argument('--store', type=Path, required=True)
    prep.add_argument('--family-id', required=True)
    prep.add_argument('--course-record', type=Path, required=True)
    prep.add_argument('--budget-file', type=Path, required=True)
    prep.add_argument('--runtime', dest='runtime_path', type=Path, required=True)
    prep.add_argument('--budget', choices=('probe', 'main', 'deep'), default='main')
    prep.add_argument('--exhausted-policy', choices=('stop', 'continue'), required=True)
    prep.add_argument('--seed-metadata', dest='seed_path', type=Path)
    prep.add_argument('--visibility-plan', type=Path,
        help='Pinned per-declaration mathematical definition expansion; no teacher proofs.')
    prep.add_argument('--premise-plan', type=Path,
        help='Reviewed fixed basic-library hints, pinned to each lesson and CPU environment.')
    prep.add_argument('--session-nonce',
        help='Explicit 12-hex fresh nonce shared only by isolated matched-pair services.')
    verification = sub.add_parser('verify-export')
    verification.add_argument('--export-dir', type=Path, required=True)
    verification.add_argument('--source-root', type=Path, required=True)
    replay = sub.add_parser('replay-success-export')
    for name in ('source-root', 'export-dir', 'output', 'lean-project'):
        replay.add_argument('--' + name, type=Path, required=True)
    for name in ('status', 'preflight', 'run', 'approval-template', 'inspect', 'export-verification', 'import-verification',
                 'import-success-replay', 'replay-success'):
        child = sub.add_parser(name)
        child.add_argument('--family', type=Path, required=True)
        if name == 'run':
            child.add_argument('--approval', dest='approval_path', type=Path, required=True)
            child.add_argument('--authorize-remote', action='store_true')
            child.add_argument('--max-lessons', type=int, default=1)
        if name == 'inspect':
            child.add_argument('--index', type=int, required=True, help='zero-based lesson index')
            child.add_argument('--phase', choices=('online', 'success-learn', 'seal', 'publish', 'retire'), required=True)
            child.add_argument('--experience-store', type=Path)
            child.add_argument('--allow-remote-read', action='store_true')
        if name in ('export-verification', 'import-verification'):
            child.add_argument('--index', type=int, required=True)
            child.add_argument('--output' if name == 'export-verification' else '--export-dir', type=Path, required=True)
        if name in ('import-success-replay', 'replay-success'):
            child.add_argument('--index', type=int, required=True)
            child.add_argument('--bundle' if name == 'import-success-replay' else '--output', type=Path, required=True)
            if name == 'replay-success':
                child.add_argument('--lean-project', type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    try:
        if command == 'prepare':
            result = {'family': str(prepare(**args)), 'gpu_started': False}
        elif command == 'verify-export':
            from verification_exchange import verify_export
            result = verify_export(**args)
        elif command == 'replay-success-export':
            from success_finalization import replay_export
            result = replay_export(**args)
        else:
            driver = Driver(args.pop('family'))
            if command == 'approval-template':
                result = {'schema_version': 'reap.teacher-course.runtime-approval.v1', 'plan_sha256': driver.plan_sha,
                          'ready': False, 'operator': '', 'checks': APPROVAL_CHECKS,
                          'warning': 'Operator readiness assertion only; confirm environment/transport before changing ready.'}
            elif command == 'status':
                result = driver.state()
            else:
                result = getattr(driver, command.replace('-', '_'))(**args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (Blocked, OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'blocked', 'reason': str(exc), 'mutation_retry_allowed': False}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
