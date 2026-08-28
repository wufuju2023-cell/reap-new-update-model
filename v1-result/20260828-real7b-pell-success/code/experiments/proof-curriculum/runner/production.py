"""Narrow adapters over pinned Reap validators, no model loading or retries."""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys
from types import FunctionType

sys.dont_write_bytecode = True


def clone(function, **bindings):
    """A private binding view; never mutate an imported module's globals."""
    result = FunctionType(function.__code__, {**function.__globals__, **bindings},
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class Production:
    def __init__(self, source_root):
        self.root = Path(source_root).absolute()
        scripts = self.root / 'v1-result/reproduction/scripts'
        sys.path[:0] = [str(self.root), str(scripts)]
        self.online = importlib.import_module('cpu_runtime.online_ttt')
        self.batch = importlib.import_module('cpu_runtime.online_batch')
        self.closed = importlib.import_module('cpu_runtime.closed_problem')
        self.policy = importlib.import_module('cpu_runtime.experience_policy')
        self.proof = importlib.import_module('extract_online_proof')
        self.cross = importlib.import_module('cross_experience')
        self.success = importlib.import_module('cpu_runtime.course_success_replay')
        success_dependencies = tuple(importlib.import_module(name) for name in (
            'cpu_runtime.verified_trajectory', 'cpu_runtime.verified_dataset_store',
            'gpu_runtime.success_finalize_objective', 'gpu_runtime.search_objective', 'gpu_runtime.verified_objective'))
        for module in (self.online, self.batch, self.closed, self.policy, self.proof, self.cross, self.success, *success_dependencies):
            if not Path(module.__file__).absolute().is_relative_to(self.root):
                raise ValueError('another source tree is already imported; start a new Python process')
        required = {'min_checkpoint_interval', 'training_snapshot_interval', 'max_request_body_bytes', 'experience_candidate',
                    'experience_weights_sha256', 'experience_snapshot_sha256'}
        if not required <= set(inspect.signature(self.online.run_online).parameters):
            raise ValueError('online_ttt lacks required explicit interfaces; integrate final source first')

    def source_files(self):
        # Pin the import closure already loaded, not a historical source copy.
        result = {Path(__file__).absolute(), Path(__file__).with_name('course_driver.py').absolute(),
                  Path(__file__).with_name('verification_exchange.py').absolute()}
        result.update(self.root / 'gpu_runtime' / name
                      for name in ('experience_store.py', 'snapshot_store.py', 'identifiers.py', 'errors.py', '__init__.py'))
        result.add(Path(__file__).with_name('success_finalization.py').absolute())
        result.add(self.root / 'containers/cpu/verified-replay/VerifiedReplay.lean')
        result.update(self.root / 'gpu_runtime' / name for name in
                      ('search_backend.py', 'search_objective.py', 'verified_objective.py',
                       'success_finalize_backend.py', 'success_finalize_objective.py', 'runtime.py', 'server.py'))
        result.update(self.root / 'cpu_runtime' / name for name in
                      ('course_success_replay.py', 'verified_trajectory.py', 'verified_dataset_store.py'))
        for module in tuple(sys.modules.values()):
            name = getattr(module, '__file__', None)
            if name and Path(name).suffix == '.py' and Path(name).absolute().is_relative_to(self.root):
                result.add(Path(name).absolute())
        return sorted(result)

    def prepare_proof_function(self, lesson):
        # Existing checker only admits five old filenames. Substitute the EXACT
        # immutable course mapping, retaining every proof/image/axiom/file gate.
        prepare = clone(self.proof.prepare_proof,
                        THEOREMS={lesson['input_file']: lesson['theorem']})
        def portable_prepare(*args):
            proof, online, bindings = prepare(*args)
            # The original uses str(relative_path), whose separator depends on
            # the host OS. Canonical relative POSIX paths preserve every file
            # identity/hash and permit CPU verification across Windows/Linux.
            return proof, online, {key: {**item, 'path': item['path'].replace('\\', '/')}
                                   for key, item in bindings.items()}
        return portable_prepare

    def proof_functions(self, lesson):
        prepare = self.prepare_proof_function(lesson)
        return (clone(self.proof.verify, prepare_proof=prepare),
                clone(self.cross.checked_proof, prepare_proof=prepare))

    def preflight(self, plan, lesson, destination):
        prepared = self.closed.prepare(
            environment_sha256=plan['runtime']['environment_sha256'],
            declarations=Path(plan['declarations']['path']).read_bytes(),
            closed_prop_name=lesson['declaration'], polarity='prove',
            budget_steps=lesson['budget']['search_steps'],
            num_samples=lesson['budget']['num_samples'], max_tokens=lesson['budget']['max_tokens'],
            max_nodes=lesson['budget'].get('max_nodes'),
            unfold_closed_prop=True, unfold_definitions=lesson.get('unfold_definitions'),
            **({'premise_hints': lesson['premise_hints']} if 'premise_hints' in lesson else {}))
        return self.closed.compile_preflight(prepared, project_dir=Path(plan['runtime']['project_dir']),
                                            output_dir=destination, lean_bin=plan['runtime']['lean_bin'])

    def execute(self, plan, lesson, family, pins):
        runtime = plan['runtime']
        extra = {'defer_experience_seal': True} if runtime.get('success_finalization') is True else {}
        snapshot_schedule = ({'training_snapshot_interval': runtime['training_snapshot_interval']}
                             if 'training_snapshot_interval' in runtime else {})
        return self.online.run_online(
            session_id=lesson['session_id'], project_dir=Path(runtime['project_dir']),
            theorem_file=str(family / 'inputs' / lesson['input_file']), output_root=family / 'outputs',
            gpu_base_url=runtime['gpu_base_url'], gamma=0.99,
            max_updates=lesson['budget']['max_updates'],
            min_checkpoint_interval=lesson['budget']['min_checkpoint_interval'],
            max_request_body_bytes=runtime['max_request_body_bytes'],
            http_timeout=runtime['http_timeout_seconds'], barrier_timeout=runtime['barrier_timeout_seconds'],
            lean_bin=runtime['lean_bin'], experience_candidate=True,
            client=self.initializing_client(plan, lesson, family), **snapshot_schedule, **extra, **(pins or {}))

    def checked_contract(self, plan, created):
        contract = created.get('initialization_contract')
        self.policy.validate_contract(contract)
        self.batch.require(contract == plan['runtime']['initialization_contract']
            and contract['search_config']['gamma'] == 0.99
            and contract['base_sha256'] == plan['runtime']['base_sha256']
            and created.get('initialization_contract_sha256') == self.policy.digest(contract),
            'base/initialization contract mismatch')

    def initializing_client(self, plan, lesson, family):
        from cpu_runtime.http_clients import GpuHttpClient
        adapter = self

        class PinnedClient(GpuHttpClient):
            def create_session(self, session_id, **kwargs):
                # Server checks the hash BEFORE creating the session, and only
                # includes the contract in its response when this is requested.
                kwargs['expected_initialization_contract_sha256'] = adapter.policy.digest(
                    plan['runtime']['initialization_contract'])
                response = super().create_session(session_id, **kwargs)
                from course_driver import write_new
                write_new(family / 'outputs' / session_id / 'create-adapter-response.json', response)
                adapter.checked_contract(plan, response)  # before run_online can start Lean
                return response

        return PinnedClient(plan['runtime']['gpu_base_url'],
                            timeout_seconds=plan['runtime']['http_timeout_seconds'])

    def batch_binding(self, plan, lesson, family, pins):
        runtime = plan['runtime']
        entry = {'session_id': lesson['session_id'],
                 'theorem_file': str(family / 'inputs' / lesson['input_file']),
                 'theorem_sha256': lesson['execution_source_sha256'], **(pins or {})}
        identity = {'config': {'project_dir': runtime['project_dir'],
                               'gpu_base_url': runtime['gpu_base_url'], 'gamma': 0.99,
                               'max_updates': lesson['budget']['max_updates']}}
        return entry, identity

    def checked_online(self, plan, lesson, family, pins):
        entry, identity = self.batch_binding(plan, lesson, family, pins)
        path = family / 'outputs' / lesson['session_id'] / 'session.json'
        original_read = self.batch.read_json

        def candidate_view(file):
            value = original_read(file)
            if file == path:
                self.batch.require(value.get('experience_candidate') is True,
                                   'course must seal eligible online experience candidates')
                # The batch verifier disallows sealing because its own executor
                # doesn't request it. Only this single checked boolean differs.
                return {**value, 'experience_candidate': False}
            return value

        record = clone(self.batch.completed_record, read_json=candidate_view)(
            family / 'outputs', entry, identity)
        session = original_read(path)
        directory = path.parent
        report = original_read(directory / 'online-result.json')
        for saved in (session, report):
            self.batch.require(saved.get('defer_experience_seal', False) is plan['runtime'].get('success_finalization', False),
                               'deferred seal mode differs from course plan')
            for name, expected in (
                ('min_checkpoint_interval', lesson['budget']['min_checkpoint_interval']),
                ('max_request_body_bytes', plan['runtime']['max_request_body_bytes'])):
                default = 1 if name == 'min_checkpoint_interval' else 8192
                self.batch.require(saved.get(name, default) == expected, 'online schedule/body mismatch')
            if 'training_snapshot_interval' in plan['runtime']:
                self.batch.require(saved.get('training_snapshot_interval') == plan['runtime']['training_snapshot_interval'],
                                   'online training snapshot schedule mismatch')
            else:
                self.batch.require('training_snapshot_interval' not in saved,
                                   'unexpected periodic training snapshot schedule')
        created = original_read(directory / 'create-receipt.json')
        self.online.validate_created(created, lesson['session_id'], 0.99,
            theorem_id=lesson['execution_source_sha256'],
            experience_id=(pins or {}).get('experience_id'),
            weights_sha256=(pins or {}).get('experience_weights_sha256'),
            snapshot_sha256=(pins or {}).get('experience_snapshot_sha256'))
        self.batch.require(created.get('role', 'theorem') == 'theorem', 'mixed/actor/learner role forbidden')
        self.checked_contract(plan, created)
        for index, request in enumerate(sorted((directory / 'checkpoints').glob('learn-*.request.json'))):
            event = original_read(request)
            self.online.validate_learn_receipt(report['learn_receipts'][index], event, index)
        return record

    def verify(self, plan, lesson, family):
        verify, checked = self.proof_functions(lesson)
        if plan['runtime'].get('verification_mode', 'local_container') == 'external_receipt':
            from verification_exchange import check_imported
            check_imported(self, plan, lesson, family)
            return checked(family, lesson['session_id'], plan['runtime']['cpu_image'])[0]
        verify(family, lesson['session_id'], lesson['input_file'], lesson['theorem'], plan['runtime']['cpu_image'])
        accepted, _ = checked(family, lesson['session_id'], plan['runtime']['cpu_image'])
        return accepted

    def acceptance(self, plan, lesson, family, pins):
        _, checked = self.proof_functions(lesson)
        accepted, online = checked(family, lesson['session_id'], plan['runtime']['cpu_image'])
        require = self.cross.require
        updates = online.get('optimizer_updates')
        terminal = plan['runtime'].get('success_finalization') is True
        if terminal:
            from success_finalization import checked_replay, receipt_rows
            event, prepared = checked_replay(self, plan, lesson, family)
            directory = family / 'ops' / f"{lesson['index']:02d}-success-learn"
            self.success.validate_success_receipt(self.batch.read_json(directory / 'response.json'), event,
                expected_rows=receipt_rows(prepared),
                expected_training_config=plan['runtime']['initialization_contract']['search_config'])
            updates += 1
        require(terminal or (type(updates) is int and updates > 0 and online.get('status') == 'passed_execution'
                and online.get('online_update_consumed_by_later_generation') is True),
                'publication requires a committed update consumed by later same-tree generation')
        candidate_path = (family / 'ops' / f"{lesson['index']:02d}-seal" / 'response.json' if terminal else
                          family / 'outputs' / lesson['session_id'] / 'experience-candidate.json')
        candidate = self.batch.read_json(candidate_path)
        source = candidate.get('source', {})
        require(candidate.get('session_id') == lesson['session_id']
                and candidate.get('snapshot') == 'experience-candidate', 'wrong candidate')
        require(set(source) == {'session_id', 'theorem_id', 'policy_version', 'snapshot',
                                'snapshot_sha256', 'parent_experience_id'}, 'incomplete candidate source')
        require(source['session_id'] == lesson['session_id']
                and source['theorem_id'] == accepted['theorem_sha256'] == lesson['execution_source_sha256']
                and type(source['policy_version']) is int and source['policy_version'] == updates
                and online.get('policy_version') + (1 if terminal else 0) == updates and source['snapshot'] == 'experience-candidate'
                and source['parent_experience_id'] == (pins or {}).get('experience_id'), 'candidate/source mismatch')
        self.cross.digest(source['snapshot_sha256'])
        return {'kind': 'independent-lean', 'passed': True, 'completed': True, 'source': source,
                'evidence_sha256': self.cross.sha(
                    (family / 'proof-check' / lesson['session_id'] / 'accepted.json').read_bytes())}

    def client(self, plan):
        from cpu_runtime.http_clients import GpuHttpClient
        return GpuHttpClient(plan['runtime']['gpu_base_url'],
                             timeout_seconds=plan['runtime']['http_timeout_seconds'])

    def publish(self, plan, intent):
        source = intent['acceptance']['source']
        return self.client(plan).publish_experience(source['session_id'], source['snapshot'],
                                                    intent['experience_id'], intent['acceptance'])

    def checked_publication(self, response, intent):
        self.policy._metadata(response)
        for key in ('experience_id', 'acceptance'):
            self.cross.same(response[key], intent[key], 'publication differs from exact intent: ' + key)
        self.cross.same(response['source'], intent['acceptance']['source'], 'publication source differs')
        return {'experience_id': response['experience_id'],
                'experience_weights_sha256': response['weights_sha256'],
                'experience_snapshot_sha256': response['source']['snapshot_sha256']}

    def retire(self, plan, intent):
        return self.client(plan).retire_session(intent['session_id'], intent['snapshot'],
                                               expected_policy_version=intent['policy_version'])

    def checked_retirement(self, response, intent):
        self.batch.validate_retirement_receipt(response, intent)

    def inspect_retirement(self, plan, intent):
        response = self.client(plan).retirement_receipt(intent['session_id'])
        self.checked_retirement(response, intent)
        return response

    def inspect_publication(self, plan, store, intent):
        # No publication GET endpoint exists. Read the ACTUAL trusted local
        # ExperienceStore; load validates persisted manifest/weights hashes.
        # Does not load a base model or decode torch tensors.
        from gpu_runtime.experience_store import ExperienceStore
        self.policy.no_links(store)
        if not Path(store).is_dir():
            raise ValueError('existing local experience store required')
        response, weights = ExperienceStore(Path(store)).load(intent['experience_id'])
        self.policy.validate_contract(weights['contract'])
        self.cross.same(weights['contract'], plan['runtime']['initialization_contract'], 'stored release contract differs')
        self.checked_publication(response, intent)
        return response
