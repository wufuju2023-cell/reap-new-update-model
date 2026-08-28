"""Mixed envelope/transaction gates. Opaque fixtures do not prove tensor training."""
import base64
from copy import deepcopy
import errno
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_runtime.learner_release_store import LearnerReleaseStore, content_sha256
from gpu_runtime.mixed_learner_store import (
    MixedLearnerStore, MIXED_RETURN, SOURCE_REDUCTION, describe_mixed_dataset, sampler_catalog,
)
from gpu_runtime.mixed_objective import (
    OBJECTIVE_KIND, SOURCE_COUNTS, REPLAY_PROFILE, SFT_PROFILE, STATE_SCHEMA,
    SAMPLER_SCHEMA, TOKENIZATION, next_mixed_batch,
)
from gpu_runtime.verified_objective import VALUE_SEMANTICS
from tests import test_learner_release_store as old


def run_record():
    run = old.run_record()
    replay = {**run['catalog'][0], 'source': 'replay', 'source_tree_id': 'source-a.tree0'}
    human_source = {'source_kind': 'mathlib_sft', 'generator_events': 'none',
                    'repository': 'https://github.com/leanprover-community/mathlib4',
                    'commit': '1'*40, 'selection': {'declaration': 'Fixture.human'}}
    human = {'source': 'mathlib_sft', 'dataset_sha256': '9'*64, 'profile': SFT_PROFILE, 'rows': 2,
        'mathlib_source': human_source, 'mathlib_source_sha256': content_sha256(human_source),
        'source_info_sha256': '2'*64, 'original_receipt_sha256': '3'*64,
        'capture_receipt_sha256': '4'*64, 'replay_receipt_sha256': '5'*64, 'trace_sha256': '6'*64,
        'source_policy_version': None}
    config = run['contract'].pop('verified_config')
    config.update(objective=OBJECTIVE_KIND, value_semantics=VALUE_SEMANTICS, eos_token_id=7,
        value_loss='categorical_cross_entropy_exact_integer_class',
        policy_loss='mean_over_sampled_rows_of_joint_tactic_plus_one_EOS_negative_log_probability',
        tokenization=TOKENIZATION, kl_reduction='mean_over_sampled_rows_of_prefix_token_sum_current_to_frozen_base',
        max_sequence_tokens=4096, max_batch_samples=10, learning_rate=1e-4, value_learning_rate=1e-3,
        value_coefficient=1.0, kl_beta=0.1, max_grad_norm=1.0, max_post_update_kl=10.0,
        mixture={'source_counts': SOURCE_COUNTS, 'sample_weight': 0.1, 'sampling_unit': 'verified_action_row',
            'ratio_scope': 'every_complete_batch', 'source_profiles': {'replay': REPLAY_PROFILE, 'mathlib_sft': SFT_PROFILE},
            'source_losses': {'replay': ['policy', 'value'], 'mathlib_sft': ['policy', 'value']}},
        source_loss_report_reduction=SOURCE_REDUCTION)
    config['support']['return'] = MIXED_RETURN
    run['contract'].update(backend='mixed-replay', objective=OBJECTIVE_KIND, mixed_config=config)
    run.update(schema_version='reap.learner.run.v2', catalog=[replay, human])
    sampler = {'schema_version': SAMPLER_SCHEMA, 'kind': 'seeded_cyclic_9_1_v1',
        'objective': OBJECTIVE_KIND, 'seed': 17, 'max_distance': 8, 'catalog': sampler_catalog(run['catalog'])}
    state = {'schema_version': STATE_SCHEMA, 'config_sha256': content_sha256(sampler),
        'step': 0, 'replay_cursor': 0, 'mathlib_sft_cursor': 0}
    run['sampler'] = {'kind': sampler['kind'], 'config': sampler, 'config_sha256': content_sha256(sampler), 'initial_state': state}
    return run


def bind_receipt(values):
    logical, _, data, _ = values
    event, receipt = data['event'], data['runtime_receipt']
    data['event_sha256'] = content_sha256(event)
    data['runtime_receipt_sha256'] = content_sha256(receipt)
    logical['event_receipts'][event['event_id']] = {'digest': content_sha256(event), 'response': deepcopy(receipt)}
    logical['buffer_metadata']['events'][event['event_id']] = {'digest': content_sha256(event),
        'status': 'consumed', 'policy_version': event['policy_version']}


def checkpoint_inputs(run, pin, parent=None):
    step = 1 if parent is None else parent['manifest']['step']+1
    before = run['sampler']['initial_state'] if parent is None else parent['sampler_state']
    refs, after = next_mixed_batch(run['sampler']['config'], before)
    descriptors = {x['dataset_sha256']: x for x in run['catalog']}
    samples = []
    for ref in refs:
        item = descriptors[ref['dataset_sha256']]
        row = {**ref, 'return': -2, 'distance': 2, 'value_class': 1}
        if ref['source'] == 'replay':
            row.update(source_session_id=item['source_session_id'], source_tree_id=item['source_tree_id'],
                source_theorem_sha256=item['theorem_sha256'], source_policy_version=4, node_index=ref['row'])
        else:
            row.update(source_policy_version=None, mathlib_source=deepcopy(item['mathlib_source']))
        samples.append(row)
    event = {'kind': OBJECTIVE_KIND, 'event_id': 'mixed-'+str(step), 'session_id': run['backend_session_id'],
        'policy_version': step-1, 'samples': refs}
    losses = {source: {'policy_loss': count*0.2, 'value_loss': count*0.1, 'kl': 0.0,
        'loss': count*0.2+count*0.1, 'rows': count, 'weight_sum': count*0.1} for source, count in SOURCE_COUNTS.items()}
    detail = {key: sum(x[key] for x in losses.values()) for key in ('policy_loss', 'value_loss', 'kl', 'loss')}
    detail.update(objective=OBJECTIVE_KIND, training_config=deepcopy(run['contract']['mixed_config']),
        optimizer_steps=step, examples_seen=step*10, finite_loss=True, finite_gradients=True,
        finite_parameters=True, finite_optimizer_state=True, samples=samples,
        source_counts=deepcopy(SOURCE_COUNTS), sample_weight=0.1, source_losses=losses, source_loss_reduction=SOURCE_REDUCTION)
    receipt = {'event_id': event['event_id'], 'applied': True, 'idempotent': False, 'policy_version': step, 'detail': detail}
    logical = deepcopy(parent['logical_state']) if parent else {
        'schema_version': 'reap.gpu.session.v1', 'session_id': run['backend_session_id'], 'role': 'learner',
        'theorem_id': None, 'lineage': {}, 'completed': False, 'policy_version': 0, 'adapter_metadata': {},
        'value_metadata': {}, 'optimizer_metadata': {}, 'reference_metadata': {},
        'buffer_metadata': {'events': {}, 'pending_event_ids': [], 'consumed_event_ids': []},
        'event_receipts': {}, 'created_at': 1.0}
    logical['policy_version'] = step; logical['optimizer_metadata'] = {'kind': 'AdamW', 'steps': step}
    logical['adapter_metadata'] = {'kind': 'lora', 'objective': OBJECTIVE_KIND, 'rank': 2, 'examples_seen': step*10}
    logical['value_metadata'] = deepcopy(run['contract']['mixed_config'])
    logical['buffer_metadata']['consumed_event_ids'].append(event['event_id'])
    backend = {'schema_version': MixedLearnerStore.SNAPSHOT_SCHEMA, 'session_id': run['backend_session_id'],
        'encoding': 'torch-save-base64', 'mixed_config': deepcopy(run['contract']['mixed_config']),
        'payload': base64.b64encode(('opaque mixed private state '+str(step)).encode()).decode()}
    data = {'schema_version': MixedLearnerStore.DATA_SCHEMA, 'run_sha256': pin, 'step': step,
        'parent_receipt_sha256': None if parent is None else content_sha256(parent['data_receipt']),
        'event': event, 'runtime_receipt': receipt, 'batch_refs': deepcopy(refs),
        'sampler_before_sha256': content_sha256(before), 'sampler_after_sha256': content_sha256(after),
        'sampler_config_sha256': run['sampler']['config_sha256'], 'source_counts': deepcopy(SOURCE_COUNTS),
        'sample_weight': 0.1, 'catalog': deepcopy(run['catalog']), 'catalog_sha256': content_sha256(run['catalog'])}
    values = [logical, backend, data, after]; bind_receipt(values)
    return values


class MixedLearnerStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)/'registry'; self.store = MixedLearnerStore(self.root)
        self.run = run_record(); self.run_pin = self.store.create_run(self.run)

    def checkpoint(self, parent=None):
        previous = self.store.load_checkpoint(parent) if parent else None
        values = checkpoint_inputs(self.run, self.run_pin, previous)
        return self.store.create_checkpoint(self.run_pin, *values, parent_checkpoint_sha256=parent), values

    def weights(self):
        return {'contract': self.run['contract'], 'encoding': 'torch-save-base64', 'payload': base64.b64encode(b'opaque weights only').decode()}

    def assert_refused(self, values, parent=None):
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with self.assertRaises((ValueError, KeyError, TypeError)):
            self.store.create_checkpoint(self.run_pin, *values, parent_checkpoint_sha256=parent)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_two_steps_publication_restart_and_exact_schedule(self):
        first, _ = self.checkpoint(); release = self.store.publish(first, self.weights())
        before = {str(p): p.read_bytes() for p in self.root.rglob('*.json')}
        second, values = self.checkpoint(first)
        restarted = MixedLearnerStore(self.root)
        cp = restarted.load_checkpoint(second)
        self.assertEqual(cp['sampler_state'], values[3])
        self.assertEqual(cp['sampler_state']['replay_cursor'], 18)
        self.assertEqual(cp['sampler_state']['mathlib_sft_cursor'], 2)
        meta, weights = restarted.load_release(release)
        self.assertEqual(meta['acceptance'], {'kind': 'mixed-training', 'committed_step': 1})
        self.assertEqual(weights['contract']['objective'], OBJECTIVE_KIND)
        self.assertEqual(meta['source']['source_kind'], 'learner_checkpoint')
        self.assertNotIn('theorem_id', meta)
        self.assertEqual(restarted.publish(first, self.weights()), release)
        for path, raw in before.items(): self.assertEqual(Path(path).read_bytes(), raw)

    def test_old_and_mixed_schemas_cannot_cross_load_or_initialize(self):
        legacy = LearnerReleaseStore(self.root)
        old_pin = legacy.create_run(old.run_record())
        with self.assertRaises(ValueError): self.store.load_run(old_pin)
        with self.assertRaises(ValueError): legacy.load_run(self.run_pin)
        cp, _ = self.checkpoint(); release = self.store.publish(cp, self.weights())
        with self.assertRaises(ValueError): legacy.load_checkpoint(cp)
        with self.assertRaises(ValueError): legacy.load_release(release)
        changed = deepcopy(self.run); changed['initialization'] = {'kind': 'learner_release', 'release_sha256': release}
        with self.assertRaisesRegex(ValueError, 'base initialization only'): self.store.create_run(changed)

    def test_run_rejects_contract_sampler_and_typed_catalog_mutations(self):
        mutations = [lambda r: r.update(schema_version='reap.learner.run.v1'),
            lambda r: r['contract'].update(objective='verified_success_replay'),
            lambda r: r['contract']['mixed_config']['mixture'].update(sample_weight=1.0),
            lambda r: r['contract']['mixed_config']['support'].update(distance_min=True),
            lambda r: r['contract']['mixed_config']['mixture']['source_losses'].update(mathlib_sft=['policy']),
            lambda r: r['catalog'][1].update(source_policy_version=0),
            lambda r: r['catalog'][1].update(source_session_id='fake'),
            lambda r: r['catalog'][1].update(profile=REPLAY_PROFILE),
            lambda r: r['catalog'][1].update(mathlib_source_sha256='f'*64),
            lambda r: r['catalog'].pop(),
            lambda r: r['catalog'].append(deepcopy(r['catalog'][0])),
            lambda r: r['sampler']['config'].update(seed=1),
            lambda r: r['sampler']['initial_state'].update(step=1),
            lambda r: r['sampler']['initial_state'].update(replay_cursor=True)]
        for mutate in mutations:
            r = deepcopy(self.run); mutate(r)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): self.store.create_run(r)

    def test_internally_rehashed_wrong_sampler_catalog_or_support_rejected(self):
        for key in ('catalog', 'max_distance'):
            r = deepcopy(self.run); config = r['sampler']['config']
            if key == 'catalog': config['catalog']['replay'][0]['rows'] += 1
            else: config['max_distance'] += 1
            r['sampler']['config_sha256'] = content_sha256(config)
            r['sampler']['initial_state']['config_sha256'] = content_sha256(config)
            with self.subTest(key=key), self.assertRaises(ValueError): self.store.create_run(r)

    def test_checkpoint_rejects_reordered_or_wrong_source_batch_even_when_rehashed(self):
        for mutate in (lambda refs: refs.reverse(), lambda refs: refs[0].update(row=99),
                       lambda refs: refs[-1].update(source='replay'), lambda refs: refs.pop()):
            values = checkpoint_inputs(self.run, self.run_pin)
            mutate(values[2]['event']['samples']); values[2]['batch_refs'] = deepcopy(values[2]['event']['samples'])
            bind_receipt(values); self.assert_refused(values)

    def test_checkpoint_rejects_omitted_source_or_invented_human_actor_and_labels(self):
        mutations = [lambda row: row.update(source_policy_version=0), lambda row: row.update(source_session_id='fake'),
            lambda row: row.update(mathlib_source={'source_kind': 'mathlib_sft'}), lambda row: row.pop('source'),
            lambda row: row.update(return_value=-2), lambda row: row.update(distance=3), lambda row: row.update(value_class=True)]
        for mutate in mutations:
            values = checkpoint_inputs(self.run, self.run_pin); mutate(values[2]['runtime_receipt']['detail']['samples'][-1])
            bind_receipt(values); self.assert_refused(values)
        for field in ('source_session_id', 'source_tree_id', 'source_theorem_sha256'):
            values = checkpoint_inputs(self.run, self.run_pin)
            values[2]['runtime_receipt']['detail']['samples'][0][field] = 'wrong'
            bind_receipt(values); self.assert_refused(values)

    def test_checkpoint_rejects_cursors_config_ratio_and_source_loss_corruption(self):
        mutations = [lambda v: v[3].update(replay_cursor=10), lambda v: v[3].update(mathlib_sft_cursor=2),
            lambda v: v[2].update(sampler_config_sha256='f'*64), lambda v: v[2].update(sample_weight=0.2),
            lambda v: v[2].update(source_counts={'replay': 10, 'mathlib_sft': 0}),
            lambda v: v[2]['runtime_receipt']['detail']['source_losses']['mathlib_sft'].update(loss=10),
            lambda v: v[2]['runtime_receipt']['detail']['source_losses']['replay'].update(rows=True),
            lambda v: v[2]['runtime_receipt']['detail'].update(source_loss_reduction='mean_per_source')]
        for mutate in mutations:
            values = checkpoint_inputs(self.run, self.run_pin); mutate(values)
            values[2]['sampler_after_sha256'] = content_sha256(values[3]); bind_receipt(values)
            self.assert_refused(values)

    def test_fixed_catalog_cannot_append_or_swap_source_between_steps(self):
        first, _ = self.checkpoint(); parent = self.store.load_checkpoint(first)
        values = checkpoint_inputs(self.run, self.run_pin, parent)
        extra = deepcopy(values[2]['catalog'][0]); extra['dataset_sha256'] = '8'*64
        values[2]['catalog'].append(extra); values[2]['catalog_sha256'] = content_sha256(values[2]['catalog'])
        self.assert_refused(values, first)

    def test_common_logical_receipt_and_parent_chain_guards_remain(self):
        for mutate in (lambda v: v[0].update(role='actor'), lambda v: v[0].update(theorem_id='fake'),
                       lambda v: v[0].update(completed=True), lambda v: v[1].update(schema_version='reap.gpu.verified-replay-backend.v1'),
                       lambda v: v[0]['buffer_metadata'].update(pending_event_ids=['pending']),
                       lambda v: v[0].update(lineage={'model_release_sha256': 'a'*64}),
                       lambda v: v[0]['adapter_metadata'].update(examples_seen=9)):
            values = checkpoint_inputs(self.run, self.run_pin); mutate(values); self.assert_refused(values)
        first, _ = self.checkpoint(); parent = self.store.load_checkpoint(first)
        values = checkpoint_inputs(self.run, self.run_pin, parent); values[2]['parent_receipt_sha256'] = 'f'*64
        self.assert_refused(values, first)

    def test_corrupt_existing_checkpoint_and_wrong_weight_contract_not_repaired(self):
        cp, _ = self.checkpoint(); weights = self.weights(); weights['contract'] = old.run_record()['contract']
        with self.assertRaises(ValueError): self.store.publish(cp, weights)
        path = self.root/'checkpoints'/cp/'sampler_state.json'; raw = path.read_bytes(); path.write_bytes(raw+b' ')
        with self.assertRaises(ValueError): self.store.load_checkpoint(cp)
        self.assertEqual(path.read_bytes(), raw+b' ')

    def test_nfs_fallback_complete_mixed_chain_and_missing_commit_fail_closed(self):
        with patch('gpu_runtime.learner_release_store._publish_noreplace', side_effect=OSError(errno.EINVAL, 'unsupported')):
            cp, _ = self.checkpoint(); release = self.store.publish(cp, self.weights())
        self.assertTrue((self.root/'checkpoints'/cp/'.COMMITTED.json').is_file())
        self.assertEqual(self.store.load_release(release)[0]['acceptance']['kind'], 'mixed-training')
        commit = self.root/'checkpoints'/cp/'.COMMITTED.json'; commit.unlink()
        with self.assertRaises(ValueError): self.store.load_checkpoint(cp)
        self.assertFalse(commit.exists())

    def test_unknown_publication_failure_never_falls_back_or_advertises_checkpoint(self):
        with patch('gpu_runtime.learner_release_store._publish_noreplace', side_effect=OSError(errno.EIO, 'unknown')) as mutation, \
                patch.object(self.store, '_commit_portable') as fallback:
            with self.assertRaises(OSError): self.checkpoint()
            self.assertEqual(mutation.call_count, 1); fallback.assert_not_called()
        self.assertEqual([p for p in (self.root/'checkpoints').iterdir() if not p.name.startswith('.')], [])

    def test_nfs_missing_final_marker_never_becomes_legacy_or_repairs_on_retry(self):
        import gpu_runtime.learner_release_store as storage
        real_link = storage._link_noreplace
        def stop_before_commit(source, source_name, target, target_name):
            if target_name == '.COMMITTED.json':
                raise OSError(errno.EIO, 'interrupted before final marker')
            real_link(source, source_name, target, target_name)
        with patch.object(storage, '_publish_noreplace', side_effect=OSError(errno.EINVAL, 'unsupported')), \
                patch.object(storage, '_link_noreplace', side_effect=stop_before_commit):
            with self.assertRaises(OSError): self.checkpoint()
        directories = [p for p in (self.root/'checkpoints').iterdir() if not p.name.startswith('.')]
        self.assertEqual(len(directories), 1)
        target = directories[0]
        self.assertTrue((target/'.PORTABLE.json').exists())
        self.assertFalse((target/'.COMMITTED.json').exists())
        before = {p.name: p.read_bytes() for p in target.iterdir()}
        with self.assertRaises(ValueError): self.store.load_checkpoint(target.name)
        with self.assertRaises(ValueError): self.checkpoint()
        self.assertEqual(before, {p.name: p.read_bytes() for p in target.iterdir()})


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT/'.downloads/verified-trajectory-local-20260828/final-v3/03'
HUMAN = ROOT/'.downloads/mathlib-sft-local-20260828/final/intro'
GENERATED_PIN = '7975e11a24d74f326387e82fd4b8f9fc2489d1d570e44c17c5f8a03e1b9555ff'
HUMAN_PIN = '98c67e8f220e8f5a9227ce010a95792f838b10a9cee6f15a98f1eae695d788cb'


@unittest.skipUnless(GENERATED.is_dir() and HUMAN.is_dir(), 'optional pinned real Lean bundles absent')
class MixedDescriptorEvidenceTests(unittest.TestCase):
    def test_actual_strict_loaders_admit_two_typed_sources_without_new_lean_or_gpu(self):
        from types import SimpleNamespace
        from cpu_runtime.verified_dataset_store import BUNDLE_FILES as generated_files
        from cpu_runtime.mathlib_trajectory import BUNDLE_FILES as human_files
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = SimpleNamespace(dataset_root=root/'replay', mathlib_dataset_root=root/'human')
            before = {}
            for source, pin, target_root, files in ((GENERATED, GENERATED_PIN, backend.dataset_root, generated_files),
                                                  (HUMAN, HUMAN_PIN, backend.mathlib_dataset_root, human_files)):
                target = target_root/pin; target.mkdir(parents=True)
                for name in files:
                    raw = (source/name).read_bytes(); before[str(source/name)] = hashlib.sha256(raw).hexdigest()
                    (target/name).write_bytes(raw)
            generated = describe_mixed_dataset(backend, 'replay', GENERATED_PIN)
            human = describe_mixed_dataset(backend, 'mathlib_sft', HUMAN_PIN)
            MixedLearnerStore._validate_catalog([generated, human])
            self.assertEqual((generated['rows'], human['rows']), (3, 2))
            self.assertIsNone(human['source_policy_version'])
            self.assertNotIn('source_session_id', human)
            self.assertEqual(human['mathlib_source']['selection']['declaration'], 'ExistsUnique.intro₂')
            with self.assertRaises((ValueError, OSError)):
                describe_mixed_dataset(backend, 'mathlib_sft', GENERATED_PIN)
            with self.assertRaises(ValueError): describe_mixed_dataset(backend, [], HUMAN_PIN)
            for path, digest in before.items(): self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
