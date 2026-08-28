"""Actual tiny Adam/parameter tests; admission fixtures are not real Lean proofs."""
from copy import deepcopy
import hashlib
from unittest.mock import patch
import unittest

from gpu_runtime import GpuRuntime
from gpu_runtime.continual_mixed_learner import ContinualMixedLearner
from gpu_runtime.continual_mixed_store import ContinualMixedStore, sampling_transition
from gpu_runtime.mixed_learner import MixedLearnerCoordinator
from gpu_runtime.mixed_learner_store import MixedLearnerStore
from gpu_runtime.learner_release_store import content_sha256
from gpu_runtime.mixed_objective import next_mixed_batch
from tests import test_mixed_learner as fixtures


class ContinualMixedTests(unittest.TestCase):
    def setUp(self):
        fixtures.MixedLearnerTests.setUp(self)
        from gpu_runtime.mixed_learner import describe_mixed_dataset
        patched = patch('gpu_runtime.continual_mixed_learner.describe_mixed_dataset', describe_mixed_dataset)
        patched.start(); self.addCleanup(patched.stop)

    fingerprint = fixtures.MixedLearnerTests.fingerprint

    def enable_v3(self):
        self.runtime.close()
        self.runtime = GpuRuntime(backend=self.backend, snapshot_root=self.root/'snapshots',
            learner_release_root=self.root/'store', learner_profile='continual-mixed-v3', max_resident_sessions=4)
        self.addCleanup(self.runtime.close)

    def learner(self, name='continuous', **options):
        learner = ContinualMixedLearner(self.runtime, learner_id=name,
            replay_pins=[fixtures.fixtures.fixtures.DIGEST], mathlib_sft_pins=[fixtures.fixtures.SFT_PIN],
            sampler_seed=17, journal_root=self.root/(name+'-journal'), **options)
        self.addCleanup(learner.close)
        return learner

    def test_v2_source_to_private_specialist_and_immutable_old_actor(self):
        source = fixtures.MixedLearnerTests.coordinator(self, name='source')
        checkpoint = source.train_next(); release = source.publish()['model_release_sha256']
        original = deepcopy(self.runtime.learner_releases.load_release(release))
        source_weights = self.fingerprint('source')
        source.close()
        # Keep the actual old session while the replacement actor owns the same
        # SessionStore; this is not a second concurrent owner of the backend.
        old_sessions = self.runtime.sessions
        self.runtime.close()
        self.runtime = GpuRuntime(backend=self.backend, sessions=old_sessions, snapshot_root=self.root/'snapshots',
            learner_release_root=self.root/'store', learner_profile='continual-mixed-v3', max_resident_sessions=4)
        self.addCleanup(self.runtime.close)
        self.runtime.create_session('old-actor', theorem_id='1'*64, model_release_sha256=release)
        actor_before = self.fingerprint('old-actor')
        specialist = self.learner('specialist', initial_model_release_sha256=release,
            scope={'kind': 'specialist', 'problem_sha256': '2'*64, 'curriculum_sha256': '3'*64})
        initialized = self.fingerprint('specialist')
        for key in ('adapter', 'value_head'):
            self.assertEqual(initialized[key], source_weights[key])
        self.assertNotEqual(initialized['optimizer'], source_weights['optimizer'])
        self.assertNotEqual(initialized['rng_counters'], source_weights['rng_counters'])
        state = self.runtime.sessions.get('specialist')
        self.assertEqual(state.policy_version, 0)
        self.assertEqual(state.buffer_metadata['events'], {})
        self.assertEqual(state.lineage['source']['checkpoint_sha256'], checkpoint['checkpoint_sha256'])
        first = specialist.train_next()
        cp = specialist.store.load_checkpoint(first['checkpoint_sha256'])
        self.assertIsNone(cp['manifest']['parent_checkpoint_sha256'])
        self.assertEqual(cp['manifest']['step'], 1)
        self.assertEqual(self.fingerprint('source'), source_weights)
        self.assertEqual(self.fingerprint('old-actor'), actor_before)
        self.assertEqual(self.runtime.learner_releases.load_release(release), original)
        self.assertEqual(MixedLearnerStore(self.root/'store').load_release(release), original)
        old_cp = self.runtime.learner_releases.load_checkpoint(checkpoint['checkpoint_sha256'])
        with self.assertRaisesRegex(ValueError, 'read-only'):
            self.runtime.learner_releases.publish(checkpoint['checkpoint_sha256'], original[1])
        with self.assertRaisesRegex(ValueError, 'read-only'):
            self.runtime.learner_releases.create_checkpoint(old_cp['manifest']['run_sha256'],
                *[old_cp[k] for k in ('logical_state','backend_state','data_receipt','sampler_state')])
        specialist.publish()
        mismatched = {**self.backend.experience_contract(), 'base_sha256': '0'*64}
        mismatched['mixed_config'] = {**mismatched['mixed_config'], 'base_tokenizer_sha256': '0'*64}
        with patch.object(self.backend, 'experience_contract', return_value=mismatched), \
                patch.object(self.runtime, 'create_session', wraps=self.runtime.create_session) as create:
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                self.learner('wrong-base', initial_model_release_sha256=release)
            create.assert_not_called()

    def test_append_actual_step_restore_and_following_batch_keep_both_cursors(self):
        self.enable_v3(); learner = self.learner()
        first = learner.train_next()
        cp1 = learner.store.load_checkpoint(first['checkpoint_sha256'])
        original_catalog = deepcopy(learner.catalog)
        before_state = deepcopy(learner.sampler_state)
        _, expected_human_after = next_mixed_batch(learner.run['sampler']['config'], before_state)
        new_pin = '8'*64
        second = learner.train_next(append_dataset_pins=[new_pin])
        cp2 = learner.store.load_checkpoint(second['checkpoint_sha256'])
        data = cp2['data_receipt']
        self.assertEqual(data['catalog'][:len(original_catalog)], original_catalog)
        self.assertEqual(data['catalog'][-1]['dataset_sha256'], new_pin)
        self.assertIn(new_pin, [r['dataset_sha256'] for r in data['batch_refs']])
        self.assertEqual(data['sampler_before_sha256'], content_sha256(before_state))
        effective = data['sampler_transition']['effective_sampler_before']
        self.assertEqual({k:v for k,v in effective.items() if k != 'config_sha256'},
                         {k:v for k,v in before_state.items() if k != 'config_sha256'})
        self.assertEqual(cp2['sampler_state']['mathlib_sft_cursor'], expected_human_after['mathlib_sft_cursor'])
        self.assertEqual(second['runtime_receipt']['detail']['optimizer_steps'], 2)
        self.assertEqual(second['runtime_receipt']['detail']['examples_seen'], 20)
        full_fingerprint = self.fingerprint('continuous')
        expected_refs, expected_state = next_mixed_batch(data['sampler_config'], cp2['sampler_state'])
        learner.close(); self.runtime.delete_session('continuous')
        restored = ContinualMixedLearner.restore(self.runtime, checkpoint_sha256=second['checkpoint_sha256'],
            journal_root=self.root/'restored')
        self.addCleanup(restored.close)
        self.assertEqual(full_fingerprint, self.fingerprint('continuous'))
        third = restored.train_next()
        cp3 = restored.store.load_checkpoint(third['checkpoint_sha256'])
        self.assertEqual(cp3['data_receipt']['batch_refs'], expected_refs)
        self.assertEqual(cp3['sampler_state'], expected_state)
        self.assertEqual(cp3['data_receipt']['sampler_config'], data['sampler_config'])
        self.assertEqual(cp3['data_receipt']['sampler_transition']['added_dataset_pins'], [])
        self.assertEqual(learner.store.load_checkpoint(first['checkpoint_sha256'])['data_receipt'], cp1['data_receipt'])

    def test_bad_append_and_missing_source_refused_before_mutation(self):
        self.enable_v3()
        with patch.object(self.runtime, 'create_session', wraps=self.runtime.create_session) as create:
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.learner('missing', initial_model_release_sha256='7'*64)
            create.assert_not_called()
        learner = self.learner(); before = self.fingerprint('continuous')
        with patch.object(self.runtime, 'learn', wraps=self.runtime.learn) as learn:
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                learner.train_next(append_dataset_pins=[fixtures.fixtures.fixtures.DIGEST])
            with patch('gpu_runtime.continual_mixed_learner.describe_mixed_dataset', side_effect=ValueError('damaged bundle')):
                with self.assertRaisesRegex(ValueError, 'damaged'):
                    learner.train_next(append_dataset_pins=['8'*64])
            learn.assert_not_called()
        self.assertEqual(before, self.fingerprint('continuous'))
        self.assertEqual([p.name for p in learner.control.iterdir()], ['created.json'])

    def test_transition_rejects_reorder_human_append_and_wrong_parent(self):
        self.enable_v3(); learner = self.learner()
        catalog = learner.catalog
        human = {**catalog[1], 'dataset_sha256': '7'*64}
        invalid = [list(reversed(catalog)), catalog+[human], [{**catalog[0], 'rows': 9}, catalog[1]]]
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                sampling_transition(learner.run, catalog, candidate, learner.sampler_state)
        with self.assertRaises(ValueError):
            sampling_transition(learner.run, catalog, catalog, {**learner.sampler_state, 'config_sha256':'0'*64})

    def test_checkpoint_unknown_leaves_old_catalog_and_blocks_reconstruction(self):
        self.enable_v3(); learner = self.learner(); first = learner.train_next()
        catalog, state = deepcopy(learner.catalog), deepcopy(learner.sampler_state)
        with patch.object(learner.store, 'create_checkpoint', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                learner.train_next(append_dataset_pins=['8'*64])
        self.assertEqual(learner.catalog, catalog)
        self.assertEqual(learner.sampler_state, state)
        after = self.fingerprint('continuous')
        with self.assertRaises(RuntimeError): learner.train_next()
        self.assertEqual(after, self.fingerprint('continuous'))
        learner.close(); self.runtime.delete_session('continuous')
        with patch.object(self.runtime, 'create_session', wraps=self.runtime.create_session) as create:
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                ContinualMixedLearner.restore(self.runtime, checkpoint_sha256=first['checkpoint_sha256'],
                    journal_root=self.root/'cannot-restore')
            create.assert_not_called()

    def test_opt_in_and_no_implicit_v2_run_conversion(self):
        with self.assertRaisesRegex(ValueError, 'profile mismatch'):
            self.learner()
        self.enable_v3()
        with self.assertRaisesRegex(ValueError, 'profile mismatch'):
            fixtures.MixedLearnerTests.coordinator(self, 'wrong')
        from tests.test_mixed_learner_store import run_record
        with self.assertRaisesRegex(ValueError, 'only creates v3'):
            self.runtime.learner_releases.create_run(run_record())
        from gpu_runtime.server import build_parser, build_backend
        parser = build_parser()
        with patch('gpu_runtime.mixed_backend.MixedReplayBackend') as load:
            for args in (['--learner-profile','continual-mixed-v3'],
                         ['--backend','mixed-replay','--learner-profile','continual-mixed-v3']):
                with self.assertRaises(ValueError): build_backend(parser.parse_args(args))
            load.assert_not_called()

    def test_tampered_transition_lineage_and_sft_receipts_rejected_without_writes(self):
        self.enable_v3(); learner = self.learner(); first = learner.train_next()
        second = learner.train_next(append_dataset_pins=['8'*64])
        cp = learner.store.load_checkpoint(second['checkpoint_sha256'])
        original = [cp[k] for k in ('logical_state', 'backend_state', 'data_receipt', 'sampler_state')]
        def files():
            return {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in learner.store.root.rglob('*') if p.is_file() and p.suffix != '.lock'}
        before = files()
        mutations = [
            lambda x: x[2]['sampler_transition']['effective_sampler_before'].update(replay_cursor=0),
            lambda x: x[2].update(sampler_before_sha256=x[2]['sampler_transition']['effective_sampler_before_sha256']),
            lambda x: x[2]['sampler_config'].update(seed=18),
            lambda x: x[2]['catalog'][0].update(rows=99),
            lambda x: x[0]['lineage'].update(model_release_sha256='0'*64),
            lambda x: x[2]['sampler_transition'].update(added_dataset_pins=[]),
        ]
        for mutation in mutations:
            value = deepcopy(original); mutation(value)
            with self.assertRaises(ValueError):
                learner.store.create_checkpoint(learner.run_sha, *value,
                    parent_checkpoint_sha256=first['checkpoint_sha256'])
            self.assertEqual(files(), before)


if __name__ == '__main__':
    unittest.main()
