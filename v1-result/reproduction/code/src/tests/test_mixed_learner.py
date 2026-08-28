"""Tiny actual autograd/Adam checks of the mixed durable coordinator, not 7B."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_runtime import GpuRuntime
from gpu_runtime.learner import LearnerCoordinator, publish_checkpoint
from gpu_runtime.learner_release_store import content_sha256
from gpu_runtime.mixed_learner import MixedLearnerCoordinator
from gpu_runtime.mixed_learner_store import MixedLearnerStore
from gpu_runtime.mixed_objective import next_mixed_batch
from tests import test_mixed_backend as fixtures
from tests import test_mixed_learner_store as store_fixtures


class MixedLearnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.backend = fixtures.MixedBackendTests.backend(self)
        self.backend._test_dataset['tree_id'] = 'old-theorem.tree0'
        for index, row in enumerate(self.backend._test_dataset['rows']):
            row['node_index'] = index
        self.backend._test_sft['source'].update(source_kind='mathlib_sft', generator_events='none')
        self.backend.model.add_adapter('__bootstrap__', self.backend.lora_config)
        self.backend.model.adapters['__bootstrap__'].requires_grad_(False)
        self.backend._checkpoint_adapter_specs = lambda: {name: (tuple(t.shape), t.dtype)
            for name, t in self.backend._adapter_state_dict('__bootstrap__').items()}
        self.runtime = GpuRuntime(backend=self.backend, snapshot_root=self.root/'snapshots',
            learner_release_root=self.root/'store', max_resident_sessions=3)
        self.addCleanup(self.runtime.close)
        # Only admission evidence is synthetic; sampler/training/store are real.
        def descriptor(backend, source, pin):
            result = deepcopy(store_fixtures.run_record()['catalog'][0 if source == 'replay' else 1])
            dataset = backend._test_dataset if source == 'replay' else backend._test_sft
            result.update(dataset_sha256=pin, rows=len(dataset['rows']))
            if source == 'replay':
                result.update(source_session_id=dataset['session_id'], source_tree_id=dataset['tree_id'],
                    theorem_sha256=dataset['theorem_sha256'])
            else:
                result.update(mathlib_source=deepcopy(dataset['source']),
                    mathlib_source_sha256=content_sha256(dataset['source']))
            return result
        admission = patch('gpu_runtime.mixed_learner.describe_mixed_dataset', side_effect=descriptor)
        admission.start(); self.addCleanup(admission.stop)

    def coordinator(self, name='mixed', **kwargs):
        coordinator = MixedLearnerCoordinator(self.runtime, learner_id=name,
            replay_pins=[fixtures.fixtures.DIGEST], mathlib_sft_pins=[fixtures.SFT_PIN],
            sampler_seed=17, journal_root=self.root/(name+'-journal'), **kwargs)
        self.addCleanup(coordinator.close)
        return coordinator

    def fingerprint(self, sid):
        return self.runtime.actor.submit(lambda: self.backend.session_fingerprints(sid))

    def test_two_steps_restore_exact_schedule_and_private_release_actors(self):
        learner = self.coordinator()
        self.assertIsInstance(self.runtime.learner_releases, MixedLearnerStore)
        refs1, state1 = next_mixed_batch(learner.run['sampler']['config'], learner.sampler_state)
        first = learner.train_next(); release1 = learner.publish()
        checkpoint1 = self.runtime.learner_releases.load_checkpoint(first['checkpoint_sha256'])
        self.assertEqual(checkpoint1['data_receipt']['event']['samples'], refs1)
        self.assertEqual(learner.sampler_state, state1)
        original_release = deepcopy(self.runtime.learner_releases.load_release(release1['model_release_sha256']))
        weights1 = self.fingerprint('mixed')
        self.runtime.create_session('actor1', theorem_id='1'*64, model_release_sha256=release1['model_release_sha256'])
        actor1 = self.fingerprint('actor1')
        for key in ('adapter', 'value_head'):
            self.assertEqual(actor1[key], weights1[key])
        self.assertNotEqual(actor1['optimizer'], weights1['optimizer'])
        self.assertEqual(self.runtime.sessions.get('actor1').policy_version, 0)
        learner.close(); self.runtime.delete_session('mixed')
        restored = MixedLearnerCoordinator.restore(self.runtime, checkpoint_sha256=first['checkpoint_sha256'],
            journal_root=self.root/'restored')
        self.addCleanup(restored.close)
        self.assertEqual(weights1, self.fingerprint('mixed'))
        refs2, state2 = next_mixed_batch(restored.run['sampler']['config'], state1)
        second = restored.train_next(); release2 = restored.publish()
        cp2 = self.runtime.learner_releases.load_checkpoint(second['checkpoint_sha256'])
        self.assertEqual(cp2['data_receipt']['event']['samples'], refs2)
        self.assertEqual(restored.sampler_state, state2)
        self.assertEqual((state2['replay_cursor'], state2['mathlib_sft_cursor']), (18, 2))
        self.assertEqual(cp2['manifest']['parent_checkpoint_sha256'], first['checkpoint_sha256'])
        self.assertEqual(second['runtime_receipt']['detail']['examples_seen'], 20)
        self.assertEqual(actor1, self.fingerprint('actor1'))
        self.assertEqual(original_release, self.runtime.learner_releases.load_release(release1['model_release_sha256']))
        self.assertNotEqual(release1['model_release_sha256'], release2['model_release_sha256'])
        self.runtime.create_session('actor2', theorem_id='2'*64, model_release_sha256=release2['model_release_sha256'])
        for key in ('adapter', 'value_head'):
            self.assertEqual(self.fingerprint('actor2')[key], self.fingerprint('mixed')[key])
        with self.assertRaisesRegex(ValueError, 'fixed-release actor'):
            self.runtime.learn('actor1', expected_policy_version=0, event=fixtures.event())

    def test_unknown_checkpoint_blocks_sampler_and_reconstruction(self):
        learner = self.coordinator(); first = learner.train_next()
        committed = deepcopy(learner.sampler_state)
        with patch.object(learner.store, 'create_checkpoint', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                learner.train_next()
        self.assertEqual(committed, learner.sampler_state)
        self.assertEqual(self.runtime.sessions.get('mixed').policy_version, 2)
        after = self.fingerprint('mixed')
        with self.assertRaisesRegex(RuntimeError, 'blocked'):
            learner.train_next()
        self.assertEqual(after, self.fingerprint('mixed'))
        learner.close(); self.runtime.delete_session('mixed')
        with patch.object(self.runtime, 'create_session', wraps=self.runtime.create_session) as create:
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                MixedLearnerCoordinator.restore(self.runtime, checkpoint_sha256=first['checkpoint_sha256'],
                    journal_root=self.root/'cannot-retry')
            create.assert_not_called()

    def test_profiles_and_append_refused_before_mutation(self):
        with self.assertRaisesRegex(ValueError, 'profile mismatch'):
            LearnerCoordinator(self.runtime, learner_id='wrong', dataset_pins=[fixtures.fixtures.DIGEST],
                batch_size=2, journal_root=self.root/'wrong')
        self.assertNotIn('wrong', self.backend.sessions)
        learner = self.coordinator(); before = self.fingerprint('mixed')
        with self.assertRaisesRegex(ValueError, 'fixed catalog'):
            learner.train_next(append_dataset_pins=['8'*64])
        self.assertEqual(before, self.fingerprint('mixed'))
        self.assertEqual(list(learner.control.iterdir()), [learner.control/'created.json'])
        first = learner.train_next(); learner.close(); self.runtime.delete_session('mixed')
        with self.assertRaisesRegex(ValueError, 'profile mismatch'):
            LearnerCoordinator.restore(self.runtime, checkpoint_sha256=first['checkpoint_sha256'],
                journal_root=self.root/'wrong-profile-restore')
        self.assertNotIn('mixed', self.backend.sessions)
        release = publish_checkpoint(self.runtime, checkpoint_sha256=first['checkpoint_sha256'],
            journal_root=self.root/'offline-publication')
        self.assertEqual(release['acceptance']['kind'], 'mixed-training')

    def test_missing_or_wrong_source_fails_before_create(self):
        with patch('gpu_runtime.mixed_learner.describe_mixed_dataset', side_effect=ValueError('damaged dataset')):
            with self.assertRaisesRegex(ValueError, 'damaged dataset'):
                self.coordinator()
        self.assertEqual(self.backend.sessions, {})
        self.assertFalse((self.root/'mixed-journal').exists())

    def test_cli_explicit_mixed_and_rejects_inapplicable_roots_before_model_load(self):
        from gpu_runtime.server import build_parser, build_backend
        parser = build_parser()
        flags = ['--backend', 'mixed-replay', '--verified-dataset-root', 'generated',
            '--mathlib-dataset-root', 'human', '--verified-max-distance', '8']
        with patch('gpu_runtime.mixed_backend.MixedReplayBackend') as mixed:
            self.assertIs(build_backend(parser.parse_args(flags)), mixed.return_value)
            self.assertEqual(mixed.call_args.kwargs['mathlib_dataset_root'], Path('human'))
            mixed.reset_mock()
            invalid = [flags+['--gamma', '0.99'], flags[:4]+flags[6:],
                ['--backend', 'verified-replay', '--mathlib-dataset-root', 'human']]
            for argv in invalid:
                with self.subTest(argv=argv), self.assertRaises(ValueError):
                    build_backend(parser.parse_args(argv))
            mixed.assert_not_called()


if __name__ == '__main__':
    unittest.main()
