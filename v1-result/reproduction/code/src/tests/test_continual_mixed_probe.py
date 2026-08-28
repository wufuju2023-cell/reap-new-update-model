"""CPU tiny autograd and orchestration gates, not real GPU/Lean acceptance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from containers.gpu import smoke_continual_mixed_learner as probe
from gpu_runtime import GpuRuntime
from tests import test_mixed_learner as fixtures


class ContinualProbeTests(unittest.TestCase):
    def setUp(self):
        fixtures.MixedLearnerTests.setUp(self)
        self.backend.max_post_update_kl = 7.0  # Explicit non-100 source contract.
        from gpu_runtime.mixed_learner import describe_mixed_dataset
        patched = patch('gpu_runtime.continual_mixed_learner.describe_mixed_dataset', describe_mixed_dataset)
        patched.start(); self.addCleanup(patched.stop)
        source = fixtures.MixedLearnerTests.coordinator(self, name='source')
        source.train_next()
        self.source = source.publish()
        self.weights = self.runtime.learner_releases.load_release(self.source['model_release_sha256'])[1]
        source.close(); self.runtime.delete_session('source'); self.runtime.close()
        self.runtime = GpuRuntime(backend=self.backend, snapshot_root=self.root/'new-snapshots',
            learner_release_root=self.root/'store', learner_profile='continual-mixed-v3', max_resident_sessions=3)
        self.addCleanup(self.runtime.close)
        self.old_pin = fixtures.fixtures.fixtures.DIGEST
        self.human_pin = fixtures.fixtures.SFT_PIN
        self.new_pin = '8'*64
        self.datasets = {'replay': {self.old_pin: self.backend._test_dataset, self.new_pin: self.backend._test_dataset},
                         'mathlib_sft': {self.human_pin: self.backend._test_sft}}
        self.plan = probe.plan_batches(self.datasets, [self.old_pin], [self.human_pin], self.new_pin,
                                      seed=17, max_distance=8)

    def run_audit(self):
        output = self.root/'probe'; output.mkdir()
        with patch.object(probe, 'base_fingerprint', return_value={'fixture': True}), \
                patch.object(self.backend, 'learn', wraps=self.backend.learn) as learns:
            report = probe.audit(self.runtime, self.backend, self.datasets, self.plan, self.source, self.weights,
                                 output, {'cpu_fixture': 'f'*64})
            self.assertEqual(learns.call_count, 2)
        return report, output

    def argv(self, output):
        return ['--model-path','fixture', '--expected-base-sha256',self.source['contract']['base_sha256'],
            '--source-release-root',str(self.root/'store'), '--initial-model-release-sha256',self.source['model_release_sha256'],
            '--replay-dataset-root',str(self.root/'replay'), '--replay-dataset-sha256',self.old_pin,
            '--mathlib-dataset-root',str(self.root/'human'), '--mathlib-dataset-sha256',self.human_pin,
            '--append-replay-dataset-sha256',self.new_pin, '--sampler-seed','17', '--output-dir',str(output)]

    def test_actual_tiny_inherit_restore_append_two_steps_and_old_source_immutable(self):
        old = probe.inventory(self.root/'store')
        report, output = self.run_audit()
        self.assertTrue(report['ok'], report['gates'])
        self.assertTrue(all(report['gates'].values()))
        self.assertEqual(report['run_scope'], {'kind':'generalist'})
        self.assertFalse(report['specialist_scope_tested'])
        self.assertEqual(report['sampler_after_second']['replay_cursor'], 18)
        self.assertEqual(report['sampler_after_second']['mathlib_sft_cursor'], 2)
        self.assertIn(self.new_pin, [r['dataset_sha256'] for r in report['second']['runtime_receipt']['detail']['samples']])
        self.assertEqual(probe.inventory(self.root/'store', old), old)
        self.assertEqual(report['first']['runtime_receipt']['detail']['training_config']['max_post_update_kl'], 7.0)
        self.assertEqual(report['append_transition']['added_dataset_pins'], [self.new_pin])
        for info in report['artifacts'].values():
            self.assertEqual(hashlib.sha256((output/info['file']).read_bytes()).hexdigest(), info['sha256'])
        changed = deepcopy(report['second']['runtime_receipt'])
        changed['detail']['kl_guard']['post_update_kl'] = 7.001
        with self.assertRaisesRegex(RuntimeError, 'KL guard'):
            probe.validate_step(changed, self.plan['refs2'], self.datasets, 2, self.backend._config())

    def test_bad_plan_rejects_before_session_or_learn(self):
        output = self.root/'probe'; output.mkdir()
        bad = deepcopy(self.plan); bad['refs2'][0]['dataset_sha256'] = '9'*64
        with patch.object(self.runtime, 'create_session', wraps=self.runtime.create_session) as create, \
                patch.object(self.backend, 'learn', wraps=self.backend.learn) as learn:
            with self.assertRaisesRegex(RuntimeError, 'sampling plan'):
                probe.audit(self.runtime, self.backend, self.datasets, bad, self.source, self.weights, output, {})
            create.assert_not_called(); learn.assert_not_called()

    def test_new_pin_absent_from_second_batch_rejected_in_preflight(self):
        datasets = deepcopy(self.datasets)
        datasets['replay'][self.old_pin]['rows'] *= 100
        # Fixed synthetic catalog/seed makes the nine-row second batch remain
        # within the 200 old rows. No search or seed tuning occurs at runtime.
        with self.assertRaisesRegex(RuntimeError, 'absent from planned second batch'):
            probe.plan_batches(datasets, [self.old_pin], [self.human_pin], self.new_pin,
                               seed=17, max_distance=8)

    def test_failed_cp1_restore_never_repeats_first_or_starts_second(self):
        output = self.root/'probe'; output.mkdir()
        with patch.object(probe, 'base_fingerprint', return_value={'fixture': True}), \
                patch.object(self.backend, 'learn', wraps=self.backend.learn) as learns, \
                patch.object(probe.ContinualMixedLearner, 'restore', side_effect=RuntimeError('restore failed')):
            with self.assertRaisesRegex(RuntimeError, 'restore failed'):
                probe.audit(self.runtime, self.backend, self.datasets, self.plan, self.source, self.weights, output, {})
            self.assertEqual(learns.call_count, 1)
        self.assertTrue((output/'step1.json').is_file())
        self.assertFalse((output/'training-continuation.json').exists())

    def test_options_derive_complete_mutable_settings_without_threshold_override(self):
        options = probe.backend_options(self.source)
        self.assertEqual(options['max_post_update_kl'], 7.0)
        self.assertEqual(options['max_distance'], 8)
        self.assertEqual(options['lora_rank'], self.backend.lora_config.r)
        for maximum in (None, 0, float('nan'), True):
            source = deepcopy(self.source); source['contract']['mixed_config']['max_post_update_kl'] = maximum
            with self.assertRaises(RuntimeError): probe.backend_options(source)
        self.assertNotIn('--max-post-update-kl', probe.build_parser().format_help())

    def test_cpu_refusal_cannot_allocate_backend_or_learn(self):
        output = self.root/'cpu-refusal'
        with patch.object(probe, 'prepare_inputs', return_value=(self.datasets, {}, self.plan)), \
                patch.object(probe.mixed, 'require_amd', side_effect=RuntimeError('actual AMD GPU required')), \
                patch.object(probe, 'MixedReplayBackend') as load:
            self.assertEqual(probe.main(self.argv(output)), 1)
            load.assert_not_called()
        report = json.loads((output/'report.json').read_bytes())
        self.assertFalse(report['ok']); self.assertFalse(report['real_7B_GPU_gate_passed'])

    def test_main_late_integrity_failure_clears_pass(self):
        output = self.root/'late-failure'
        # The actual GPU allocation/device gate is separately required by main;
        # this orchestration test injects only an explicit fake loaded device.
        from unittest.mock import MagicMock
        backend = MagicMock(hidden_size=3584)
        backend.device.type = 'cuda'; backend.model.parameters.return_value = []
        backend.experience_contract.return_value = self.source['contract']
        fake = dict(ok=True, second={'checkpoint_sha256':'e'*64}, release2={'model_release_sha256':'f'*64},
                    sampler_after_second=self.plan['after2'])
        with patch.object(probe, 'prepare_inputs', side_effect=[(self.datasets, {}, self.plan), RuntimeError('late changed inputs')]), \
                patch.object(probe.mixed, 'require_amd'), patch.object(probe, 'MixedReplayBackend', return_value=backend), \
                patch.object(probe, 'GpuRuntime'), patch.object(probe, 'audit', return_value=fake):
            self.assertEqual(probe.main(self.argv(output)), 1)
        report = json.loads((output/'report.json').read_bytes())
        self.assertFalse(report['ok']); self.assertFalse(report['real_7B_GPU_gate_passed'])
        self.assertIn('late changed', report['error']['message'])
        self.assertFalse(json.loads((output/'final-continuation.json').read_bytes())['stage_passed'])

    def test_new_dataset_lineage_must_be_initial_release_and_no_training(self):
        args = probe.build_parser().parse_args(self.argv(self.root/'lineage'))
        good = dict(model_release_sha256=self.source['model_release_sha256'], role='actor', policy_version=0,
                    training_enabled=False, lineage={'model_release_sha256':self.source['model_release_sha256']})
        for session in (good, {**good, 'training_enabled':True}, {**good, 'policy_version':False},
                        {**good, 'model_release_sha256':'1'*64}, {**good, 'lineage':{'model_release_sha256':'2'*64}}):
            with patch.object(probe.mixed, 'load_inputs', return_value=(self.datasets, {}, {}, {})), \
                    patch.object(probe.safe, 'read_bundle', return_value={'session.json':json.dumps(session).encode()}):
                if session is good:
                    self.assertEqual(probe.prepare_inputs(args, self.source)[2], self.plan)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'lineage'): probe.prepare_inputs(args, self.source)


class ContinualProbeFinalizationTests(unittest.TestCase):
    def test_final_continuation_failure_clears_pass_before_atomic_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); original = probe.save
            report = dict(ok=True, real_7B_GPU_gate_passed=True, second={'checkpoint_sha256':'a'*64},
                          release2={'model_release_sha256':'b'*64}, sampler_after_second={})
            def save(path, name, value):
                if name == 'final-continuation.json': raise OSError('final disk failure')
                return original(path, name, value)
            with patch.object(probe, 'save', side_effect=save):
                self.assertEqual(probe.finalize(root, report), 1)
            saved = json.loads((root/'report.json').read_bytes())
            self.assertFalse(saved['ok']); self.assertFalse(saved['real_7B_GPU_gate_passed'])
            self.assertIn('finalization_error', saved)

    def test_unknown_report_publication_never_returns_success_or_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); original = probe.save
            attempts = []
            def save(path, name, value):
                attempts.append(name)
                if name == 'report.json':
                    original(path, name, value)
                    raise OSError('post-publication fsync unknown')
                return original(path, name, value)
            with patch.object(probe, 'save', side_effect=save):
                with self.assertRaisesRegex(OSError, 'unknown'):
                    probe.finalize(root, dict(ok=True, real_7B_GPU_gate_passed=True))
            self.assertEqual(attempts.count('report.json'), 1)
            worker = json.loads((root/'worker-result.json').read_bytes())
            self.assertTrue(worker['report_publication_pending'])
            self.assertTrue(worker['external_process_exit_still_required'])

    def test_atomic_artifacts_never_replace_and_source_inventory_detects_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); receipt = probe.save(root, 'sample.json', {'old':True})
            original = (root/'sample.json').read_bytes()
            with self.assertRaises(FileExistsError): probe.save(root, 'sample.json', {'old':False})
            self.assertEqual((root/'sample.json').read_bytes(), original)
            before = probe.inventory(root)
            (root/'sample.json').write_bytes(b'changed')
            self.assertNotEqual(probe.inventory(root, before), before)
            self.assertEqual(hashlib.sha256(original).hexdigest(), receipt['sha256'])


if __name__ == '__main__':
    unittest.main()
