"""Wrapper identity/rejection mechanics; actual Lean checks are separate."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cpu_runtime.closed_problem import prepare, compile_preflight, digest

DECLARATIONS = b'import ReapRuntime\ndef Test.target : Prop := forall n : Nat, n = n\n'


class ClosedProblemTests(unittest.TestCase):
    def prepared(self, **kwargs):
        return prepare(environment_sha256='1'*64, declarations=DECLARATIONS,
            closed_prop_name='Test.target', polarity='disprove', budget_steps=8, **kwargs)

    def test_closed_negation_and_budget_independent_problem_identity(self):
        negative = self.prepared()
        self.assertIn('theorem ReapCurriculumWrapper.attempt : _root_.Not (_root_.Test.target) := by', negative['execution_source'].decode())
        positive = prepare(environment_sha256='1'*64, declarations=DECLARATIONS,
            closed_prop_name='Test.target', polarity='prove', budget_steps=32)
        self.assertEqual(positive['problem'], negative['problem'])
        self.assertNotEqual(positive['attempted']['attempted_prop_sha256'], negative['attempted']['attempted_prop_sha256'])
        larger = prepare(environment_sha256='1'*64, declarations=DECLARATIONS,
            closed_prop_name='Test.target', polarity='disprove', budget_steps=32)
        self.assertEqual(negative['attempted'], larger['attempted'])
        self.assertNotEqual(negative['execution_source_sha256'], larger['execution_source_sha256'])

    def test_forged_local_negation_or_identity_fails_before_process(self):
        with tempfile.TemporaryDirectory() as directory, patch('cpu_runtime.closed_problem.subprocess.run') as run:
            root = Path(directory)
            bad = self.prepared()
            bad['execution_source'] = bad['execution_source'].replace(b'_root_.Not (_root_.Test.target)', b'(forall n : Nat, Not (n = n))')
            bad['execution_source_sha256'] = digest(bad['execution_source'])
            with self.assertRaisesRegex(ValueError, 'changed'):
                compile_preflight(bad, project_dir=root, output_dir=root/'rejected')
            self.assertFalse((root/'rejected').exists())
            run.assert_not_called()

    def test_compile_receipt_is_preflight_only_and_failed_result_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('cpu_runtime.closed_problem.subprocess.run', return_value=SimpleNamespace(returncode=0)):
                descriptor = compile_preflight(self.prepared(), project_dir=root, output_dir=root/'good')
            receipt = json.loads((root/'good/receipt.json').read_bytes())
            self.assertFalse(receipt['proof_verified'])
            self.assertEqual(descriptor['compilation_receipt_sha256'], digest(receipt))
            self.assertEqual(receipt['execution_source_sha256'], self.prepared()['execution_source_sha256'])
            with patch('cpu_runtime.closed_problem.subprocess.run', return_value=SimpleNamespace(returncode=1)):
                with self.assertRaisesRegex(ValueError, 'preflight failed'):
                    compile_preflight(self.prepared(), project_dir=root, output_dir=root/'bad')
            self.assertEqual(json.loads((root/'bad/receipt.json').read_bytes())['returncode'], 1)

    def test_names_markers_and_budgets_are_bounded(self):
        for name in ('target', 'Foo.target\naxiom bad : False', 'Foo.(target)'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                prepare(environment_sha256='1'*64, declarations=DECLARATIONS,
                    closed_prop_name=name, polarity='prove', budget_steps=8)
        for budget in (True, 0, 1000001):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                prepare(environment_sha256='1'*64, declarations=DECLARATIONS,
                    closed_prop_name='Test.target', polarity='prove', budget_steps=budget)
        with self.assertRaises(ValueError):
            prepare(environment_sha256='1'*64, declarations=DECLARATIONS+b'-- reapTrainingMCTS',
                closed_prop_name='Test.target', polarity='prove', budget_steps=8)

    def test_opt_in_unfold_preserves_default_bytes_identity_and_complete_not(self):
        # Golden digests independently computed from the frozen r2 wrapper.
        golden = {
            'prove': ('8578c575c32d8d388be9f73e62b26c0878913ad8e1e74af714d1cd4fc92a346d',
                      'a4c5d5c1b70b74b376030e9e1a45c12ba78d42e507997d54c2e13e73bfb09f21'),
            'disprove': ('a385b16fc7245c698da0055a1db10149990818977418144b00a32383251e7e66',
                         '47d25348f0c971806bbd0085e5ef8092d434174b8dedb9f3051f739a98417493')}
        for polarity in golden:
            args = dict(environment_sha256='1'*64, declarations=DECLARATIONS,
                closed_prop_name='Test.target', polarity=polarity, budget_steps=8)
            default = prepare(**args)
            self.assertEqual(default, prepare(**args, unfold_closed_prop=False))
            self.assertEqual((default['preflight_source_sha256'], default['execution_source_sha256']), golden[polarity])
            expanded = prepare(**args, unfold_closed_prop=True)
            self.assertEqual(default['problem'], expanded['problem'])
            self.assertEqual(default['attempted'], expanded['attempted'])
            self.assertEqual(default['preflight_source'], expanded['preflight_source'])
            self.assertEqual(expanded['execution_source'], default['execution_source'].replace(
                b'  reapTrainingMCTS', b'  unfold _root_.Test.target\n  reapTrainingMCTS'))
            if polarity == 'disprove':
                self.assertIn(b': _root_.Not (_root_.Test.target) := by', expanded['execution_source'])
            self.assertIn(b'info.type == expected', expanded['execution_source'])
        for value in (0, 1, 'true', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.prepared(unfold_closed_prop=value)

    def test_opt_in_preflight_rebuild_carries_flag_and_refuses_omission(self):
        prepared = self.prepared(unfold_closed_prop=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('cpu_runtime.closed_problem.subprocess.run', return_value=SimpleNamespace(returncode=0)) as run:
                descriptor = compile_preflight(prepared, project_dir=root, output_dir=root/'good')
                self.assertEqual(run.call_count, 1)
            self.assertEqual(descriptor['execution_source_sha256'], prepared['execution_source_sha256'])
            self.assertEqual((root/'good/attempt.lean').read_bytes(), prepared['execution_source'])
            self.assertFalse(json.loads((root/'good/receipt.json').read_bytes())['proof_verified'])
            bad = deepcopy(prepared)
            del bad['unfold_closed_prop']
            with patch('cpu_runtime.closed_problem.subprocess.run') as run:
                with self.assertRaisesRegex(ValueError, 'changed'):
                    compile_preflight(bad, project_dir=root, output_dir=root/'rejected')
                run.assert_not_called()
            self.assertFalse((root/'rejected').exists())


if __name__ == '__main__':
    unittest.main()
