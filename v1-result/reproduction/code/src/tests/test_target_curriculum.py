"""Curriculum identity/publication mechanics; every Lean subprocess is mocked.

These tests do not establish Lean validity, relation validity, or proof success.
The actual generator is used to reconstruct all source bytes and identities.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cpu_runtime import target_curriculum as curriculum
from cpu_runtime import target_variants
from cpu_runtime.closed_problem import canonical, digest


DECLARATIONS = b'import ReapRuntime\ndef Test.target : Prop := forall n : Nat, n = n\n'
AXIOMS = ("'" + target_variants.RELATION_THEOREM + "' does not depend on any axioms\n").encode()


class TargetCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / 'project'
        self.project.mkdir()

    def plan(self, **kwargs):
        values = dict(environment_sha256='1' * 64, declarations=DECLARATIONS,
            target_name='Test.target', variants=[dict(problem_id='zero', variant_name='Test.zero',
                transform={'kind': 'nat_forall_instance', 'value': 0})])
        values.update(kwargs)
        return curriculum.prepare_curriculum(**values)

    @staticmethod
    def lean_success(command, **kwargs):
        # Deliberately only a protocol fixture, never a simulated Lean kernel.
        return SimpleNamespace(returncode=0,
            stdout=AXIOMS if Path(command[-1]).name == 'relation.lean' else b'', stderr=b'')

    def verified(self, name='verified', plan=None):
        with patch.object(curriculum.subprocess, 'run', side_effect=self.lean_success) as run:
            result = curriculum.verify_curriculum(plan or self.plan(), project_dir=self.project,
                output_dir=self.root / name)
        self.assertEqual(run.call_count, 3)
        return result

    def assert_failed_artifacts(self, output):
        self.assertFalse((output / 'curriculum.json').exists())
        failure = json.loads((output / 'failed.json').read_bytes())
        self.assertFalse(failure['verified'])
        self.assertFalse(failure['automatic_retry_allowed'])

    def test_plan_and_generated_members_reconstruct_without_io(self):
        plan = self.plan()
        self.assertEqual(canonical(plan), canonical(self.plan()))
        with patch.object(curriculum.subprocess, 'run') as run:
            members = curriculum._members(plan)
        run.assert_not_called()
        self.assertEqual(members[0]['declarations'], DECLARATIONS)
        self.assertTrue(members[1]['declarations'].startswith(DECLARATIONS))
        self.assertNotIn(target_variants.RELATION_THEOREM.encode(), members[1]['declarations'])
        self.assertIn(target_variants.RELATION_THEOREM.encode(), members[1]['relation_source'])
        self.assertEqual(members[1]['identity']['problem_sha256'], plan['variants'][0]['problem_sha256'])

    def test_changed_parent_source_parameter_or_generator_refused_before_output(self):
        variants = []
        for label in ('parent', 'source', 'parameter', 'generator', 'child', 'transform_pin'):
            plan = self.plan()
            if label == 'parent': plan['target']['problem_sha256'] = '2' * 64
            elif label == 'source': plan['target']['declarations_utf8'] += '-- changed\n'
            elif label == 'parameter': plan['variants'][0]['transform']['value'] = 1
            elif label == 'generator': plan['generator_sha256'] = '2' * 64
            elif label == 'child': plan['variants'][0]['problem_sha256'] = '2' * 64
            else: plan['variants'][0]['transformation_sha256'] = '2' * 64
            variants.append((label, plan))
        for label, plan in variants:
            with self.subTest(label=label), patch.object(curriculum.subprocess, 'run') as run:
                with self.assertRaises(ValueError):
                    curriculum.verify_curriculum(plan, project_dir=self.project, output_dir=self.root / label)
                run.assert_not_called()
                self.assertFalse((self.root / label).exists())

    def test_duplicate_transforms_ids_and_names_and_bad_parameters_rejected(self):
        first = self.plan()['variants'][0]
        spec = {k: first[k] for k in ('problem_id', 'variant_name', 'transform')}
        for changes in ({'problem_id': 'other', 'variant_name': 'Test.other'},
                        {'variant_name': 'Test.other', 'transform': {'kind': 'nat_forall_instance', 'value': 1}},
                        {'problem_id': 'other', 'transform': {'kind': 'nat_forall_instance', 'value': 1}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.plan(variants=[spec, {**deepcopy(spec), **changes}])
        for value in (True, -1, 1000001, 0.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.plan(variants=[{**spec, 'transform': {'kind': 'nat_forall_instance', 'value': value}}])
        with self.assertRaises(ValueError):
            self.plan(variants=[{**spec, 'transform': {'kind': 'and_left', 'value': 0}}])

    def test_verification_publishes_loadable_frozen_bundle_without_proof_claim(self):
        result = self.verified()
        self.assertFalse(result['target_proved'])
        self.assertFalse(result['variant_proved'])
        manifest = curriculum.load_curriculum(result['path'], expected_sha256=result['curriculum_sha256'])
        self.assertEqual(canonical(manifest), canonical(result['manifest']))
        problems = manifest['matchmaker_curriculum']['problems']
        self.assertEqual(problems[1]['parent_problem_sha256'], problems[0]['problem_sha256'])
        self.assertEqual(problems[1]['compilation_receipt_sha256'], digest(manifest['checks']['zero']))
        self.assertIsNone(manifest['checks']['target']['relation'])
        sources = list((self.root / 'verified').rglob('*.lean'))
        self.assertEqual(len(sources), 3)
        self.assertEqual(sum(b'ReapTargetVariantRelation.witness' in p.read_bytes() for p in sources), 1)

    def test_manifest_rejects_parent_family_source_or_receipt_relabelling(self):
        manifest = self.verified()['manifest']
        for label in ('parent', 'family', 'declarations', 'extra', 'command', 'returncode_bool', 'source_sha', 'missing_check'):
            bad = deepcopy(manifest)
            if label == 'parent': bad['matchmaker_curriculum']['problems'][1]['parent_problem_sha256'] = '2' * 64
            elif label == 'family': bad['matchmaker_curriculum']['problems'][1]['family_id'] = 'unrelated'
            elif label == 'declarations': bad['plan']['target']['declarations_utf8'] += '\n-- tampered'
            elif label == 'extra': bad['checks']['zero']['accepted'] = True
            elif label == 'command': bad['checks']['zero']['relation']['command'][0] = 'false-lean'
            elif label == 'returncode_bool': bad['checks']['zero']['relation']['returncode'] = False
            elif label == 'source_sha': bad['checks']['zero']['preflight']['source_sha256'] = '2' * 64
            else: del bad['checks']['zero']
            with self.subTest(label=label), self.assertRaises(ValueError):
                curriculum.validate_manifest(bad)

    def test_relation_axiom_report_must_be_unique_exact_and_approved(self):
        manifest = self.verified()['manifest']
        samples = (b'', AXIOMS + AXIOMS, AXIOMS.replace(b'witness', b'other'),
            AXIOMS.replace(b'does not depend on any axioms', b'depends on axioms: [sorryAx]'),
            AXIOMS.replace(b'does not depend on any axioms', b'depends on axioms: [Untrusted.fake]'))
        for sample in samples:
            bad = deepcopy(manifest)
            bad['checks']['zero']['relation']['stdout'] = sample.decode()
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                curriculum.validate_manifest(bad)

    def test_compile_failure_keeps_receipt_no_manifest_and_never_retries(self):
        output = self.root / 'failed'
        with patch.object(curriculum.subprocess, 'run',
                return_value=SimpleNamespace(returncode=1, stdout=b'bad statement', stderr=b'error')) as run:
            with self.assertRaises(ValueError):
                curriculum.verify_curriculum(self.plan(), project_dir=self.project, output_dir=output)
            self.assertEqual(run.call_count, 1)
            with self.assertRaises((FileExistsError, ValueError, OSError)):
                curriculum.verify_curriculum(self.plan(), project_dir=self.project, output_dir=output)
            self.assertEqual(run.call_count, 1)
        self.assert_failed_artifacts(output)
        self.assertEqual(json.loads((output / 'target/preflight-receipt.json').read_bytes())['returncode'], 1)
        self.assertEqual((output / 'target/preflight.stdout').read_bytes(), b'bad statement')

    def test_relation_failure_keeps_first_checks_but_never_publishes(self):
        output = self.root / 'relation-failed'
        def reject_relation(command, **kwargs):
            result = self.lean_success(command, **kwargs)
            if Path(command[-1]).name == 'relation.lean':
                result.stdout = b"'ReapTargetVariantRelation.witness' depends on axioms: [sorryAx]\n"
            return result
        with patch.object(curriculum.subprocess, 'run', side_effect=reject_relation) as run:
            with self.assertRaises(ValueError):
                curriculum.verify_curriculum(self.plan(), project_dir=self.project, output_dir=output)
        self.assertEqual(run.call_count, 3)
        self.assert_failed_artifacts(output)
        self.assertTrue((output / 'target/preflight-receipt.json').exists())
        self.assertIn(b'sorryAx', (output / 'zero/relation.stdout').read_bytes())

    def test_timeout_retains_stdout_stderr_intent_and_does_not_repeat(self):
        output = self.root / 'timeout'
        error = subprocess.TimeoutExpired(['lake'], 0.25, output=b'partial stdout', stderr=b'partial stderr')
        with patch.object(curriculum.subprocess, 'run', side_effect=error) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                curriculum.verify_curriculum(self.plan(), project_dir=self.project, output_dir=output, timeout=0.25)
        self.assertEqual(run.call_count, 1)
        self.assert_failed_artifacts(output)
        self.assertEqual((output / 'target/preflight.stdout').read_bytes(), b'partial stdout')
        self.assertEqual((output / 'target/preflight.stderr').read_bytes(), b'partial stderr')
        intent = json.loads((output / 'target/preflight-intent.json').read_bytes())
        self.assertFalse(intent['automatic_retry_allowed'])
        self.assertEqual(intent['timeout_seconds'], 0.25)
        self.assertFalse((output / 'target/preflight-receipt.json').exists())

    def test_invalid_timeout_rejected_before_output_or_process(self):
        for timeout in (True, 0, -1, float('nan'), float('inf')):
            with self.subTest(timeout=timeout), patch.object(curriculum.subprocess, 'run') as run:
                with self.assertRaises(ValueError):
                    curriculum.verify_curriculum(self.plan(), project_dir=self.project,
                        output_dir=self.root / 'invalid-timeout', timeout=timeout)
                run.assert_not_called()
                self.assertFalse((self.root / 'invalid-timeout').exists())

    def test_changed_current_compile_source_refused(self):
        output = self.root / 'changed-current'
        def corrupt(command, **kwargs):
            Path(command[-1]).write_bytes(b'-- changed while Lean ran\n')
            return self.lean_success(command, **kwargs)
        with patch.object(curriculum.subprocess, 'run', side_effect=corrupt) as run:
            with self.assertRaises(ValueError):
                curriculum.verify_curriculum(self.plan(), project_dir=self.project, output_dir=output)
        self.assertEqual(run.call_count, 1)
        self.assert_failed_artifacts(output)

    def test_earlier_frozen_source_rechecked_before_manifest_publication(self):
        output = self.root / 'changed-earlier'
        def corrupt_earlier(command, **kwargs):
            if Path(command[-1]).name == 'relation.lean':
                (output / 'target/preflight.lean').write_bytes(b'-- prior source changed\n')
            return self.lean_success(command, **kwargs)
        with patch.object(curriculum.subprocess, 'run', side_effect=corrupt_earlier) as run:
            with self.assertRaises(ValueError):
                curriculum.verify_curriculum(self.plan(), project_dir=self.project, output_dir=output)
        self.assertEqual(run.call_count, 3)
        self.assert_failed_artifacts(output)

    def test_frozen_files_rechecked_by_loader(self):
        result = self.verified()
        directory = Path(result['path'])
        for relative in ('plan.json', 'target/preflight.lean', 'zero/relation.lean',
                         'zero/relation-receipt.json', 'zero/relation.stdout', 'zero/relation.stderr'):
            path = directory / relative
            original = path.read_bytes()
            try:
                path.write_bytes(original + b'\n')
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    curriculum.load_curriculum(directory, expected_sha256=result['curriculum_sha256'])
            finally:
                path.write_bytes(original)
        self.assertEqual(curriculum.load_curriculum(directory, expected_sha256=result['curriculum_sha256']), result['manifest'])

    def test_frozen_compile_intent_rechecked_by_loader(self):
        result = self.verified()
        path = Path(result['path']) / 'zero/relation-intent.json'
        original = path.read_bytes()
        for label in ('command', 'timeout', 'retry', 'source', 'extra'):
            intent = json.loads(original)
            if label == 'command': intent['command'][-1] = 'unrelated.lean'
            elif label == 'timeout': intent['timeout_seconds'] = 1
            elif label == 'retry': intent['automatic_retry_allowed'] = 0
            elif label == 'source': intent['source_sha256'] = '2' * 64
            else: intent['accepted'] = True
            try:
                path.write_bytes(canonical(intent))
                with self.subTest(label=label), self.assertRaises(ValueError):
                    curriculum.load_curriculum(result['path'], expected_sha256=result['curriculum_sha256'])
            finally:
                path.write_bytes(original)

    def test_digest_and_canonical_manifest_rejected(self):
        result = self.verified()
        with self.assertRaises(ValueError):
            curriculum.load_curriculum(result['path'], expected_sha256='2' * 64)
        path = Path(result['path']) / 'curriculum.json'
        path.write_bytes(json.dumps(result['manifest'], indent=2).encode())
        with self.assertRaisesRegex(ValueError, 'canonical'):
            curriculum.load_curriculum(result['path'], expected_sha256=digest(path.read_bytes()))

    def test_postpublication_failure_is_unknown_and_loader_refuses_visible_manifest(self):
        output = self.root / 'postlink-failed'
        publish = curriculum.mm._publish
        def fail_after_link(*args, **kwargs):
            publish(*args, **kwargs)
            raise OSError('injected post-publication fsync uncertainty')
        with patch.object(curriculum.subprocess, 'run', side_effect=self.lean_success) as run, \
                patch.object(curriculum.mm, '_publish', side_effect=fail_after_link) as publishing:
            with self.assertRaises(OSError):
                curriculum.verify_curriculum(self.plan(), project_dir=self.project, output_dir=output)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(publishing.call_count, 1)
        self.assertTrue((output / 'curriculum.json').exists())
        self.assertTrue((output / 'failed.json').exists())
        with self.assertRaisesRegex(ValueError, 'failed or publication is unknown'):
            curriculum.load_curriculum(output, expected_sha256=digest((output / 'curriculum.json').read_bytes()))

    def test_prepare_member_wraps_complete_not_and_budgets_keep_problem_identity(self):
        manifest = self.verified()['manifest']
        small = curriculum.prepare_member(manifest, 'zero', polarity='disprove', budget_steps=2,
            num_samples=1, max_tokens=32)
        large = curriculum.prepare_member(manifest, 'zero', polarity='disprove', budget_steps=16,
            num_samples=2, max_tokens=128)
        self.assertEqual(small['problem'], large['problem'])
        self.assertEqual(small['attempted'], large['attempted'])
        self.assertNotEqual(small['execution_source_sha256'], large['execution_source_sha256'])
        self.assertIn(b'theorem ReapCurriculumWrapper.attempt : _root_.Not (_root_.Test.zero) := by', small['execution_source'])
        self.assertNotIn(target_variants.RELATION_THEOREM.encode(), small['execution_source'])
        self.assertIn(b'info.type == expected', small['execution_source'])
        for member in ('not-a-member', 'Test.zero'):
            with self.subTest(member=member), self.assertRaises(ValueError):
                curriculum.prepare_member(manifest, member, polarity='prove', budget_steps=2,
                    num_samples=1, max_tokens=32)


if __name__ == '__main__':
    unittest.main()
