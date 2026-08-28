"""Generation/identity tests; real offline Lean acceptance has separate evidence."""
from copy import deepcopy
import unittest

from cpu_runtime.closed_problem import identity, prepare, digest
from cpu_runtime.target_variants import generate_variant, GENERATOR_VERSION, MAX_NAT_INSTANCE, MAX_DECLARATION_BYTES

DECLARATIONS = b'import ReapRuntime\ndef Fixture.target : Prop := forall n : Nat, n = 0\n'


class TargetVariantsTests(unittest.TestCase):
    def generate(self, **options):
        args = dict(environment_sha256='1'*64, declarations=DECLARATIONS,
                    target_name='Fixture.target', variant_name='Fixture.instance',
                    transform={'kind':'nat_forall_instance', 'value':3})
        args.update(options)
        return generate_variant(**args)

    def test_exact_rebuild_preserves_original_prefix_and_expr_derivation(self):
        first, second = self.generate(), self.generate()
        self.assertEqual(first, second)
        self.assertTrue(first['declarations'].startswith(DECLARATIONS+b'\n'))
        self.assertEqual(first['target_identity'], identity(environment_sha256='1'*64,
            declarations=DECLARATIONS, closed_prop_name='Fixture.target'))
        self.assertIn(b'body.instantiate1 (Lean.mkNatLit 3)', first['declarations'])
        self.assertIn(b'Lean.Meta.whnf (Lean.mkConst targetName)', first['declarations'])
        self.assertNotIn(b'def Fixture.instance : Prop := 3 = 0', first['declarations'])
        self.assertNotIn('relationship_verified', first)
        self.assertNotIn('proof_verified', first)
        relation = dict(schema_version='reap.target-variant-relation.v1', generator_version=GENERATOR_VERSION,
            target_problem_sha256=first['target_identity']['problem_sha256'],
            variant_problem_sha256=first['variant_identity']['problem_sha256'], transform=first['transform'])
        self.assertEqual(first['transformation_sha256'], digest(relation))

    def test_variant_works_as_closed_problem_input_for_both_polarities(self):
        generated = self.generate()
        for polarity in ('prove', 'disprove'):
            prepared = prepare(environment_sha256='1'*64, declarations=generated['declarations'],
                closed_prop_name='Fixture.instance', polarity=polarity, budget_steps=8)
            self.assertEqual(prepared['problem'], generated['variant_identity'])
            self.assertIn(b'target variant requires an exact closed Prop declaration', prepared['preflight_source'])
        self.assertIn(b'_root_.Not (_root_.Fixture.instance)', prepared['execution_source'])

    def test_target_bytes_name_environment_and_rule_change_identity(self):
        first = self.generate()
        cases = [dict(declarations=DECLARATIONS+b'-- another pinned source\n'),
                 dict(environment_sha256='2'*64), dict(target_name='Other.target'),
                 dict(variant_name='Fixture.other'), dict(transform={'kind':'nat_forall_instance','value':4})]
        for change in cases:
            with self.subTest(change=change):
                generated = self.generate(**change)
                self.assertNotEqual(first['variant_identity']['problem_sha256'], generated['variant_identity']['problem_sha256'])
                self.assertNotEqual(first['transformation_sha256'], generated['transformation_sha256'])
        self.assertEqual(first['target_identity'], self.generate(transform={'kind':'and_left'})['target_identity'])

    def test_and_projection_uses_checked_expr_arguments(self):
        for kind, index in (('and_left',0), ('and_right',1)):
            generated = self.generate(transform={'kind':kind})
            self.assertIn(b'normalized.isAppOfArity ``And 2', generated['declarations'])
            self.assertIn(f'normalized.getAppArgs[{index}]!'.encode(), generated['declarations'])
            self.assertNotIn(b'body.instantiate1', generated['declarations'])
        self.assertNotEqual(self.generate(transform={'kind':'and_left'})['transformation_sha256'],
                            self.generate(transform={'kind':'and_right'})['transformation_sha256'])

    def test_relation_witness_is_separate_from_search_declarations(self):
        for transform, proof in (({'kind':'nat_forall_instance','value':3}, 'exact @h 3'),
                                 ({'kind':'and_left'}, 'exact h.1'), ({'kind':'and_right'}, 'exact h.2')):
            result = self.generate(transform=transform)
            self.assertTrue(result['relation_source'].startswith(result['declarations']))
            self.assertNotIn(b'ReapTargetVariantRelation', result['declarations'])
            self.assertIn(b'_root_.Fixture.target \xe2\x86\x92 _root_.Fixture.instance', result['relation_source'])
            self.assertIn(proof.encode(), result['relation_source'])
            self.assertIn(('#print axioms '+result['relation_theorem']).encode(), result['relation_source'])
            self.assertEqual(result['relation_source_sha256'], digest(result['relation_source']))

    def test_strict_rule_and_bounded_nat(self):
        bad = [None, [], {}, {'kind':'or_left'}, {'kind':'and_left','value':0},
               {'kind':'nat_forall_instance'}, {'kind':'nat_forall_instance','value':0,'extra':1}]
        bad += [{'kind':'nat_forall_instance','value':v} for v in (-1, MAX_NAT_INSTANCE+1, True, 1.0, '1')]
        for transform in bad:
            with self.subTest(transform=transform), self.assertRaises(ValueError): self.generate(transform=transform)
        for value in (0, MAX_NAT_INSTANCE):
            self.assertEqual(self.generate(transform={'kind':'nat_forall_instance','value':value})['transform']['value'], value)

    def test_names_declarations_and_environment_reject_without_io(self):
        bad = [dict(variant_name='Fixture.target'),dict(target_name='target'), dict(variant_name='X.x\naxiom bad : False'),
               dict(environment_sha256='A'*64),dict(declarations='not bytes'),dict(declarations=b'import Mathlib\n'),
               dict(declarations=b'import ReapRuntime\n'+b' '*MAX_DECLARATION_BYTES),
               dict(declarations=DECLARATIONS+b'-- ReapCurriculumWrapper')]
        for options in bad:
            with self.subTest(options=repr(options)[:100]), self.assertRaises(ValueError): self.generate(**options)

    def test_returned_transform_does_not_alias_caller(self):
        transform = {'kind':'nat_forall_instance','value':3}
        generated = self.generate(transform=transform)
        before = deepcopy(generated)
        transform['value'] = 9
        self.assertEqual(generated, before)


if __name__ == '__main__':
    unittest.main()
