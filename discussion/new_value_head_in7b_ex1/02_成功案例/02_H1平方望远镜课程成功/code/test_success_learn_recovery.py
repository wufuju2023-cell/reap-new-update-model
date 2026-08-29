import copy
import unittest

from cpu_runtime.course_success_replay import validate_success_receipt as scalar_validate
from success_learn_recovery import validate_success_receipt


EVENT = {'kind': 'verified_success_discounted_v1', 'event_id': 'x.success',
         'session_id': 'x', 'tree_id': 'x.tree0', 'theorem_id': 'a' * 64,
         'policy_version': 2, 'dataset_sha256': 'b' * 64,
         'course_acceptance_sha256': 'c' * 64}
ROW = {'row': 0, 'node_index': 4, 'generation_sequence': 8, 'eval_sequence': 9,
       'source_policy_version': 2, 'return': -5, 'value_target': 0.96059601,
       'prompt_sha256': 'd' * 64, 'tactic_sha256': 'e' * 64}


def config(categorical=False):
    value = {'objective': EVENT['kind']}
    if categorical:
        value.update(head='categorical-64', categorical_value={'support': {
            'distance_min': 1, 'distance_max': 64}})
    return value


def receipt(categorical=False):
    sample = {**ROW, 'target_tokens': 3}
    if categorical:
        sample['value_target'] = 5.0
        sample['value_training'] = {'clipped_distance': 5.0, 'distance': 5.0,
            'loss_kind': 'two_hot_categorical_cross_entropy', 'lower_bin': 5,
            'lower_weight': 1.0, 'prediction': 4.5, 'saturated': False,
            'support_max': 64, 'support_min': 1, 'target': 5.0,
            'target_nonzero': [{'bin': 5, 'weight': 1.0}], 'upper_bin': 5,
            'upper_weight': 0.0}
    return {'event_id': EVENT['event_id'], 'policy_version': 3, 'applied': True,
        'idempotent': False, 'detail': {'objective': EVENT['kind'], 'source': EVENT,
        'optimizer_steps': 3, 'training_config': config(categorical),
        'online_update_consumed_by_later_generation': False, 'samples': [sample],
        **{k: True for k in ('finite_loss', 'finite_gradients', 'finite_parameters',
                             'finite_optimizer_state', 'base_parameters_frozen')},
        **{k: 0.1 for k in ('loss', 'policy_loss', 'kl', 'value_loss', 'grad_norm')},
        'parameter_diffs': {k: {'before_sha256': '0'*64, 'after_sha256': '1'*64,
                                'changed_tensors': 1}
                            for k in ('adapter', 'value_head', 'optimizer')}}}


class ReceiptTests(unittest.TestCase):
    def test_scalar_contract_is_unchanged(self):
        value = receipt(False)
        scalar_validate(value, EVENT, expected_rows=[ROW], expected_training_config=config(False))
        validate_success_receipt(value, EVENT, expected_rows=[ROW], expected_training_config=config(False))

    def test_valid_categorical_extension_is_strictly_admitted(self):
        validate_success_receipt(receipt(True), EVENT, expected_rows=[ROW], expected_training_config=config(True))

    def test_tampered_categorical_fields_are_rejected(self):
        changes = [('distance', 4.0), ('target', 4.0), ('prediction', 65.0),
                   ('lower_bin', 4), ('lower_weight', 0.5), ('saturated', True),
                   ('target_nonzero', [{'bin': 4, 'weight': 1.0}])]
        for field, value in changes:
            bad = copy.deepcopy(receipt(True)); bad['detail']['samples'][0]['value_training'][field] = value
            with self.subTest(field=field), self.assertRaises((ValueError, RuntimeError)):
                validate_success_receipt(bad, EVENT, expected_rows=[ROW], expected_training_config=config(True))

    def test_unexpected_extension_is_rejected_for_scalar(self):
        bad = receipt(False); bad['detail']['samples'][0]['value_training'] = {}
        with self.assertRaises(ValueError):
            validate_success_receipt(bad, EVENT, expected_rows=[ROW], expected_training_config=config(False))


if __name__ == '__main__':
    unittest.main()
