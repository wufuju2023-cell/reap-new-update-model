"""Stdlib v3 envelope/append tests. Opaque tensor bytes do not prove training."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from gpu_runtime.continual_mixed_store import ContinualMixedStore, SAMPLER_KIND, sampling_transition
from gpu_runtime.learner_release_store import content_sha256
from gpu_runtime.mixed_learner_store import MixedLearnerStore
from tests import test_mixed_learner_store as fixtures


def run_record():
    run = fixtures.run_record()
    run['schema_version'] = ContinualMixedStore.RUN_SCHEMA
    run['backend_session_id'] = run['learner_id']
    run['scope'] = {'kind':'generalist'}
    run['sampler']['kind'] = SAMPLER_KIND
    return run


def inputs(run, pin, parent=None, additions=()):
    previous_catalog = run['catalog'] if parent is None else parent['data_receipt']['catalog']
    before = run['sampler']['initial_state'] if parent is None else parent['sampler_state']
    catalog = deepcopy(previous_catalog)+deepcopy(list(additions))
    config, transition, refs, after = sampling_transition(run, previous_catalog, catalog, before)
    projected = deepcopy(run)
    projected['catalog'] = catalog
    projected['sampler']['config'] = config
    projected['sampler']['config_sha256'] = content_sha256(config)
    if parent is None:
        projected['sampler']['initial_state'] = transition['effective_sampler_before']
        projected_parent = None
    else:
        projected_parent = deepcopy(parent)
        projected_parent['sampler_state'] = transition['effective_sampler_before']
    values = fixtures.checkpoint_inputs(projected, pin, projected_parent)
    data = values[2]
    data.update(schema_version=ContinualMixedStore.DATA_SCHEMA, sampler_config=config,
        sampler_transition=transition, sampler_before_sha256=content_sha256(before))
    assert data['batch_refs'] == refs and values[3] == after
    return values


class ContinualMixedStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)/'store'
        self.store = ContinualMixedStore(self.root)
        self.run = run_record(); self.pin = self.store.create_run(self.run)

    def files(self):
        return {str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def test_append_checkpoint_load_keeps_flat_prefix_and_actual_parent_hash(self):
        first = self.store.create_checkpoint(self.pin, *inputs(self.run, self.pin))
        cp1 = self.store.load_checkpoint(first)
        extra = {**self.run['catalog'][0], 'dataset_sha256':'8'*64}
        values = inputs(self.run, self.pin, cp1, [extra])
        second = self.store.create_checkpoint(self.pin, *values, parent_checkpoint_sha256=first)
        restored = ContinualMixedStore(self.root).load_checkpoint(second)
        self.assertEqual(restored['data_receipt'], values[2])
        self.assertEqual(restored['sampler_state'], values[3])
        self.assertEqual(values[2]['catalog'][:-1], cp1['data_receipt']['catalog'])
        self.assertEqual(values[2]['sampler_before_sha256'], content_sha256(cp1['sampler_state']))
        self.assertIn('8'*64, [r['dataset_sha256'] for r in values[2]['batch_refs']])

    def test_v2_read_only_schema_dispatch_never_retries_invalid_schema(self):
        old = MixedLearnerStore(self.root); run = fixtures.run_record(); pin = old.create_run(run)
        values = fixtures.checkpoint_inputs(run,pin)
        cp = old.create_checkpoint(pin,*values)
        weights = {'contract':run['contract'],'encoding':'torch-save-base64','payload':'b3BhcXVl'}
        release = old.publish(cp,weights)
        original = self.files()
        self.assertEqual(self.store.load_release(release),old.load_release(release))
        self.assertEqual(self.store.load_run(pin), run)
        for mutate in (lambda:self.store.create_run(run),lambda:self.store.create_checkpoint(pin,*values),
                       lambda:self.store.publish(cp,weights)):
            with self.assertRaises(ValueError):mutate()
            self.assertEqual(self.files(),original)
        bad = deepcopy(self.run);bad['schema_version']='reap.learner.run.v4'
        with self.assertRaises(ValueError):self.store.create_run(bad)

    def test_all_transition_fields_are_checked_before_checkpoint_writes(self):
        cp1pin = self.store.create_checkpoint(self.pin,*inputs(self.run,self.pin))
        parent = self.store.load_checkpoint(cp1pin)
        extra = {**self.run['catalog'][0], 'dataset_sha256':'8'*64}
        original = inputs(self.run,self.pin,parent,[extra]); before = self.files()
        mutations = [lambda x:x[2]['sampler_config'].update(seed=18),
            lambda x:x[2]['sampler_transition'].update(previous_config_sha256='0'*64),
            lambda x:x[2]['sampler_transition'].update(added_dataset_pins=[]),
            lambda x:x[2]['sampler_transition']['effective_sampler_before'].update(mathlib_sft_cursor=0),
            lambda x:x[2].update(sampler_before_sha256=x[2]['sampler_transition']['effective_sampler_before_sha256']),
            lambda x:x[2]['catalog'][1].update(rows=3)]
        for mutate in mutations:
            values=deepcopy(original);mutate(values)
            with self.assertRaises(ValueError):self.store.create_checkpoint(self.pin,*values,parent_checkpoint_sha256=cp1pin)
            self.assertEqual(self.files(),before)

    def test_source_contract_and_identity_rejected_before_new_run_write(self):
        cp = self.store.create_checkpoint(self.pin,*inputs(self.run,self.pin))
        weights={'contract':self.run['contract'],'encoding':'torch-save-base64','payload':'b3BhcXVl'}
        release=self.store.publish(cp,weights);before=self.files()
        bad=deepcopy(self.run);bad['initialization']={'kind':'learner_release','release_sha256':release}
        with self.assertRaisesRegex(ValueError,'identity'):self.store.create_run(bad)
        bad.update(learner_id='new-learner',backend_session_id='new-learner')
        bad['contract']['base_sha256']='0'*64
        bad['contract']['mixed_config']['base_tokenizer_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'contract differs'):self.store.create_run(bad)
        self.assertEqual(self.files(),before)


if __name__ == '__main__':
    unittest.main()
