"""Stdlib publication protocol tests; opaque CAS payloads are not GPU acceptance."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gpu_runtime import release_head as rh
from gpu_runtime.learner import _write
from gpu_runtime.learner_release_store import LearnerReleaseStore, content_sha256
from tests.test_learner_release_store import run_record, checkpoint_inputs
from tests import test_learner_release_store as store_fixtures


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.store = LearnerReleaseStore(self.root/'store')
        self.run = run_record(); self.run_sha = self.store.create_run(self.run)
        self.control = self.store.root/'control'/self.run_sha; self.control.mkdir(parents=True)
        _write(self.control/'created.json', {'run_sha256':self.run_sha})
        self.directory = self.control/rh.DIRECTORY; self.directory.mkdir()
        rh._write_record(self.directory,'mode.json',rh._mode(self.run_sha,self.run))
        self.parent = None; self.step = 0

    def commit(self):
        cp = self.store.load_checkpoint(self.parent) if self.parent else None
        values = checkpoint_inputs(self.run,self.run_sha,parent=cp)
        pin = self.store.create_checkpoint(self.run_sha,*values,parent_checkpoint_sha256=self.parent)
        self.step += 1
        intent = {'run_sha256':self.run_sha,'step':self.step,'parent_checkpoint_sha256':self.parent}
        _write(self.control/f'intent-{self.step:08d}.json',intent)
        _write(self.control/f'commit-{self.step:08d}.json',{'step':self.step,'checkpoint_sha256':pin,'intent_sha256':content_sha256(intent)})
        self.parent = pin
        return pin

    def begin(self):
        intent = rh.publication_state(self.store,self.run_sha)['pending']['expected_intent']
        rh._write_record(self.directory,f'intent-{self.step:08d}.json',intent)
        return intent

    def release(self):
        return self.store.publish(self.parent,store_fixtures.LearnerReleaseStoreTests.weights(self))

    def test_no_bootstrap_unpublished_blocks_and_reconcile_is_read_only(self):
        with self.assertRaisesRegex(RuntimeError,'no published'):rh.read_publication_head(self.store,self.run_sha)
        self.commit()
        with self.assertRaisesRegex(RuntimeError,'unpublished'):rh.read_publication_head(self.store,self.run_sha)
        self.begin(); pin=self.release()
        with patch.object(self.store,'publish',side_effect=AssertionError('no retry')):
            head=rh.reconcile_published(self.store,run_sha=self.run_sha,model_release_sha256=pin)
            self.assertEqual(head,rh.read_publication_head(self.store,self.run_sha))
            self.assertEqual(head,rh.reconcile_published(self.store,run_sha=self.run_sha,model_release_sha256=pin))

    def test_chain_rollback_cross_run_and_train_past_unpublished_rejected(self):
        self.commit();self.begin();pin=self.release()
        first=rh.reconcile_published(self.store,run_sha=self.run_sha,model_release_sha256=pin)
        self.commit();self.begin()
        with self.assertRaisesRegex(RuntimeError,'different run/checkpoint'):
            rh.reconcile_published(self.store,run_sha=self.run_sha,model_release_sha256=pin)
        second_pin=self.release();second=rh.reconcile_published(self.store,run_sha=self.run_sha,model_release_sha256=second_pin)
        self.assertEqual(second['previous_head_sha256'],content_sha256(first))
        self.commit();self.commit()
        with self.assertRaisesRegex(RuntimeError,'advanced'):rh.read_publication_head(self.store,self.run_sha)
        with self.assertRaisesRegex(RuntimeError,'invalid publication run'):rh.read_publication_head(self.store,'../bad')

    def test_partial_and_orphan_stage_fail_closed(self):
        self.commit();(self.directory/'intent-00000001.json').write_bytes(b'{')
        with self.assertRaises(ValueError):rh.read_publication_head(self.store,self.run_sha)
        (self.directory/'intent-00000001.json').unlink()
        _write(self.directory/'.staged-orphan',{'partial':'unpublished'})
        with self.assertRaisesRegex(RuntimeError,'staging'):rh.read_publication_head(self.store,self.run_sha)

    def test_atomic_postlink_sync_failure_is_not_retried(self):
        self.commit();intent=self.begin();pin=self.release()
        from cpu_runtime.verified_dataset_store import _Directory
        original=_Directory.sync; calls=[]
        def fail(handle):
            if handle.path==self.directory:
                calls.append(1);raise OSError('postlink fsync unknown')
            return original(handle)
        with patch.object(_Directory,'sync',fail):
            with self.assertRaises(OSError):rh.reconcile_published(self.store,run_sha=self.run_sha,model_release_sha256=pin)
        self.assertEqual(len(calls),1)
        # Explicit read-only reconciliation of a visible complete record is safe;
        # it does not issue another CAS publication or training mutation.
        with patch.object(self.store,'publish',side_effect=AssertionError('no publish')):
            self.assertEqual(rh.reconcile_published(self.store,run_sha=self.run_sha,model_release_sha256=pin)['model_release_sha256'],pin)

    def test_complete_orphan_confirmation_reconciles_only_exact_pinned_release(self):
        self.commit();intent=self.begin();pin=self.release()
        confirmation={**intent,'publication_intent_sha256':content_sha256(intent),'model_release_sha256':pin}
        _write(self.directory/'.staged-confirmation',confirmation)
        with self.assertRaisesRegex(RuntimeError,'staging'):rh.read_publication_head(self.store,self.run_sha)
        with patch.object(self.store,'publish',side_effect=AssertionError('do not republish')):
            self.assertEqual(rh.reconcile_published(self.store,run_sha=self.run_sha,model_release_sha256=pin),confirmation)
        self.assertTrue((self.directory/'.staged-confirmation').exists())

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None,'tiny autograd requires local torch; stdlib protocol tests remain runnable')
class TinyPublicationTests(unittest.TestCase):
    def setUp(self):
        from tests.test_continual_mixed_learner import ContinualMixedTests
        ContinualMixedTests.setUp(self);ContinualMixedTests.enable_v3(self)
        self.learner=ContinualMixedTests.learner(self)
        self.learner.enable_release_head()

    def test_actual_train_publish_restore_and_next_step(self):
        from gpu_runtime.continual_mixed_learner import ContinualMixedLearner
        learner=self.learner;one=learner.train_next_and_publish()
        self.assertEqual(rh.read_publication_head(learner.store,learner.run_sha),one['head_receipt'])
        learner.close();self.runtime.delete_session(learner.learner_id)
        with patch.object(self.runtime,'learn',wraps=self.runtime.learn) as learn:
            restored=ContinualMixedLearner.restore(self.runtime,checkpoint_sha256=one['checkpoint_sha256'],journal_root=self.root/'restore')
            self.addCleanup(restored.close);learn.assert_not_called()
            two=restored.train_next_and_publish();self.assertEqual(learn.call_count,1)
        self.assertEqual(two['step'],2)
        self.assertEqual(two['head_receipt']['previous_head_sha256'],content_sha256(one['head_receipt']))

    def test_unknown_publish_blocks_train_restore_until_explicit_reconcile(self):
        from gpu_runtime.continual_mixed_learner import ContinualMixedLearner
        learner=self.learner;original=learner.store.publish;pins=[]
        def lost(*args,**kwargs):
            pin=original(*args,**kwargs);pins.append(pin);raise TimeoutError('ACK lost')
        with patch.object(learner.store,'publish',side_effect=lost),patch.object(self.runtime,'learn',wraps=self.runtime.learn) as learn:
            with self.assertRaises(TimeoutError):learner.train_next_and_publish()
            with self.assertRaises(RuntimeError):learner.train_next()
            with self.assertRaises(RuntimeError):learner.publish()
            self.assertEqual(learn.call_count,1)
        checkpoint=learner.parent_checkpoint;learner.close();self.runtime.delete_session(learner.learner_id)
        with patch.object(self.runtime,'create_session',wraps=self.runtime.create_session) as create:
            with self.assertRaisesRegex(RuntimeError,'unresolved publication'):
                ContinualMixedLearner.restore(self.runtime,checkpoint_sha256=checkpoint,journal_root=self.root/'bad-restore')
            create.assert_not_called()
        rh.reconcile_published(learner.store,run_sha=learner.run_sha,model_release_sha256=pins[0])
        restored=ContinualMixedLearner.restore(self.runtime,checkpoint_sha256=checkpoint,journal_root=self.root/'good-restore')
        self.addCleanup(restored.close)
        self.assertEqual(restored.sampler_state['step'],1)

    def test_direct_train_requires_publish_and_historical_pin_cannot_advance_head(self):
        learner=self.learner;one=learner.train_next()
        with patch.object(self.runtime,'learn',wraps=self.runtime.learn) as learn:
            with self.assertRaisesRegex(RuntimeError,'unpublished'):learner.train_next()
            learn.assert_not_called()
        learner.publish();two=learner.train_next()
        with self.assertRaisesRegex(RuntimeError,'latest committed'):learner.publish(one['checkpoint_sha256'])
        self.assertEqual(learner.sampler_state['step'],2)

    def test_real_owner_provider_reselects_after_prepare_without_touching_fixed_actor(self):
        from cpu_runtime.released_attempts import ReleasedAttemptProvider
        from cpu_runtime import matchmaker as mm
        from tests.test_matchmaker import config, curriculum, source
        learner=self.learner;first=learner.train_next_and_publish()
        self.runtime.create_session('fixed-actor',theorem_id='1'*64,model_release_sha256=first['release']['model_release_sha256'])
        actor_before=self.runtime.actor.submit(lambda:self.backend.session_fingerprints('fixed-actor'))
        scheduler=mm.Matchmaker.create(self.root/'scheduler',curriculum(),config(max_attempts=1));self.addCleanup(scheduler.close)
        calls=[]
        def prepare(proposal):
            calls.append(proposal);self.assertFalse(learner.lock._is_owned())
            learner.train_next_and_publish();return source(proposal)
        intent,selection=ReleasedAttemptProvider(learner)(scheduler,prepare)
        self.assertEqual(len(calls),1);self.assertEqual(selection['head']['step'],2)
        self.assertNotEqual(intent['proposal']['model_release_sha256'],first['release']['model_release_sha256'])
        self.assertEqual(actor_before,self.runtime.actor.submit(lambda:self.backend.session_fingerprints('fixed-actor')))
