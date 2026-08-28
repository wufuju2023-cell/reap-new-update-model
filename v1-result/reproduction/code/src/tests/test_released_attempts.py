"""Same-process scheduling mechanism tests with explicit mock publication state."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch
from cpu_runtime import matchmaker as mm
from cpu_runtime.released_attempts import ReleasedAttemptProvider, validate_selection
from cpu_runtime.collector_batch import run_collector_batch
from gpu_runtime.learner_release_store import content_sha256
from tests.test_matchmaker import config, curriculum, source, pin
from tests.test_collector_batch import search_response, retirement, verified


def head(step=1, previous=None):
    intent={'schema_version':'reap.learner.release-head.v1','run_sha256':pin('run'),'step':step,
        'checkpoint_sha256':pin('cp'+str(step)),'previous_head_sha256':previous}
    return {**intent,'publication_intent_sha256':content_sha256(intent),'model_release_sha256':pin('release'+str(step))}


class ReleasedAttemptTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.learner=SimpleNamespace(run_sha=pin('run'),lock=threading.RLock(),head=head(),blocked=False)
        def available(learner):
            self.assertTrue(learner.lock._is_owned())
            if learner.blocked:raise RuntimeError('unknown publication')
            return deepcopy(learner.head)
        mock=patch('cpu_runtime.released_attempts.available_head',side_effect=available);mock.start();self.addCleanup(mock.stop)
        self.provider=ReleasedAttemptProvider(self.learner)
        self.scheduler=mm.Matchmaker.create(self.root/'scheduler',curriculum(),config(max_attempts=2))
        self.addCleanup(self.scheduler.close)

    def test_prepare_once_outside_lock_new_head_bound_old_intent_immutable(self):
        before=deepcopy(self.learner.head);calls=[]
        def prepare(proposal):
            self.assertFalse(self.learner.lock._is_owned());calls.append(proposal)
            def publish():
                with self.learner.lock:self.learner.head=head(2,content_sha256(before))
            worker=threading.Thread(target=publish);worker.start();worker.join(2);self.assertFalse(worker.is_alive())
            return source(proposal)
        intent,selection=self.provider(self.scheduler,prepare)
        self.assertEqual(len(calls),1)
        self.assertEqual(intent['proposal']['model_release_sha256'],head(2,content_sha256(before))['model_release_sha256'])
        validate_selection(selection,intent,self.learner.run_sha)
        saved=deepcopy(intent);self.learner.head=head(3,content_sha256(self.learner.head))
        self.assertEqual(self.scheduler.status()['attempts'][intent['proposal']['attempt_id']]['intent'],saved)

    def test_unknown_after_prepare_prevents_reserve_without_second_prepare(self):
        calls=[]
        def prepare(proposal):
            calls.append(proposal);self.learner.blocked=True;return source(proposal)
        with self.assertRaisesRegex(RuntimeError,'unknown'):self.provider(self.scheduler,prepare)
        self.assertEqual(len(calls),1);self.assertEqual(self.scheduler.status()['attempts'],{})

    def test_scheduler_change_during_prepare_refused_without_retry(self):
        def prepare(proposal):
            other=self.scheduler.plan_next(pin('explicit-other'))
            self.scheduler.reserve(other,source(other));return source(proposal)
        with self.assertRaisesRegex(mm.MatchmakerError,'scheduler changed'):self.provider(self.scheduler,prepare)
        self.assertEqual(len(self.scheduler.status()['attempts']),1)

    def test_batch_persists_selection_before_search_and_completed_resume_is_readonly(self):
        def search(intent):
            path=self.root/'batch/attempts'/intent['proposal']['attempt_id']/'publication-selection.json'
            self.assertTrue(path.is_file());return search_response(intent)
        arguments=dict(prepare_source=source,provide_release=lambda: (_ for _ in ()).throw(AssertionError('old provider')),
            search=search,retire=retirement,verify=verified,reserve_attempt=self.provider)
        report=run_collector_batch(self.scheduler,self.root/'batch',**arguments)
        self.assertEqual(report['status'],'completed');self.assertEqual(len(report['started']),2)
        self.learner.blocked=True
        # All known completed attempts must audit their pinned sidecars without
        # requiring a live latest head or reissuing search/retirement callbacks.
        report=run_collector_batch(self.scheduler,self.root/'batch',**arguments)
        self.assertEqual(report['status'],'completed');self.assertEqual(report['started'],[])

    def test_sidecar_cross_run_release_and_intent_changes_rejected(self):
        intent,selection=self.provider(self.scheduler,source)
        for key,value in [('intent_sha256',pin('wrong')),('learner_run_sha256',pin('other')),('head_sha256',pin('wrong'))]:
            changed=deepcopy(selection);changed[key]=value
            with self.assertRaises(mm.MatchmakerError):validate_selection(changed,intent,self.learner.run_sha)
