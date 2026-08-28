"""Two real tiny Adam releases and private actor sessions; no GPU/Lean proof.

Dataset admission and base identity use the existing explicit tiny fixtures.
Actual learner/store/publication/HTTP initialization/retirement are not mocked.
"""
from copy import deepcopy
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch=None

from cpu_runtime import matchmaker as mm
from cpu_runtime.replica_collector import ReplicaCollector, run_replica_collector_batch
from cpu_runtime.released_attempts import ReleasedAttemptProvider
from cpu_runtime.http_clients import GpuHttpClient
from tests.test_matchmaker import curriculum, config, source
from tests.test_collector_batch import search_response, verified
from tests.test_replica_collector import endpoints, Client


@unittest.skipIf(torch is None,'actual tiny Adam fixture requires local torch')
class LatestReplicaTests(unittest.TestCase):
    def setUp(self):
        from tests.test_release_head import TinyPublicationTests
        TinyPublicationTests.setUp(self)
        from gpu_runtime.continual_mixed_learner import ContinualMixedLearner
        from gpu_runtime.continual_mixed_store import ContinualMixedStore
        self.assertIsInstance(self.learner,ContinualMixedLearner)
        self.assertIsInstance(self.learner.store,ContinualMixedStore)
        self.assertEqual(self.learner.run['schema_version'],'reap.learner.run.v3')
        self.provider=ReleasedAttemptProvider(self.learner)
        self.scheduler=mm.Matchmaker.create(self.root/'scheduler',curriculum(),config(max_attempts=2))
        self.addCleanup(self.scheduler.close)

    def bindings(self,pin):
        values=endpoints()
        for item in values:item['model_release_sha256']=pin
        return values

    def runtime_pair(self,pin):
        from tests.test_mixed_backend import MixedBackendTests
        from gpu_runtime import GpuRuntime
        from gpu_runtime.server import RuntimeHandler
        values=self.bindings(pin); runtimes={}
        for index in range(2):
            backend=MixedBackendTests.backend(self)
            backend.model.add_adapter('__bootstrap__',backend.lora_config)
            backend.model.adapters['__bootstrap__'].requires_grad_(False)
            backend._checkpoint_adapter_specs=lambda b=backend:{name:(tuple(t.shape),t.dtype)
                for name,t in b._adapter_state_dict('__bootstrap__').items()}
            runtime=GpuRuntime(backend=backend,snapshot_root=self.root/f'endpoint-{index}',
                learner_release_root=self.root/'store',learner_profile='continual-mixed-v3',max_resident_sessions=1)
            self.addCleanup(runtime.close)
            handler=type(f'LatestHandler{index}',(RuntimeHandler,),{'runtime':runtime,'backend_name':'mixed-replay'})
            server=ThreadingHTTPServer(('127.0.0.1',0),handler)
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            def cleanup(s=server,w=worker):s.shutdown();s.server_close();w.join(3)
            self.addCleanup(cleanup)
            values[index]['base_url']=f'http://127.0.0.1:{server.server_port}'
            runtimes[values[index]['replica_id']]=runtime
        return values,runtimes

    def test_real_two_generations_use_idle_endpoint_without_mutating_live_old_actor(self):
        first=self.learner.train_next_and_publish();r1=first['release']['model_release_sha256']
        weights1=self.runtime.actor.submit(lambda:self.backend.session_fingerprints(self.learner.learner_id))
        values,runtimes=self.runtime_pair(r1)
        old_ready=threading.Event();new_ready=threading.Event(); observed={}; second={}
        def prepare(proposal):
            if proposal['attempt_index']==2:
                self.assertTrue(old_ready.wait(5))
                second.update(self.learner.train_next_and_publish(append_dataset_pins=['8'*64]))
            return source(proposal)
        def search(intent,endpoint):
            aid=intent['proposal']['attempt_id'];index=intent['proposal']['attempt_index']
            runtime=runtimes[endpoint['replica_id']]
            created=GpuHttpClient(endpoint['base_url']).create_session(aid,theorem_id=str(index)*64,
                model_release_sha256=endpoint['model_release_sha256'])
            self.assertEqual(created['policy_version'],0);self.assertEqual(created['role'],'actor')
            before=runtime.actor.submit(lambda:runtime.backend.session_fingerprints(aid))
            observed[index]={'release':endpoint['model_release_sha256'],'endpoint':endpoint['replica_id'],
                             'fingerprint':before,'created':deepcopy(created)}
            for key in ('adapter','value_head'):
                wanted=weights1 if index==1 else self.runtime.actor.submit(lambda:self.backend.session_fingerprints(self.learner.learner_id))
                self.assertEqual(before[key],wanted[key])
            if index==1:
                old_ready.set();self.assertTrue(new_ready.wait(8))
                self.assertEqual(before,runtime.actor.submit(lambda:runtime.backend.session_fingerprints(aid)))
                self.assertEqual(runtime.sessions.get(aid).policy_version,0)
            else:
                # R1 is still resident while R2 has already initialized privately.
                old_runtime=runtimes[observed[1]['endpoint']]
                self.assertTrue(old_runtime._resident_session_ids)
                new_ready.set()
            self.assertTrue((self.root/'batch/attempts'/aid/'publication-selection.json').is_file())
            return search_response(intent)
        with ReplicaCollector(self.root/'router',values,search,publication_provider=self.provider) as router:
            report=run_replica_collector_batch(self.scheduler,self.root/'batch',router=router,prepare_source=prepare,verify=verified)
        self.assertEqual(report['status'],'completed',report)
        self.assertEqual(self.learner.sampler_state['step'],2)
        checkpoint=self.learner.store.load_checkpoint(second['checkpoint_sha256'])
        self.assertEqual(checkpoint['data_receipt']['schema_version'],'reap.learner.data-receipt.v3')
        self.assertEqual(checkpoint['data_receipt']['catalog'][-1]['dataset_sha256'],'8'*64)
        self.assertEqual(checkpoint['manifest']['parent_checkpoint_sha256'],first['checkpoint_sha256'])
        self.assertEqual(observed[1]['release'],r1)
        self.assertEqual(observed[2]['release'],second['release']['model_release_sha256'])
        self.assertNotEqual(observed[1]['release'],observed[2]['release'])
        self.assertNotEqual(observed[1]['endpoint'],observed[2]['endpoint'])
        self.assertTrue(all(not runtime._resident_session_ids for runtime in runtimes.values()))
        # Completed restore validates old/new route pins, no current head lookup
        # or training/search/retirement replay is needed.
        with patch.object(self.runtime,'learn',side_effect=AssertionError('must not retrain')):
            with ReplicaCollector(self.root/'router',values,lambda *a:self.fail('search repeated'),publication_provider=self.provider) as router:
                again=run_replica_collector_batch(self.scheduler,self.root/'batch',router=router,prepare_source=lambda *a:self.fail('prepare repeated'),verify=verified)
        self.assertEqual(again['started'],[])

    def test_unknown_second_publication_prevents_new_reservation_but_old_actor_can_retire(self):
        first=self.learner.train_next_and_publish();pin=first['release']['model_release_sha256']
        calls=[];old_ready=threading.Event();failure=threading.Event()
        def search(intent,endpoint):
            calls.append((intent,endpoint));old_ready.set();self.assertTrue(failure.wait(6));return search_response(intent)
        def prepare(proposal):
            if proposal['attempt_index']==2:
                self.assertTrue(old_ready.wait(5))
                original=self.learner.store.publish
                def lost(*args,**kwargs):original(*args,**kwargs);raise TimeoutError('lost publication ACK')
                try:
                    with patch.object(self.learner.store,'publish',side_effect=lost):self.learner.train_next_and_publish()
                finally:failure.set()
            return source(proposal)
        with ReplicaCollector(self.root/'router',self.bindings(pin),search,client_factory=Client,publication_provider=self.provider) as router:
            report=run_replica_collector_batch(self.scheduler,self.root/'batch',router=router,prepare_source=prepare,verify=verified)
            self.assertEqual(router.active,{})
        self.assertEqual(report['status'],'blocked');self.assertEqual(len(calls),1)
        self.assertEqual(len(self.scheduler.status()['attempts']),1)
        self.assertEqual(self.learner.sampler_state['step'],2)
        with self.assertRaises(RuntimeError):self.provider(self.scheduler,source)

    def test_unconfirmed_release_contract_change_and_cross_run_are_rejected_before_callback(self):
        first=self.learner.train_next_and_publish();pin=first['release']['model_release_sha256'];calls=[]
        with ReplicaCollector(self.root/'router',self.bindings(pin),lambda *a:calls.append(a),client_factory=Client,publication_provider=self.provider) as router:
            checkpoint=self.learner.train_next()
            from gpu_runtime.learner import publish_checkpoint
            unconfirmed=publish_checkpoint(self.runtime,checkpoint_sha256=checkpoint['checkpoint_sha256'],
                journal_root=self.root/'outside-head-publication',expected_run=self.learner.run)
            # A real release CAS exists, but no published-00000002 confirmation.
            # Mere existence must not alias an unknown publication into the head.
            with self.assertRaises(FileNotFoundError):router._validate_release(unconfirmed['model_release_sha256'])
            # Existing committed source is accepted, but tampered metadata cannot
            # enter cache even when the URI uses an otherwise valid release pin.
            metadata,weights=self.learner.store.load_release(pin)
            for field in ('contract','run_sha256'):
                altered=deepcopy(metadata)
                if field=='contract':altered[field]['base_sha256']='0'*64
                else:altered[field]='0'*64
                with patch.object(self.learner.store,'load_release',return_value=(altered,weights)):
                    with self.assertRaises(ValueError):router._validate_release(pin)
            self.assertEqual(calls,[])
        changed=self.bindings(pin)
        with self.assertRaisesRegex(ValueError,'identity changed'):
            ReplicaCollector(self.root/'router',changed,lambda *a:None,client_factory=Client)
