from copy import deepcopy
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from cpu_runtime import matchmaker as mm
from cpu_runtime.replica_collector import ReplicaCollector, run_replica_collector_batch, validate_endpoints
from cpu_runtime.http_clients import GpuHttpClient
from cpu_runtime.transport_budget import DEFAULT_BUDGET
from gpu_runtime.runtime import GpuRuntime
from gpu_runtime.server import RuntimeHandler
from gpu_runtime.toy_backend import ToyBackend
from tests.test_matchmaker import RELEASE, config, curriculum, source
from tests.test_collector_batch import search_response, retirement, verified


def endpoints():
    return [{"replica_id":f"replica-{i}","deployment_id":f"local-process-{i}",
             "base_url":f"http://127.0.0.1:{18770+i}","model_release_sha256":RELEASE} for i in range(2)]


class Client:
    def __init__(self, url, **kwargs):
        self.url=url; self.calls=[]
    def retire_session(self, sid, name, *, expected_policy_version):
        self.calls.append(sid)
        return {"schema_version":"reap.gpu.retirement.v1","session_id":sid,"policy_version":0,
            "snapshot":name,"snapshot_sha256":"a"*64,"status":"released",
            "mutation_retry_allowed":False,"tombstone_scope":"current_runtime"}


class ReplicaCollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.scheduler=mm.Matchmaker.create(self.root/'scheduler',curriculum(),config(max_inflight=4,max_attempts=12))
        self.addCleanup(self.scheduler.close)

    def intent(self,release=RELEASE):
        proposal=self.scheduler.plan_next(release)
        return self.scheduler.reserve(proposal,source(proposal))

    def router(self, callback=None, **kwargs):
        result=ReplicaCollector(self.root/'router',endpoints(),callback or (lambda intent,endpoint:search_response(intent)),
                                client_factory=kwargs.pop('client_factory',Client),**kwargs)
        self.addCleanup(result.close);return result

    def test_default_retirement_timeout_matches_bridge_budget(self):
        observed=[]
        def factory(url,*,timeout_seconds):
            observed.append(timeout_seconds);return Client(url)
        self.router(client_factory=factory)
        self.assertEqual(observed,[DEFAULT_BUDGET.client]*2)

    def test_twelve_attempts_use_existing_batch_and_bounded_slots(self):
        calls=[]
        def search(intent,endpoint):
            calls.append((intent['proposal']['attempt_id'],endpoint['replica_id']))
            time.sleep(0.002)
            return search_response(intent)
        router=self.router(search)
        result=run_replica_collector_batch(self.scheduler,self.root/'batch',router=router,prepare_source=source,verify=verified)
        self.assertEqual(result['status'],'completed');self.assertEqual(len(result['exhausted']),12)
        self.assertEqual(set(e for _,e in calls),{'replica-0','replica-1'})
        self.assertEqual(router.active,{})
        self.assertEqual(len({aid for aid,_ in calls}),12)
        for aid,endpoint in calls:
            record=json.loads((self.root/'router/attempts'/aid/'route.json').read_bytes())
            self.assertEqual(record['endpoint']['replica_id'],endpoint)
        router.close()
        again=self.router(lambda *args:self.fail('completed search repeated'))
        resumed=run_replica_collector_batch(self.scheduler,self.root/'batch',router=again,prepare_source=source,verify=verified)
        self.assertEqual(resumed['started'],[])

    def test_two_search_callbacks_overlap_without_holding_router_lock(self):
        barrier=threading.Barrier(2);actual=[]
        def callback(intent,endpoint):
            aid=intent['proposal']['attempt_id']
            self.assertTrue((self.root/'router/attempts'/aid/'route.json').is_file())
            barrier.wait(3);actual.append(endpoint['replica_id']);return search_response(intent)
        router=self.router(callback);intents=[self.intent(),self.intent()]
        errors=[]
        def run(intent):
            try:router.retire(intent,router.search(intent))
            except Exception as exc:errors.append(exc)
        workers=[threading.Thread(target=run,args=(intent,)) for intent in intents]
        for worker in workers:worker.start()
        for worker in workers:worker.join(5)
        self.assertEqual(errors,[]);self.assertEqual(set(actual),{'replica-0','replica-1'})

    def test_release_or_intent_mismatch_does_not_dispatch(self):
        calls=[];router=self.router(lambda *args:calls.append(args))
        wrong=self.intent('f'*64)
        with self.assertRaisesRegex(ValueError,'new release'):router.search(wrong)
        wrong=deepcopy(wrong);wrong['proposal']['model_release_sha256']=RELEASE
        with self.assertRaisesRegex(ValueError,'pin mismatch'):router.search(wrong)
        self.assertEqual(calls,[]);self.assertEqual(router.active,{})

    def test_retire_unknown_keeps_route_and_blocks_new_work_and_restart(self):
        class Unknown(Client):
            def retire_session(self,*args,**kwargs):
                self.calls.append(args[0]);raise TimeoutError('do not publish exception detail')
        router=self.router(client_factory=Unknown);intent=self.intent();response=router.search(intent)
        with self.assertRaises(TimeoutError):router.retire(intent,response)
        self.assertEqual(len(router.active),1)
        with self.assertRaises(ValueError):router.retire(intent,response)
        with self.assertRaisesRegex(ValueError,'unknown'):router.search(self.intent())
        self.assertEqual(sum(len(c.calls) for c in router.clients.values()),1)
        raw=b''.join(p.read_bytes() for p in (self.root/'router').rglob('*.json'))
        self.assertNotIn(b'do not publish exception detail',raw)
        router.close()
        with self.assertRaisesRegex(ValueError,'incomplete/extra'):self.router()

    def test_search_unknown_has_no_retire_or_failover(self):
        calls=[]
        def bad(intent,endpoint):calls.append(endpoint);raise TimeoutError()
        router=self.router(bad);intent=self.intent()
        with self.assertRaises(TimeoutError):router.search(intent)
        self.assertEqual(len(calls),1);self.assertEqual(len(router.active),1)
        self.assertEqual(sum(len(c.calls) for c in router.clients.values()),0)
        router.close()
        with self.assertRaises(ValueError):self.router()

    def test_capacity_held_until_successfully_persisted_retirement(self):
        router=self.router();first,second=self.intent(),self.intent()
        a,b=router.search(first),router.search(second)
        with self.assertRaisesRegex(ValueError,'both replica'):router.search(self.intent())
        original=router._write
        def fail(aid,name,value):
            if name=='retirement.json':raise OSError('disk failure')
            return original(aid,name,value)
        with patch.object(router,'_write',side_effect=fail):
            with self.assertRaises(OSError):router.retire(first,a)
        self.assertEqual(len(router.active),2)
        # Already admitted work may retire even when another route is unknown.
        router.retire(second,b);self.assertEqual(len(router.active),1)

    def test_changed_receipt_or_duplicate_completed_sid_cannot_retire_or_search(self):
        router=self.router();intent=self.intent();response=router.search(intent)
        changed=deepcopy(response);changed['artifacts_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'receipt changed'):router.retire(intent,changed)
        router.retire(intent,response)
        with self.assertRaisesRegex(ValueError,'already routed'):router.search(intent)
        self.assertEqual(sum(len(c.calls) for c in router.clients.values()),1)

    def test_pool_reconfiguration_and_single_writer_rejected(self):
        router=self.router()
        with self.assertRaises(Exception):self.router()
        router.close();changed=endpoints();changed[0]['deployment_id']='new-process'
        with self.assertRaisesRegex(ValueError,'identity changed'):
            ReplicaCollector(self.root/'router',changed,lambda *args:None,client_factory=Client)

    def test_endpoint_schema_rejects_credentials_duplicates_and_different_releases(self):
        for field,value in [('base_url','http://user:secret@host:1234'),('replica_id','replica-1'),
                            ('deployment_id','local-process-1'),('model_release_sha256','f'*64),
                            ('base_url','http://localhost:1234/path')]:
            with self.subTest(field=field,value=value):
                changed=endpoints();changed[0][field]=value
                with self.assertRaises(ValueError):validate_endpoints(changed)

    def test_unknown_search_stops_batch_but_other_inflight_route_retires(self):
        barrier=threading.Barrier(2)
        def search(intent,endpoint):
            barrier.wait(3)
            if intent['proposal']['attempt_index']==1:raise TimeoutError()
            time.sleep(0.02)
            return search_response(intent)
        router=self.router(search)
        result=run_replica_collector_batch(self.scheduler,self.root/'batch',router=router,prepare_source=source,verify=verified)
        self.assertEqual(result['status'],'blocked');self.assertEqual(len(result['started']),2)
        self.assertEqual(len(result['unknown']),1);self.assertEqual(len(result['exhausted']),1)
        self.assertEqual(len(router.active),1)
        self.assertEqual(sum(len(c.calls) for c in router.clients.values()),1)

    def test_route_publication_ack_unknown_never_invokes_search(self):
        calls=[];router=self.router(lambda *args:calls.append(args));intent=self.intent()
        original=mm._publish
        def after_link(directory,name,value):
            original(directory,name,value)
            if name=='route.json':raise OSError('unknown durability')
        with patch.object(mm,'_publish',side_effect=after_link):
            with self.assertRaises(OSError):router.search(intent)
        self.assertEqual(calls,[]);self.assertTrue(router.blocked)
        router.close()
        with self.assertRaisesRegex(ValueError,'incomplete/extra'):self.router()

    def test_completed_record_tampering_rejected_without_callback(self):
        router=self.router();intent=self.intent();router.retire(intent,router.search(intent));router.close()
        path=self.root/'router/attempts'/intent['proposal']['attempt_id']/'retirement.json'
        record=json.loads(path.read_bytes());record['session_id']='other'
        path.write_bytes(mm.canonical_bytes(record))
        with self.assertRaises(ValueError):self.router(lambda *args:self.fail('must not invoke'))

    def test_real_two_http_runtimes_keep_capacity_and_retire_on_original_endpoint(self):
        servers=[];runtimes=[];bindings=endpoints()
        for i in range(2):
            runtime=GpuRuntime(backend=ToyBackend(),snapshot_root=self.root/f'snap-{i}',max_resident_sessions=1)
            handler=type(f'Handler{i}',(RuntimeHandler,),{'runtime':runtime,'backend_name':'toy'})
            server=ThreadingHTTPServer(('127.0.0.1',0),handler)
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            servers.append((server,worker));runtimes.append(runtime)
            bindings[i]['base_url']=f'http://127.0.0.1:{server.server_port}'
        def cleanup():
            for server,worker in servers:server.shutdown();server.server_close();worker.join(3)
            for runtime in runtimes:runtime.close()
        self.addCleanup(cleanup)
        barrier=threading.Barrier(2)
        def callback(intent,endpoint):
            client=GpuHttpClient(endpoint['base_url']);sid=intent['proposal']['attempt_id']
            created=client.create_session(sid)
            self.assertEqual(created['policy_version'],0)
            # Toy HTTP exercises real routing/retirement; it does not impersonate
            # the real collector's model-release or independent Lean gates.
            if intent['proposal']['attempt_index'] <= 2:barrier.wait(3)
            request={'messages':[{'role':'user','content':'toy routing fixture'}],'n':1,'max_tokens':8}
            policy=client._request('POST',f'/sessions/{sid}/policy/v1/chat/completions',request)
            value=client._request('POST',f'/sessions/{sid}/value/v1/chat/completions',request)
            self.assertEqual(policy['policy_version'],0);self.assertEqual(value['policy_version'],0)
            return search_response(intent)
        with ReplicaCollector(self.root/'http-router',bindings,callback) as router:
            result=run_replica_collector_batch(self.scheduler,self.root/'http-batch',router=router,prepare_source=source,verify=verified)
        self.assertEqual(result['status'],'completed');self.assertEqual(len(result['exhausted']),12)
        self.assertTrue(all(not runtime.backend._states for runtime in runtimes))
        self.assertTrue(all(runtime.actor.metrics()['max_active']==1 for runtime in runtimes))


if __name__=='__main__':unittest.main()
