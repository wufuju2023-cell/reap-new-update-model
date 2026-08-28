"""Explicit --run only: three-process 7B publication/consumption mechanism.

No Lean/search/verification callbacks, mathematical verdict, new curriculum,
mutation retries or automatic restarts. Scheduler reservations intentionally
remain pending: they identify inference probes, not completed proof attempts.
Only accept report.ok AND external main exit=0 AND both child exit=0.
"""
from __future__ import annotations
import argparse, base64, hashlib, io, json, math, multiprocessing as mp
import os, shutil, sys, threading, time
from copy import deepcopy
from pathlib import Path
from http.server import ThreadingHTTPServer
from containers.gpu import smoke_continual_mixed_learner as old
from containers.gpu import smoke_mixed_learner as mixed
from containers.gpu.smoke_gpu import require, equal_tree, base_fingerprint
from gpu_runtime import GpuRuntime
from gpu_runtime.continual_mixed_learner import ContinualMixedLearner
from gpu_runtime.continual_mixed_store import ContinualMixedStore
from gpu_runtime.learner_release_store import content_sha256, canonical_bytes
from gpu_runtime.mixed_backend import MixedReplayBackend
from gpu_runtime.server import RuntimeHandler
from gpu_runtime.snapshot_store import _json_bytes
from cpu_runtime import matchmaker as mm
from cpu_runtime.http_clients import GpuHttpClient
from cpu_runtime.released_attempts import ReleasedAttemptProvider, validate_selection

save = old.save

def tree_digest(torch, value):
    """Exact recursive tensor dtype/shape/bytes plus canonical scalar identities."""
    if torch.is_tensor(value):
        x=value.detach().cpu().contiguous()
        return {'tensor_dtype':str(x.dtype),'shape':list(x.shape),
                'sha256':hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value,dict):
        return {'dict':sorted([[tree_digest(torch,k),tree_digest(torch,v)] for k,v in value.items()],key=lambda x:canonical_bytes(x[0]))}
    if isinstance(value,(list,tuple)):
        return {type(value).__name__:[tree_digest(torch,x) for x in value]}
    return {'scalar_type':type(value).__name__,'value':value}

def state_summary(runtime,backend,sid):
    state=mixed.capture(runtime,backend,sid)
    return {'metadata':state['metadata'], 'complete_sha256':content_sha256(tree_digest(backend.torch,state)),
            'parameters':{key:content_sha256(tree_digest(backend.torch,state['backend'][key])) for key in ('adapter','value_head')},
            'optimizer_steps':state['backend']['optimizer_steps'],'examples_seen':state['backend']['examples_seen'],
            'optimizer_empty':state['backend']['optimizer']['state']=={}}

def build_backend(config):
    import torch
    require(torch.cuda.is_available() and bool(torch.version.hip),'actual AMD GPU required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    backend=MixedReplayBackend(config['model_path'],dataset_root=config['replay_dataset_root'],
        mathlib_dataset_root=config['mathlib_dataset_root'],**old.backend_options(config['source']))
    require(backend.hidden_size==3584 and backend.device.type=='cuda'
        and all(t.device==backend.device for t in backend.model.parameters()),'actual 7B device mismatch')
    require(canonical_bytes(backend.experience_contract())==canonical_bytes(config['source']['contract']),'exact source contract mismatch')
    return backend

def child(pipe,config,root,factory=build_backend):
    runtime=server=thread=None;status={'ok':False,'pid':os.getpid()}
    root=Path(root)
    try:
        backend=factory(config)
        runtime=GpuRuntime(backend=backend,snapshot_root=root/'snapshots',learner_release_root=config['store'],
                           learner_profile='continual-mixed-v3',max_resident_sessions=1)
        base=base_fingerprint(backend,8*1024*1024)
        handler=type('ProbeHandler',(RuntimeHandler,),{'runtime':runtime,'backend_name':'mixed-replay'})
        server=ThreadingHTTPServer(('127.0.0.1',0),handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        ready={'pid':os.getpid(),'url':f'http://127.0.0.1:{server.server_port}',
               'contract':backend.experience_contract(),'device':str(backend.device),
               'hip':backend.torch.version.hip,'base_fingerprint':base}
        save(root,'ready.json',ready);pipe.send(ready)
        sequence=0
        while True:
            require(pipe.poll(900),'child control wait expired; no restart')
            command=pipe.recv();sequence+=1
            save(root,f'command-{sequence:02}.json',command)
            if command['action']=='inspect':
                result=state_summary(runtime,backend,command['session_id'])
            elif command['action']=='close':
                require(not runtime._resident_session_ids,'all real actors must retire before close')
                after=base_fingerprint(backend,8*1024*1024)
                require(after==base,'child full base changed')
                result={'base_unchanged':True,'resident_ids':sorted(runtime._resident_session_ids)}
            else:raise ValueError('unsupported read-only control action')
            save(root,f'response-{sequence:02}.json',result);pipe.send(result)
            if command['action']=='close':break
        status['ok']=True
    except BaseException as exc:
        status['error']={'type':type(exc).__name__,'message':str(exc)}
        try:pipe.send({'child_error':status['error']})
        except BaseException:pass
    finally:
        try:
            if server is not None:server.shutdown();server.server_close()
            if thread is not None:thread.join(5)
            if runtime is not None:runtime.close()
        except BaseException as exc:status.update(ok=False,cleanup_error=str(exc))
        save(root,'worker-exit.json',status);pipe.close()
    raise SystemExit(0 if status['ok'] else 1)

def receive(pipe,seconds=300):
    require(pipe.poll(seconds),'child response timeout; unknown; do not retry')
    result=pipe.recv();require('child_error' not in result,str(result));return result

def inspect(pipe,sid):
    pipe.send({'action':'inspect','session_id':sid});return receive(pipe)

def http(client,root,label,method,path,body=None):
    save(root,label+'-intent.json',{'method':method,'path':path,'body':body,'mutation_retry_allowed':False})
    result=client._request(method,path,body)
    save(root,label+'-response.json',result);return result

def inference(client,root,label,sid,prompt,max_distance):
    request={'model':'reap','messages':[{'role':'user','content':prompt}],
             'n':1,'max_tokens':8,'temperature':0.0,'logprobs':True}
    outputs={}
    for kind in ('policy','value'):
        result=http(client,root,label+'-'+kind,'POST',f'/sessions/{sid}/{kind}/v1/chat/completions',request)
        require(type(result.get('policy_version')) is int and result['policy_version']==0,'actor local version changed')
        require(len(result['choices'])==1,'one actual inference result required')
        if kind=='value':
            score=json.loads(result['choices'][0]['message']['content'])['score']
            require(type(score) in (int,float) and math.isfinite(score) and 1<=score<=max_distance,'invalid categorical distance')
        else:
            require(type(result['choices'][0]['message']['content']) is str,'missing actual policy text')
            tokens=result['choices'][0].get('logprobs',{}).get('content',[])
            require(tokens and all(type(t.get('logprob')) in (int,float) and math.isfinite(t['logprob'])
                and t['logprob']<=0 for t in tokens),'actual finite policy token logprobs required')
        outputs[kind]=result
    return outputs

def run_joint(runtime,backend,learner,datasets,plan,campaign,config,output,implementation,factory=build_backend):
    """Actual provider, HTTP and independent processes; no search envelope."""
    children=[];scheduler=None
    try:
        learner.enable_release_head();provider=ReleasedAttemptProvider(learner)
        curriculum=deepcopy(campaign['curriculum'])
        for p in curriculum['problems']:p['family_id']='latest-'+p['family_id']
        scheduler=mm.Matchmaker.create(output/'reservations',curriculum,campaign['config'])
        def prepare(proposal):
            case=next(c for c in campaign['cases'] if c['problem_id']==proposal['problem_id'])
            require(case['desired_polarity']==proposal['polarity'],'reservation polarity mismatch')
            descriptor=deepcopy(case['prior_descriptor'])
            require(descriptor['budget_steps']==proposal['budget_steps'] and descriptor['attempted_prop_sha256']==proposal['attempted_prop_sha256'],'prior descriptor mismatch')
            return descriptor
        # An imported initial source is never a publication of the new run.
        try:provider(scheduler,prepare)
        except RuntimeError:pass
        else:raise AssertionError('unpublished own-run reservation was accepted')
        require(not scheduler.status()['attempts'],'unpublished check created an intent')
        ctx=mp.get_context('spawn')
        for index in range(2):
            root=output/f'service-{index}';root.mkdir()
            parent,remote=ctx.Pipe();process=ctx.Process(target=child,args=(remote,config,root,factory))
            process.start();remote.close();children.append((process,parent,root))
        ready=[receive(p) for _,p,_ in children]
        require(len({r['pid'] for r in ready}|{os.getpid()})==3,'three separate process IDs required')
        require(all(r['contract']==backend.experience_contract() for r in ready),'service contracts differ')
        base=runtime.actor.submit(lambda:base_fingerprint(backend,8*1024*1024))
        require(all(r['base_fingerprint']==base for r in ready),'independent base tensors differ')
        clients=[GpuHttpClient(r['url'],timeout_seconds=300) for r in ready]
        before=state_summary(runtime,backend,learner.learner_id)
        if 'source' in config:
            source,weights=learner.store.load_release(config['source']['model_release_sha256'])
            decoded=backend.torch.load(io.BytesIO(base64.b64decode(weights['payload'],validate=True)),map_location='cpu',weights_only=True)
            wanted={k:content_sha256(tree_digest(backend.torch,decoded[k])) for k in ('adapter','value_head')}
            require(before['parameters']==wanted and before['metadata']['policy_version']==0
                and before['optimizer_steps']==0 and before['optimizer_empty'],'source parameters/fresh learner state mismatch')
        first=learner.train_next_and_publish()
        old.validate_step(first['runtime_receipt'],plan['refs1'],datasets,1,backend._config())
        state1=state_summary(runtime,backend,learner.learner_id)
        save(output,'step1.json',first)
        intent1,selection1=provider(scheduler,prepare)
        validate_selection(selection1,intent1,learner.run_sha);save(output,'selection1.json',selection1);save(output,'intent1.json',intent1)
        require(selection1['head']==first['head_receipt'],'first reservation selected a different head')
        def create(index,intent,selection,expected):
            sid=intent['proposal']['attempt_id'];pin=intent['proposal']['model_release_sha256']
            release,_=learner.store.load_release(pin)
            require(release['run_sha256']==learner.run_sha and release['contract']==learner.run['contract'],'release ownership/contract mismatch')
            body={'theorem_id':intent['attempt_source']['execution_source_sha256'],
                  'model_release_sha256':pin,'expected_initialization_contract_sha256':hashlib.sha256(_json_bytes(learner.run['contract'])).hexdigest()}
            created=http(clients[index],output,f'actor{index+1}-create','POST',f'/sessions/{sid}',body)
            require(created['role']=='actor' and type(created['policy_version']) is int and created['policy_version']==0
                and created['session_id']==sid and created['theorem_id']==body['theorem_id'] and created['lineage']['model_release_sha256']==pin
                and created['initialization_contract']==learner.run['contract']
                and created['initialization_contract_sha256']==body['expected_initialization_contract_sha256'],'HTTP actor initialization mismatch')
            initial=inspect(children[index][1],sid)
            require(initial['parameters']==expected['parameters'] and initial['optimizer_empty']
                and initial['optimizer_steps']==0 and initial['examples_seen']==0,'actor parameters/private reset mismatch')
            logical=initial['metadata']
            require(logical['lineage']['source']==release['source'] and logical['lineage']['weights_sha256']==release['weights_sha256']
                and logical['lineage']['reset']==release['reset'] and logical['theorem_id']==body['theorem_id']
                and logical['event_receipts']=={} and logical['completed'] is False
                and logical['buffer_metadata']=={'events':{},'pending_event_ids':[],'consumed_event_ids':[]},'full actor lineage/reset mismatch')
            save(output,f'actor{index+1}-initial.json',initial)
            return sid,initial
        sid1,actor1_initial=create(0,intent1,selection1,state1)
        prompt=datasets['replay'][plan['initial_replay'][0]]['rows'][0]['prompt']
        inference(clients[0],output,'actor1',sid1,prompt,backend.max_distance)
        actor1_before=inspect(children[0][1],sid1);save(output,'actor1-before-second.json',actor1_before)
        # Release changes during preparation; provider must reserve the new head
        # without invoking preparation twice or changing the problem descriptor.
        second={};calls=[]
        def publish_during_prepare(proposal):
            calls.append(deepcopy(proposal));descriptor=prepare(proposal)
            second.update(learner.train_next_and_publish(append_dataset_pins=[plan['append_pin']]))
            return descriptor
        intent2,selection2=provider(scheduler,publish_during_prepare)
        require(len(calls)==1 and calls[0]['model_release_sha256']==first['release']['model_release_sha256'],'prepare was repeated or did not start at R1')
        old.validate_step(second['runtime_receipt'],plan['refs2'],datasets,2,backend._config())
        require(selection2['head']==second['head_receipt'] and selection2['head']['previous_head_sha256']==content_sha256(selection1['head']),'head chain/final reservation mismatch')
        save(output,'step2.json',second);save(output,'intent2.json',intent2);save(output,'selection2.json',selection2)
        state2=state_summary(runtime,backend,learner.learner_id)
        sid2,actor2_initial=create(1,intent2,selection2,state2)
        actor1_after=inspect(children[0][1],sid1);save(output,'actor1-after-second.json',actor1_after)
        require(actor1_before==actor1_after,'old actor complete private state changed during publication/new initialization')
        inference(clients[1],output,'actor2',sid2,prompt,backend.max_distance)
        cp2=learner.store.load_checkpoint(second['checkpoint_sha256'])
        require(cp2['manifest']['parent_checkpoint_sha256']==first['checkpoint_sha256'] and learner.sampler_state==plan['after2'],'checkpoint/sampler chain differs')
        require(cp2['data_receipt']['sampler_transition']['added_dataset_pins']==[plan['append_pin']],'append identity differs')
        require(all(before['parameters'][k]!=state1['parameters'][k]!=state2['parameters'][k] for k in ('adapter','value_head')),'both steps must really change parameters')
        require(runtime.actor.submit(lambda:base_fingerprint(backend,8*1024*1024))==base,'learner base changed')
        require(all(row['result'] is None for row in scheduler.status()['attempts'].values()),'mechanism reservations must not gain mathematical outcomes')
        for index,sid in enumerate((sid1,sid2)):
            receipt=http(clients[index],output,f'actor{index+1}-retire','POST',f'/sessions/{sid}/retire/v1',{'name':'mechanism-final','expected_policy_version':0})
            require(receipt['status']=='released' and receipt['session_id']==sid,'retirement not confirmed')
            children[index][1].send({'action':'close'});require(receive(children[index][1])['base_unchanged'],'child base gate failed')
        for process,pipe,root in children:
            process.join(30);require(process.exitcode==0,'child cleanup/exit failed');pipe.close()
            require(json.loads((root/'worker-exit.json').read_bytes())['ok'] is True,'child exit receipt failed')
        return {'ok':True,'first':first,'second':second,'release1':first['release'],'release2':second['release'],
            'sampler_after_second':learner.sampler_state,'service_ready':ready,
            'gates':{'two_real_commits_and_head_chain':True,'prepare_once_reselected_latest':True,
            'three_independent_processes_same_contract_and_base':True,'two_real_HTTP_create_policy_value':True,
            'R1_R2_exact_distinct_parameters':True,'old_actor_complete_state_unchanged':True,
            'new_actor_optimizer_local0_reset':True,'no_mathematical_outcome_fabricated':True,
            'both_children_retired_and_exit0':True,'full_frozen_base_exact_all_processes':True},
            'actor1_parameters':actor1_initial['parameters'],'actor2_parameters':actor2_initial['parameters']}
    finally:
        if scheduler is not None:scheduler.close()
        for process,pipe,_ in children:
            if process.is_alive():process.terminate();process.join(10)
            pipe.close()

def parser():
    p=old.build_parser();p.description=__doc__
    p.add_argument('--run',action='store_true');p.add_argument('--campaign-plan',type=Path,required=True)
    p.add_argument('--campaign-plan-sha256',required=True)
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    if not args.run:
        print(json.dumps({'run':False,'note':'No GPU allocation or mutation; add --run explicitly'}));return 0
    raw=args.campaign_plan.read_bytes();require(hashlib.sha256(raw).hexdigest()==args.campaign_plan_sha256,'campaign metadata pin differs')
    campaign=json.loads(raw)
    source_root=args.source_release_root.absolute();output=args.output_dir.absolute()
    require(source_root!=output and source_root not in output.parents and output not in source_root.parents,'output overlaps source')
    old_files=old.inventory(source_root);source,weights=ContinualMixedStore(source_root).load_release(args.initial_model_release_sha256)
    require(source['contract']['base_sha256']==args.expected_base_sha256,'base pin differs')
    datasets,input_before,plan=old.prepare_inputs(args,source)
    output.mkdir(parents=True,exist_ok=False)
    # Independent copies: old source CAS never receives a new write/hardlink.
    store=output/'learner-store';store.mkdir()
    for name in old_files:
        dest=store/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with (source_root/name).open('rb') as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
    require(old.inventory(store)==old_files,'source copy exact hash failed')
    source_before=mixed.source_hashes()
    root=Path(mixed.__file__).resolve().parents[2]
    source_before['containers/gpu/smoke_continual_mixed_learner.py']=hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest()
    for path in (root/'cpu_runtime').glob('*.py'):source_before[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    source_before['runner']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config={'model_path':args.model_path,'replay_dataset_root':str(args.replay_dataset_root),
        'mathlib_dataset_root':str(args.mathlib_dataset_root),'source':source,'store':str(store)}
    save(output,'input-manifest.json',{'datasets':input_before,'source_files':old_files,'source_release':source,
        'implementation':source_before,'sampling_plan':plan,'campaign_plan_sha256':args.campaign_plan_sha256,'command':[sys.executable,*sys.argv]})
    runtime=learner=None
    report={'schema_version':'reap.latest-release.mechanism.v1','ok':False,'real_7B_GPU_gate_passed':False,
        'new_Lean_search':False,'mathematical_verification_performed':False,'ReplicaCollector_search_or_batch_executed':False,
        'curriculum_generation':False,'performance_claimed':False,'mutation_retry_allowed':False,
        'unknown_publication_fault_injected':False,'timing_started_epoch':time.time()}
    try:
        backend=build_backend(config)
        runtime=GpuRuntime(backend=backend,snapshot_root=output/'snapshots',learner_release_root=store,
                           learner_profile='continual-mixed-v3',max_resident_sessions=1)
        learner=ContinualMixedLearner(runtime,learner_id='latest-mechanism-central',replay_pins=plan['initial_replay'],
            mathlib_sft_pins=plan['human'],sampler_seed=plan['config']['seed'],journal_root=output/'journal',
            initial_model_release_sha256=source['model_release_sha256'],implementation=source_before)
        report.update(run_joint(runtime,backend,learner,datasets,plan,campaign,config,output,source_before))
        _,after,afterplan=old.prepare_inputs(args,source)
        require(after==input_before and afterplan==plan and old.inventory(source_root)==old_files,'input/source integrity changed')
        for name,digest in source_before.items():
            path=Path(__file__) if name=='runner' else root/name
            require(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'implementation changed')
        report['real_7B_GPU_gate_passed']=True
    except BaseException as exc:old.fail(report,'error',exc)
    finally:
        try:
            if learner is not None:learner.close()
            if runtime is not None:runtime.close()
        except BaseException as exc:old.fail(report,'cleanup_error',exc)
    report['timing_finished_epoch']=time.time()
    return old.finalize(output,report)

if __name__=='__main__':raise SystemExit(main())
