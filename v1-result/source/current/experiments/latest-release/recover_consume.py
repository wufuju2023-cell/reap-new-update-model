"""Explicit recovery-only consumption of an already published R2; never train.

The original failed run stays failed. Restore the original R1 full snapshot in
a new process; do not describe that as uninterrupted actor lifetime. Original
reservation identities/receipts are read, never rewritten or completed.
"""
from __future__ import annotations
import argparse,base64,hashlib,io,json,multiprocessing as mp,os,threading,time
from pathlib import Path
from http.server import ThreadingHTTPServer
import latest_probe as shared
from gpu_runtime import GpuRuntime
from gpu_runtime.server import RuntimeHandler
from gpu_runtime.snapshot_store import SnapshotStore,_json_bytes
from gpu_runtime.continual_mixed_store import ContinualMixedStore
from gpu_runtime.learner_release_store import canonical_bytes,content_sha256
from gpu_runtime.release_head import _validate_confirmation
from cpu_runtime.released_attempts import validate_selection
from cpu_runtime.http_clients import GpuHttpClient
from cpu_runtime.verified_dataset_store import safe_directory

require=shared.require;save=shared.save

def read(path):return json.loads(Path(path).read_bytes())

def pinned(path,pin):
    raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==pin,'pinned file differs: '+str(path));return json.loads(raw)

def child(pipe,config,root,factory=shared.build_backend):
    root=Path(root);runtime=server=thread=None;status={'ok':False,'pid':os.getpid()}
    try:
        backend=factory(config)
        runtime=GpuRuntime(backend=backend,snapshot_root=root/'snapshots',learner_release_root=config['store'],
            learner_profile='continual-mixed-v3',max_resident_sessions=1)
        base=shared.base_fingerprint(backend,8*1024*1024)
        Handler=type('RecoveryHandler',(RuntimeHandler,),{'runtime':runtime,'backend_name':'mixed-replay'})
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        ready={'pid':os.getpid(),'url':f'http://127.0.0.1:{server.server_port}','contract':backend.experience_contract(),
            'base_fingerprint':base,'device':str(backend.device),'hip':backend.torch.version.hip,'control_timeout_seconds':1800}
        save(root,'ready.json',ready);pipe.send(ready)
        seq=0
        while True:
            require(pipe.poll(1800),'recovery child control timeout; no retry')
            command=pipe.recv();seq+=1;save(root,f'command-{seq:02}.json',command)
            if command['action']=='inspect':result=shared.state_summary(runtime,backend,command['session_id'])
            elif command['action']=='close':
                require(not runtime._resident_session_ids,'recovery actor still resident')
                require(shared.base_fingerprint(backend,8*1024*1024)==base,'recovery base changed')
                result={'base_unchanged':True,'resident_ids':[]}
            else:raise ValueError('unsupported control action')
            save(root,f'response-{seq:02}.json',result);pipe.send(result)
            if command['action']=='close':break
        status['ok']=True
    except BaseException as exc:
        status.update(error={'type':type(exc).__name__,'message':str(exc)})
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

def preflight(args):
    require(os.name=='posix','actual recovery entry requires Linux')
    original=args.original_run.absolute();result=original/'result'
    report=pinned(result/'report.json',args.original_report_sha256)
    require(report['ok'] is False and report['real_7B_GPU_gate_passed'] is False,'original run must explicitly have failed')
    outer=read(original/'worker-exit.json')
    require(outer['returncode']!=0 and type(outer['probe_returncode']) is int and outer['probe_returncode']!=0,'original main exit not confirmed')
    oldpids=[read(original/'probe-launch.json')['pid']]+[read(result/f'service-{i}/ready.json')['pid'] for i in range(2)]
    require(all(type(pid) is int and pid>0 and not Path('/proc',str(pid)).exists() for pid in oldpids),'original process remains; do not recover alongside it')
    inp=read(result/'input-manifest.json');runsha=read(result/'step1.json')['release']['run_sha256']
    headpath=result/'learner-store/control'/runsha/'release-head/published-00000002.json'
    head=pinned(headpath,args.head_sha256)
    selection=pinned(result/'selection2.json',args.selection_sha256);intent=read(result/'intent2.json')
    validate_selection(selection,intent,runsha)
    require(selection['head']==head and head['step']==2,'original second selection/head mismatch')
    store=ContinualMixedStore(result/'learner-store')
    release,weights=store.load_release(head['model_release_sha256'])
    _validate_confirmation(store,runsha,{k:head[k] for k in ('schema_version','run_sha256','step','checkpoint_sha256','previous_head_sha256')},head,verify_release=False)
    require(release['run_sha256']==runsha and release['source']['checkpoint_sha256']==head['checkpoint_sha256']
        and release['source']['learner_step']==2,'release/head source mismatch')
    cp=store.load_checkpoint(head['checkpoint_sha256'])
    require(cp['manifest']['step']==2 and cp['manifest']['parent_checkpoint_sha256']==read(result/'step1.json')['checkpoint_sha256'],'committed CP2 parent differs')
    baseline=read(result/'actor1-before-second.json')
    require(baseline['complete_sha256']==args.expected_r1_complete_sha256,'R1 original baseline differs')
    sid=baseline['metadata']['session_id'];snapshot=result/'service-0/snapshots'/sid/args.snapshot_name
    require(hashlib.sha256((snapshot/'manifest.json').read_bytes()).hexdigest()==args.snapshot_manifest_sha256,'safety snapshot manifest differs')
    logical,backend=SnapshotStore(result/'service-0/snapshots').load(sid,args.snapshot_name)
    import torch
    decode=lambda raw:torch.load(io.BytesIO(base64.b64decode(raw['payload'],validate=True)),map_location='cpu',weights_only=True)
    payload=decode(backend)
    require(logical==baseline['metadata'] and content_sha256(shared.tree_digest(torch,{'backend':payload,'metadata':logical}))==baseline['complete_sha256'],'safety snapshot is not exact original complete private state')
    cpstate=decode(cp['backend_state']);released=decode(weights)
    parameters=lambda s:{k:content_sha256(shared.tree_digest(torch,s[k])) for k in ('adapter','value_head')}
    require(parameters(cpstate)==parameters(released),'published R2 parameters differ from actual CP2')
    return {'original':original,'result':result,'input':inp,'head':head,'selection':selection,'intent':intent,
        'release':release,'baseline':baseline,'snapshot':snapshot,'snapshot_name':args.snapshot_name,
        'R2_parameters':parameters(cpstate),'store':store,'source_files':shared.old.inventory(store.root),
        'original_pids':oldpids}

def run_recovery(data,output,config,factory=shared.build_backend):
    children=[]
    def stage(name,value=None):save(output,'stage-'+name+'.json',{'epoch':time.time(),'detail':value})
    try:
        stage('00-preflight-complete',{'head':data['head'],'original_run_preserved_failed':True})
        for index in range(2):
            root=output/f'service-{index}';root.mkdir()
            if index==0:
                dest=root/'snapshots'/data['baseline']['metadata']['session_id']/data['snapshot_name'];dest.mkdir(parents=True)
                with safe_directory(data['snapshot']) as source:
                    for name in ('manifest.json','session.json','backend.json'):
                        raw=source.read(name)
                        with (dest/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
                require(shared.old.inventory(dest)==shared.old.inventory(data['snapshot']),'copied R1 snapshot differs')
            parent,remote=mp.get_context('spawn').Pipe();process=mp.get_context('spawn').Process(target=child,args=(remote,config,root,factory))
            process.start();remote.close();children.append((process,parent,root))
        ready=[shared.receive(pipe,seconds=600) for _,pipe,_ in children]
        require(len({r['pid'] for r in ready})==2 and all(r['pid'] not in data['original_pids'] for r in ready),'new independent recovery processes required')
        require(ready[0]['base_fingerprint']==ready[1]['base_fingerprint'] and all(r['contract']==data['release']['contract'] for r in ready),'recovery base/contract mismatch')
        stage('01-services-ready',ready)
        clients=[GpuHttpClient(r['url'],timeout_seconds=600) for r in ready]
        old=data['baseline']['metadata'];sid1=old['session_id'];sid2='latest-recovery-r2-actor'
        def create(index,sid,theorem,pin):
            body={'theorem_id':theorem,'model_release_sha256':pin,
                'expected_initialization_contract_sha256':hashlib.sha256(_json_bytes(data['release']['contract'])).hexdigest()}
            created=shared.http(clients[index],output,f'actor{index+1}-create','POST',f'/sessions/{sid}',body)
            require(created['session_id']==sid and created['role']=='actor' and created['policy_version']==0
                and created['theorem_id']==theorem and created['lineage']['model_release_sha256']==pin
                and created['initialization_contract']==data['release']['contract']
                and created['initialization_contract_sha256']==body['expected_initialization_contract_sha256'],'recovery create contract/identity differs')
        create(0,sid1,old['theorem_id'],old['lineage']['model_release_sha256'])
        restored=shared.http(clients[0],output,'actor1-full-restore','POST',f'/sessions/{sid1}/restore/v1',{'name':data['snapshot_name']})
        require(restored==old,'full restore metadata differs')
        before=shared.inspect(children[0][1],sid1);save(output,'actor1-restored-before.json',before)
        require(before==data['baseline'],'R1 full private restoration differs from original baseline')
        stage('02-original-R1-restored',{'complete_sha256':before['complete_sha256']})
        create(1,sid2,data['intent']['attempt_source']['execution_source_sha256'],data['head']['model_release_sha256'])
        second=shared.inspect(children[1][1],sid2);save(output,'actor2-initial.json',second)
        require(second['parameters']==data['R2_parameters'] and second['optimizer_empty'] and second['optimizer_steps']==0
            and second['examples_seen']==0 and second['metadata']['lineage']['source']==data['release']['source'],'R2 actual parameters/reset mismatch')
        prompt='User: STATE:\n⊢ True\nTACTIC:'
        shared.inference(clients[1],output,'actor2',sid2,prompt,data['release']['contract']['mixed_config']['support']['distance_max'])
        after=shared.inspect(children[0][1],sid1);save(output,'actor1-restored-after-R2.json',after)
        require(after==before,'restored original R1 changed during R2 consumption')
        stage('03-R2-consumed-original-R1-preserved')
        for index,sid in enumerate((sid1,sid2)):
            receipt=shared.http(clients[index],output,f'actor{index+1}-retire','POST',f'/sessions/{sid}/retire/v1',{'name':'recovery-final','expected_policy_version':0})
            require(receipt['status']=='released' and receipt['session_id']==sid,'recovery retirement not confirmed')
            children[index][1].send({'action':'close'});require(shared.receive(children[index][1],600)['base_unchanged'],'base changed')
        for process,pipe,root in children:
            process.join(30);require(process.exitcode==0 and read(root/'worker-exit.json')['ok'] is True,'recovery child exit failed')
        stage('04-retired-and-closed')
        return {'ok':True,'R1_full_private_state_restoration':True,'head':data['head'],'release':data['release'],'service_ready':ready,
            'restored_original_complete_sha256':before['complete_sha256'],'R2_parameters':second['parameters'],
            'gates':{'original_cp2_release_parameters_exact':True,'original_head_selection_bound':True,
                'original_R1_complete_private_state_restored':True,'two_new_independent_7B_processes':True,
                'actual_R2_HTTP_policy_value':True,'restored_R1_unchanged_during_R2_consumption':True,
                'both_retired_and_child_exit0':True,'both_base_fingerprints_unchanged':True}}
    finally:
        for process,pipe,_ in children:
            if process.is_alive():process.terminate();process.join(10)
            pipe.close()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original-run','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('original-report-sha256','head-sha256','selection-sha256','snapshot-name','snapshot-manifest-sha256','expected-r1-complete-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--run',action='store_true');args=p.parse_args(argv)
    if not args.run:print(json.dumps({'run':False,'GPU_allocated':False}));return 0
    output=args.output_dir.absolute();require(not output.exists(),'new recovery output required')
    data=preflight(args);output.mkdir(parents=True,exist_ok=False)
    original_command=read(data['original']/'actual-command.json')
    option=lambda name:original_command[original_command.index('--'+name)+1]
    config={'model_path':option('model-path'),'replay_dataset_root':option('replay-dataset-root'),
            'mathlib_dataset_root':option('mathlib-dataset-root'),'source':data['release'],'store':str(data['store'].root)}
    code={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__),Path(shared.__file__))}
    save(output,'input-manifest.json',{'args':{k:str(v) for k,v in vars(args).items()},'code':code,
        'original_failed_report_sha256':args.original_report_sha256,'source_files':data['source_files'],
        'head':data['head'],'selection':data['selection'],'original_snapshot_manifest_sha256':args.snapshot_manifest_sha256})
    report={'schema_version':'reap.latest-release.recovery-consumption.v1','ok':False,'real_7B_GPU_gate_passed':False,
        'new_training_steps':0,'publish_calls':0,'learner_restore_performed':False,'original_run_still_failed':True,
        'original_actor_uninterrupted':False,'R1_full_private_state_restoration':False,
        'new_Lean_search':False,'new_mathematical_verification':False,'ReplicaCollector_batch_executed':False,
        'automatic_retry_allowed':False}
    try:
        report.update(run_recovery(data,output,config))
        require(shared.old.inventory(data['store'].root)==data['source_files'],'original CAS changed during read-only recovery')
        require(all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in code.items()),'recovery source changed')
        report['real_7B_GPU_gate_passed']=True
    except BaseException as exc:shared.old.fail(report,'error',exc)
    return shared.old.finalize(output,report)

if __name__=='__main__':raise SystemExit(main())
