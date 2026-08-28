"""Prepare complete evidence datasets and run explicit, bounded TTT checks.

The default preparation is offline. GPU modes require --run; they never download
weights, restart an existing output, or send anything to GitHub.
"""
import argparse,hashlib,importlib.util,json,os,re,subprocess,sys,zipfile
from pathlib import Path,PurePosixPath

_spec=importlib.util.spec_from_file_location('_reap_delivery_verify',Path(__file__).with_name('verify.py'))
_verify=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_verify)

PACKAGE=Path(__file__).resolve().parents[2]
BASE='2ff73d37f6f4edad02f5c2e67bdabeeecef97a187b4747834e6acdf648980839'
REPLAY=['b89032bee7d7d75449f5395e9c34c99ff538624bd21abc057ae4be76ae8441eb','7975e11a24d74f326387e82fd4b8f9fc2489d1d570e44c17c5f8a03e1b9555ff','179c51f5f802311f9d1f292902fb4f0d452159dbe33afd6e14d823cae948fa0a','fae47cd667809fa30c7a87b782b98e2e4943ba47144ddc69eee930142552402e']
HUMAN=['633da6857da2df468f5d9e6c2aee30f0b424505d95ee4c94cbbfe575fd203b5d','98c67e8f220e8f5a9227ce010a95792f838b10a9cee6f15a98f1eae695d788cb','390db65cb72310385919dfe5207659c51614962420b9a5ff39ec14fe42d874d1']
APPEND='33a3be06d72cacd9dfa78c5e3f82465f3e1f6ccbf12506fad67fd423a93114ad'
sha=lambda b:hashlib.sha256(b).hexdigest()

def evidence(stage):
 root=PACKAGE/'evidence/current'/stage
 manifest=json.loads((root/'manifest.json').read_bytes())
 raw=(root/'raw.zip').read_bytes()
 if len(raw)!=manifest['archive']['bytes'] or sha(raw)!=manifest['archive']['sha256']:raise ValueError('archive mismatch '+stage)
 return _verify.checked_zip(raw,manifest['files'],2*1024*1024)

def prepare(destination):
 destination=_verify.no_links(destination)
 if destination.exists() or destination.is_symlink():raise FileExistsError('use a new dataset directory')
 groups=[('verified-replay','frozen/datasets/','replay'),('central-learner','verified-actor-01-host-recovery/registry/','replay'),('mixed-learner','frozen-data/mathlib-datasets/','mathlib'),('collector-r3','collected/volume/dataset-store/','replay')]
 payload={}
 for stage,prefix,kind in groups:
  for name,b in evidence(stage).items():
   if name.startswith(prefix):
    suffix=name[len(prefix):];parts=PurePosixPath(suffix).parts
    if len(parts)!=2 or not re.fullmatch('[0-9a-f]{64}',parts[0]):continue
    dest=kind+'/'+suffix
    if dest in payload and payload[dest]!=b:raise ValueError('conflicting dataset member')
    payload[dest]=b
 destination.mkdir(parents=True,exist_ok=False)
 _verify.no_links(destination)
 for name,b in payload.items():
  p=destination/name;p.parent.mkdir(parents=True,exist_ok=True);_verify.no_links(p)
  with p.open('xb') as stream:stream.write(b)
 result=check(destination)
 (destination/'prepared.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 return result

def check(root):
 from cpu_runtime.verified_trajectory import load_verified_dataset
 from cpu_runtime.mathlib_trajectory import load_mathlib_dataset
 result={'offline_integrity_and_strict_loader':True,'new_Lean_or_GPU_run':False,'datasets':{}}
 for kind,loader in [('replay',load_verified_dataset),('mathlib',load_mathlib_dataset)]:
  for p in sorted((Path(root)/kind).iterdir()):
   if p.is_dir():
    value=loader(p,expected_sha256=p.name)
    result['datasets'][p.name]={'kind':kind,'rows':len(value['rows']),'files':len(list(p.iterdir()))}
 for pin in [*REPLAY,*HUMAN,APPEND]:
  if pin not in result['datasets']:raise ValueError('required pinned dataset missing '+pin)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('mode',choices=['prepare','check','local-tests','mixed','continual'])
 p.add_argument('--source-root',type=Path,required=True)
 p.add_argument('--data-root',type=Path)
 p.add_argument('--output',type=Path)
 p.add_argument('--model-path',type=Path)
 p.add_argument('--source-release-root',type=Path)
 p.add_argument('--source-release-sha256')
 p.add_argument('--run',action='store_true',help='execute GPU mode; otherwise print exact command only')
 args=p.parse_args();source=args.source_root.resolve(strict=True)
 sys.path.insert(0,str(source))
 if args.mode=='local-tests':
  return subprocess.call([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_*.py'],cwd=source)
 if args.data_root is None:p.error('--data-root required')
 data=args.data_root.absolute()
 if args.mode=='prepare':print(json.dumps(prepare(data),indent=2));return 0
 if args.mode=='check':print(json.dumps(check(data),indent=2));return 0
 check(data)
 if args.model_path is None or args.output is None:p.error('--model-path and --output required for GPU modes')
 if not args.model_path.is_dir():p.error('existing model directory required; no download')
 output=args.output.absolute()
 if output.exists() or output.is_symlink():p.error('output already exists: inspect it, do not replay training')
 _verify.no_links(output)
 intent=output.with_name(output.name+'.launch-intent.json')
 exit_path=output.with_name(output.name+'.exit.json')
 for marker in (intent,exit_path):
  _verify.no_links(marker)
  if marker.exists():p.error('existing launch evidence: inspect it, do not replay training')
 module='smoke_mixed_learner' if args.mode=='mixed' else 'smoke_continual_mixed_learner'
 cmd=[sys.executable,'-B','-m','containers.gpu.'+module,'--model-path',str(args.model_path.resolve()),'--expected-base-sha256',BASE,'--replay-dataset-root',str(data/'replay'),'--mathlib-dataset-root',str(data/'mathlib'),'--output-dir',str(output),'--sampler-seed','0']
 for pin in REPLAY:cmd.extend(['--replay-dataset-sha256',pin])
 for pin in HUMAN:cmd.extend(['--mathlib-dataset-sha256',pin])
 if args.mode=='mixed':cmd.extend(['--max-distance','8'])
 else:
  if args.source_release_root is None or not re.fullmatch('[0-9a-f]{64}',args.source_release_sha256 or ''):p.error('continual requires explicit source release root and SHA from the preceding report')
  cmd.extend(['--source-release-root',str(args.source_release_root.resolve(strict=True)),'--initial-model-release-sha256',args.source_release_sha256,'--append-replay-dataset-sha256',APPEND])
 print(json.dumps({'command':cmd,'cwd':str(source),'execute':args.run},indent=2),flush=True)
 if not args.run:return 0
 intent.parent.mkdir(parents=True,exist_ok=True)
 _verify.no_links(intent)
 with intent.open('x') as f:
  json.dump({'command':cmd,'cwd':str(source),'automatic_retry_allowed':False},f);f.flush();os.fsync(f.fileno())
 rc=subprocess.call(cmd,cwd=source)
 with exit_path.open('x') as f:json.dump({'returncode':rc},f)
 return rc

if __name__=='__main__':raise SystemExit(main())

