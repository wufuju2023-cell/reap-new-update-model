"""Recompile complete original proof bodies offline, without GPU or training."""
import argparse,hashlib,json,re,subprocess,time
from pathlib import Path
from verify_package import verify

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True,help='your actual locally built CPU image ID')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--all',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];verify(root)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    files=sorted((root/'proofs').glob('*.lean')) if args.all else [root/'proofs/07-original-target.lean']
    results=[]
    for proof in files:
        command=['podman','run','--rm','--pull','never','--network','none',
          '--volume',str(root/'proofs')+':/proof:ro','--workdir','/opt/reap-runtime',
          '--entrypoint','lake',args.image,'env','lean','/proof/'+proof.name]
        start=time.monotonic();run=subprocess.run(command,capture_output=True,text=True)
        (out/(proof.stem+'.stdout')).write_text(run.stdout,encoding='utf-8')
        (out/(proof.stem+'.stderr')).write_text(run.stderr,encoding='utf-8')
        axioms=re.findall(r'depends on axioms:\s*\[([^\]]*)\]',run.stdout)
        assert run.returncode==0 and axioms,'Lean failed; inspect output'
        names={s.strip() for s in axioms[-1].split(',') if s.strip()}
        assert names<={'propext','Classical.choice','Quot.sound'},names
        results.append({'proof':proof.name,'sha256':hashlib.sha256(proof.read_bytes()).hexdigest(),
          'returncode':run.returncode,'axioms':sorted(names),'seconds':time.monotonic()-start})
    (out/'receipt.json').write_text(json.dumps({'image':args.image,'network':'none','results':results},indent=2)+'\n')
    print(json.dumps(results))

if __name__=='__main__':main()
