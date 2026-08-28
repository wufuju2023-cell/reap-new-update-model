"""Explicit operator assertion after preflight; never calls the GPU."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-root',type=Path,required=True)
p.add_argument('--operator',required=True)
p.add_argument('--confirm-ready',action='store_true',required=True)
a=p.parse_args();root=a.run_root.resolve()
family=json.loads((root/'portable-run.json').read_bytes())['family']
wrapper=Path(__file__).resolve().parent/'cpu.py'
result=subprocess.run([sys.executable,str(wrapper),'--run-root',str(root),'approval-template','--family',family],capture_output=True,text=True,check=True)
value=json.loads(result.stdout);assert value['ready'] is False
value.update(ready=True,operator=a.operator)
with (root/'approval.json').open('x') as f:json.dump(value,f,indent=2)
print('Approval created for this exact plan. It does not replace successful preflight.')
