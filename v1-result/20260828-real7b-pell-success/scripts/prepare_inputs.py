"""Make a NEW portable CPU run directory; never call a GPU endpoint."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from verify_package import verify

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lesson',choices=['indexed-witness','unbounded-sequence','original-target'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family-id',required=True)
    p.add_argument('--cpu-image',required=True)
    p.add_argument('--gpu-url',required=True)
    p.add_argument('--contract',type=Path,required=True,help='actual new server initialization-contract.json')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--seed-metadata',type=Path)
    group.add_argument('--fresh-base',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];verify(root)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    template=root/'inputs'/a.lesson
    shutil.copytree(root/'code',out/'code')
    shutil.copytree(template/'student',out/'student')
    if (template/'student-proof-source').exists():shutil.copytree(template/'student-proof-source',out/'student-proof-source')
    record=json.loads((template/'course-plan.template.json').read_bytes())
    record['problems'][0]['id']=a.family_id
    for h in record.get('student_proof_library',[]):
        for k in ('acceptance_path','proof_path'):
            h[k]=str((out/'student'/h[k]).resolve())
        h['acceptance_sha256']=digest(Path(h['acceptance_path']))
    course=out/'student/course_plan.json';save(course,record)
    visibility=json.loads((out/'student/visibility.json').read_bytes())
    visibility['course_record_sha256']=digest(course);save(out/'student/visibility.json',visibility)
    image=a.cpu_image.removeprefix('sha256:')
    assert len(image)==64 and all(c in '0123456789abcdef' for c in image),'use actual image ID, not tag'
    premise=out/'student/premise_plan.json'
    if premise.exists():
        data=json.loads(premise.read_bytes())
        data.update(course_record_sha256=digest(course),environment_sha256=image,cpu_image=image)
        for key in data['lessons']:data['lessons'][key]=record.get('student_proof_library',[])
        save(premise,data)
    runtime=json.loads((template/'historical-runtime.json').read_bytes())
    contract=json.loads(a.contract.read_bytes())
    assert contract['backend']=='real-search' and contract['search_config']['gamma']==.99
    runtime.update(cpu_image=image,environment_sha256=image,gpu_base_url=a.gpu_url,
      initialization_contract=contract,base_sha256=contract['base_sha256'])
    save(out/'runtime.json',runtime)
    driver=out/'code/experiments/proof-curriculum/runner/course_driver.py'
    command=[sys.executable,str(driver),'prepare','--source-root',str(out/'code'),
      '--store',str(out/'store'),'--family-id',a.family_id,'--course-record',str(course),
      '--budget-file',str(out/'student/budgets.json'),'--runtime',str(out/'runtime.json'),
      '--budget','probe','--exhausted-policy','stop','--visibility-plan',str(out/'student/visibility.json')]
    if premise.exists():command+=['--premise-plan',str(premise)]
    if a.seed_metadata:command+=['--seed-metadata',str(a.seed_metadata.resolve())]
    # Preparation checks /opt/reap-runtime; run it in the fixed CPU environment.
    subprocess.run([sys.executable,str(root/'scripts/cpu.py'),'--run-root',str(out)]+command[2:],check=True)
    save(out/'portable-run.json',{'family':str(out/'store'/a.family_id),'mode':'fresh-base-new-experiment' if a.fresh_base else 'published-parameter-inheritance','gpu_started':False})

if __name__=='__main__':main()
