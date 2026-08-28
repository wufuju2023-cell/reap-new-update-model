"""Run the unchanged course driver inside the user's fixed CPU image."""
import argparse,json,subprocess,sys,uuid
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-root',type=Path,required=True)
p.add_argument('driver_args',nargs=argparse.REMAINDER)
a=p.parse_args();root=a.run_root.resolve()
runtime=json.loads((root/'runtime.json').read_bytes())
command=['podman','run','--name','pell-course-'+uuid.uuid4().hex[:12],
 '--pull','never','--network','host','--userns','keep-id:uid=10001,gid=10001',
 '--user','10001:10001','--volume',str(root)+':'+str(root)+':rw',
 '--volume',str(root/'code')+':'+str(root/'code')+':ro',
 '--workdir','/opt/reap-runtime','--entrypoint','python3',runtime['cpu_image'],
 '-B',str(root/'code/experiments/proof-curriculum/runner/course_driver.py')]+a.driver_args
sys.exit(subprocess.call(command))
