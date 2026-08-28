"""Bounded read-only AMD status via Edge/OpenCLI, with one fresh-tab fallback.

No arbitrary command, HTTP mutation, instance start/stop, model load, or learning
is exposed. A timed-out status read can finish later without changing the run.
This shorter deadline must never be used for the training transport.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time

from .http_bridge import OpenCliTransport

READ_TIMEOUT = 25
OPEN_TIMEOUT = 30


def status_source(root: str) -> str:
    path = PurePosixPath(root)
    if (not root.startswith('/mnt/workspace/') or path.as_posix() != root
            or '..' in path.parts or not re.fullmatch(r'/[A-Za-z0-9_./-]+', root)):
        raise ValueError('explicit canonical run directory under /mnt/workspace required')
    return f'''import json,time
from pathlib import Path
r=Path({root!r})
if not r.is_dir() or r.is_symlink():raise ValueError('verified run directory missing')
result={{'schema_version':'reap.amd.read-only-status.v1','epoch':time.time(),'root':str(r),'gpu_busy_percent':{{}},'processes':[],'job_roots':{{}}}}
for p in Path('/sys/class/drm').glob('card*/device/gpu_busy_percent'):
 result['gpu_busy_percent'][str(p)]=int(p.read_text())
for p in r.glob('*.pid'):
 if not p.is_file() or p.is_symlink() or p.stat().st_size>32:continue
 text=p.read_text().strip()
 if not text.isdigit():continue
 proc=Path('/proc')/text
 row={{'pid_file':p.name,'pid':int(text),'exists':proc.exists()}}
 if proc.exists():
  try:
   argv=(proc/'cmdline').read_bytes().split(b'\\0')
   row['kind']=next((x.decode() for x in argv if x in (b'gpu_runtime.server',b'containers.gpu.smoke_experience_gpu',b'containers.gpu.smoke_search_gpu')), 'other')
  except OSError:row['kind']='exited_during_read'
 result['processes'].append(row)
for jobs in r.glob('*http-jobs'):
 if not jobs.is_dir() or jobs.is_symlink():continue
 counts={{'done':0,'unknown':0,'pending':0,'invalid':0,'last_completed_gpu_request_epoch':None}}
 for request in jobs.glob('*/request.json'):
  if request.parent.is_symlink() or request.is_symlink() or request.stat().st_size>65536:counts['invalid']+=1;continue
  try:
   q=json.loads(request.read_text());p=request.parent/'result.json'
   if not p.exists():counts['pending']+=1;continue
   if p.is_symlink() or p.stat().st_size>16384:counts['invalid']+=1;continue
   v=json.loads(p.read_text());state=v.get('state')
   if state not in ('done','unknown'):counts['invalid']+=1;continue
   counts[state]+=1
   # Health/read-only polling and snapshots never reset the GPU idle clock.
   if state=='done' and v.get('status') in (200,201) and any(q.get('path','').endswith(a) for a in ('/learn/v1','/policy/v1/chat/completions','/value/v1/chat/completions')):
    counts['last_completed_gpu_request_epoch']=max(counts['last_completed_gpu_request_epoch'] or 0,v.get('completed_at',0))
  except (OSError,ValueError,TypeError):counts['invalid']+=1
 result['job_roots'][jobs.name]=counts
print(json.dumps(result))
'''


def run_probe(transport: OpenCliTransport, *, remote_root: str, recovery_session: str,
              runner=subprocess.run, clock=time.time) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', recovery_session) or recovery_session == transport.browser_session:
        raise ValueError('a distinct explicit recovery browser session is required')
    code = 'import base64;exec(base64.b64decode(' + repr(base64.b64encode(status_source(remote_root).encode()).decode()) + '))'
    expression = transport._expression(shlex.join(['python3', '-c', code]))
    expression = expression.replace('AbortSignal.timeout(20000)', 'AbortSignal.timeout(10000)')
    attempts = []

    def call(session, arguments, timeout):
        with transport._locked():
            result = runner([transport.node, transport.cli, '--profile', transport.profile,
                             'browser', session, *arguments], capture_output=True,
                            text=True, encoding='utf-8', timeout=timeout)
        if result.returncode:
            # Keep only structured error class; no raw console/argv/credentials.
            try:
                code = json.loads(result.stdout).get('error', {}).get('code', 'cli_error')
            except (ValueError, AttributeError):
                code = 'cli_error'
            raise RuntimeError(code if re.fullmatch('[a-z_]{1,50}', str(code)) else 'cli_error')
        return json.loads(result.stdout)

    for index, session in enumerate((transport.browser_session, recovery_session)):
        start = clock()
        try:
            if index:
                call(session, ['open', 'https://www.modelscope.cn/code/workspace'], OPEN_TIMEOUT)
            envelope = call(session, ['eval', expression], READ_TIMEOUT)
            if envelope.get('status') != 200:
                raise RuntimeError('command_response_error')
            status = json.loads(json.loads(envelope['text'])['output'])
            if status.get('schema_version') != 'reap.amd.read-only-status.v1' or status.get('root') != remote_root:
                raise RuntimeError('status_identity_error')
            attempts.append({'session': session, 'ok': True, 'elapsed_seconds': clock() - start})
            return {'ok': True, 'instance_id': transport.instance_id, 'browser_session': session,
                    'read_only': True, 'attempts': attempts, 'status': status,
                    'note': 'Status alone does not authorize retrying an unresolved mutation or prove idle duration.'}
        except (subprocess.TimeoutExpired, RuntimeError, ValueError, KeyError, OSError, TypeError, AttributeError) as exc:
            attempts.append({'session': session, 'ok': False, 'elapsed_seconds': clock() - start,
                             'error': str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__})
    return {'ok': False, 'instance_id': transport.instance_id, 'read_only': True,
            'attempts': attempts, 'status': None, 'idle_confirmed': False,
            'note': 'No automatic restart, stop, mutation, or further tab retries were performed.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node', required=True)
    p.add_argument('--cli', required=True)
    p.add_argument('--profile', required=True)
    p.add_argument('--browser-session', required=True)
    p.add_argument('--recovery-session', required=True)
    p.add_argument('--instance-id', required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--lock-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    # Reserve output before any browser operation; accidental reruns do nothing.
    with a.output.open('x', encoding='utf8') as output:
        t = OpenCliTransport(node=a.node, cli=a.cli, profile=a.profile,
            browser_session=a.browser_session, instance_id=a.instance_id,
            remote_python='python3', remote_helper='', remote_root=a.remote_root, lock_file=a.lock_file)
        result = run_probe(t, remote_root=a.remote_root, recovery_session=a.recovery_session)
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps({'ok': result['ok'], 'output': str(a.output), 'read_only': True}))
    return 0 if result['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
