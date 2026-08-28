"""Local status recovery fixtures; no browser, remote instance or GPU used."""
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

from tools.amd_jupyter.http_bridge import OpenCliTransport
from tools.amd_jupyter.status_probe import run_probe, status_source, READ_TIMEOUT

ROOT = '/mnt/workspace/reap-status-test'


def response(root=ROOT):
    payload = {'schema_version': 'reap.amd.read-only-status.v1', 'root': root, 'epoch': 123,
               'gpu_busy_percent': {}, 'processes': [], 'job_roots': {}}
    return SimpleNamespace(returncode=0, stdout=json.dumps({'status': 200, 'text': json.dumps({'output': json.dumps(payload)})}))


class StatusProbeTests(unittest.TestCase):
    def transport(self):
        t = OpenCliTransport(node='node', cli='main.js', profile='explicit-edge', browser_session='primary',
            instance_id='dsw-123', remote_python='python3', remote_helper='', remote_root=ROOT, lock_file=Path('unused'))
        t._locked = nullcontext
        return t

    def test_success_uses_instance_guard_and_never_opens_extra_tab(self):
        calls = []
        def runner(args, **kwargs):
            calls.append((args, kwargs))
            return response()
        r = run_probe(self.transport(), remote_root=ROOT, recovery_session='recovery', runner=runner)
        self.assertTrue(r['ok'])
        self.assertEqual(len(calls), 1)
        args, kw = calls[0]
        self.assertEqual(args[4:7], ['browser', 'primary', 'eval'])
        self.assertIn('/dsw-123/dsw/commands', args[-1])
        self.assertIn('Wrong or ambiguous instance', args[-1])
        self.assertEqual(kw['timeout'], READ_TIMEOUT)

    def test_read_timeout_opens_one_recovery_tab_then_reads(self):
        calls = []
        def runner(args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired('sensitive argv is not logged', 25)
            if args[6] == 'open':
                return SimpleNamespace(returncode=0, stdout='{}')
            return response()
        r = run_probe(self.transport(), remote_root=ROOT, recovery_session='recovery', runner=runner)
        self.assertTrue(r['ok'])
        self.assertEqual([c[6] for c in calls], ['eval', 'open', 'eval'])
        self.assertEqual(r['browser_session'], 'recovery')
        self.assertNotIn('sensitive', json.dumps(r))

    def test_two_failures_stop_without_claiming_idle_or_retrying_mutations(self):
        calls = []
        def runner(args, **kwargs):
            calls.append(args)
            if args[6] == 'open':
                return SimpleNamespace(returncode=0, stdout='{}')
            return SimpleNamespace(returncode=1, stdout=json.dumps({'error': {'code': 'cdp_timeout', 'message': 'private text'}}))
        r = run_probe(self.transport(), remote_root=ROOT, recovery_session='recovery', runner=runner)
        self.assertFalse(r['ok'])
        self.assertFalse(r['idle_confirmed'])
        self.assertEqual(len(calls), 3)
        self.assertNotIn('private text', json.dumps(r))

    def test_wrong_status_root_never_accepted(self):
        def runner(args, **kwargs):
            return SimpleNamespace(returncode=0, stdout='{}') if args[6] == 'open' else response('/other')
        r = run_probe(self.transport(), remote_root=ROOT, recovery_session='recovery', runner=runner)
        self.assertFalse(r['ok'])

    def test_root_and_recovery_validation_precede_dispatch(self):
        for root in ('/tmp/other', '/mnt/workspace/a/../b', '/mnt/workspace//a', '/mnt/workspace/x;id'):
            with self.subTest(root=root), self.assertRaises(ValueError):
                status_source(root)
        with self.assertRaises(ValueError):
            run_probe(self.transport(), remote_root=ROOT, recovery_session='primary')
        # The fixed remote program is syntactically valid and has no mutation API.
        source = status_source(ROOT)
        compile(source, '<status>', 'exec')
        for forbidden in ('write_text', 'write_bytes', 'subprocess', 'os.kill', 'requests.post'):
            self.assertNotIn(forbidden, source)


if __name__ == '__main__':
    unittest.main()
