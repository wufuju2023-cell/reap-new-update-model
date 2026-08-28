#!/usr/bin/env python3
"""Real Generator HTTP error behavior using a local stub, never a GPU test."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import unittest


PROJECT = Path('/opt/reap-runtime')
FIXTURE = Path(__file__).with_name('fixtures') / 'StrictValue.lean'


class StrictValueTests(unittest.TestCase):
    def probe(self, enabled, response):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                calls.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                status, body = response
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        env = os.environ.copy()
        env.pop('REAP_OBSERVER_PATH', None)
        if enabled:
            env['REAP_OBSERVER_PATH'] = '/tmp/strict-value-test-observer.jsonl'
        env['REAP_VALUE_ENDPOINT'] = f'http://127.0.0.1:{server.server_port}'
        try:
            result = subprocess.run(['lake', 'env', 'lean', str(FIXTURE)], cwd=PROJECT,
                                    env=env, capture_output=True, text=True, timeout=90)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        return result, calls

    def test_http_500_strict_fails_closed(self):
        result, calls = self.probe(True, (500, {'error': 'test unavailable'}))
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('refusing fallback sentinel', result.stdout + result.stderr)
        self.assertNotIn('VALUE_PROBE:', result.stdout)
        self.assertEqual(len(calls), 3)

    def test_http_500_default_off_preserves_fallback(self):
        result, calls = self.probe(False, (500, {'error': 'test unavailable'}))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('VALUE_PROBE: -1000', result.stdout)
        self.assertEqual(len(calls), 3)

    @staticmethod
    def chat(content):
        return {'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'test',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content},
                             'finish_reason': 'stop'}]}

    def test_real_score_1000_is_not_misclassified_as_error(self):
        result, calls = self.probe(True, (200, self.chat('{"score":1000.0}')))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('VALUE_PROBE: -1000', result.stdout)
        self.assertEqual(len(calls), 1)

    def test_invalid_value_json_strict_fails_closed(self):
        result, calls = self.probe(True, (200, self.chat('not-json')))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('refusing fallback sentinel', result.stdout + result.stderr)
        self.assertNotIn('VALUE_PROBE:', result.stdout)
        self.assertEqual(len(calls), 3)

    def test_empty_choices_strict_fails_closed(self):
        body = self.chat('unused')
        body['choices'] = []
        result, calls = self.probe(True, (200, body))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('refusing fallback sentinel', result.stdout + result.stderr)
        self.assertNotIn('VALUE_PROBE:', result.stdout)
        self.assertEqual(len(calls), 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', type=Path, default=PROJECT)
    args, rest = parser.parse_known_args()
    PROJECT = args.project_dir
    unittest.main(argv=['test_strict_value.py', *rest], verbosity=2)
