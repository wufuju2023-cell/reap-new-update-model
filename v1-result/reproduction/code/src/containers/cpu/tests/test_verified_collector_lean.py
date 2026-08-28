"""Real Lean 4.28 collector + loopback mock model; never a GPU/model test."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import os
from pathlib import Path
import subprocess
import threading
import unittest
from unittest.mock import patch

from cpu_runtime.verified_collector import run_collector, atomic_new_json
from cpu_runtime.verified_trajectory import plan_success_path, validate_trace, checked_axioms, export_verified, load_verified_dataset
from cpu_runtime.verified_dataset_store import install_verified_dataset
from tests.test_verified_collector import created, PIN

PROJECT = Path('/opt/reap-runtime')
OUTPUT = Path('/tmp/collector-tests')
REPLAY = Path(__file__).resolve().parents[1]/'verified-replay/VerifiedReplay.lean'


class CollectorLeanTests(unittest.TestCase):
    def run_case(self, name, mode, gamma=.9, bad_ack=None):
        root = OUTPUT/name
        root.mkdir(parents=True, exist_ok=False)
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                prompt = body['messages'][0]['content']
                calls.append({'path': self.path, 'prompt': prompt})
                if '/value/' in self.path:
                    content = json.dumps({'score': -3.0 if mode == 'invalid-value' else 3.0})
                elif mode == 'exhausted': content = 'unknown_collector_tactic'
                elif mode == 'or3':
                    content = ('exact h2' if 'h2 : True' in prompt else
                        'have h2 : True := True.intro' if 'h1 : True' in prompt else 'have h1 : True := True.intro')
                elif mode == 'and':
                    if '⊢ True ∧ True' in prompt: content = 'constructor'
                    else:
                        maximum = 3 if 'case right' in prompt else 1
                        present = max([0]+[i for i in range(1,4) if f'h{i} : True' in prompt])
                        content = f'exact h{maximum}' if present == maximum else f'have h{present+1} : True := True.intro'
                else: content = 'exact True.intro'
                payload = json.dumps({'id': 'local-only', 'choices': [{'index': 0,
                    'message': {'role': 'assistant', 'content': content},
                    'logprobs': {'content': [{'token': content, 'logprob': -.1}]}}]}).encode()
                self.send_response(200); self.send_header('Content-Length', str(len(payload)))
                self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(payload)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        client = type('Client', (), {'create_session': lambda self, sid, **kw: created(sid, kw['theorem_id'])})()
        goal = 'True ∧ True' if mode == 'and' else 'True'
        source = root/'source.lean'
        source.write_text('import ReapRuntime\nset_option reap.num_premises 0\nset_option reap.num_samples 1\n'
            'set_option reap.progressive_sampling_c 0\nset_option reap.max_steps 24\nset_option reap.max_goals 96\n'
            f'theorem collectorFixture : {goal} := by\n  reapTrainingMCTS\n', encoding='utf8')

        def write(path, value):
            value = dict(value)
            if bad_ack == 'ready' and path.name == 'collector-ready.ack.json': value['model_release_sha256'] = 'f'*64
            if bad_ack == 'final' and path.name.startswith('checkpoint-'):
                # or1 solves at checkpoint zero; the Lean final-path guard must still execute.
                value['policy_version'] = 1
            atomic_new_json(path, value)
        try:
            with patch('cpu_runtime.verified_collector.atomic_new_json', side_effect=write):
                report = run_collector(session_id=name, project_dir=PROJECT, theorem_file=str(source),
                    theorem='collectorFixture', output_root=root/'out', gpu_base_url=f'http://127.0.0.1:{server.server_port}',
                    model_release_sha256=PIN, puct_value_gamma=gamma, client=client, barrier_timeout=10)
        finally:
            server.shutdown(); server.server_close(); thread.join()
        directory = root/'out'/name
        (root/'http-calls.json').write_text(json.dumps(calls), encoding='utf8')
        return directory, report, calls

    def replay(self, directory, expected_return):
        session = json.loads((directory/'session.json').read_bytes())
        events = [json.loads(x) for x in (directory/'observer.jsonl').read_bytes().splitlines()]
        candidate = plan_success_path(json.loads((directory/'raw_tree.json').read_bytes()), events, session)
        self.assertEqual(candidate['root_return'], expected_return)
        plan = directory/'replay-plan.json'; trace = directory/'replay-trace.json'
        plan.write_text(json.dumps(candidate['plan']), encoding='utf8')
        source = (directory/'source.lean').read_text()
        source = source.replace('import ReapRuntime\n', 'import ReapRuntime\n'+REPLAY.read_text()+'\n')
        source = source.replace('  reapTrainingMCTS', f'  reapVerifiedReplay {json.dumps(str(plan))} {json.dumps(str(trace))}')
        source += '\n#print axioms collectorFixture\n'
        replay = directory/'replay.lean'; replay.write_text(source, encoding='utf8')
        for label, path in (('independent-proof', directory/'proof.lean'), ('independent-replay', replay)):
            outcome = subprocess.run(['lake','env','lean',str(path)], cwd=PROJECT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
            (directory/(label+'.log')).write_bytes(outcome.stdout)
            self.assertEqual(outcome.returncode, 0, outcome.stdout.decode())
            checked_axioms(outcome.stdout.decode(), 'collectorFixture')
        actual = json.loads(trace.read_bytes())
        validate_trace(actual, candidate, 'collectorFixture')
        self.assertEqual(actual['root_return'], expected_return)
        return candidate

    def test_real_lean_core_cost_backup_puct_and_invalid_values(self):
        fixture = Path(__file__).with_name('fixtures')/'VerifiedCollectorUnit.lean'
        outcome = subprocess.run(['lake','env','lean',str(fixture)], cwd=PROJECT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
        (OUTPUT/'unit.log').write_bytes(outcome.stdout)
        self.assertEqual(outcome.returncode, 0, outcome.stdout.decode())
        self.assertIn(b'VERIFIED_COLLECTOR_UNIT_OK', outcome.stdout)

    def test_complete_bundle_retains_new_actor_profile_receipt_and_ready_event(self):
        directory, report, _ = self.run_case('bundle', 'or1')
        self.assertEqual(report['status'], 'solved_pending_independent_verification', report)
        image = os.environ['REAP_TEST_CONTAINER_IMAGE']
        self.assertRegex(image, r'^[0-9a-f]{64}$')
        proof = directory/'proof.lean'
        outcome = subprocess.run(['lake','env','lean',str(proof)], cwd=PROJECT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
        self.assertEqual(outcome.returncode, 0, outcome.stdout.decode()+outcome.stderr.decode())
        checked_axioms(outcome.stdout.decode(), 'collectorFixture')
        (directory/'proof.stdout').write_bytes(outcome.stdout); (directory/'proof.stderr').write_bytes(outcome.stderr)
        receipt = {'proof_from_session': 'bundle', 'source_theorem_sha256': report['theorem_sha256'],
            'generated_proof_sha256': hashlib.sha256(proof.read_bytes()).hexdigest(),
            'image': image, 'network': 'none', 'returncode': outcome.returncode,
            'stdout': outcome.stdout.decode(), 'stderr': outcome.stderr.decode()}
        receipt_path=directory/'proof-receipt.json'; receipt_path.write_text(json.dumps(receipt))
        bundle=directory/'verified-dataset'
        dataset=export_verified(session_dir=directory, source=directory/'source.lean', proof=proof,
            proof_receipt=receipt_path, theorem='collectorFixture', output=bundle,
            lean_project=PROJECT, replay_module=REPLAY)
        pin=hashlib.sha256((bundle/'dataset.json').read_bytes()).hexdigest()
        self.assertEqual(load_verified_dataset(bundle,expected_sha256=pin),dataset)
        installed=install_verified_dataset(bundle,OUTPUT/'registry',expected_sha256=pin)
        copied=Path(installed['path'])
        self.assertEqual(load_verified_dataset(copied,expected_sha256=pin),dataset)
        self.assertEqual(len(list(copied.iterdir())),17)
        session=json.loads((copied/'session.json').read_bytes())
        self.assertEqual(session['profile'],'verified-release-collector-v1')
        self.assertEqual(session['role'],'actor')
        self.assertEqual(session['model_release_sha256'],PIN)
        self.assertEqual(session['lineage'],session['initialization_receipt']['lineage'])
        self.assertEqual(json.loads((copied/'observer.jsonl').read_text().splitlines()[0])['kind'],'collector_contract')
        self.assertEqual(dataset['root_return'],-1)

    def test_real_http_positive_distance_once_and_or_one_three_steps(self):
        for mode, count in [('or1',1),('or3',3)]:
            for gamma in [.9,.99]:
                with self.subTest(mode=mode,gamma=gamma):
                    directory, report, calls = self.run_case(mode+str(int(gamma*1000)),mode,gamma)
                    self.assertEqual(report['status'],'solved_pending_independent_verification',report)
                    candidate = self.replay(directory,-count)
                    rows=[json.loads(x) for x in (directory/'observer.jsonl').read_bytes().splitlines()]
                    generated=[x for x in rows if x['kind']=='generation']
                    self.assertTrue(generated)
                    self.assertTrue(all(x['search_value']==-3.0 and x['policy_version']==0 for x in generated))
                    self.assertTrue(all(x['gamma']==gamma for x in rows if x['kind']=='checkpoint'))
                    self.assertEqual(len([x for x in calls if '/value/' in x['path']]),count)
                    self.assertEqual(len(candidate['rows']),count)

    def test_real_independent_and_uses_longest_branch_and_zero_focus_cost(self):
        directory, report, calls = self.run_case('and','and')
        self.assertEqual(report['status'],'solved_pending_independent_verification',report)
        candidate=self.replay(directory,-5) # constructor + max(left2,right4), never 1+2+4
        self.assertEqual(len(candidate['rows']),7)
        self.assertEqual(len([x for x in calls if '/value/' in x['path']]),7)

    def test_invalid_distance_stops_without_proof_or_new_checkpoint(self):
        directory, report, _ = self.run_case('invalid-value','invalid-value')
        self.assertEqual(report['status'],'failed_unknown')
        self.assertFalse((directory/'proof.lean').exists())
        self.assertEqual(report['checkpoint_count'],0)

    def test_known_exhaustion_has_complete_final_ack_but_no_proof(self):
        directory, report, _ = self.run_case('exhausted','exhausted')
        self.assertEqual(report['status'],'exhausted',report)
        self.assertFalse((directory/'proof.lean').exists())
        self.assertGreater(report['checkpoint_count'],0)

    def test_ready_pin_mismatch_never_calls_model(self):
        directory, report, calls = self.run_case('ready-bad','or1',bad_ack='ready')
        self.assertEqual(report['status'],'failed_unknown')
        self.assertEqual(calls,[])
        self.assertFalse((directory/'proof.lean').exists())

    def test_final_solved_ack_version_change_rejected_by_lean(self):
        directory, report, _ = self.run_case('final-bad','or1',bad_ack='final')
        self.assertEqual(report['status'],'failed_unknown')
        self.assertFalse((directory/'proof.lean').exists())
        self.assertIn('Fixed release collector cannot change policy version',(directory/'stdout.log').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--project-dir',type=Path,default=PROJECT)
    args,remaining=p.parse_known_args(); OUTPUT=args.output_dir; PROJECT=args.project_dir
    OUTPUT.mkdir(parents=True,exist_ok=False)
    unittest.main(argv=[__file__,*remaining])
