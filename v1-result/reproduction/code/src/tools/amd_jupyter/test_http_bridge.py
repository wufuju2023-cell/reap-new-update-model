"""Local transport tests. Never attach a browser or contact a remote instance."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid

try:
    from .http_bridge import Bridge, OpenCliTransport, UnknownOutcome, _FifoLock, make_server
    from .remote_http_job import json_bytes, poll_job, run_job, submit_job, validate_request, validate_route
except ImportError:
    from http_bridge import Bridge, OpenCliTransport, UnknownOutcome, _FifoLock, make_server
    from remote_http_job import json_bytes, poll_job, run_job, submit_job, validate_request, validate_route


class Response(io.BytesIO):
    def __init__(self, data: bytes, status=200):
        super().__init__(data)
        self.status = status


class StubTransport:
    def __init__(self, root: Path, response: bytes = b'{"ok":true}', status=200):
        self.root, self.response, self.status = root, response, status
        self.submits, self.launches, self.polls, self.requests = [], [], [], []
        self.lose_submit_ack = False
        self.corrupt_response = False

    def open(self, request, *, timeout):
        self.requests.append(request)
        if not request.full_url.startswith("http://127.0.0.1:8760/"):
            raise AssertionError("non-loopback target")
        return Response(self.response, self.status)

    def __call__(self, action, request=None, *, request_id="", offset=0):
        if action == "submit":
            self.submits.append(request["request_id"])
            result = submit_job(self.root, request, launcher=self.launches.append)
            run_job(self.root, request["request_id"], opener=self.open)
            if self.lose_submit_ack:
                raise TimeoutError("injected acknowledgement loss")
            return result
        self.polls.append((request_id, offset))
        result = poll_job(self.root, request_id, offset)
        if self.corrupt_response and result.get("state") == "done":
            result["sha256"] = "0" * 64
        return result


class HttpBridgeTests(unittest.TestCase):
    def test_replica_port_is_bound_to_uuid_and_worker_exact_loopback(self):
        for port in (8760, 8761, 8762):
            request = self.request(target_port=port)
            launches = []; calls = []
            submit_job(self.root, request, launcher=launches.append)
            with self.assertRaisesRegex(ValueError, "different payload"):
                submit_job(self.root, {**request, "target_port":8762 if port != 8762 else 8761}, launcher=launches.append)
            def opened(req, **kwargs):
                calls.append(req.full_url)
                return Response(b'{}')
            run_job(self.root, request['request_id'], opener=opened)
            run_job(self.root, request['request_id'], opener=opened)
            self.assertEqual(calls, [f'http://127.0.0.1:{port}/sessions/a/learn/v1'])
            self.assertEqual(len(launches), 1)
            self.assertEqual(poll_job(self.root, request['request_id'])['request_sha256'], validate_request(request))

    def test_replica_port_invalid_before_any_dispatch_and_old_bytes_unchanged(self):
        request = self.request()
        self.assertNotIn('target_port', request)
        for port in (True, 8761.0, '8761', 80, 65536, None):
            with self.subTest(port=port), self.assertRaises(ValueError):
                validate_request({**request, 'target_port':port})
        for port in (True, 80, 8761.0):
            with self.assertRaises(ValueError):
                Bridge(lambda *a, **k: self.fail('unexpected dispatch'), ['a'], target_port=port)
        transport = StubTransport(self.root / 'old')
        Bridge(transport,['a']).request('POST','/sessions/a',{})
        saved = json.loads(next((self.root/'old').glob('*/request.json')).read_bytes())
        self.assertNotIn('target_port', saved)

    def test_replica_bridge_lost_ack_keeps_single_target_and_no_retry(self):
        class ReplicaTransport(StubTransport):
            def open(self, request, **kwargs):
                self.requests.append(request)
                self.asserted_url = request.full_url
                return Response(self.response)
        transport = ReplicaTransport(self.root/'replica')
        transport.lose_submit_ack = True
        bridge = Bridge(transport, ['a'], target_port=8762)
        self.assertEqual(bridge.request('POST','/sessions/a',{})[0],200)
        self.assertEqual(transport.asserted_url,'http://127.0.0.1:8762/sessions/a')
        self.assertEqual(len(transport.submits),1); self.assertEqual(len(transport.requests),1)
        saved = json.loads(next((self.root/'replica').glob('*/request.json')).read_bytes())
        self.assertEqual(saved['target_port'],8762)

    def test_retirement_read_is_allowlisted_after_unknown_but_never_unfreezes_mutations(self):
        transport = StubTransport(self.root / "retire-jobs", b'{"status":"released"}')
        bridge = Bridge(transport, ["a"])
        transport.corrupt_response = True
        with self.assertRaises(UnknownOutcome):
            bridge.request("POST", "/sessions/a/retire/v1", {"name": "final", "expected_policy_version": 0})
        transport.corrupt_response = False
        self.assertEqual(bridge.request("GET", "/sessions/a/retire/v1", {})[0], 200)
        self.assertEqual([request.method for request in transport.requests], ["POST", "GET"])
        for method, path in (("POST", "/sessions/a/retire/v1"), ("POST", "/sessions/a/learn/v1"),
                             ("DELETE", "/sessions/a")):
            with self.subTest(path=path), self.assertRaises(UnknownOutcome):
                bridge.request(method, path, {})
        for path in ("/sessions/b/retire/v1", "/sessions/a/retire/v1/extra", "/sessions/a/learn/v1"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                bridge.request("GET", path, {})
        self.assertEqual(len(transport.submits), 2)

    def test_deadline_order_and_no_call_started_without_full_budget(self):
        from cpu_runtime.transport_budget import DEFAULT_BUDGET, TransportBudget
        budget = DEFAULT_BUDGET
        self.assertLess(budget.daemon_http, budget.cli)
        self.assertLess(budget.bridge(), budget.client)
        self.assertLess(budget.client, budget.barrier)
        with self.assertRaises(ValueError):
            TransportBudget(cli=55)
        now = [0.0]
        calls = []
        class SlowTransport:
            max_call_seconds = budget.transport_call
            def __call__(self, action, request=None, *, request_id="", offset=0):
                calls.append((action, now[0]))
                now[0] += self.max_call_seconds
                if action == "submit":
                    return {"request_id": request["request_id"], "request_sha256": validate_request(request)}
                return {"state": "pending", "request_id": request_id}
        bridge = Bridge(SlowTransport(), ["a"], clock=lambda: now[0],
                        sleep=lambda delay: now.__setitem__(0, now[0] + delay))
        with self.assertRaisesRegex(UnknownOutcome, "deadline"):
            bridge.request("POST", "/sessions/a/learn/v1", {})
        self.assertLess(now[0], budget.bridge())
        self.assertEqual([c[0] for c in calls].count("submit"), 1)
        previous = list(calls)
        with self.assertRaises(UnknownOutcome):
            bridge.request("POST", "/sessions/a/learn/v1", {})
        self.assertEqual(previous, calls)

    def test_outer_deadline_exceeds_inner_http_and_lost_ack_is_recovered_once(self):
        from cpu_runtime.transport_budget import DEFAULT_BUDGET
        transport = OpenCliTransport(node="node", cli="main.js", profile="p", browser_session="ms",
            instance_id="dsw-123", remote_python="python3", remote_helper="/job.py",
            remote_root="/jobs", lock_file=self.root / "lock")
        def fake_run(args, **kwargs):
            self.assertEqual(kwargs["timeout"], DEFAULT_BUDGET.cli)
            self.assertGreater(kwargs["timeout"], 130)
            self.assertIn("AbortSignal.timeout(20000)", args[-1])
            return subprocess.CompletedProcess(args, 0, json.dumps({"status": 200,
                "text": json.dumps({"output": '{"state":"pending"}'})}), "")
        with patch.object(subprocess, "run", side_effect=fake_run):
            self.assertEqual(transport("poll", request_id="a" * 32), {"state": "pending"})
        durable = StubTransport(self.root / "jobs")
        durable.lose_submit_ack = True
        bridge = Bridge(durable, ["a"])
        self.assertEqual(bridge.request("POST", "/sessions/a/learn/v1", {})[0], 200)
        self.assertEqual(len(durable.submits), 1)
        self.assertEqual(len(durable.requests), 1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.events = io.StringIO()
        stderr = patch("sys.stderr", self.events)
        stderr.start()
        self.addCleanup(stderr.stop)

    def tearDown(self):
        self.temp.cleanup()

    def request(self, **changes):
        return {"request_id": uuid.uuid4().hex, "method": "POST", "path": "/sessions/a/learn/v1",
                "body": {"event": {"event_id": "test", "reward": -1}},
                "allowed_sessions": ["a", "b"], "timeout_seconds": 600, **changes}

    def test_job_submit_and_worker_are_idempotent_and_reject_uuid_conflicts(self):
        request = self.request()
        launches = []
        first = submit_job(self.root, request, launcher=launches.append)
        second = submit_job(self.root, request, launcher=launches.append)
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(len(launches), 1)
        with self.assertRaisesRegex(ValueError, "different payload"):
            submit_job(self.root, {**request, "body": {"changed": True}}, launcher=launches.append)
        calls = []
        def opener(request, **kwargs):
            calls.append(request)
            return Response(b'{"policy_version":1}')
        with ThreadPoolExecutor(2) as executor:
            list(executor.map(lambda _: run_job(self.root, request["request_id"], opener), range(2)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(poll_job(self.root, request["request_id"])["state"], "done")

    def test_lost_ack_only_polls_same_id_and_large_response_is_checked(self):
        payload = json_bytes({"candidate": "证明" * 6000})
        transport = StubTransport(self.root, payload, 201)
        transport.lose_submit_ack = True
        bridge = Bridge(transport, ["a", "b"])
        status, body = bridge.request("POST", "/sessions/a", {})
        self.assertEqual((status, body), (201, payload))
        self.assertEqual(len(transport.submits), 1)
        self.assertGreater(len(transport.polls), 1)
        self.assertTrue(all(item[0] == transport.submits[0] for item in transport.polls))
        self.assertEqual([item[1] for item in transport.polls], list(range(0, len(payload), 4096)))

    def test_failed_submit_and_missing_job_fails_fast_without_repeating_or_leaking_body(self):
        now, calls = [0.0], []
        def missing(action, request=None, *, request_id="", offset=0):
            identity = request["request_id"] if action == "submit" else request_id
            calls.append((action, identity))
            if action == "submit":
                raise TimeoutError("SECRET_BODY_AND_AUTH_MUST_NOT_APPEAR")
            return {"state": "not_found", "request_id": request_id}
        bridge = Bridge(missing, ["a"], clock=lambda: now[0],
                        sleep=lambda delay: now.__setitem__(0, now[0] + delay))
        with self.assertRaisesRegex(UnknownOutcome, "not found in three read-only polls") as caught:
            bridge.request("POST", "/sessions/a/learn/v1", {"secret": "SECRET_BODY_AND_AUTH_MUST_NOT_APPEAR"})
        self.assertEqual([action for action, _ in calls], ["submit", "poll", "poll", "poll"])
        self.assertTrue(all(identity == caught.exception.request_id for _, identity in calls))
        self.assertEqual(now[0], 2.0)
        with self.assertRaisesRegex(UnknownOutcome, "suspended"):
            bridge.request("POST", "/sessions/a/learn/v1", {})
        self.assertEqual(len(calls), 4)
        events = [json.loads(line) for line in self.events.getvalue().splitlines()]
        self.assertIn("submit_unacknowledged", [event["event"] for event in events])
        self.assertEqual(events[-1]["event"], "request_unknown")
        self.assertTrue(all(event["request_id"] == caught.exception.request_id for event in events))
        self.assertNotIn("SECRET_BODY_AND_AUTH_MUST_NOT_APPEAR", self.events.getvalue())

    def test_lost_ack_can_recover_after_two_not_found_reads_using_same_uuid(self):
        transport = StubTransport(self.root)
        transport.lose_submit_ack = True
        missing_ids = []
        def delayed_visibility(action, request=None, *, request_id="", offset=0):
            if action == "poll" and len(missing_ids) < 2:
                missing_ids.append(request_id)
                return {"state": "not_found", "request_id": request_id}
            return transport(action, request, request_id=request_id, offset=offset)
        bridge = Bridge(delayed_visibility, ["a"], sleep=lambda _: None)
        self.assertEqual(bridge.request("POST", "/sessions/a/learn/v1", {}), (200, b'{"ok":true}'))
        self.assertEqual(len(transport.submits), 1)
        self.assertEqual(missing_ids, transport.submits * 2)
        self.assertEqual(transport.polls, [(transport.submits[0], 0)])

    def test_bad_hash_suspends_session_without_resubmission(self):
        transport = StubTransport(self.root)
        transport.corrupt_response = True
        bridge = Bridge(transport, ["a", "b"])
        with self.assertRaisesRegex(UnknownOutcome, "checksum"):
            bridge.request("POST", "/sessions/a/learn/v1", {})
        with self.assertRaisesRegex(UnknownOutcome, "suspended"):
            bridge.request("POST", "/sessions/a/learn/v1", {})
        self.assertEqual(len(transport.submits), 1)
        transport.corrupt_response = False
        self.assertEqual(bridge.request("POST", "/sessions/b", {})[0], 200)

    def test_worker_disconnect_is_unknown_and_wait_expiry_never_resubmits(self):
        request = self.request()
        submit_job(self.root, request, launcher=lambda _: None)
        def disconnected(*args, **kwargs):
            raise TimeoutError("injected server disconnect after possible commit")
        run_job(self.root, request["request_id"], disconnected)
        self.assertEqual(poll_job(self.root, request["request_id"])["state"], "unknown")
        now = [0.0]
        calls = []
        def never_finished(action, request=None, *, request_id="", offset=0):
            calls.append(action)
            if action == "submit":
                return {"request_id": request["request_id"], "request_sha256": validate_request(request)}
            return {"state": "pending", "request_id": request_id}
        bridge = Bridge(never_finished, ["a"], timeout_seconds=1, clock=lambda: now[0],
                        sleep=lambda delay: now.__setitem__(0, now[0] + 20))
        with self.assertRaisesRegex(UnknownOutcome, "wait expired"):
            bridge.request("POST", "/sessions/a", {})
        self.assertEqual(calls.count("submit"), 1)
        calls.clear()
        def failed_polls(action, request=None, *, request_id="", offset=0):
            calls.append(action)
            if action == "submit":
                return {"request_id": request["request_id"], "request_sha256": validate_request(request)}
            raise ConnectionError("injected bridge disconnect")
        bridge = Bridge(failed_polls, ["a"], sleep=lambda _: None)
        with self.assertRaisesRegex(UnknownOutcome, "three read-only polls"):
            bridge.request("POST", "/sessions/a", {})
        self.assertEqual(calls, ["submit", "poll", "poll", "poll"])

    def test_route_body_limits_and_redirect_status(self):
        for method, path in (("POST", "/sessions/other"), ("POST", "/sessions/a/../../evil"),
                             ("GET", "http://evil/health"), ("POST", "/sessions/a?x=1"), ("DELETE", "/health")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_route(method, path, ["a"])
        with self.assertRaises(ValueError):
            validate_request(self.request(body={"text": "x" * 8192}))
        request = self.request()
        submit_job(self.root, request, launcher=lambda _: None)
        def redirect(http_request, **kwargs):
            raise HTTPError(http_request.full_url, 302, "redirect forbidden", {}, io.BytesIO(b'{"error":"redirect"}'))
        run_job(self.root, request["request_id"], redirect)
        self.assertEqual(poll_job(self.root, request["request_id"])["status"], 302)

    def test_real_loopback_http_server_preserves_json_status_and_rejects_web_origin(self):
        transport = StubTransport(self.root, b'{"error":"VersionConflictError"}', 409)
        bridge = Bridge(transport, ["a", "b"])
        server = make_server(bridge, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            def call(session):
                request = Request(base + f"/sessions/{session}/learn/v1", data=b"{}", headers={"Content-Type": "application/json"})
                try:
                    return urlopen(request, timeout=5)
                except HTTPError as response:
                    with response:
                        return response.code, json.loads(response.read())
            with ThreadPoolExecutor(2) as executor:
                responses = list(executor.map(call, ["a", "b"]))
            self.assertEqual(responses, [(409, {"error": "VersionConflictError"})] * 2)
            request = Request(base + "/sessions/a", data=b"{}", headers={"Content-Type": "application/json", "Origin": "https://unrelated.example"})
            with self.assertRaises(HTTPError) as caught:
                urlopen(request, timeout=5)
            self.assertEqual(caught.exception.code, 400)
            self.assertEqual(server.server_address[0], "127.0.0.1")
            self.assertEqual(len(transport.submits), 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_opencli_uses_argv_fixed_instance_and_serializes_calls_without_browser(self):
        transport = OpenCliTransport(node="node.exe", cli="C:/path with spaces/main.js", profile="p",
                                     browser_session="ms", instance_id="dsw-123", remote_python="python3",
                                     remote_helper="/mnt/workspace/remote_http_job.py", remote_root="/mnt/workspace/out/jobs",
                                     lock_file=self.root / "opencli.lock")
        active, maximum = [0], [0]
        guard = threading.Lock()
        def fake_run(args, **kwargs):
            with guard:
                active[0] += 1
                maximum[0] = max(maximum[0], active[0])
            try:
                self.assertFalse(kwargs["shell"])
                self.assertIn('"/dsw-123/dsw/commands"', args[-1])
                self.assertNotIn("document.cookie", args[-1])
                self.assertNotIn("localStorage", args[-1])
                time.sleep(0.01)
                return subprocess.CompletedProcess(args, 0, json.dumps({"status": 200, "text": json.dumps({"output": '{"state":"pending"}'})}), "")
            finally:
                with guard:
                    active[0] -= 1
        with patch.object(subprocess, "run", side_effect=fake_run), ThreadPoolExecutor(2) as executor:
            results = list(executor.map(lambda _: transport("poll", request_id="a" * 32), range(2)))
        self.assertEqual(results, [{"state": "pending"}] * 2)
        self.assertEqual(maximum[0], 1)

    def test_fifo_lock_serves_queued_submit_before_later_poll(self):
        lock = _FifoLock()
        self.assertTrue(lock.acquire(timeout=0))
        order = []
        threads = []
        def caller(label):
            acquired = lock.acquire(timeout=2)
            if acquired:
                try:
                    order.append(label)
                finally:
                    lock.release()
        for index, label in enumerate(("existing_poll", "submit", "later_poll"), start=1):
            thread = threading.Thread(target=caller, args=(label,))
            threads.append(thread)
            thread.start()
            with lock._condition:
                self.assertTrue(lock._condition.wait_for(lambda: len(lock._waiters) == index, timeout=1))
        lock.release()
        for thread in threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(order, ["existing_poll", "submit", "later_poll"])

    def test_fifo_timeout_removes_ticket_and_does_not_block_following_caller(self):
        lock = _FifoLock()
        self.assertTrue(lock.acquire(timeout=0))
        expired = []
        thread = threading.Thread(target=lambda: expired.append(lock.acquire(timeout=0.02)))
        thread.start()
        thread.join(timeout=1)
        self.assertEqual(expired, [False])
        lock.release()
        self.assertTrue(lock.acquire(timeout=0))
        lock.release()

    def test_opencli_timeout_logs_phase_but_never_exception_argv(self):
        transport = OpenCliTransport(node="node.exe", cli="main.js", profile="p", browser_session="ms",
                                     instance_id="dsw-123", remote_python="python3", remote_helper="helper.py",
                                     remote_root="/jobs", lock_file=self.root / "opencli.lock")
        identity = "a" * 32
        with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("SECRET_ARGV", 55)):
            with self.assertRaises(subprocess.TimeoutExpired):
                transport("poll", request_id=identity)
        event = json.loads(self.events.getvalue().splitlines()[-1])
        self.assertEqual((event["request_id"], event["stage"], event["exception_type"]),
                         (identity, "cli_run", "TimeoutExpired"))
        self.assertNotIn("SECRET_ARGV", self.events.getvalue())


if __name__ == "__main__":
    unittest.main()
