#!/usr/bin/env python3
"""WSL loopback HTTP bridge to a fixed AMD instance via Edge/OpenCLI.

Run only after the user starts the instance. The remote helper must already be
uploaded. This is a transport for the temporary host-B experiment, not evidence
that a GPU container ran. OpenCLI/browser integration requires a separate gate.
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cpu_runtime.transport_budget import DEFAULT_BUDGET

try:
    from .remote_http_job import MAX_BODY, MAX_CHUNK, MAX_RESPONSE, json_bytes, validate_request, validate_route
except ImportError:  # direct script execution
    from remote_http_job import MAX_BODY, MAX_CHUNK, MAX_RESPONSE, json_bytes, validate_request, validate_route


class UnknownOutcome(RuntimeError):
    def __init__(self, request_id: str, reason: str):
        self.request_id = request_id
        super().__init__(reason)


def _event(event: str, request_id: str, **fields) -> None:
    """Log only caller-selected metadata, never exception text or subprocess argv."""
    try:
        print(json.dumps({"event": event, "request_id": request_id,
                          "monotonic_seconds": time.monotonic(), **fields}, ensure_ascii=True),
              file=sys.stderr, flush=True)
    except OSError:
        pass  # Diagnostic output must not change whether a remote request is sent.


class _FifoLock:
    """Bounded process-local FIFO; a polling thread cannot jump queued submits."""
    def __init__(self):
        self._condition = threading.Condition()
        self._waiters = deque()
        self._held = False

    def acquire(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        ticket = object()
        with self._condition:
            self._waiters.append(ticket)
            self._condition.notify_all()
            try:
                while self._held or self._waiters[0] is not ticket:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._waiters.remove(ticket)
                        self._condition.notify_all()
                        return False
                    self._condition.wait(remaining)
                self._waiters.popleft()
                self._held = True
                return True
            except BaseException:
                self._waiters.remove(ticket)
                self._condition.notify_all()
                raise

    def release(self) -> None:
        with self._condition:
            if not self._held:
                raise RuntimeError("release of unlocked FIFO lock")
            self._held = False
            self._condition.notify_all()


class OpenCliTransport:
    """One short eval per call; no authentication storage or daemon bypass."""
    _thread_lock = _FifoLock()
    max_call_seconds = DEFAULT_BUDGET.transport_call

    def __init__(self, *, node: str, cli: str, profile: str, browser_session: str, instance_id: str,
                 remote_python: str, remote_helper: str, remote_root: str, lock_file: Path):
        if not re.fullmatch(r"dsw-\d+", instance_id):
            raise ValueError("an explicit dsw-NNN instance ID is required")
        self.node, self.cli, self.profile = node, cli, profile
        self.browser_session, self.instance_id = browser_session, instance_id
        self.remote_python, self.remote_helper, self.remote_root = remote_python, remote_helper, remote_root
        self.lock_file = lock_file

    @contextmanager
    def _locked(self):
        # All bridge processes use the same lock path. Other manually launched
        # OpenCLI commands must also avoid this session while the bridge runs.
        deadline = time.monotonic() + DEFAULT_BUDGET.lock
        if not self._thread_lock.acquire(timeout=DEFAULT_BUDGET.lock):
            raise TimeoutError("OpenCLI bridge mutex wait expired")
        try:
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            with self.lock_file.open("a+b") as handle:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    handle.write(b"0")
                    handle.flush()
                    handle.seek(0)
                    while True:
                        try:
                            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                            break
                        except OSError:
                            if time.monotonic() >= deadline:
                                raise TimeoutError("OpenCLI bridge file-lock wait expired")
                            time.sleep(0.05)
                    try:
                        yield
                    finally:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    while True:
                        try:
                            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.monotonic() >= deadline:
                                raise TimeoutError("OpenCLI bridge file-lock wait expired")
                            time.sleep(0.05)
                    try:
                        yield
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._thread_lock.release()

    def _expression(self, command: str) -> str:
        payload = json.dumps({"command": command}, ensure_ascii=True, separators=(",", ":"))
        expected = json.dumps("/" + self.instance_id + "/dsw/commands")
        return """(async()=>{
if(location.origin!=='https://www.modelscope.cn') throw new Error('Wrong origin');
if(!document.body.innerText.includes('DSW-AMD')) throw new Error('Not AMD workspace');
const frame=document.querySelector('iframe[title="notebook-ide"]');
if(!frame) throw new Error('No IDE frame');
const urls=[...new Set(frame.contentWindow.performance.getEntriesByType('resource')
.map(e=>new URL(e.name)).filter(u=>u.hostname==='dsw-gateway-cn-hangzhou.data.aliyun.com'
&& /^\\/dsw-\\d+\\/dsw\\/commands$/.test(u.pathname)).map(u=>u.origin+u.pathname))];
if(urls.length!==1 || new URL(urls[0]).pathname!==__EXPECTED__) throw new Error('Wrong or ambiguous instance');
const r=await fetch(urls[0]+'?type=status&name=next-ide',{method:'POST',credentials:'include',
headers:{'Content-Type':'application/json'},signal:AbortSignal.timeout(__FETCH_MS__),body:JSON.stringify(__PAYLOAD__)});
return {status:r.status,text:await r.text()};
})()""".replace("__EXPECTED__", expected).replace("__PAYLOAD__", payload).replace(
            "__FETCH_MS__", str(int(DEFAULT_BUDGET.fetch * 1000)))

    def __call__(self, action: str, request: dict | None = None, *, request_id: str = "", offset: int = 0) -> dict:
        command = [self.remote_python, self.remote_helper, action, "--root", self.remote_root]
        if action == "submit":
            command += ["--request-base64", base64.b64encode(json_bytes(request)).decode("ascii")]
        elif action == "poll":
            if not re.fullmatch(r"[0-9a-f]{32}", request_id) or offset < 0:
                raise ValueError("invalid poll request")
            command += ["--request-id", request_id, "--offset", str(offset)]
        else:
            raise ValueError("transport only supports submit and poll")
        expression = self._expression(shlex.join(command))
        # Windows CreateProcess has a 32767 UTF-16-unit command-line limit.
        if len(expression.encode("utf-16-le")) // 2 + len(self.cli) + 1024 > 30000:
            raise ValueError("OpenCLI expression exceeds the bounded command-line budget")
        identity = request["request_id"] if action == "submit" else request_id
        stage = "lock_wait"
        try:
            with self._locked():
                stage = "cli_run"
                completed = subprocess.run(
                    [self.node, self.cli, "--profile", self.profile, "browser", self.browser_session, "eval", expression],
                    shell=False, capture_output=True, text=True, encoding="utf-8", timeout=DEFAULT_BUDGET.cli,
                )
            stage = "cli_exit"
            if completed.returncode:
                _event("transport_exit", identity, action=action, returncode=completed.returncode)
                raise RuntimeError("OpenCLI failed; command outcome is not established")
            stage = "dsw_response"
            envelope = json.loads(completed.stdout)
            if envelope.get("status") != 200:
                raise RuntimeError("DSW command API failed")
            stage = "job_response"
            body = json.loads(envelope["text"])
            result = json.loads(body["output"])
            if not isinstance(result, dict):
                raise ValueError("invalid remote job response")
            return result
        except Exception as exc:
            _event("transport_error", identity, action=action, stage=stage,
                   exception_type=type(exc).__name__)
            raise


class Bridge:
    def __init__(self, transport, allowed_sessions: list[str], *, timeout_seconds: float = DEFAULT_BUDGET.worker,
                 poll_interval: float = 1.0, clock=time.monotonic, sleep=time.sleep, target_port: int | None = None):
        validate_route("GET", "/health", allowed_sessions)
        validate_request({"request_id": "0" * 32, "method": "GET", "path": "/health", "body": {},
                          "allowed_sessions": allowed_sessions, "timeout_seconds": timeout_seconds,
                          **({} if target_port is None else {"target_port": target_port})})
        self.target_port = target_port
        self.transport, self.allowed_sessions = transport, list(allowed_sessions)
        self.timeout_seconds, self.poll_interval = timeout_seconds, poll_interval
        self.clock, self.sleep = clock, sleep
        self._blocked: set[str] = set()
        self._state_lock = threading.Lock()

    def request(self, method: str, path: str, body: dict) -> tuple[int, bytes]:
        session = validate_route(method, path, self.allowed_sessions)
        with self._state_lock:
            retirement_read = method == "GET" and path == f"/sessions/{session}/retire/v1"
            if session in self._blocked and not retirement_read:
                raise UnknownOutcome("", "session suspended after an unknown request; inspect its existing job before restarting bridge")
        request_id = uuid.uuid4().hex
        request = {"request_id": request_id, "method": method, "path": path, "body": body,
                   "allowed_sessions": self.allowed_sessions, "timeout_seconds": self.timeout_seconds}
        if self.target_port is not None:
            request["target_port"] = self.target_port
        digest = validate_request(request)
        # Reserve submit and final-response transport windows outside GPU work.
        # Do not launch a call that could outlive this bridge request deadline.
        call_budget = getattr(self.transport, "max_call_seconds", 0)
        deadline = self.clock() + self.timeout_seconds + 2 * call_budget + DEFAULT_BUDGET.recovery

        def call(action, *args, **kwargs):
            if self.clock() + call_budget >= deadline:
                raise UnknownOutcome(request_id, "insufficient deadline for another transport call; do not resubmit")
            return self.transport(action, *args, **kwargs)
        _event("request_started", request_id, method=method, path=path)
        try:
            # Exactly one submit attempt. A lost acknowledgement is recovered
            # solely by reading this same UUID, never by dispatching a new job.
            try:
                accepted = call("submit", request)
                if accepted.get("request_id") != request_id or accepted.get("request_sha256") != digest:
                    raise UnknownOutcome(request_id, "submission identity/digest mismatch")
                _event("submit_acknowledged", request_id)
            except UnknownOutcome:
                raise
            except Exception as exc:
                _event("submit_unacknowledged", request_id, stage="submit",
                       exception_type=type(exc).__name__)
            chunks = bytearray()
            expected = None
            poll_failures = 0
            missing_polls = 0
            while self.clock() < deadline:
                try:
                    result = call("poll", request_id=request_id, offset=len(chunks))
                except UnknownOutcome:
                    raise
                except Exception as exc:
                    poll_failures += 1
                    _event("poll_error", request_id, consecutive_failures=poll_failures,
                           exception_type=type(exc).__name__)
                    if poll_failures >= 3:
                        raise UnknownOutcome(request_id, "three read-only polls failed; inspect the existing job before retrying")
                    self.sleep(self.poll_interval)
                    continue
                poll_failures = 0
                if result.get("request_id") != request_id:
                    raise UnknownOutcome(request_id, "response identity mismatch")
                if result.get("state") == "not_found":
                    missing_polls += 1
                    _event("job_not_found", request_id, consecutive_missing_polls=missing_polls)
                    if missing_polls >= 3:
                        raise UnknownOutcome(request_id, "remote job not found in three read-only polls; outcome remains unknown; do not resubmit")
                    self.sleep(self.poll_interval)
                    continue
                missing_polls = 0
                if result.get("state") == "pending":
                    if result.get("request_sha256", digest) != digest:
                        raise UnknownOutcome(request_id, "pending request digest mismatch")
                    self.sleep(self.poll_interval)
                    continue
                if result.get("state") != "done" or result.get("request_sha256") != digest:
                    raise UnknownOutcome(request_id, "remote outcome unknown or request digest mismatch")
                current = (result.get("status"), result.get("bytes"), result.get("sha256"))
                status, length, checksum = current
                if not isinstance(status, int) or not 200 <= status <= 599 or not isinstance(length, int) or not 0 <= length <= MAX_RESPONSE:
                    raise UnknownOutcome(request_id, "invalid remote response metadata")
                if expected is None:
                    expected = current
                if current != expected or result.get("offset") != len(chunks):
                    raise UnknownOutcome(request_id, "response changed while reading chunks")
                try:
                    chunk = base64.b64decode(result["chunk_base64"], validate=True)
                except (KeyError, ValueError) as exc:
                    raise UnknownOutcome(request_id, "invalid response chunk") from exc
                if len(chunk) > MAX_CHUNK or len(chunks) + len(chunk) > length or (not chunk and len(chunks) < length):
                    raise UnknownOutcome(request_id, "truncated or oversized response chunk")
                chunks.extend(chunk)
                if len(chunks) == length:
                    if hashlib.sha256(chunks).hexdigest() != checksum:
                        raise UnknownOutcome(request_id, "response checksum mismatch")
                    _event("request_completed", request_id, status=status, response_bytes=length)
                    return status, bytes(chunks)
            raise UnknownOutcome(request_id, "request wait expired; do not resubmit or assume rollback")
        except UnknownOutcome:
            _event("request_unknown", request_id, session_suspended=bool(session))
            if session:
                with self._state_lock:
                    self._blocked.add(session)
            raise


def make_server(bridge: Bridge, port: int = 18760) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Do not log request bodies, browser output, or credentials.

        def _send(self, status: int, data: bytes):
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _handle(self):
            try:
                if self.headers.get("Origin") or self.headers.get("Transfer-Encoding"):
                    raise ValueError("browser-origin and chunked requests are not accepted")
                validate_route(self.command, self.path, bridge.allowed_sessions)
                length = int(self.headers.get("Content-Length", "0"))
                self.connection.settimeout(10)
                if not 0 <= length <= MAX_BODY:
                    # Consume only a small bounded invalid body before sending
                    # 400. Closing a socket with unread request bytes can reset
                    # it on Windows and hide the rejection response. Never drain
                    # arbitrary large/chunked requests or dispatch these bytes.
                    if MAX_BODY < length <= 2 * MAX_BODY:
                        self.rfile.read(length)
                    raise ValueError("request exceeds 8 KiB")
                if self.command == "POST" and self.headers.get_content_type() != "application/json":
                    raise ValueError("application/json required")
                self.connection.settimeout(10)
                data = self.rfile.read(length)
                if len(data) != length:
                    raise ValueError("truncated request body")
                body = json.loads(data or b"{}")
                status, response = bridge.request(self.command, self.path, body)
                self._send(status, response)
            except UnknownOutcome as exc:
                self._send(502, json_bytes({"error": "outcome_unknown", "request_id": exc.request_id, "message": str(exc)}))
            except (ValueError, TypeError) as exc:
                self._send(400, json_bytes({"error": "invalid_request", "message": str(exc)}))
            except (BrokenPipeError, ConnectionResetError):
                pass  # Client loss never cancels or repeats the remote update.

        do_GET = do_POST = do_DELETE = _handle
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default="/mnt/c/Program Files/nodejs/node.exe")
    parser.add_argument("--cli", required=True, help="Windows path of the installed OpenCLI main.js")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--browser-session", default="ms")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument("--remote-helper", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--session-id", action="append", required=True)
    parser.add_argument("--port", type=int, default=18760)
    parser.add_argument("--target-port", type=int, choices=(8760, 8761, 8762), default=None,
                        help="explicit loopback GPU service; omitted keeps original durable request bytes")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_BUDGET.worker)
    parser.add_argument("--lock-file", type=Path, default=Path.home() / ".cache/reap-http-bridge/opencli.lock")
    args = parser.parse_args()
    transport = OpenCliTransport(**{key: getattr(args, key) for key in (
        "node", "cli", "profile", "browser_session", "instance_id", "remote_python", "remote_helper", "remote_root", "lock_file")})
    bridge = Bridge(transport, args.session_id, timeout_seconds=args.timeout_seconds, target_port=args.target_port)
    server = make_server(bridge, args.port)
    print(json_bytes({"ready": True, "bind": "127.0.0.1", "port": server.server_port,
                      "deadlines_seconds": {**DEFAULT_BUDGET.__dict__,
                          "bridge": DEFAULT_BUDGET.bridge(args.timeout_seconds),
                          "minimum_client": DEFAULT_BUDGET.bridge(args.timeout_seconds) + 60},
                      "instance_id": args.instance_id, "remote_verified": False}).decode(), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
