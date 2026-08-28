#!/usr/bin/env python3
"""Small durable job adapter for a loopback-only GPU HTTP server.

No browser credentials are accepted or stored. submit/poll are short commands;
the detached worker alone waits for the GPU response. An existing job is never
automatically restarted, including after an interrupted submit or worker crash.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

MAX_BODY = 8192
MAX_RESPONSE = 8 * 1024 * 1024
MAX_CHUNK = 4096
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
ACTIONS = {"", "policy/v1/chat/completions", "value/v1/chat/completions", "learn/v1", "snapshot/v1", "restore/v1", "experience/v1", "retire/v1"}


def json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def validate_route(method: str, path: str, allowed: list[str]) -> str | None:
    if not allowed or any(not isinstance(item, str) or not IDENTIFIER.fullmatch(item) for item in allowed):
        raise ValueError("explicit valid session allowlist required")
    if method == "GET" and path == "/health":
        return None
    match = re.fullmatch(r"/sessions/([A-Za-z0-9][A-Za-z0-9_.-]{0,63})(?:/(.*))?", path)
    if not match or match[1] not in allowed:
        raise ValueError("route/session is not allowed")
    action = match[2] or ""
    if not ((method == "POST" and action in ACTIONS) or (method == "DELETE" and action == "")
            or (method == "GET" and action == "retire/v1")):
        raise ValueError("method/action is not allowed")
    return match[1]


def validate_request(request: dict) -> str:
    if not isinstance(request, dict) or not re.fullmatch(r"[0-9a-f]{32}", str(request.get("request_id", ""))):
        raise ValueError("invalid request UUID")
    validate_route(request.get("method"), request.get("path"), request.get("allowed_sessions", []))
    if not isinstance(request.get("body"), dict) or len(json_bytes(request["body"])) > MAX_BODY:
        raise ValueError("request body must be an object of at most 8 KiB")
    timeout = request.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 3600:
        raise ValueError("invalid per-request transport timeout")
    if "target_port" in request and (type(request["target_port"]) is not int or request["target_port"] not in (8760, 8761, 8762)):
        raise ValueError("target_port must explicitly be 8760, 8761, or 8762")
    return hashlib.sha256(json_bytes(request)).hexdigest()


def job_dir(root: Path, request_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", request_id):
        raise ValueError("invalid request UUID")
    root = root.resolve()
    target = (root / request_id).resolve()
    if target.parent != root:
        raise ValueError("job path escapes root")
    return target


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def submit_job(root: Path, request: dict, launcher=None) -> dict:
    digest = validate_request(request)
    target = job_dir(root, request["request_id"])
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        target.mkdir(mode=0o700)
    except FileExistsError:
        # A missing request after mkdir means interrupted submission. Never
        # guess that it is safe to dispatch a second mutation.
        try:
            existing = json.loads((target / "request.json").read_bytes())
        except (OSError, ValueError) as exc:
            raise RuntimeError("existing job has an unknown submission outcome") from exc
        if validate_request(existing) != digest:
            raise ValueError("request UUID reused with a different payload")
        return {"state": "accepted", "request_id": request["request_id"], "request_sha256": digest, "duplicate": True}
    atomic_write(target / "request.json", json_bytes(request))
    command = [sys.executable, str(Path(__file__).resolve()), "run", "--root", str(target.parent), "--request-id", request["request_id"]]
    if launcher is not None:
        launcher(command)
    else:
        with (target / "worker.log").open("ab") as log:
            subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    return {"state": "accepted", "request_id": request["request_id"], "request_sha256": digest, "duplicate": False}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def run_job(root: Path, request_id: str, opener=None) -> None:
    target = job_dir(root, request_id)
    request = json.loads((target / "request.json").read_bytes())
    digest = validate_request(request)
    try:
        fd = os.open(target / "running.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    os.close(fd)
    result = {"request_id": request_id, "request_sha256": digest}
    response_body = b""
    try:
        raw_body = json_bytes(request["body"]) if request["method"] == "POST" else None
        # The optional port is part of the immutable request digest. Existing
        # requests without it retain their original 8760 payload and identity.
        http_request = Request(f"http://127.0.0.1:{request.get('target_port', 8760)}" + request["path"], data=raw_body,
                               method=request["method"], headers={"Content-Type": "application/json"})
        open_request = opener or build_opener(ProxyHandler({}), NoRedirect()).open
        try:
            response = open_request(http_request, timeout=request["timeout_seconds"])
        except HTTPError as exc:
            response = exc  # Preserve B's 400/409/etc. response, not a transport error.
        with response:
            response_body = response.read(MAX_RESPONSE + 1)
            if len(response_body) > MAX_RESPONSE:
                raise ValueError("GPU response exceeds 8 MiB")
            result.update(state="done", status=response.status)
    except Exception as exc:
        # A timeout/disconnect cannot establish whether a mutation committed.
        # Do not retry, restart this job, or claim an update was unapplied.
        result.update(state="unknown", error=type(exc).__name__)
    result.update(bytes=len(response_body), sha256=hashlib.sha256(response_body).hexdigest(), completed_at=time.time())
    atomic_write(target / "response.bin", response_body)
    atomic_write(target / "result.json", json_bytes(result))


def poll_job(root: Path, request_id: str, offset: int = 0, chunk_bytes: int = MAX_CHUNK) -> dict:
    if isinstance(offset, bool) or offset < 0 or not 1 <= chunk_bytes <= MAX_CHUNK:
        raise ValueError("invalid response chunk range")
    target = job_dir(root, request_id)
    if not target.exists():
        return {"state": "not_found", "request_id": request_id}
    request_path = target / "request.json"
    if not request_path.exists():
        return {"state": "pending", "request_id": request_id}
    digest = validate_request(json.loads(request_path.read_bytes()))
    result_path = target / "result.json"
    if not result_path.exists():
        return {"state": "pending", "request_id": request_id, "request_sha256": digest}
    result = json.loads(result_path.read_bytes())
    if offset > result["bytes"]:
        raise ValueError("response offset exceeds length")
    with (target / "response.bin").open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read(chunk_bytes)
    return {**result, "offset": offset, "chunk_base64": base64.b64encode(chunk).decode("ascii")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "poll", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--request-base64")
    parser.add_argument("--request-id")
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if args.action == "submit":
        request = json.loads(base64.b64decode(args.request_base64, validate=True))
        result = submit_job(args.root, request)
    elif args.action == "poll":
        result = poll_job(args.root, args.request_id, args.offset)
    else:
        run_job(args.root, args.request_id)
        return 0
    print(json_bytes(result).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
