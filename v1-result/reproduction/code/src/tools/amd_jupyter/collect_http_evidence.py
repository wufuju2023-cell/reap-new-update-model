#!/usr/bin/env python3
"""Read existing HTTP jobs into a new compressed evidence snapshot; never send HTTP.

Original jobs are read only.  Only the explicitly named new snapshot/assembled
output is written.  No browser, credentials, model, Torch or runtime imports.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

SCHEMA = "reap.http-evidence.collection.v1"
CHUNK_SCHEMA = "reap.http-evidence.chunk.v1"
UUID = re.compile(r"[0-9a-f]{32}")
SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
ROUTE = re.compile(r"/sessions/([A-Za-z0-9][A-Za-z0-9_.-]{0,63})(?:/(.*))?")
ACTIONS = {"", "policy/v1/chat/completions", "value/v1/chat/completions", "learn/v1", "snapshot/v1", "restore/v1", "experience/v1", "retire/v1"}
REQUEST_KEYS = {"request_id", "method", "path", "body", "allowed_sessions", "timeout_seconds", "target_port"}
LIMITS = {"request.json": 64 * 1024, "result.json": 16 * 1024, "response.bin": 8 * 1024 * 1024}
MAX_RAW_TOTAL = 64 * 1024 * 1024
MAX_JSON = 128 * 1024 * 1024
PAGE_BYTES = 24 * 1024
MAX_STDOUT = 40 * 1024


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_stable(path: Path, limit: int) -> bytes | None:
    if not path.exists():
        require(not path.is_symlink(), "dangling symlink refused")
        return None
    require(not path.is_symlink() and path.is_file(), "evidence must be an ordinary file")
    before = path.stat()
    require(before.st_size <= limit, "evidence file exceeds explicit size limit")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    after = path.stat()
    require(len(data) <= limit and (before.st_size, before.st_mtime_ns) ==
            (after.st_size, after.st_mtime_ns) and len(data) == after.st_size,
            "job changed while being collected; stop bridge/workers and retry with a new snapshot")
    return data


def blob(data: bytes | None):
    if data is None:
        return {"exists": False}
    return {"exists": True, "bytes": len(data), "sha256": sha(data),
            "base64": base64.b64encode(data).decode("ascii")}


def decode_blob(item):
    if item.get("exists") is False:
        return None
    require(item.get("exists") is True, "invalid blob presence")
    data = base64.b64decode(item["base64"], validate=True)
    require(len(data) == item["bytes"] and sha(data) == item["sha256"], "blob hash/length mismatch")
    return data


def collect(jobs_root: Path, output: Path, sessions: list[str]):
    require(sessions and all(isinstance(sid, str) and SESSION.fullmatch(sid) for sid in sessions)
            and len(set(sessions)) == len(sessions),
            "sessions must be explicit, unique, valid identifiers")
    require(not jobs_root.is_symlink(), "symlink jobs root refused")
    jobs_root = jobs_root.resolve(strict=True)
    output = output.resolve()
    require(jobs_root.is_dir() and output != jobs_root and jobs_root not in output.parents,
            "snapshot output must be outside the original jobs tree")
    require(not output.exists() and output.suffix == ".gz", "snapshot must be a new .gz file")
    directories = sorted(p for p in jobs_root.iterdir() if UUID.fullmatch(p.name))
    entries, excluded, count, raw_total = [], [], Counter(), 0
    for directory in directories:
        require(not directory.is_symlink() and directory.is_dir(), "job directory symlink/non-directory refused")
        request_raw = read_stable(directory / "request.json", LIMITS["request.json"])
        if request_raw is None:
            excluded.append({"request_id": directory.name, "reason": "missing_request"})
            continue
        try:
            request = json.loads(request_raw)
        except ValueError:
            raise ValueError("malformed job request; no snapshot emitted") from None
        require(isinstance(request, dict) and set(request) <= REQUEST_KEYS,
                "unexpected request fields; refuse exporting unknown/authentication fields")
        if "target_port" in request:
            require(type(request["target_port"]) is int and request["target_port"] in (8760, 8761, 8762),
                    "invalid explicit replica target port")
        match = ROUTE.fullmatch(str(request.get("path", "")))
        if not match or match[1] not in sessions:
            excluded.append({"request_id": directory.name, "reason": "outside_session_allowlist"})
            continue
        action, method = match[2] or "", request.get("method")
        require((method == "POST" and action in ACTIONS) or (method == "DELETE" and action == "")
                or (method == "GET" and action == "retire/v1"), "unexpected session action")
        allowed = request.get("allowed_sessions")
        require(isinstance(allowed, list) and match[1] in allowed,
                "selected session was not authorized by the retained request")
        require(request.get("request_id") == directory.name, "job/request UUID mismatch")
        data = {"request.json": request_raw}
        data.update({name: read_stable(directory / name, limit) for name, limit in LIMITS.items() if name != "request.json"})
        raw_total += sum(len(value) for value in data.values() if value is not None)
        require(raw_total <= MAX_RAW_TOTAL, "collection exceeds 64 MiB raw limit; collect fewer sessions")
        # Recheck immutable request after reading the other files.
        require(read_stable(directory / "request.json", LIMITS["request.json"]) == request_raw,
                "request changed during collection")
        state, integrity = "pending", None
        if data["result.json"] is not None:
            result = json.loads(data["result.json"])
            require(result.get("request_id") == directory.name and result.get("request_sha256") == sha(json_bytes(request)),
                    "result/request identity mismatch")
            state = result.get("state")
            require(state in ("done", "unknown"), "unexpected durable result state")
            response = data["response.bin"]
            require(response is not None and len(response) == result.get("bytes") and sha(response) == result.get("sha256"),
                    "result/response bytes or SHA mismatch")
            integrity = True
        entries.append({"request_id": directory.name, "session_id": match[1], "state": state,
                        "result_response_hash_matches": integrity,
                        "files": {name: blob(value) for name, value in data.items()}})
        count[match[1]] += 1
    require([p.name for p in directories] == sorted(p.name for p in jobs_root.iterdir() if UUID.fullmatch(p.name)),
            "job directory set changed during collection; collect again after stopping submissions")
    require(all(count[sid] > 0 for sid in sessions), "requested session has no matching retained jobs")
    document = {"schema_version": SCHEMA, "collected_at_epoch": time.time(),
                "jobs_root": str(jobs_root), "allowed_sessions": sessions,
                "source_jobs_modified": False, "credential_sources_read": False,
                "entries": entries, "excluded": excluded,
                "note": "Fixed export of observed files; pending/unknown jobs remain pending/unknown. No request is reissued."}
    encoded = json_bytes(document)
    require(len(encoded) <= MAX_JSON, "encoded snapshot exceeds safety bound")
    compressed = gzip.compress(encoded, mtime=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(compressed)
    return {"snapshot": str(output), "encoding": "gzip-json", "sha256": sha(compressed),
            "bytes": len(compressed), "json_sha256": sha(encoded), "json_bytes": len(encoded),
            "page_bytes": PAGE_BYTES, "pages": (len(compressed) + PAGE_BYTES - 1) // PAGE_BYTES,
            "selected_jobs": len(entries), "jobs_by_session": dict(count), "excluded_jobs": len(excluded),
            "states": dict(Counter(e["state"] for e in entries))}


def read_page(snapshot: Path, expected_sha: str, offset: int):
    require(offset >= 0 and offset % PAGE_BYTES == 0, "offset must be a nonnegative page boundary")
    data = read_stable(snapshot, MAX_JSON)
    require(data is not None and sha(data) == expected_sha, "snapshot hash mismatch")
    require(data.startswith(b"\x1f\x8b") and offset < len(data), "invalid gzip snapshot or offset")
    chunk = data[offset:offset + PAGE_BYTES]
    result = {"schema_version": CHUNK_SCHEMA, "snapshot_sha256": expected_sha,
              "total_bytes": len(data), "offset": offset, "chunk_bytes": len(chunk),
              "chunk_sha256": sha(chunk), "base64": base64.b64encode(chunk).decode("ascii"),
              "next_offset": offset + len(chunk), "done": offset + len(chunk) == len(data)}
    require(len(json_bytes(result)) <= MAX_STDOUT, "page output exceeds 40 KiB")
    return result


def assemble(chunks: list[Path], output: Path, expected_sha: str):
    pages = [json.loads(p.read_bytes()) for p in chunks]
    require(pages, "no pages supplied")
    pages.sort(key=lambda p: p["offset"])
    data = bytearray()
    total = pages[0]["total_bytes"]
    require(total <= MAX_JSON, "snapshot exceeds safety bound")
    for index, page in enumerate(pages):
        require(page["schema_version"] == CHUNK_SCHEMA and page["snapshot_sha256"] == expected_sha
                and page["total_bytes"] == total and page["offset"] == len(data), "page gap/duplicate/identity mismatch")
        chunk = base64.b64decode(page["base64"], validate=True)
        require(len(chunk) == page["chunk_bytes"] and sha(chunk) == page["chunk_sha256"], "chunk corruption")
        data.extend(chunk)
        require(page["next_offset"] == len(data) and page["done"] == (len(data) == total), "page boundary mismatch")
    require(len(data) == total and sha(data) == expected_sha, "incomplete/corrupt snapshot")
    # Stream decompression with a limit, avoiding unbounded gzip expansion.
    import io
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
        raw = handle.read(MAX_JSON + 1)
    require(len(raw) <= MAX_JSON, "decompressed snapshot exceeds safety bound")
    document = json.loads(raw)
    require(document.get("schema_version") == SCHEMA, "wrong collection schema")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(raw)
    return {"collection": str(output), "sha256": sha(raw), "bytes": len(raw),
            "jobs": len(document["entries"]), "archive_sha256": expected_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--jobs-root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--session-id", action="append", required=True)
    read = commands.add_parser("read")
    read.add_argument("--snapshot", type=Path, required=True)
    read.add_argument("--sha256", required=True)
    read.add_argument("--offset", type=int, required=True)
    join = commands.add_parser("assemble")
    join.add_argument("--chunk", type=Path, action="append", required=True)
    join.add_argument("--output", type=Path, required=True)
    join.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "create": result = collect(args.jobs_root, args.output, args.session_id)
    elif args.command == "read": result = read_page(args.snapshot, args.sha256, args.offset)
    else: result = assemble(args.chunk, args.output, args.sha256)
    print(json_bytes(result).decode("utf-8"), flush=True)


if __name__ == "__main__":
    main()
