#!/usr/bin/env python3
"""Download five official REAL-Prover LFS objects from private expiring URLs.

Only validated filenames and sanitized progress are written. This is a weights
transfer, not validation of the complete 16-file model and not a TTT result.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import socket
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

REPO = "FrenzyMath/REAL-Prover"
REVISION = "fe76f68d9a88f342cb7b546307c20292fea9cced"
ALLOWED_HOSTS = {"us.aws.cdn.hf.co"}
LFS_NAMES = {f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)} | {"tokenizer.json"}
CHUNK_BYTES = 1024 * 1024


class TransferError(Exception):
    """Contains a constant safe code, never a URL or underlying exception text."""


def require(condition, code):
    if not condition:
        raise TransferError(code)


def validate_url(url):
    try:
        parsed = urlparse(url)
        require(parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS and parsed.port in (None, 443)
                and not parsed.username and not parsed.password and not parsed.fragment, "unapproved_download_url")
    except (TypeError, ValueError):
        raise TransferError("invalid_download_url") from None
    return url


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def load_manifest(manifest_path, expected_hash):
    content = Path(manifest_path).read_bytes()
    require(re.fullmatch(r"[0-9a-f]{64}", expected_hash) and hashlib.sha256(content).hexdigest() == expected_hash,
            "official_manifest_sha256_mismatch")
    manifest = json.loads(content.decode("utf-8-sig"))
    require(manifest.get("schema_version") == "reap.official-model-manifest.v1"
            and manifest.get("repo") == REPO and manifest.get("revision") == REVISION
            and manifest.get("source_api") == f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true",
            "official_manifest_identity_mismatch")
    files = {name: entry for name, entry in manifest["files"].items() if "lfs_sha256" in entry}
    require(set(files) == LFS_NAMES, "expected_exactly_five_known_LFS_files")
    for entry in files.values():
        require(type(entry.get("size")) is int and entry["size"] > 0
                and isinstance(entry.get("lfs_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", entry["lfs_sha256"]), "invalid_official_LFS_metadata")
    return files


def load_inputs(manifest_path, expected_hash, urls_path):
    files = load_manifest(manifest_path, expected_hash)
    private = json.loads(Path(urls_path).read_text(encoding="utf-8-sig"))
    require(private.get("repo") == REPO and private.get("revision") == REVISION, "URL_file_identity_mismatch")
    urls = private.get("urls")
    require(isinstance(urls, dict) and set(urls) == LFS_NAMES, "URL_file_names_mismatch")
    for url in urls.values():
        validate_url(url)
    return files, urls


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


class Progress:
    def __init__(self, path, files):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.guard = threading.Lock()
        self.last_write = 0.0
        self.data = {"schema_version": 1, "repo": REPO, "revision": REVISION,
                     "status": "running", "weights_only": True, "model_lock_written": False,
                     "started_at_unix": time.time(), "completed": [],
                     "files": {name: {"state": "pending", "bytes": 0, "total": entry["size"]} for name, entry in files.items()}}
        self.update(force=True)

    def update(self, name=None, *, force=False, overall=None, **fields):
        with self.guard:
            if name is not None:
                self.data["files"][name].update(fields)
            if overall is not None:
                self.data["status"] = overall
            if not force and time.monotonic() - self.last_write < 1:
                return
            self.data["completed"] = sorted(name for name, entry in self.data["files"].items() if entry["state"] == "complete")
            self.data["updated_at_unix"] = time.time()
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                 prefix=".transfer-progress-", suffix=".tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                    json.dump(self.data, handle, sort_keys=True, allow_nan=False)
                    handle.write("\n")
                os.replace(temporary, self.path)
                self.last_write = time.monotonic()
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def stream_response(response, partial, expected_size, offset, progress, name):
    status = response.status
    require(status in (200, 206), "unexpected_HTTP_status")
    if status == 206:
        match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", response.headers.get("Content-Range", ""))
        require(match is not None, "missing_or_invalid_Content_Range")
        begin, end, total = map(int, match.groups())
        require(begin == offset and total == expected_size and begin <= end < total, "Content_Range_mismatch")
        response_bytes = end - begin + 1
        mode = "ab" if offset else "wb"
    else:
        # Some CDNs ignore Range. Replacing the partial is mandatory; appending
        # a full 200 response to old bytes would silently corrupt the model.
        offset, response_bytes, mode = 0, expected_size, "wb"
    length = response.headers.get("Content-Length")
    require(length is None or (length.isdigit() and int(length) == response_bytes), "Content_Length_mismatch")
    encoding = response.headers.get("Content-Encoding", "identity").lower()
    require(encoding == "identity", "unexpected_Content_Encoding")
    received = 0
    with partial.open(mode) as target:
        progress.update(name, bytes=offset, state="downloading", force=True)
        while chunk := response.read(CHUNK_BYTES):
            require(received + len(chunk) <= response_bytes and offset + received + len(chunk) <= expected_size,
                    "response_exceeds_official_size")
            target.write(chunk)
            received += len(chunk)
            progress.update(name, bytes=offset + received)
        target.flush()
        os.fsync(target.fileno())
    require(received == response_bytes, "response_truncated")
    return offset + received


def transfer_one(name, entry, url, output, progress, *, attempts=3, opener=None, sleep=time.sleep):
    final = output / name
    partial = output / (name + ".partial")
    expected_size, expected_hash = entry["size"], entry["lfs_sha256"]
    opener = opener or build_opener(SafeRedirect())
    try:
        require(not final.is_symlink() and not partial.is_symlink(), "symlink_model_file_rejected")
        if final.exists() and final.stat().st_size == expected_size:
            progress.update(name, state="verifying_existing", bytes=expected_size, force=True)
            if digest_file(final) == expected_hash:
                progress.update(name, state="complete", bytes=expected_size, sha256_verified=True, force=True)
                return True
        for attempt in range(1, attempts + 1):
            offset = partial.stat().st_size if partial.exists() else 0
            require(offset <= expected_size, "partial_exceeds_official_size")
            progress.update(name, attempt=attempt, state="connecting", bytes=offset, error=None, force=True)
            try:
                if offset < expected_size:
                    request = Request(validate_url(url), headers={"Range": f"bytes={offset}-", "Accept-Encoding": "identity",
                                                                "User-Agent": "reap-public-model-transfer/1"})
                    with opener.open(request, timeout=30) as response:
                        offset = stream_response(response, partial, expected_size, offset, progress, name)
                if offset != expected_size:
                    raise TransferError("response_truncated")
                progress.update(name, state="verifying_sha256", bytes=offset, force=True)
                require(digest_file(partial) == expected_hash, "official_LFS_SHA256_mismatch")
                os.replace(partial, final)
                progress.update(name, state="complete", bytes=expected_size, sha256_verified=True, error=None, force=True)
                return True
            except (HTTPError, URLError, HTTPException, TimeoutError, socket.timeout, ConnectionError, OSError, TransferError) as error:
                if isinstance(error, TransferError):
                    code = str(error)
                    retryable = code == "response_truncated"
                    details = {"code": code}
                elif isinstance(error, HTTPError):
                    details = {"code": "HTTP_error", "status": error.code}
                    retryable = error.code in (408, 429) or error.code >= 500
                    error.close()
                else:
                    details = {"code": "network_or_IO_error", "type": type(error).__name__}
                    retryable = True
                remaining = partial.stat().st_size if partial.exists() else 0
                progress.update(name, state="retrying" if retryable and attempt < attempts else "failed",
                                bytes=remaining, error=details, force=True)
                if not retryable or attempt == attempts:
                    return False
                sleep(min(attempt * 2, 6))
    except Exception as error:
        progress.update(name, state="failed", error={"code": str(error) if isinstance(error, TransferError) else "local_error",
                                                     "type": type(error).__name__}, force=True)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--urls", type=Path, required=True, help="private temporary signed URL JSON; never copied to progress")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True, help="JSON progress outside model output directory")
    args = parser.parse_args()
    try:
        files, urls = load_inputs(args.manifest, args.manifest_sha256, args.urls)
        require(not args.progress.resolve().is_relative_to(args.output.resolve()), "progress_must_be_outside_model_directory")
        require(not args.urls.resolve().is_relative_to(args.output.resolve()), "private_URL_file_must_be_outside_model_directory")
        args.output.mkdir(parents=True, exist_ok=True)
        progress = Progress(args.progress, files)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(transfer_one, name, entry, urls[name], args.output, progress) for name, entry in files.items()]
            results = [future.result() for future in futures]
        complete = all(results)
        progress.update(force=True, overall="complete" if complete else "failed")
        print(json.dumps({"completed": sum(results), "total": len(results), "weights_only": True, "model_lock_written": False}))
        return 0 if complete else 1
    except Exception as error:
        print(json.dumps({"status": "failed", "code": str(error) if isinstance(error, TransferError) else "local_error",
                          "error_type": type(error).__name__, "model_lock_written": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
