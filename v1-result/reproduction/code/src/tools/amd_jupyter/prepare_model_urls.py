#!/usr/bin/env python3
"""Prepare private expiring URLs using HEAD only; never fetch weight bodies.

Run locally on a POSIX filesystem that enforces chmod 0600 (for example WSL
/tmp, not a Windows-mounted directory). Existing output files are never replaced.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

try:
    from .download_from_urls import LFS_NAMES, REPO, REVISION, TransferError, load_manifest, require, validate_url
except ImportError:  # Direct `python3 tools/amd_jupyter/prepare_model_urls.py`.
    from download_from_urls import LFS_NAMES, REPO, REVISION, TransferError, load_manifest, require, validate_url


class NoFollowRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def head_url(filename, *, opener=None):
    require(filename in LFS_NAMES, "HEAD_filename_not_in_fixed_LFS_allowlist")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{quote(filename, safe='')}"
    request = Request(url, method="HEAD", headers={"Accept-Encoding": "identity", "User-Agent": "reap-public-model-HEAD/1"})
    opener = opener or build_opener(NoFollowRedirect())
    try:
        response = opener.open(request, timeout=15)
    except HTTPError as response:
        try:
            code, location = response.code, response.headers.get("Location")
        finally:
            response.close()
    else:
        try:
            code, location = response.status, response.headers.get("Location")
        finally:
            response.close()
    require(code in (301, 302, 303, 307, 308), "official_HEAD_did_not_return_redirect")
    require(isinstance(location, str) and bool(location), "official_HEAD_missing_Location")
    return validate_url(location)


def prepare_urls(manifest_path, manifest_sha256, output, *, head=head_url):
    files = load_manifest(manifest_path, manifest_sha256)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the path exclusively and verify privacy BEFORE obtaining/writing
    # bearer URLs. chmod alone does not provide Unix privacy on Windows/DrvFS.
    descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        else:
            os.chmod(output, 0o600)
        require(os.name == "posix" and stat.S_IMODE(os.fstat(descriptor).st_mode) == 0o600,
                "output_filesystem_cannot_enforce_0600_use_WSL_native_tmp")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {name: executor.submit(head, name) for name in sorted(files)}
            urls = {name: validate_url(future.result()) for name, future in futures.items()}
        payload = {"repo": REPO, "revision": REVISION, "urls": urls,
                   "fetched_at": datetime.now(timezone.utc).isoformat()}
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = None
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return {"repo": REPO, "revision": REVISION, "url_count": len(urls), "file_mode": "0600",
                "fetched_at": payload["fetched_at"], "weight_bodies_downloaded": False}
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        output.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new private URL JSON, on a POSIX filesystem enforcing 0600")
    args = parser.parse_args()
    try:
        result = prepare_urls(args.manifest, args.manifest_sha256, args.output)
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "code": str(error) if isinstance(error, TransferError) else "local_or_HEAD_request_failed",
                          "weight_bodies_downloaded": False}))
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
