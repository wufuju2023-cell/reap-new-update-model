#!/usr/bin/env python3
"""Download a fixed REAL-Prover revision and validate against official HF hashes.

An explicit mirror endpoint supplies bytes only. --manifest accepts an OFFICIAL
manifest previously fetched on a trusted machine, not metadata from the mirror.
Use --manifest-sha256 to authenticate that file against a separately kept digest.
Verification and manifest-only mode need only the Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import quote, urlparse
from urllib.request import urlopen

SCHEMA = "reap.official-model-manifest.v1"
LOCK_NAME = "reap-model-lock.json"
OFFICIAL_ENDPOINT = "https://huggingface.co"
CHUNK_BYTES = 8 * 1024 * 1024
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
               "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)


def official_url(repo, revision):
    require(isinstance(repo, str) and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo), "invalid model repository")
    require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision), "revision must be a full lowercase 40-hex commit, not main/tag")
    return f"{OFFICIAL_ENDPOINT}/api/models/{quote(repo, safe='/')}/revision/{revision}?blobs=true"


def safe_name(name):
    require(isinstance(name, str) and bool(name), "empty manifest filename")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name and all(part not in {".", ".."} for part in path.parts)
            and "\\" not in name and ":" not in name and "\x00" not in name,
            "unsafe manifest filename")
    require(name != LOCK_NAME and path.parts[0] != ".cache", "reserved manifest filename")
    return path


def normalize_official_metadata(raw, repo, revision):
    source = official_url(repo, revision)
    require(raw.get("id") == repo and raw.get("sha") == revision, "official API repo/revision mismatch")
    require(raw.get("private") is False and raw.get("gated") is False, "expected public, ungated model")
    files = {}
    for sibling in raw.get("siblings", []):
        name = sibling["rfilename"]
        safe_name(name)
        require(name not in files, "duplicate official filename")
        entry = {"size": sibling.get("size"), "git_blob_sha1": sibling.get("blobId")}
        if sibling.get("lfs"):
            require(sibling["lfs"].get("size") == entry["size"], "LFS size disagrees with official file size")
            entry["lfs_sha256"] = sibling["lfs"].get("sha256")
        files[name] = entry
    manifest = {"schema_version": SCHEMA, "source_api": source, "repo": repo, "revision": revision, "files": files}
    validate_manifest(manifest, repo, revision)
    return manifest


def validate_manifest(manifest, repo, revision):
    source = official_url(repo, revision)
    require(manifest.get("schema_version") == SCHEMA and manifest.get("source_api") == source,
            "manifest must be the normalized official HF API manifest")
    require(manifest.get("repo") == repo and manifest.get("revision") == revision, "manifest repo/revision mismatch")
    files = manifest.get("files")
    require(isinstance(files, dict) and bool(files), "official manifest has no files")
    require({"config.json", "model.safetensors.index.json"} <= files.keys(), "official manifest lacks config or shard index")
    for name, entry in files.items():
        safe_name(name)
        require(isinstance(entry, dict), "invalid official file entry")
        require(type(entry.get("size")) is int and entry["size"] > 0, "invalid official file size")
        require(isinstance(entry.get("git_blob_sha1"), str) and re.fullmatch(r"[0-9a-f]{40}", entry["git_blob_sha1"]), "invalid official git blob SHA1")
        if "lfs_sha256" in entry:
            require(isinstance(entry["lfs_sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["lfs_sha256"]), "invalid official LFS SHA256")
    return manifest


def fetch_official_manifest(repo, revision):
    # Never honor HF_ENDPOINT for this trust source, and never attach a token.
    source = official_url(repo, revision)
    with urlopen(source, timeout=30) as response:
        require(response.geturl() == source, "official metadata request redirected away from its fixed URL")
        data = response.read(10 * 1024 * 1024 + 1)
    require(len(data) <= 10 * 1024 * 1024, "official metadata response too large")
    return normalize_official_metadata(json.loads(data, object_pairs_hook=unique_object), repo, revision)


def write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=".reap-manifest-", suffix=".tmp", delete=False) as handle:
            name = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def file_hashes(path):
    before = path.stat()
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {before.st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            sha256.update(chunk)
            blob.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "model file changed during verification")
    return {"size": after.st_size, "sha256": sha256.hexdigest(), "git_blob_sha1": blob.hexdigest()}


def safetensors_header(path):
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        require(len(prefix) == 8, "truncated safetensors prefix")
        length = int.from_bytes(prefix, "little")
        require(0 < length <= min(100_000_000, size - 8), "invalid/truncated safetensors header length")
        encoded = handle.read(length)
    require(encoded.startswith(b"{"), "safetensors header must begin with JSON object")
    header = json.loads(encoded, object_pairs_hook=unique_object)
    require(isinstance(header, dict), "safetensors header is not an object")
    data_size = size - 8 - length
    ranges, tensors = [], {}
    for name, entry in header.items():
        if name == "__metadata__":
            require(isinstance(entry, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in entry.items()), "invalid safetensors metadata")
            continue
        require(isinstance(entry, dict), "invalid tensor header")
        shape, offsets, dtype = entry.get("shape"), entry.get("data_offsets"), entry.get("dtype")
        require(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape), "invalid tensor shape")
        require(isinstance(dtype, str) and dtype in DTYPE_BYTES, "unsupported tensor dtype in REAL-Prover header")
        require(isinstance(offsets, list) and len(offsets) == 2 and all(type(n) is int for n in offsets), "invalid tensor offsets")
        begin, end = offsets
        require(0 <= begin <= end <= data_size, "tensor offsets exceed shard payload")
        require(end - begin == math.prod(shape) * DTYPE_BYTES[dtype], "tensor shape/dtype byte length mismatch")
        ranges.append((begin, end))
        tensors[name] = {"dtype": dtype, "shape": shape, "bytes": end - begin}
    require(bool(tensors), "safetensors shard contains no tensors")
    cursor = 0
    for begin, end in sorted(ranges):
        require(begin == cursor, "safetensors payload contains overlapping tensors or holes")
        cursor = end
    require(cursor == data_size, "safetensors payload has unindexed trailing bytes")
    return tensors, data_size


def validate_and_lock(output, manifest, repo, revision, expected_hidden_size=3584):
    output = Path(output)
    lock_path = output / LOCK_NAME
    lock_path.unlink(missing_ok=True)
    validate_manifest(manifest, repo, revision)
    require(expected_hidden_size == 3584, "this REAL-Prover gate requires hidden_size 3584")
    root = output.resolve()
    actual = {}
    for name, expected in sorted(manifest["files"].items()):
        path = output.joinpath(*safe_name(name).parts)
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), f"missing/nonlocal model file: {name}")
        require(path.stat().st_size == expected["size"], f"file size mismatch: {name}")
        hashes = file_hashes(path)
        if "lfs_sha256" in expected:
            require(hashes["sha256"] == expected["lfs_sha256"], f"official LFS SHA256 mismatch: {name}")
        else:
            require(hashes["git_blob_sha1"] == expected["git_blob_sha1"], f"official git blob SHA1 mismatch: {name}")
        actual[name] = {**hashes, "official_hash_kind": "lfs_sha256" if "lfs_sha256" in expected else "git_blob_sha1"}
    # Exclude only downloader cache bookkeeping, not unexpected model artifacts.
    for path in output.rglob("*"):
        if path.is_file() and path.relative_to(output).parts[0] != ".cache":
            require(path.relative_to(output).as_posix() in actual, "unmanifested file in model directory")
    config = read_json(output / "config.json")
    require(type(config.get("hidden_size")) is int and config["hidden_size"] == expected_hidden_size, "REAL-Prover hidden_size mismatch")
    index = read_json(output / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict) and bool(weight_map), "missing safetensors weight_map")
    shards = set(weight_map.values())
    require(all(isinstance(shard, str) and shard.endswith(".safetensors") and shard in actual for shard in shards), "index references a missing/untrusted shard")
    require(shards == {name for name in actual if name.endswith(".safetensors")}, "index/shard file set mismatch")
    found, total_bytes, shard_reports = {}, 0, {}
    for shard in sorted(shards):
        tensors, data_bytes = safetensors_header(output / shard)
        for name in tensors:
            require(name not in found and weight_map.get(name) == shard, "tensor index/header mismatch or duplicate tensor")
            found[name] = shard
        total_bytes += data_bytes
        shard_reports[shard] = {"tensors": len(tensors), "payload_bytes": data_bytes}
    require(found == weight_map, "index contains tensors absent from shard headers")
    total_declared = index.get("metadata", {}).get("total_size")
    require(type(total_declared) is int and total_declared == total_bytes, "safetensors index total_size mismatch")
    manifest_digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    lock = {"schema_version": "reap.model-lock.v2", "repo": repo, "revision": revision,
            "hidden_size": config["hidden_size"], "model_type": config.get("model_type"), "architectures": config.get("architectures"),
            "verified_against": {"source_api": manifest["source_api"], "canonical_manifest_sha256": manifest_digest},
            "files": actual, "safetensors": {"shards": shard_reports, "tensor_count": len(found), "total_tensor_bytes": total_bytes}}
    write_json_atomic(lock_path, lock)
    return lock


def endpoint_value(value):
    parsed = urlparse(value)
    require(parsed.scheme == "https" and bool(parsed.hostname) and not parsed.username and not parsed.password
            and not parsed.query and not parsed.fragment and parsed.path in {"", "/"}, "endpoint must be a plain HTTPS origin without credentials")
    return value.rstrip("/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-hidden-size", type=int, default=3584)
    parser.add_argument("--endpoint", default=OFFICIAL_ENDPOINT)
    parser.add_argument("--manifest", type=Path, help="trusted manifest previously fetched from official HF API")
    parser.add_argument("--manifest-sha256", help="SHA256 of the trusted manifest file, obtained separately")
    parser.add_argument("--save-manifest", type=Path)
    parser.add_argument("--manifest-only", action="store_true", help="fetch/save metadata only; never download model files")
    parser.add_argument("--verify-only", action="store_true", help="validate existing files without downloading")
    args = parser.parse_args()
    official_url(args.repo, args.revision)
    require(not (args.manifest_only and args.verify_only), "choose manifest-only or verify-only")
    require(args.expected_hidden_size == 3584, "this REAL-Prover gate requires hidden_size 3584")
    require(args.manifest_only or args.output is not None, "--output is required except in manifest-only mode")
    require(not args.manifest_only or args.save_manifest is not None, "manifest-only requires --save-manifest")
    require(not args.manifest_sha256 or args.manifest is not None, "manifest-sha256 requires --manifest")
    endpoint = endpoint_value(args.endpoint)
    if args.output and not args.manifest_only:
        (args.output / LOCK_NAME).unlink(missing_ok=True)
    if args.manifest:
        if args.manifest_sha256:
            require(hashlib.sha256(args.manifest.read_bytes()).hexdigest() == args.manifest_sha256, "trusted manifest file SHA256 mismatch")
        manifest = validate_manifest(read_json(args.manifest), args.repo, args.revision)
    else:
        manifest = fetch_official_manifest(args.repo, args.revision)
    if args.save_manifest:
        write_json_atomic(args.save_manifest, manifest)
    if args.manifest_only:
        print(json.dumps({"manifest_only": True, "repo": args.repo, "revision": args.revision,
                          "file_count": len(manifest["files"]), "total_bytes": sum(entry["size"] for entry in manifest["files"].values()),
                          "manifest_sha256": hashlib.sha256(args.save_manifest.read_bytes()).hexdigest()}))
        return 0
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        args.output.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=args.output,
                          endpoint=endpoint, token=False, allow_patterns=list(manifest["files"]))
    lock = validate_and_lock(args.output, manifest, args.repo, args.revision, args.expected_hidden_size)
    print(json.dumps(lock, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
