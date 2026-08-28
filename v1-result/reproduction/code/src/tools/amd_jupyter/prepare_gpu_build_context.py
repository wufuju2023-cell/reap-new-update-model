#!/usr/bin/env python3
"""Prepare an existing-model B build context on an authorized remote builder.

No downloads, container execution, or registry pushes occur. Do not run this
against full weights on the local PC: the model must remain on the approved
remote builder. Small fixture tests do not prove a 7B image or GPU runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from containers.gpu.download_model import validate_manifest, validate_and_lock
from tools.amd_jupyter.prepare_deployment import git_state, trusted_model_manifest


REPO = "FrenzyMath/REAL-Prover"
REVISION = "fe76f68d9a88f342cb7b546307c20292fea9cced"
CHUNK_BYTES = 8 * 1024 * 1024
CONTEXT_MANIFEST = "build-context-manifest.json"
FIXED_SOURCES = (
    "containers/gpu/Containerfile", "containers/gpu/download_model.py",
    "containers/gpu/smoke_gpu.py", "containers/gpu/requirements-gpu-hashed.lock",
    "cpu_runtime/__init__.py", "cpu_runtime/verified_trajectory.py", "cpu_runtime/verified_dataset_store.py",
    "cpu_runtime/mathlib_trajectory.py",
)
CORE_PINS = {"transformers": "4.57.1", "peft": "0.17.1", "accelerate": "1.10.1",
             "huggingface-hub": "0.34.4", "safetensors": "0.6.2"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_name(name):
    require(isinstance(name, str) and bool(name), "empty context path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name and "\\" not in name and ":" not in name
            and all(part not in {".", ".."} for part in path.parts), "unsafe context path")
    return path


def reject_link_components(path):
    # Inspect lexical ancestors before resolve() can erase a symlink/junction.
    for component in (path, *path.parents):
        require(not component.is_symlink(), "symlink input/output path rejected")
        if hasattr(component, "is_junction"):
            require(not component.is_junction(), "junction input/output path rejected")


def local_regular(root, name):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "input directory missing or symlink")
    current = root
    for part in relative_name(name).parts:
        current = current / part
        require(not current.is_symlink(), "symlink input rejected")
        if hasattr(current, "is_junction"):
            require(not current.is_junction(), "junction input rejected")
    require(current.is_file() and current.resolve().is_relative_to(root.resolve()), "input missing or outside declared directory")
    return current


def validate_hash_lock(content):
    pins = {}
    for line in content.decode("utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([a-z0-9-]+)==([^ ]+) --hash=sha256:([0-9a-f]{64})", line)
        require(match is not None, "dependency lock must contain exact wheel SHA256 hashes")
        name, version, digest = match.groups()
        require(name not in pins and name not in {"torch", "torchvision", "torchaudio", "triton", "pytorch-triton", "pytorch-triton-rocm"}
                and not name.startswith(("torch-", "rocm-", "nvidia-")), "duplicate or forbidden GPU package in dependency lock")
        pins[name] = {"version": version, "sha256": digest}
    require(len(pins) == 22 and all(pins.get(name, {}).get("version") == version for name, version in CORE_PINS.items()),
            "dependency lock must preserve the reviewed 22 packages and five core pins")
    return pins


def stream_copy(source, destination):
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {before.st_size}\0".encode("ascii"))
    size = 0
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        while chunk := incoming.read(CHUNK_BYTES):
            outgoing.write(chunk)
            sha256.update(chunk)
            blob.update(chunk)
            size += len(chunk)
    after = source.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns) and size == before.st_size,
            "input changed while copying; context is incomplete")
    return {"size": size, "sha256": sha256.hexdigest(), "git_blob_sha1": blob.hexdigest()}


def check_official_hash(hashes, expected, name):
    require(hashes["size"] == expected["size"], f"official model size mismatch: {name}")
    if "lfs_sha256" in expected:
        require(hashes["sha256"] == expected["lfs_sha256"], f"official model LFS SHA256 mismatch: {name}")
    else:
        require(hashes["git_blob_sha1"] == expected["git_blob_sha1"], f"official model git blob SHA1 mismatch: {name}")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_new(path, content):
    with path.open("xb") as handle:
        handle.write(content)
    return {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def strict_ignore(names):
    allowed = set()
    for name in names:
        path = relative_name(name)
        for parent in path.parents:
            if str(parent) != ".":
                allowed.add(parent.as_posix() + "/")
        allowed.add(name)
    # Parent directories must be re-included before their children.
    return ("**\n" + "".join("!" + name + "\n" for name in sorted(allowed, key=lambda item: (item.count("/"), item)))).encode()


def prepare_context(repo_root, model_dir, manifest_path, manifest_sha256, output):
    repo_root, model_dir, manifest_path, output = map(Path, (repo_root, model_dir, manifest_path, output))
    for path in (repo_root, model_dir, manifest_path, output):
        reject_link_components(path)
    require(not output.exists() and not output.is_symlink(), "context already exists; refusing to overwrite")
    require(repo_root.is_dir() and model_dir.is_dir() and not repo_root.is_symlink() and not model_dir.is_symlink(),
            "source/model directory missing or symlink")
    require(not output.resolve().is_relative_to(model_dir.resolve()), "context must not be inside its source model directory")
    require(manifest_path.is_file() and not manifest_path.is_symlink() and manifest_path.stat().st_size < 2 * 1024 * 1024,
            "trusted manifest missing, symlink, or too large")
    manifest_bytes, manifest = trusted_model_manifest(manifest_path, manifest_sha256)
    validate_manifest(manifest, REPO, REVISION)
    runtime_names = sorted(path.relative_to(repo_root).as_posix() for path in (repo_root / "gpu_runtime").glob("*.py")
                           if not path.name.startswith("."))
    require("gpu_runtime/__init__.py" in runtime_names, "GPU runtime source files missing")
    names = sorted(runtime_names + list(FIXED_SOURCES))
    sources = {name: local_regular(repo_root, name) for name in names}
    require(all(path.stat().st_size < 2 * 1024 * 1024 for path in sources.values()), "oversize source file")
    dependencies = validate_hash_lock(sources["containers/gpu/requirements-gpu-hashed.lock"].read_bytes())
    models = {name: local_regular(model_dir, name) for name in manifest["files"]}
    for name, source in models.items():
        require(source.stat().st_size == manifest["files"][name]["size"], f"official model size mismatch: {name}")

    # A failure after this point leaves an explicitly incomplete new directory.
    # Never delete user directories or reuse it; a success manifest is written
    # only after independent verification of all copied model files.
    output.mkdir(parents=True, exist_ok=False)
    files, source_hashes, model_source_hashes = {}, {}, {}
    for name, source in sources.items():
        copied = stream_copy(source, output / name)
        files[name] = {"size": copied["size"], "sha256": copied["sha256"], "kind": "source"}
        source_hashes[name] = {"size": copied["size"], "sha256": copied["sha256"]}
    for name, source in sorted(models.items()):
        copied = stream_copy(source, output / "model" / name)
        check_official_hash(copied, manifest["files"][name], name)
        files["model/" + name] = {"size": copied["size"], "sha256": copied["sha256"], "kind": "model"}
        model_source_hashes[name] = {"size": copied["size"], "sha256": copied["sha256"]}
    files["model-manifest.json"] = {**write_new(output / "model-manifest.json", manifest_bytes), "kind": "trusted_manifest"}
    model_lock = validate_and_lock(output / "model", manifest, REPO, REVISION, 3584)
    for name, entry in model_lock["files"].items():
        require(entry["sha256"] == files["model/" + name]["sha256"], "copied model changed after its first checksum")
    generated_lock = (output / "model/reap-model-lock.json").read_bytes()
    files["model/reap-model-lock.json"] = {"size": len(generated_lock), "sha256": hashlib.sha256(generated_lock).hexdigest(), "kind": "verified_model_lock"}
    ignores = strict_ignore(list(files) + [".dockerignore", ".containerignore", CONTEXT_MANIFEST])
    for name in (".dockerignore", ".containerignore"):
        files[name] = {**write_new(output / name, ignores), "kind": "generated_context_whitelist"}
    metadata = {
        "schema_version": "reap.gpu-build-context.v1", "target": "release-existing", "image_built": False, "GPU_verified": False,
        "model": {"repo": REPO, "revision": REVISION, "hidden_size": 3584, "manifest_sha256": manifest_sha256,
                  "file_count": len(model_lock["files"]), "tensor_count": model_lock["safetensors"]["tensor_count"],
                  "bytes": sum(entry["size"] for entry in model_lock["files"].values()), "source_files": model_source_hashes},
        "source": {"git": git_state(repo_root, names), "files": source_hashes},
        "dependencies": {"torch": "unchanged digest-pinned ROCm base; candidate GPU runtime still requires validation", "packages": dependencies},
        "files": dict(sorted(files.items())),
        "build_args": {"REAL_PROVER_MANIFEST_SHA256": manifest_sha256},
    }
    encoded = json_bytes(metadata)
    write_new(output / CONTEXT_MANIFEST, encoded)
    return {"context": str(output), "target": "release-existing", "context_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
            "model_files": metadata["model"]["file_count"], "model_bytes": metadata["model"]["bytes"],
            "image_built": False, "GPU_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model-dir", type=Path, required=True, help="verified model on authorized remote builder; never download it to the local PC")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new context directory; incomplete existing directories are also rejected")
    args = parser.parse_args()
    result = prepare_context(args.repo_root, args.model_dir, args.manifest, args.manifest_sha256, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
