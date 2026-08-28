#!/usr/bin/env python3
"""Prepare a weight-free, offline GPU image context; never downloads or builds.

Only runtime Python sources, four pure dataset loaders and the exact reviewed
22 wheels are admitted. The model must be mounted read-only at runtime. Partial
outputs are never reused, and neither the old recipe nor old evidence is changed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.amd_jupyter.prepare_gpu_build_context import (
    json_bytes, local_regular, reject_link_components, require, stream_copy,
    strict_ignore, validate_hash_lock, write_new,
)

MANIFEST = "runtime-context-manifest.json"
BASE = "docker.io/rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e"
FIXED = (
    "containers/gpu/Containerfile.runtime", "containers/gpu/requirements-gpu-hashed.lock",
    "containers/gpu/smoke_gpu.py", "cpu_runtime/__init__.py",
    "cpu_runtime/verified_trajectory.py", "cpu_runtime/verified_dataset_store.py",
    "cpu_runtime/mathlib_trajectory.py",
    "containers/gpu/install_runtime.sh",
)


def check_imports(sources):
    """Reject undeclared static project imports, including delayed imports."""
    modules = {name[:-3].replace("/", ".") for name in sources if name.endswith(".py")}
    modules |= {name[:-9] for name in modules if name.endswith(".__init__")}
    for name, source in sources.items():
        if not name.endswith(".py"):
            continue
        tree = ast.parse(source.read_bytes(), filename=name)
        package = name.rsplit("/", 1)[0].replace("/", ".")
        for node in ast.walk(tree):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    require(node.level == 1, "unsupported cross-package relative import")
                    module = package + ("." + module if module else "")
                imports = [module]
            for module in imports:
                if module.split(".", 1)[0] in {"gpu_runtime", "cpu_runtime", "containers", "tools"}:
                    require(module in modules, f"undeclared project import: {name}: {module}")


def prepare_context(repo_root, wheelhouse, output):
    repo_root, wheelhouse, output = map(Path, (repo_root, wheelhouse, output))
    for path in (repo_root, wheelhouse, output):
        reject_link_components(path)
    require(repo_root.is_dir() and wheelhouse.is_dir(), "source/wheelhouse missing")
    require(not output.exists(), "context exists; refusing overwrite or incomplete reuse")
    require(not output.resolve().is_relative_to(wheelhouse.resolve()), "context inside wheelhouse")
    runtime = sorted(p.relative_to(repo_root).as_posix() for p in (repo_root / "gpu_runtime").glob("*.py")
                     if not p.name.startswith("."))
    require("gpu_runtime/__init__.py" in runtime and "gpu_runtime/server.py" in runtime, "runtime missing")
    sources = {name: local_regular(repo_root, name) for name in sorted(runtime + list(FIXED))}
    require(all(p.stat().st_size < 2 * 1024 * 1024 for p in sources.values()), "oversize source")
    check_imports(sources)
    recipe = sources[FIXED[0]].read_text(encoding="utf-8")
    installer = sources["containers/gpu/install_runtime.sh"].read_text(encoding="utf-8")
    require(BASE in recipe and "--no-index" in installer and "--require-hashes" in installer,
            "runtime recipe lost fixed/offline dependency contract")
    require("download_model.py" not in recipe + installer and "COPY model" not in recipe and "FROM model-download" not in recipe,
            "runtime recipe must not include model download/copy")
    dependencies = validate_hash_lock(sources[FIXED[1]].read_bytes())
    required = {entry["sha256"] for entry in dependencies.values()}
    require(len(required) == 22, "duplicate wheel hash")
    wheels = {}
    for path in sorted(wheelhouse.glob("*.whl")):
        source = local_regular(wheelhouse, path.name)
        require(source.stat().st_size < 128 * 1024 * 1024, "oversize wheel")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        require(digest in required and digest not in wheels, "unexpected or duplicate wheel")
        wheels[digest] = source
    require(set(wheels) == required, "missing reviewed wheel")

    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, source in sources.items():
        record = stream_copy(source, output / name)
        files[name] = {"size": record["size"], "sha256": record["sha256"], "kind": "source"}
    for digest, source in sorted(wheels.items()):
        name = "wheels/" + source.name
        record = stream_copy(source, output / name)
        require(record["sha256"] == digest, "wheel changed during copy")
        files[name] = {"size": record["size"], "sha256": digest, "kind": "wheel"}
    ignore = strict_ignore([*files, MANIFEST, ".dockerignore", ".containerignore"])
    for name in (".dockerignore", ".containerignore"):
        files[name] = {**write_new(output / name, ignore), "kind": "whitelist"}
    # Recheck copied source bytes; this output is a frozen build input, not HEAD.
    for name, entry in files.items():
        raw = (output / name).read_bytes()
        require(len(raw) == entry["size"] and hashlib.sha256(raw).hexdigest() == entry["sha256"], "context changed")
    metadata = {"schema": "reap.gpu-runtime-context.v1", "base_image": BASE,
                "recipe": FIXED[0], "model_files": 0, "model_mount": "/opt/models/REAL-Prover:ro",
                "build_network": "none", "image_built": False, "gpu_verified": False,
                "dependencies": dependencies, "files": files}
    raw = json_bytes(metadata)
    write_new(output / MANIFEST, raw)
    return {"context": str(output), "files": len(files), "bytes": sum(v["size"] for v in files.values()),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(), "model_files": 0,
            "image_built": False, "gpu_verified": False}


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--wheelhouse", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_context(args.repo_root, args.wheelhouse, args.output), allow_nan=False))


if __name__ == "__main__":
    main()
