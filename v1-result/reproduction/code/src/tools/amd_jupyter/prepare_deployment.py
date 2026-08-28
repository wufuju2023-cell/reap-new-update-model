#!/usr/bin/env python3
"""Create a small explicit-whitelist deployment archive; never deploy or install.

The input model manifest must have been fetched from the official HF API and its
SHA256 supplied separately. No model files, browser state, credentials, Git
metadata, histories, or arbitrary .downloads contents enter the archive.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile


MODEL_REPO = "FrenzyMath/REAL-Prover"
MODEL_REVISION = "fe76f68d9a88f342cb7b546307c20292fea9cced"
MODEL_MANIFEST = "model-manifest.json"
ARCHIVE_NAME = "deployment.tar.gz"
DEPLOYMENT_MANIFEST = "deployment-manifest.json"
FIXED_FILES = (
    "containers/gpu/download_model.py", "containers/gpu/smoke_gpu.py",
    "containers/gpu/smoke_search_gpu.py", "containers/gpu/smoke_experience_gpu.py",
    "containers/gpu/requirements-gpu-hashed.lock",
    "containers/gpu/requirements-gpu.lock", "containers/gpu/Containerfile",
    "tools/amd_jupyter/probe_remote_amd.py",
    "cpu_runtime/__init__.py", "cpu_runtime/verified_trajectory.py", "cpu_runtime/verified_dataset_store.py",
)
REMOTE_HTTP_JOB = "tools/amd_jupyter/remote_http_job.py"
REMOTE_REQUIREMENTS = "containers/gpu/requirements-remote.lock"
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
LOCKED_PACKAGES = {"transformers", "peft", "accelerate", "huggingface-hub", "safetensors"}

SETUP_VENV = r'''#!/bin/sh
# Run manually after verifying/extracting this bundle into a fresh directory.
# Optional argument: wheelhouse directory. No services/models are started.
set -eu
wheelhouse=${1:-}
if [ -n "$wheelhouse" ]; then
  if [ ! -d "$wheelhouse" ]; then
    echo "Wheelhouse directory is missing" >&2
    exit 1
  fi
  wheelhouse=$(CDPATH= cd -- "$wheelhouse" && pwd)
fi
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
requirements=__REQUIREMENTS_LOCK__
venv_root="$(pwd)/venv"
if [ -e ./venv ]; then
  echo "Refusing to overwrite existing ./venv" >&2
  exit 1
fi
host_torch_identity=$(python3 -c 'import json,os,torch; print(json.dumps([str(torch.__version__), os.path.realpath(torch.__file__), torch.version.hip]))')
python3 -m venv --system-site-packages ./venv
venv_torch_identity=$(./venv/bin/python -c 'import json,os,torch; print(json.dumps([str(torch.__version__), os.path.realpath(torch.__file__), torch.version.hip]))')
if [ "$host_torch_identity" != "$venv_torch_identity" ]; then
  echo "The venv does not expose the unchanged host torch; stopping" >&2
  exit 1
fi
if [ -n "$wheelhouse" ]; then
  ./venv/bin/python -m pip --isolated --require-virtualenv --disable-pip-version-check --no-cache-dir install \
    --no-index --find-links "$wheelhouse" --no-deps --ignore-installed \
    --prefix "$venv_root" -r "$requirements"
else
  ./venv/bin/python -m pip --isolated --require-virtualenv --disable-pip-version-check --no-cache-dir install \
    --no-deps --ignore-installed --prefix "$venv_root" -r "$requirements"
fi
venv_torch_identity=$(./venv/bin/python -c 'import json,os,torch; print(json.dumps([str(torch.__version__), os.path.realpath(torch.__file__), torch.version.hip]))')
if [ "$host_torch_identity" != "$venv_torch_identity" ]; then
  echo "Torch identity changed unexpectedly; stopping" >&2
  exit 1
fi
./venv/bin/python - "$requirements" <<'PY'
from importlib import metadata
from pathlib import Path
import sys
from packaging.requirements import Requirement
from packaging.version import Version

root = Path(sys.prefix).resolve()
failures = []
for line in Path(sys.argv[1]).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name, expected = line.split("==", 1)
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        failures.append(f"Missing locked distribution: {name}")
        continue
    if distribution.version != expected:
        failures.append(f"Locked version mismatch: {name}")
    if not Path(distribution.locate_file("")).resolve().is_relative_to(root):
        failures.append(f"Locked distribution is not installed locally in the venv: {name}")
    for raw in distribution.requires or []:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            failures.append(f"Missing dependency of {name}: {requirement.name}")
            continue
        if requirement.specifier and Version(installed) not in requirement.specifier:
            failures.append(f"Dependency version mismatch for {name}: {requirement.name}")
if failures:
    raise SystemExit("\n".join(failures) + "\nStop: prepare a reviewed dependency lock/wheelhouse; never auto-install torch or unpinned dependencies.")
print("Locked venv distributions and their active dependency requirements are satisfied.")
PY
./venv/bin/python -c 'import transformers,peft,accelerate,huggingface_hub,safetensors; print("Locked GPU libraries import successfully; no model or service started.")'
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def safe_relative(name):
    path = PurePosixPath(name)
    require(isinstance(name, str) and bool(name) and not path.is_absolute()
            and path.as_posix() == name and all(part not in {".", ".."} for part in path.parts)
            and "\\" not in name and ":" not in name and "\x00" not in name,
            "unsafe archive member path")
    return path


def source_bytes(repo_root, relative):
    path = repo_root.joinpath(*safe_relative(relative).parts)
    require(path.is_file(), f"missing required deployment file: {relative}")
    require(not path.is_symlink() and path.resolve().is_relative_to(repo_root.resolve()),
            f"nonlocal or symlink deployment source: {relative}")
    require(path.stat().st_size <= MAX_FILE_BYTES, f"oversize deployment source: {relative}")
    content = path.read_bytes()
    require(len(content) <= MAX_FILE_BYTES, f"oversize deployment source: {relative}")
    return content


def validate_requirements(content, *, extra_packages=False):
    packages = {}
    for line in content.decode("utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9.]*)?)", line)
        require(match is not None, "requirements must contain only exact reviewed package pins")
        name = re.sub(r"[-_.]+", "-", match.group(1).lower())
        require(name not in {"torch", "torchvision", "torchaudio", "triton", "pytorch-triton", "pytorch-triton-rocm"}
                and not name.startswith(("nvidia-", "rocm-", "torch-")), "requirements must not install host GPU/torch packages")
        require((extra_packages or name in LOCKED_PACKAGES) and name not in packages,
                "requirements contain duplicate or unreviewed packages")
        packages[name] = match.group(2)
    require(LOCKED_PACKAGES <= packages.keys(), "requirements do not contain all five GPU library pins")
    return packages


def trusted_model_manifest(path, expected_sha256):
    require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256),
            "model-manifest-sha256 must be a full lowercase SHA256")
    require(path.is_file() and path.stat().st_size <= MAX_FILE_BYTES, "model manifest missing or too large")
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == expected_sha256, "trusted model manifest SHA256 mismatch")
    metadata = json.loads(content.decode("utf-8-sig"))
    require(isinstance(metadata, dict) and set(metadata) == {"schema_version", "repo", "revision", "source_api", "files"},
            "model manifest contains unexpected metadata fields")
    expected_source = f"https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REVISION}?blobs=true"
    require(metadata.get("schema_version") == "reap.official-model-manifest.v1"
            and metadata.get("repo") == MODEL_REPO and metadata.get("revision") == MODEL_REVISION
            and metadata.get("source_api") == expected_source and isinstance(metadata.get("files"), dict) and bool(metadata["files"]),
            "model manifest is not the fixed official REAL-Prover revision")
    for name, entry in metadata["files"].items():
        safe_relative(name)
        require(isinstance(entry, dict) and {"size", "git_blob_sha1"} <= set(entry)
                and set(entry) <= {"size", "git_blob_sha1", "lfs_sha256"}, "unexpected model file metadata fields")
        require(type(entry["size"]) is int and entry["size"] > 0
                and re.fullmatch(r"[0-9a-f]{40}", str(entry["git_blob_sha1"])), "invalid model file metadata")
        if "lfs_sha256" in entry:
            require(re.fullmatch(r"[0-9a-f]{64}", str(entry["lfs_sha256"])), "invalid model file SHA256")
    return content, metadata


def git_state(repo_root, source_names):
    result = {"head": None, "dirty": None, "included_source_changes": [],
              "description": "Archive contains current workspace bytes. File hashes, not HEAD alone, identify the deployed code."}
    try:
        head = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              text=True, timeout=5, check=False)
        status = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain=v1", "--no-renames",
                                 "--untracked-files=all", "-z", "--", *source_names],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, timeout=5, check=False)
        if head.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", head.stdout.strip()):
            result["head"] = head.stdout.strip()
        if status.returncode == 0:
            changes = [{"status": item[:2], "path": item[3:]} for item in status.stdout.split("\0") if item]
            require(all(item["path"] in source_names for item in changes), "git reported an unexpected source path")
            result["dirty"] = bool(changes)
            result["included_source_changes"] = changes
    except (OSError, subprocess.TimeoutExpired):
        # A non-Git source tree is allowed, but never reported as clean.
        pass
    return result


def build_deployment(repo_root, output, model_manifest, model_manifest_sha256, *, include_remote_http_job=False,
                     include_remote_requirements=False):
    repo_root, output, model_manifest = Path(repo_root), Path(output), Path(model_manifest)
    require(repo_root.is_dir(), "repository root is missing")
    require(not output.exists(), "output directory already exists; refusing to overwrite")
    runtime_names = sorted(path.relative_to(repo_root).as_posix() for path in (repo_root / "gpu_runtime").glob("*.py")
                           if not path.name.startswith("."))
    require(bool(runtime_names) and "gpu_runtime/__init__.py" in runtime_names, "GPU runtime Python sources are missing")
    source_names = sorted(runtime_names + list(FIXED_FILES) + ([REMOTE_HTTP_JOB] if include_remote_http_job else [])
                          + ([REMOTE_REQUIREMENTS] if include_remote_requirements else []))
    payload = {name: source_bytes(repo_root, name) for name in source_names}
    top_level = validate_requirements(payload["containers/gpu/requirements-gpu.lock"])
    selected_requirements = "containers/gpu/requirements-gpu.lock"
    if include_remote_requirements:
        remote_pins = validate_requirements(payload[REMOTE_REQUIREMENTS], extra_packages=True)
        require(all(remote_pins[name] == version for name, version in top_level.items()),
                "remote dependency lock changes a pinned top-level GPU library")
        selected_requirements = REMOTE_REQUIREMENTS
    manifest_bytes, model = trusted_model_manifest(model_manifest, model_manifest_sha256)
    payload[MODEL_MANIFEST] = manifest_bytes
    payload["setup_venv.sh"] = SETUP_VENV.replace("__REQUIREMENTS_LOCK__", selected_requirements).encode("utf-8")
    require(sum(map(len, payload.values())) <= MAX_TOTAL_BYTES, "deployment payload exceeds small-bundle limit")
    manifest = {
        "schema_version": "reap.remote-deployment.v1",
        "scope": "host_GPU_preparation_only_not_a_container_image_or_TTT_result",
        "model": {"repo": model["repo"], "revision": model["revision"], "hidden_size": 3584,
                  "manifest_path": MODEL_MANIFEST, "manifest_sha256": model_manifest_sha256,
                  "weights_included": False},
        "source": git_state(repo_root, source_names),
        "remote_http_job_included": include_remote_http_job,
        "remote_requirements_included": include_remote_requirements,
        "setup": {"executed": False, "host_torch_replacement_allowed": False,
                  "requirements": selected_requirements,
                  "dependencies": "exact selected lock only; active dependency closure checked; host torch remains external"},
        "files": {name: {"size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                         "mode": "0755" if name == "setup_venv.sh" else "0644"}
                  for name, content in sorted(payload.items())},
    }
    metadata = canonical_json(manifest)
    payload[DEPLOYMENT_MANIFEST] = metadata
    # Validate and collect everything before creating the destination. Exclusive
    # creation also protects against another process creating the same output.
    output.mkdir(parents=True, exist_ok=False)
    with (output / DEPLOYMENT_MANIFEST).open("xb") as handle:
        handle.write(metadata)
    archive = output / ARCHIVE_NAME
    with archive.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name, content in sorted(payload.items()):
                safe_relative(name)
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.uid, info.gid = len(content), 0, 0, 0
                info.mode = 0o755 if name == "setup_venv.sh" else 0o644
                tar.addfile(info, io.BytesIO(content))
    checksums = "".join(f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n"
                        for name in (ARCHIVE_NAME, DEPLOYMENT_MANIFEST))
    with (output / "manifest.sha256").open("x", encoding="ascii", newline="\n") as handle:
        handle.write(checksums)
    return {"output": str(output), "archive_bytes": archive.stat().st_size, "payload_files": len(manifest["files"]),
            "model_revision": MODEL_REVISION, "dirty": manifest["source"]["dirty"], "ttt_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True, help="new directory; any existing path is rejected")
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--include-remote-http-job", action="store_true", help="include the one explicitly allowlisted bridge worker; missing file is an error")
    parser.add_argument("--include-remote-requirements", action="store_true", help="require/include the reviewed auxiliary dependency lock and use it for the venv")
    args = parser.parse_args()
    result = build_deployment(args.repo_root, args.output, args.model_manifest, args.model_manifest_sha256,
                              include_remote_http_job=args.include_remote_http_job,
                              include_remote_requirements=args.include_remote_requirements)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
