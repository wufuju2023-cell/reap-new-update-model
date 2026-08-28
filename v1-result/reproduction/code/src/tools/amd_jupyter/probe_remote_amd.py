#!/usr/bin/env python3
"""Bounded host preflight; optional tiny ROCm update is NOT a REAP/TTT test.

Run on the remote host with its existing Python, without installing dependencies:
    python3 probe_remote_amd.py
    python3 probe_remote_amd.py --gpu-smoke

Default checks do not allocate GPU tensors. Runtime `info` probes neither start
containers nor download images. No browser state, environment variables, raw
subprocess output, or exception messages are included in the JSON report.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


TORCH_PROBE = r'''
import json, sys
report = {"status": "unavailable", "gpu_smoke": {"requested": sys.argv[1] == "yes", "status": "not_requested"}}
try:
    import torch
    report.update(status="ok", version=str(torch.__version__), hip=torch.version.hip,
                  cuda_build=torch.version.cuda, gpu_available=torch.cuda.is_available())
    report["gpu_count"] = torch.cuda.device_count()
    report["devices"] = []
    for i in range(report["gpu_count"]):
        props = torch.cuda.get_device_properties(i)
        device = {"index": i, "name": props.name, "total_memory_bytes": props.total_memory}
        try:
            free, total = torch.cuda.mem_get_info(i)
            device.update(free_memory_bytes=free, usable_total_memory_bytes=total)
        except Exception as exc:
            device["memory_query_error_type"] = type(exc).__name__
        report["devices"].append(device)
    if sys.argv[1] == "yes":
        if not report["gpu_available"] or not report["hip"]:
            report["gpu_smoke"] = {"requested": True, "status": "skipped", "reason": "ROCm_GPU_required"}
        else:
            x = torch.nn.Parameter(torch.ones((128, 128), device="cuda", dtype=torch.float32))
            before = x.detach().clone()
            optimizer = torch.optim.SGD([x], lr=0.01)
            loss = x.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_finite = bool(torch.isfinite(x.grad).all().item())
            optimizer.step()
            torch.cuda.synchronize()
            changed = not torch.equal(before, x.detach())
            finite = bool(torch.isfinite(x).all().item())
            report["gpu_smoke"] = {"requested": True, "status": "passed" if changed and finite and gradient_finite else "failed",
                "device_index": 0, "parameter_changed": changed, "parameter_finite": finite,
                "gradient_finite": gradient_finite, "loss": float(loss.item()), "scope": "tiny_ROCm_SGD_only_NOT_TTT"}
except Exception as exc:
    report["status"] = "error"
    report["error_type"] = type(exc).__name__
    if sys.argv[1] == "yes":
        report["gpu_smoke"] = {"requested": True, "status": "error", "error_type": type(exc).__name__}
print(json.dumps(report, allow_nan=False))
'''


def bounded_json(command: list[str], timeout: float) -> tuple[dict, object | None]:
    """Retain only parsed JSON; never expose stderr, command text, or raw output."""
    started = time.monotonic()
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
        )
        metadata = {"status": "ok", "exit_code": result.returncode}
        if result.returncode:
            metadata["status"] = "nonzero_exit"
            data = None
        else:
            try:
                data = json.loads(result.stdout)
            except ValueError:
                metadata["status"] = "invalid_json"
                data = None
    except subprocess.TimeoutExpired:
        metadata, data = {"status": "timeout", "timeout_seconds": timeout}, None
    except OSError as exc:
        metadata, data = {"status": "error", "error_type": type(exc).__name__}, None
    metadata["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return metadata, data


def _nested(data: dict, *keys: str) -> object:
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def runtime_probe(name: str, timeout: float) -> dict:
    executable = shutil.which(name)
    if executable is None:
        return {"present": False, "info": {"status": "missing"}}
    if name == "docker":
        command = [executable, "info", "--format", "{{json .}}"]
        fields = {"server_version": ("ServerVersion",), "storage_driver": ("Driver",),
                  "os_type": ("OSType",), "architecture": ("Architecture",)}
    elif name == "podman":
        command = [executable, "info", "--format", "json"]
        fields = {"version": ("version", "Version"), "storage_driver": ("store", "graphDriverName"),
                  "os_type": ("host", "os"), "architecture": ("host", "arch"),
                  "rootless": ("host", "security", "rootless")}
    else:
        command = [executable, "info"]
        fields = {"storage_driver": ("store", "GraphDriverName"),
                  "architecture": ("host", "arch"), "os_type": ("host", "os")}
    metadata, data = bounded_json(command, timeout)
    if isinstance(data, dict):
        # Explicit allowlist excludes registry credentials, endpoints, paths,
        # proxy settings, hostnames, labels, and the surrounding full info JSON.
        metadata["details"] = {
            key: value for key, path in fields.items()
            if isinstance(value := _nested(data, *path), (str, bool, int))
        }
    elif data is not None:
        metadata["status"] = "unexpected_json_type"
    return {"present": True, "info": metadata}


def memory_probe() -> dict:
    result = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            name, _, value = line.partition(":")
            if name in {"MemTotal", "MemAvailable"}:
                result[name + "_bytes"] = int(value.split()[0]) * 1024
    except (OSError, ValueError) as exc:
        result["status"] = "unavailable"
        result["error_type"] = type(exc).__name__
    return result


def device_probe(path: str) -> dict:
    return {"present": os.path.exists(path), "readable": os.access(path, os.R_OK),
            "writable": os.access(path, os.W_OK)}


def timeout_value(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 60:
        raise argparse.ArgumentTypeError("timeout must be finite and in (0, 60] seconds")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/mnt/workspace"),
                        help="directory to check disk capacity; path is not included in output")
    parser.add_argument("--gpu-smoke", action="store_true", help="explicitly run one tiny ROCm SGD step; not TTT")
    parser.add_argument("--runtime-timeout-seconds", type=timeout_value, default=4.0)
    parser.add_argument("--torch-timeout-seconds", type=timeout_value, default=30.0)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        usage = shutil.disk_usage(args.workspace)
        disk = {"status": "ok", "total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}
    except OSError as exc:
        disk = {"status": "unavailable", "error_type": type(exc).__name__}
    report = {
        "schema_version": 1,
        "scope": "host_preflight_only",
        "ttt_verified": False,
        "container_gpu_passthrough_verified": False,
        "system": {"os": platform.system(), "architecture": platform.machine(),
                   "python": platform.python_version(), "uid": os.getuid() if hasattr(os, "getuid") else None},
        "memory": memory_probe(),
        "workspace_disk": disk,
        "devices": {"kfd": device_probe("/dev/kfd"),
                    "dri_render_nodes": [{"name": Path(path).name, **device_probe(path)}
                                         for path in sorted(glob.glob("/dev/dri/render*"))]},
        "tools": {name: {"present": shutil.which(name) is not None} for name in ("rocm-smi", "amd-smi")},
        "container_runtimes": {name: runtime_probe(name, args.runtime_timeout_seconds)
                               for name in ("docker", "podman", "buildah")},
    }
    metadata, torch_report = bounded_json(
        [sys.executable, "-B", "-c", TORCH_PROBE, "yes" if args.gpu_smoke else "no"],
        args.torch_timeout_seconds,
    )
    report["torch_probe"] = metadata
    if isinstance(torch_report, dict):
        report["torch"] = torch_report
    elif torch_report is not None:
        report["torch_probe"]["status"] = "unexpected_json_type"
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    # Diagnostic collection finishing is not a GPU/TTT success exit status.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
