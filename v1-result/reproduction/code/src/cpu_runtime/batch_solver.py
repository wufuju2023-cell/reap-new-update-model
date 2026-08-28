#!/usr/bin/env python3
"""Run isolated Lean/Reap sessions concurrently.

Each JSONL manifest record describes one theorem process. Isolation is explicit:
the session gets its own output directory and policy/value endpoint, allowing a
GPU service to bind an independent LoRA/value/optimizer state to the URL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import time
from typing import Iterable


SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    theorem_file: str
    policy_base_url: str
    value_base_url: str
    ps_endpoint: str = ""


@dataclass
class SessionResult:
    session_id: str
    theorem_file: str
    returncode: int
    timed_out: bool
    elapsed_seconds: float
    solved: bool
    status: str
    output_dir: str


def _require_string(obj: dict, name: str) -> str:
    value = obj.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def parse_spec(obj: dict) -> SessionSpec:
    session_id = _require_string(obj, "session_id")
    if not SESSION_RE.fullmatch(session_id):
        raise ValueError(f"invalid session_id: {session_id!r}")
    return SessionSpec(
        session_id=session_id,
        theorem_file=_require_string(obj, "theorem_file"),
        policy_base_url=_require_string(obj, "policy_base_url").rstrip("/"),
        value_base_url=_require_string(obj, "value_base_url").rstrip("/"),
        ps_endpoint=str(obj.get("ps_endpoint", "")),
    )


def load_manifest(path: Path) -> list[SessionSpec]:
    specs: list[SessionSpec] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            obj = json.loads(line)
            spec = parse_spec(obj)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if spec.session_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate session_id {spec.session_id!r}")
        seen.add(spec.session_id)
        specs.append(spec)
    if not specs:
        raise ValueError(f"manifest contains no sessions: {path}")
    return specs


async def _stop_process_tree(process: asyncio.subprocess.Process) -> None:
    """Stop the exact solver process tree without touching unrelated sessions."""
    if process.returncode is not None:
        return
    if os.name == "nt":
        killer = await asyncio.create_subprocess_exec(
            "taskkill", "/PID", str(process.pid), "/T", "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
    else:
        os.killpg(process.pid, signal.SIGTERM)


async def run_process(command: list[str], cwd: Path, env: dict[str, str], timeout: float) -> tuple[int, bool, bytes, bytes]:
    process_options: dict[str, object] = {}
    if os.name == "nt":
        process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        process_options["start_new_session"] = True
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **process_options,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return int(process.returncode or 0), False, stdout, stderr
    except asyncio.TimeoutError:
        await _stop_process_tree(process)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
        return 124, True, stdout, stderr


def _read_solver_result(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing_result"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "invalid_result"
    return bool(data.get("solved", False)), str(data.get("status", "unknown"))


def _finalize_solver_status(path: Path, returncode: int, timed_out: bool) -> tuple[bool, str]:
    solved, status = _read_solver_result(path)
    if timed_out:
        return False, "timeout"
    if returncode != 0:
        return False, "lean_failed_after_solution" if solved else "lean_failed"
    return solved, status


async def run_session(
    spec: SessionSpec,
    project_dir: Path,
    output_root: Path,
    timeout_seconds: float,
    semaphore: asyncio.Semaphore,
    lean_bin: str,
) -> SessionResult:
    async with semaphore:
        session_dir = output_root / spec.session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        theorem_path = Path(spec.theorem_file)
        if not theorem_path.is_absolute():
            theorem_path = project_dir / theorem_path
        env = os.environ.copy()
        env.update(
            {
                "REAP_SESSION_ID": spec.session_id,
                "REAP_SESSION_DIR": str(session_dir),
                "REAP_POLICY_ENDPOINT": spec.policy_base_url,
                "REAP_VALUE_ENDPOINT": spec.value_base_url,
                "REAP_PS_ENDPOINT": spec.ps_endpoint,
            }
        )
        (session_dir / "session.json").write_text(
            json.dumps(asdict(spec), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        started = time.perf_counter()
        returncode, timed_out, stdout, stderr = await run_process(
            [lean_bin, "env", "lean", str(theorem_path)],
            project_dir,
            env,
            timeout_seconds,
        )
        elapsed = time.perf_counter() - started
        (session_dir / "stdout.log").write_bytes(stdout)
        (session_dir / "stderr.log").write_bytes(stderr)
        solved, status = _finalize_solver_status(
            session_dir / "result.json", returncode, timed_out
        )
        return SessionResult(
            session_id=spec.session_id,
            theorem_file=str(theorem_path),
            returncode=returncode,
            timed_out=timed_out,
            elapsed_seconds=round(elapsed, 6),
            solved=solved,
            status=status,
            output_dir=str(session_dir),
        )


async def run_all(args: argparse.Namespace) -> list[SessionResult]:
    project_dir = args.project_dir.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    specs = load_manifest(args.manifest.resolve())
    collisions = [spec.session_id for spec in specs if (output_root / spec.session_id).exists()]
    if collisions:
        raise FileExistsError(f"session output already exists: {', '.join(collisions)}")
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        run_session(spec, project_dir, output_root, args.timeout_seconds, semaphore, args.lean_bin)
        for spec in specs
    ]
    return await asyncio.gather(*tasks)


def write_summary(path: Path, results: Iterable[SessionResult]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=Path(os.getenv("REAP_PROJECT_DIR", "/opt/reap-runtime")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.getenv("REAP_OUTPUT_DIR", "/workspace/out")))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--lean-bin", default="lake")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be > 0")
    started = time.perf_counter()
    results = asyncio.run(run_all(args))
    write_summary(args.output_dir.resolve() / "summary.jsonl", results)
    report = {
        "sessions": len(results),
        "solved": sum(item.solved for item in results),
        "wall_seconds": round(time.perf_counter() - started, 6),
        "max_session_seconds": max(item.elapsed_seconds for item in results),
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0 if all(item.solved for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
