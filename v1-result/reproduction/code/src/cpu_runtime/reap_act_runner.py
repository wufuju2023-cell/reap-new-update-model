"""Bridge one segmented ACT request to a fresh Lean/Reap process."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time

from .batch_solver import SessionSpec, run_session
from .normalize_rollout import normalize_session
from .segmented_ttt import ActRequest, ActResult, RolloutEvent


class ReapActRunner:
    def __init__(
        self,
        *,
        project_dir: Path,
        theorem_file: str,
        output_root: Path,
        gpu_base_url: str,
        lean_bin: str = "lake",
        per_segment_timeout_seconds: float = 90.0,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.theorem_file = theorem_file
        self.output_root = Path(output_root).resolve()
        self.gpu_base_url = gpu_base_url.rstrip("/")
        self.lean_bin = lean_bin
        self.per_segment_timeout_seconds = float(per_segment_timeout_seconds)

    def __call__(self, request: ActRequest) -> ActResult:
        if not request.restart_from_root:
            raise ValueError("Reap ACT must restart from theorem root")
        timeout = self.per_segment_timeout_seconds
        if request.deadline_monotonic is not None:
            remaining = request.deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("TTT deadline expired before ACT")
            timeout = min(remaining, timeout)
        segment_root = self.output_root / request.session_id / f"segment-{request.segment_index:03d}"
        spec = SessionSpec(
            session_id=request.session_id,
            theorem_file=self.theorem_file,
            policy_base_url=f"{self.gpu_base_url}/sessions/{request.session_id}/policy/v1",
            value_base_url=f"{self.gpu_base_url}/sessions/{request.session_id}/value/v1",
        )
        result = asyncio.run(run_session(
            spec,
            self.project_dir,
            segment_root,
            timeout,
            asyncio.Semaphore(1),
            self.lean_bin,
        ))
        session_dir = Path(result.output_dir)
        rollout: list[RolloutEvent] = []
        seen_transitions: set[tuple[str, str, str]] = set()
        normalized = normalize_session(session_dir)
        root_state = ""
        for record in normalized:
            if record.get("event") == "tree_edge" and not root_state:
                root_state = str(record.get("state", ""))
            if record.get("event") != "tactic_eval":
                continue
            transition_key = (
                str(record.get("state", "")),
                str(record.get("tactic", "")),
                str(record.get("verdict", "unknown")),
            )
            if transition_key in seen_transitions:
                continue
            seen_transitions.add(transition_key)
            rollout.append(RolloutEvent(
                state=transition_key[0],
                tactic=transition_key[1],
                verdict=transition_key[2],
                logprob_old=record.get("logprob"),
                next_state=record.get("next_state"),
                metadata={
                    "prompt": record.get("prompt"),
                    "eval_result": record.get("eval_result"),
                    "start_ns": record.get("start_ns"),
                    "stop_ns": record.get("stop_ns"),
                },
            ))
        if result.solved and result.returncode == 0 and not result.timed_out:
            final_record = next(
                (record for record in normalized if record.get("event") == "session_result"),
                {},
            )
            proof_script = str(final_record.get("proof_script") or "")
            if proof_script:
                root_prompt = next(
                    (record["prompt"] for record in normalized
                     if record.get("state") == root_state and record.get("prompt")),
                    None,
                )
                rollout.append(RolloutEvent(
                    state=root_state or request.theorem,
                    tactic=proof_script,
                    verdict="root_verified",
                    logprob_old=None,
                    terminal_verified=True,
                    metadata={"final_check": True, "prompt": root_prompt},
                ))
        return ActResult(
            root_verified=bool(result.solved and result.returncode == 0 and not result.timed_out),
            rollout=tuple(rollout),
            status=result.status,
            metadata={
                "output_dir": result.output_dir,
                "elapsed_seconds": result.elapsed_seconds,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "policy_version": request.policy_version,
            },
        )
