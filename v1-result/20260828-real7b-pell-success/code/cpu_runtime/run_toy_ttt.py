#!/usr/bin/env python3
"""Run Lean/Reap ACT→LEARN→fresh-root ACT against a GPU runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from .http_clients import GpuHttpClient
from .reap_act_runner import ReapActRunner
from .segmented_ttt import SegmentedTTTController


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--gpu-base-url", default="http://127.0.0.1:18760")
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--theorem-file", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lean-bin", default="lake")
    parser.add_argument("--deadline-seconds", type=float, default=None,
                        help="optional total runtime limit; omitted means no total deadline")
    parser.add_argument("--segment-timeout-seconds", type=float, default=90.0,
                        help="per-ACT process timeout, independent of total TTT runtime")
    parser.add_argument("--http-timeout-seconds", type=float, default=30.0,
                        help="per-request GPU transport timeout; allow longer for an asynchronous remote bridge")
    parser.add_argument(
        "--learn-verified-proof",
        action="store_true",
        help="train once on a root-checked proof and require a fresh-root recheck",
    )
    args = parser.parse_args()
    for name in ("deadline_seconds", "segment_timeout_seconds", "http_timeout_seconds"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")

    client = GpuHttpClient(args.gpu_base_url, timeout_seconds=args.http_timeout_seconds)
    client.create_session(args.session_id)
    runner = ReapActRunner(
        project_dir=args.project_dir,
        theorem_file=args.theorem_file,
        output_root=args.output_dir,
        gpu_base_url=args.gpu_base_url,
        lean_bin=args.lean_bin,
        per_segment_timeout_seconds=args.segment_timeout_seconds,
    )
    controller = SegmentedTTTController(
        runner,
        client.learn,
        args.output_dir / args.session_id / "ttt-journal.jsonl",
    )
    outcome = controller.run(
        session_id=args.session_id,
        theorem=args.theorem_file,
        max_segments=2,
        deadline_seconds=args.deadline_seconds,
        learn_verified_proof=args.learn_verified_proof,
    )
    print(json.dumps(asdict(outcome), ensure_ascii=False))
    return 0 if outcome.root_verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
