#!/usr/bin/env python3
"""Prepare two unchanged Lean inputs and a four-arm plan; never launch GPU/network."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from cpu_runtime.transport_budget import DEFAULT_BUDGET

ARCHIVE_SHA256 = "9183583475aef37bafd4d0f88e32cc958f5baa335d9b06456ee41dd02f5a6450"
SELECTED = ("inputs/01-CoupledOddSquare.lean", "inputs/05-CubeAccumulator.lean")


def prepare(repo: Path, output: Path) -> dict:
    evidence = repo / "v1-result/evidence/multiround"
    manifest = json.loads((evidence / "raw-evidence-manifest.json").read_bytes())
    archive = evidence / "raw-evidence.tar.gz"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256 or digest != manifest["archive"]["sha256"]:
        raise ValueError("fixed historical evidence archive digest mismatch")
    expected = {item["path"]: item for item in manifest["files"]}
    selected = {}
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("archive member set mismatch")
        for member in members:
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("non-regular evidence member")
            data = tar.extractfile(member).read()
            item = expected[member.name]
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("evidence member checksum mismatch")
            if member.name in SELECTED:
                selected[Path(member.name).name] = data
    if len(selected) != 2:
        raise ValueError("the two fixed search inputs are missing")
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    for name, data in selected.items():
        (output / "inputs" / name).write_bytes(data)
    plan = {"schema_version": "reap.cross-theorem.validation-plan.v1", "execution_started": False,
        "archive_sha256": digest, "verified_archive_members": len(expected),
        "inputs": {name: hashlib.sha256(data).hexdigest() for name, data in selected.items()},
        "source": {"input": "01-CoupledOddSquare.lean", "session_id": "inherit-source-01",
                   "max_updates": 5, "experience_candidate": True,
                   "publish_only_after": "new successful trained run plus independent Lean/wire/parameter acceptance"},
        "target": "05-CubeAccumulator.lean", "gamma": 0.99,
        "arms": [{"name": name, "experience_id": exp, "max_updates": updates,
                  "session_id": "inherit-target-05", "separate_fresh_runtime_required": True}
                 for name, exp, updates in (("fresh-no-ttt", None, 0), ("fresh-ttt", None, 5),
                                           ("inherited-no-ttt", "accepted-source-01", 0),
                                           ("inherited-ttt", "accepted-source-01", 5))],
        "pairing": "Same target/session ID per arm in separate fresh service roots: same RNG seed and input; no live session deletion/reuse to hide unknown outcomes.",
        "pilot_order": ["source", "GPU inheritance mechanism audit", "fresh-ttt", "inherited-ttt"],
        "later_controls": ["fresh-no-ttt", "inherited-no-ttt", "repeat with several predeclared paired session IDs"],
        "deadlines_seconds": {"worker": DEFAULT_BUDGET.worker, "bridge": DEFAULT_BUDGET.bridge(),
                              "client": DEFAULT_BUDGET.client, "barrier": DEFAULT_BUDGET.barrier},
        "metrics": ["independent proof accepted", "search node/checkpoint count", "updates", "search elapsed",
                    "session initialization elapsed", "snapshot/audit elapsed", "transport failures"],
        "limits": ["Two selected historical questions are not a benchmark.",
                   "GPU synthetic mechanism audit is not real Lean TTT.",
                   "No performance claim from one paired run.",
                   "If source solves with zero updates, keep valid proof but do not publish untrained experience.",
                   "No model download, instance start, bridge connection, upload, or image build is performed here."]}
    (output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = prepare(Path(__file__).resolve().parents[1], args.output_dir.resolve())
    print(json.dumps({"prepared": True, "output": str(args.output_dir),
                      "verified_archive_members": plan["verified_archive_members"], "gpu_started": False}))


if __name__ == "__main__":
    main()
