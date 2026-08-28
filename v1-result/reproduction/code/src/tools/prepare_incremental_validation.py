"""Prepare only new inheritance/concurrency gates, reusing audited prior work.

This local preflight does not execute experiments, contact Edge, start instances,
or certify proof quality. It checks the recorded admission and content hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from tools.prepare_experience_validation import ARCHIVE_SHA256


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(root: Path, relative: str, hashes: dict) -> dict:
    name = PurePosixPath(relative)
    if name.is_absolute() or name.as_posix() != relative or any(p in (".", "..") for p in name.parts):
        raise ValueError("evidence path must stay inside the prior run")
    path = root.joinpath(*name.parts)
    if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()) for p in (path, *path.parents)):
        raise ValueError("linked evidence rejected")
    data = path.read_bytes()
    hashes[relative] = sha(data)
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("evidence must be an object")
    return value


def prepare(repo: Path, prior: Path, output: Path) -> dict:
    hashes = {}
    release = read(prior, "publicationation/receipt.json", hashes)
    admission = read(prior, "source-acceptance-evidence.json", hashes)
    if (release.get("schema_version") != "reap.gpu.experience.v1"
            or release.get("transfer") != ["adapter", "value_head"]
            or admission.get("passed") is not True or release.get("source") != admission.get("source")
            or release.get("acceptance", {}).get("source") != admission.get("source")
            or release["acceptance"].get("kind") != "independent-lean"
            or release["acceptance"].get("passed") is not True
            or release["acceptance"].get("completed") is not True
            or release["acceptance"].get("evidence_sha256") != hashes["source-acceptance-evidence.json"]):
        raise ValueError("source admission does not bind the published release")
    for name, expected in admission["files"].items():
        read(prior, name, hashes)
        if hashes[name] != expected:
            raise ValueError("previously admitted source evidence changed")
    mechanism = read(prior, "mechanism-report.json", hashes)
    required = {"initial_parameters_equal_release", "fresh_training_state", "first_update_v1",
                "control_unchanged", "release_unchanged", "restore_exact"}
    if (mechanism.get("ok") is not True or set(mechanism.get("gates", {})) != required
            or not all(v is True for v in mechanism["gates"].values())
            or mechanism.get("base_before", {}).get("sha256") != mechanism.get("base_after", {}).get("sha256")
            or not mechanism.get("base_before", {}).get("sha256")):
        raise ValueError("prior mechanism gate is incomplete")
    fresh = read(prior, "fresh-ttt-final/online-result.json", hashes)
    session = read(prior, "fresh-ttt-final/session.json", hashes)
    proof = read(prior, "proof-checkfresh-ttt/receipt.json", hashes)
    wire = read(prior, "fresh-target-wire-audit.json", hashes)
    if (fresh.get("status") != "passed_execution" or fresh.get("root_verified") is not True
            or fresh.get("experience_id") is not None or fresh.get("error") is not None
            or fresh.get("optimizer_updates", 0) < 1
            or proof.get("returncode") != 0 or proof.get("container_exit_code") != 0
            or proof.get("network") != "none" or proof.get("proof_from_session") != fresh.get("session_id")
            or wire.get("status") != "MATCHED_EXECUTION_WIRE" or wire.get("gaps") != []
            or wire.get("unknown_mutation_request_ids") != []
            or wire.get("session_id") != fresh.get("session_id")
            or session.get("session_id") != fresh.get("session_id")
            or proof.get("source_theorem_sha256") != session.get("theorem_sha256")):
        raise ValueError("fresh baseline evidence is incomplete; inspect it, do not auto-rerun")
    evidence = repo / "v1-result/evidence/multiround"
    archive = evidence / "raw-evidence.tar.gz"
    if sha(archive.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError("historical input archive identity changed")
    entries = json.loads((evidence / "raw-evidence-manifest.json").read_bytes())["files"]
    expected = {item["path"]: item["sha256"] for item in entries}
    selected = ("05-CubeAccumulator.lean", "03-ScaledTriangular.lean", "02-AffineAccumulator.lean")
    inputs = {}
    with tarfile.open(archive, "r:gz") as tar:
        for name in selected:
            key = "inputs/" + name
            members = [m for m in tar.getmembers() if m.name == key]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError("selected input is missing or ambiguous")
            data = tar.extractfile(members[0]).read()
            if sha(data) != expected[key]:
                raise ValueError("selected input checksum mismatch")
            inputs[name] = data
    if sha(inputs[selected[0]]) != session["theorem_sha256"]:
        raise ValueError("target input changed from recorded baseline")
    pins = {"experience_id": release["experience_id"], "experience_weights_sha256": release["weights_sha256"],
            "experience_snapshot_sha256": release["source"]["snapshot_sha256"]}
    batch = [{"session_id": sid, "theorem_file": "inputs/" + name, **pins} for sid, name in zip(
        (fresh["session_id"], "inherit-concurrent-03", "inherit-refill-02"), selected)]
    plan = {"schema_version": "reap.incremental-gpu-plan.v1", "execution_started": False,
        "reused_evidence_sha256": hashes, "input_archive_sha256": ARCHIVE_SHA256,
        "input_sha256": {name: sha(data) for name, data in inputs.items()}, "experience": pins,
        "do_not_repeat": ["source training/publication", "7B inheritance mechanism", "fresh target", "historical five-question runs"],
        "batch": batch, "concurrency": 2, "max_resident_sessions": 2, "retire_completed": True,
        "max_updates": 5, "gamma": 0.99, "policy_scoring": "tokenwise", "max_post_update_kl": None,
        "new_acceptance": ["pinned inherited initialization before real Lean search",
            "real search feedback updates consumed later by the same tree",
            "overlapping CPU sessions with serialized, identity-correct GPU operations",
            "third task admitted after durable terminal artifacts and explicit retirement",
            "independent offline Lean verification for every resulting proof"],
        "separate_optional_probe": "fixed-token policy scorer equivalence and timing; no synthetic training rerun",
        "run_boundaries": ["fresh service/snapshot/job/output roots required; never reuse an uncertain job UUID",
            "keep existing experience root immutable; recheck all model files and source manifests before GPU load",
            "default KL guard stays disabled to match the accepted source contract",
            "no automatic start, download, publish, experiment submission, or GPU execution by this preparer"],
        "limits": ["known search exhaustion is a recorded terminal result, not a proof",
            "overlap must be measured, not inferred from requested concurrency",
            "concurrency and source-code differences prevent strict speed comparison with the old fresh baseline",
            "one inherited run does not establish TTT/inheritance performance improvement",
            "preparation checks recorded evidence only; it does not rerun Lean or validate GPU tensors"]}
    # Validate all inputs and records before creating any output. Never replace prior evidence.
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    for name, data in inputs.items():
        (output / "inputs" / name).write_bytes(data)
    (output / "batch.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in batch), encoding="utf8")
    (output / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = prepare(Path(__file__).resolve().parents[1], args.prior_run.resolve(), args.output_dir.resolve())
    print(json.dumps({"prepared": True, "experiments": len(plan["batch"]), "gpu_started": False,
                      "reused_files": len(plan["reused_evidence_sha256"]), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
