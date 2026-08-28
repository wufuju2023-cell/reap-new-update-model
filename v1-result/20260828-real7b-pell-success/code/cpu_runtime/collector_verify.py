"""Verify an already completed fixed-release collector; never search or train.

Run inside the caller-pinned offline CPU container. ``image_sha256`` and
``network`` are explicit launcher attestations, not facts inferred by Python.
Put the dataset registry on a native Linux named volume, not an NTFS bind mount.
Every output directory is exclusive. Failure preserves evidence and never
automatically repeats Lean or an ambiguous registry publication.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import textwrap
import time

from .verified_collector import (CollectorCoordinator, OBJECTIVE, PROFILE,
    classify_terminal, digest, gamma_option, validate_created)
from .verified_dataset_store import install_verified_dataset, safe_directory
from .verified_trajectory import (THEOREM, checked_axioms, encode, export_verified,
    load_verified_dataset, plan_success_path, require, sha)


INPUT_FILES = (
    "session.json", "collector-result.json", "create-receipt.json", "result.json",
    "raw_tree.json", "observer.jsonl", "source.lean", "run.lean", "process.json",
    "proof.lean", "proof-candidate.json",
)


def _absolute(path: Path) -> Path:
    require(".." not in Path(path).parts, "parent path traversal refused")
    return Path(os.path.abspath(path))


def _same_json(left, right) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _validate(files: dict[str, bytes], output: Path, release: str, source_pin: str,
              expected_theorem: str | None) -> tuple[dict, dict]:
    session, report, created, result, tree, candidate, process = (
        json.loads(files[name]) for name in (
            "session.json", "collector-result.json", "create-receipt.json", "result.json",
            "raw_tree.json", "proof-candidate.json", "process.json"))
    require(all(isinstance(v, dict) for v in (session, report, created, result, tree, candidate, process)),
            "collector metadata must be objects")
    theorem = session.get("theorem")
    require(isinstance(theorem, str) and THEOREM.fullmatch(theorem)
            and (expected_theorem is None or theorem == expected_theorem), "theorem identity mismatch")
    require(session.get("profile") == PROFILE and session.get("role") == "actor"
            and session.get("training_enabled") is False
            and type(session.get("policy_version")) is int and session["policy_version"] == 0
            and type(session.get("return_discount")) is int and session["return_discount"] == 1
            and session.get("model_release_sha256") == release
            and session.get("theorem_sha256") == source_pin == sha(files["source.lean"]),
            "fixed release/source/session contract mismatch")
    require(all(_same_json(report.get(key), value) for key, value in session.items())
            and report.get("schema_version") == "reap.verified-collector.result.v1"
            and report.get("status") == "solved_pending_independent_verification"
            and report.get("root_verified") is True and report.get("independent_verified") is False
            and type(report.get("optimizer_updates")) is int and report["optimizer_updates"] == 0
            and type(report.get("returncode")) is int and report["returncode"] == 0
            and report.get("error") is None and report.get("mutation_retry_allowed") is False,
            "collector is not a complete solved fixed-v0 candidate")
    maximum = validate_created(created, session["session_id"], theorem_sha256=source_pin,
        model_release_sha256=release, learner_objective=session.get("learner_objective", OBJECTIVE))
    require(_same_json(session.get("initialization_receipt"), created)
            and session.get("initialization_receipt_sha256") == sha(files["create-receipt.json"])
            and _same_json(session.get("lineage"), created["lineage"]), "actor initialization receipt mismatch")
    option = gamma_option(session.get("puct_value_gamma"))
    source = files["source.lean"].decode("utf8")
    require(source.startswith("import ReapRuntime\n") and source.count("  reapTrainingMCTS") == 1,
            "unsupported source marker/import")
    run_source = ("import Reap.VerifiedCollector\n" + source.replace("  reapTrainingMCTS",
        f"  set_option reap.visit_discount {option} in\n    reapVerifiedCollectorMCTS")).encode()
    require(files["run.lean"] == run_source and process.get("run_source_sha256") == sha(run_source)
            and process.get("tree_id") == session["tree_id"]
            and process.get("pid") == report.get("lean_pid"), "collector execution source/process mismatch")
    coordinator = CollectorCoordinator(session["session_id"], session["tree_id"], output/"checked-acks",
        model_release_sha256=release, puct_value_gamma=session["puct_value_gamma"], max_distance=maximum)
    events = [json.loads(line) for line in files["observer.jsonl"].splitlines() if line.strip()]
    for event in events:
        coordinator.accept(event)
    coordinator.finish()
    status, _ = classify_terminal(output, coordinator, report["returncode"])
    require(status == "solved_pending_independent_verification"
            and type(report.get("checkpoint_count")) is int
            and report["checkpoint_count"] == coordinator.last_step+1, "terminal checkpoint mismatch")
    selected = plan_success_path(tree, events, session)
    proof = source.replace("  reapTrainingMCTS", textwrap.indent(selected["proof_script"], "  "))
    proof += "\n#print axioms " + theorem + "\n"
    require(result.get("proof_script") == selected["proof_script"] and files["proof.lean"] == proof.encode(),
            "proof bytes differ from selected successful path")
    require(candidate == {"proof_from_session": session["session_id"],
        "source_theorem_sha256": source_pin, "generated_proof_sha256": sha(files["proof.lean"]),
        "model_release_sha256": release, "independent_verified": False,
        "candidate_root_return": selected["root_return"], "candidate_rows": len(selected["rows"])},
        "proof candidate binding mismatch")
    return session, selected


def verify_collected(session_dir: Path, output_dir: Path, project_dir: Path,
                     dataset_store: Path, expected_release_sha256: str,
                     expected_execution_source_sha256: str, lean_bin: str = "lake", *,
                     image_sha256: str, network: str, expected_theorem: str | None = None,
                     replay_module: Path | None = None, timeout: float = 180) -> dict:
    """Verify once, export using the unchanged loader, and install the 17 files.

    The caller must bind image/network to its real container launch receipt.
    Input pins identify the *execution wrapper*, not the underlying proposition.
    Returns only after both Lean processes and strict registry admission pass.
    """
    require(digest(expected_release_sha256) and digest(expected_execution_source_sha256)
            and digest(image_sha256) and network == "none", "explicit release/source/offline-image pins required")
    require(lean_bin == "lake", "existing replay exporter requires lake; alternate binary unsupported")
    require(type(timeout) in (float, int) and math.isfinite(timeout) and timeout > 0,
            "timeout must be positive finite")
    if expected_theorem is not None:
        require(isinstance(expected_theorem, str) and THEOREM.fullmatch(expected_theorem), "invalid expected theorem")
    session_dir, output_dir, project_dir, dataset_store = map(_absolute,
        (session_dir, output_dir, project_dir, dataset_store))
    require(not output_dir.is_relative_to(session_dir) and not session_dir.is_relative_to(output_dir)
            and not dataset_store.is_relative_to(output_dir) and not output_dir.is_relative_to(dataset_store),
            "source, evidence output and registry must be separate directories")
    module = _absolute(replay_module or Path(__file__).resolve().parents[1]/"containers/cpu/verified-replay/VerifiedReplay.lean")
    # Retain safe directory handles; no resolve() through caller-supplied links.
    with safe_directory(session_dir) as source_dir, safe_directory(project_dir), \
            safe_directory(module.parent) as module_dir, safe_directory(output_dir.parent, create=True) as parent:
        with parent.child(output_dir.name, create=True) as output:
            started, stage = time.perf_counter(), "freeze_inputs"
            inputs: dict[str, bytes] = {}
            try:
                inputs = {name: source_dir.read(name) for name in INPUT_FILES}
                for name, data in inputs.items():
                    output.write_new(name, data)
                module_bytes = module_dir.read(module.name)
                output.write_new("VerifiedReplay.lean", module_bytes)
                hashes = {name: sha(data) for name, data in inputs.items()}
                hashes["VerifiedReplay.lean"] = sha(module_bytes)
                implementation_dir = Path(__file__).absolute().parent
                with safe_directory(implementation_dir) as code:
                    implementation = {name: code.read(name) for name in (
                        "collector_verify.py", "verified_collector.py", "verified_trajectory.py", "verified_dataset_store.py")}
                output.write_new("input-manifest.json", encode({"schema_version": "reap.collector-verify.inputs.v1",
                    "inputs_sha256": hashes, "expected_release_sha256": expected_release_sha256,
                    "expected_execution_source_sha256": expected_execution_source_sha256,
                    "implementation_sha256": {name: sha(data) for name, data in implementation.items()},
                    "launcher_attestation": {"image": image_sha256, "network": network},
                    "scope": "launcher must independently bind container image/network; not inferred by verifier"}))

                def unchanged() -> None:
                    require(all(output.read(name) == data and source_dir.read(name) == data
                                for name, data in inputs.items()), "collector input changed during verification")
                    require(output.read("VerifiedReplay.lean") == module_bytes
                            and module_dir.read(module.name) == module_bytes, "replay module changed during verification")
                    with safe_directory(implementation_dir) as code:
                        require(all(code.read(name) == data for name, data in implementation.items()),
                                "verification implementation changed during execution")

                stage = "validate_collector"
                session, selected = _validate(inputs, output_dir, expected_release_sha256,
                    expected_execution_source_sha256, expected_theorem)
                stage = "whole_proof"
                command = [lean_bin, "env", "lean", str(output_dir/"proof.lean")]
                output.write_new("proof-intent.json", encode({"command": command, "cwd": str(project_dir),
                    "timeout_seconds": timeout, "proof_sha256": hashes["proof.lean"],
                    "mutation_retry_allowed": False}))
                before = time.perf_counter()
                try:
                    process = subprocess.run(command, cwd=project_dir, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, timeout=timeout)
                except subprocess.TimeoutExpired as error:
                    output.write_new("proof.stdout", error.stdout or b"")
                    output.write_new("proof.stderr", error.stderr or b"")
                    raise
                output.write_new("proof.stdout", process.stdout)
                output.write_new("proof.stderr", process.stderr)
                proof_receipt = {"schema_version": "reap.collector-verify.proof.v1",
                    "proof_from_session": session["session_id"], "theorem": session["theorem"],
                    "source_theorem_sha256": expected_execution_source_sha256,
                    "generated_proof_sha256": hashes["proof.lean"],
                    "model_release_sha256": expected_release_sha256,
                    "image": image_sha256, "network": network, "command": command, "cwd": str(project_dir),
                    "returncode": process.returncode, "elapsed_seconds": time.perf_counter()-before,
                    "stdout": process.stdout.decode("utf8"), "stderr": process.stderr.decode("utf8")}
                output.write_new("proof-receipt.json", encode(proof_receipt))
                require(type(process.returncode) is int and process.returncode == 0, "independent whole proof failed")
                axioms = checked_axioms(proof_receipt["stdout"], session["theorem"])
                unchanged()
                stage = "replay"
                dataset = export_verified(session_dir=output_dir, source=output_dir/"source.lean",
                    proof=output_dir/"proof.lean", proof_receipt=output_dir/"proof-receipt.json",
                    theorem=session["theorem"], output=output_dir/"dataset", lean_project=project_dir,
                    replay_module=output_dir/"VerifiedReplay.lean", timeout=timeout)
                unchanged()
                with safe_directory(output_dir/"dataset") as bundle:
                    dataset_pin = sha(bundle.read("dataset.json"))
                    replay_receipt_pin = sha(bundle.read("replay-receipt.json"))
                require(load_verified_dataset(output_dir/"dataset", expected_sha256=dataset_pin) == dataset,
                        "exported dataset admission mismatch")
                stage = "registry_install"
                output.write_new("install-intent.json", encode({"dataset_sha256": dataset_pin,
                    "dataset_store": str(dataset_store), "mutation_retry_allowed": False}))
                installed = install_verified_dataset(output_dir/"dataset", dataset_store, expected_sha256=dataset_pin)
                require(load_verified_dataset(Path(installed["path"]), expected_sha256=dataset_pin) == dataset,
                        "installed dataset admission mismatch")
                unchanged()
                output.write_new("dataset-receipt.json", encode({**installed,
                    "schema_version": "reap.collector-verify.dataset-install.v1",
                    "verification_profile": PROFILE, "model_release_sha256": expected_release_sha256,
                    "execution_source_sha256": expected_execution_source_sha256,
                    "proof_receipt_sha256": sha(output.read("proof-receipt.json")),
                    "replay_receipt_sha256": replay_receipt_pin}))
                receipt = {"schema_version": "reap.collector-verify.result.v1", "verified": True,
                    "session_id": session["session_id"], "theorem": session["theorem"],
                    "model_release_sha256": expected_release_sha256,
                    "execution_source_sha256": expected_execution_source_sha256,
                    "proof_sha256": hashes["proof.lean"], "proof_path": str(output_dir/"proof.lean"),
                    "proof_receipt_sha256": sha(output.read("proof-receipt.json")),
                    "proof_receipt_path": str(output_dir/"proof-receipt.json"),
                    "dataset_sha256": dataset_pin, "dataset_path": installed["path"],
                    "dataset_receipt_sha256": sha(output.read("dataset-receipt.json")),
                    "dataset_receipt_path": str(output_dir/"dataset-receipt.json"),
                    "replay_receipt_sha256": replay_receipt_pin,
                    "input_manifest_sha256": sha(output.read("input-manifest.json")),
                    "root_return": selected["root_return"], "rows": len(dataset["rows"]), "axioms": axioms,
                    "new_search": False, "optimizer_updates": 0, "mutation_retry_allowed": False,
                    "elapsed_seconds": time.perf_counter()-started}
                output.write_new("verification-receipt.json", encode(receipt))
                return {**receipt, "verification_receipt_sha256": sha(output.read("verification-receipt.json")),
                        "verification_receipt_path": str(output_dir/"verification-receipt.json")}
            except BaseException as error:
                # Preserve any successful install if the outcome was uncertain;
                # never delete, overwrite, resume search, or implicitly retry it.
                output.write_new("failed.json", encode({"schema_version": "reap.collector-verify.failure.v1",
                    "stage": stage, "error_type": type(error).__name__, "error": str(error),
                    "verified": False, "mutation_retry_allowed": False,
                    "registry_may_already_be_published": stage == "registry_install",
                    "elapsed_seconds": time.perf_counter()-started}))
                raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("session-dir", "output-dir", "project-dir", "dataset-store"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("expected-release-sha256", "expected-execution-source-sha256", "image-sha256"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--network", choices=("none",), required=True)
    parser.add_argument("--expected-theorem")
    parser.add_argument("--replay-module", type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)
    result = verify_collected(**vars(args))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
