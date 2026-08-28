"""Bind an accepted Matchmaker attempt to its actual target-family dataset.

Read-only with respect to evidence; opening Matchmaker takes its existing writer
lease and validates the durable event chain. Run after releasing that writer.
This module does not train, infer target truth from a variant, or promote a
specialist to generalist. It does not retroactively strengthen old v3 scopes.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import closed_problem as closed
from .matchmaker import Matchmaker, _sha
from .target_curriculum import load_curriculum, prepare_member
from .verified_collector import PROFILE, MIXED_OBJECTIVE
from .verified_dataset_store import read_bundle, safe_directory
from .verified_trajectory import load_verified_dataset


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def admit_dataset(*, curriculum_dir, curriculum_sha256, scheduler_dir, scheduler_run_sha256,
                  attempt_id, verification_dir, dataset_root, dataset_sha256,
                  expected_model_release_sha256, num_samples, max_tokens):
    """Return a reproducible admission receipt only after all source checks.

    Curriculum/source identities are independent of execution-wrapper hashes.
    The expected release is explicit: a later specialist may collect more family
    data, but cannot silently substitute an arbitrary actor model. The future
    specialist coordinator must bind this receipt into every training catalog.
    """
    for value in (dataset_sha256, expected_model_release_sha256, scheduler_run_sha256):
        _sha(value, "admission pin")
    curriculum = load_curriculum(curriculum_dir, expected_sha256=curriculum_sha256)
    with Matchmaker.open(scheduler_dir, scheduler_run_sha256) as scheduler:
        _require(closed.canonical(scheduler._run["curriculum"]) ==
            closed.canonical(curriculum["matchmaker_curriculum"]), "scheduler uses another curriculum")
        status = scheduler.status()
        _require(type(attempt_id) is str and attempt_id in status["attempts"], "unknown curriculum attempt")
        item = status["attempts"][attempt_id]
        intent, result = item["intent"], item["result"]
        _require(result is not None and result["outcome"] == "accepted", "only independently accepted attempts can train")
        proposal, evidence = intent["proposal"], result["evidence"]
        _require(proposal["model_release_sha256"] == expected_model_release_sha256
            and evidence["dataset_sha256"] == dataset_sha256, "attempt release or dataset differs")
        prepared = prepare_member(curriculum, proposal["problem_id"], polarity=proposal["polarity"],
            budget_steps=proposal["budget_steps"], num_samples=num_samples, max_tokens=max_tokens)
        source_pin = prepared["execution_source_sha256"]
        _require(proposal["problem_sha256"] == prepared["problem"]["problem_sha256"]
            and proposal["attempted_prop_sha256"] == prepared["attempted"]["attempted_prop_sha256"]
            and intent["attempt_source"]["execution_source_sha256"] == source_pin
            and evidence["execution_source_sha256"] == source_pin,
            "attempt does not execute the mechanically reconstructed curriculum member")

        directory = Path(dataset_root)/dataset_sha256
        dataset = load_verified_dataset(directory, expected_sha256=dataset_sha256)
        bundle = read_bundle(directory)
        _require(closed.digest(bundle["dataset.json"]) == dataset_sha256
            and json.loads(bundle["dataset.json"]) == dataset
            and all(closed.digest(bundle[name]) == pin for name, pin in dataset["inputs_sha256"].items()),
            "dataset changed while binding its curriculum provenance")
        session = json.loads(bundle["session.json"])
        _require(bundle["source.lean"] == prepared["execution_source"]
            and dataset["theorem_sha256"] == source_pin
            and dataset["theorem"] == prepared["theorem"]
            and dataset["session_id"] == attempt_id == session.get("session_id")
            and dataset["tree_id"] == session.get("tree_id"), "dataset statement/actor identity differs")
        _require(session.get("profile") == PROFILE and session.get("role") == "actor"
            and session.get("training_enabled") is False
            and session.get("learner_objective") == MIXED_OBJECTIVE
            and type(session.get("policy_version")) is int and session["policy_version"] == 0
            and type(dataset.get("final_policy_version")) is int and dataset["final_policy_version"] == 0
            and session.get("model_release_sha256") == expected_model_release_sha256
            and session.get("lineage", {}).get("model_release_sha256") == expected_model_release_sha256,
            "curriculum data requires the pinned fixed-v0 mixed collector")

        with safe_directory(Path(verification_dir)) as verified:
            files = {name: verified.read(name) for name in (
                "verification-receipt.json", "dataset-receipt.json", "proof-receipt.json")}
        verify, installed = (json.loads(files[name]) for name in ("verification-receipt.json", "dataset-receipt.json"))
        _require(closed.digest(files["verification-receipt.json"]) == evidence["verification_receipt_sha256"]
            and closed.digest(files["dataset-receipt.json"]) == evidence["dataset_receipt_sha256"]
            and files["proof-receipt.json"] == bundle["historical-proof-receipt.json"],
            "independent verification receipts are not those accepted by Matchmaker")
        _require(verify.get("schema_version") == "reap.collector-verify.result.v1" and verify.get("verified") is True
            and verify.get("session_id") == attempt_id and verify.get("theorem") == prepared["theorem"]
            and verify.get("model_release_sha256") == expected_model_release_sha256
            and verify.get("execution_source_sha256") == source_pin
            and verify.get("dataset_sha256") == dataset_sha256
            and verify.get("dataset_receipt_sha256") == closed.digest(files["dataset-receipt.json"])
            and verify.get("proof_receipt_sha256") == closed.digest(files["proof-receipt.json"])
            and verify.get("proof_sha256") == evidence["proof_sha256"] == closed.digest(bundle["accepted-proof.lean"])
            and verify.get("replay_receipt_sha256") == dataset["replay_receipt_sha256"]
            and verify.get("root_return") == dataset["root_return"] and verify.get("rows") == len(dataset["rows"])
            and verify.get("axioms") == dataset["axioms"]
            and type(verify.get("optimizer_updates")) is int and verify["optimizer_updates"] == 0,
            "verification did not accept this exact curriculum proof and replay")
        _require(installed.get("schema_version") == "reap.collector-verify.dataset-install.v1"
            and installed.get("status") in {"installed", "existing"}
            and installed.get("dataset_sha256") == dataset_sha256
            and installed.get("verification_profile") == PROFILE
            and installed.get("model_release_sha256") == expected_model_release_sha256
            and installed.get("execution_source_sha256") == source_pin
            and installed.get("proof_receipt_sha256") == verify["proof_receipt_sha256"]
            and installed.get("replay_receipt_sha256") == dataset["replay_receipt_sha256"],
            "registry installation receipt differs")
        # Re-admit after reading the verification files; a mixed-time bundle
        # cannot authorize a catalog entry merely because the first read passed.
        _require(load_verified_dataset(directory, expected_sha256=dataset_sha256) == dataset
            and read_bundle(directory) == bundle, "dataset changed during admission")
        receipt = {"schema_version": "reap.curriculum.dataset-admission.v1",
            "curriculum_sha256": curriculum_sha256,
            "target_problem_sha256": curriculum["plan"]["target"]["problem_sha256"],
            "scheduler_run_sha256": scheduler_run_sha256, "scheduler_head_sha256": status["head_sha256"],
            "intent_sha256": intent["intent_sha256"], "result_event_sha256": item["result_event_sha256"],
            "attempt_id": attempt_id, "problem_sha256": proposal["problem_sha256"],
            "attempted_prop_sha256": proposal["attempted_prop_sha256"], "polarity": proposal["polarity"],
            "execution_source_sha256": source_pin, "model_release_sha256": expected_model_release_sha256,
            "tree_id": dataset["tree_id"], "proof_sha256": evidence["proof_sha256"],
            "verification_receipt_sha256": evidence["verification_receipt_sha256"],
            "dataset_receipt_sha256": evidence["dataset_receipt_sha256"],
            "replay_receipt_sha256": dataset["replay_receipt_sha256"], "dataset_sha256": dataset_sha256,
            "rows": len(dataset["rows"])}
        return {**receipt, "admission_sha256": closed.digest(receipt)}
