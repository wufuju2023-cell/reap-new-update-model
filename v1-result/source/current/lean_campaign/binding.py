"""CPU image rebinding: identities stay pending until actual preflight receipts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from cpu_runtime import closed_problem as cp
from cpu_runtime import matchmaker as mm


def prepared_for(plan, declarations, case):
    problem = next(x for x in plan["curriculum"]["problems"] if x["problem_id"] == case["problem_id"])
    return cp.prepare(environment_sha256=plan["curriculum"]["environment_sha256"],
        declarations=declarations, closed_prop_name=problem["closed_prop_name"],
        polarity=case["desired_polarity"], budget_steps=8, unfold_closed_prop=True)


def make_pending(plan, original_declarations, environment):
    plan = deepcopy(plan)
    env_sha = mm.content_sha256(environment)
    declarations = original_declarations.rstrip() + b"\n-- portable CPU environment sha256: " + env_sha.encode() + b"\n"
    plan["curriculum"]["environment_sha256"] = env_sha
    plan["cpu_rebinding"] = {"state": "pending_actual_preflight", "environment": environment,
        "original_declarations_sha256": mm.content_sha256(original_declarations), "new_Lean_executed": False}
    plan["run_sha256"] = None
    plan["expected_attempts"] = []
    inputs = {"inputs/environment.json": mm.canonical_bytes(environment), "inputs/declarations.lean": declarations}
    for case in plan["cases"]:
        prepared = prepared_for(plan, declarations, case)
        problem = next(x for x in plan["curriculum"]["problems"] if x["problem_id"] == case["problem_id"])
        problem.update(problem_sha256=prepared["problem"]["problem_sha256"],
            declarations_sha256=prepared["problem"]["declarations_sha256"], compilation_receipt_sha256=None)
        case.pop("prior_descriptor", None)
        case["prepared_descriptor"] = {"execution_source_sha256": prepared["execution_source_sha256"],
            "attempted_prop_sha256": prepared["attempted"]["attempted_prop_sha256"], "budget_steps": 8}
        inputs[f'inputs/{case["problem_id"]}/attempt.lean'] = prepared["execution_source"]
        inputs[f'inputs/{case["problem_id"]}/preflight.lean'] = prepared["preflight_source"]
    plan["campaign_sha256"] = mm.content_sha256(plan)
    return plan, inputs


def validate_pending(plan, declarations):
    mm.require(plan["cpu_rebinding"]["state"] == "pending_actual_preflight" and
        plan["run_sha256"] is None and plan["expected_attempts"] == [], "preflight identity prematurely committed")
    mm.require(mm.content_sha256({k: v for k, v in plan.items() if k != "campaign_sha256"}) == plan["campaign_sha256"],
               "pending campaign pin differs")
    environment = plan["cpu_rebinding"]["environment"]
    mm.require(environment["cpu_image_id"] == plan["image_id"] and environment["network"] == "none" and
        environment["project_dir"] == plan["project_dir"] and
        mm.content_sha256(environment) == plan["curriculum"]["environment_sha256"], "environment pin differs")
    for case in plan["cases"]:
        prepared = prepared_for(plan, declarations, case)
        problem = next(x for x in plan["curriculum"]["problems"] if x["problem_id"] == case["problem_id"])
        mm.require(problem["compilation_receipt_sha256"] is None and
            problem["problem_sha256"] == prepared["problem"]["problem_sha256"] and
            problem["declarations_sha256"] == prepared["problem"]["declarations_sha256"], "pending problem differs")
        expected = {"execution_source_sha256": prepared["execution_source_sha256"],
            "attempted_prop_sha256": prepared["attempted"]["attempted_prop_sha256"], "budget_steps": 8}
        mm.require(case["prepared_descriptor"] == expected, "pending source differs")


def finalize(plan, declarations, records):
    """records contain raw actual receipt bytes and strict returned descriptors."""
    validate_pending(plan, declarations)
    result = deepcopy(plan)
    mm.require(set(records) == {c["problem_id"] for c in plan["cases"]}, "both preflight receipts required")
    for case in result["cases"]:
        label = case["problem_id"]
        raw, descriptor = records[label]
        receipt = json.loads(raw)
        prepared = prepared_for(plan, declarations, case)
        expected = {**case["prepared_descriptor"], "compilation_receipt_sha256": mm.content_sha256(raw)}
        mm.require(descriptor == expected, "actual preflight descriptor/receipt pin differs")
        required = {"schema_version": "reap.closed-prop.preflight.v1", "returncode": 0,
            "proof_verified": False, "scope": "closed_propositions_and_search_options_only",
            "environment_sha256": plan["curriculum"]["environment_sha256"],
            "problem_sha256": prepared["problem"]["problem_sha256"],
            "attempted_prop_sha256": prepared["attempted"]["attempted_prop_sha256"],
            "execution_source_sha256": prepared["execution_source_sha256"],
            "preflight_source_sha256": prepared["preflight_source_sha256"]}
        mm.require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k, v in required.items()),
                   "failed, old, or mismatched preflight receipt")
        problem = next(x for x in result["curriculum"]["problems"] if x["problem_id"] == label)
        problem["compilation_receipt_sha256"] = descriptor["compilation_receipt_sha256"]
        case["prior_descriptor"] = descriptor
    result["cpu_rebinding"]["state"] = "actual_preflights_bound"
    result["cpu_rebinding"]["new_Lean_executed"] = True
    with tempfile.TemporaryDirectory(prefix="bound-campaign-") as directory:
        with mm.Matchmaker.create(Path(directory) / "scheduler", result["curriculum"], result["config"]) as scheduler:
            result["run_sha256"] = scheduler.run_sha256
            result["expected_attempts"] = []
            for _ in range(2):
                proposal = scheduler.plan_next(result["model_release_sha256"])
                case = next(c for c in result["cases"] if c["problem_id"] == proposal["problem_id"])
                mm.require(proposal["polarity"] == case["desired_polarity"], "branch selection differs")
                result["expected_attempts"].append({k: proposal[k] for k in (
                    "attempt_id", "attempt_index", "attempted_prop_sha256", "budget_steps", "family_id",
                    "model_release_sha256", "polarity", "problem_id", "problem_sha256")})
                scheduler.reserve(proposal, case["prior_descriptor"])
    return result
