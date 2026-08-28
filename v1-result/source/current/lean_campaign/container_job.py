"""One container phase; invoked only by the independently audited launcher."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if sys.flags.optimize:
    raise RuntimeError("optimized Python is forbidden for this assertion-checked experiment")
BUNDLE = Path(__file__).absolute().parent
sys.path.insert(0, str(BUNDLE / "frozen"))
sys.path.insert(0, str(BUNDLE))
from cpu_runtime import closed_problem as cp
from cpu_runtime import matchmaker as mm
from binding import finalize
from cpu_runtime.verified_collector import (run_collector, CollectorCoordinator, classify_terminal,
    validate_created, MIXED_OBJECTIVE)
from cpu_runtime.collector_verify import verify_collected
from cpu_runtime.verified_dataset_store import safe_directory

DATA = Path("/campaign")
PROJECT = Path(json.loads((BUNDLE / "plan.json").read_bytes())["project_dir"])


def read(path):
    with safe_directory(path.parent) as directory:
        return json.loads(directory.read(path.name))


def write(path, value):
    with safe_directory(path.parent, create=True) as directory:
        directory.write_new(path.name, mm.canonical_bytes(value))
        directory.sync()


def sha(path):
    with safe_directory(path.parent) as directory:
        return mm.content_sha256(directory.read(path.name))


def check_bundle(phase):
    manifest = read(BUNDLE / "manifest.json")
    for relative, expected in manifest["files_sha256"].items():
        assert sha(BUNDLE / relative) == expected, "frozen bundle hash mismatch"
    plan = read(BUNDLE / "plan.json")
    if "cpu_rebinding" in plan and phase != "preflight":
        binding = read(DATA / "runtime-plan-binding.json")
        assert binding["prepared_plan_sha256"] == sha(BUNDLE / "plan.json")
        assert binding["runtime_plan_sha256"] == sha(DATA / "runtime-plan.json")
        records = {}
        for case in plan["cases"]:
            directory = DATA / "preflight" / case["problem_id"]
            raw = (directory / "receipt.json").read_bytes(); receipt = json.loads(raw)
            assert sha(directory / "stdout.log") == receipt["stdout_sha256"]
            assert sha(directory / "stderr.log") == receipt["stderr_sha256"]
            records[case["problem_id"]] = (raw, read(directory / "descriptor.json"))
        expected = finalize(plan, (BUNDLE / "inputs/declarations.lean").read_bytes(), records)
        assert expected == read(DATA / "runtime-plan.json")
        plan = expected
    return plan


def check_intent(plan, sid):
    intent = read(DATA / "intents" / (sid + ".json"))
    expected = next(item for item in plan["expected_attempts"] if item["attempt_id"] == sid)
    assert all(intent["proposal"][key] == value for key, value in expected.items())
    assert intent["proposal"]["run_sha256"] == plan["run_sha256"]
    assert intent["intent_sha256"] == mm.content_sha256({k: v for k, v in intent.items() if k != "intent_sha256"})
    label = intent["proposal"]["problem_id"]
    assert intent["attempt_source"] == read(DATA / "preflight" / label / "descriptor.json")
    assert sha(DATA / "preflight" / label / "attempt.lean") == intent["attempt_source"]["execution_source_sha256"]
    return intent


def preflight(plan, label):
    case = next(item for item in plan["cases"] if item["problem_id"] == label)
    problem = next(item for item in plan["curriculum"]["problems"] if item["problem_id"] == label)
    command = ["lake", "env", "lean", "--version"]
    probe = DATA / "version-probe" / label
    write(probe / "intent.json", {"command": command, "mutation_retry_allowed": False})
    process = subprocess.run(command, cwd=PROJECT, capture_output=True, timeout=60)
    write(probe / "receipt.json", {"command": command, "returncode": process.returncode,
        "stdout": process.stdout.decode(), "stderr": process.stderr.decode()})
    assert process.returncode == 0, "Lean version check failed; inspect its full receipt"
    version = process.stdout.decode().strip()
    assert version == read(BUNDLE / "inputs/environment.json")["lean_version"]
    if "cpu_rebinding" in plan:
        module_file = probe / "required-modules.lean"
        with module_file.open("xb") as stream:
            stream.write(b"import ReapRuntime\nimport Reap.VerifiedCollector\n")
        command = ["lake", "env", "lean", str(module_file)]
        write(probe / "modules-intent.json", {"command": command, "mutation_retry_allowed": False})
        process = subprocess.run(command, cwd=PROJECT, capture_output=True, timeout=180)
        write(probe / "modules-receipt.json", {"command": command, "returncode": process.returncode,
            "stdout": process.stdout.decode(), "stderr": process.stderr.decode()})
        assert process.returncode == 0, "required Reap modules failed before any GPU session creation"
    prepared = cp.prepare(environment_sha256=plan["curriculum"]["environment_sha256"],
        declarations=(BUNDLE / "inputs/declarations.lean").read_bytes(), closed_prop_name=problem["closed_prop_name"],
        polarity=case["desired_polarity"], budget_steps=8, unfold_closed_prop=True)
    assert prepared["execution_source"] == (BUNDLE / "inputs" / label / "attempt.lean").read_bytes()
    descriptor = cp.compile_preflight(prepared, project_dir=PROJECT, output_dir=DATA / "preflight" / label)
    assert descriptor["execution_source_sha256"] == case.get("prepared_descriptor", case.get("prior_descriptor"))["execution_source_sha256"]
    assert descriptor["attempted_prop_sha256"] == case.get("prepared_descriptor", case.get("prior_descriptor"))["attempted_prop_sha256"]
    write(DATA / "preflight" / label / "descriptor.json", descriptor)
    return descriptor


def normalize_terminal(plan, intent, directory):
    sid = intent["proposal"]["attempt_id"]
    audit = DATA / "terminal-audit" / sid
    audit.mkdir(parents=True, exist_ok=False)
    required = {"session-intent.json", "session.json", "create-intent.json", "create-receipt.json", "collector-result.json",
                "source.lean", "run.lean", "process.json", "observer.jsonl", "result.json", "raw_tree.json", "stdout.log", "stderr.log"}
    assert required <= {p.name for p in directory.iterdir() if p.is_file()}, "terminal evidence missing"
    report = read(directory / "collector-result.json")
    session = read(directory / "session.json")
    route = read(DATA / "routes" / (sid + ".json"))
    assert route["intent_sha256"] == intent["intent_sha256"] and route["endpoint"] in plan["endpoints"]
    base = route["endpoint"]["base_url"]
    assert session["policy_base_url"] == base + f"/sessions/{sid}/policy/v1"
    assert session["value_base_url"] == base + f"/sessions/{sid}/value/v1"
    created = read(directory / "create-receipt.json")
    process = read(directory / "process.json")
    assert report["schema_version"] == "reap.verified-collector.result.v1"
    assert all(mm.canonical_bytes(report[key]) == mm.canonical_bytes(value) for key, value in session.items())
    assert report["session_id"] == sid and report["model_release_sha256"] == plan["model_release_sha256"]
    assert report["theorem_sha256"] == intent["attempt_source"]["execution_source_sha256"] == sha(directory / "source.lean")
    assert report["training_enabled"] is False and type(report["optimizer_updates"]) is int and report["optimizer_updates"] == 0
    assert report["policy_version"] == 0 and type(report["policy_version"]) is int
    assert report["error"] is None and report["mutation_retry_allowed"] is False and report["independent_verified"] is False
    assert report["learner_objective"] == MIXED_OBJECTIVE and report["puct_value_gamma"] == .99 and report["return_discount"] == 1
    maximum = validate_created(created, sid, theorem_sha256=report["theorem_sha256"],
        model_release_sha256=plan["model_release_sha256"], learner_objective=MIXED_OBJECTIVE)
    assert mm.canonical_bytes(session["initialization_receipt"]) == mm.canonical_bytes(created)
    assert session["initialization_receipt_sha256"] == sha(directory / "create-receipt.json")
    assert mm.canonical_bytes(session["lineage"]) == mm.canonical_bytes(created["lineage"])
    source = (directory / "source.lean").read_text()
    expected_run = ("import Reap.VerifiedCollector\n" + source.replace("  reapTrainingMCTS",
        "  set_option reap.visit_discount 990 in\n    reapVerifiedCollectorMCTS")).encode()
    assert (directory / "run.lean").read_bytes() == expected_run
    assert process["run_source_sha256"] == mm.content_sha256(expected_run) and process["pid"] == report["lean_pid"]
    coordinator = CollectorCoordinator(sid, session["tree_id"], audit / "checked-acks",
        model_release_sha256=plan["model_release_sha256"], puct_value_gamma=.99, max_distance=maximum)
    for line in (directory / "observer.jsonl").read_bytes().splitlines():
        if line.strip():
            coordinator.accept(json.loads(line))
    coordinator.finish()
    status, result = classify_terminal(directory, coordinator, report["returncode"])
    assert status == report["status"] and report["checkpoint_count"] == coordinator.last_step + 1
    assert type(coordinator.last_step) is int and 0 <= coordinator.last_step <= intent["proposal"]["budget_steps"]
    if status == "solved_pending_independent_verification":
        assert report["root_verified"] is True
        candidate = read(directory / "proof-candidate.json")
        assert candidate["generated_proof_sha256"] == sha(directory / "proof.lean")
        assert candidate["proof_from_session"] == sid and candidate["source_theorem_sha256"] == report["theorem_sha256"]
    else:
        assert status == "exhausted" and report["root_verified"] is False
    files = {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob("*")) if p.is_file()}
    write(audit / "files.json", files)
    envelope = {"schema_version": "reap.collector-batch.search.v1", "attempt_id": sid,
        "intent_sha256": intent["intent_sha256"], "outcome": "solved_candidate" if report["root_verified"] else "exhausted",
        "artifacts_complete": True, "artifacts_sha256": sha(audit / "files.json"),
        "terminal_receipt_sha256": sha(directory / "collector-result.json"),
        "actual_steps": coordinator.last_step, "policy_version": 0, "raw": report}
    write(audit / "terminal-envelope.json", envelope)
    return envelope


def search(plan, sid):
    intent = check_intent(plan, sid); label = intent["proposal"]["problem_id"]
    route = read(DATA / "routes" / (sid + ".json"))
    assert set(route) == {"endpoint", "intent_sha256"}
    assert route["intent_sha256"] == intent["intent_sha256"]
    assert route["endpoint"] in plan["endpoints"]
    assert route["endpoint"]["model_release_sha256"] == intent["proposal"]["model_release_sha256"]
    result = run_collector(session_id=sid, project_dir=PROJECT,
        theorem_file=str(DATA / "preflight" / label / "attempt.lean"), theorem=plan["expected_theorem"],
        output_root=DATA / "sessions", gpu_base_url=route["endpoint"]["base_url"],
        model_release_sha256=plan["model_release_sha256"], puct_value_gamma=.99, learner_objective=MIXED_OBJECTIVE)
    assert result["status"] in {"solved_pending_independent_verification", "exhausted"}, "unknown collector outcome; never retry"
    return normalize_terminal(plan, intent, DATA / "sessions" / sid)


def verify(plan, sid):
    intent = check_intent(plan, sid)
    result = verify_collected(DATA / "sessions" / sid, DATA / "verified" / sid, PROJECT,
        DATA / "dataset-store", plan["model_release_sha256"], intent["attempt_source"]["execution_source_sha256"],
        image_sha256=plan["image_id"], network="none", expected_theorem=plan["expected_theorem"],
        replay_module=BUNDLE / "frozen/VerifiedReplay.lean", timeout=180)
    assert result["verified"] is True and result["session_id"] == sid and result["new_search"] is False
    assert result["model_release_sha256"] == plan["model_release_sha256"]
    assert result["execution_source_sha256"] == intent["attempt_source"]["execution_source_sha256"]
    terminal = read(DATA / "terminal-audit" / sid / "terminal-envelope.json")
    accepted = {"schema_version": "reap.matchmaker.result.v1", "attempt_id": sid,
        "intent_sha256": intent["intent_sha256"], "outcome": "accepted", "evidence": {
            "verdict": {"prove": "proved", "disprove": "disproved"}[intent["proposal"]["polarity"]],
            "problem_sha256": intent["proposal"]["problem_sha256"],
            "attempted_prop_sha256": intent["proposal"]["attempted_prop_sha256"],
            "execution_source_sha256": result["execution_source_sha256"], "model_release_sha256": result["model_release_sha256"],
            "proof_sha256": result["proof_sha256"], "verification_receipt_sha256": result["verification_receipt_sha256"],
            "dataset_sha256": result["dataset_sha256"], "dataset_receipt_sha256": result["dataset_receipt_sha256"],
            "actual_steps": terminal["actual_steps"]}}
    write(DATA / "verify-results" / (sid + ".json"), {"verification": result, "matchmaker_result": accepted})
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "search", "verify"))
    parser.add_argument("--label", choices=("true-positive", "false-negative"))
    parser.add_argument("--session-id")
    args = parser.parse_args(); plan = check_bundle(args.phase)
    if args.phase == "preflight":
        assert args.label is not None and args.session_id is None
        result = preflight(plan, args.label)
    else:
        assert args.session_id in {item["attempt_id"] for item in plan["expected_attempts"]} and args.label is None
        result = {"search": search, "verify": verify}[args.phase](plan, args.session_id)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
