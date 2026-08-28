"""Pinned two-branch real campaign. Default/check mode never contacts the bridge."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

if sys.flags.optimize:
    raise RuntimeError("optimized Python is forbidden for this assertion-checked experiment")
BUNDLE = Path(__file__).absolute().parent
sys.path.insert(0, str(BUNDLE / "frozen"))
sys.path.insert(0, str(BUNDLE))
from cpu_runtime import matchmaker as mm
from binding import validate_pending, finalize
from cpu_runtime.replica_collector import ReplicaCollector, run_replica_collector_batch, validate_endpoints
from cpu_runtime.http_clients import GpuHttpClient
from cpu_runtime.transport_budget import DEFAULT_BUDGET
from cpu_runtime.verified_dataset_store import safe_directory


def read(path):
    with safe_directory(path.parent) as directory:
        return json.loads(directory.read(path.name))


def write(path, value):
    with safe_directory(path.parent, create=True) as directory:
        directory.write_new(path.name, mm.canonical_bytes(value)); directory.sync()


def file_hash(path):
    with safe_directory(path.parent) as directory:
        return mm.content_sha256(directory.read(path.name))


def check_bundle():
    manifest = read(BUNDLE / "manifest.json")
    for relative, expected in manifest["files_sha256"].items():
        assert file_hash(BUNDLE / relative) == expected, "frozen bundle hash mismatch"
    plan = read(BUNDLE / "plan.json")
    mm.validate_config(plan["config"])
    if "cpu_rebinding" in plan:
        validate_pending(plan, (BUNDLE / "inputs/declarations.lean").read_bytes())
        validate_endpoints(plan["endpoints"])
        return plan
    mm.validate_curriculum(plan["curriculum"])
    assert plan["config"]["max_attempts"] == 2 and plan["config"]["total_steps"] == 16
    assert plan["config"]["base_steps"] == plan["config"]["cap_steps"] == 8
    validate_endpoints(plan["endpoints"])
    assert plan["project_dir"].startswith("/") and ".." not in Path(plan["project_dir"]).parts
    assert len(plan["curriculum"]["problems"]) == 2
    with tempfile.TemporaryDirectory(prefix="check-matchmaker-") as directory:
        with mm.Matchmaker.create(Path(directory) / "scheduler", plan["curriculum"], plan["config"]) as scheduler:
            assert scheduler.run_sha256 == plan["run_sha256"]
            for expected in plan["expected_attempts"]:
                proposal = scheduler.plan_next(plan["model_release_sha256"])
                assert all(proposal[key] == value for key, value in expected.items())
                case = next(item for item in plan["cases"] if item["problem_id"] == proposal["problem_id"])
                assert proposal["polarity"] == case["desired_polarity"]
                scheduler.reserve(proposal, case["prior_descriptor"])
    return plan


TRUSTED_REPOSITORIES = ['/opt/reap-runtime/.lake/packages/Qq', '/opt/reap-runtime/.lake/packages/plausible', '/opt/reap-runtime/.lake/packages/batteries', '/opt/reap-runtime/.lake/packages/mathlib', '/opt/reap-runtime/.lake/packages/aesop', '/opt/reap-runtime/.lake/packages/LeanSearchClient', '/opt/reap-runtime/.lake/packages/openAI_client', '/opt/reap-runtime/.lake/packages/importGraph', '/opt/reap-runtime/.lake/packages/requests', '/opt/reap-runtime/.lake/packages/Cli', '/opt/reap-runtime/.lake/packages/proofwidgets', '/opt/reap', '/opt/reap/.lake/packages/batteries', '/opt/reap/.lake/packages/openAI_client', '/opt/reap/.lake/packages/requests']

def git_environment(project="/opt/reap-runtime"):
    paths = [p.replace("/opt/reap-runtime", project) for p in TRUSTED_REPOSITORIES]
    result = {'GIT_CONFIG_COUNT': str(len(paths))}
    for i, path in enumerate(paths):
        result[f'GIT_CONFIG_KEY_{i}'] = 'safe.directory'
        result[f'GIT_CONFIG_VALUE_{i}'] = path
    return result

def git_flags(project):
    return [part for key, value in git_environment(project).items() for part in ('--env', key+'='+value)]

def campaign_identity(plan):
    return plan.get("campaign_sha256") or plan["run_sha256"]


def command_for(plan, phase, key):
    assert phase in {"preflight", "search", "verify"}
    network = "host" if phase == "search" else "none"
    name = "portable-" + plan["run_id"] + "-" + phase + "-" + key
    args = ["podman", "create", "--pull=never", "--name", name,
        "--network", network, "--user", "0:0", "--workdir", plan["project_dir"], "--entrypoint", "python3",
        "--label", "reap.campaign=" + campaign_identity(plan), "--env", "PYTHONPATH=/bundle/frozen",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONOPTIMIZE=", "--env", "HTTP_PROXY=",
        "--env", "HTTPS_PROXY=", "--env", "ALL_PROXY=", "--env", "NO_PROXY=127.0.0.1,localhost",
        *git_flags(plan["project_dir"]),
        "--volume", str(BUNDLE) + ":/bundle:ro", "--mount", "type=volume,source=" + plan["volume"] + ",destination=/campaign",
        plan["image_id"], "-B", "/bundle/container_job.py", phase]
    args += ["--label", key] if phase == "preflight" else ["--session-id", key]
    return args


def validate_inspect(plan, inspection, phase):
    assert inspection["Image"].removeprefix("sha256:") == plan["image_id"]
    assert inspection["HostConfig"]["NetworkMode"] == ("host" if phase == "search" else "none")
    assert inspection["Config"]["User"] == "0:0"
    actual = dict(item.split("=", 1) for item in inspection["Config"]["Env"] if item.startswith("GIT_CONFIG"))
    assert actual == git_environment(plan["project_dir"])
    assert inspection["Config"]["Labels"]["reap.campaign"] == campaign_identity(plan)
    data = next(item for item in inspection["Mounts"] if item["Destination"] == "/campaign")
    assert data["Type"] == "volume" and data["Name"] == plan["volume"] and data["RW"] is True
    bundle = next(item for item in inspection["Mounts"] if item["Destination"] == "/bundle")
    assert bundle["Type"] == "bind" and Path(bundle["Source"]).absolute() == BUNDLE and bundle["RW"] is False


def podman_json(args):
    return json.loads(subprocess.run(["podman", *args], check=True, capture_output=True).stdout)


class Containers:
    def __init__(self, plan, native_root, mountpoint):
        self.plan, self.root, self.mountpoint = plan, native_root, mountpoint

    def run(self, phase, key):
        directory = self.root / "containers" / (phase + "-" + key)
        directory.mkdir(parents=True, exist_ok=False)
        command = command_for(self.plan, phase, key)
        write(directory / "create-intent.json", {"command": command, "phase": phase, "mutation_retry_allowed": False})
        created = subprocess.run(command, check=True, capture_output=True)
        cid = created.stdout.decode().strip()
        before = podman_json(["inspect", cid])[0]
        write(directory / "inspect-before.json", before)
        validate_inspect(self.plan, before, phase)
        assert before["State"]["Status"] == "created"
        started = time.perf_counter()
        # Timeout leaves the named container intact and outcome unknown. No
        # automatic kill, second start, delete, or bridge mutation is attempted.
        with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
            try:
                completed = subprocess.run(["podman", "start", "--attach", cid], stdout=stdout, stderr=stderr,
                    timeout=2400 if phase == "search" else 600)
            except BaseException as error:
                write(directory / "unknown.json", {"container_id": cid, "phase": phase,
                    "exception_type": type(error).__name__, "mutation_retry_allowed": False})
                raise
        after = podman_json(["inspect", cid])[0]
        write(directory / "inspect-after.json", after); validate_inspect(self.plan, after, phase)
        receipt = {"container_id": cid, "phase": phase, "podman_returncode": completed.returncode,
            "container_returncode": after["State"]["ExitCode"], "image": self.plan["image_id"],
            "network": after["HostConfig"]["NetworkMode"], "elapsed_seconds": time.perf_counter()-started,
            "stdout_sha256": file_hash(directory / "stdout.log"), "stderr_sha256": file_hash(directory / "stderr.log"),
            "inspect_before_sha256": file_hash(directory / "inspect-before.json"),
            "inspect_after_sha256": file_hash(directory / "inspect-after.json"), "mutation_retry_allowed": False}
        write(directory / "receipt.json", receipt)
        assert completed.returncode == 0 and after["State"]["ExitCode"] == 0 and after["State"]["Running"] is False, \
            "container phase failed; inspect retained evidence, never repeat"
        return receipt


def run(plan):
    if os.name != "posix" or not sys.platform.startswith("linux"):
        raise RuntimeError("--run requires native WSL Linux Python")
    native = Path(plan["native_root"])
    assert native.is_absolute() and not str(native).startswith("/mnt/") and ".." not in native.parts
    native.parent.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.run(["stat", "-f", "-c", "%T", str(native.parent)], check=True, capture_output=True).stdout.decode().strip()
    assert filesystem in {"ext2/ext3", "ext4", "xfs", "btrfs"}, "durable native Linux filesystem required"
    native.mkdir(exist_ok=False)
    write(native / "launch-intent.json", {"plan": plan, "bundle_manifest_sha256": file_hash(BUNDLE / "manifest.json"),
        "journal_filesystem": filesystem, "mutation_retry_allowed": False})
    inspected_image = podman_json(["image", "inspect", plan["image_id"]])[0]
    assert inspected_image["Id"].removeprefix("sha256:") == plan["image_id"]
    write(native / "image-inspect.json", inspected_image)
    # No --ignore and no cleanup: existing volume names are refused by Podman.
    exists = subprocess.run(["podman", "volume", "exists", plan["volume"]])
    assert exists.returncode == 1, "volume exists or existence is unknown; never reuse"
    created = subprocess.run(["podman", "volume", "create", "--label", "reap.campaign=" + campaign_identity(plan), plan["volume"]],
                             check=True, capture_output=True)
    assert created.stdout.decode().strip() == plan["volume"]
    volume = podman_json(["volume", "inspect", plan["volume"]])[0]
    assert volume["Name"] == plan["volume"] and volume["Driver"] == "local"
    assert volume["Labels"]["reap.campaign"] == campaign_identity(plan)
    mountpoint = Path(volume["Mountpoint"])
    assert not str(mountpoint).startswith("/mnt/")
    write(native / "volume-inspect.json", volume)
    containers = Containers(plan, native, mountpoint)
    report = None
    try:
        # Finish both CPU preflights before any session creation; the callbacks
        # below only return these immutable descriptors, avoiding serial compile
        # work in the GPU search dispatch path.
        for case in plan["cases"]:
            containers.run("preflight", case["problem_id"])
        if "cpu_rebinding" in plan:
            records = {}
            for case in plan["cases"]:
                directory = mountpoint / "preflight" / case["problem_id"]
                raw = (directory / "receipt.json").read_bytes()
                receipt = json.loads(raw)
                assert file_hash(directory / "stdout.log") == receipt["stdout_sha256"]
                assert file_hash(directory / "stderr.log") == receipt["stderr_sha256"]
                records[case["problem_id"]] = (raw, read(directory / "descriptor.json"))
            bound = finalize(plan, (BUNDLE / "inputs/declarations.lean").read_bytes(), records)
            write(mountpoint / "runtime-plan.json", bound)
            write(mountpoint / "runtime-plan-binding.json", {"prepared_plan_sha256": file_hash(BUNDLE / "plan.json"),
                "runtime_plan_sha256": file_hash(mountpoint / "runtime-plan.json")})
            plan = bound; containers.plan = bound
        with mm.Matchmaker.create(native / "scheduler", plan["curriculum"], plan["config"]) as scheduler:
            assert scheduler.run_sha256 == plan["run_sha256"]

            def prepare_source(proposal):
                expected = next(item for item in plan["expected_attempts"] if item["attempt_id"] == proposal["attempt_id"])
                assert all(proposal[key] == value for key, value in expected.items())
                descriptor = read(mountpoint / "preflight" / proposal["problem_id"] / "descriptor.json")
                assert descriptor["attempted_prop_sha256"] == proposal["attempted_prop_sha256"] and descriptor["budget_steps"] == 8
                return descriptor

            def search(intent, endpoint):
                sid = intent["proposal"]["attempt_id"]
                write(mountpoint / "intents" / (sid + ".json"), intent)
                write(mountpoint / "routes" / (sid + ".json"), {"endpoint": endpoint, "intent_sha256": intent["intent_sha256"]})
                containers.run("search", sid)
                envelope = read(mountpoint / "terminal-audit" / sid / "terminal-envelope.json")
                files = read(mountpoint / "terminal-audit" / sid / "files.json")
                assert file_hash(mountpoint / "terminal-audit" / sid / "files.json") == envelope["artifacts_sha256"]
                assert all(file_hash(mountpoint / "sessions" / sid / name) == value for name, value in files.items())
                return envelope

            def verify(intent, envelope):
                sid = intent["proposal"]["attempt_id"]
                containers.run("verify", sid)
                result = read(mountpoint / "verify-results" / (sid + ".json"))
                assert result["verification"]["verified"] is True
                accepted = result["matchmaker_result"]
                assert file_hash(mountpoint / "verified" / sid / "verification-receipt.json") == accepted["evidence"]["verification_receipt_sha256"]
                assert file_hash(mountpoint / "verified" / sid / "dataset-receipt.json") == accepted["evidence"]["dataset_receipt_sha256"]
                return accepted

            with ReplicaCollector(native / "router", plan["endpoints"], search) as router:
                report = run_replica_collector_batch(scheduler, native / "batch", router=router, prepare_source=prepare_source, verify=verify)
            write(native / "batch-report.json", report)
            expected_verdicts = {p["family_id"]: {"true-positive": "proved", "false-negative": "disproved"}[p["problem_id"]] for p in plan["curriculum"]["problems"]}
            families = report["scheduler"]["terminal_families"]
            verdicts = {key: value["verdict"] for key, value in families.items()}
            actual_routes = [read(mountpoint / "routes" / (item["attempt_id"] + ".json")) for item in plan["expected_attempts"]]
            used = {item["endpoint"]["replica_id"] for item in actual_routes}
            summary = {"schema_version": "reap.replica-collector.campaign-result.v1", "execution_completed": report["status"] == "completed",
                "both_independently_verified_branches": verdicts == expected_verdicts and len(report["accepted"]) == 2,
                "batch_report_sha256": file_hash(native / "batch-report.json"), "expected_attempts": plan["expected_attempts"],
                "seed_selection": plan["seed_selection"], "model_release_sha256": plan["model_release_sha256"],
                "image_id": plan["image_id"], "new_training_steps": 0, "performance_gain_claimed": False,
                "remote_tensor_and_wire_audit_performed_here": False, "mutation_retry_allowed": False}
            summary["both_replica_endpoints_used"] = used == {e["replica_id"] for e in plan["endpoints"]}
            summary["actual_routes"] = actual_routes
            summary["actual_MCTS_overlap_measured_here"] = False
            summary["ok"] = summary["execution_completed"] and summary["both_independently_verified_branches"] and summary["both_replica_endpoints_used"]
            write(native / "campaign-report.json", summary)
            print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
            return 0 if summary["ok"] else 1
    finally:
        # All originals remain in native storage. This is a read-only evidence
        # export, never a restore/retry or deletion of the volume/containers.
        destination = BUNDLE / "collected"
        destination.mkdir(exist_ok=False)
        shutil.copytree(native, destination / "host")
        shutil.copytree(mountpoint, destination / "volume")
        files = {p.relative_to(destination).as_posix(): file_hash(p) for p in sorted(destination.rglob("*")) if p.is_file()}
        write(destination / "export-manifest.json", {"files_sha256": files, "native_originals_retained": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate frozen inputs/IDs only; never call bridge")
    mode.add_argument("--run", action="store_true", help="execute once after the parent has configured the exact allowlist")
    args = parser.parse_args()
    plan = check_bundle()
    if not args.run:
        print(json.dumps({"prepared": True, "GPU_or_bridge_called": False, "run_sha256": plan["run_sha256"],
            "attempts": plan["expected_attempts"], "bundle_manifest_sha256": file_hash(BUNDLE / "manifest.json")}, indent=2))
        return 0
    return run(plan)


if __name__ == "__main__":
    raise SystemExit(main())
