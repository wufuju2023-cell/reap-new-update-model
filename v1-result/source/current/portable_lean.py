"""Prepare/check/run a bounded two-service Lean campaign, without model downloads.

Preparation copies explicit source bytes into a new bundle. Check is offline;
run requires Linux Podman and two already configured fixed-release endpoints.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import uuid

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("_portable_verify", HERE / "verify.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)
TEMPLATES = HERE / "lean_campaign"
SHA = re.compile(r"[0-9a-f]{64}\Z")
ID = re.compile(r"[a-z0-9][a-z0-9-]{0,31}\Z")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def put(path, raw):
    VERIFY.no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def template_bytes():
    manifest = json.loads((TEMPLATES / "template-manifest.json").read_bytes())
    values = {}
    for name, pin in manifest["files_sha256"].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in name, "unsafe template path")
        p = VERIFY.no_links(TEMPLATES / name)
        raw = p.read_bytes()
        require(digest(raw) == pin, "template SHA differs: " + name)
        values[name] = raw
    return values


def prepare(*, source_root, output, image_id, endpoints, deployments, release, project_dir, run_id=None, rebind_cpu_image=False):
    output = VERIFY.no_links(Path(output).absolute())
    require(not output.exists(), "output exists; inspect it, never repeat")
    source = VERIFY.no_links(Path(source_root).absolute()).resolve(strict=True)
    require(SHA.fullmatch(image_id or "") and SHA.fullmatch(release or ""), "explicit image ID and release SHA required")
    run_id = run_id or uuid.uuid4().hex
    require(ID.fullmatch(run_id), "run-id must be 1..32 lower-case letters/digits/hyphens")
    project = PurePosixPath(project_dir)
    require(project.is_absolute() and ".." not in project.parts and project.as_posix() == project_dir
            and not any(ord(c) < 32 for c in project_dir), "absolute container Lean project path required")
    require(len(endpoints) == len(deployments) == 2, "two endpoints and deployment IDs required")
    values = template_bytes()
    environment = json.loads(values["inputs/environment.json"])
    require(rebind_cpu_image or image_id == environment["cpu_image_id"],
            "different CPU image requires explicit --rebind-cpu-image and new runtime preflights")
    copied = {}
    for package in ("cpu_runtime", "gpu_runtime"):
        paths = sorted((source / package).glob("*.py"))
        require(paths, "missing source package: " + package)
        for path in paths:
            VERIFY.no_links(path)
            copied[path.relative_to(source).as_posix()] = path.read_bytes()
    replay = "containers/cpu/verified-replay/VerifiedReplay.lean"
    copied[replay] = VERIFY.no_links(source / replay).read_bytes()
    # Import only the explicitly selected stdlib control modules in this process.
    for name in ("cpu_runtime", "gpu_runtime"):
        existing = sys.modules.get(name)
        if existing is not None:
            require(Path(existing.__file__).resolve().is_relative_to(source), "a different source package is already imported")
    sys.path.insert(0, str(source))
    from cpu_runtime import matchmaker as mm
    from cpu_runtime.replica_collector import validate_endpoints
    plan = json.loads(values["plan-template.json"])
    plan.update(schema_version="reap.portable-lean.plan.v1", image_id=image_id,
                model_release_sha256=release, project_dir=project_dir, run_id=run_id,
                native_root=str(output / "native"), volume="reap-portable-" + run_id)
    plan["endpoints"] = validate_endpoints([
        {"replica_id": "replica-" + str(i), "base_url": endpoint,
         "deployment_id": deployments[i], "model_release_sha256": release}
        for i, endpoint in enumerate(endpoints)])
    for index, problem in enumerate(plan["curriculum"]["problems"]):
        # Matchmaker IDs have a 40-character bound; retain the full run nonce.
        problem["family_id"] = "p-" + run_id + "-" + str(index)
    if rebind_cpu_image:
        binding_spec = importlib.util.spec_from_file_location("_portable_binding", TEMPLATES / "binding.py")
        binding = importlib.util.module_from_spec(binding_spec); binding_spec.loader.exec_module(binding)
        new_environment = {**environment, "cpu_image_id": image_id, "project_dir": project_dir,
            "source_guard_sha256": {name: digest(copied[name]) for name in (
                "cpu_runtime/closed_problem.py", "cpu_runtime/verified_collector.py")}}
        plan, new_inputs = binding.make_pending(plan, values["inputs/declarations.lean"], new_environment)
        values = {("provenance/original-" + name if name.startswith("inputs/") or name == "plan-template.json" else name): raw
                  for name, raw in values.items()}
        values.update(new_inputs)
    else:
        with tempfile.TemporaryDirectory(prefix="portable-lean-plan-") as temporary:
            with mm.Matchmaker.create(Path(temporary) / "scheduler", plan["curriculum"], plan["config"]) as scheduler:
                plan["run_sha256"] = scheduler.run_sha256
                plan["expected_attempts"] = []
                for _ in range(2):
                    proposal = scheduler.plan_next(release)
                    require(proposal is not None, "expected two bounded attempts")
                    case = next(x for x in plan["cases"] if x["problem_id"] == proposal["problem_id"])
                    require(proposal["polarity"] == case["desired_polarity"], "branch selection differs")
                    plan["expected_attempts"].append({k: proposal[k] for k in (
                        "attempt_id", "attempt_index", "attempted_prop_sha256", "budget_steps", "family_id",
                        "model_release_sha256", "polarity", "problem_id", "problem_sha256")})
                    scheduler.reserve(proposal, case["prior_descriptor"])
    require(all((source / name).read_bytes() == raw for name, raw in copied.items()), "source changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in values.items():
        if name != "plan-template.json":
            put(output / name, raw)
    for name, raw in copied.items():
        target = "VerifiedReplay.lean" if name == replay else name
        put(output / "frozen" / target, raw)
    put(output / "plan.json", canonical(plan))
    put(output / "preparation.json", canonical({"source_sha256": {name: digest(raw) for name, raw in copied.items()},
        "source_root_is_not_an_experiment": True, "new_GPU_or_Lean_run": False,
        "run_id": run_id, "automatic_retry_allowed": False}))
    manifest = {"schema_version": "reap.portable-lean.bundle.v1", "files_sha256": {
        p.relative_to(output).as_posix(): digest(p.read_bytes()) for p in sorted(output.rglob("*")) if p.is_file()}}
    put(output / "manifest.json", canonical(manifest))
    return {"prepared": True, "output": str(output), "run_sha256": plan["run_sha256"],
            "attempts": plan["expected_attempts"], "new_GPU_or_Lean_run": False}


def validate_run_platform():
    require(sys.platform.startswith("linux") and os.name == "posix", "run requires native Linux/WSL Python and Podman")


def execute(output, run=False):
    output = VERIFY.no_links(Path(output).absolute()).resolve(strict=True)
    manifest = json.loads((output / "manifest.json").read_bytes())
    for name, pin in manifest["files_sha256"].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in name, "unsafe manifest path")
        require(digest(VERIFY.no_links(output / name).read_bytes()) == pin, "prepared file changed: " + name)
    intent, exit_path = output / "run-intent.json", output / "run-exit.json"
    if run:
        validate_run_platform()
        require(not any(p.exists() for p in (intent, exit_path, output / "native", output / "collected")),
                "existing run evidence; inspect original outcome without retry")
        put(intent, canonical({"command": [sys.executable, "-B", str(output / "campaign.py"), "--run"],
                               "automatic_retry_allowed": False}))
    command = [sys.executable, "-B", str(output / "campaign.py"), "--run" if run else "--check"]
    if not run:
        return subprocess.call(command)
    # A lost subprocess result leaves run-intent intact; no second launch.
    with (output / "run-stdout.log").open("xb") as stdout, (output / "run-stderr.log").open("xb") as stderr:
        code = subprocess.call(command, stdout=stdout, stderr=stderr)
    put(exit_path, canonical({"returncode": code, "automatic_retry_allowed": False}))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--image-id")
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--deployment-id", action="append", default=[])
    parser.add_argument("--release-sha256")
    parser.add_argument("--project-dir", default="/opt/reap-runtime")
    parser.add_argument("--run-id")
    parser.add_argument("--rebind-cpu-image", action="store_true", help="derive new environment; require actual CPU preflights before scheduler creation")
    args = parser.parse_args(argv)
    if args.prepare:
        if args.source_root is None:
            parser.error("--source-root required for preparation")
        result = prepare(source_root=args.source_root, output=args.output, image_id=args.image_id,
            endpoints=args.endpoint, deployments=args.deployment_id, release=args.release_sha256,
            project_dir=args.project_dir, run_id=args.run_id, rebind_cpu_image=args.rebind_cpu_image)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    return execute(args.output, args.run)


if __name__ == "__main__":
    raise SystemExit(main())
