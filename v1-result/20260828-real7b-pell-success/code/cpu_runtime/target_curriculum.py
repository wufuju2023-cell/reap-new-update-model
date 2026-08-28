"""A fixed target and mechanically derived, independently checked variants.

This is separate from the permissive Matchmaker manifest and v3 purpose scope.
Compilation checks statements; relation checks establish T -> V; neither proves
T or V. Search sources never include the relation certificate. The operator must
pin the actual Lean environment and the returned curriculum digest. Hashes bind
trusted local receipts; they are not signatures for arbitrary remote evidence.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess

from . import closed_problem as closed
from . import matchmaker as mm
from .verified_dataset_store import safe_directory
from .verified_trajectory import checked_axioms


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _fields(value, names, label):
    _require(type(value) is dict and set(value) == set(names.split()), "invalid "+label+" fields")


def _id(value):
    _require(type(value) is str and mm.ID.fullmatch(value), "invalid curriculum member ID")


def _variants():
    # Kept lazy so import/CLI inspection does not run Lean or import Torch.
    from . import target_variants
    return target_variants


def prepare_curriculum(*, environment_sha256, declarations, target_name, variants,
                       family_id="target-family", target_id="target"):
    """Prepare one target and 1..3 direct variants; no compilation or proof."""
    _id(family_id); _id(target_id)
    _require(type(variants) is list and 1 <= len(variants) <= 3, "one to three explicit variants required")
    target = closed.identity(environment_sha256=environment_sha256,
        declarations=declarations, closed_prop_name=target_name)
    generator = _variants()
    entries, ids, names, transforms = [], {target_id}, {target_name}, set()
    for spec in variants:
        _fields(spec, "problem_id variant_name transform", "variant specification")
        _id(spec["problem_id"])
        _require(spec["problem_id"] not in ids and spec["variant_name"] not in names,
                 "duplicate member ID or declaration name")
        transform_pin = closed.digest(spec["transform"])
        _require(transform_pin not in transforms, "duplicate transformation specification")
        made = generator.generate_variant(environment_sha256=environment_sha256,
            declarations=declarations, target_name=target_name,
            variant_name=spec["variant_name"], transform=spec["transform"])
        entries.append({**deepcopy(spec), "problem_sha256": made["variant_identity"]["problem_sha256"],
            "transformation_sha256": made["transformation_sha256"]})
        ids.add(spec["problem_id"]); names.add(spec["variant_name"]); transforms.add(transform_pin)
    return {"schema_version": "reap.target-curriculum.plan.v1",
        "environment_sha256": environment_sha256, "wrapper_version": closed.WRAPPER_VERSION,
        "generator_version": generator.GENERATOR_VERSION,
        "generator_sha256": closed.digest(Path(generator.__file__).read_bytes()),
        "family_id": family_id, "target": {"problem_id": target_id,
            "closed_prop_name": target_name, "declarations_utf8": declarations.decode("utf8"),
            "problem_sha256": target["problem_sha256"]}, "variants": entries}


def _members(plan):
    _fields(plan, "schema_version environment_sha256 wrapper_version generator_version generator_sha256 family_id target variants", "plan")
    _fields(plan["target"], "problem_id closed_prop_name declarations_utf8 problem_sha256", "target")
    _require(type(plan["target"]["declarations_utf8"]) is str and type(plan["variants"]) is list,
             "invalid declarations or variants")
    specs = []
    for entry in plan["variants"]:
        _fields(entry, "problem_id variant_name transform problem_sha256 transformation_sha256", "variant")
        specs.append({key: entry[key] for key in ("problem_id", "variant_name", "transform")})
    target = plan["target"]
    declarations = target["declarations_utf8"].encode("utf8")
    expected = prepare_curriculum(environment_sha256=plan["environment_sha256"], declarations=declarations,
        target_name=target["closed_prop_name"], variants=specs, family_id=plan["family_id"], target_id=target["problem_id"])
    _require(closed.canonical(plan) == closed.canonical(expected), "curriculum plan/source/generator differs")
    members = [{"problem_id": target["problem_id"], "identity": closed.identity(
        environment_sha256=plan["environment_sha256"], declarations=declarations,
        closed_prop_name=target["closed_prop_name"]), "declarations": declarations,
        "is_target": True, "transformation_sha256": None, "relation_source": None, "relation_theorem": None}]
    for entry in plan["variants"]:
        made = _variants().generate_variant(environment_sha256=plan["environment_sha256"],
            declarations=declarations, target_name=target["closed_prop_name"],
            variant_name=entry["variant_name"], transform=entry["transform"])
        members.append({"problem_id": entry["problem_id"], "identity": made["variant_identity"],
            "declarations": made["declarations"], "is_target": False,
            "transformation_sha256": made["transformation_sha256"],
            "relation_source": made["relation_source"], "relation_theorem": made["relation_theorem"]})
    return members


def _prepared(plan, member, *, polarity="prove", budget_steps=1, num_samples=1, max_tokens=1):
    return closed.prepare(environment_sha256=plan["environment_sha256"], declarations=member["declarations"],
        closed_prop_name=member["identity"]["closed_prop_name"], polarity=polarity,
        budget_steps=budget_steps, num_samples=num_samples, max_tokens=max_tokens)


def _check(receipt, source, label, theorem=None):
    _fields(receipt, "schema_version source_sha256 intent_sha256 command returncode stdout stderr", "compile receipt")
    mm._sha(receipt["intent_sha256"], "compile intent")
    _require(receipt["schema_version"] == "reap.target-curriculum.compile.v1"
        and receipt["source_sha256"] == closed.digest(source)
        and type(receipt["returncode"]) is int and receipt["returncode"] == 0,
        label+" did not compile successfully")
    command = receipt["command"]
    _require(type(command) is list and len(command) == 4 and command[:3] == ["lake", "env", "lean"]
        and type(command[3]) is str and Path(command[3]).name == label+".lean",
        "compile command does not match its source")
    _require(type(receipt["stdout"]) is str and type(receipt["stderr"]) is str,
        "compile streams required")
    if theorem is not None:
        checked_axioms(receipt["stdout"], theorem)


def _curriculum(plan, members, checks):
    _require(type(checks) is dict and set(checks) == {m["problem_id"] for m in members}, "incomplete member checks")
    problems = []
    for member in members:
        receipt = checks[member["problem_id"]]
        _fields(receipt, "preflight relation", "member checks")
        _check(receipt["preflight"], _prepared(plan, member)["preflight_source"], "preflight")
        if member["is_target"]:
            _require(receipt["relation"] is None, "target must not claim a derived relation")
        else:
            _check(receipt["relation"], member["relation_source"], "relation", member["relation_theorem"])
        identity = member["identity"]
        problems.append({"problem_id": member["problem_id"], "family_id": plan["family_id"],
            "is_target": member["is_target"], "problem_sha256": identity["problem_sha256"],
            "declarations_sha256": identity["declarations_sha256"], "closed_prop_name": identity["closed_prop_name"],
            "compilation_receipt_sha256": closed.digest(receipt),
            "parent_problem_sha256": None if member["is_target"] else plan["target"]["problem_sha256"],
            "transformation_sha256": member["transformation_sha256"]})
    value = {"schema_version": "reap.matchmaker.curriculum.v1", "environment_sha256": plan["environment_sha256"],
        "wrapper_version": closed.WRAPPER_VERSION, "problems": problems}
    mm.validate_curriculum(value)
    return value


def validate_manifest(manifest):
    _fields(manifest, "schema_version plan checks matchmaker_curriculum", "curriculum manifest")
    _require(manifest["schema_version"] == "reap.target-curriculum.verified.v1", "unsupported curriculum manifest")
    members = _members(manifest["plan"])
    expected = _curriculum(manifest["plan"], members, manifest["checks"])
    _require(closed.canonical(expected) == closed.canonical(manifest["matchmaker_curriculum"]),
        "scheduler curriculum differs from verified target derivations")
    return members


def _check_frozen(folder, manifest, members):
    _require(folder.read("plan.json") == closed.canonical(manifest["plan"]), "frozen plan differs")
    for member in members:
        with folder.child(member["problem_id"]) as evidence:
            for label, receipt in manifest["checks"][member["problem_id"]].items():
                if receipt is None:
                    continue
                source = (_prepared(manifest["plan"], member)["preflight_source"]
                    if label == "preflight" else member["relation_source"])
                raw = evidence.read(label+"-intent.json")
                intent = json.loads(raw)
                _fields(intent, "command source_sha256 timeout_seconds automatic_retry_allowed", "compile intent")
                timeout = intent["timeout_seconds"]
                _require(raw == closed.canonical(intent) and closed.digest(raw) == receipt["intent_sha256"]
                    and intent["command"] == receipt["command"]
                    and intent["source_sha256"] == receipt["source_sha256"]
                    and intent["automatic_retry_allowed"] is False
                    and type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0,
                    "frozen compile intent differs")
                _require(evidence.read(label+".lean") == source
                    and evidence.read(label+"-receipt.json") == closed.canonical(receipt)
                    and evidence.read(label+".stdout") == receipt["stdout"].encode("utf8")
                    and evidence.read(label+".stderr") == receipt["stderr"].encode("utf8"),
                    "frozen curriculum compile evidence differs")


def verify_curriculum(plan, *, project_dir, output_dir, timeout=180):
    """Compile statements and relations once in the caller-pinned environment.

    Output is exclusive and all failures are retained. No search, training or
    proof of the target is performed. A timeout never triggers another command.
    """
    members = _members(plan)
    _require(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0, "invalid compile timeout")
    output_dir, project_dir = Path(output_dir).absolute(), Path(project_dir).absolute()
    with safe_directory(project_dir), safe_directory(output_dir.parent) as parent:
        with parent.child(output_dir.name, create=True) as out:
            out.write_new("plan.json", closed.canonical(plan)); out.sync(); parent.sync()
            checks = {}
            try:
                for member in members:
                    with out.child(member["problem_id"], create=True) as folder:
                        receipt = {"preflight": None, "relation": None}
                        sources = {"preflight": _prepared(plan, member)["preflight_source"]}
                        if not member["is_target"]:
                            sources["relation"] = member["relation_source"]
                        for label, source in sources.items():
                            folder.write_new(label+".lean", source)
                            command = ["lake", "env", "lean", str(folder.path/(label+".lean"))]
                            folder.write_new(label+"-intent.json", closed.canonical({"command": command,
                                "source_sha256": closed.digest(source), "timeout_seconds": timeout,
                                "automatic_retry_allowed": False}))
                            folder.sync()
                            try:
                                process = subprocess.run(command, cwd=project_dir, capture_output=True, timeout=timeout)
                            except subprocess.TimeoutExpired as error:
                                folder.write_new(label+".stdout", error.stdout or b"")
                                folder.write_new(label+".stderr", error.stderr or b"")
                                raise
                            folder.write_new(label+".stdout", process.stdout)
                            folder.write_new(label+".stderr", process.stderr)
                            value = {"schema_version": "reap.target-curriculum.compile.v1",
                                "source_sha256": closed.digest(source), "command": command,
                                "intent_sha256": closed.digest(folder.read(label+"-intent.json")),
                                "returncode": process.returncode, "stdout": process.stdout.decode("utf8"),
                                "stderr": process.stderr.decode("utf8")}
                            folder.write_new(label+"-receipt.json", closed.canonical(value)); folder.sync()
                            _require(folder.read(label+".lean") == source, "compile source changed")
                            _check(value, source, label, member["relation_theorem"] if label == "relation" else None)
                            receipt[label] = value
                        checks[member["problem_id"]] = receipt
                manifest = {"schema_version": "reap.target-curriculum.verified.v1", "plan": deepcopy(plan),
                    "checks": checks, "matchmaker_curriculum": _curriculum(plan, members, checks)}
                validate_manifest(manifest)
                _check_frozen(out, manifest, members)
                # Atomic no-replace publication. Failure after link/fsync remains
                # unknown; the caller also requires this process to exit zero.
                mm._publish(out, "curriculum.json", manifest)
                _require(load_curriculum(output_dir, expected_sha256=closed.digest(manifest)) == manifest,
                    "published curriculum evidence differs")
                return {"curriculum_sha256": closed.digest(manifest), "path": str(output_dir),
                    "target_proved": False, "variant_proved": False, "manifest": manifest}
            except BaseException as error:
                out.write_new("failed.json", closed.canonical({"error_type": type(error).__name__,
                    "automatic_retry_allowed": False, "verified": False})); out.sync()
                raise


def load_curriculum(directory, *, expected_sha256):
    mm._sha(expected_sha256, "curriculum")
    with safe_directory(Path(directory)) as folder:
        # An error after publishing is not converted into successful admission.
        try:
            folder.read("failed.json")
        except FileNotFoundError:
            pass
        else:
            raise ValueError("curriculum verification failed or publication is unknown")
        raw = folder.read("curriculum.json")
        _require(closed.digest(raw) == expected_sha256, "curriculum digest differs")
        manifest = json.loads(raw)
        _require(raw == closed.canonical(manifest), "curriculum must be canonical JSON")
        members = validate_manifest(manifest)
        _check_frozen(folder, manifest, members)
        return manifest


def prepare_member(manifest, problem_id, *, polarity, budget_steps, num_samples, max_tokens):
    members = validate_manifest(manifest)
    matching = [m for m in members if m["problem_id"] == problem_id]
    _require(len(matching) == 1, "problem is not a member of this curriculum")
    return _prepared(manifest["plan"], matching[0], polarity=polarity, budget_steps=budget_steps,
        num_samples=num_samples, max_tokens=max_tokens)
