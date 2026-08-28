#!/usr/bin/env python3
"""Explicit reproduction steps; no training, implicit retries, or base-model load.

Use the frozen source through PYTHONPATH. Lean receipts are trusted local
operator records, checked against their files. Tensor checking reads trusted
local snapshots on CPU and never changes the GPU stores.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import uuid
from extract_online_proof import SCHEMA, prepare_proof


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def same(left, right, message):
    require(canonical(left) == canonical(right), message)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=unique)
    canonical(value)
    return value


def write(path, value):
    with Path(path).open("xb") as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", value),
            "invalid session/experience ID")
    return value


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value


def checked_proof(run, session_id, cpu_image):
    directory = run / "proof-check" / identifier(session_id)
    accepted, receipt = read(directory / "accepted.json"), read(directory / "receipt.json")
    require(accepted.get("verification_receipt_sha256") == sha((directory / "receipt.json").read_bytes()),
            "proof receipt hash mismatch")
    same({k: v for k, v in accepted.items() if k != "verification_receipt_sha256"}, receipt,
         "accepted record differs from independent receipt")
    require(receipt.get("schema_version") == SCHEMA and receipt.get("passed") is True
            and receipt.get("session_id") == session_id, "missing independent proof acceptance")
    require(type(receipt.get("lean_returncode")) is int and receipt["lean_returncode"] == 0
            and type(receipt.get("container_exit_code")) is int and receipt["container_exit_code"] == 0
            and receipt.get("container_running") is False and receipt.get("network") == "none",
            "independent Lean did not finish successfully offline")
    require(receipt.get("image", "").removeprefix("sha256:") == cpu_image.removeprefix("sha256:"),
            "proof uses a different CPU image")
    require(isinstance(receipt.get("axioms"), list)
            and set(receipt["axioms"]) <= {"propext", "Classical.choice", "Quot.sound"}, "unexpected proof axioms")
    files = receipt.get("files", {})
    require({"proof.lean", "stdout.log", "stderr.log", "inspect.stdout.json", "launch-intent.json"}
            <= set(files), "incomplete proof evidence")
    for name, item in files.items():
        require(name == Path(name).name and "/" not in name and "\\" not in name, "unsafe evidence name")
        raw = (directory / name).read_bytes()
        require(type(item.get("size")) is int and item["size"] == len(raw)
                and item.get("sha256") == sha(raw), "proof evidence changed: " + name)
    binding = receipt["input_files"]
    proof, online, actual = prepare_proof(run, session_id, Path(binding["source"]["path"]).name, receipt["theorem"])
    same(actual, binding, "searched input/result changed since proof verification")
    require(proof == (directory / "proof.lean").read_bytes() and receipt.get("proof_sha256") == sha(proof)
            and receipt.get("theorem_sha256") == actual["source"]["sha256"], "proof/search binding differs")
    return accepted, online


def client(url):
    from cpu_runtime.http_clients import GpuHttpClient
    from cpu_runtime.transport_budget import DEFAULT_BUDGET
    return GpuHttpClient(url, timeout_seconds=DEFAULT_BUDGET.client)


def publish(run_root, session_id, gpu_url, cpu_image, *, gpu=None):
    run = Path(run_root).resolve()
    accepted, online = checked_proof(run, session_id, cpu_image)
    updates = online.get("optimizer_updates")
    require(type(updates) is int and updates >= 1 and online.get("status") == "passed_execution"
            and online.get("online_update_consumed_by_later_generation") is True,
            "source needs a real update consumed by later generation")
    candidate = read(run / "outputs" / session_id / "experience-candidate.json")
    source = candidate.get("source", {})
    require(candidate.get("session_id") == session_id and candidate.get("snapshot") == "experience-candidate",
            "wrong experience candidate")
    require(set(source) == {"session_id", "theorem_id", "policy_version", "snapshot",
                            "snapshot_sha256", "parent_experience_id"}, "incomplete candidate source")
    require(source["session_id"] == session_id and source["theorem_id"] == accepted["theorem_sha256"]
            and type(source["policy_version"]) is int and source["policy_version"] == updates
            and online.get("policy_version") == updates and source["snapshot"] == "experience-candidate",
            "candidate differs from independently verified source")
    digest(source["snapshot_sha256"])
    output = run / "cross"
    output.mkdir(exist_ok=False)
    experience_id, target_id = "experience-" + uuid.uuid4().hex[:16], "cross-" + uuid.uuid4().hex[:16]
    acceptance = {"kind": "independent-lean", "passed": True, "completed": True, "source": source,
                  "evidence_sha256": sha((run / "proof-check" / session_id / "accepted.json").read_bytes())}
    intent = {"source_session_id": session_id, "target_session_id": target_id, "experience_id": experience_id,
              "snapshot": source["snapshot"], "acceptance": acceptance, "gpu_url": gpu_url,
              "cpu_image": cpu_image, "mutation_retry_allowed": False}
    write(output / "publish-intent.json", intent)
    response = (gpu or client(gpu_url)).publish_experience(session_id, source["snapshot"], experience_id, acceptance)
    write(output / "publish-response.json", response)
    require(response.get("schema_version") == "reap.gpu.experience.v1"
            and response.get("experience_id") == experience_id, "wrong publish response")
    same(response.get("source"), source, "published source mismatch")
    same(response.get("acceptance"), acceptance, "published acceptance mismatch")
    same(response.get("transfer"), ["adapter", "value_head"], "unexpected transfer scope")
    pins = {"experience_id": experience_id, "experience_weights_sha256": digest(response.get("weights_sha256")),
            "experience_snapshot_sha256": source["snapshot_sha256"]}
    write(output / "pins.json", pins)
    settings = {"SOURCE_SID": session_id, "TARGET_SID": target_id, "EXPERIENCE_ID": experience_id,
                "EXPERIENCE_WEIGHTS_SHA256": pins["experience_weights_sha256"],
                "EXPERIENCE_SNAPSHOT_SHA256": pins["experience_snapshot_sha256"]}
    with (output / "settings.sh").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("".join("export " + key + "=" + shlex.quote(value) + "\n" for key, value in settings.items()))
        handle.flush()
        os.fsync(handle.fileno())
    return pins


def settings(cross):
    intent, response, pins = (read(cross / name) for name in
                               ("publish-intent.json", "publish-response.json", "pins.json"))
    same(response.get("source"), intent["acceptance"]["source"], "source receipt mismatch")
    same(response.get("acceptance"), intent["acceptance"], "acceptance receipt mismatch")
    same(pins, {"experience_id": intent["experience_id"], "experience_weights_sha256": response["weights_sha256"],
                "experience_snapshot_sha256": response["source"]["snapshot_sha256"]}, "pin receipt mismatch")
    for key in ("source_session_id", "target_session_id", "experience_id"):
        identifier(intent[key])
    return intent, pins


def source_snapshot(run_root, gpu_url, phase, *, gpu=None):
    run = Path(run_root).resolve()
    intent, _ = settings(run / "cross")
    require(intent["gpu_url"] == gpu_url, "use the same running GPU service")
    require(phase in ("before", "after"), "unknown snapshot phase")
    name = "cross-source-" + phase
    request = {"session_id": intent["source_session_id"], "name": name, "mutation_retry_allowed": False}
    target = intent["target_session_id"]
    if phase == "before":
        require(not (run / "outputs" / target).exists(), "take before snapshot before starting the target")
    else:
        accepted, _ = checked_proof(run, target, intent["cpu_image"])
        request.update(target_session_id=target, target_theorem_sha256=accepted["theorem_sha256"],
            target_accepted_sha256=sha((run / "proof-check" / target / "accepted.json").read_bytes()),
            target_online_result_sha256=sha((run / "outputs" / target / "online-result.json").read_bytes()))
    write(run / "cross" / (name + "-intent.json"), request)
    response = (gpu or client(gpu_url)).snapshot(request["session_id"], name)
    write(run / "cross" / (name + "-response.json"), response)
    require(response.get("session_id") == request["session_id"] and response.get("snapshot") == name,
            "wrong source snapshot ACK")
    return response


def nested_equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(nested_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def decode(envelope):
    import torch
    return torch.load(io.BytesIO(base64.b64decode(envelope["payload"], validate=True)),
                      map_location="cpu", weights_only=True)


def check_tensors(gpu_run_root, cross_dir, output):
    from gpu_runtime.snapshot_store import SnapshotStore
    from gpu_runtime.experience_store import ExperienceStore
    root, cross = Path(gpu_run_root).resolve(), Path(cross_dir).resolve()
    require(not Path(output).exists(), "audit output already exists")
    intent, pins = settings(cross)
    source_id, target_id = intent["source_session_id"], intent["target_session_id"]
    guards = {}
    for phase in ("before", "after"):
        name = "cross-source-" + phase
        request, response = read(cross / (name + "-intent.json")), read(cross / (name + "-response.json"))
        require(request.get("session_id") == source_id and request.get("name") == name
                and request.get("mutation_retry_allowed") is False
                and response.get("session_id") == source_id and response.get("snapshot") == name,
                "source guard request/ACK missing or mismatched")
        guards[phase] = request
    require(source_id != target_id, "cross-problem sessions must differ")
    require((root / "snapshots").is_dir() and (root / "experiences").is_dir(), "missing actual GPU stores")
    snapshots, experiences = SnapshotStore(root / "snapshots"), ExperienceStore(root / "experiences")
    candidate, source_backend = snapshots.load(source_id, "experience-candidate")
    release, weights = experiences.load(pins["experience_id"])
    initial, target_backend = snapshots.load(target_id, "before-online-ttt")
    before, before_backend = snapshots.load(source_id, "cross-source-before")
    after, after_backend = snapshots.load(source_id, "cross-source-after")
    source_manifest = root / "snapshots" / source_id / "experience-candidate" / "manifest.json"
    require(sha(source_manifest.read_bytes()) == pins["experience_snapshot_sha256"], "source snapshot pin mismatch")
    require(release["weights_sha256"] == pins["experience_weights_sha256"], "release weights pin mismatch")
    same(release["source"], intent["acceptance"]["source"], "release source changed")
    same(release["acceptance"], intent["acceptance"], "release acceptance changed")
    same(weights["contract"], source_backend["experience_contract"], "source/release contract differs")
    require(weights["contract"].get("backend") == "real-search"
            and weights["contract"].get("objective") == "search_visit_backup", "wrong reproduction profile")
    for backend_id, backend in ((source_id, source_backend), (target_id, target_backend),
                                (source_id, before_backend), (source_id, after_backend)):
        require(backend.get("schema_version") == "reap.gpu.real-search-backend.v1"
                and backend.get("session_id") == backend_id, "snapshot backend identity mismatch")
        same(backend.get("search_config"), weights["contract"]["search_config"], "snapshot search contract differs")
    require(candidate["session_id"] == source_id and candidate["completed"] is True
            and candidate["theorem_id"] == release["source"]["theorem_id"]
            and candidate["policy_version"] == release["source"]["policy_version"], "wrong source candidate state")
    require(initial["session_id"] == target_id and initial.get("role", "theorem") == "theorem"
            and initial["theorem_id"] != candidate["theorem_id"], "target/source identities must differ")
    digest(initial["theorem_id"])
    require(guards["after"].get("target_session_id") == target_id
            and guards["after"].get("target_theorem_sha256") == initial["theorem_id"],
            "after snapshot does not bind the completed target")
    digest(guards["after"].get("target_accepted_sha256"))
    digest(guards["after"].get("target_online_result_sha256"))
    same(initial["lineage"], {"experience_id": pins["experience_id"],
         "weights_sha256": pins["experience_weights_sha256"], "source": release["source"],
         "reset": ["optimizer", "rng", "buffer", "policy_version", "event_receipts"]},
         "target did not receive the pinned lineage/reset scope")
    require(type(initial["policy_version"]) is int and initial["policy_version"] == 0
            and initial["completed"] is False and initial["event_receipts"] == {}, "target local state is not fresh")
    same(initial["buffer_metadata"], {"events": {}, "pending_event_ids": [], "consumed_event_ids": []},
         "target inherited a source buffer")
    source_data, released_data, target_data = decode(source_backend), decode(weights), decode(target_backend)
    counts = {}
    for key in ("adapter", "value_head"):
        require(bool(source_data[key]) and nested_equal(source_data[key], released_data[key])
                and nested_equal(released_data[key], target_data[key]), "initial parameter copy differs: " + key)
        counts[key] = len(source_data[key])
    require(type(target_data["optimizer_steps"]) is int and target_data["optimizer_steps"] == 0
            and target_data["examples_seen"] == 0 and target_data["optimizer"]["state"] == {},
            "target optimizer inherited history")
    expected_seed = int.from_bytes(hashlib.sha256(target_id.encode()).digest()[:8], "big") % (2 ** 63)
    require(target_data["rng"]["seed"] == expected_seed, "target RNG did not use its own ID")
    same(before, after, "source logical state changed during target search")
    same({k: v for k, v in before_backend.items() if k != "payload"},
         {k: v for k, v in after_backend.items() if k != "payload"}, "source backend metadata changed")
    require(nested_equal(decode(before_backend), decode(after_backend)), "source full private state changed")
    report = {"schema_version": "reap.reproduction.cross-tensors.v1", "passed": True,
              "source_session_id": source_id, "target_session_id": target_id,
              "target_theorem_sha256": initial["theorem_id"], "pins": pins, "tensor_counts": counts,
              "gates": {"candidate_release_target_parameters_equal": True, "target_fresh_private_state": True,
                        "distinct_theorem_and_session": True, "source_full_private_state_unchanged": True},
              "scope": "CPU decode of actual GPU-host snapshots; no base-model load or new inference"}
    write(output, report)
    return report


def retire(run_root, gpu_url, *, gpu=None):
    run = Path(run_root).resolve()
    intent, pins = settings(run / "cross")
    require(intent["gpu_url"] == gpu_url, "use the original service")
    audit = read(run / "cross" / "tensor-audit.json")
    require(audit.get("schema_version") == "reap.reproduction.cross-tensors.v1" and audit.get("passed") is True
            and audit.get("source_session_id") == intent["source_session_id"]
            and audit.get("target_session_id") == intent["target_session_id"], "missing exact tensor audit")
    same(audit["pins"], pins, "tensor audit pins differ")
    checked = [(session_id, *checked_proof(run, session_id, intent["cpu_image"]))
               for session_id in (intent["source_session_id"], intent["target_session_id"])]
    require(checked[1][1]["theorem_sha256"] == audit["target_theorem_sha256"], "target proof/tensor identity differs")
    after = read(run / "cross" / "cross-source-after-intent.json")
    target = intent["target_session_id"]
    require(after["target_accepted_sha256"] == sha((run / "proof-check" / target / "accepted.json").read_bytes())
            and after["target_online_result_sha256"] == sha((run / "outputs" / target / "online-result.json").read_bytes()),
            "target changed since after snapshot")
    service = gpu or client(gpu_url)
    for session_id, _, online in checked:
        name, version = "cross-retired-final", online["policy_version"]
        request = {"session_id": session_id, "name": name, "expected_policy_version": version,
                   "mutation_retry_allowed": False}
        write(run / "cross" / ("retire-" + session_id + "-intent.json"), request)
        response = service.retire_session(session_id, name, expected_policy_version=version)
        write(run / "cross" / ("retire-" + session_id + "-response.json"), response)
        require(response.get("schema_version") == "reap.gpu.retirement.v1"
                and response.get("status") == "released" and response.get("session_id") == session_id
                and type(response.get("policy_version")) is int and response["policy_version"] == version
                and response.get("snapshot") == name and response.get("mutation_retry_allowed") is False
                and response.get("tombstone_scope") == "current_runtime", "retirement did not confirm exact release")
        digest(response.get("snapshot_sha256"))
    return {"released_sessions": [item[0] for item in checked]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pub = commands.add_parser("publish")
    pub.add_argument("--session-id", required=True)
    pub.add_argument("--cpu-image", required=True)
    snap = commands.add_parser("source-snapshot")
    snap.add_argument("--phase", choices=("before", "after"), required=True)
    end = commands.add_parser("retire")
    for item in (pub, snap, end):
        item.add_argument("--run-root", type=Path, required=True)
        item.add_argument("--gpu-url", required=True)
    tensor = commands.add_parser("check-tensors")
    tensor.add_argument("--gpu-run-root", type=Path, required=True)
    tensor.add_argument("--cross-dir", type=Path, required=True)
    tensor.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    function = {"publish": publish, "source-snapshot": source_snapshot,
                "check-tensors": check_tensors, "retire": retire}[args.pop("command")]
    print(json.dumps(function(**args), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
