"""Explicit fresh/chain/bank routing for legacy theorem experience.

The catalog contains metadata and pins, never tensor payloads.
Family/tags/priority are operator routing declarations, not mathematical
relationships, semantic retrieval, or independently verified proof evidence.
Mixed learner model releases intentionally use a different system.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import uuid

CATALOG = "reap.theorem-experience.catalog.v1"
SELECTION = "reap.theorem-experience.selection.v1"
PROFILE = "legacy-theorem-search-visit-backup-v1"
RULE = "exact-family-and-required-tags;priority-descending;experience-id-ascending"
ATTESTATION = "operator routing labels; acceptance is a recorded attestation, not reverified here"
PINS = ("experience_id", "experience_weights_sha256", "experience_snapshot_sha256")
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
SHA = re.compile(r"[0-9a-f]{64}")
MAX_BYTES = 4 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def _sha(value, name):
    require(isinstance(value, str) and SHA.fullmatch(value), name+" must be a lowercase SHA256")


def _id(value, name):
    require(isinstance(value, str) and ID.fullmatch(value), name+" must be an explicit identifier")


def _fields(value, names, name):
    require(type(value) is dict and set(value) == set(names.split()), "invalid "+name+" fields")


def no_links(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (path, *path.parents)),
            "linked catalog/source/store path rejected")
    return path


def decode(raw):
    require(len(raw) <= MAX_BYTES, "catalog/selection exceeds the explicit size limit")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: require(False, "nonfinite JSON rejected"))


def read_json(path):
    from .verified_dataset_store import safe_directory
    path = no_links(path)
    with safe_directory(path.parent) as directory:
        return decode(directory.read(path.name))


def write_new(path, value):
    """Complete-file no-replace link; failed/unknown writes are never retried."""
    from .verified_dataset_store import safe_directory
    path = no_links(path)
    with safe_directory(path.parent) as directory:
        staging = ".experience-staged-"+uuid.uuid4().hex
        directory.write_new(staging, encoded(value))
        if os.name == "nt":
            os.link(directory.path/staging, directory.path/path.name, follow_symlinks=False)
        else:
            os.link(staging, path.name, src_dir_fd=directory.handle, dst_dir_fd=directory.handle, follow_symlinks=False)
        directory.sync()


def validate_contract(contract):
    _fields(contract, "backend base_sha256 objective hidden_size lora_rank lora_alpha lora_dropout target_modules value_head search_config", "real-search contract")
    require(type(contract) is dict and contract.get("backend") == "real-search"
        and contract.get("objective") == "search_visit_backup"
        and contract.get("value_head") == "linear-silu-linear-sigmoid-v1", "only legacy real-search theorem contracts are routable")
    _sha(contract.get("base_sha256"), "base/tokenizer identity")
    search = contract.get("search_config")
    require(type(search) is dict and search.get("objective") == "search_visit_backup"
        and type(search.get("gamma")) in (int, float) and math.isfinite(search["gamma"])
        and 0 < search["gamma"] < 1, "strict search objective/gamma contract required")
    for name in ("hidden_size", "lora_rank"):
        require(type(contract.get(name)) is int and contract[name] > 0, "invalid contract "+name)
    require(type(contract.get("target_modules")) is list and bool(contract["target_modules"])
        and all(isinstance(v, str) and v for v in contract["target_modules"]), "contract target modules required")
    encoded(contract)  # finite JSON; the actual server must match every field.


def _labels(family, tags, priority):
    _id(family, "family_id")
    require(type(tags) is list and tags == sorted(set(tags)) and len(tags) <= 32, "tags must be unique and sorted")
    for tag in tags:
        _id(tag, "tag")
    require(type(priority) is int and -1000000 <= priority <= 1000000, "priority must be an explicit bounded integer")


def _metadata(value):
    _fields(value, "schema_version experience_id source acceptance transfer weights_sha256", "experience metadata")
    require(value["schema_version"] == "reap.gpu.experience.v1" and value["transfer"] == ["adapter", "value_head"],
            "only theorem experience metadata is supported")
    _id(value["experience_id"], "experience_id"); _sha(value["weights_sha256"], "weights pin")
    source = value["source"]
    _fields(source, "session_id theorem_id policy_version snapshot snapshot_sha256 parent_experience_id", "source")
    for name in ("session_id", "snapshot"):
        _id(source[name], name)
    for name in ("theorem_id", "snapshot_sha256"):
        _sha(source[name], name)
    require(type(source["policy_version"]) is int and source["policy_version"] > 0, "source must have a committed update")
    if source["parent_experience_id"] is not None:
        _id(source["parent_experience_id"], "parent experience")
    acceptance = value["acceptance"]
    _fields(acceptance, "completed passed kind evidence_sha256 source", "acceptance")
    require(acceptance["completed"] is True and acceptance["passed"] is True and acceptance["kind"] == "independent-lean"
        and encoded(acceptance["source"]) == encoded(source), "completed independent-lean acceptance attestation required")
    _sha(acceptance["evidence_sha256"], "acceptance evidence")


def validate_catalog(catalog):
    _fields(catalog, "schema_version profile contract contract_sha256 entries routing_attestation", "catalog")
    require(catalog["schema_version"] == CATALOG and catalog["profile"] == PROFILE
        and catalog["routing_attestation"] == ATTESTATION, "catalog profile differs")
    validate_contract(catalog["contract"])
    require(catalog["contract_sha256"] == digest(catalog["contract"]), "catalog contract hash differs")
    require(type(catalog["entries"]) is list and len(catalog["entries"]) <= 128, "catalog permits at most 128 approved sources")
    ids = []
    for entry in catalog["entries"]:
        _fields(entry, "metadata metadata_sha256 release_manifest_sha256 family_id tags priority", "catalog entry")
        _metadata(entry["metadata"]); _labels(entry["family_id"], entry["tags"], entry["priority"])
        require(entry["metadata_sha256"] == digest(entry["metadata"]), "experience metadata hash differs")
        _sha(entry["release_manifest_sha256"], "release manifest")
        ids.append(entry["metadata"]["experience_id"])
    require(ids == sorted(set(ids)), "catalog IDs must be unique and sorted")
    require(len(encoded(catalog)) <= MAX_BYTES, "catalog too large")
    return catalog


def load_catalog(path, expected_sha256):
    from .verified_dataset_store import safe_directory
    path = no_links(path); _sha(expected_sha256, "catalog pin")
    with safe_directory(path.parent) as directory:
        raw = directory.read(path.name)
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "catalog file SHA differs")
    catalog = decode(raw)
    require(encoded(catalog) == raw, "catalog must preserve canonical immutable export bytes")
    return validate_catalog(catalog)


def export_catalog(store_root, approvals, contract):
    """Read actual ExperienceStore packages; export no weights and change no source."""
    from gpu_runtime.experience_store import ExperienceStore
    from .verified_dataset_store import safe_directory
    validate_contract(contract)
    root = no_links(store_root); require(root.is_dir(), "existing experience store required")
    require(type(approvals) is list and len(approvals) <= 128, "explicit approved source list required")
    store = ExperienceStore(root); entries = []; seen = set()
    for approval in approvals:
        _fields(approval, "experience_id family_id tags priority", "approval")
        sid = approval["experience_id"]; _id(sid, "approved experience")
        require(sid not in seen, "duplicate approved experience"); seen.add(sid)
        _labels(approval["family_id"], approval["tags"], approval["priority"])
        folder = no_links(root/sid/"release")
        with safe_directory(folder) as directory:
            before = {name: hashlib.sha256(directory.read(name)).hexdigest()
                      for name in ("manifest.json", "session.json", "backend.json")}
            metadata, weights = store.load(sid)
            _metadata(metadata)
            require(encoded(weights["contract"]) == encoded(contract), "approved source contract differs")
            require(all(hashlib.sha256(directory.read(name)).hexdigest() == value for name, value in before.items()),
                    "experience source changed while cataloging")
        entries.append({"metadata": metadata, "metadata_sha256": digest(metadata),
            "release_manifest_sha256": before["manifest.json"],
            **{name: approval[name] for name in ("family_id", "tags", "priority")}})
    catalog = {"schema_version": CATALOG, "profile": PROFILE, "contract": deepcopy(contract),
        "contract_sha256": digest(contract), "entries": sorted(entries, key=lambda e: e["metadata"]["experience_id"]),
        "routing_attestation": ATTESTATION}
    return validate_catalog(catalog)


def select(catalog, query):
    validate_catalog(catalog)
    _fields(query, "mode session_id target_theorem_sha256 family_id tags predecessor_experience_id allow_fresh_fallback", "selection query")
    require(query["mode"] in ("fresh", "chain", "bank"), "explicit fresh/chain/bank mode required")
    _id(query["session_id"], "target session"); require(len(query["session_id"]) <= 40, "target session exceeds online limit")
    _sha(query["target_theorem_sha256"], "target theorem")
    _labels(query["family_id"], query["tags"], 0)
    require(type(query["allow_fresh_fallback"]) is bool and (query["mode"] == "bank" or not query["allow_fresh_fallback"]),
            "fresh fallback must be explicitly enabled only for bank")
    predecessor = query["predecessor_experience_id"]
    if query["mode"] == "chain":
        _id(predecessor, "explicit chain predecessor")
    else:
        require(predecessor is None, "predecessor is only valid for chain")
    eligible = [e for e in catalog["entries"] if e["family_id"] == query["family_id"]
        and set(query["tags"]) <= set(e["tags"])
        and e["metadata"]["source"]["session_id"] != query["session_id"]
        and e["metadata"]["source"]["theorem_id"] != query["target_theorem_sha256"]]
    if query["mode"] == "chain":
        eligible = [e for e in eligible if e["metadata"]["experience_id"] == predecessor]
    chosen = None; outcome = "fresh"
    if query["mode"] != "fresh":
        if not eligible:
            require(query["allow_fresh_fallback"], "no approved compatible source; no implicit fresh fallback")
            outcome = "fresh_fallback"
        else:
            entry = sorted(eligible, key=lambda e: (-e["priority"], e["metadata"]["experience_id"]))[0]
            meta = entry["metadata"]
            chosen = {"experience_id": meta["experience_id"], "experience_weights_sha256": meta["weights_sha256"],
                "experience_snapshot_sha256": meta["source"]["snapshot_sha256"], "metadata_sha256": entry["metadata_sha256"]}
            outcome = "inherited"
    return {"schema_version": SELECTION, "catalog_sha256": digest(catalog), "query": deepcopy(query),
            "rule": RULE, "outcome": outcome, "chosen": chosen,
            "expected_initialization_contract_sha256": catalog["contract_sha256"]}


def validate_policy(policy, *, session_id, theorem_sha256, gamma=None):
    _fields(policy, "catalog selection", "experience policy")
    selection = policy["selection"]
    require(type(selection) is dict and "query" in selection, "selection query required")
    expected = select(policy["catalog"], selection["query"])
    require(encoded(expected) == encoded(selection), "selection differs from pinned deterministic rule")
    require(selection["query"]["session_id"] == session_id and selection["query"]["target_theorem_sha256"] == theorem_sha256,
            "policy targets a different session/theorem")
    if gamma is not None:
        require(policy["catalog"]["contract"]["search_config"]["gamma"] == gamma, "policy contract gamma differs from execution")
    return {} if selection["chosen"] is None else {key: selection["chosen"][key] for key in PINS}


def validate_execution_contract(created, policy):
    contract = policy["catalog"]["contract"]
    require(created.get("initialization_contract_sha256") == digest(contract)
        and encoded(created.get("initialization_contract")) == encoded(contract), "server initialization contract/base differs; no Lean may start")
    # Legacy theorem snapshots intentionally omit the default role; actor and
    # learner roles are explicit and must never enter this policy family.
    require(created.get("role", "theorem") == "theorem" and created.get("completed") is False
        and type(created.get("policy_version")) is int and created["policy_version"] == 0
        and type(created.get("optimizer_metadata", {}).get("steps")) is int and created["optimizer_metadata"]["steps"] == 0
        and created.get("event_receipts") == {}
        and created.get("buffer_metadata") == {"events": {}, "pending_event_ids": [], "consumed_event_ids": []},
        "policy initialization is not a fresh theorem session")
    chosen = policy["selection"]["chosen"]
    if chosen is None:
        require(created.get("lineage") == {}, "fresh requires base initialization with empty lineage")
    else:
        metadata = next(e["metadata"] for e in policy["catalog"]["entries"] if e["metadata"]["experience_id"] == chosen["experience_id"])
        require(encoded(created.get("lineage", {}).get("source")) == encoded(metadata["source"]), "actual source differs from selected catalog metadata")


def materialize(catalog, queries, project):
    project = no_links(project); require(project.is_dir(), "project directory required")
    rows = []; seen = set()
    for request in queries:
        _fields(request, "mode session_id theorem_file family_id tags predecessor_experience_id allow_fresh_fallback", "batch policy request")
        relative = PurePosixPath(request["theorem_file"])
        require(not relative.is_absolute() and relative.as_posix() == request["theorem_file"] and "\\" not in str(relative)
            and ":" not in str(relative) and all(p not in (".", "..") for p in relative.parts), "theorem must stay inside project")
        source = no_links(project.joinpath(*relative.parts)); require(source.is_file(), "theorem source missing")
        sid = request["session_id"]; require(sid not in seen, "duplicate target session"); seen.add(sid)
        query = {k: v for k, v in request.items() if k != "theorem_file"}
        query["target_theorem_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        selection = select(catalog, query); policy = {"catalog": deepcopy(catalog), "selection": selection}
        pins = validate_policy(policy, session_id=sid, theorem_sha256=query["target_theorem_sha256"])
        rows.append({"session_id": sid, "theorem_file": request["theorem_file"], **pins, "experience_policy": policy})
    require(0 < len(rows) <= 10000, "materialization requires 1..10000 targets")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export-catalog"); export.add_argument("--store-root", type=Path, required=True)
    export.add_argument("--approvals-json", type=Path, required=True); export.add_argument("--contract-json", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    for name in ("select", "materialize-batch"):
        child = sub.add_parser(name); child.add_argument("--catalog", type=Path, required=True)
        child.add_argument("--catalog-sha256", required=True); child.add_argument("--query-json", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if name == "materialize-batch": child.add_argument("--project-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export-catalog":
        value = export_catalog(args.store_root, read_json(args.approvals_json), read_json(args.contract_json))
        write_new(args.output, value)
    else:
        catalog = load_catalog(args.catalog, args.catalog_sha256)
        query = read_json(args.query_json)
        if args.command == "select":
            write_new(args.output, {"catalog": catalog, "selection": select(catalog, query)})
        else:
            rows = materialize(catalog, query, args.project_dir)
            # The existing batch format is JSONL. Publish a complete payload once.
            path = no_links(args.output); require(not path.exists(), "output already exists")
            from .verified_dataset_store import safe_directory
            with safe_directory(path.parent) as directory:
                staging = ".experience-staged-"+uuid.uuid4().hex
                raw = b"".join(encoded(row) for row in rows); require(len(raw) <= 16*1024*1024, "batch manifest too large")
                directory.write_new(staging, raw)
                if os.name == "nt": os.link(directory.path/staging, directory.path/path.name, follow_symlinks=False)
                else: os.link(staging, path.name, src_dir_fd=directory.handle, dst_dir_fd=directory.handle, follow_symlinks=False)
                directory.sync()
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "GPU_used": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
