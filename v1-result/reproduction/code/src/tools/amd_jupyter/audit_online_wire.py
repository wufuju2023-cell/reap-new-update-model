#!/usr/bin/env python3
"""Independently match local Lean observer/receipts to exported raw HTTP bytes.

Local files only.  No remote calls or mutation retries.  A wire match alone is
not a proof, an independent tensor recomputation, or a GPU-container gate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

try:
    from .collect_http_evidence import SCHEMA, decode_blob, json_bytes, require, sha
except ImportError:
    from collect_http_evidence import SCHEMA, decode_blob, json_bytes, require, sha


def jsonl(path):
    data = path.read_bytes()
    require(not data or data.endswith(b"\n"), "partial final JSONL record")
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def strip_thinking(text):
    text = text.lstrip(" \t\r\n\f\v")
    if text.startswith("<think>") and "</think>" in text:
        text = text.split("</think>", 1)[1].lstrip(" \t\r\n\f\v")
    return text


def prompt_of(request):
    messages = request.get("body", {}).get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        return None
    return messages[0].get("content")


def load_jobs(collection: Path):
    raw = collection.read_bytes()
    document = json.loads(raw)
    require(document.get("schema_version") == SCHEMA, "unsupported collection schema")
    jobs, seen = [], set()
    for entry in document["entries"]:
        request_id = entry["request_id"]
        require(request_id not in seen, "duplicate collected request UUID")
        seen.add(request_id)
        files = {name: decode_blob(entry["files"][name]) for name in ("request.json", "result.json", "response.bin")}
        require(files["request.json"] is not None, "collected job lacks a request")
        request = json.loads(files["request.json"])
        require(request["request_id"] == request_id, "request UUID mismatch")
        require(request["path"].startswith("/sessions/" + entry["session_id"] + "/")
                or request["path"] == "/sessions/" + entry["session_id"], "collected session/path mismatch")
        result = None if files["result.json"] is None else json.loads(files["result.json"])
        response = None
        if result is not None:
            require(result["request_id"] == request_id and result["request_sha256"] == sha(json_bytes(request)),
                    "result request identity/digest mismatch")
            require(files["response.bin"] is not None and len(files["response.bin"]) == result["bytes"]
                    and sha(files["response.bin"]) == result["sha256"], "raw HTTP response corruption")
            if result["state"] == "done":
                response = json.loads(files["response.bin"])
        jobs.append({"request_id": request_id, "session_id": entry["session_id"], "request": request,
                     "result": result, "response": response,
                     "raw_sha256": {name: None if data is None else sha(data) for name, data in files.items()}})
    return document, jobs, sha(raw)


def audit(collection: Path, session_dir: Path):
    document, all_jobs, collection_sha = load_jobs(collection)
    session = json.loads((session_dir / "session.json").read_bytes())
    sid, tree = session["session_id"], session["tree_id"]
    require(sid in document["allowed_sessions"], "session was not included in collection allowlist")
    jobs = [job for job in all_jobs if job["session_id"] == sid]
    require(jobs, "no collected jobs for requested session")
    success = [job for job in jobs if job["result"] is not None
               and job["result"].get("state") == "done" and job["result"].get("status") in (200, 201)]
    events = jsonl(session_dir / "observer.jsonl")
    require([e["sequence"] for e in events] == list(range(len(events))), "observer sequence gap/replay")
    require(all(e["session_id"] == sid and e["tree_id"] == tree for e in events), "observer identity mismatch")
    require(all(a["monotonic_ns"] <= b["monotonic_ns"] for a, b in zip(events, events[1:])), "observer time decreased")
    wall = [w for w in jsonl(session_dir / "wall_clock.jsonl") if w.get("name") == "tactic_gen" and "stop" in w]
    gaps, learns, generations = [], [], []
    prefix = "/sessions/" + sid
    create = json.loads((session_dir / "create-receipt.json").read_bytes())
    creates = [job for job in success if job["request"]["path"] == prefix and job["response"] == create]
    if len(creates) != 1:
        gaps.append("missing/ambiguous exact raw create-session receipt")
    metadata = create["value_metadata"]
    require(metadata["objective"] == "search_visit_backup" and metadata["gamma"] == session["gamma"]
            and metadata["value_semantics"] == "reap.search_backup_discounted_return.v1", "strict session semantics mismatch")
    previous = None
    for path in sorted((session_dir / "checkpoints").glob("learn-*.request.json")):
        event = json.loads(path.read_bytes())
        receipt_path = path.with_name(path.name.replace("request", "receipt"))
        if not receipt_path.exists():
            gaps.append("local learn intent has no receipt: " + event["event_id"])
            continue
        receipt = json.loads(receipt_path.read_bytes())
        version = event["policy_version"]
        matches = [job for job in success if job["request"]["path"] == prefix + "/learn/v1"
                   and job["request"]["body"] == {"expected_policy_version": version, "event": event}
                   and job["response"] == receipt]
        if len(matches) != 1:
            gaps.append("missing/ambiguous exact raw learn receipt: " + event["event_id"])
        require(receipt["applied"] is True and receipt["idempotent"] is False
                and receipt["policy_version"] == version + 1, "invalid local applied/version chain")
        detail = receipt["detail"]
        require(detail["optimizer_steps"] == version + 1 and detail["objective"] == "search_visit_backup", "wrong optimizer/objective")
        for key in ("finite_loss", "finite_gradients", "finite_parameters", "finite_optimizer_state", "base_parameters_frozen"):
            require(detail.get(key) is True, "missing numeric receipt gate")
        require(detail["prompt_sha256"] == sha(event["prompt"].encode()), "learn prompt hash mismatch")
        trace = detail["search_trace"]
        require(all(trace[k] == event[k] for k in ("tree_id", "step", "node_index", "policy_version", "gamma", "terminal_verified")),
                "learn source trace mismatch")
        expected = max(detail["training_config"]["value_floor"], event["gamma"] ** (-event["backup"]["value_sum"] / event["backup"]["visits"] - 1))
        require(math.isclose(expected, detail["value_target"], rel_tol=1e-12), "backup target mismatch")
        for name in ("adapter", "value_head", "optimizer"):
            current = detail["parameter_diffs"][name]
            require(current["before_sha256"] != current["after_sha256"], "no parameter/optimizer change")
            if previous is not None:
                require(previous[name]["after_sha256"] == current["before_sha256"], "parameter hash chain broken")
        previous = detail["parameter_diffs"]
        checkpoint = [e for e in events if e["kind"] == "checkpoint" and e["step"] == event["step"]]
        ack = [e for e in events if e["kind"] == "checkpoint_ack" and e["step"] == event["step"]]
        require(len(checkpoint) == len(ack) == 1 and checkpoint[0]["policy_version"] == version
                and ack[0]["policy_version"] == version + 1, "checkpoint/ACK version chain mismatch")
        learns.append({"event_id": event["event_id"], "version_before": version, "version_after": version + 1,
                       "step": event["step"], "request_ids": [m["request_id"] for m in matches],
                       "checkpoint_sequence": checkpoint[0]["sequence"], "ack_sequence": ack[0]["sequence"],
                       "raw_receipt_exactly_matched": len(matches) == 1,
                       "parameter_diffs": detail["parameter_diffs"]})
    groups = defaultdict(list)
    for event in events:
        if event["kind"] == "generation":
            groups[tuple(event[k] for k in ("step", "node_index", "generation_index", "policy_version"))].append(event)
    for (step, node, index, version), group in groups.items():
        first = group[0]
        selections = [e for e in events if e["kind"] == "selection" and e["step"] == step
                      and e["node_index"] == node and e["sequence"] < first["sequence"]]
        require(selections, "generation lacks preceding node selection")
        start = selections[-1]["monotonic_ns"]
        calls = [w for w in wall if w["start"] >= start and w["stop"] <= first["monotonic_ns"]
                 and isinstance(w.get("extra", {}).get("result"), dict)]
        require(len(calls) == 1, "generation cannot be uniquely associated with Lean policy response ID")
        response_id = calls[0]["extra"]["result"]["id"]
        policies = [job for job in success if job["request"]["path"] == prefix + "/policy/v1/chat/completions"
                    and job["response"].get("id") == response_id]
        policy_ok = len(policies) == 1
        value_matches = []
        if not policy_ok:
            gaps.append("missing/ambiguous policy raw response: " + response_id)
        else:
            policy = policies[0]
            require(policy["response"].get("policy_version") == version, "raw policy version differs from observer ACK label")
            require(prompt_of(policy["request"]) == first["prompt"], "raw policy prompt differs from observer prompt")
            for generated in group:
                require(generated["prompt"] == first["prompt"] and generated["search_value"] == first["search_value"], "group prompt/value mismatch")
                matching_choices = []
                for choice in policy["response"]["choices"]:
                    if strip_thinking(choice["message"]["content"]) != generated["tactic"]:
                        continue
                    tokens = choice.get("logprobs", {}).get("content")
                    require(tokens and all(math.isfinite(t["logprob"]) and t["logprob"] <= 0 for t in tokens), "invalid raw token logprobs")
                    if math.isclose(sum(t["logprob"] for t in tokens), generated["raw_logprob"], abs_tol=1e-8, rel_tol=1e-10):
                        matching_choices.append(choice)
                require(matching_choices, "generated tactic/raw logprob not present in exact HTTP response")
            for job in success:
                if job["request"]["path"] != prefix + "/value/v1/chat/completions" or prompt_of(job["request"]) != first["prompt"]:
                    continue
                response = job["response"]
                if response.get("policy_version") != version:
                    continue
                score = json.loads(response["choices"][0]["message"]["content"])["score"]
                require(math.isfinite(score) and 1 <= score <= metadata["max_distance"], "value is not bounded distance")
                if math.isclose(-score, first["search_value"], rel_tol=1e-10, abs_tol=1e-8):
                    value_matches.append(job["request_id"])
            if not value_matches:
                gaps.append(f"no raw value response matches version/prompt/distance at step {step}, node {node}")
        generations.append({"step": step, "node_index": node, "generation_index": index, "policy_version": version,
            "observer_sequences": [e["sequence"] for e in group], "response_id": response_id,
            "policy_request_ids": [m["request_id"] for m in policies], "prompt_sha256": sha(first["prompt"].encode()),
            "policy_prompt_version_tactics_raw_logprobs_match": policy_ok,
            "value_request_ids": value_matches, "value_match_unique": len(value_matches) == 1,
            "value_note": "Multiple equal-input/version/value matches are equivalent evidence, not uniquely paired calls."})
    known_ids = {row["event_id"] for row in learns}
    unknown_mutations = []
    for job in jobs:
        if job["request"]["path"] == prefix + "/learn/v1":
            result = job["result"]
            event_id = job["request"]["body"].get("event", {}).get("event_id")
            if event_id not in known_ids or result is None or result.get("state") != "done":
                unknown_mutations.append(job["request_id"])
    if unknown_mutations:
        gaps.append("unacknowledged/pending/unknown LEARN jobs exist; do not retry")
    for row in learns:
        row["later_generation_groups"] = [g["response_id"] for g in generations if g["policy_version"] == row["version_after"]
                                          and min(g["observer_sequences"]) > row["ack_sequence"]]
        row["later_generation_wire_verified"] = any(g["response_id"] in row["later_generation_groups"]
            and g["policy_prompt_version_tactics_raw_logprobs_match"] and g["value_request_ids"] for g in generations)
    final = json.loads((session_dir / "result.json").read_bytes()) if (session_dir / "result.json").exists() else None
    online = json.loads((session_dir / "online-result.json").read_bytes()) if (session_dir / "online-result.json").exists() else None
    final_execution = bool(final and online and final.get("solved") is True and online.get("root_verified") is True
                           and online.get("returncode") == 0 and online.get("status") == "passed_execution")
    consumed = any(row["later_generation_wire_verified"] for row in learns)
    return {"schema_version": "reap.independent-online-wire-audit.v1", "session_id": sid, "tree_id": tree,
        "status": "GAPS" if gaps else "MATCHED_EXECUTION_WIRE" if final_execution and consumed else "MATCHED_PARTIAL_WIRE",
        "collection_sha256": collection_sha, "cpu_input_sha256": {p.name: sha(p.read_bytes()) for p in
            (session_dir / "session.json", session_dir / "observer.jsonl", session_dir / "wall_clock.jsonl")},
        "session_jobs": len(jobs), "create_receipt_matched": len(creates) == 1, "learns": learns,
        "generation_groups": generations, "unknown_mutation_request_ids": unknown_mutations,
        "final_lean_execution_report_present": final_execution, "some_update_consumed_by_later_wire_generation": consumed,
        "gaps": gaps,
        "limits": ["No remote requests were sent. Raw byte hashes/lengths and semantic links were verified locally.",
                   "Tensor changes are receipt/hash-chain evidence; no independent raw tensor or full-base recomputation in this audit.",
                   "A partial run has no final proof even if its raw HTTP chain matches.",
                   "No GPU container execution or proof-quality improvement is established.",
                   "Value calls with identical prompt/version/score may not be individually distinguishable; counts are explicit."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--cpu-session-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "audit output must be new")
    report = audit(args.collection, args.cpu_session_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(json_bytes(report) + b"\n")
    print(json_bytes({"output": str(args.output), "status": report["status"], "session_id": report["session_id"],
                      "learns": len(report["learns"]), "generation_groups": len(report["generation_groups"]),
                      "gaps": report["gaps"]}).decode(), flush=True)
    return 1 if report["gaps"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
