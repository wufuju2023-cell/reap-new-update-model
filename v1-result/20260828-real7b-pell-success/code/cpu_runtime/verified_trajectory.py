"""Independent successful-path replay/export; never a search_visit_backup batch.

Run inside the offline CPU environment with an existing Lean project. All inputs
are copied and hashed before Lean runs. Only a successful whole-theorem replay,
exact per-action state trace, and axiom check permit dataset publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import textwrap
import time
from typing import Any

PROFILE = "verified-generated-action-negative-longest-branch-v1"
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
THEOREM = re.compile(r"[A-Za-z_][A-Za-z_0-9']*(?:\.[A-Za-z_][A-Za-z_0-9']*)*\Z")


class TrajectoryRejected(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrajectoryRejected(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)


def checked_axioms(stdout: str, theorem: str) -> list[str]:
    require(bool(THEOREM.fullmatch(theorem)), "unsupported theorem identifier")
    matches = re.findall(r"^'" + re.escape(theorem) + r"' depends on axioms: \[([^\]]*)\]$",
                         stdout, re.MULTILINE)
    no_axioms = re.findall(r"^'" + re.escape(theorem) + r"' does not depend on any axioms$",
                           stdout, re.MULTILINE)
    matches.extend("" for _ in no_axioms)
    require(len(matches) == 1, "missing or ambiguous exact theorem axiom report")
    axioms = [s.strip() for s in matches[0].split(",") if s.strip()]
    require(set(axioms) <= ALLOWED_AXIOMS and "sorryAx" not in stdout,
            "unapproved proof axiom")
    return axioms


def _trim(text: str) -> str:
    return text.strip(" \t\r\n\v\f")  # Lean trimAscii, not Unicode whitespace.


def _bullet(script: str) -> str:
    lines = (_trim(script) or "skip").split("\n")
    return "· " + lines[0] + "".join("\n  " + line for line in lines[1:])


def plan_success_path(tree: dict, events: list[dict], session: dict) -> dict:
    """Choose exactly Reap's first solved OR edge and all sorted AND children.

    This is only a candidate plan. Its rows MUST NOT be consumed before replay.
    Strict append-only tree topology and independent focused branches are the
    supported v1 subset; no invented labels for cross-branch metavariables.
    """
    nodes = tree.get("nodes")
    require(isinstance(nodes, list) and nodes and type(tree.get("solution")) is int and tree["solution"] == 0,
            "missing solved root")
    sid, tid = session["session_id"], session["tree_id"]
    for sequence, event in enumerate(events):
        require(event.get("schema_version") == "reap.training.observer.v1" and
                event.get("session_id") == sid and event.get("tree_id") == tid and
                type(event.get("sequence")) is int and event["sequence"] == sequence,
                "observer identity or sequence mismatch")
    checkpoints = [e for e in events if e.get("kind") == "checkpoint"]
    require(bool(checkpoints), "missing final checkpoint")
    final = checkpoints[-1]
    require(final.get("root_is_solved") is True and final.get("tree", {}).get("root_index") == 0
            and final["tree"].get("nodes") == nodes, "final checkpoint/tree mismatch")
    require(events[-1].get("kind") == "checkpoint_ack" and
            events[-1].get("step") == final.get("step") and
            events[-1].get("policy_version") == final.get("policy_version"),
            "missing exact final ACK")
    generations: dict[tuple, dict] = {}
    creations: dict[tuple, dict] = {}
    for event in events:
        key = (event.get("node_index"), event.get("generation_index"), event.get("candidate_index"))
        if event.get("kind") == "generation":
            require(key not in generations, "duplicate generation identity")
            generations[key] = event
        if event.get("kind") == "eval" and event.get("disposition") == "created":
            edge_key = (event.get("node_index"), event.get("child_index"))
            require(edge_key not in creations, "duplicate child creation")
            creations[edge_key] = event

    plan_nodes: list[dict] = []
    seen: set[int] = set()

    def visit(index: int, in_focus: bool = False) -> tuple[int, str, int, list[dict]]:
        require(type(index) is int and 0 <= index < len(nodes) and index not in seen,
                "cycle, repeated subtree, or invalid child index")
        seen.add(index)
        data, edges = nodes[index]["data"], nodes[index]["children"]
        states, kind = data.get("state"), data.get("toPlay")
        require(data.get("isSolved") is True and data.get("isPartial") is in_focus,
                "unsolved node or partial-goal flag outside its focused branch")
        require(isinstance(states, list) and all(isinstance(s, str) and s for s in states),
                "invalid saved states")
        require(kind in ("OR", "AND"), "invalid node kind")
        for edge in edges:
            child = edge.get("childIndex")
            require(type(child) is int and index < child < len(nodes), "non-append-only edge")
            require(type(edge.get("edge", {}).get("isFocus")) is bool, "missing edge kind")
        position = len(plan_nodes)
        planned = {"node_index": index, "state": states, "kind": kind,
                   "tactic": "", "children": []}
        plan_nodes.append(planned)
        if kind == "OR":
            require(not any(e["edge"]["isFocus"] for e in edges), "focus edge on OR node")
            choices = [e for e in edges if nodes[e["childIndex"]]["data"].get("isSolved") is True]
            if not choices:
                require(not states and not edges, "isSolved flag without empty terminal")
                planned["kind"] = "terminal"
                return position, "", 0, []
            require(len(states) == 1, "OR replay requires one goal")
            chosen = choices[0]  # Same array order as solvedOrChild?.
            child, tactic = chosen["childIndex"], chosen["edge"].get("tacticStr")
            require(isinstance(tactic, str) and bool(_trim(tactic)), "empty successful action")
            event = creations.get((index, child))
            require(event is not None and event.get("eval_result") == {"ok": None} and
                    event.get("partial_goal") is in_focus and event.get("tactic") == tactic,
                    "selected edge lacks successful original creation")
            key = (index, event["generation_index"], event["candidate_index"])
            generation = generations.get(key)
            require(generation is not None and generation.get("tactic") == tactic and
                    generation["sequence"] < event["sequence"] and
                    generation.get("policy_version") == event.get("policy_version") and
                    generation.get("goal_state") == states[0] and
                    json.loads(generation.get("state_key", "null")) == states and
                    isinstance(generation.get("prompt"), str) and generation["prompt"],
                    "generation/state/version provenance mismatch")
            child_position, rest_script, rest_value, rest_rows = visit(child, in_focus)
            planned.update(tactic=tactic, children=[child_position])
            value = rest_value - 1
            row = {"node_index": index, "child_index": child, "state": states,
                   "next_state": nodes[child]["data"]["state"], "tactic": tactic,
                   "return": value, "prompt": generation["prompt"],
                   "policy_version": generation["policy_version"],
                   "generation_sequence": generation["sequence"], "eval_sequence": event["sequence"],
                   "generation_index": event["generation_index"], "candidate_index": event["candidate_index"]}
            script = "\n".join(s for s in (_trim(tactic), _trim(rest_script)) if s)
            return position, script, value, [row, *rest_rows]
        require(len(states) >= 2 and edges and all(e["edge"]["isFocus"] for e in edges),
                "AND must contain every focused goal")
        ordered = sorted(edges, key=lambda e: e["edge"]["focusIndex"])
        require([e["edge"].get("focusIndex") for e in ordered] == list(range(len(states))),
                "AND missing, duplicated, or out-of-range focus index")
        scripts, values, rows = [], [], []
        for goal, edge in zip(states, ordered):
            child = edge["childIndex"]
            require(nodes[child]["data"].get("state") == [goal], "focus child state mismatch")
            p, script, value, branch_rows = visit(child, True)
            planned["children"].append(p)
            scripts.append(_bullet(script)); values.append(value); rows.extend(branch_rows)
        return position, "\n".join(scripts), min(values), rows

    _, script, value, rows = visit(0)
    require(bool(rows), "empty successful dataset")
    return {"plan": {"schema_version": "reap.verified-replay.plan.v1", "nodes": plan_nodes},
            "proof_script": script, "root_return": value, "rows": rows,
            "final_policy_version": final["policy_version"]}


def validate_historical_proof(source: bytes, proof: bytes, receipt: dict, result: dict,
                              session: dict, theorem: str, script: str) -> None:
    require(result.get("schema_version") == "reap.training.result.v1" and
            result.get("session_id") == session["session_id"] and result.get("solved") is True and
            result.get("status") == "solved" and result.get("error") is None and
            result.get("proof_script") == script, "result/successful path mismatch")
    require(receipt.get("proof_from_session") == session["session_id"] and
            receipt.get("source_theorem_sha256") == sha(source) == session.get("theorem_sha256") and
            receipt.get("generated_proof_sha256") == sha(proof), "historical proof identity/hash mismatch")
    require(type(receipt.get("returncode")) is int and receipt["returncode"] == 0 and
            receipt.get("network") == "none" and
            re.fullmatch(r"[a-f0-9]{64}", receipt.get("image", "")) is not None,
            "independent Lean acceptance receipt required")
    if "container_exit_code" in receipt:
        require(receipt["container_exit_code"] == 0 and
                receipt.get("container_image", "").removeprefix("sha256:") == receipt["image"],
                "historical container receipt mismatch")
    checked_axioms(receipt.get("stdout", ""), theorem)
    text = source.decode("utf8")
    require(text.count("  reapTrainingMCTS") == 1, "unsupported theorem source insertion")
    expected = text.replace("  reapTrainingMCTS", textwrap.indent(script, "  "))
    expected += "\n#print axioms " + theorem + "\n"
    require(proof == expected.encode(), "independent accepted proof is not this exact selected path")


def validate_trace(trace: dict, candidate: dict, theorem: str | None = None) -> None:
    require(trace.get("schema_version") == "reap.verified-replay.trace.v1" and
            trace.get("complete") is True and trace.get("root_return") == candidate["root_return"],
            "incomplete or wrong-return replay")
    if theorem is not None:
        require(trace.get("theorem") == theorem, "replay declaration differs from claimed theorem")
    expected = [{k: row[k] for k in ("node_index", "state", "next_state", "tactic", "return")}
                for row in candidate["rows"]]
    require(trace.get("rows") == expected, "actual Lean states/actions/returns differ from selected path")


def load_verified_dataset(directory: Path, *, expected_sha256: str | None = None) -> dict:
    """Read-only admission check for a future, separate replay learner.

    Pin the dataset digest in a trusted run manifest. Hashes establish binding
    and corruption detection, not authenticity against an attacker replacing
    the entire local evidence bundle and its digest together.
    """
    from .verified_dataset_store import read_bundle
    bundle = read_bundle(directory)
    raw = bundle["dataset.json"]
    if expected_sha256 is not None:
        require(re.fullmatch(r"[a-f0-9]{64}", expected_sha256) is not None and sha(raw) == expected_sha256,
                "dataset content pin mismatch")
    dataset = json.loads(raw)
    require(dataset.get("schema_version") == "reap.verified-success-trajectory.v1" and
            dataset.get("profile") == PROFILE, "unsupported successful replay profile")
    required = {"session.json", "result.json", "raw_tree.json", "observer.jsonl", "source.lean",
                "accepted-proof.lean", "historical-proof-receipt.json", "VerifiedReplay.lean",
                "historical-proof.stdout", "historical-proof.stderr", "plan.json", "replay.lean"}
    inputs = dataset.get("inputs_sha256")
    require(isinstance(inputs, dict) and set(inputs) == required, "incomplete frozen replay inputs")
    files = {name: bundle[name] for name in required}
    require(all(sha(content) == inputs[name] for name, content in files.items()), "frozen replay input hash mismatch")
    receipt_bytes = bundle["replay-receipt.json"]
    trace_bytes = bundle["trace.json"]
    require(sha(receipt_bytes) == dataset.get("replay_receipt_sha256") and
            sha(trace_bytes) == dataset.get("trace_sha256"), "replay receipt/trace hash mismatch")
    session, result, tree = (json.loads(files[name]) for name in ("session.json", "result.json", "raw_tree.json"))
    events = [json.loads(line) for line in files["observer.jsonl"].splitlines() if line.strip()]
    candidate = plan_success_path(tree, events, session)
    require(json.loads(files["plan.json"]) == candidate["plan"], "replay plan differs from successful path")
    historical = json.loads(files["historical-proof-receipt.json"])
    theorem = dataset.get("theorem", "")
    validate_historical_proof(files["source.lean"], files["accepted-proof.lean"], historical,
                              result, session, theorem, candidate["proof_script"])
    for stream in ("stdout", "stderr"):
        require(files["historical-proof." + stream].decode() == historical.get(stream, ""),
                "historical process stream mismatch")
    receipt = json.loads(receipt_bytes)
    require(receipt.get("schema_version") == "reap.verified-replay.receipt.v1" and
            receipt.get("session_id") == session["session_id"] and receipt.get("theorem") == theorem and
            type(receipt.get("returncode")) is int and receipt["returncode"] == 0 and
            receipt.get("inputs_sha256") == inputs, "independent replay receipt not accepted")
    stdout = bundle["replay.stdout"]
    stderr = bundle["replay.stderr"]
    require(sha(stdout) == receipt.get("stdout_sha256") and sha(stderr) == receipt.get("stderr_sha256"),
            "independent replay process stream hash mismatch")
    axioms = checked_axioms(stdout.decode(), theorem)
    validate_trace(json.loads(trace_bytes), candidate, theorem)
    require(dataset.get("session_id") == session["session_id"] and dataset.get("tree_id") == session["tree_id"] and
            dataset.get("theorem_sha256") == sha(files["source.lean"]) and dataset.get("axioms") == axioms and
            dataset.get("root_return") == candidate["root_return"] and
            dataset.get("final_policy_version") == candidate["final_policy_version"] and
            dataset.get("rows") == candidate["rows"], "dataset identity or target rows differ from verified evidence")
    return dataset


def export_verified(*, session_dir: Path, source: Path, proof: Path, proof_receipt: Path,
                    theorem: str, output: Path, lean_project: Path,
                    replay_module: Path, timeout: float = 180) -> dict:
    require(bool(THEOREM.fullmatch(theorem)), "unsupported theorem identifier")
    require(math.isfinite(timeout) and timeout > 0, "timeout must be positive finite")
    files = {name: (session_dir / name).read_bytes() for name in
             ("session.json", "result.json", "raw_tree.json", "observer.jsonl")}
    files.update({"source.lean": source.read_bytes(), "accepted-proof.lean": proof.read_bytes(),
                  "historical-proof-receipt.json": proof_receipt.read_bytes(),
                  "VerifiedReplay.lean": replay_module.read_bytes()})
    session, result, tree = (json.loads(files[name]) for name in ("session.json", "result.json", "raw_tree.json"))
    events = [json.loads(line) for line in files["observer.jsonl"].splitlines() if line.strip()]
    candidate = plan_success_path(tree, events, session)
    historical = json.loads(files["historical-proof-receipt.json"])
    validate_historical_proof(files["source.lean"], files["accepted-proof.lean"], historical,
                              result, session, theorem, candidate["proof_script"])
    # Recheck preserved independent process streams, not a JSON-only claim.
    for name in ("stdout", "stderr"):
        stream = proof_receipt.parent / ("proof." + name)
        files["historical-proof." + name] = stream.read_bytes()
        require(files["historical-proof." + name].decode() == historical.get(name, ""),
                "historical receipt/stream mismatch")
    text = files["source.lean"].decode()
    require(text.startswith("import ReapRuntime\n"), "standalone replay requires initial ReapRuntime import")
    output = output.resolve()
    require(not output.exists(), "output already exists; replay/export is never automatically retried")
    output.mkdir(parents=True)
    for name, content in files.items():
        write_new(output / name, content)
    write_new(output / "plan.json", encode(candidate["plan"]))
    tactic = "  reapVerifiedReplay " + json.dumps(str(output / "plan.json")) + " " + json.dumps(str(output / "trace.json"))
    replay_source = text.replace("  reapTrainingMCTS", tactic)
    # Course declarations also import Mathlib. Embedding declarations after
    # only the first import would strand later imports below declarations.
    # Preserve the exact leading contiguous imports; never rewrite a theorem.
    imports = re.match(r"(?:import [^\n]+\n)+", replay_source)
    require(imports is not None, "missing initial import block")
    end = imports.end()
    replay_source = replay_source[:end] + files["VerifiedReplay.lean"].decode() + "\n" + replay_source[end:]
    replay_source += "\n#print axioms " + theorem + "\n"
    write_new(output / "replay.lean", replay_source.encode())
    inputs = {p.name: sha(p.read_bytes()) for p in output.iterdir() if p.is_file()}
    command = ["lake", "env", "lean", str(output / "replay.lean")]
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, cwd=lean_project, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=timeout)
        elapsed = time.perf_counter() - started
        write_new(output / "replay.stdout", completed.stdout)
        write_new(output / "replay.stderr", completed.stderr)
        receipt = {"schema_version": "reap.verified-replay.receipt.v1", "session_id": session["session_id"],
                   "theorem": theorem, "command": command, "cwd": str(lean_project),
                   "returncode": completed.returncode, "elapsed_seconds": elapsed,
                   "inputs_sha256": inputs, "stdout_sha256": sha(completed.stdout),
                   "stderr_sha256": sha(completed.stderr)}
        write_new(output / "replay-receipt.json", encode(receipt))
        require(completed.returncode == 0, "standalone Lean replay failed; dataset not published")
        axioms = checked_axioms(completed.stdout.decode(), theorem)
        trace_bytes = (output / "trace.json").read_bytes()
        validate_trace(json.loads(trace_bytes), candidate, theorem)
        require(all(sha((output / name).read_bytes()) == digest for name, digest in inputs.items()),
                "replay input mutated during verification")
        dataset = {"schema_version": "reap.verified-success-trajectory.v1", "profile": PROFILE,
                   "session_id": session["session_id"], "tree_id": session["tree_id"], "theorem": theorem,
                   "theorem_sha256": sha(files["source.lean"]), "root_return": candidate["root_return"],
                   "action_unit": "one generated tactic string; focus transitions cost zero",
                   "and_return": "minimum branch return; shared-metavariable branch skipping rejected",
                   "state_origin": "exact saved state matched against independent Lean replay",
                   "training_integration": "none; not search_visit_backup or categorical-head training",
                   "axioms": axioms, "final_policy_version": candidate["final_policy_version"],
                   "inputs_sha256": inputs, "trace_sha256": sha(trace_bytes),
                   "replay_receipt_sha256": sha((output / "replay-receipt.json").read_bytes()),
                   "rows": candidate["rows"]}
        # Publish a complete closed file without overwrite. An interrupted write
        # leaves only a .prepared file; dataset.json can never be a partial JSON.
        with (output / "dataset.prepared.json").open("xb") as handle:
            handle.write(encode(dataset))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(output / "dataset.prepared.json", output / "dataset.json")
        return dataset
    except BaseException as error:
        write_new(output / "failed.json", encode({"error_type": type(error).__name__, "error": str(error),
                   "elapsed_seconds": time.perf_counter() - started,
                   "dataset_published": (output / "dataset.json").exists()}))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("session-dir", "source", "proof", "proof-receipt", "output", "lean-project"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--theorem", required=True)
    parser.add_argument("--replay-module", type=Path, default=Path(__file__).resolve().parents[1] /
                        "containers/cpu/verified-replay/VerifiedReplay.lean")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    dataset = export_verified(**vars(args))
    print(json.dumps({"status": "verified", "rows": len(dataset["rows"]),
                      "root_return": dataset["root_return"], "profile": PROFILE}))


if __name__ == "__main__":
    main()
