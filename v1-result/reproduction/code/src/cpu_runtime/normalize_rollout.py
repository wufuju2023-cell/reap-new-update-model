#!/usr/bin/env python3
"""Normalize Reap wall-clock events and final MCTS trees into training JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .reap_prompt import logged_prompts


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _choice_logprob(choice: dict[str, Any]) -> float | None:
    content = ((choice.get("logprobs") or {}).get("content") or [])
    values = [item.get("logprob") for item in content if isinstance(item.get("logprob"), (int, float))]
    return float(sum(values)) if values else None


def strip_thinking_prefix(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith("<think>"):
        return stripped
    marker = "</think>"
    if marker not in stripped:
        return stripped
    return stripped.split(marker, 1)[1].lstrip()


def generated_logprobs(events: Iterable[dict[str, Any]]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for event in events:
        if event.get("name") != "tactic_gen":
            continue
        extra = event.get("extra") or {}
        state = str(extra.get("goal", ""))
        response = extra.get("result") or {}
        for choice in response.get("choices", []):
            tactic = strip_thinking_prefix(
                str(((choice.get("message") or {}).get("content", "")))
            ).strip()
            logprob = _choice_logprob(choice)
            if tactic and logprob is not None:
                result[(state, tactic)] = logprob
    return result


def evaluation_records(session_id: str, events: Iterable[dict[str, Any]], priors: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    output = []
    for event in events:
        if event.get("name") != "tactic_eval":
            continue
        extra = event.get("extra") or {}
        state = str(extra.get("state", ""))
        tactic = str(extra.get("tactic", ""))
        verdict = extra.get("result")
        ok = isinstance(verdict, dict) and "ok" in verdict
        output.append(
            {
                "schema_version": "reap.training.transition.v1",
                "session_id": session_id,
                "event": "tactic_eval",
                "state": state,
                "tactic": tactic,
                "verdict": "accepted" if ok else "rejected",
                "eval_result": verdict,
                # Reap's EvalResult.ok only means the tactic was accepted.
                # It is not a positive reward unless final root replay passes.
                "reward": 0.0 if ok else None,
                "terminal_verified": False,
                "logprob": priors.get((state, tactic)),
                "start_ns": event.get("start"),
                "stop_ns": event.get("stop"),
            }
        )
    return output


def tree_records(session_id: str, tree: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    nodes = tree.get("nodes") or []
    for parent_index, node in enumerate(nodes):
        data = node.get("data") or {}
        state_parts = data.get("state") or []
        state = "\n".join(str(part) for part in state_parts)
        for child in node.get("children") or []:
            edge = child.get("edge") or {}
            probability = edge.get("probability")
            child_index = child.get("childIndex")
            child_data: dict[str, Any] = {}
            if isinstance(child_index, int) and 0 <= child_index < len(nodes):
                child_data = (nodes[child_index] or {}).get("data") or {}
            next_state = "\n".join(str(part) for part in (child_data.get("state") or []))
            output.append(
                {
                    "schema_version": "reap.training.transition.v1",
                    "session_id": session_id,
                    "event": "tree_edge",
                    "parent_index": parent_index,
                    "child_index": child_index,
                    "state": state,
                    "next_state": next_state,
                    "tactic": edge.get("tacticStr"),
                    "verdict": "verified_child",
                    # This is exp(raw_logprob / prior_temperature), possibly
                    # merged across duplicate siblings. It is not old_logprob.
                    "logprob": None,
                    "prior_weight": probability,
                    "value": edge.get("value"),
                    "visits": edge.get("numVisit"),
                    # A proof may be found during expansion before this edge is
                    # selected by a later PUCT step, so visits=0 is valid even
                    # for a solved child. Keep the two facts separate.
                    "visited": bool((edge.get("numVisit") or 0) > 0),
                    "solved_child": bool(child_data.get("isSolved", False)),
                    "puct": child.get("extra"),
                    "parent_solved": data.get("isSolved"),
                    "child_solved": child_data.get("isSolved"),
                }
            )
    return output


def normalize_session(session_dir: Path) -> list[dict[str, Any]]:
    session = read_json(session_dir / "session.json")
    session_id = str(session["session_id"])
    events = list(read_jsonl(session_dir / "wall_clock.jsonl"))
    output = evaluation_records(session_id, events, generated_logprobs(events))
    prompts = logged_prompts(events)
    for record in output:
        # The training model must see the same prompt (including premises)
        # as the policy that generated this tactic, not only the bare state.
        record["prompt"] = prompts.get(record["state"])
    tree_path = session_dir / "raw_tree.json"
    if tree_path.exists():
        output.extend(tree_records(session_id, read_json(tree_path)))
    result_path = session_dir / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        output.append(
            {
                "schema_version": "reap.training.session_result.v1",
                "session_id": session_id,
                "event": "session_result",
                "solved": bool(result.get("solved", False)),
                "status": result.get("status"),
                "proof_script": result.get("proof_script"),
                "reward": 1.0 if bool(result.get("solved", False)) else 0.0,
                "terminal_verified": bool(result.get("solved", False)),
                "error": result.get("error"),
                "elapsed_ns": result.get("elapsed_ns"),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    session_dirs = sorted(path.parent for path in args.sessions_dir.glob("*/session.json"))
    if not session_dirs:
        raise SystemExit(f"no session outputs found under {args.sessions_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for session_dir in session_dirs:
            for record in normalize_session(session_dir):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(json.dumps({"sessions": len(session_dirs), "records": count, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
