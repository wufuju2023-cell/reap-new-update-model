"""Reconstruct the pinned Reap Generator.mkPrompt from its logged inputs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def reap_prompt(state: str, premises: Iterable[Mapping[str, Any]] = ()) -> str:
    related = "\n".join(
        "Formal name: " + premise["formal_name"]
        + "\nFormal statement: " + premise["formal_statement"]
        for premise in premises
    )
    return (
        "User: Please generate a tactic in lean4 to solve the state.\n"
        "Here're some theorems that may be helpful:\n"
        + related + "\nSTATE:\n" + state + "\nTACTIC:\n\nAssistant:"
    )


def logged_prompts(events: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Fail rather than silently choose one of multiple prompts for a state."""
    prompts: dict[str, str] = {}
    for event in events:
        if event.get("name") != "tactic_gen":
            continue
        extra = event.get("extra") or {}
        state = str(extra.get("goal", ""))
        prompt = reap_prompt(state, extra.get("ps") or [])
        if state in prompts and prompts[state] != prompt:
            raise ValueError("ambiguous generation prompt for a repeated Lean state")
        prompts[state] = prompt
    return prompts
