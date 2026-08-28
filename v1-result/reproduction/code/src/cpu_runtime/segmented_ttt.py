#!/usr/bin/env python3
"""Segmented test-time training orchestration.

The controller deliberately treats every ACT segment as a new search from the
theorem root.  A policy update may therefore happen only between two trees;
one MCTS tree never contains nodes produced by different policy versions.

This module contains no network or Reap-specific process code.  Callers inject
an ``act_runner`` and a ``learn_client``, which makes the state machine usable
with local processes, HTTP clients, or deterministic tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Protocol


SCHEMA_VERSION = "reap.training.segmented_ttt.v1"


class SegmentedTTTError(RuntimeError):
    """Base class for controller contract violations."""


class PolicyVersionError(SegmentedTTTError):
    """Raised when LEARN does not return a strictly newer policy version."""


class EventAcknowledgementError(SegmentedTTTError):
    """Raised when LEARN does not acknowledge exactly the submitted events."""


@dataclass(frozen=True)
class RolloutEvent:
    state: str
    tactic: str
    verdict: str
    logprob_old: float | None = None
    next_state: str | None = None
    event_id: str | None = None
    terminal_verified: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActRequest:
    session_id: str
    theorem: str
    segment_index: int
    policy_version: int
    deadline_monotonic: float | None
    restart_from_root: bool = True


@dataclass(frozen=True)
class ActResult:
    root_verified: bool
    rollout: tuple[RolloutEvent, ...] = ()
    status: str = "completed"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingEvent:
    event_id: str
    session_id: str
    theorem: str
    segment_index: int
    policy_version: int
    state: str
    tactic: str
    verdict: str
    reward: float
    logprob_old: float | None
    next_state: str | None
    root_verified: bool = False
    prompt: str | None = None


@dataclass(frozen=True)
class LearnRequest:
    session_id: str
    theorem: str
    segment_index: int
    policy_version: int
    deadline_monotonic: float | None
    events: tuple[TrainingEvent, ...]


@dataclass(frozen=True)
class LearnResult:
    policy_version: int
    event_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionOutcome:
    session_id: str
    theorem: str
    status: str
    root_verified: bool
    policy_version: int
    segments_run: int
    learn_steps: int


class ActRunner(Protocol):
    def __call__(self, request: ActRequest) -> ActResult: ...


class LearnClient(Protocol):
    def __call__(self, request: LearnRequest) -> LearnResult: ...


_NEGATIVE_VERDICTS = {
    "parse",
    "parseerror",
    "forbidden",
    "forbiddentactic",
    "exception",
    "tacticexception",
    "tactictimeout",
    "errormsg",
    "tacticerrormessages",
    "unassignedgoal",
    "assignedproofhasmvarorsorry",
    "auxproofhasmvarorsorry",
    "auxproofkernelcheckfailed",
    "finalproofcheckfailed",
    "rejected",
    "error",
    "failed",
}


def _normalized_verdict(verdict: str) -> str:
    return "".join(character for character in verdict.lower() if character.isalnum())


def _training_reward(verdict: str) -> float:
    """Map a non-terminal rollout verdict to a safe immediate reward.

    A segment reaches LEARN only when its root was *not* verified.  Consequently
    no event in that segment may receive a positive reward.  Explicit verifier
    model-attributable verifier/tactic failures receive -1 and all other
    events remain neutral. A generic infrastructure timeout is deliberately
    neutral; only an explicit tacticTimeout is attributable to the candidate.
    """

    return -1.0 if _normalized_verdict(verdict) in _NEGATIVE_VERDICTS else 0.0


def build_training_events(
    *,
    session_id: str,
    theorem: str,
    segment_index: int,
    policy_version: int,
    rollout: Iterable[RolloutEvent],
) -> tuple[TrainingEvent, ...]:
    events: list[TrainingEvent] = []
    seen_ids: set[str] = set()
    for event_index, event in enumerate(rollout):
        event_id = event.event_id or f"{session_id}.{segment_index}.{event_index}"
        if event_id in seen_ids:
            raise ValueError(f"duplicate rollout event_id: {event_id}")
        seen_ids.add(event_id)
        if event.terminal_verified and _normalized_verdict(event.verdict) != "rootverified":
            raise ValueError("terminal_verified event must use root_verified verdict")
        events.append(
            TrainingEvent(
                event_id=event_id,
                session_id=session_id,
                theorem=theorem,
                segment_index=segment_index,
                policy_version=policy_version,
                state=event.state,
                tactic=event.tactic,
                verdict=event.verdict,
                reward=1.0 if event.terminal_verified else _training_reward(event.verdict),
                logprob_old=event.logprob_old,
                next_state=event.next_state,
                root_verified=event.terminal_verified,
                prompt=event.metadata.get("prompt"),
            )
        )
    return tuple(events)


class JsonlJournal:
    """Append-only structured journal with flush-on-record semantics."""

    def __init__(self, path: Path, clock: Callable[[], float]) -> None:
        self.path = path
        self.clock = clock

    def append(self, kind: str, **payload: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "monotonic_seconds": self.clock(),
            **payload,
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


class SegmentedTTTController:
    def __init__(
        self,
        act_runner: ActRunner,
        learn_client: LearnClient,
        journal_path: Path,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.act_runner = act_runner
        self.learn_client = learn_client
        self.clock = clock
        self.journal = JsonlJournal(journal_path, clock)

    def _outcome(
        self,
        *,
        session_id: str,
        theorem: str,
        status: str,
        root_verified: bool,
        policy_version: int,
        segments_run: int,
        learn_steps: int,
    ) -> SessionOutcome:
        outcome = SessionOutcome(
            session_id=session_id,
            theorem=theorem,
            status=status,
            root_verified=root_verified,
            policy_version=policy_version,
            segments_run=segments_run,
            learn_steps=learn_steps,
        )
        self.journal.append("session_finished", **asdict(outcome))
        return outcome

    def run(
        self,
        *,
        session_id: str,
        theorem: str,
        initial_policy_version: int = 0,
        max_segments: int,
        deadline_seconds: float | None = None,
        learn_verified_proof: bool = False,
    ) -> SessionOutcome:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if not theorem:
            raise ValueError("theorem must be non-empty")
        if initial_policy_version < 0:
            raise ValueError("initial_policy_version must be >= 0")
        if max_segments < 1:
            raise ValueError("max_segments must be >= 1")
        if deadline_seconds is not None and (
            not math.isfinite(deadline_seconds) or deadline_seconds <= 0
        ):
            raise ValueError("deadline_seconds must be finite and positive, or None")

        # Five minutes is a performance preference, not a proof acceptance gate.
        # Callers may still opt into a total runtime budget explicitly.
        deadline = None if deadline_seconds is None else self.clock() + deadline_seconds
        policy_version = initial_policy_version
        segments_run = 0
        learn_steps = 0
        self.journal.append(
            "session_started",
            session_id=session_id,
            theorem=theorem,
            max_segments=max_segments,
            deadline_seconds=deadline_seconds,
            initial_policy_version=initial_policy_version,
        )

        for segment_index in range(max_segments):
            if deadline is not None and self.clock() >= deadline:
                return self._outcome(
                    session_id=session_id,
                    theorem=theorem,
                    status="deadline",
                    root_verified=False,
                    policy_version=policy_version,
                    segments_run=segments_run,
                    learn_steps=learn_steps,
                )

            act_request = ActRequest(
                session_id=session_id,
                theorem=theorem,
                segment_index=segment_index,
                policy_version=policy_version,
                deadline_monotonic=deadline,
            )
            self.journal.append("act_started", **asdict(act_request))
            try:
                act_result = self.act_runner(act_request)
            except TimeoutError as exc:
                self.journal.append(
                    "act_timeout",
                    session_id=session_id,
                    theorem=theorem,
                    segment_index=segment_index,
                    policy_version=policy_version,
                    error=str(exc),
                )
                return self._outcome(
                    session_id=session_id,
                    theorem=theorem,
                    status="deadline" if deadline is not None and self.clock() >= deadline else "act_timeout",
                    root_verified=False,
                    policy_version=policy_version,
                    segments_run=segments_run,
                    learn_steps=learn_steps,
                )
            except Exception as exc:
                self.journal.append(
                    "act_failed",
                    session_id=session_id,
                    theorem=theorem,
                    segment_index=segment_index,
                    policy_version=policy_version,
                    error=repr(exc),
                )
                return self._outcome(
                    session_id=session_id,
                    theorem=theorem,
                    status="act_failed",
                    root_verified=False,
                    policy_version=policy_version,
                    segments_run=segments_run,
                    learn_steps=learn_steps,
                )

            if not isinstance(act_result, ActResult):
                raise TypeError("act_runner must return ActResult")
            segments_run += 1
            self.journal.append(
                "act_finished",
                session_id=session_id,
                theorem=theorem,
                segment_index=segment_index,
                policy_version=policy_version,
                root_verified=act_result.root_verified,
                status=act_result.status,
                rollout_events=len(act_result.rollout),
                metadata=dict(act_result.metadata),
            )

            # A result that arrives after the total deadline is not accepted as
            # a successful proof and never becomes a positive training signal.
            if deadline is not None and self.clock() >= deadline:
                return self._outcome(
                    session_id=session_id,
                    theorem=theorem,
                    status="deadline",
                    root_verified=False,
                    policy_version=policy_version,
                    segments_run=segments_run,
                    learn_steps=learn_steps,
                )

            has_next_segment = segment_index + 1 < max_segments
            if act_result.root_verified and (not learn_verified_proof or not has_next_segment):
                return self._outcome(
                    session_id=session_id,
                    theorem=theorem,
                    status="solved",
                    root_verified=True,
                    policy_version=policy_version,
                    segments_run=segments_run,
                    learn_steps=learn_steps,
                )

            # No update is useful after the final allowed ACT segment because
            # there is no subsequent fresh-root segment that could consume it.
            if not has_next_segment:
                break

            training_rollout = act_result.rollout
            if act_result.root_verified:
                training_rollout = tuple(event for event in act_result.rollout if event.terminal_verified)
            events = build_training_events(
                session_id=session_id,
                theorem=theorem,
                segment_index=segment_index,
                policy_version=policy_version,
                rollout=training_rollout,
            )
            if not events:
                return self._outcome(
                    session_id=session_id,
                    theorem=theorem,
                    status="no_training_events",
                    root_verified=False,
                    policy_version=policy_version,
                    segments_run=segments_run,
                    learn_steps=learn_steps,
                )

            learn_request = LearnRequest(
                session_id=session_id,
                theorem=theorem,
                segment_index=segment_index,
                policy_version=policy_version,
                deadline_monotonic=deadline,
                events=events,
            )
            self.journal.append(
                "learn_started",
                session_id=session_id,
                theorem=theorem,
                segment_index=segment_index,
                policy_version=policy_version,
                event_ids=[event.event_id for event in events],
                rewards=[event.reward for event in events],
            )
            try:
                learn_result = self.learn_client(learn_request)
            except TimeoutError as exc:
                self.journal.append(
                    "learn_timeout",
                    session_id=session_id,
                    theorem=theorem,
                    segment_index=segment_index,
                    policy_version=policy_version,
                    error=str(exc),
                )
                return self._outcome(
                    session_id=session_id,
                    theorem=theorem,
                    status="deadline" if deadline is not None and self.clock() >= deadline else "learn_timeout",
                    root_verified=False,
                    policy_version=policy_version,
                    segments_run=segments_run,
                    learn_steps=learn_steps,
                )
            except Exception as exc:
                self.journal.append(
                    "learn_failed",
                    session_id=session_id,
                    theorem=theorem,
                    segment_index=segment_index,
                    policy_version=policy_version,
                    error=repr(exc),
                )
                return self._outcome(
                    session_id=session_id,
                    theorem=theorem,
                    status="learn_failed",
                    root_verified=False,
                    policy_version=policy_version,
                    segments_run=segments_run,
                    learn_steps=learn_steps,
                )
            if not isinstance(learn_result, LearnResult):
                raise TypeError("learn_client must return LearnResult")

            if deadline is not None and self.clock() >= deadline:
                return self._outcome(
                    session_id=session_id,
                    theorem=theorem,
                    status="deadline",
                    root_verified=False,
                    policy_version=policy_version,
                    segments_run=segments_run,
                    learn_steps=learn_steps,
                )

            submitted_ids = tuple(event.event_id for event in events)
            acknowledged_ids = learn_result.event_ids
            if len(set(acknowledged_ids)) != len(acknowledged_ids) or set(acknowledged_ids) != set(submitted_ids):
                self.journal.append(
                    "learn_rejected",
                    session_id=session_id,
                    theorem=theorem,
                    segment_index=segment_index,
                    policy_version=policy_version,
                    reason="event_acknowledgement_mismatch",
                    submitted_event_ids=list(submitted_ids),
                    acknowledged_event_ids=list(acknowledged_ids),
                )
                raise EventAcknowledgementError("LEARN must acknowledge every submitted event_id exactly once")

            if learn_result.policy_version <= policy_version:
                self.journal.append(
                    "learn_rejected",
                    session_id=session_id,
                    theorem=theorem,
                    segment_index=segment_index,
                    policy_version=policy_version,
                    returned_policy_version=learn_result.policy_version,
                    reason="policy_version_not_increasing",
                )
                raise PolicyVersionError(
                    f"LEARN policy_version must increase: current={policy_version}, "
                    f"returned={learn_result.policy_version}"
                )

            previous_version = policy_version
            policy_version = learn_result.policy_version
            learn_steps += 1
            self.journal.append(
                "learn_finished",
                session_id=session_id,
                theorem=theorem,
                segment_index=segment_index,
                previous_policy_version=previous_version,
                policy_version=policy_version,
                event_ids=list(acknowledged_ids),
                metadata=dict(learn_result.metadata),
            )

        return self._outcome(
            session_id=session_id,
            theorem=theorem,
            status="max_segments",
            root_verified=False,
            policy_version=policy_version,
            segments_run=segments_run,
            learn_steps=learn_steps,
        )


__all__ = [
    "ActRequest",
    "ActResult",
    "EventAcknowledgementError",
    "LearnRequest",
    "LearnResult",
    "PolicyVersionError",
    "RolloutEvent",
    "SegmentedTTTController",
    "SessionOutcome",
    "TrainingEvent",
    "build_training_events",
]
