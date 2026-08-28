from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cpu_runtime.segmented_ttt import (
    ActResult,
    LearnResult,
    PolicyVersionError,
    RolloutEvent,
    SegmentedTTTController,
    build_training_events,
)


def failed_rollout(event_id: str = "event-0") -> tuple[RolloutEvent, ...]:
    return (
        RolloutEvent(
            event_id=event_id,
            state="⊢ True",
            tactic="exact False.elim (by contradiction)",
            verdict="tacticErrorMessages",
            logprob_old=-2.5,
        ),
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class SegmentedTTTTests(unittest.TestCase):
    def make_controller(self, directory: str, act_runner, learn_client, clock=None):
        journal = Path(directory) / "journal.jsonl"
        controller = SegmentedTTTController(
            act_runner,
            learn_client,
            journal,
            clock=clock or FakeClock(),
        )
        return controller, journal

    def test_transport_timeout_is_not_a_total_ttt_deadline(self) -> None:
        def timeout(_request):
            raise TimeoutError("transport timeout")

        for phase in ("act", "learn"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                runner = timeout if phase == "act" else lambda request: ActResult(
                    root_verified=False, rollout=failed_rollout())
                controller, _ = self.make_controller(directory, runner, timeout)
                outcome = controller.run(session_id="s", theorem="True", max_segments=2)
                self.assertEqual(outcome.status, phase + "_timeout")
                self.assertFalse(outcome.root_verified)

    def test_first_segment_fails_second_succeeds_from_new_root(self) -> None:
        act_requests = []
        learn_requests = []

        def act_runner(request):
            act_requests.append(request)
            if request.segment_index == 0:
                return ActResult(root_verified=False, rollout=failed_rollout())
            return ActResult(root_verified=True, status="verified")

        def learn_client(request):
            learn_requests.append(request)
            return LearnResult(
                policy_version=request.policy_version + 1,
                event_ids=tuple(event.event_id for event in request.events),
            )

        with tempfile.TemporaryDirectory() as directory:
            controller, journal = self.make_controller(directory, act_runner, learn_client)
            outcome = controller.run(
                session_id="session-a",
                theorem="theorem demo : True",
                max_segments=3,
            )

            self.assertEqual(outcome.status, "solved")
            self.assertTrue(outcome.root_verified)
            self.assertEqual(outcome.segments_run, 2)
            self.assertEqual(outcome.learn_steps, 1)
            self.assertEqual([request.policy_version for request in act_requests], [0, 1])
            self.assertTrue(all(request.restart_from_root for request in act_requests))
            self.assertEqual(learn_requests[0].events[0].reward, -1.0)
            records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[-1]["kind"], "session_finished")
            self.assertEqual(records[-1]["status"], "solved")

    def test_successful_first_segment_does_not_learn(self) -> None:
        learn_calls = []

        def act_runner(_request):
            return ActResult(root_verified=True, status="verified")

        def learn_client(request):
            learn_calls.append(request)
            raise AssertionError("LEARN must not run after a verified root")

        with tempfile.TemporaryDirectory() as directory:
            controller, _journal = self.make_controller(directory, act_runner, learn_client)
            outcome = controller.run(session_id="s", theorem="theorem t : True", max_segments=2)

        self.assertEqual(outcome.status, "solved")
        self.assertEqual(outcome.learn_steps, 0)
        self.assertEqual(learn_calls, [])

    def test_verified_root_can_learn_then_require_fresh_root_recheck(self) -> None:
        act_versions = []
        learned_rewards = []

        def act_runner(request):
            act_versions.append(request.policy_version)
            return ActResult(
                root_verified=True,
                status="verified",
                rollout=(RolloutEvent(
                    state="⊢ True",
                    tactic="trivial",
                    verdict="root_verified",
                    terminal_verified=True,
                ),),
            )

        def learn_client(request):
            learned_rewards.extend((event.reward, event.root_verified) for event in request.events)
            return LearnResult(
                policy_version=request.policy_version + 1,
                event_ids=tuple(event.event_id for event in request.events),
            )

        with tempfile.TemporaryDirectory() as directory:
            controller, _journal = self.make_controller(directory, act_runner, learn_client)
            outcome = controller.run(
                session_id="s",
                theorem="theorem t : True",
                max_segments=2,
                learn_verified_proof=True,
            )

        self.assertEqual(outcome.status, "solved")
        self.assertEqual(outcome.learn_steps, 1)
        self.assertEqual(act_versions, [0, 1])
        self.assertEqual(learned_rewards, [(1.0, True)])

    def test_non_increasing_policy_version_is_rejected(self) -> None:
        def act_runner(_request):
            return ActResult(root_verified=False, rollout=failed_rollout())

        def learn_client(request):
            return LearnResult(
                policy_version=request.policy_version,
                event_ids=tuple(event.event_id for event in request.events),
            )

        with tempfile.TemporaryDirectory() as directory:
            controller, journal = self.make_controller(directory, act_runner, learn_client)
            with self.assertRaises(PolicyVersionError):
                controller.run(session_id="s", theorem="theorem t : True", max_segments=2)
            records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[-1]["kind"], "learn_rejected")
            self.assertEqual(records[-1]["reason"], "policy_version_not_increasing")

    def test_total_deadline_rejects_late_act_success(self) -> None:
        clock = FakeClock()
        learn_calls = []

        def act_runner(_request):
            clock.now = 2.0
            return ActResult(root_verified=True, status="verified_but_late")

        def learn_client(request):
            learn_calls.append(request)
            raise AssertionError("late ACT must not be learned")

        with tempfile.TemporaryDirectory() as directory:
            controller, _journal = self.make_controller(directory, act_runner, learn_client, clock)
            outcome = controller.run(
                session_id="s",
                theorem="theorem t : True",
                max_segments=2,
                deadline_seconds=1.0,
            )

        self.assertEqual(outcome.status, "deadline")
        self.assertFalse(outcome.root_verified)
        self.assertEqual(outcome.segments_run, 1)
        self.assertEqual(learn_calls, [])

    def test_max_segments_stops_without_trailing_learn(self) -> None:
        act_requests = []
        learn_requests = []

        def act_runner(request):
            act_requests.append(request)
            return ActResult(
                root_verified=False,
                rollout=failed_rollout(f"event-{request.segment_index}"),
            )

        def learn_client(request):
            learn_requests.append(request)
            return LearnResult(
                policy_version=request.policy_version + 1,
                event_ids=tuple(event.event_id for event in request.events),
            )

        with tempfile.TemporaryDirectory() as directory:
            controller, _journal = self.make_controller(directory, act_runner, learn_client)
            outcome = controller.run(session_id="s", theorem="theorem t : True", max_segments=2)

        self.assertEqual(outcome.status, "max_segments")
        self.assertEqual(outcome.segments_run, 2)
        self.assertEqual(outcome.learn_steps, 1)
        self.assertEqual(len(act_requests), 2)
        self.assertEqual(len(learn_requests), 1)

    def test_five_minutes_is_not_a_total_runtime_cap(self) -> None:
        for options in ({}, {"deadline_seconds": 900.0}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                clock = FakeClock()
                versions = []

                def act_runner(request):
                    versions.append(request.policy_version)
                    clock.now += 200
                    return ActResult(
                        root_verified=request.segment_index == 1,
                        rollout=failed_rollout(),
                    )

                def learn_client(request):
                    clock.now += 200
                    return LearnResult(
                        policy_version=request.policy_version + 1,
                        event_ids=tuple(event.event_id for event in request.events),
                    )

                controller, journal = self.make_controller(directory, act_runner, learn_client, clock)
                outcome = controller.run(session_id="s", theorem="True", max_segments=2, **options)
                self.assertTrue(outcome.root_verified)
                self.assertEqual(versions, [0, 1])
                records = [json.loads(line) for line in journal.read_text().splitlines()]
                self.assertEqual(records[0]["deadline_seconds"], options.get("deadline_seconds"))

    def test_invalid_explicit_deadlines_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller, _journal = self.make_controller(
                directory, lambda _request: ActResult(root_verified=True),
                lambda _request: LearnResult(policy_version=1, event_ids=()),
            )
            for deadline in (0, -1, float("nan"), float("inf")):
                with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                    controller.run(session_id="s", theorem="True", max_segments=1,
                                   deadline_seconds=deadline)

    def test_failed_and_timeout_events_never_have_positive_reward(self) -> None:
        events = build_training_events(
            session_id="s",
            theorem="theorem t : True",
            segment_index=0,
            policy_version=0,
            rollout=(
                RolloutEvent(state="s0", tactic="bad", verdict="error"),
                RolloutEvent(state="s0", tactic="slow", verdict="timeout"),
                RolloutEvent(state="s0", tactic="ok", verdict="accepted"),
            ),
        )
        self.assertEqual([event.reward for event in events], [-1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
