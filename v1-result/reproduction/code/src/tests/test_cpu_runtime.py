from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from cpu_runtime.batch_solver import _finalize_solver_status, load_manifest, parse_spec, run_process
from cpu_runtime.normalize_rollout import (
    evaluation_records,
    generated_logprobs,
    strip_thinking_prefix,
    tree_records,
)
from cpu_runtime.reap_prompt import logged_prompts, reap_prompt
from cpu_runtime.segmented_ttt import RolloutEvent, build_training_events


class ManifestTests(unittest.TestCase):
    def test_valid_manifest_and_unique_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "session_id": "s-1",
                        "theorem_file": "Smoke.lean",
                        "policy_base_url": "http://gpu/sessions/s-1/policy/v1/",
                        "value_base_url": "http://gpu/sessions/s-1/value/v1/",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            specs = load_manifest(path)
            self.assertEqual(specs[0].session_id, "s-1")
            self.assertFalse(specs[0].policy_base_url.endswith("/"))

    def test_invalid_session_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid session_id"):
            parse_spec(
                {
                    "session_id": "../escape",
                    "theorem_file": "x.lean",
                    "policy_base_url": "http://gpu/policy/v1",
                    "value_base_url": "http://gpu/value/v1",
                }
            )

    def test_duplicate_session_id_is_rejected(self) -> None:
        item = {
            "session_id": "same",
            "theorem_file": "x.lean",
            "policy_base_url": "http://gpu/policy/v1",
            "value_base_url": "http://gpu/value/v1",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.jsonl"
            path.write_text(json.dumps(item) + "\n" + json.dumps(item) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate session_id"):
                load_manifest(path)


class ProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_timeout(self) -> None:
        command = [sys.executable, "-c", "import time; time.sleep(2)"]
        code, timed_out, _stdout, _stderr = await run_process(
            command, Path.cwd(), dict(**__import__("os").environ), timeout=0.05
        )
        self.assertEqual(code, 124)
        self.assertTrue(timed_out)

    async def test_timeout_result_cannot_be_solved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_path = root / "result.json"
            result_path.write_text(
                json.dumps({"solved": True, "status": "solved"}), encoding="utf-8"
            )
            solved, status = _finalize_solver_status(result_path, 124, True)
            self.assertFalse(solved)
            self.assertEqual(status, "timeout")

    async def test_two_processes_really_overlap(self) -> None:
        command = [sys.executable, "-c", "import time; time.sleep(0.25)"]
        started = time.perf_counter()
        results = await asyncio.gather(
            run_process(command, Path.cwd(), dict(**__import__("os").environ), timeout=2),
            run_process(command, Path.cwd(), dict(**__import__("os").environ), timeout=2),
        )
        elapsed = time.perf_counter() - started
        self.assertTrue(all(result[0] == 0 for result in results))
        self.assertLess(elapsed, 0.48)


class NormalizationTests(unittest.TestCase):
    def test_training_preserves_reap_prompt_and_premises(self) -> None:
        state = "⊢ True"
        premises = [{"formal_name": "True.intro", "formal_statement": "True"}]
        prompt = logged_prompts([
            {"name": "tactic_gen", "extra": {"goal": state, "ps": premises}}
        ])[state]
        self.assertEqual(prompt,
            "User: Please generate a tactic in lean4 to solve the state.\n"
            "Here're some theorems that may be helpful:\n"
            "Formal name: True.intro\nFormal statement: True\n"
            "STATE:\n⊢ True\nTACTIC:\n\nAssistant:")
        training = build_training_events(
            session_id="s", theorem="True", segment_index=0, policy_version=0,
            rollout=[RolloutEvent(state, "trivial", "accepted", metadata={"prompt": prompt})],
        )
        self.assertEqual(training[0].prompt, prompt)
        self.assertIn("helpful:\n\nSTATE:", reap_prompt(state))

    def test_ambiguous_logged_prompt_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            logged_prompts([
                {"name": "tactic_gen", "extra": {"goal": "⊢ True", "ps": []}},
                {"name": "tactic_gen", "extra": {"goal": "⊢ True", "ps": [
                    {"formal_name": "True.intro", "formal_statement": "True"}]}},
            ])

    def test_logprob_and_tree_fields(self) -> None:
        events = [
            {
                "name": "tactic_gen",
                "extra": {
                    "goal": "⊢ True",
                    "result": {
                        "choices": [
                            {
                                "message": {"content": "<think>easy</think>\ntrivial"},
                                "logprobs": {"content": [{"logprob": -0.2}, {"logprob": -0.3}]},
                            }
                        ]
                    },
                },
            }
        ]
        self.assertAlmostEqual(generated_logprobs(events)[("⊢ True", "trivial")], -0.5)
        tree = {
            "nodes": [
                {
                    "data": {"state": ["⊢ True"], "isSolved": True},
                    "children": [
                        {
                            "edge": {"tacticStr": "trivial", "probability": 0.5, "value": 1.0, "numVisit": 2},
                            "childIndex": 1,
                            "extra": {"Q": 1.0},
                        }
                    ],
                },
                {"data": {"state": [], "isSolved": True}, "children": []},
            ]
        }
        record = tree_records("s", tree)[0]
        self.assertTrue(record["visited"])
        self.assertTrue(record["solved_child"])
        self.assertIsNone(record["logprob"])
        self.assertEqual(record["prior_weight"], 0.5)

    def test_accepted_tactic_is_not_positive_reward(self) -> None:
        records = evaluation_records(
            "s",
            [{"name": "tactic_eval", "extra": {"state": "⊢ True", "tactic": "trivial", "result": {"ok": None}}}],
            {("⊢ True", "trivial"): -0.1},
        )
        self.assertEqual(records[0]["reward"], 0.0)
        self.assertFalse(records[0]["terminal_verified"])

    def test_strip_thinking_prefix(self) -> None:
        self.assertEqual(strip_thinking_prefix("<think>x</think>  exact h"), "exact h")


if __name__ == "__main__":
    unittest.main()
