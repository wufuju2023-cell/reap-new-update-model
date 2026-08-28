#!/usr/bin/env python3
"""Real Lean / deterministic mock observer regression, never a GPU TTT test.

Run inside an observer-enabled CPU image with its cached runtime project:
python3 test_observer.py --project-dir /opt/reap-runtime --output-dir /workspace/out/new-run
The adjacent fixtures/ directory must be copied alongside this script.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import unittest


PROJECT: Path
OUTPUT: Path
FIXTURE = Path(__file__).with_name("fixtures") / "ObserverSmoke.lean"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class ObserverTests(unittest.TestCase):
    def run_case(self, name, *, enabled=True, ack=None, initial_version=0, timeout=15):
        root = OUTPUT / name
        root.mkdir()
        checkpoints = root / "acks"
        checkpoints.mkdir()
        trace = root / "observer.jsonl"
        env = os.environ.copy()
        for key in ("REAP_OBSERVER_PATH", "REAP_CHECKPOINT_DIR", "REAP_POLICY_VERSION",
                    "REAP_CHECKPOINT_TIMEOUT_SECONDS", "REAP_TREE_ID"):
            env.pop(key, None)
        env.update(REAP_SESSION_ID=name, OBSERVER_TEST_OUT=str(root),
                   OBSERVER_TEST_POISON="1", REAP_TREE_ID=name + "-tree")
        if enabled:
            env.update(REAP_OBSERVER_PATH=str(trace), REAP_CHECKPOINT_DIR=str(checkpoints),
                       REAP_POLICY_VERSION=str(initial_version),
                       REAP_CHECKPOINT_TIMEOUT_SECONDS=str(timeout))
        else:
            # Observer-off must not even parse unrelated observer-only settings.
            env["REAP_CHECKPOINT_TIMEOUT_SECONDS"] = "not-a-number"
        handled = set()
        live_frames = []
        lean_pids = set()
        stdout_path, stderr_path = root / "stdout.log", root / "stderr.log"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            proc = subprocess.Popen(["lake", "env", "lean", str(FIXTURE)], cwd=PROJECT,
                                    env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                    start_new_session=True)
            deadline = time.monotonic() + 60
            try:
                while proc.poll() is None:
                    self.assertLess(time.monotonic(), deadline, "fixture exceeded test protection")
                    rows = []
                    if trace.exists():
                        # Ignore only the final unterminated write; JSONL flush must finish it.
                        raw = trace.read_bytes()
                        rows = [json.loads(line) for line in raw.split(b"\n")[:-1] if line]
                    for frame in rows:
                        if frame["kind"] != "checkpoint" or frame["step"] in handled:
                            continue
                        step = frame["step"]
                        handled.add(step)
                        self.assertFalse((root / "result.json").exists(), "checkpoint was not live")
                        self.assertIsNone(proc.poll(), "Lean exited before ACK")
                        # /proc confirms the same actual Lean executable, not just its launcher.
                        current = set()
                        for path in Path("/proc").glob("[0-9]*/cmdline"):
                            try:
                                args = path.read_bytes().split(b"\0")
                            except (OSError, PermissionError):
                                continue
                            if args and Path(os.fsdecode(args[0])).name == "lean" and os.fsencode(FIXTURE) in args:
                                current.add(int(path.parent.name))
                        self.assertEqual(len(current), 1, "expected one live Lean fixture PID")
                        lean_pids.update(current)
                        live_frames.append(frame)
                        # Without ACK, no next iteration or final proof may appear.
                        time.sleep(0.06)
                        self.assertFalse((root / "result.json").exists())
                        if ack is not None:
                            payload = ack(frame)
                            temporary = checkpoints / f"checkpoint-{step:06d}.tmp"
                            target = checkpoints / f"checkpoint-{step:06d}.ack.json"
                            temporary.write_text(payload if isinstance(payload, str) else json.dumps(payload),
                                                 encoding="utf-8")
                            os.replace(temporary, target)
                    time.sleep(0.01)
            finally:
                if proc.poll() is None:
                    # Only the process group created above; no broad process-name kill.
                    os.killpg(proc.pid, 15)
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, 9)
                        proc.wait(timeout=3)
        rows = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        self.assertLessEqual(len(lean_pids), 1, "Lean was restarted across checkpoints")
        return root, proc.returncode, rows, live_frames, lean_pids

    @staticmethod
    def continuation(frame):
        return {"session_id": frame["session_id"], "step": frame["step"],
                "status": "continue", "policy_version": 1}

    def test_live_same_tree_and_observer_off_equivalence(self):
        off, code, rows, _, _ = self.run_case("observer-off", enabled=False)
        self.assertEqual(code, 0, (off / "stdout.log").read_text())
        self.assertFalse(rows)
        on, code, rows, frames, pids = self.run_case("observer-on", ack=self.continuation)
        self.assertEqual(code, 0, (on / "stdout.log").read_text())
        self.assertEqual(len(pids), 1)
        self.assertEqual([f["step"] for f in frames], [0, 1, 2])
        self.assertEqual([f["policy_version"] for f in frames], [0, 1, 1])
        self.assertEqual([f["root_is_solved"] for f in frames], [False, False, True])
        self.assertTrue(all(f["gamma"] == frames[0]["gamma"] for f in frames))
        self.assertEqual([r["sequence"] for r in rows], list(range(len(rows))))
        self.assertTrue(all(r["session_id"] == "observer-on" and r["tree_id"] == "observer-on-tree" for r in rows))
        self.assertEqual(read_json(off / "tree.json"), read_json(on / "tree.json"))
        self.assertEqual(read_json(off / "result.json"), read_json(on / "result.json"))
        self.assertTrue(read_json(on / "result.json")["solved"])
        generated = {(r["step"], r["node_index"], r["generation_index"], r["candidate_index"]): r
                     for r in rows if r["kind"] == "generation"}
        evaluated = [r for r in rows if r["kind"] == "eval"]
        self.assertTrue(evaluated)
        for event in evaluated:
            original = generated[(event["step"], event["node_index"], event["generation_index"], event["candidate_index"])]
            self.assertEqual(original["tactic"], event["tactic"])
            self.assertIn(original["goal_state"], original["prompt"])
            self.assertIn("ps", original)
            self.assertIn("raw_logprob", original)
            self.assertIn("eval_result", event)
        rejected = [r for r in evaluated if r["disposition"] == "eval_rejected"]
        self.assertTrue(rejected)
        self.assertTrue(all("error" in r["eval_result"] and r["child_index"] is None for r in rejected))
        self.assertTrue(any(r["kind"] == "backup" and r["edge"]["isFocus"] for r in rows))
        nodes = frames[-1]["tree"]["nodes"]
        self.assertTrue(any(n["data"]["toPlay"] == "AND" for n in nodes))
        self.assertTrue(all("valueSum" in n["data"] and "numVisit" in n["data"] for n in nodes))
        (OUTPUT / "live-evidence.json").write_text(json.dumps({
            "mock_only": True, "same_lean_pid": next(iter(pids)), "checkpoint_steps": [0, 1, 2],
            "observer_poisoned_goals_but_restored": True, "on_off_tree_and_proof_equal": True,
            "frames": frames,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def assert_rejected(self, name, mutate, *, initial_version=0):
        def bad_ack(frame):
            payload = self.continuation(frame)
            return mutate(payload)
        root, code, rows, frames, _ = self.run_case(name, ack=bad_ack, initial_version=initial_version)
        self.assertNotEqual(code, 0)
        self.assertFalse((root / "result.json").exists())
        self.assertEqual([f["step"] for f in frames], [0])
        self.assertFalse(any(r["kind"] == "checkpoint_ack" for r in rows))

    def test_wrong_session_fails_closed(self):
        self.assert_rejected("wrong-session", lambda p: {**p, "session_id": "other"})

    def test_wrong_step_fails_closed(self):
        self.assert_rejected("wrong-step", lambda p: {**p, "step": 1})

    def test_version_rollback_fails_closed(self):
        self.assert_rejected("version-rollback", lambda p: p, initial_version=2)

    def test_error_ack_fails_closed(self):
        self.assert_rejected("error-ack", lambda p: {**p, "status": "error", "error": "test rejection"})

    def test_malformed_ack_fails_closed(self):
        self.assert_rejected("malformed-ack", lambda _: "{not-json")

    def test_invalid_status_fails_closed(self):
        self.assert_rejected("invalid-status", lambda p: {**p, "status": "skip"})

    def test_timeout_fails_closed(self):
        root, code, rows, frames, _ = self.run_case("timeout", timeout=1)
        self.assertNotEqual(code, 0)
        self.assertFalse((root / "result.json").exists())
        self.assertEqual([f["step"] for f in frames], [0])
        self.assertIn("timed out", (root / "stdout.log").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    PROJECT, OUTPUT = args.project_dir.resolve(), args.output_dir.resolve()
    if os.name != "posix" or not Path("/proc").is_dir():
        parser.error("run inside the Linux CPU runtime; this test uses /proc")
    if OUTPUT.exists():
        parser.error("output directory must be new")
    OUTPUT.mkdir(parents=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ObserverTests))
    raise SystemExit(0 if result.wasSuccessful() else 1)
