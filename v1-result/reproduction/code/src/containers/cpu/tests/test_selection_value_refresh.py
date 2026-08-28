#!/usr/bin/env python3
"""Real local Lean + mock values; no network model, GPU or learning evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import test_observer as protocol


class SelectionValueRefreshTests(unittest.TestCase):
    def run_case(self, name, *, mode="selection-value-refresh", value_error=False, **kwargs):
        with patch.dict(os.environ, {"REAP_SELECTION_VALUE_REFRESH": mode,
                                    "OBSERVER_TEST_VALUE_ERROR": "1" if value_error else ""}):
            return protocol.ObserverTests.run_case(self, name, **kwargs)

    @staticmethod
    def continue_at(version):
        return lambda frame: {"session_id": frame["session_id"], "step": frame["step"],
                              "status": "continue", "policy_version": version}

    def test_real_lean_cache_statistics_failure_and_version_unit_contract(self):
        fixture = protocol.FIXTURE.with_name("SelectionValueRefreshUnit.lean")
        result = subprocess.run(["lake", "env", "lean", str(fixture)], cwd=protocol.PROJECT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        (protocol.OUTPUT / "unit.log").write_bytes(result.stdout)
        self.assertEqual(result.returncode, 0, result.stdout.decode())
        self.assertIn(b"SELECTION_VALUE_REFRESH_UNIT_OK", result.stdout)

    def test_explicit_off_keeps_original_tree_and_proof(self):
        off, code, _, _, _ = self.run_case("refresh-off", enabled=False, mode="")
        self.assertEqual(code, 0, (off / "stdout.log").read_text())
        unchanged, code, rows, _, _ = self.run_case("refresh-same-version", ack=self.continue_at(0))
        self.assertEqual(code, 0, (unchanged / "stdout.log").read_text())
        self.assertFalse(any(row["kind"] == "selection_value_refresh" for row in rows))
        self.assertEqual(protocol.read_json(off / "tree.json"), protocol.read_json(unchanged / "tree.json"))
        self.assertEqual(protocol.read_json(off / "result.json"), protocol.read_json(unchanged / "result.json"))

    def test_ack_version_advances_cache_before_next_selection_and_kernel_proof_survives(self):
        output, code, rows, frames, pids = self.run_case("refresh-on", ack=self.continue_at(1))
        self.assertEqual(code, 0, (output / "stdout.log").read_text())
        self.assertEqual(len(pids), 1)
        refreshes = [row for row in rows if row["kind"] == "selection_value_refresh"]
        self.assertEqual(len(refreshes), 1)
        refresh = refreshes[0]
        self.assertEqual((refresh["policy_version"], refresh["selection_value_version"]), (1, 1))
        self.assertEqual(refresh["scope"], "selection-only-not-training-Q")
        ack = next(row for row in rows if row["kind"] == "checkpoint_ack")
        next_selection = next(row for row in rows if row["kind"] == "selection" and row["step"] == 1)
        self.assertLess(ack["sequence"], refresh["sequence"])
        self.assertLess(refresh["sequence"], next_selection["sequence"])
        self.assertTrue(protocol.read_json(output / "result.json")["solved"])
        self.assertNotIn("sorryAx", (output / "stdout.log").read_text())
        for node in refresh["nodes"]:
            old = frames[0]["tree"]["nodes"][node["node_index"]]["data"]
            self.assertFalse(old["isSolved"])
            self.assertGreater(old["numVisit"], 0)
        (protocol.OUTPUT / "live-evidence.json").write_text(json.dumps({
            "scope": "local Lean with deterministic mock values; no TTT/GPU", "same_lean_pid": next(iter(pids)),
            "checkpoint_steps": [frame["step"] for frame in frames], "refresh": refresh,
            "proof": protocol.read_json(output / "result.json")}, indent=2), encoding="utf-8")

    def test_failed_value_does_not_continue_under_old_cache(self):
        output, code, rows, frames, _ = self.run_case("refresh-failed-value", value_error=True,
                                                    ack=self.continue_at(1))
        self.assertNotEqual(code, 0)
        self.assertEqual([frame["step"] for frame in frames], [0])
        self.assertFalse(any(row["kind"] == "selection_value_refresh" for row in rows))
        self.assertFalse(any(row["kind"] == "selection" and row["step"] > 0 for row in rows))
        self.assertFalse((output / "result.json").exists())

    def test_failed_ack_and_version_jump_never_refresh_or_continue(self):
        for name, change in (("error", {"status": "error", "error": "rejected update"}),
                              ("jump", {"policy_version": 2})):
            with self.subTest(name=name):
                ack = lambda frame: {**self.continue_at(1)(frame), **change}
                _, code, rows, frames, _ = self.run_case("refresh-" + name, ack=ack)
                self.assertNotEqual(code, 0)
                self.assertEqual([frame["step"] for frame in frames], [0])
                self.assertFalse(any(row["kind"] == "selection_value_refresh" for row in rows))
                self.assertFalse(any(row["kind"] == "selection" and row["step"] > 0 for row in rows))

    def test_invalid_mode_or_missing_barrier_is_rejected(self):
        for name, mode, enabled in (("invalid-mode", "automatic", True),
                                     ("missing-barrier", "selection-value-refresh", False)):
            with self.subTest(name=name):
                _, code, rows, frames, _ = self.run_case("refresh-" + name, mode=mode, enabled=enabled)
                self.assertNotEqual(code, 0)
                self.assertFalse(rows or frames)

    def test_solved_final_checkpoint_still_rejects_a_jumped_ack_without_value_queries(self):
        def ack(frame):
            return self.continue_at(2 if frame["root_is_solved"] else 0)(frame)
        output, code, rows, frames, _ = self.run_case("refresh-terminal-jump", ack=ack)
        self.assertNotEqual(code, 0)
        self.assertTrue(frames[-1]["root_is_solved"])
        self.assertFalse((output / "result.json").exists())
        self.assertFalse(any(row["kind"] == "selection_value_refresh" for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path("/opt/reap-runtime"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol.PROJECT, protocol.OUTPUT = args.project_dir, args.output_dir
    protocol.OUTPUT.mkdir(parents=True, exist_ok=False)
    protocol.FIXTURE = Path(__file__).with_name("fixtures") / "SelectionValueRefreshSmoke.lean"
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelectionValueRefreshTests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
