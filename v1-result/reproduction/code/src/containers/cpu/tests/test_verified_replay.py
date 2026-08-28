"""Offline real Lean checks for the independent successful-path replayer."""
import argparse
import json
from pathlib import Path
import subprocess
import unittest

PROJECT = Path("/opt/reap-runtime")
MODULE = Path(__file__).resolve().parents[1] / "verified-replay/VerifiedReplay.lean"
OUTPUT = Path("/tmp/verified-replay-tests")


def node(index, state, kind="OR", tactic="", children=()):
    return {"node_index": index, "state": state, "kind": kind,
            "tactic": tactic, "children": list(children)}


class LeanVerifiedReplayTests(unittest.TestCase):
    def run_case(self, name, goal, nodes, success=True):
        directory = OUTPUT / name
        directory.mkdir(parents=True, exist_ok=False)
        plan = directory / "plan.json"
        trace = directory / "trace.json"
        plan.write_text(json.dumps({"schema_version": "reap.verified-replay.plan.v1", "nodes": nodes}), encoding="utf8")
        text = "import ReapRuntime\n" + MODULE.read_text(encoding="utf8")
        text += f'\ntheorem verifiedReplayFixture : {goal} := by\n  reapVerifiedReplay {json.dumps(str(plan))} {json.dumps(str(trace))}\n'
        text += "\n#print axioms verifiedReplayFixture\n"
        source = directory / "case.lean"
        source.write_text(text, encoding="utf8")
        completed = subprocess.run(["lake", "env", "lean", str(source)], cwd=PROJECT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (directory / "lean.log").write_bytes(completed.stdout)
        if success:
            self.assertEqual(completed.returncode, 0, completed.stdout.decode())
            self.assertNotIn(b"sorryAx", completed.stdout)
            return json.loads(trace.read_bytes())
        self.assertNotEqual(completed.returncode, 0, completed.stdout.decode())
        self.assertFalse(trace.exists(), "failed replay must not write a complete trace")
        return completed.stdout.decode()

    def test_real_linear_proof_has_exact_states_tactic_and_return(self):
        trace = self.run_case("linear", "True", [node(0, ["⊢ True"], tactic="exact True.intro", children=[1]),
                                                    node(1, [], "terminal")])
        self.assertEqual(trace["root_return"], -1)
        self.assertEqual(trace["theorem"], "verifiedReplayFixture")
        self.assertEqual(trace["rows"], [{"node_index": 0, "state": ["⊢ True"], "next_state": [],
                                          "tactic": "exact True.intro", "return": -1}])

    def test_real_and_longest_branch_preserves_each_focused_goal(self):
        left, right, final = "case left\n⊢ True", "case right\n⊢ True", "case right\nh : True\n⊢ True"
        nodes = [node(0, ["⊢ True ∧ True"], tactic="constructor", children=[1]),
                 node(1, [left, right], "AND", children=[2, 4]),
                 node(2, [left], tactic="exact True.intro", children=[3]), node(3, [], "terminal"),
                 node(4, [right], tactic="have h : True := True.intro", children=[5]),
                 node(5, [final], tactic="exact h", children=[6]), node(6, [], "terminal")]
        trace = self.run_case("and", "True ∧ True", nodes)
        self.assertEqual(trace["root_return"], -3)
        self.assertEqual([row["return"] for row in trace["rows"]], [-3, -1, -2, -1])

    def test_saved_state_mismatch_is_not_relabelled(self):
        output = self.run_case("wrong-state", "True", [node(0, ["⊢ False"], tactic="trivial", children=[1]),
                                                         node(1, [], "terminal")], False)
        self.assertIn("state mismatch", output)

    def test_failed_tactic_cannot_write_success_trace(self):
        output = self.run_case("bad-tactic", "False", [node(0, ["⊢ False"], tactic="exact True.intro", children=[1]),
                                                          node(1, [], "terminal")], False)
        self.assertIn("tactic failed", output)

    def test_unproved_terminal_flag_does_not_close_goal(self):
        output = self.run_case("false-terminal", "True", [node(0, ["⊢ True"], "terminal")], False)
        self.assertIn("nonterminal leaf", output)

    def test_dependent_and_is_rejected_before_any_branch_replay(self):
        directory = OUTPUT / "dependent-and"
        directory.mkdir(parents=True, exist_ok=False)
        source = directory / "case.lean"
        text = "import ReapRuntime\n" + MODULE.read_text(encoding="utf8")
        text += '''
open Reap.VerifiedReplay
theorem verifiedReplayDependentFixture : ∃ n : Nat, n = 0 := by
  refine ⟨?_, ?_⟩
  run_tac do
    let before ← actualState
    let plan : Array ReplayNode := #[{
      node_index := 0, state := before, kind := "AND", tactic := "", children := #[1, 2]
    }]
    let ctx ← mkProofCheckContext
    let rejected ← try
      discard <| replay ctx plan 0 3
      pure false
    catch error =>
      unless (← error.toMessageData.toString).contains "dependent/metavariable" do throw error
      pure true
    unless rejected do throwError "dependent AND was accepted"
    unless (← actualState) == before do throwError "rejected AND changed goals"
  all_goals first | exact 0 | rfl
#print axioms verifiedReplayDependentFixture
'''
        source.write_text(text, encoding="utf8")
        completed = subprocess.run(["lake", "env", "lean", str(source)], cwd=PROJECT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (directory / "lean.log").write_bytes(completed.stdout)
        self.assertEqual(completed.returncode, 0, completed.stdout.decode())
        self.assertNotIn(b"sorryAx", completed.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=PROJECT)
    args, remaining = parser.parse_known_args()
    OUTPUT, PROJECT = args.output_dir, args.project_dir
    unittest.main(argv=[__file__, *remaining])
