import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cpu_runtime.verified_trajectory import (
    TrajectoryRejected, checked_axioms, encode, export_verified, plan_success_path,
    sha, validate_historical_proof, validate_trace, load_verified_dataset,
)


def fixture(branch=False):
    def node(state, kind="OR", children=()):
        return {"data": {"state": state, "toPlay": kind, "isSolved": True,
                         "isPartial": False}, "children": list(children)}

    def edge(child, tactic="", focus=False, index=0):
        return {"childIndex": child, "edge": {"tacticStr": tactic,
                "isFocus": focus, "focusIndex": index}}

    if branch:
        nodes = [node(["root"], children=[edge(1, "constructor")]),
                 node(["left", "right"], "AND", [edge(2, focus=True), edge(4, focus=True, index=1)]),
                 node(["left"], children=[edge(3, "trivial")]), node([]),
                 node(["right"], children=[edge(5, "step")]),
                 node(["last"], children=[edge(6, "finish")]), node([])]
        for focused in nodes[2:]:
            focused["data"]["isPartial"] = True
    else:
        nodes = [node(["root"], children=[edge(1, "step")]),
                 node(["last"], children=[edge(2, "finish")]), node([])]
    session = {"session_id": "test", "tree_id": "test.tree0"}
    events = []

    def add(event):
        events.append({"schema_version": "reap.training.observer.v1", "session_id": "test",
                       "tree_id": "test.tree0", "sequence": len(events), "policy_version": 0, **event})

    for i, n in enumerate(nodes):
        for e in n["children"]:
            if e["edge"]["isFocus"]:
                continue
            base = {"node_index": i, "generation_index": 0, "candidate_index": 0,
                    "tactic": e["edge"]["tacticStr"]}
            add({**base, "kind": "generation", "prompt": "prompt", "state_key": json.dumps(n["data"]["state"]),
                 "goal_state": n["data"]["state"][0]})
            add({**base, "kind": "eval", "child_index": e["childIndex"], "disposition": "created",
                 "eval_result": {"ok": None}, "partial_goal": n["data"]["isPartial"]})
    add({"kind": "checkpoint", "step": 2, "root_is_solved": True,
         "tree": {"root_index": 0, "nodes": copy.deepcopy(nodes)}})
    add({"kind": "checkpoint_ack", "step": 2})
    return {"nodes": nodes, "solution": 0}, events, session


def sync_tree(tree, events):
    events[-2]["tree"]["nodes"] = copy.deepcopy(tree["nodes"])


class SuccessfulPathTests(unittest.TestCase):
    def test_linear_actual_action_count_and_no_search_value_target(self):
        candidate = plan_success_path(*fixture())
        self.assertEqual(candidate["proof_script"], "step\nfinish")
        self.assertEqual(candidate["root_return"], -2)
        self.assertEqual([r["return"] for r in candidate["rows"]], [-2, -1])
        self.assertNotIn("gamma", candidate)

    def test_and_uses_longest_branch_not_total_action_count(self):
        candidate = plan_success_path(*fixture(True))
        self.assertEqual(candidate["proof_script"], "constructor\n· trivial\n· step\n  finish")
        self.assertEqual(candidate["root_return"], -3)
        self.assertEqual([r["return"] for r in candidate["rows"]], [-3, -1, -2, -1])
        self.assertEqual(len(candidate["rows"]), 4)

    def test_and_missing_or_duplicate_focus_and_unsolved_branch_rejected(self):
        for mutate in (
            lambda t: t["nodes"][1]["children"].pop(),
            lambda t: t["nodes"][1]["children"][1]["edge"].update(focusIndex=0),
            lambda t: t["nodes"][4]["data"].update(isSolved=False),
        ):
            tree, events, session = fixture(True); mutate(tree); sync_tree(tree, events)
            with self.assertRaises(TrajectoryRejected):
                plan_success_path(tree, events, session)

    def test_is_solved_flag_without_empty_terminal_is_not_proof(self):
        tree, events, session = fixture()
        tree["nodes"][-1]["data"]["state"] = ["unproved"]
        sync_tree(tree, events)
        with self.assertRaisesRegex(TrajectoryRejected, "empty terminal"):
            plan_success_path(tree, events, session)

    def test_failed_guess_cannot_enter_success_rows(self):
        for patch_data in ({"disposition": "eval_rejected"}, {"eval_result": {"error": "failed"}},
                           {"partial_goal": True}, {"tactic": "guess"}):
            tree, events, session = fixture(); events[1].update(patch_data)
            with self.assertRaises(TrajectoryRejected):
                plan_success_path(tree, events, session)

    def test_wrong_state_version_identity_and_tree_rejected(self):
        for event_index, change in ((0, {"state_key": '["other"]'}), (0, {"goal_state": "other"}),
                                    (0, {"policy_version": 1}), (1, {"session_id": "other"}),
                                    (-1, {"policy_version": 1}), (-2, {"root_is_solved": False})):
            tree, events, session = fixture(); events[event_index].update(change)
            with self.assertRaises(TrajectoryRejected):
                plan_success_path(tree, events, session)

    def test_first_solved_edge_order_not_later_shorter_edge(self):
        tree, events, session = fixture()
        tree["nodes"][0]["children"].append({"childIndex": 2, "edge": {
            "isFocus": False, "focusIndex": 0, "tacticStr": "later direct proof"}})
        sync_tree(tree, events)
        self.assertEqual(plan_success_path(tree, events, session)["proof_script"], "step\nfinish")

    def test_cycle_partial_node_and_repeated_subtree_rejected(self):
        for mutate in (lambda t: t["nodes"][0]["children"][0].update(childIndex=0),
                       lambda t: t["nodes"][0]["data"].update(isPartial=True)):
            tree, events, session = fixture(); mutate(tree); sync_tree(tree, events)
            with self.assertRaises(TrajectoryRejected):
                plan_success_path(tree, events, session)

    def test_exact_trace_required_including_order_states_and_returns(self):
        c = plan_success_path(*fixture())
        trace = {"schema_version": "reap.verified-replay.trace.v1", "complete": True,
                 "root_return": c["root_return"], "rows": [{k: r[k] for k in
                 ("node_index", "state", "next_state", "tactic", "return")} for r in c["rows"]]}
        validate_trace(trace, c)
        with self.assertRaisesRegex(TrajectoryRejected, "declaration"):
            validate_trace(trace, c, "claimed_theorem")
        for mutate in (lambda t: t.update(complete=False), lambda t: t["rows"].reverse(),
                       lambda t: t["rows"][0].update(state=["wrong"]),
                       lambda t: t["rows"][0].update(**{"return": -99})):
            altered = copy.deepcopy(trace); mutate(altered)
            with self.assertRaises(TrajectoryRejected): validate_trace(altered, c)

    def test_axiom_whitelist_and_exact_theorem_binding(self):
        self.assertEqual(checked_axioms("'A.x' depends on axioms: [propext]\n", "A.x"), ["propext"])
        self.assertEqual(checked_axioms("'A.x' does not depend on any axioms\n", "A.x"), [])
        for stdout in ("'B.x' depends on axioms: [propext]\n", "'A.x' depends on axioms: [sorryAx]\n",
                       "'A.x' depends on axioms: [untrusted]\n", ""):
            with self.assertRaises(TrajectoryRejected): checked_axioms(stdout, "A.x")


class PublicationTests(unittest.TestCase):
    def setup_inputs(self, root):
        tree, events, session = fixture()
        source = b"import ReapRuntime\n\ntheorem x : True := by\n  reapTrainingMCTS\n"
        proof = b"import ReapRuntime\n\ntheorem x : True := by\n  step\n  finish\n\n#print axioms x\n"
        session["theorem_sha256"] = sha(source)
        result = {"schema_version": "reap.training.result.v1", "session_id": "test", "solved": True,
                  "status": "solved", "error": None, "proof_script": "step\nfinish"}
        receipt = {"proof_from_session": "test", "source_theorem_sha256": sha(source),
                   "generated_proof_sha256": sha(proof), "returncode": 0, "network": "none",
                   "image": "a" * 64, "stdout": "'x' depends on axioms: []\n", "stderr": ""}
        for name, data in {"source.lean": source, "proof.lean": proof,
                           "session.json": encode(session), "result.json": encode(result),
                           "raw_tree.json": encode(tree), "receipt.json": encode(receipt),
                           "observer.jsonl": b"\n".join(json.dumps(e).encode() for e in events),
                           "proof.stdout": receipt["stdout"].encode(), "proof.stderr": b"",
                           "module.lean": b"-- fixture module"}.items():
            (root / name).write_bytes(data)
        return dict(session_dir=root, source=root / "source.lean", proof=root / "proof.lean",
                    proof_receipt=root / "receipt.json", theorem="x", output=root / "out",
                    lean_project=root, replay_module=root / "module.lean")

    def test_invalid_historical_proof_rejected_before_process_or_output(self):
        with tempfile.TemporaryDirectory() as temp, patch("cpu_runtime.verified_trajectory.subprocess.run") as run:
            root = Path(temp); args = self.setup_inputs(root)
            (root / "proof.lean").write_bytes(b"different accepted proof")
            with self.assertRaises(TrajectoryRejected): export_verified(**args)
            run.assert_not_called(); self.assertFalse(args["output"].exists())

    def test_failed_lean_no_dataset_and_never_auto_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.setup_inputs(Path(temp))
            with patch("cpu_runtime.verified_trajectory.subprocess.run", return_value=
                       subprocess.CompletedProcess([], 1, b"failure", b"")) as run:
                with self.assertRaises(TrajectoryRejected): export_verified(**args)
                self.assertEqual(run.call_count, 1)
            self.assertFalse((args["output"] / "dataset.json").exists())
            self.assertTrue((args["output"] / "failed.json").exists())
            with patch("cpu_runtime.verified_trajectory.subprocess.run") as run:
                with self.assertRaisesRegex(TrajectoryRejected, "already exists"): export_verified(**args)
                run.assert_not_called()

    def test_receipt_success_without_actual_trace_cannot_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.setup_inputs(Path(temp))
            with patch("cpu_runtime.verified_trajectory.subprocess.run", return_value=
                       subprocess.CompletedProcess([], 0, b"'x' depends on axioms: []\n", b"")):
                with self.assertRaises(FileNotFoundError): export_verified(**args)
            self.assertFalse((args["output"] / "dataset.json").exists())

    def test_mock_complete_publication_loader_rechecks_every_binding_and_content_pin(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.setup_inputs(Path(temp))
            candidate = plan_success_path(*fixture())
            def run(*unused, **kwargs):
                trace = {"schema_version": "reap.verified-replay.trace.v1", "complete": True,
                         "theorem": "x", "root_return": -2, "rows": [{k: r[k] for k in
                         ("node_index", "state", "next_state", "tactic", "return")} for r in candidate["rows"]]}
                (args["output"] / "trace.json").write_bytes(encode(trace))
                return subprocess.CompletedProcess([], 0, b"'x' depends on axioms: []\n", b"")
            with patch("cpu_runtime.verified_trajectory.subprocess.run", side_effect=run):
                expected = export_verified(**args)
            output = args["output"]
            pin = sha((output / "dataset.json").read_bytes())
            self.assertEqual(load_verified_dataset(output, expected_sha256=pin), expected)
            with self.assertRaisesRegex(TrajectoryRejected, "content pin"):
                load_verified_dataset(output, expected_sha256="b" * 64)
            original = (output / "dataset.json").read_bytes()
            altered = copy.deepcopy(expected); altered["rows"][0]["return"] = -99
            (output / "dataset.json").write_bytes(encode(altered))
            with self.assertRaisesRegex(TrajectoryRejected, "target rows"):
                load_verified_dataset(output)
            (output / "dataset.json").write_bytes(original)
            for name in ("plan.json", "trace.json", "replay.stdout", "historical-proof-receipt.json"):
                content = (output / name).read_bytes(); (output / name).write_bytes(content + b" ")
                with self.assertRaises(TrajectoryRejected): load_verified_dataset(output)
                (output / name).write_bytes(content)


if __name__ == "__main__":
    unittest.main()
