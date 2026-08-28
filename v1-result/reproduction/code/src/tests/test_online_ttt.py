import json
import hashlib
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cpu_runtime.online_ttt import JsonlTail, OnlineCoordinator, atomic_new_json
from cpu_runtime.online_ttt import run_online, validate_created


class OnlineCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.calls = []

        def learn(version, event):
            self.calls.append(event)
            return {"event_id": event["event_id"], "policy_version": version + 1,
                    "applied": True, "idempotent": False, "detail": {
                        "objective": "search_visit_backup", "optimizer_steps": version + 1,
                        **{k: True for k in ("finite_loss", "finite_gradients", "finite_parameters", "finite_optimizer_state", "base_parameters_frozen")},
                        **{k: 0.1 for k in ("loss", "policy_loss", "kl", "value_loss", "grad_norm", "value_target")},
                        "search_trace": {k: event[k] for k in ("tree_id", "step", "node_index", "policy_version", "gamma", "terminal_verified")},
                        "prompt_sha256": hashlib.sha256(event["prompt"].encode()).hexdigest(),
                        "parameter_diffs": {k: {"before_sha256": "0" * 64, "after_sha256": "1" * 64, "changed_tensors": 1}
                            for k in ("adapter", "value_head", "optimizer")}}}
        self.coordinator = OnlineCoordinator("unit-a", "unit-a.tree0", self.root / "acks", learn, gamma=0.99)
        self.sequence = 0

    def tearDown(self):
        self.temp.cleanup()

    def event(self, kind, **fields):
        value = {"schema_version": "reap.training.observer.v1", "session_id": "unit-a",
                 "tree_id": "unit-a.tree0", "sequence": self.sequence,
                 "policy_version": self.coordinator.version, "kind": kind, **fields}
        self.sequence += 1
        return value

    def checkpoint(self, *, gamma=0.99, solved=False, visits=2, total=-4.0):
        return self.event("checkpoint", step=0, gamma=gamma, root_is_solved=solved,
            tree={"root_index": 0, "nodes": [{"data": {"toPlay": "OR", "isSolved": False,
                "numVisit": visits, "valueSum": total}, "children": [{"childIndex": 1,
                "edge": {"tacticStr": "intro n", "numVisit": 1, "isFocus": False}}]}]})

    def seed(self):
        self.coordinator.accept(self.event("generation", step=0, node_index=0, tactic="intro n",
            generation_index=0, candidate_index=0,
            prompt="the original prompt", raw_logprob=-2.0))
        self.coordinator.accept(self.event("eval", step=0, node_index=0, tactic="intro n",
            generation_index=0, candidate_index=0, disposition="created", child_index=1))
        self.coordinator.accept(self.event("backup", step=0, node_index=0))

    def test_same_tree_continues_only_after_committed_learn(self):
        self.seed()
        self.coordinator.accept(self.checkpoint())
        ack = json.loads((self.root / "acks/checkpoint-000000.ack.json").read_text())
        self.assertEqual(ack["policy_version"], 1)
        self.assertEqual(self.calls[0]["reward"], 0)
        self.assertEqual(self.calls[0]["candidates"][0]["visits"], 1)
        self.assertEqual(self.calls[0]["backup"], {"value_sum": -4.0, "visits": 2, "kind": "OR", "valid": True})
        self.coordinator.accept(self.event("checkpoint_ack", step=0))
        self.coordinator.accept(self.event("generation", step=1, node_index=1, tactic="rfl",
            generation_index=0, candidate_index=0, prompt="next prompt", raw_logprob=-0.1))
        self.assertEqual(len(self.coordinator.post_update_generations), 1)
        self.assertEqual(self.coordinator.post_update_generations[0]["tree_id"], "unit-a.tree0")

    def test_zero_update_control_keeps_observer_and_ack_without_learning(self):
        self.coordinator.max_updates = 0
        self.seed()
        self.coordinator.accept(self.checkpoint())
        ack = json.loads((self.root / "acks/checkpoint-000000.ack.json").read_text())
        self.assertEqual(ack["policy_version"], 0)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.coordinator.version, 0)

    def test_failure_ack_never_releases_a_successful_continue(self):
        self.seed()
        def failed(*args):
            raise TimeoutError("unknown update outcome")
        self.coordinator.learn = failed
        with self.assertRaises(TimeoutError):
            self.coordinator.accept(self.checkpoint())
        ack = json.loads((self.root / "acks/checkpoint-000000.ack.json").read_text())
        self.assertEqual(ack["status"], "error")
        self.assertEqual(ack["policy_version"], 0)
        self.assertTrue((self.root / "acks/learn-000000.request.json").exists())
        self.assertFalse((self.root / "acks/learn-000000.receipt.json").exists())

    def test_gamma_mismatch_is_error_not_silent_conversion(self):
        self.seed()
        with self.assertRaisesRegex(ValueError, "gamma"):
            self.coordinator.accept(self.checkpoint(gamma=0.9))
        self.assertEqual(self.calls, [])

    def test_root_search_flag_does_not_manufacture_terminal_reward(self):
        self.seed()
        self.coordinator.accept(self.checkpoint(solved=True))
        self.assertEqual(self.calls, [])

    def test_unvisited_and_invalid_distance_are_not_targets(self):
        self.seed()
        self.coordinator.accept(self.checkpoint(visits=0, total=0))
        self.assertEqual(self.calls, [])

    def test_sequence_tree_and_version_must_match(self):
        record = self.event("selection", step=0)
        record["tree_id"] = "other.tree"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.coordinator.accept(record)

    def test_ack_publication_does_not_overwrite(self):
        path = self.root / "ack.json"
        atomic_new_json(path, {"version": 0})
        with self.assertRaises(FileExistsError):
            atomic_new_json(path, {"version": 1})
        self.assertEqual(json.loads(path.read_text())["version"], 0)

    def test_partial_json_line_is_not_consumed(self):
        path = self.root / "stream.jsonl"
        tail = JsonlTail(path)
        path.write_bytes(b'{"a":')
        self.assertEqual(tail.read(), [])
        with path.open("ab") as handle:
            handle.write(b'1}\n')
        self.assertEqual(tail.read(), [{"a": 1}])
        self.assertEqual(tail.read(), [])

    def test_missing_strict_receipt_detail_stops_before_ack(self):
        self.seed()
        self.coordinator.learn = lambda version, event: {
            "event_id": event["event_id"], "policy_version": version + 1,
            "applied": True, "idempotent": False}
        with self.assertRaisesRegex(ValueError, "objective"):
            self.coordinator.accept(self.checkpoint())
        self.assertEqual(json.loads((self.root / "acks/checkpoint-000000.ack.json").read_text())["status"], "error")

    def start_refresh(self):
        self.coordinator.selection_value_refresh = True
        self.seed()
        checkpoint = self.checkpoint()
        checkpoint["tree"]["nodes"] += [
            {"data": {"toPlay": "AND", "isSolved": False, "numVisit": 1, "valueSum": -2},
             "children": [{"childIndex": 2}, {"childIndex": 3}, {"childIndex": 4}]},
            {"data": {"toPlay": "OR", "isSolved": False, "numVisit": 1, "valueSum": -3}, "children": []},
            {"data": {"toPlay": "OR", "isSolved": True, "numVisit": 1, "valueSum": 0}, "children": []},
            {"data": {"toPlay": "OR", "isSolved": False, "numVisit": 0, "valueSum": 0}, "children": []}]
        self.coordinator.accept(checkpoint)
        return checkpoint

    def refreshed(self):
        return self.event("selection_value_refresh", step=0, selection_value_version=1,
            scope="selection-only-not-training-Q", nodes=[
                {"node_index": 2, "node_kind": "OR", "value": -7.0},
                {"node_index": 1, "node_kind": "AND", "value": -7.0},
                {"node_index": 0, "node_kind": "OR", "value": -5.0}])

    def test_opt_in_refresh_follows_ack_then_selection_and_keeps_training_stats(self):
        checkpoint = self.start_refresh()
        original = deepcopy(checkpoint)
        self.coordinator.accept(self.event("checkpoint_ack", step=0))
        self.coordinator.accept(self.refreshed())
        self.coordinator.accept(self.event("selection", step=1))
        self.coordinator.finish()
        self.assertEqual(checkpoint, original)
        self.assertEqual(self.coordinator.selection_refreshes[0]["node_count"], 3)
        self.assertEqual(len(self.calls), 1)

    def test_refresh_cannot_be_omitted_or_emitted_before_ack(self):
        self.start_refresh()
        with self.assertRaisesRegex(ValueError, "missing ACK"):
            self.coordinator.accept(self.refreshed())
        with self.assertRaisesRegex(ValueError, "before continuing"):
            self.coordinator.accept(self.event("selection", step=1))
        with self.assertRaisesRegex(ValueError, "overlay capability"):
            self.coordinator.finish()

    def test_refresh_rejects_wrong_version_scope_nodes_and_and_minimum(self):
        self.start_refresh()
        self.coordinator.accept(self.event("checkpoint_ack", step=0))
        cases = []
        for key, value in (("selection_value_version", 2), ("selection_value_version", True),
                           ("step", 1), ("scope", "training-Q")):
            record = self.refreshed();record[key] = value;cases.append(record)
        record = self.refreshed();record["nodes"][1]["value"] = -6;cases.append(record)
        record = self.refreshed();record["nodes"].pop();cases.append(record)
        record = self.refreshed();record["nodes"].append(dict(record["nodes"][0]));cases.append(record)
        for index in (3, 4, True):
            record = self.refreshed();record["nodes"][0]["node_index"] = index;cases.append(record)
        for value in (1.0, float("nan"), True):
            record = self.refreshed();record["nodes"][0]["value"] = value;cases.append(record)
        for record in cases:
            record["sequence"] = self.coordinator.next_sequence
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.coordinator.accept(record)
            self.assertIsNotNone(self.coordinator.pending_refresh)
        self.sequence = self.coordinator.next_sequence
        self.coordinator.accept(self.refreshed())
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.coordinator.accept(self.refreshed())

    def test_disabled_refresh_rejects_unrecorded_ambient_mode(self):
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.coordinator.accept(self.refreshed())


class OnlineInitializationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base_root = self.root
        self.project = self.root / "project"
        self.project.mkdir()
        self.theorem = self.project / "Proof.lean"
        self.theorem.write_text("example : True := by trivial\n", encoding="utf-8")
        self.theorem_id = hashlib.sha256(self.theorem.read_bytes()).hexdigest()

    def created(self):
        return {"schema_version": "reap.gpu.session.v1", "session_id": "target", "theorem_id": self.theorem_id,
                "policy_version": 0, "completed": False, "optimizer_metadata": {"kind": "AdamW", "steps": 0},
                "buffer_metadata": {"events": {}, "pending_event_ids": [], "consumed_event_ids": []},
                "event_receipts": {}, "value_metadata": {"objective": "search_visit_backup", "gamma": 0.99,
                    "value_semantics": "reap.search_backup_discounted_return.v1"},
                "lineage": {"experience_id": "release", "weights_sha256": "a" * 64,
                    "reset": ["optimizer", "rng", "buffer", "policy_version", "event_receipts"],
                    "source": {"session_id": "source", "theorem_id": "source-theorem", "policy_version": 2,
                        "snapshot": "candidate", "snapshot_sha256": "b" * 64, "parent_experience_id": None}}}

    def client(self, created):
        return SimpleNamespace(create_session=Mock(return_value=deepcopy(created)),
                               snapshot=Mock(side_effect=RuntimeError("passed initialization gate")))

    def invoke(self, client, **kwargs):
        return run_online(session_id="target", project_dir=self.project, theorem_file="Proof.lean",
            output_root=self.root / "output", gpu_base_url="http://127.0.0.1:1", gamma=0.99,
            client=client, **kwargs)

    def test_pinned_and_unpinned_inherited_create_preserve_explicit_contract(self):
        for pinned in (False, True):
            with self.subTest(pinned=pinned):
                self.root = self.base_root / ("pinned" if pinned else "id-only")
                self.root.mkdir()
                client = self.client(self.created())
                pins = {"experience_weights_sha256": "a" * 64, "experience_snapshot_sha256": "b" * 64} if pinned else {}
                with patch("cpu_runtime.online_ttt.subprocess.Popen") as launch:
                    with self.assertRaisesRegex(RuntimeError, "passed initialization gate"):
                        self.invoke(client, experience_id="release", **pins)
                client.create_session.assert_called_once_with("target", theorem_id=self.theorem_id,
                                                               experience_id="release", **pins)
                client.snapshot.assert_called_once_with("target", "before-online-ttt")
                launch.assert_not_called()

    def test_same_id_wrong_content_stops_before_snapshot_or_Lean_and_never_retries(self):
        for field in ("weights_sha256", "snapshot_sha256"):
            with self.subTest(field=field):
                self.root = self.base_root / field
                self.root.mkdir()
                created = self.created()
                location = created["lineage"] if field == "weights_sha256" else created["lineage"]["source"]
                location[field] = "c" * 64
                client = self.client(created)
                pins = {"experience_id": "release", "experience_weights_sha256": "a" * 64,
                        "experience_snapshot_sha256": "b" * 64}
                with patch("cpu_runtime.online_ttt.subprocess.Popen") as launch:
                    with self.assertRaisesRegex(ValueError, "differs from pinned"):
                        self.invoke(client, **pins)
                    receipt = self.root / "output/target/create-receipt.json"
                    raw = receipt.read_bytes()
                    self.assertEqual(json.loads(raw), created)
                    with self.assertRaises(FileExistsError):
                        self.invoke(client, **pins)
                    self.assertEqual(receipt.read_bytes(), raw)
                client.create_session.assert_called_once()
                client.snapshot.assert_not_called()
                launch.assert_not_called()

    def test_incomplete_lineage_reset_identity_or_nonfresh_state_is_rejected(self):
        cases = [("lineage.reset", ["optimizer"]), ("lineage.source", {}), ("lineage", None),
                 ("lineage.source.session_id", "target"), ("lineage.source.theorem_id", self.theorem_id),
                 ("lineage.source.session_id", ""), ("lineage.source.theorem_id", "../bad"),
                 ("lineage.source.policy_version", True), ("lineage.source.policy_version", 0),
                 ("lineage.source.snapshot", "../bad"), ("lineage.source.parent_experience_id", "../bad"),
                 ("lineage.source.snapshot_sha256", "x" * 64), ("lineage.weights_sha256", "A" * 64),
                 ("optimizer_metadata.steps", 1), ("optimizer_metadata.steps", False),
                 ("policy_version", False), ("completed", True), ("buffer_metadata.events", {"old": {}}),
                 ("event_receipts", {"old": {}})]
        for index, (path, value) in enumerate(cases):
            with self.subTest(path=path, value=value):
                self.root = self.base_root / str(index)
                self.root.mkdir()
                created = self.created()
                parent = created
                fields = path.split(".")
                for field in fields[:-1]:
                    parent = parent[field]
                parent[fields[-1]] = value
                client = self.client(created)
                with patch("cpu_runtime.online_ttt.subprocess.Popen") as launch, self.assertRaises(ValueError):
                    self.invoke(client, experience_id="release")
                client.create_session.assert_called_once()
                client.snapshot.assert_not_called()
                launch.assert_not_called()

    def test_invalid_reference_is_rejected_before_create_or_output(self):
        for kwargs in ({"experience_id": "../release"}, {"experience_weights_sha256": "a" * 64},
                       {"experience_id": "release", "experience_weights_sha256": "a" * 64},
                       {"experience_id": "release", "experience_weights_sha256": "a" * 64,
                        "experience_snapshot_sha256": "B" * 64}):
            with self.subTest(kwargs=kwargs):
                client = self.client(self.created())
                with self.assertRaises(ValueError):
                    self.invoke(client, **kwargs)
                client.create_session.assert_not_called()
                self.assertFalse((self.root / "output").exists())

    def test_fresh_default_still_uses_legacy_create_signature(self):
        created = self.created()
        created.pop("lineage")
        client = self.client(created)
        with self.assertRaisesRegex(RuntimeError, "passed initialization gate"):
            self.invoke(client)
        client.create_session.assert_called_once_with("target")
        validate_created({"session_id": "target", "policy_version": 0,
                          "value_metadata": created["value_metadata"]}, "target", 0.99)


if __name__ == "__main__":
    unittest.main()
