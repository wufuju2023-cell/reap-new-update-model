import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cpu_runtime.verified_collector import (CollectorCoordinator, PROFILE, OBJECTIVE, VALUE_SEMANTICS,
    MIXED_OBJECTIVE, RESETS, gamma_option, run_collector, validate_created)
from tests.test_verified_trajectory import fixture

PIN = "a" * 64


def created(sid="test", theorem_sha256="b"*64):
    return {"schema_version": "reap.gpu.session.v1", "session_id": sid, "theorem_id": theorem_sha256,
        "role": "actor", "completed": False, "policy_version": 0,
        "optimizer_metadata": {"steps": 0}, "event_receipts": {},
        "buffer_metadata": {"events": {}, "pending_event_ids": [], "consumed_event_ids": []},
        "lineage": {"model_release_sha256": PIN, "weights_sha256": "c"*64,
            "source": {"source_kind": "learner_checkpoint", "learner_id": "central",
                "learner_step": 2, "checkpoint_sha256": "d"*64}, "reset": list(RESETS)},
        "value_metadata": {"objective": OBJECTIVE, "value_semantics": VALUE_SEMANTICS,
            "support": {"distance_min": 1, "distance_max": 64,
                "return": "negative_integer_longest_generated_action_branch", "overflow": "reject"}}}


def contract():
    return {"schema_version": "reap.training.observer.v1", "session_id": "test", "tree_id": "test.tree0",
        "sequence": 0, "policy_version": 0, "kind": "collector_contract", "profile": PROFILE,
        "model_release_sha256": PIN, "return_discount": 1, "puct_value_gamma": 0.9,
        "max_distance": 64, "value_adapter": "positive-distance-negated-once", "training_enabled": False}


def mixed_created(sid="test", theorem_sha256="b"*64):
    result = created(sid, theorem_sha256)
    metadata = result["value_metadata"]
    metadata.update(objective=MIXED_OBJECTIVE, max_batch_samples=10, mixture={
        "source_counts": {"replay": 9, "mathlib_sft": 1}, "sample_weight": 0.1,
        "sampling_unit": "verified_action_row", "ratio_scope": "every_complete_batch",
        "source_profiles": {"replay": "verified-generated-action-negative-longest-branch-v1",
            "mathlib_sft": "mathlib_sft_linear_negative_remaining_actions_v1"},
        "source_losses": {"replay": ["policy", "value"], "mathlib_sft": ["policy", "value"]}})
    metadata["support"]["return"] = "negative_integer_verified_action_longest_branch; human_profile_linear_remaining_actions"
    return result


def frames():
    tree, events, _ = fixture()
    events.insert(0, contract())
    for seq, event in enumerate(events):
        event.update(sequence=seq, step=0)
        if event["kind"] == "generation":
            event["search_value"] = -3.0
        if event["kind"] == "checkpoint":
            event["gamma"] = 0.9
    return tree, events


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def coordinator(self):
        return CollectorCoordinator("test", "test.tree0", self.root/"acks",
            model_release_sha256=PIN, puct_value_gamma=0.9, max_distance=64)

    def test_release_lineage_and_local_state_accepted(self):
        self.assertEqual(validate_created(created(), "test", theorem_sha256="b"*64,
                                         model_release_sha256=PIN), 64)

    def test_mixed_profile_requires_explicit_choice_and_keeps_strict_support(self):
        kwargs = {"theorem_sha256": "b"*64, "model_release_sha256": PIN}
        with self.assertRaises(ValueError):
            validate_created(mixed_created(), "test", **kwargs)
        self.assertEqual(validate_created(mixed_created(), "test", learner_objective=MIXED_OBJECTIVE, **kwargs), 64)
        with self.assertRaises(ValueError):
            validate_created(created(), "test", learner_objective=MIXED_OBJECTIVE, **kwargs)
        for change in (lambda m: m["support"].update(overflow="clip"),
                       lambda m: m["mixture"].update(sample_weight=1),
                       lambda m: m["mixture"]["source_counts"].update(mathlib_sft=True),
                       lambda m: m["mixture"]["source_losses"].update(mathlib_sft=["policy"]),
                       lambda m: m.update(max_batch_samples=True)):
            record = mixed_created(); change(record["value_metadata"])
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_created(record, "test", learner_objective=MIXED_OBJECTIVE, **kwargs)

    def test_incomplete_or_mismatched_release_identity_rejected(self):
        changes = [lambda c: c.update(role="learner"), lambda c: c.update(policy_version=True),
            lambda c: c.update(theorem_id="f"*64), lambda c: c["lineage"].update(model_release_sha256="f"*64),
            lambda c: c["lineage"].update(weights_sha256="bad"), lambda c: c["lineage"]["reset"].pop(),
            lambda c: c["lineage"]["source"].pop("checkpoint_sha256"),
            lambda c: c["lineage"]["source"].update(learner_id="test"),
            lambda c: c["lineage"]["source"].update(learner_step=True),
            lambda c: c["buffer_metadata"]["pending_event_ids"].append("old"),
            lambda c: c["optimizer_metadata"].update(steps=1), lambda c: c.update(event_receipts={"old": {}}),
            lambda c: c["value_metadata"].update(objective="search_visit_backup"),
            lambda c: c["value_metadata"]["support"].update(overflow="clip"),
            lambda c: c["value_metadata"]["support"].update(distance_min=True)]
        for change in changes:
            with self.subTest(change=change):
                record = created(); change(record)
                with self.assertRaises(ValueError):
                    validate_created(record, "test", theorem_sha256="b"*64, model_release_sha256=PIN)

    def test_gamma_is_only_explicit_representable_puct_parameter(self):
        self.assertEqual(gamma_option(.99), 990)
        for bad in (True, 0, 1, -.1, float("nan"), float("inf"), .9999, "0.9"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gamma_option(bad)

    def test_fixed_release_ready_and_checkpoints_write_only_version_zero(self):
        c = self.coordinator()
        _, events = frames()
        for event in events:
            c.accept(event)
        c.finish()
        self.assertEqual(c.last_step, 0)
        self.assertEqual({p.name for p in c.acks.iterdir()}, {"collector-ready.ack.json", "checkpoint-000000.ack.json"})
        for path in c.acks.iterdir():
            self.assertEqual(json.loads(path.read_bytes())["policy_version"], 0)

    def test_wrong_lean_contract_refused_before_ready_ack(self):
        for field, value in (("model_release_sha256", "e"*64), ("return_discount", .9),
                             ("puct_value_gamma", 1), ("training_enabled", True),
                             ("value_adapter", "sigmoid")):
            with self.subTest(field=field):
                c = CollectorCoordinator("test", "test.tree0", self.root/field,
                    model_release_sha256=PIN, puct_value_gamma=.9, max_distance=64)
                event = contract(); event[field] = value
                with self.assertRaises(ValueError): c.accept(event)
                self.assertEqual(list(c.acks.iterdir()), [])

    def test_final_solved_ack_cannot_change_version(self):
        c = self.coordinator(); _, events = frames()
        for event in events[:-1]: c.accept(event)
        events[-1]["policy_version"] = 1
        with self.assertRaisesRegex(ValueError, "version changed"): c.accept(events[-1])
        with self.assertRaises(ValueError): c.finish()

    def test_missing_ack_sequence_or_unexpected_refresh_fails(self):
        for change in (lambda e: e.update(sequence=99), lambda e: e.update(kind="selection_value_refresh")):
            with self.subTest(change=change):
                c = self.coordinator(); c.accept(contract())
                event = {**contract(), "sequence": 1, "kind": "selection", "step": 0}; change(event)
                with self.assertRaises(ValueError): c.accept(event)
                for p in c.acks.iterdir(): p.unlink()
                c.acks.rmdir()

    def test_positive_or_nonfinite_search_value_is_refused(self):
        c = self.coordinator(); c.accept(contract())
        _, events = frames()
        for bad in (3, float("nan"), -65, 0):
            event = {**events[1], "search_value": bad}
            with self.assertRaises(ValueError): c.accept(event)

    def test_generation_must_have_matching_eval_before_checkpoint(self):
        c = self.coordinator(); _, events = frames()
        c.accept(events[0]); c.accept(events[1])
        event = {**events[-2], "sequence": 2}
        with self.assertRaisesRegex(ValueError, "incomplete generation"): c.accept(event)

    def setup_run(self):
        source = self.root/"case.lean"
        source.write_text("import ReapRuntime\ntheorem sample : True := by\n  reapTrainingMCTS\n", encoding="utf8")
        client = Mock()
        client.create_session.side_effect = lambda sid, **kw: created(sid, kw["theorem_id"])
        kwargs = dict(session_id="test", project_dir=self.root, theorem_file="case.lean", theorem="sample",
            output_root=self.root/"out", gpu_base_url="http://unused", model_release_sha256=PIN,
            puct_value_gamma=.9, client=client, command=["fixture"])
        return source, client, kwargs

    def fake_process(self, *, exhausted=False, partial=False, wrong_tree=False):
        def start(*args, env, **kwargs):
            directory = Path(env["REAP_SESSION_DIR"])
            tree, events = frames()
            if exhausted:
                tree["solution"] = None
                for node in tree["nodes"]: node["data"]["isSolved"] = False
                events[-2]["root_is_solved"] = False
                events[-2]["tree"]["nodes"] = copy.deepcopy(tree["nodes"])
            if wrong_tree: tree["nodes"][0]["data"]["state"] = ["wrong"]
            (directory/"observer.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events)+("{" if partial else ""))
            (directory/"raw_tree.json").write_text(json.dumps(tree))
            (directory/"result.json").write_text(json.dumps({"schema_version": "reap.training.result.v1",
                "session_id": "test", "solved": not exhausted, "status": "exhausted" if exhausted else "solved",
                "proof_script": None if exhausted else "step\nfinish", "error": None}))
            self.assertEqual(env["REAP_SELECTION_VALUE_REFRESH"], "")
            self.assertEqual(env["REAP_COLLECTOR_RETURN_DISCOUNT"], "1")
            return SimpleNamespace(pid=123, returncode=1 if exhausted else 0, poll=lambda: 1 if exhausted else 0)
        return start

    def test_solved_exports_exact_candidate_never_claims_independent_verification(self):
        source, client, kwargs = self.setup_run()
        with patch("cpu_runtime.verified_collector.subprocess.Popen", side_effect=self.fake_process()):
            result = run_collector(**kwargs)
        self.assertEqual(result["status"], "solved_pending_independent_verification", result)
        self.assertFalse(result["independent_verified"])
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertEqual([c[0] for c in client.method_calls], ["create_session"])
        out = self.root/"out/test"
        expected = source.read_text().replace("  reapTrainingMCTS", "  step\n  finish")+"\n#print axioms sample\n"
        self.assertEqual((out/"proof.lean").read_text(), expected)
        self.assertEqual(json.loads((out/"proof-candidate.json").read_bytes())["candidate_root_return"], -2)
        session = json.loads((out/"session.json").read_bytes())
        self.assertEqual(session["profile"], PROFILE)
        self.assertEqual(session["role"], "actor")
        self.assertEqual(session["initialization_receipt"], json.loads((out/"create-receipt.json").read_bytes()))
        self.assertEqual(session["initialization_receipt_sha256"], hashlib.sha256((out/"create-receipt.json").read_bytes()).hexdigest())

    def test_known_exhausted_has_no_success_proof_or_learning(self):
        _, client, kwargs = self.setup_run()
        with patch("cpu_runtime.verified_collector.subprocess.Popen", side_effect=self.fake_process(exhausted=True)):
            result = run_collector(**kwargs)
        self.assertEqual(result["status"], "exhausted", result)
        self.assertFalse((self.root/"out/test/proof.lean").exists())
        self.assertEqual([c[0] for c in client.method_calls], ["create_session"])

    def test_mixed_collection_records_objective_without_enabling_local_training(self):
        _, client, kwargs = self.setup_run()
        client.create_session.side_effect = lambda sid, **kw: mixed_created(sid, kw["theorem_id"])
        with patch("cpu_runtime.verified_collector.subprocess.Popen", side_effect=self.fake_process()):
            result = run_collector(**kwargs, learner_objective=MIXED_OBJECTIVE)
        self.assertEqual(result["status"], "solved_pending_independent_verification")
        self.assertEqual(result["learner_objective"], MIXED_OBJECTIVE)
        self.assertFalse(result["training_enabled"])
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertEqual([c[0] for c in client.method_calls], ["create_session"])

    def test_invalid_pin_or_discount_rejected_before_create_or_output(self):
        _, client, kwargs = self.setup_run()
        for changes in ({"model_release_sha256": "bad"}, {"return_discount": .99}, {"puct_value_gamma": 1}):
            with self.assertRaises(ValueError): run_collector(**{**kwargs, **changes})
        client.create_session.assert_not_called()
        self.assertFalse((self.root/"out").exists())

    def test_abnormal_exit_with_exhausted_marker_is_unknown(self):
        _, _, kwargs = self.setup_run()
        original = self.fake_process(exhausted=True)
        def killed(*args, **options):
            process = original(*args, **options)
            process.returncode = -9
            return process
        with patch("cpu_runtime.verified_collector.subprocess.Popen", side_effect=killed):
            result = run_collector(**kwargs)
        self.assertEqual(result["status"], "failed_unknown")

    def test_changed_release_receipt_stops_before_any_lean_or_snapshot(self):
        _, client, kwargs = self.setup_run()
        def changed(sid, **kw):
            value = created(sid, kw["theorem_id"]); value["lineage"]["model_release_sha256"] = "e"*64
            return value
        client.create_session.side_effect = changed
        with patch("cpu_runtime.verified_collector.subprocess.Popen") as launch:
            result = run_collector(**kwargs)
        launch.assert_not_called()
        self.assertEqual(result["status"], "failed_unknown")
        self.assertEqual(client.create_session.call_count, 1)
        self.assertTrue((self.root/"out/test/create-receipt.json").exists())

    def test_unknown_create_is_never_resubmitted_or_cleaned(self):
        _, client, kwargs = self.setup_run()
        client.create_session.side_effect = TimeoutError("unknown")
        with patch("cpu_runtime.verified_collector.subprocess.Popen") as launch:
            result = run_collector(**kwargs)
        launch.assert_not_called()
        self.assertEqual(result["status"], "failed_unknown")
        self.assertEqual([c[0] for c in client.method_calls], ["create_session"])
        self.assertFalse((self.root/"out/test/create-receipt.json").exists())

    def test_partial_observer_or_mismatched_final_tree_cannot_publish_proof(self):
        _, _, kwargs = self.setup_run()
        for option in ("partial", "wrong_tree"):
            with self.subTest(option=option):
                kwargs["output_root"] = self.root/option
                with patch("cpu_runtime.verified_collector.subprocess.Popen", side_effect=self.fake_process(**{option: True})):
                    result = run_collector(**kwargs)
                self.assertEqual(result["status"], "failed_unknown")
                self.assertFalse((kwargs["output_root"]/"test/proof.lean").exists())


if __name__ == "__main__":
    unittest.main()
