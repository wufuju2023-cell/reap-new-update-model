"""Local mechanism tests: mocked HTTP/Lean, tiny CPU tensors, no real search."""
import base64
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import cross_experience as helper
import test_extract_online_proof as proof_tests
from test_extract_online_proof import IMAGE, SID


class CrossTests(unittest.TestCase):
    def setUp(self):
        self.proof = proof_tests.ProofCheckTests(methodName="runTest")
        self.proof.setUp()
        self.addCleanup(self.proof.doCleanups)
        self.run = self.proof.root
        online = helper.read(self.proof.session / "online-result.json")
        online.update(policy_version=1, online_update_consumed_by_later_generation=True)
        self.proof.put("online-result.json", online)
        accepted = self.proof.run_check()
        self.source = {"session_id": SID, "theorem_id": accepted["theorem_sha256"], "policy_version": 1,
                       "snapshot": "experience-candidate", "snapshot_sha256": "b" * 64,
                       "parent_experience_id": None}
        self.proof.put("experience-candidate.json", {"session_id": SID, "snapshot": "experience-candidate",
                                                   "source": self.source})
        self.gpu = Mock()
        self.gpu.publish_experience.side_effect = lambda sid, name, expid, acceptance: {
            "schema_version": "reap.gpu.experience.v1", "experience_id": expid,
            "source": self.source, "acceptance": acceptance,
            "transfer": ["adapter", "value_head"], "weights_sha256": "c" * 64}

    def publish(self):
        return helper.publish(self.run, SID, "http://test.invalid", IMAGE, gpu=self.gpu)

    def test_publish_pins_and_single_attempt(self):
        pins = self.publish()
        self.assertEqual(pins["experience_snapshot_sha256"], "b" * 64)
        intent, actual = helper.settings(self.run / "cross")
        self.assertEqual(pins, actual)
        self.assertNotEqual(intent["source_session_id"], intent["target_session_id"])
        self.assertIn("export TARGET_SID=", (self.run / "cross/settings.sh").read_text())
        with self.assertRaises(FileExistsError):
            self.publish()
        self.assertEqual(self.gpu.publish_experience.call_count, 1)

    def test_unknown_publish_preserves_intent_and_cannot_retry(self):
        self.gpu.publish_experience.side_effect = TimeoutError("unknown")
        with self.assertRaises(TimeoutError):
            self.publish()
        self.assertTrue((self.run / "cross/publish-intent.json").is_file())
        self.assertFalse((self.run / "cross/pins.json").exists())
        with self.assertRaises(FileExistsError):
            self.publish()
        self.assertEqual(self.gpu.publish_experience.call_count, 1)

    def test_changed_proof_or_input_refused_before_publish(self):
        (self.run / "proof-check" / SID / "proof.lean").write_text("changed")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            self.publish()
        self.gpu.publish_experience.assert_not_called()
        self.assertFalse((self.run / "cross").exists())

    def test_candidate_wrong_version_refused(self):
        self.source["policy_version"] = 2
        self.proof.put("experience-candidate.json", {"session_id": SID, "snapshot": "experience-candidate",
                                                   "source": self.source})
        with self.assertRaisesRegex(ValueError, "candidate differs"):
            self.publish()
        self.gpu.publish_experience.assert_not_called()

    def test_snapshot_ack_and_unknown_are_not_reissued(self):
        self.publish()
        self.gpu.snapshot.side_effect = TimeoutError("unknown")
        with self.assertRaises(TimeoutError):
            helper.source_snapshot(self.run, "http://test.invalid", "before", gpu=self.gpu)
        with self.assertRaises(FileExistsError):
            helper.source_snapshot(self.run, "http://test.invalid", "before", gpu=self.gpu)
        self.assertEqual(self.gpu.snapshot.call_count, 1)

    def test_retire_requires_proof_and_tensor_evidence(self):
        self.publish()
        with self.assertRaises(FileNotFoundError):
            helper.retire(self.run, "http://test.invalid", gpu=self.gpu)
        self.gpu.retire_session.assert_not_called()

    def test_before_guard_refuses_target_already_started(self):
        self.publish()
        intent, _ = helper.settings(self.run / "cross")
        (self.run / "outputs" / intent["target_session_id"]).mkdir()
        with self.assertRaisesRegex(ValueError, "before snapshot"):
            helper.source_snapshot(self.run, "http://test.invalid", "before", gpu=self.gpu)
        self.gpu.snapshot.assert_not_called()

    def test_after_guard_requires_target_independent_proof(self):
        self.publish()
        with self.assertRaises(FileNotFoundError):
            helper.source_snapshot(self.run, "http://test.invalid", "after", gpu=self.gpu)
        self.gpu.snapshot.assert_not_called()

    def retirement_fixture(self):
        self.publish()
        intent, pins = helper.settings(self.run / "cross")
        target = intent["target_session_id"]
        (self.run / "outputs" / target).mkdir()
        (self.run / "proof-check" / target).mkdir()
        helper.write(self.run / "outputs" / target / "online-result.json", {"test_fixture": True})
        helper.write(self.run / "proof-check" / target / "accepted.json", {"test_fixture": True})
        helper.write(self.run / "cross/tensor-audit.json", {
            "schema_version": "reap.reproduction.cross-tensors.v1", "passed": True,
            "source_session_id": SID, "target_session_id": target, "target_theorem_sha256": "e" * 64, "pins": pins})
        helper.write(self.run / "cross/cross-source-after-intent.json", {
            "target_accepted_sha256": helper.sha((self.run / "proof-check" / target / "accepted.json").read_bytes()),
            "target_online_result_sha256": helper.sha((self.run / "outputs" / target / "online-result.json").read_bytes())})
        self.gpu.retire_session.side_effect = lambda sid, name, expected_policy_version: {
            "schema_version": "reap.gpu.retirement.v1", "status": "released", "session_id": sid,
            "policy_version": expected_policy_version, "snapshot": name, "snapshot_sha256": "f" * 64,
            "mutation_retry_allowed": False, "tombstone_scope": "current_runtime"}
        return patch.object(helper, "checked_proof", return_value=({"theorem_sha256": "e" * 64}, {"policy_version": 1}))

    def test_retire_two_explicit_snapshots_once(self):
        with self.retirement_fixture():
            result = helper.retire(self.run, "http://test.invalid", gpu=self.gpu)
            self.assertEqual(len(result["released_sessions"]), 2)
            with self.assertRaises(FileExistsError):
                helper.retire(self.run, "http://test.invalid", gpu=self.gpu)
        self.assertEqual(self.gpu.retire_session.call_count, 2)

    def test_retire_wrong_snapshot_preserves_raw_response_and_stops(self):
        context = self.retirement_fixture()
        self.gpu.retire_session.side_effect = None
        self.gpu.retire_session.return_value = {"status": "released", "session_id": SID, "policy_version": 1}
        with context, self.assertRaisesRegex(ValueError, "exact release"):
            helper.retire(self.run, "http://test.invalid", gpu=self.gpu)
        self.assertEqual(self.gpu.retire_session.call_count, 1)
        self.assertTrue((self.run / "cross" / ("retire-" + SID + "-response.json")).is_file())


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "tiny tensor audit requires CPU Torch")
class TensorTests(unittest.TestCase):
    def fixture(self, mutate=None):
        from gpu_runtime.snapshot_store import SnapshotStore
        from gpu_runtime.experience_store import ExperienceStore
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        cross = root / "cross"
        cross.mkdir()
        snapshots, experiences = SnapshotStore(root / "snapshots"), ExperienceStore(root / "experiences")
        def envelope(data, sid="source"):
            buffer = io.BytesIO()
            torch.save(data, buffer)
            return {"encoding": "torch-save-base64", "payload": base64.b64encode(buffer.getvalue()).decode(),
                    "schema_version": "reap.gpu.real-search-backend.v1", "session_id": sid,
                    "search_config": {"objective": "search_visit_backup", "gamma": 0.99}}
        data = {"adapter": {"a": torch.tensor([1., 2.])}, "value_head": {"v": torch.tensor([3.])},
                "optimizer": {"state": {0: {"step": torch.tensor(1.)}}, "param_groups": []},
                "optimizer_steps": 1, "examples_seen": 1,
                "rng": {"seed": 12, "cpu": torch.tensor([4], dtype=torch.uint8), "device": None}}
        contract = {"backend": "real-search", "objective": "search_visit_backup",
                    "search_config": {"objective": "search_visit_backup", "gamma": 0.99}}
        source_backend = {**envelope(data), "experience_contract": contract}
        source = {"session_id": "source", "theorem_id": "a" * 64, "policy_version": 1, "completed": True}
        snapshots.create("source", "experience-candidate", session_state=source, backend_state=source_backend)
        manifest = root / "snapshots/source/experience-candidate/manifest.json"
        source_id = {"session_id": "source", "theorem_id": "a" * 64, "policy_version": 1,
                     "snapshot": "experience-candidate", "snapshot_sha256": helper.sha(manifest.read_bytes()),
                     "parent_experience_id": None}
        acceptance = {"source": source_id, "kind": "independent-lean", "completed": True,
                      "passed": True, "evidence_sha256": "d" * 64}
        release = experiences.publish("exp", source=source_id, acceptance=acceptance,
                    weights={**envelope({k: data[k] for k in ("adapter", "value_head")}), "contract": contract})
        target = deepcopy(data)
        target.update(optimizer={"state": {}, "param_groups": []}, optimizer_steps=0, examples_seen=0)
        target["rng"]["seed"] = int.from_bytes(helper.hashlib.sha256(b"target").digest()[:8], "big") % (2 ** 63)
        initial = {"session_id": "target", "theorem_id": "b" * 64, "policy_version": 0, "completed": False,
                   "event_receipts": {}, "buffer_metadata": {"events": {}, "pending_event_ids": [], "consumed_event_ids": []},
                   "lineage": {"experience_id": "exp", "weights_sha256": release["weights_sha256"], "source": source_id,
                   "reset": ["optimizer", "rng", "buffer", "policy_version", "event_receipts"]}}
        after_data = deepcopy(data)
        if mutate:
            mutate(target, after_data, initial)
        snapshots.create("target", "before-online-ttt", session_state=initial, backend_state=envelope(target, "target"))
        snapshots.create("source", "cross-source-before", session_state=source, backend_state=envelope(data))
        snapshots.create("source", "cross-source-after", session_state=source, backend_state=envelope(after_data))
        helper.write(cross / "publish-intent.json", {"source_session_id": "source", "target_session_id": "target",
                     "experience_id": "exp", "acceptance": acceptance})
        helper.write(cross / "publish-response.json", release)
        helper.write(cross / "pins.json", {"experience_id": "exp", "experience_weights_sha256": release["weights_sha256"],
                     "experience_snapshot_sha256": source_id["snapshot_sha256"]})
        for phase in ("before", "after"):
            name = "cross-source-" + phase
            request = {"session_id": "source", "name": name, "mutation_retry_allowed": False}
            if phase == "after":
                request.update(target_session_id="target", target_theorem_sha256="b" * 64,
                               target_accepted_sha256="e" * 64, target_online_result_sha256="f" * 64)
            helper.write(cross / (name + "-intent.json"), request)
            helper.write(cross / (name + "-response.json"), {"session_id": "source", "snapshot": name})
        return root, cross

    def test_actual_small_tensor_copy_and_full_state_audit(self):
        root, cross = self.fixture()
        result = helper.check_tensors(root, cross, cross / "audit.json")
        self.assertTrue(result["passed"])
        self.assertEqual(result["tensor_counts"], {"adapter": 1, "value_head": 1})

    def test_wrong_target_parameter_refused(self):
        root, cross = self.fixture(lambda target, after, initial: target["adapter"]["a"].add_(1))
        with self.assertRaisesRegex(ValueError, "parameter copy differs"):
            helper.check_tensors(root, cross, cross / "audit.json")
        self.assertFalse((cross / "audit.json").exists())

    def test_source_optimizer_pollution_refused(self):
        root, cross = self.fixture(lambda target, after, initial: after["optimizer"]["state"][0]["step"].add_(1))
        with self.assertRaisesRegex(ValueError, "full private state changed"):
            helper.check_tensors(root, cross, cross / "audit.json")

    def test_target_inherited_buffer_refused(self):
        root, cross = self.fixture(lambda target, after, initial: initial["buffer_metadata"]["pending_event_ids"].append("old"))
        with self.assertRaisesRegex(ValueError, "buffer"):
            helper.check_tensors(root, cross, cross / "audit.json")

    def test_rehashed_target_outer_gamma_mismatch_refused(self):
        root, cross = self.fixture()
        directory = root / "snapshots/target/before-online-ttt"
        backend = helper.read(directory / "backend.json")
        backend["search_config"]["gamma"] = 0.9
        raw = helper.canonical(backend)
        (directory / "backend.json").write_bytes(raw)
        manifest = helper.read(directory / "manifest.json")
        manifest["files"]["backend.json"] = {"bytes": len(raw), "sha256": helper.sha(raw)}
        (directory / "manifest.json").write_bytes(helper.canonical(manifest))
        with self.assertRaisesRegex(ValueError, "search contract differs"):
            helper.check_tensors(root, cross, cross / "audit.json")

    def test_rehashed_target_outer_session_mismatch_refused(self):
        root, cross = self.fixture()
        directory = root / "snapshots/target/before-online-ttt"
        backend = helper.read(directory / "backend.json")
        backend["session_id"] = "source"
        raw = helper.canonical(backend)
        (directory / "backend.json").write_bytes(raw)
        manifest = helper.read(directory / "manifest.json")
        manifest["files"]["backend.json"] = {"bytes": len(raw), "sha256": helper.sha(raw)}
        (directory / "manifest.json").write_bytes(helper.canonical(manifest))
        with self.assertRaisesRegex(ValueError, "backend identity mismatch"):
            helper.check_tensors(root, cross, cross / "audit.json")


if __name__ == "__main__":
    unittest.main()
