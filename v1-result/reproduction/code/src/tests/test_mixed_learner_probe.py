"""CPU probe orchestration and failure-report gates, not real GPU acceptance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from containers.gpu import smoke_mixed_learner as probe
from gpu_runtime.mixed_objective import make_mixed_sampler
from tests import test_mixed_learner as fixtures
from tests import test_mixed_objective as data_fixtures


class MixedProbeTests(unittest.TestCase):
    def setUp(self):
        fixtures.MixedLearnerTests.setUp(self)
        self.backend.max_post_update_kl = 100.0

    def inputs(self):
        replay_pin, human_pin = fixtures.fixtures.fixtures.DIGEST, fixtures.fixtures.SFT_PIN
        datasets = {"replay": {replay_pin: self.backend._test_dataset},
                    "mathlib_sft": {human_pin: self.backend._test_sft}}
        config, initial = make_mixed_sampler(replay_pins=[replay_pin], mathlib_sft_pins=[human_pin],
            seed=17, max_distance=8, load_replay=self.backend._load_dataset,
            load_mathlib_sft=self.backend._load_mathlib_dataset)
        return datasets, config, initial

    def run_audit(self):
        datasets, config, initial = self.inputs()
        output = self.root/"probe"; output.mkdir()
        with patch.object(probe, "base_fingerprint", return_value={"fixture": True}), \
                patch.object(self.backend, "learn", wraps=self.backend.learn) as learns:
            report = probe.audit(self.runtime, self.backend, datasets, config, initial, output,
                {"cpu_fixture": "f"*64})
            self.assertEqual(learns.call_count, 2)
        return report, output, datasets

    def test_actual_tiny_two_steps_restore_and_both_release_actor_gates(self):
        report, output, datasets = self.run_audit()
        self.assertTrue(report["ok"], report["gates"])
        self.assertTrue(all(report["gates"].values()))
        self.assertEqual(report["sampler_after_second"]["replay_cursor"], 18)
        self.assertEqual(report["sampler_after_second"]["mathlib_sft_cursor"], 2)
        self.assertEqual(report["sampled_unique_rows"], {"replay": 2, "mathlib_sft": 1})
        self.assertEqual(report["catalog_rows"], {"replay": 2, "mathlib_sft": 1})
        for info in report["artifacts"].values():
            raw = (output/info["file"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), info["sha256"])
            self.assertEqual(len(raw), info["bytes"])
        continuation = json.loads((output/"training-continuation.json").read_bytes())
        self.assertTrue(continuation["training_gates_passed"])
        self.assertTrue(continuation["input_integrity_check_pending"])
        self.assertFalse(continuation["goal_complete"])
        self.assertFalse(continuation["instance_stop_requested"])
        self.assertEqual(continuation["checkpoint_sha256"], report["second"]["checkpoint_sha256"])

    def test_receipt_wrong_rows_provenance_weights_and_losses_fail_closed(self):
        report, _, datasets = self.run_audit()
        receipt = report["first"]["runtime_receipt"]
        refs = [{k: r[k] for k in ("source", "dataset_sha256", "row")} for r in receipt["detail"]["samples"]]
        changes = []
        r = deepcopy(receipt); r["policy_version"] = 2; changes.append(r)
        r = deepcopy(receipt); r["detail"]["samples"][0]["return"] = -8; changes.append(r)
        r = deepcopy(receipt); r["detail"]["samples"][-1]["source_policy_version"] = 0; changes.append(r)
        r = deepcopy(receipt); r["detail"]["samples"][-1]["source_session_id"] = "invented"; changes.append(r)
        r = deepcopy(receipt); r["detail"]["sample_weight"] = 1.0; changes.append(r)
        r = deepcopy(receipt); r["detail"]["source_losses"]["mathlib_sft"]["policy_loss"] *= 0.1; changes.append(r)
        r = deepcopy(receipt); r["detail"]["source_losses"]["mathlib_sft"]["loss"] = float("nan"); changes.append(r)
        for key, value in (("post_update_kl", float("nan")), ("post_update_kl", -1.0),
                           ("post_update_kl", 100.1), ("maximum", 1000.0),
                           ("timing", "before_update"), ("reduction", "different"), ("scope", "unknown")):
            r = deepcopy(receipt); r["detail"]["kl_guard"][key] = value; changes.append(r)
        for changed in changes:
            with self.subTest(receipt=changed), self.assertRaises(RuntimeError):
                probe.validate_step(changed, refs, datasets, expected_step=1)

    def test_actor_freshness_binds_release_hash_reset_theorem_and_logical_state(self):
        report, output, _ = self.run_audit()
        state = self.backend.torch.load(output/"actor1-before.pt", weights_only=True)
        theorem = state["metadata"]["theorem_id"]
        self.assertTrue(probe.fresh_actor(state, report["release1"], theorem))
        for key, value in (("completed", True), ("theorem_id", "wrong"),
                           ("optimizer_metadata", {"steps": 1})):
            altered = deepcopy(state); altered["metadata"][key] = value
            self.assertFalse(probe.fresh_actor(altered, report["release1"], theorem))
        for key, value in (("weights_sha256", "a"*64), ("reset", [])):
            altered = deepcopy(state); altered["metadata"]["lineage"][key] = value
            self.assertFalse(probe.fresh_actor(altered, report["release1"], theorem))


class MixedProbeInputTests(unittest.TestCase):
    def argv(self, root):
        return ["--model-path", "fixture", "--expected-base-sha256", "b"*64,
            "--replay-dataset-root", str(root/"replay"), "--replay-dataset-sha256", "a"*64,
            "--mathlib-dataset-root", str(root/"human"), "--mathlib-dataset-sha256", "d"*64,
            "--sampler-seed", "17", "--max-distance", "64", "--output-dir", str(root/"output")]

    def test_bad_pin_or_empty_source_rejects_before_loaders(self):
        for replay, human in (([], ["b"*64]), (["a"*64], []), (["A"*64], ["b"*64]),
                              (["a"*64], ["a"*64]), (["a"*64]*2, ["b"*64])):
            with self.subTest(replay=replay, human=human), \
                    patch.object(probe, "load_verified_dataset") as r, patch.object(probe, "load_mathlib_dataset") as h:
                with self.assertRaises(RuntimeError):
                    probe.load_inputs(Path("missing"), replay, Path("missing"), human, seed=17, max_distance=8)
                r.assert_not_called(); h.assert_not_called()

    def test_error_after_successful_audit_clears_pass_and_records_no_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = MagicMock(hidden_size=3584)
            backend.experience_contract.return_value = {"base_sha256": "b"*64}
            with patch.object(probe, "load_inputs", return_value=({}, {}, {}, {})), \
                    patch.object(probe, "source_hashes", side_effect=[{"source": "a"*64}, {"source": "b"*64}]), \
                    patch.object(probe, "require_amd"), patch.object(probe, "MixedReplayBackend", return_value=backend), \
                    patch.object(probe, "GpuRuntime"), patch.object(probe, "audit", return_value={"ok": True,
                        "second": {"checkpoint_sha256": "e"*64}, "release2": {"model_release_sha256": "f"*64},
                        "sampler_after_second": {"fixture": True}}):
                self.assertEqual(probe.main(self.argv(root)), 1)
            report = json.loads((root/"output/report.json").read_bytes())
            self.assertFalse(report["ok"]); self.assertFalse(report["real_7B_GPU_gate_passed"])
            self.assertFalse(report["error"]["mutation_retry_allowed"])
            self.assertIn("source changed", report["error"]["message"])
            self.assertFalse(json.loads((root/"output/final-continuation.json").read_bytes())["stage_passed"])

    def test_cleanup_failure_cannot_leave_gpu_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); backend = MagicMock(hidden_size=3584)
            backend.experience_contract.return_value = {"base_sha256": "b"*64}
            runtime = MagicMock(); runtime.close.side_effect = RuntimeError("cleanup failed")
            with patch.object(probe, "load_inputs", return_value=({}, {}, {}, {})), \
                    patch.object(probe, "source_hashes", return_value={"source": "a"*64}), \
                    patch.object(probe, "require_amd"), patch.object(probe, "MixedReplayBackend", return_value=backend), \
                    patch.object(probe, "GpuRuntime", return_value=runtime), patch.object(probe, "audit", return_value={"ok": True}):
                self.assertEqual(probe.main(self.argv(root)), 1)
            report = json.loads((root/"output/report.json").read_bytes())
            self.assertFalse(report["ok"]); self.assertFalse(report["real_7B_GPU_gate_passed"])
            self.assertIn("cleanup_error", report)

    @unittest.skipUnless(data_fixtures.REPLAY_DIR.is_dir() and data_fixtures.SFT_DIR.is_dir(),
                         "optional pinned local evidence not present")
    def test_real_input_manifests_cover_separate_17_and_22_files(self):
        from cpu_runtime.verified_dataset_store import BUNDLE_FILES as replay_files
        from cpu_runtime.mathlib_trajectory import BUNDLE_FILES as human_files
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder, source, pin, names in (
                ("replay", data_fixtures.REPLAY_DIR, data_fixtures.REPLAY_PIN, replay_files),
                ("human", data_fixtures.SFT_DIR, data_fixtures.SFT_PIN, human_files)):
                target = root/folder/pin; target.mkdir(parents=True)
                for name in names:
                    (target/name).write_bytes((source/name).read_bytes())
            datasets, manifest, config, initial = probe.load_inputs(root/"replay", [data_fixtures.REPLAY_PIN],
                root/"human", [data_fixtures.SFT_PIN], seed=17, max_distance=64)
            self.assertEqual(len(manifest["replay"][data_fixtures.REPLAY_PIN]), 17)
            self.assertEqual(len(manifest["mathlib_sft"][data_fixtures.SFT_PIN]), 22)
            self.assertEqual(len(datasets["mathlib_sft"][data_fixtures.SFT_PIN]["rows"]), 2)
            self.assertEqual(initial["step"], 0)
            (root/"human"/data_fixtures.SFT_PIN/"source.lean").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                probe.load_inputs(root/"replay", [data_fixtures.REPLAY_PIN], root/"human", [data_fixtures.SFT_PIN],
                    seed=17, max_distance=64)


if __name__ == "__main__":
    unittest.main()
