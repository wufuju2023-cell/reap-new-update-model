"""Small CPU tensors exercise the replay probe; never real 7B/GPU evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from containers.gpu import smoke_kl_guard as probe
from tests import test_search_objective as fixtures


class KLGuardProbeTests(unittest.TestCase):
    def source(self, root, source=None):
        path = root / "event.json"
        raw = json.dumps(source or fixtures.event()).encode()
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def fixture(self, maximum):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch unavailable")
        with torch.random.fork_rng():
            torch.manual_seed(1729)
            backend = fixtures.SearchBackendCPUTests.make_backend(self)
        backend.max_post_update_kl = maximum
        runtime = fixtures.SearchBackendCPUTests.runtime(self, backend)
        return backend, runtime

    def run_case(self, case, maximum):
        backend, runtime = self.fixture(maximum)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, sha = self.source(root)
            _, event, _ = probe.read_event(path, sha, 0.99)
            result = probe.audit(runtime, backend, event, case=case, output_dir=root)
            for artifact in result["artifacts"].values():
                raw = (root / artifact["file"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), artifact["sha256"])
            observed = json.loads((root / "tentative-observations.json").read_bytes())
            return result, observed

    def test_actual_Adam_KL_accept_and_native_post_hook_no_mocking(self):
        result, observed = self.run_case("accept", 100.0)
        self.assertTrue(result["ok"], result)
        self.assertTrue(all(result["gates"].values()))
        self.assertEqual(result["learn_receipt"]["policy_version"], 1)
        self.assertEqual(len(observed), 1)
        self.assertGreater(observed[0]["parameter_deltas"]["adapter"]["changed_tensors"], 0)
        self.assertGreater(result["post_step_guard"]["post_update_kl"], 0)
        self.assertNotIn("real_7B_GPU_gate_passed", result)

    def test_actual_Adam_KL_reject_restores_all_captures_and_control(self):
        result, observed = self.run_case("reject", 1e-12)
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["gates"]["rollback_exact_full_target"])
        self.assertGreater(observed[0]["optimizer_state_entries"], 0)
        self.assertEqual(result["guard_rejection"]["type"], "KLGuardExceeded")
        for key in ("adapter", "value_head", "optimizer"):
            self.assertEqual(result["committed_parameter_deltas"][key]["changed_tensors"], 0)
            self.assertEqual(result["committed_parameter_deltas"][key]["added_tensors"], 0)
        self.assertEqual(result["artifacts"]["before_" + probe.TARGET]["decoded_state_sha256"],
                         result["artifacts"]["after_" + probe.TARGET]["decoded_state_sha256"])

    def test_wrong_expected_case_fails_without_changing_threshold_or_retrying(self):
        result, observed = self.run_case("reject", 100.0)
        self.assertFalse(result["ok"])
        self.assertFalse(result["gates"]["rejected_exact_KL_exception"])
        self.assertEqual(len(observed), 1)
        self.assertEqual(result["training_contract"]["kl_guard"]["maximum"], 100.0)
        self.assertEqual(result["learn_receipt"]["policy_version"], 1)

    def test_pinned_event_preserves_training_fields_and_rejects_version_relabel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = fixtures.event()
            path, sha = self.source(root, source)
            raw, replay, provenance = probe.read_event(path, sha, 0.99)
            self.assertEqual(raw, path.read_bytes())
            self.assertEqual(set(provenance["identity_rebindings"]), {"event_id", "session_id", "tree_id"})
            for key in set(source) - set(provenance["identity_rebindings"]):
                self.assertEqual(source[key], replay[key])
            self.assertEqual(replay["session_id"], probe.TARGET)
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                probe.read_event(path, "0" * 64, 0.99)
            for key, value in (("policy_version", 1), ("gamma", 0.98)):
                bad = deepcopy(source)
                bad[key] = value
                path, sha = self.source(root, bad)
                with self.assertRaises((ValueError, RuntimeError)):
                    probe.read_event(path, sha, 0.99)
            path.write_bytes(b'{"session_id":"a","session_id":"b"}')
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                probe.read_event(path, hashlib.sha256(path.read_bytes()).hexdigest(), 0.99)

    def test_positive_fixed_threshold_and_exclusive_artifacts(self):
        for bad in (0, -1, True, False, "nan", "inf", "-inf"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                probe.positive_threshold(bad)
        self.assertEqual(probe.positive_threshold("1e-12"), 1e-12)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"
            probe.write_bytes(path, b"original")
            with self.assertRaises(FileExistsError):
                probe.write_bytes(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"original")

    def test_GPU_precheck_failure_persists_report_without_loading_or_overwriting(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, sha = self.source(root)
            argv = ["--model-path", str(root / "model"), "--event-file", str(path), "--event-sha256", sha,
                    "--gamma", "0.99", "--case", "reject", "--max-post-update-kl", "1e-12",
                    "--output-dir", str(root / "out")]
            with patch.object(torch.cuda, "is_available", return_value=False), patch.object(probe, "RealSearchBackend") as load:
                self.assertEqual(probe.main(argv), 1)
                load.assert_not_called()
                report = json.loads((root / "out" / "report.json").read_bytes())
                self.assertFalse(report["ok"])
                self.assertFalse(report["real_7B_GPU_gate_passed"])
                self.assertIn("HIP", report["error"])
                with self.assertRaises(FileExistsError):
                    probe.main(argv)


if __name__ == "__main__":
    unittest.main()
