"""CPU-only probe orchestration checks; never a real GPU acceptance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from containers.gpu import smoke_policy_scoring as probe
from tests import test_policy_scoring as fixtures


class FakeMeter:
    def __init__(self):
        self.syncs = self.resets = 0
    def synchronize(self):
        self.syncs += 1
    def reset(self):
        self.resets += 1
    def memory(self):
        return {"allocated_bytes": 1, "peak_allocated_bytes": 2,
                "reserved_bytes": 3, "peak_reserved_bytes": 4}


class PolicyScoringProbeTests(unittest.TestCase):
    def setup_backend(self):
        backend = fixtures.PolicyScoringTests().backend(batch=2, chunk=2)
        backend.model.config.vocab_size = 11
        data = {"prompt_ids": [0, 0, 1, 3], "attention_mask": [0, 0, 1, 1],
                "candidates": [[2, 3, 9], [5, 9], [4, 6, 7, 9]]}
        return backend, data

    def test_fixed_inputs_no_generation_and_explicit_warmup_alternating_order(self):
        backend, data = self.setup_backend()
        meter = FakeMeter()
        result = probe.audit(backend, "scoring", data, backend.policy_scoring,
                             atol=1e-6, repeats=2, warmups=1, meter=meter)
        self.assertTrue(result["ok"])
        self.assertTrue(result["unchanged_rng"] and result["unchanged_state"])
        self.assertFalse(backend.model.generate_calls)
        self.assertEqual(len(result["warmups"]), 2)
        self.assertEqual([r["mode"] for r in result["measurements"]],
                         ["tokenwise", "candidate_chunks", "candidate_chunks", "tokenwise"])
        self.assertEqual((meter.resets, meter.syncs), (6, 12))
        self.assertEqual(result["comparisons"][0]["maximum_absolute_error"], 0)
        self.assertFalse(result["full_frozen_base_content_hash_checked"])

    def test_numerical_mismatch_is_failure_independent_of_speed(self):
        backend, data = self.setup_backend()
        score = backend._score_generated_candidates
        def wrong(*args):
            rows = score(*args)
            rows[0][0]["logprob"] += 0.25
            return rows
        with patch.object(backend, "_score_generated_candidates", side_effect=wrong):
            result = probe.audit(backend, "scoring", data, backend.policy_scoring,
                                 atol=1e-6, repeats=1, meter=FakeMeter())
        self.assertFalse(result["ok"])
        self.assertEqual(result["comparisons"][0]["mismatch_count"], 1)
        self.assertEqual(result["comparisons"][0]["maximum_absolute_error"], 0.25)

    def test_diagnostic_matrix_same_session_separates_chunk_and_batch(self):
        backend, data = self.setup_backend()
        meter = FakeMeter()
        result = probe.audit_matrix(backend, "scoring", data, atol=1e-6, repeats=1, meter=meter)
        self.assertTrue(result["ok"])
        self.assertEqual(list(result["variants"]), ["batch1_chunk1", "batch1_chunk8", "batch2_chunk1"])
        self.assertEqual([(r["config"]["candidate_batch_size"], r["config"]["token_chunk_size"])
                         for r in result["variants"].values()], [(1, 1), (1, 8), (2, 1)])
        self.assertTrue(all(r["maximum_absolute_error"] == 0 for r in result["cross_variant_tokenwise_references"]))
        self.assertEqual(meter.resets, 12)
        self.assertEqual(list(backend.sessions), ["scoring"])
        self.assertFalse(backend.model.generate_calls)

    def test_deferred_probe_keeps_explicit_mode_and_exact_scores(self):
        backend, data = self.setup_backend()
        config = probe.PolicyScoringConfig("tokenwise_deferred")
        result = probe.audit(backend, "scoring", data, config, atol=0.0, repeats=2, meter=FakeMeter())
        self.assertTrue(result["ok"])
        self.assertEqual(result["comparison_mode"], "tokenwise_deferred")
        self.assertNotIn("median_speed_ratio_tokenwise_over_chunks", result)
        self.assertIn("median_speed_ratio_tokenwise_over_comparison", result)
        self.assertTrue(all(row["maximum_absolute_error"] == 0 for row in result["comparisons"]))
        self.assertEqual([r["mode"] for r in result["measurements"]],
                         ["tokenwise", "tokenwise_deferred", "tokenwise_deferred", "tokenwise"])

    def test_matrix_continues_numerical_failure_but_stops_mutation(self):
        for mutate in (False, True):
            backend, data = self.setup_backend()
            original = backend._score_generated_candidates
            def bad(*args):
                rows = original(*args)
                rows[0][0]["logprob"] += 1
                if mutate:
                    backend.sessions["scoring"].examples_seen += 1
                return rows
            with patch.object(backend, "_score_generated_candidates", side_effect=bad):
                result = probe.audit_matrix(backend, "scoring", data, atol=1e-6, repeats=1, meter=FakeMeter())
            self.assertFalse(result["ok"])
            self.assertEqual(len(result["variants"]), 1 if mutate else 3)

    def test_validation_finite_tolerance_ids_eos_mask_and_context(self):
        backend, data = self.setup_backend()
        for tolerance in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                probe.finite_nonnegative(tolerance)
        for key, value in (("prompt_ids", [True, 0, 1, 3]), ("attention_mask", [1, 0, 1, 1]),
                           ("candidates", [[9, 2]]), ("candidates", [[11]]),
                           ("candidates", [[]]), ("candidates", [[2] * 4093])):
            bad = copy.deepcopy(data)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.validate_inputs(bad, 11, [9])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            probe.compare_rows([[{"token": "a", "logprob": 1.0}]],
                               [[{"token": "a", "logprob": float("nan")}]], 1, 0)

    def test_unknown_mutation_rng_and_state_are_rejected(self):
        backend, data = self.setup_backend()
        score = backend._score_generated_candidates
        def bad(*args):
            rows = score(*args)
            backend.torch.rand(1)
            backend.sessions["scoring"].examples_seen += 1
            return rows
        with patch.object(backend, "_score_generated_candidates", side_effect=bad):
            result = probe.audit(backend, "scoring", data, backend.policy_scoring,
                                 atol=1e-6, repeats=1, meter=FakeMeter())
        self.assertFalse(result["ok"])
        self.assertFalse(result["unchanged_state"])
        self.assertFalse(result["unchanged_rng"])

    def test_main_gpu_guard_failure_report_and_output_is_exclusive(self):
        backend, data = self.setup_backend()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            (model / "reap-model-lock.json").write_text(json.dumps({
                "schema_version": "reap.model-lock.v2", "repo": "FrenzyMath/REAL-Prover",
                "revision": "fe76f68d9a88f342cb7b546307c20292fea9cced", "hidden_size": 3584}))
            inputs = root / "input.json"
            inputs.write_text(json.dumps(data))
            args = ["probe", "--model-path", str(model), "--input-json", str(inputs),
                    "--output-dir", str(root / "out"), "--atol", "0.001"]
            with patch("sys.argv", args), patch.object(backend.torch.cuda, "is_available", return_value=False), \
                    patch.object(probe, "RealProverBackend") as factory:
                self.assertEqual(probe.main(), 1)
                factory.assert_not_called()
                result = json.loads((root / "out" / "report.json").read_text())
                self.assertFalse(result["ok"])
                self.assertIn("HIP", result["error"])
                with self.assertRaises(FileExistsError):
                    probe.main()
            inputs.write_text('{"prompt_ids": [], "prompt_ids": []}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                probe.read_json(inputs)


if __name__ == "__main__":
    unittest.main()
