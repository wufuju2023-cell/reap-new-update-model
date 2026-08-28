"""Post-update KL gates on tiny CPU tensors, not real-model/GPU evidence."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_runtime.search_backend import KLGuardExceeded, KL_REDUCTION, RealSearchBackend
from tests import test_search_objective as fixtures

event = fixtures.event


class SearchKLGuardTests(unittest.TestCase):
    def backend(self, maximum=None):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch unavailable")
        with torch.random.fork_rng():
            torch.manual_seed(1729)
            backend = fixtures.SearchBackendCPUTests.make_backend(self)
        backend.max_post_update_kl = maximum
        return backend

    def runtime(self, backend):
        return fixtures.SearchBackendCPUTests.runtime(self, backend)

    def test_constructor_rejects_invalid_threshold_before_loading_model(self):
        for maximum in (0, -1, True, False, "2", float("nan"), float("inf"), -float("inf")):
            with self.subTest(maximum=maximum), \
                 patch("gpu_runtime.real_backend.RealProverBackend.__init__") as initialize:
                with self.assertRaises(ValueError):
                    RealSearchBackend("unused", gamma=0.99, max_post_update_kl=maximum)
                initialize.assert_not_called()
        def initialize(backend, *args, **kwargs):
            backend.learning_rate = backend.value_learning_rate = backend.max_grad_norm = 1.0
            backend.kl_beta = backend.value_coefficient = 0.0
        for maximum in (None, 1, 0.5):
            with self.subTest(maximum=maximum), \
                 patch("gpu_runtime.real_backend.RealProverBackend.__init__", new=initialize):
                self.assertEqual(RealSearchBackend("unused", gamma=0.99,
                    max_post_update_kl=maximum).max_post_update_kl, maximum)

    def test_default_disabled_preserves_contract_and_never_measures_guard(self):
        backend = self.backend()
        runtime = self.runtime(backend)
        runtime.create_session("search-a")
        config = backend._search_config()
        self.assertNotIn("kl_guard", config)
        backend._experience_base_digest = "b" * 64
        backend.lora_config.lora_alpha, backend.lora_config.lora_dropout = 2, 0.0
        old_contract = backend.experience_contract()
        before = backend.export_session("search-a")
        with patch.object(backend, "_measure_post_update_kl", side_effect=AssertionError("disabled guard called")):
            receipt = runtime.learn("search-a", expected_policy_version=0, event=event())
        self.assertNotIn("kl_guard", receipt["detail"])
        backend.import_session("search-a", before)
        backend.max_post_update_kl = 2.0
        enabled = backend.experience_contract()
        self.assertNotEqual(enabled, old_contract)
        self.assertEqual(enabled["search_config"]["kl_guard"]["maximum"], 2.0)
        with self.assertRaisesRegex(ValueError, "semantics mismatch"):
            backend.import_session("search-a", before)
        backend.max_post_update_kl = None
        self.assertEqual(backend.experience_contract(), old_contract)

    def test_enabled_measurement_is_after_step_and_matches_direction_weighting_and_token_sum(self):
        backend = self.backend(100.0)
        runtime = self.runtime(backend)
        runtime.create_session("search-a")
        before = backend.session_fingerprints("search-a")
        original = backend._measure_post_update_kl
        order = []
        def measure(*args):
            session = backend.sessions["search-a"]
            self.assertTrue(session.optimizer.state)
            self.assertEqual(session.optimizer_steps, 0)  # tentative, not committed
            self.assertNotEqual(backend.session_fingerprints("search-a")["adapter"], before["adapter"])
            order.append("post-step")
            return original(*args)
        with patch.object(backend, "_measure_post_update_kl", side_effect=measure):
            receipt = runtime.learn("search-a", expected_policy_version=0, event=event())
        expected = reverse = token_mean = 0.0
        with backend.torch.no_grad():
            for tokens, weight in (([1], 0.75), ([1, 2, 3], 0.25)):
                ids = backend.torch.tensor([[1, 2, *tokens]])
                logs = backend.model(ids).logits[:, 1:-1].float().log_softmax(-1)
                with backend.model.disable_adapter():
                    reference = backend.model(ids).logits[:, 1:-1].float().log_softmax(-1)
                forward_kl = (logs.exp() * (logs - reference)).sum().item()
                expected += weight * forward_kl
                token_mean += weight * forward_kl / len(tokens)
                reverse += weight * (reference.exp() * (reference - logs)).sum().item()
        guard = receipt["detail"]["kl_guard"]
        self.assertEqual(order, ["post-step"])
        self.assertAlmostEqual(guard["post_update_kl"], expected, places=7)
        self.assertGreater(abs(expected - reverse), 1e-6)
        self.assertGreater(abs(expected - token_mean), 1e-6)
        self.assertTrue(guard["accepted"])
        self.assertEqual(guard["reduction"], KL_REDUCTION)
        self.assertGreaterEqual(guard["measurement_seconds"], 0)
        self.assertEqual(receipt["policy_version"], 1)
        self.assertNotEqual(receipt["detail"]["kl"], guard["post_update_kl"])

    def test_excessive_real_kl_rolls_back_all_private_state_and_does_not_affect_other_session(self):
        backend = self.backend(1e-20)
        runtime = self.runtime(backend)
        runtime.create_session("search-a")
        runtime.create_session("search-b")
        # Start with populated Adam moments and an existing committed receipt;
        # rejection must restore these too, not only an empty fresh optimizer.
        with patch.object(backend, "_measure_post_update_kl", return_value=0.0):
            runtime.learn("search-a", expected_policy_version=0, event=event())
        before_a = backend.session_fingerprints("search-a")
        before_b = backend.session_fingerprints("search-b")
        logical = runtime.sessions.get("search-a").snapshot()
        global_rng = backend.torch.get_rng_state().clone()
        step = backend.sessions["search-a"].optimizer.step
        def step_with_rng():
            step()
            backend.torch.rand(4)  # failed tentative work must not consume RNG
        with patch.object(backend.sessions["search-a"].optimizer, "step", side_effect=step_with_rng):
            with self.assertRaises(KLGuardExceeded) as caught:
                runtime.learn("search-a", expected_policy_version=1, event=event(version=1))
        self.assertGreater(caught.exception.detail["post_update_kl"], 1e-20)
        self.assertFalse(caught.exception.detail["accepted"])
        self.assertEqual(runtime.sessions.get("search-a").snapshot(), logical)
        self.assertEqual(backend.session_fingerprints("search-a"), before_a)
        self.assertEqual(backend.session_fingerprints("search-b"), before_b)
        self.assertTrue(backend.torch.equal(global_rng, backend.torch.get_rng_state()))
        self.assertTrue(backend.model.config.use_cache)
        self.assertFalse(backend.model.training)
        self.assertTrue(all(parameter.grad is None for parameter in backend.model.parameters()))

    def test_equal_threshold_accepts_and_adjacent_float_above_rejects(self):
        for measured, accepted in ((math.nextafter(1.0, 0.0), True), (1.0, True),
                                    (math.nextafter(1.0, math.inf), False)):
            with self.subTest(measured=measured):
                backend = self.backend(1.0)
                runtime = self.runtime(backend)
                runtime.create_session("search-a")
                before = backend.session_fingerprints("search-a")
                with patch.object(backend, "_measure_post_update_kl", return_value=measured):
                    if accepted:
                        receipt = runtime.learn("search-a", expected_policy_version=0, event=event())
                        self.assertTrue(receipt["detail"]["kl_guard"]["accepted"])
                    else:
                        with self.assertRaises(KLGuardExceeded):
                            runtime.learn("search-a", expected_policy_version=0, event=event())
                        self.assertEqual(backend.session_fingerprints("search-a"), before)
                self.assertEqual(runtime.sessions.get("search-a").policy_version, int(accepted))

    def test_nonfinite_or_negative_measurement_rolls_back_instead_of_passing_comparison(self):
        for measured in (float("nan"), float("inf"), -1.0, math.nextafter(0.0, -math.inf)):
            with self.subTest(measured=measured):
                backend = self.backend(1.0)
                runtime = self.runtime(backend)
                runtime.create_session("search-a")
                before = backend.session_fingerprints("search-a")
                with patch.object(backend, "_measure_post_update_kl", return_value=measured):
                    with self.assertRaises((ValueError, FloatingPointError)):
                        runtime.learn("search-a", expected_policy_version=0, event=event())
                self.assertEqual(backend.session_fingerprints("search-a"), before)
                self.assertEqual(runtime.sessions.get("search-a").event_receipts, {})

    def test_rejected_guard_produces_error_ack_and_does_not_continue_old_tree(self):
        from cpu_runtime.online_ttt import OnlineCoordinator
        backend = self.backend(1e-20)
        runtime = self.runtime(backend)
        runtime.create_session("search-a")
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "checkpoints"
            coordinator = OnlineCoordinator("search-a", "tree-1", directory,
                lambda version, payload: runtime.learn("search-a", expected_policy_version=version, event=payload),
                gamma=0.99, max_updates=1)
            checkpoint = {"schema_version": "reap.training.observer.v1", "session_id": "search-a",
                "tree_id": "tree-1", "policy_version": 0, "sequence": 0, "kind": "checkpoint", "step": 0}
            with patch.object(coordinator, "_target", return_value=(event(), (1,))):
                with self.assertRaises(KLGuardExceeded):
                    coordinator.accept(checkpoint)
            ack = json.loads((directory / "checkpoint-000000.ack.json").read_bytes())
            self.assertEqual(ack["status"], "error")
            self.assertEqual(ack["policy_version"], 0)
            self.assertIn("KLGuardExceeded", ack["error"])
            self.assertEqual(coordinator.receipts, [])
            self.assertEqual(list(directory.glob("learn-*.receipt.json")), [])


if __name__ == "__main__":
    unittest.main()
