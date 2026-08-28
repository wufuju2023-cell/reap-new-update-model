"""Strict objective contracts; optional tiny CPU tensors are NOT a GPU gate."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_runtime.search_objective import (
    VALUE_SEMANTICS, distance_to_value, prepare_search_event, value_to_distance,
    visit_distribution, weighted_joint_nll,
)


def event(session_id="search-a", version=0):
    return {"kind": "search_visit_backup", "event_id": f"event-{version}",
        "session_id": session_id, "tree_id": "tree-1", "step": 2,
        "node_index": 0, "policy_version": version, "prompt": "original inference prompt",
        "gamma": 0.99, "candidates": [
            {"tactic": "short", "visits": 3, "raw_logprob": -2.0, "behavior_version": 0},
            {"tactic": "long", "visits": 1, "raw_logprob": -3.0, "behavior_version": 0}],
        "backup": {"value_sum": -6.0, "visits": 2, "kind": "OR", "valid": True},
        "reward": 0, "terminal_verified": False}


class SearchObjectiveTests(unittest.TestCase):
    def prepare(self, item):
        return prepare_search_event(item, session_id="search-a", policy_version=item["policy_version"], gamma=0.99)

    def test_nonterminal_value_distance_offset_round_trip(self):
        for gamma in (0.5, 0.99, 0.999):
            for distance in (1, 2, 3, 10):
                value = gamma ** (distance - 1)
                self.assertAlmostEqual(distance_to_value(distance, gamma), value)
                self.assertAlmostEqual(value_to_distance(value, gamma), distance)
        self.assertEqual(value_to_distance(1, 0.99), 1)
        with self.assertRaisesRegex(ValueError, "solved"):
            distance_to_value(0, 0.99)

    def test_floor_and_backup_rounding_are_explicit(self):
        self.assertTrue(math.isfinite(value_to_distance(0, 0.99)))
        self.assertEqual(distance_to_value(1e10, 0.99), 1e-6)
        item = event()
        item["backup"]["value_sum"] = -2e10
        prepared = self.prepare(item)
        self.assertTrue(prepared["value_trace"]["floor_clipped"])
        self.assertEqual(prepared["value_trace"]["target"], 1e-6)
        item["backup"]["value_sum"] = -2 + 1e-10
        self.assertTrue(self.prepare(item)["value_trace"]["roundoff_clipped"])

    def test_bad_gamma_and_nonfinite_values_fail_closed(self):
        for gamma in (0, 1, -0.1, float("nan"), float("inf"), True, "0.99"):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                value_to_distance(0.5, gamma)
        for value in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                value_to_distance(value, 0.99)

    def test_visit_targets_joint_probability_not_mean_tokens_or_prior(self):
        self.assertEqual(visit_distribution([3, 1, 0]), [0.75, 0.25, 0.0])
        result = weighted_joint_nll([[-1], [-1, -2, -3]], [3, 1])
        self.assertEqual(result, 0.75 * 1 + 0.25 * 6)
        self.assertNotEqual(result, 0.75 * 1 + 0.25 * 2)
        original = self.prepare(event())
        changed = event()
        changed["candidates"][0]["raw_logprob"] = -1000
        self.assertEqual([c["target_probability"] for c in original["candidates"]],
                         [c["target_probability"] for c in self.prepare(changed)["candidates"]])
        for visits in ([], [0, 0], [True], [-1], [1.5], [2**53]):
            with self.subTest(visits=visits), self.assertRaises(ValueError):
                visit_distribution(visits)

    def test_backup_target_comes_from_distance_not_event_reward(self):
        prepared = self.prepare(event())
        self.assertEqual(prepared["reward"], 0)
        self.assertAlmostEqual(prepared["value_target"], 0.99 ** 2)
        self.assertEqual(prepared["value_trace"]["value_semantics"], VALUE_SEMANTICS)
        self.assertEqual(prepared["value_trace"]["source_value_sum"], -6)
        item = event()
        item["backup"]["kind"] = "AND"
        self.assertAlmostEqual(self.prepare(item)["value_target"], 0.99 ** 2)
        # E[gamma^(D-1)] != gamma^(E[D]-1); the mode names the latter honestly.
        self.assertNotAlmostEqual((0.99 ** 0 + 0.99 ** 4) / 2, prepared["value_target"])

    def test_invalid_event_provenance_focus_and_sentinels(self):
        mutations = [
            lambda e: e.update(kind="legacy"), lambda e: e.update(session_id="other"),
            lambda e: e.update(tree_id="../tree"), lambda e: e.update(gamma=0.98),
            lambda e: e.update(reward=-1), lambda e: e.update(reward=1),
            lambda e: e.update(terminal_verified=True), lambda e: e.update(is_focus=True),
            lambda e: e.update(prompt=""), lambda e: e.update(step=True),
            lambda e: e["backup"].update(valid=False), lambda e: e["backup"].update(visits=0),
            lambda e: e["backup"].update(value_sum=0), lambda e: e["backup"].update(value_sum=float("nan")),
            lambda e: e["candidates"][0].update(is_focus=True),
            lambda e: e["candidates"][0].update(behavior_version=1),
            lambda e: e["candidates"][0].update(raw_logprob=1),
            lambda e: e["candidates"][1].update(tactic="short"),
        ]
        for index, mutate in enumerate(mutations):
            item = event()
            mutate(item)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.prepare(item)
        with self.assertRaisesRegex(ValueError, "policy_version"):
            prepare_search_event(event(), session_id="search-a", policy_version=1, gamma=0.99)
        item = event(version=2)
        self.assertEqual(self.prepare(item)["candidates"][0]["behavior_version"], 0)


class SearchBackendCPUTests(unittest.TestCase):
    def make_backend(self):
        # Reuse the existing tiny CPU fixture; never load a real model/PEFT.
        path = Path(__file__).with_name("test_gpu_runtime.py")
        spec = importlib.util.spec_from_file_location("_reused_gpu_fixture", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        backend = module.GpuRuntimeTests._numeric_backend(self)
        from gpu_runtime.search_backend import RealSearchBackend
        backend.__class__ = RealSearchBackend
        backend.gamma, backend.value_floor = 0.99, 1e-6
        backend.max_sequence_tokens, backend.max_candidates = 4096, 64
        backend.sessions.clear()
        original_set = backend.model.set_adapter

        def activate(session_id):
            original_set(session_id)
            for name, adapter in backend.model.adapters.items():
                adapter.requires_grad_(name == session_id)

        backend.model.set_adapter = activate
        torch = backend.torch
        previous_tokenizer = backend.tokenizer

        class VariableTokenizer(type(previous_tokenizer)):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                ids = [1] if text == "short" else [1, 2, 3] if text == "long" else [1, 2]
                result["input_ids"] = torch.tensor([ids])
                result["attention_mask"] = torch.ones_like(result["input_ids"])
                return result

        backend.tokenizer = VariableTokenizer()
        return backend

    def runtime(self, backend):
        from gpu_runtime import GpuRuntime
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        runtime = GpuRuntime(backend=backend, snapshot_root=Path(temporary.name))
        self.addCleanup(runtime.close)
        return runtime

    def test_true_cpu_backward_joint_token_loss_and_tensor_receipts(self):
        backend = self.make_backend()
        runtime = self.runtime(backend)
        metadata = runtime.create_session("search-a")
        runtime.create_session("search-b")
        torch = backend.torch
        before_b = backend.session_fingerprints("search-b")
        backend._activate("search-a")
        base = {name: p.detach().clone() for name, p in backend.model.named_parameters() if "adapters" not in name}
        expected = 0.0
        with torch.no_grad():
            for ids, weight in (([1], 0.75), ([1, 2, 3], 0.25)):
                tokens = torch.tensor([[1, 2, *ids]])
                logs = backend.model(tokens).logits[:, 1:-1].log_softmax(-1)
                expected -= weight * logs.gather(-1, torch.tensor([[ids]]).reshape(1, len(ids), 1)).sum().item()
        global_rng = torch.get_rng_state().clone()
        receipt = runtime.learn("search-a", expected_policy_version=0, event=event())
        self.assertTrue(receipt["applied"])
        self.assertEqual(receipt["policy_version"], 1)
        self.assertAlmostEqual(receipt["detail"]["policy_loss"], expected, places=5)
        self.assertTrue(receipt["detail"]["finite_gradients"])
        self.assertGreater(receipt["detail"]["parameter_diffs"]["adapter"]["changed_tensors"], 0)
        self.assertGreater(receipt["detail"]["parameter_diffs"]["value_head"]["changed_tensors"], 0)
        self.assertGreater(receipt["detail"]["parameter_diffs"]["optimizer"]["added_tensors"], 0)
        self.assertEqual(backend.session_fingerprints("search-b"), before_b)
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        for name, parameter in backend.model.named_parameters():
            if name in base:
                self.assertTrue(torch.equal(parameter, base[name]))
        self.assertEqual(metadata["value_metadata"]["gamma"], 0.99)
        self.assertEqual(metadata["value_metadata"]["head"], "sigmoid")
        duplicate = runtime.learn("search-a", expected_policy_version=0, event=event())
        self.assertTrue(duplicate["idempotent"])
        self.assertFalse(duplicate["applied"])

    def test_strict_snapshot_restore_and_cross_mode_gamma_rejection(self):
        from gpu_runtime.real_backend import RealProverBackend
        from gpu_runtime.schemas import parse_chat_request
        backend = self.make_backend()
        runtime = self.runtime(backend)
        runtime.create_session("search-a")
        before = backend.session_fingerprints("search-a")
        runtime.snapshot("search-a", "initial")
        original = backend.export_session("search-a")
        runtime.learn("search-a", expected_policy_version=0, event=event())
        runtime.restore("search-a", "initial")
        self.assertEqual(backend.session_fingerprints("search-a"), before)
        for wrong in ({**original, "schema_version": "reap.gpu.real-prover-backend.v1"},
                      {**original, "session_id": "other"},
                      {**original, "search_config": {**original["search_config"], "gamma": 0.98}}):
            with self.assertRaisesRegex(ValueError, "semantics mismatch"):
                backend.import_session("search-a", wrong)
        with self.assertRaisesRegex(ValueError, "snapshot schema"):
            RealProverBackend.import_session(backend, "search-a", original)
        request = parse_chat_request({"messages": [{"role": "user", "content": "prompt"}], "model": "test"})
        probability = RealProverBackend.value(backend, "search-a", request)
        self.assertAlmostEqual(backend.value("search-a", request), value_to_distance(probability, 0.99))

    def test_changed_visits_change_loss_on_identical_restored_state(self):
        backend = self.make_backend()
        runtime = self.runtime(backend)
        runtime.create_session("search-a")
        runtime.snapshot("search-a", "initial")
        first = runtime.learn("search-a", expected_policy_version=0, event=event())
        runtime.restore("search-a", "initial")
        changed = event()
        changed["candidates"][0]["visits"] = 1
        changed["candidates"][1]["visits"] = 3
        second = runtime.learn("search-a", expected_policy_version=0, event=changed)
        self.assertEqual(first["detail"]["parameter_diffs"]["adapter"]["before_sha256"],
                         second["detail"]["parameter_diffs"]["adapter"]["before_sha256"])
        self.assertGreater(second["detail"]["policy_loss"], first["detail"]["policy_loss"])

    def test_late_optimizer_failure_restores_actual_state_and_version(self):
        backend = self.make_backend()
        runtime = self.runtime(backend)
        runtime.create_session("search-a")
        before = backend.session_fingerprints("search-a")
        session = backend.sessions["search-a"]
        step = session.optimizer.step

        def broken_step():
            step()
            with backend.torch.no_grad():
                next(backend.model.adapters["search-a"].parameters()).fill_(float("nan"))

        with patch.object(session.optimizer, "step", side_effect=broken_step):
            with self.assertRaises(FloatingPointError):
                runtime.learn("search-a", expected_policy_version=0, event=event())
        self.assertEqual(backend.session_fingerprints("search-a"), before)
        self.assertEqual(runtime.sessions.get("search-a").policy_version, 0)
        self.assertTrue(backend.model.config.use_cache)
        self.assertFalse(backend.model.training)
        self.assertTrue(all(p.grad is None for p in backend.model.parameters()))

    def test_invalid_event_and_sequence_limit_never_step(self):
        backend = self.make_backend()
        runtime = self.runtime(backend)
        runtime.create_session("search-a")
        before = backend.session_fingerprints("search-a")
        bad = event()
        bad["backup"]["valid"] = False
        with self.assertRaises(ValueError):
            runtime.learn("search-a", expected_policy_version=0, event=bad)
        backend.max_sequence_tokens = 2
        with self.assertRaisesRegex(ValueError, "never truncate"):
            runtime.learn("search-a", expected_policy_version=0, event=event())
        self.assertEqual(backend.session_fingerprints("search-a"), before)
        self.assertEqual(runtime.sessions.get("search-a").policy_version, 0)


if __name__ == "__main__":
    unittest.main()
