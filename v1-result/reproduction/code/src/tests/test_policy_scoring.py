"""CPU causal-model equivalence and work-count tests, not 7B/GPU speed evidence."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gpu_runtime.real_backend import PolicyScoringConfig, RealProverBackend, _RealSession
from gpu_runtime.schemas import parse_chat_request


class PolicyScoringTests(unittest.TestCase):
    def backend(self, *, mode="candidate_chunks", batch=2, chunk=8):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch is unavailable")

        class CausalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.arange(1, 12, dtype=torch.float64) / 13, requires_grad=False)
                self.calls = []
                self.generate_calls = []
                self.config = SimpleNamespace(use_cache=True)
                self.generated = None

            def set_adapter(self, sid):
                self.active_adapter = sid

            def forward(self, input_ids, attention_mask, position_ids, past_key_values=None,
                        use_cache=True, logits_to_keep=0, **kwargs):
                if torch.is_grad_enabled() or self.training:
                    raise AssertionError("scoring must stay eval/no_grad")
                previous_length = 0 if past_key_values is None else past_key_values.length
                if attention_mask.shape != (input_ids.shape[0], previous_length + input_ids.shape[1]):
                    raise AssertionError("KV/mask length mismatch")
                expected_positions = attention_mask.long().cumsum(-1) - 1
                expected_positions.masked_fill_(attention_mask == 0, 1)
                if not torch.equal(position_ids, expected_positions[:, -input_ids.shape[1]:]):
                    raise AssertionError("wrong position IDs")
                self.calls.append({"batch": input_ids.shape[0], "input_tokens": input_ids.shape[1],
                    "logits_to_keep": logits_to_keep, "prefill": past_key_values is None,
                    "mask": attention_mask.tolist(), "input_ids": input_ids.tolist(),
                    "position_ids": position_ids.tolist()})
                # A causal prefix-dependent function with an exact integer KV
                # summary detects shift, padding, row and chunk-boundary bugs.
                values = (input_ids.long() * 3 + position_ids) * attention_mask[:, -input_ids.shape[1]:].long()
                hidden = values.cumsum(-1)
                if past_key_values is not None:
                    hidden = hidden + past_key_values.prefix[:, None]
                logits = torch.sin(hidden.double()[..., None] * self.weight) + self.weight
                if logits_to_keep:
                    logits = logits[:, -logits_to_keep:, :]
                return SimpleNamespace(logits=logits, past_key_values=SimpleNamespace(
                    prefix=hidden[:, -1].clone(), length=previous_length + input_ids.shape[1]))

            def generate(self, **kwargs):
                self.generate_calls.append(kwargs)
                torch.rand(3)  # unchanged generation consumes the same private RNG
                return self.generated.clone()

        class Tokenizer:
            eos_token_id = 9
            pad_token_id = 0
            def decode(self, ids, skip_special_tokens=False):
                eos = self.eos_token_id if isinstance(self.eos_token_id, list) else [self.eos_token_id]
                return ":".join(str(int(token)) for token in ids
                                if not skip_special_tokens or int(token) not in [self.pad_token_id, *eos])

        backend = RealProverBackend.__new__(RealProverBackend)
        backend.torch = torch
        backend.device = torch.device("cpu")
        backend.model = CausalModel().eval()
        backend.tokenizer = Tokenizer()
        backend.policy_scoring = PolicyScoringConfig(mode, batch, chunk)
        backend._test_encoded = {"input_ids": torch.tensor([[0, 0, 1, 3]]),
                                 "attention_mask": torch.tensor([[0, 0, 1, 1]])}
        backend._encoded_prompt = lambda request: backend._test_encoded
        seed, cpu, device = backend._new_rng("scoring")
        backend.sessions = {"scoring": _RealSession(None, None, rng_seed=seed, cpu_rng_state=cpu, device_rng_state=device)}
        return backend

    def request(self, n=2, max_tokens=24, temperature=1.7):
        return parse_chat_request({"model": "test", "messages": [{"role": "user", "content": "prompt"}],
                                   "n": n, "temperature": temperature, "max_tokens": max_tokens, "logprobs": True})

    def assert_rows_equal(self, left, right):
        self.assertEqual([[token["token"] for token in row] for row in left],
                         [[token["token"] for token in row] for row in right])
        for a, b in zip(left, right):
            for x, y in zip(a, b):
                self.assertAlmostEqual(x["logprob"], y["logprob"], delta=1e-6)

    def test_variable_lengths_padding_and_chunk_boundaries_match_tokenwise(self):
        candidates = [[], [9], [2, 9], [4, 5, 6, 2, 3, 9], [1] * 17 + [9]]
        for batch in (1, 2, 4):
            for chunk in (1, 2, 8, 32):
                with self.subTest(batch=batch, chunk=chunk):
                    backend = self.backend(batch=batch, chunk=chunk)
                    torch = backend.torch
                    before = {key: value.clone() for key, value in backend._test_encoded.items()}
                    with torch.no_grad():
                        expected = [backend._score_generated_tokens(backend._test_encoded, row) for row in candidates]
                        old_calls = len(backend.model.calls)
                        backend.model.calls.clear()
                        actual = backend._score_generated_candidates(backend._test_encoded, candidates, backend.policy_scoring)
                    self.assert_rows_equal(actual, expected)
                    nonempty = [row for row in candidates if row]
                    needed = sum(math.ceil(max(map(len, nonempty[i:i + batch])) / chunk)
                                 for i in range(0, len(nonempty), batch))
                    self.assertEqual(old_calls, sum(map(len, candidates)))
                    self.assertEqual(len(backend.model.calls), needed)
                    self.assertTrue(all(call["batch"] <= batch and call["logits_to_keep"] <= chunk
                                        for call in backend.model.calls))
                    self.assertTrue(all(torch.equal(before[key], backend._test_encoded[key]) for key in before))

    def test_deferred_preserves_exact_forward_trace_scores_and_scalar_storage(self):
        backend = self.backend(mode="tokenwise_deferred")
        torch = backend.torch
        candidates = [[], [2, 3, 9], [4, 9], [1] * 20 + [9]]
        with torch.no_grad():
            expected = [backend._score_generated_tokens(backend._test_encoded, row) for row in candidates]
            trace = list(backend.model.calls)
            backend.model.calls.clear()
            stack, cpu = torch.stack, torch.Tensor.cpu
            scalar_sizes, transfers = [], []
            def checked_stack(tensors, *args, **kwargs):
                scalar_sizes.extend((tensor.numel(), tensor.untyped_storage().nbytes()) for tensor in tensors)
                return stack(tensors, *args, **kwargs)
            def copied(tensor, *args, **kwargs):
                transfers.append(tensor.numel())
                return cpu(tensor, *args, **kwargs)
            with patch.object(torch, "stack", side_effect=checked_stack), \
                    patch.object(torch.Tensor, "cpu", copied), \
                    patch.object(torch.Tensor, "item", side_effect=AssertionError("per-token host extraction")):
                actual = backend._score_generated_candidates_deferred(backend._test_encoded, candidates)
        self.assertEqual(actual, expected)
        self.assertEqual(backend.model.calls, trace)
        self.assertEqual(scalar_sizes, [(1, 4)] * sum(map(len, candidates)))
        self.assertEqual(transfers, [1 + sum(map(len, candidates))])

    def test_deferred_constructs_device_tokens_once_per_nonempty_candidate(self):
        backend = self.backend(mode="tokenwise_deferred")
        torch = backend.torch
        candidates = [[], [2, 3, 9], [4, 9], [1] * 20 + [9]]
        with torch.no_grad():
            expected = [backend._score_generated_tokens(backend._test_encoded, row) for row in candidates]
            trace = list(backend.model.calls)
            backend.model.calls.clear()
            construct = torch.tensor
            constructed = []
            def tracked(data, *args, **kwargs):
                constructed.append((data, kwargs.get("device"), kwargs.get("dtype")))
                return construct(data, *args, **kwargs)
            with patch.object(torch, "tensor", side_effect=tracked):
                actual = backend._score_generated_candidates_deferred(backend._test_encoded, candidates)
        self.assertEqual(actual, expected)
        self.assertEqual(backend.model.calls, trace)
        self.assertEqual(constructed, [([row], backend.device, backend._test_encoded["input_ids"].dtype)
                                       for row in candidates if row])

    def test_deferred_public_policy_same_eos_generation_weights_and_rng(self):
        old, new = self.backend(mode="tokenwise"), self.backend(mode="tokenwise_deferred")
        for backend in (old, new):
            backend.model.generated = backend.torch.tensor([[0, 0, 1, 3, 2, 9, 0], [0, 0, 1, 3, 4, 5, 9]])
        before = new.model.weight.clone()
        outside = new.torch.get_rng_state().clone()
        expected = old.policy("scoring", self.request())
        self.assertEqual(new.policy("scoring", self.request()), expected)
        self.assertEqual(old.model.calls, new.model.calls)
        self.assertEqual(len(new.model.generate_calls), 1)
        self.assertTrue(new.torch.equal(new.model.weight, before))
        self.assertTrue(new.torch.equal(new.torch.get_rng_state(), outside))
        self.assertTrue(new.torch.equal(new.sessions["scoring"].cpu_rng_state, old.sessions["scoring"].cpu_rng_state))

    def test_deferred_bounds_and_nonfinite_fail_without_returning_partial_scores(self):
        backend = self.backend(mode="tokenwise_deferred")
        torch = backend.torch
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "4096 retained"):
            backend._score_generated_candidates_deferred(backend._test_encoded, [[1] * 2049] * 2)
        with self.assertRaisesRegex(ValueError, "4096 retained"):
            backend.policy("scoring", self.request(n=64, max_tokens=65))
        self.assertFalse(backend.model.calls or backend.model.generate_calls)
        with torch.no_grad():
            self.assertEqual(backend._score_generated_candidates_deferred(backend._test_encoded, [[], []]), [[], []])
        backend.model.generated = torch.tensor([[0, 0, 1, 3, 4, 5, 6, 9]])
        original = backend.model.forward
        private, outside = backend.sessions["scoring"].cpu_rng_state.clone(), torch.get_rng_state().clone()
        def bad(*args, **kwargs):
            result = original(*args, **kwargs)
            if len(backend.model.calls) == 2:
                result.logits[0, 0, 0] = float("nan")
            return result
        with patch.object(backend.model, "forward", side_effect=bad), self.assertRaises(FloatingPointError):
            backend.policy("scoring", self.request(n=1))
        self.assertEqual(len(backend.model.calls), 4)
        self.assertTrue(torch.equal(private, backend.sessions["scoring"].cpu_rng_state))
        self.assertTrue(torch.equal(outside, torch.get_rng_state()))

    def test_deferred_exact_limit_and_greedy_copy_budget(self):
        backend = self.backend(mode="tokenwise_deferred")
        with backend.torch.no_grad(), patch.object(backend.model, "forward", side_effect=RuntimeError("passed preflight")) as forward:
            with self.assertRaisesRegex(RuntimeError, "passed preflight"):
                backend._score_generated_candidates_deferred(backend._test_encoded, [[1] * 2048] * 2)
            forward.assert_called_once()
        # Greedy generates/scores only one row before repeating response rows.
        backend.model.generated = backend.torch.tensor([[0, 0, 1, 3, 2, 9]])
        contents, rows = backend.policy("scoring", self.request(n=64, max_tokens=65, temperature=0))
        self.assertEqual(len(contents), 64)
        self.assertEqual(len(rows), 64)
        self.assertTrue(all(row == rows[0] for row in rows))
        self.assertEqual(len(backend.model.calls), 2)

    def test_deferred_last_candidate_last_token_failure_and_multiple_eos(self):
        backend = self.backend(mode="tokenwise_deferred")
        torch = backend.torch
        backend.tokenizer.eos_token_id = [9, 10]
        backend.model.generated = torch.tensor([[0, 0, 1, 3, 2, 9, 0], [0, 0, 1, 3, 3, 4, 10]])
        original = backend.model.forward
        private, outside = backend.sessions["scoring"].cpu_rng_state.clone(), torch.get_rng_state().clone()
        def bad(*args, **kwargs):
            result = original(*args, **kwargs)
            if len(backend.model.calls) == 5:
                result.logits[0, 0, 10] = float("inf")
            return result
        with patch.object(backend.model, "forward", side_effect=bad), self.assertRaises(FloatingPointError):
            backend.policy("scoring", self.request())
        self.assertEqual(len(backend.model.calls), 5)
        self.assertTrue(torch.equal(private, backend.sessions["scoring"].cpu_rng_state))
        self.assertTrue(torch.equal(outside, torch.get_rng_state()))
        backend.model.calls.clear()
        _, rows = backend.policy("scoring", self.request())
        self.assertEqual([len(row) for row in rows], [2, 3])

    def test_policy_generation_eos_raw_scores_weights_and_rng_are_unchanged(self):
        baseline = self.backend(mode="tokenwise")
        optimized = self.backend(batch=2, chunk=2)
        torch = baseline.torch
        sequence = torch.tensor([[0, 0, 1, 3, 2, 9, 0, 0], [0, 0, 1, 3, 4, 5, 6, 9]])
        for backend in (baseline, optimized):
            backend.model.generated = sequence
        global_before = torch.get_rng_state().clone()
        weights = optimized.model.weight.clone()
        old_contents, old_rows = baseline.policy("scoring", self.request())
        contents, rows = optimized.policy("scoring", self.request())
        self.assertEqual(contents, old_contents)
        self.assertEqual([len(row) for row in rows], [2, 4])
        self.assert_rows_equal(rows, old_rows)
        self.assertEqual(len(baseline.model.calls), 6)
        self.assertEqual(len(optimized.model.calls), 2)
        self.assertTrue(torch.equal(baseline.sessions["scoring"].cpu_rng_state, optimized.sessions["scoring"].cpu_rng_state))
        self.assertTrue(torch.equal(torch.get_rng_state(), global_before))
        self.assertTrue(torch.equal(weights, optimized.model.weight))
        options = optimized.model.generate_calls[0]
        self.assertEqual(options["temperature"], 1.7)
        self.assertFalse(options["output_scores"])
        self.assertFalse(options["output_logits"])

    def test_multiple_eos_and_shared_padding_stop_at_first_eos(self):
        backend = self.backend(chunk=2)
        backend.tokenizer.eos_token_id = [9, 10]
        backend.tokenizer.pad_token_id = 9
        backend.model.generated = backend.torch.tensor([[0, 0, 1, 3, 2, 10, 9, 9], [0, 0, 1, 3, 9, 9, 9, 9]])
        _, rows = backend.policy("scoring", self.request())
        self.assertEqual([[token["token"] for token in row] for row in rows], [["2", "10"], ["9"]])

    def test_greedy_repeated_candidates_score_once_and_no_eos_is_added(self):
        backend = self.backend(chunk=2)
        backend.model.generated = backend.torch.tensor([[0, 0, 1, 3, 4, 5, 6]])
        contents, rows = backend.policy("scoring", self.request(n=3, max_tokens=3, temperature=0))
        self.assertEqual(contents, ["4:5:6"] * 3)
        self.assertEqual([[token["token"] for token in row] for row in rows], [["4", "5", "6"]] * 3)
        self.assertEqual(len(backend.model.calls), 2)
        self.assertEqual(backend.model.generate_calls[0]["num_return_sequences"], 1)

    def test_empty_candidates_make_no_forward_calls(self):
        backend = self.backend()
        with backend.torch.no_grad():
            self.assertEqual(backend._score_generated_candidates(backend._test_encoded, [[], []], backend.policy_scoring), [[], []])
        self.assertEqual(backend.model.calls, [])

    def test_limits_reject_before_generation_and_default_config_is_legacy(self):
        backend = self.backend()
        with self.assertRaisesRegex(ValueError, "4096"):
            backend.policy("scoring", self.request(max_tokens=4093))
        self.assertEqual(backend.model.generate_calls, [])
        with backend.torch.no_grad(), self.assertRaisesRegex(ValueError, "64"):
            backend._score_generated_candidates(backend._test_encoded, [[1]] * 65, backend.policy_scoring)
        self.assertEqual(PolicyScoringConfig().mode, "tokenwise")
        for kwargs in ({"mode": "automatic"}, {"candidate_batch_size": True}, {"candidate_batch_size": 5},
                       {"token_chunk_size": 0}, {"token_chunk_size": 33}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PolicyScoringConfig(**kwargs)

    def test_prompt_mask_shape_and_budget_boundary_are_validated(self):
        backend = self.backend()
        torch = backend.torch
        for mask in ([[1, 1, 1, 0]], [[0, 1, 0, 1]], [[0, 0, 2, 1]], [[0, 0, 0, 0]]):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "left-padding"):
                backend._scoring_prompt({**backend._test_encoded, "attention_mask": torch.tensor(mask)}, 2)
        prompt = {"input_ids": torch.ones((1, 4094), dtype=torch.long)}
        backend._scoring_prompt(prompt, 2)
        with self.assertRaisesRegex(ValueError, "4096"):
            backend._scoring_prompt(prompt, 3)
        with self.assertRaisesRegex(ValueError, "one nonempty"):
            backend._scoring_prompt({"input_ids": torch.ones((2, 4), dtype=torch.long)}, 2)

    def test_nonfinite_shape_cache_and_eos_errors_rollback_rng(self):
        for failure in ("nonfinite", "shape", "cache", "eos"):
            with self.subTest(failure=failure):
                backend = self.backend(chunk=2)
                torch = backend.torch
                backend.model.generated = torch.tensor([[0, 0, 1, 3, 4, 5, 6, 9]])
                private = backend.sessions["scoring"].cpu_rng_state.clone()
                outside = torch.get_rng_state().clone()
                original_forward = backend.model.forward

                def forward(*args, **kwargs):
                    output = original_forward(*args, **kwargs)
                    if failure == "nonfinite":
                        output.logits[0, 0, 0] = float("nan")
                    elif failure == "shape":
                        output.logits = output.logits[:, :1]
                    elif failure == "cache":
                        output.past_key_values = None
                    return output

                if failure == "eos":
                    backend.model.generated[0, -2] = 9  # following 9 is not pad=0
                with patch.object(backend.model, "forward", side_effect=forward), \
                        self.assertRaises((FloatingPointError, RuntimeError, ValueError)):
                    backend.policy("scoring", self.request(n=1))
                self.assertTrue(torch.equal(private, backend.sessions["scoring"].cpu_rng_state))
                self.assertTrue(torch.equal(outside, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
