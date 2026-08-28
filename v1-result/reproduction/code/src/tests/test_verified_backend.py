"""Small CPU mechanism gates, not Lean verification or real 7B evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_runtime.verified_backend import VerifiedReplayBackend
from gpu_runtime.verified_objective import DATA_PROFILE, OBJECTIVE_KIND
from gpu_runtime.search_backend import KLGuardExceeded
from tests import test_search_objective as fixtures

DIGEST = "a" * 64


def event(version=0):
    return {"kind": OBJECTIVE_KIND, "event_id": f"verified-{version}",
            "session_id": "learner", "policy_version": version,
            "samples": [{"dataset_sha256": DIGEST, "row": i} for i in range(2)]}


class VerifiedBackendTests(unittest.TestCase):
    def backend(self):
        backend = fixtures.SearchBackendCPUTests.make_backend(self)
        backend.__class__ = VerifiedReplayBackend
        backend.max_distance, backend.max_post_update_kl = 8, None
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        backend.dataset_root = Path(temporary.name)
        backend._experience_base_digest = "b" * 64
        backend.lora_config.lora_alpha, backend.lora_config.lora_dropout = 2, 0.0
        backend._test_dataset = {"profile": DATA_PROFILE, "session_id": "old-theorem",
            "theorem_sha256": "c" * 64, "rows": [
                {"prompt": "prompt", "tactic": "short", "return": -1, "policy_version": 4, "node_id": 0},
                {"prompt": "prompt", "tactic": "long", "return": -3, "policy_version": 9, "node_id": 1}]}
        backend._load_dataset = lambda digest: copy.deepcopy(backend._test_dataset)
        return backend

    def runtime(self, backend):
        return fixtures.SearchBackendCPUTests.runtime(self, backend)

    def test_joint_eos_ce_categorical_return_and_isolated_actual_step(self):
        backend = self.backend(); runtime = self.runtime(backend)
        runtime.create_session("learner"); runtime.create_session("control")
        torch = backend.torch
        before = backend.session_fingerprints("learner")
        control = backend.session_fingerprints("control")
        host_rng = torch.get_rng_state().clone()
        backend._activate("learner")
        policy_loss = value_loss = 0.0
        with torch.no_grad():
            for tokens, value_class in (([1, 7], 0), ([1, 2, 3, 7], 2)):
                output = backend.model(torch.tensor([[1, 2, *tokens]]))
                logs = output.logits[:, 1:-1].log_softmax(-1)
                policy_loss -= 0.5 * logs.gather(-1, torch.tensor([[tokens]]).reshape(1, len(tokens), 1)).sum().item()
                logits = backend.sessions["learner"].value_head(output.hidden_states[-1][:, 1].float())
                value_loss += 0.5 * backend.functional.cross_entropy(logits, torch.tensor([value_class])).item()
        receipt = runtime.learn("learner", expected_policy_version=0, event=event())
        self.assertEqual(receipt["policy_version"], 1)
        self.assertAlmostEqual(receipt["detail"]["policy_loss"], policy_loss, places=5)
        self.assertAlmostEqual(receipt["detail"]["value_loss"], value_loss, places=5)
        after = backend.session_fingerprints("learner")
        for name in ("adapter", "value_head", "optimizer"):
            self.assertNotEqual(before[name], after[name])
        self.assertEqual(control, backend.session_fingerprints("control"))
        self.assertTrue(torch.equal(host_rng, torch.get_rng_state()))
        self.assertEqual(backend.sessions["learner"].examples_seen, 2)
        self.assertEqual([r["source_policy_version"] for r in receipt["detail"]["samples"]], [4, 9])
        duplicate = runtime.learn("learner", expected_policy_version=0, event=event())
        self.assertTrue(duplicate["idempotent"])
        self.assertEqual(backend.session_fingerprints("learner"), after)

    def test_value_is_expected_distance_without_gamma_and_last_prompt_hidden(self):
        backend = self.backend(); runtime = self.runtime(backend)
        runtime.create_session("learner")
        from gpu_runtime.schemas import parse_chat_request
        request = parse_chat_request({"messages": [{"role": "user", "content": "prompt"}], "model": "test"})
        with backend.torch.no_grad():
            for parameter in backend.sessions["learner"].value_head.parameters():
                parameter.zero_()
        self.assertAlmostEqual(backend.value("learner", request), 4.5)

    def test_same_prompt_value_logits_ignore_targets_and_value_gradient_reaches_adapter(self):
        backend = self.backend(); runtime = self.runtime(backend)
        runtime.create_session("learner")
        torch = backend.torch
        session = backend._activate("learner")
        adapter_parameters = backend._adapter_parameters("learner")
        with torch.no_grad():
            prompt_output = backend.model(torch.tensor([[1, 2]]), output_hidden_states=True)
            prompt_hidden = prompt_output.hidden_states[-1][:, -1, :]
            prompt_logits = session.value_head(prompt_hidden.float()).clone()
            eos_output = backend.model(torch.tensor([[7]]), output_hidden_states=True)
            eos_logits = session.value_head(eos_output.hidden_states[-1][:, -1, :].float())
        # A constant head would hide a wrong causal position.
        self.assertGreater(float(prompt_logits.std()), 0)
        self.assertFalse(torch.allclose(prompt_logits, eos_logits))
        observed = []

        def inspect_value_branch(module, inputs, output):
            value_class = (0, 2)[len(observed)]
            value_only_loss = backend.functional.cross_entropy(output, torch.tensor([value_class]))
            # This inspects the actual learn graph. autograd.grad neither fills
            # parameter.grad nor changes the loss/optimizer used by learn.
            gradients = torch.autograd.grad(value_only_loss, adapter_parameters, retain_graph=True)
            observed.append((inputs[0].detach().clone(), output.detach().clone(),
                             [gradient.detach().clone() for gradient in gradients]))

        handle = session.value_head.register_forward_hook(inspect_value_branch)
        try:
            receipt = runtime.learn("learner", expected_policy_version=0, event=event())
        finally:
            handle.remove()
        self.assertEqual(receipt["policy_version"], 1)
        self.assertEqual(len(observed), 2)
        for hidden, logits, gradients in observed:
            # Both different-length targets see the same pre-update prompt-only
            # value logits, while their distinct return labels yield gradients.
            torch.testing.assert_close(hidden, prompt_hidden, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(logits, prompt_logits, rtol=1e-6, atol=1e-7)
            self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
        self.assertTrue(all(parameter.grad is None for name, parameter in backend.model.named_parameters()
                            if "adapters" not in name))

    def test_broadcastable_reference_shape_is_rejected_after_first_row_backward(self):
        backend = self.backend(); runtime = self.runtime(backend)
        runtime.create_session("learner"); runtime.create_session("control")
        before = backend.session_fingerprints("learner")
        control = backend.session_fingerprints("control")
        logical = runtime.sessions.get("learner").snapshot()
        host_rng = backend.torch.get_rng_state().clone()
        native_forward = backend.model.forward
        references = []

        def mismatched_reference(*args, **kwargs):
            output = native_forward(*args, **kwargs)
            if not backend.model.enabled:
                references.append(output.logits.shape)
                if len(references) == 2:
                    # Slicing in learn produces [1, 1, vocab], which would
                    # silently broadcast against the four-token current logits.
                    output.logits = output.logits[:, -2:, :]
            return output

        with patch.object(backend.model, "forward", side_effect=mismatched_reference), \
                patch.object(backend.sessions["learner"].optimizer, "step") as step:
            with self.assertRaisesRegex(ValueError, "reference logits shape mismatch"):
                runtime.learn("learner", expected_policy_version=0, event=event())
            step.assert_not_called()
        self.assertEqual(len(references), 2)
        self.assertEqual(backend.session_fingerprints("learner"), before)
        self.assertEqual(backend.session_fingerprints("control"), control)
        self.assertEqual(runtime.sessions.get("learner").snapshot(), logical)
        self.assertTrue(backend.torch.equal(host_rng, backend.torch.get_rng_state()))
        self.assertTrue(backend.model.config.use_cache)
        self.assertTrue(all(parameter.grad is None for parameter in backend.model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in backend.sessions["learner"].value_head.parameters()))

    def test_negative_infinity_in_unused_value_class_is_rejected_without_partial_commit(self):
        backend = self.backend(); runtime = self.runtime(backend)
        runtime.create_session("learner"); runtime.create_session("control")
        torch = backend.torch
        before = backend.session_fingerprints("learner")
        control = backend.session_fingerprints("control")
        logical = runtime.sessions.get("learner").snapshot()
        host_rng = torch.get_rng_state().clone()
        calls = []

        def corrupt_second_value(module, inputs, output):
            calls.append(True)
            if len(calls) == 2:
                changed = output.clone()
                changed[:, -1] = float("-inf")
                # The non-target -inf alone need not make cross entropy fail.
                self.assertTrue(bool(torch.isfinite(backend.functional.cross_entropy(changed, torch.tensor([2])))))
                return changed
            return output

        handle = backend.sessions["learner"].value_head.register_forward_hook(corrupt_second_value)
        try:
            with patch.object(backend.sessions["learner"].optimizer, "step") as step:
                with self.assertRaisesRegex(FloatingPointError, "verified categorical logits"):
                    runtime.learn("learner", expected_policy_version=0, event=event())
                step.assert_not_called()
        finally:
            handle.remove()
        self.assertEqual(len(calls), 2)
        self.assertEqual(backend.session_fingerprints("learner"), before)
        self.assertEqual(backend.session_fingerprints("control"), control)
        self.assertEqual(runtime.sessions.get("learner").snapshot(), logical)
        self.assertTrue(torch.equal(host_rng, torch.get_rng_state()))
        self.assertTrue(backend.model.config.use_cache)
        self.assertTrue(all(parameter.grad is None for parameter in backend.model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in backend.sessions["learner"].value_head.parameters()))

    def test_snapshot_restore_and_cross_objective_support_identity_rejected(self):
        backend = self.backend(); runtime = self.runtime(backend)
        runtime.create_session("learner")
        before = backend.session_fingerprints("learner")
        snapshot = backend.export_session("learner")
        runtime.snapshot("learner", "initial")
        runtime.learn("learner", expected_policy_version=0, event=event())
        runtime.restore("learner", "initial")
        self.assertEqual(backend.session_fingerprints("learner"), before)
        for changes in ({"session_id": "foreign"}, {"schema_version": "reap.gpu.real-search-backend.v1"},
                        {"verified_config": {**snapshot["verified_config"], "support": {}}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                backend.import_session("learner", {**snapshot, **changes})
        self.assertEqual(backend.session_fingerprints("learner"), before)

    def test_changed_alpha_eos_and_base_identity_reject_before_import(self):
        backend = self.backend(); runtime = self.runtime(backend)
        runtime.create_session("learner")
        snapshot = backend.export_session("learner")
        before = backend.session_fingerprints("learner")
        for owner, key, changed in ((backend.lora_config, "lora_alpha", 999),
                                   (backend.tokenizer, "eos_token_id", 6),
                                   (backend, "_experience_base_digest", "d"*64)):
            with self.subTest(key=key), patch.object(owner, key, changed):
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    backend.import_session("learner", snapshot)
        self.assertEqual(backend.session_fingerprints("learner"), before)

    def test_probe_orchestration_cpu_fixture_not_real_base_or_Lean_gate(self):
        from containers.gpu import smoke_verified_replay as probe
        backend = self.backend(); runtime = self.runtime(backend)
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, "base_fingerprint", return_value={"fixture": True}):
            report = probe.audit(runtime, backend, {DIGEST: backend._test_dataset}, Path(directory))
        self.assertTrue(report["ok"], report["gates"])
        self.assertEqual(report["rows"], 2)

    def test_corrupt_dataset_unsupported_return_and_sequence_limit_do_not_step(self):
        for failure in ("corrupt", "overflow", "sequence", "eos"):
            with self.subTest(failure=failure):
                backend = self.backend(); runtime = self.runtime(backend)
                runtime.create_session("learner")
                if failure == "corrupt":
                    backend._load_dataset = lambda digest: (_ for _ in ()).throw(ValueError("corrupt bundle"))
                elif failure == "overflow":
                    backend._test_dataset["rows"][0]["return"] = -9
                elif failure == "sequence":
                    backend.max_sequence_tokens = 3
                else:
                    backend.tokenizer.eos_token_id = 1
                before = backend.session_fingerprints("learner")
                with self.assertRaises(ValueError):
                    runtime.learn("learner", expected_policy_version=0, event=event())
                self.assertEqual(backend.session_fingerprints("learner"), before)
                self.assertEqual(runtime.sessions.get("learner").policy_version, 0)

    def test_kl_and_late_mutation_rollback_populated_adam_and_control(self):
        for failure in ("guard", "nonfinite"):
            with self.subTest(failure=failure):
                backend = self.backend(); runtime = self.runtime(backend)
                backend.max_post_update_kl = 100.0
                runtime.create_session("learner"); runtime.create_session("control")
                runtime.learn("learner", expected_policy_version=0, event=event())
                before = backend.session_fingerprints("learner")
                control = backend.session_fingerprints("control")
                logical = runtime.sessions.get("learner").snapshot()
                step = backend.sessions["learner"].optimizer.step
                def broken():
                    step()
                    with backend.torch.no_grad():
                        next(backend.sessions["learner"].value_head.parameters()).fill_(float("nan"))
                context = (patch.object(backend, "_measure_prefix_kl", return_value=101.0) if failure == "guard"
                           else patch.object(backend.sessions["learner"].optimizer, "step", side_effect=broken))
                with context, self.assertRaises((KLGuardExceeded, FloatingPointError)):
                    runtime.learn("learner", expected_policy_version=1, event=event(1))
                self.assertEqual(backend.session_fingerprints("learner"), before)
                self.assertEqual(backend.session_fingerprints("control"), control)
                self.assertEqual(runtime.sessions.get("learner").snapshot(), logical)
                self.assertTrue(backend.model.config.use_cache)
                self.assertTrue(all(p.grad is None for p in backend.model.parameters()))


class VerifiedBackendCliTests(unittest.TestCase):
    def test_explicit_verified_profile_routes_only_its_contract(self):
        from gpu_runtime.server import build_backend, build_parser
        arguments = build_parser().parse_args(["--backend", "verified-replay", "--model-path", "fixture-model",
            "--device", "cpu", "--verified-dataset-root", "fixture-store", "--verified-max-distance", "32"])
        with patch("gpu_runtime.verified_backend.VerifiedReplayBackend") as verified, \
                patch("gpu_runtime.server.RealSearchBackend") as search, \
                patch("gpu_runtime.server.RealProverBackend") as real:
            self.assertIs(build_backend(arguments), verified.return_value)
            verified.assert_called_once_with("fixture-model", device="cpu", dataset_root=Path("fixture-store"),
                                             max_distance=32)
            search.assert_not_called(); real.assert_not_called()

    def test_verified_profile_forwards_explicit_guard_and_scoring(self):
        from gpu_runtime.real_backend import PolicyScoringConfig
        from gpu_runtime.server import build_backend, build_parser
        arguments = build_parser().parse_args(["--backend", "verified-replay", "--verified-dataset-root", "store",
            "--verified-max-distance", "8", "--max-post-update-kl", "2.5", "--policy-scoring", "tokenwise_deferred"])
        with patch("gpu_runtime.verified_backend.VerifiedReplayBackend") as verified:
            build_backend(arguments)
        verified.assert_called_once_with("/opt/models/REAL-Prover", device="cuda:0", dataset_root=Path("store"),
            max_distance=8, max_post_update_kl=2.5, policy_scoring=PolicyScoringConfig("tokenwise_deferred", 2, 8))

    def test_missing_or_cross_profile_flags_reject_before_model_allocation(self):
        from gpu_runtime.server import build_backend, build_parser
        cases = [
            ["--backend", "verified-replay"],
            ["--backend", "verified-replay", "--verified-dataset-root", "store"],
            ["--backend", "verified-replay", "--verified-max-distance", "8"],
            ["--backend", "verified-replay", "--verified-dataset-root", "store",
             "--verified-max-distance", "8", "--gamma", "0.99"],
        ]
        for backend in ("toy", "real", "real-search"):
            for flag, value in (("--verified-dataset-root", "store"), ("--verified-max-distance", "8")):
                cases.append(["--backend", backend, flag, value])
        for argv in cases:
            with self.subTest(argv=argv), patch("gpu_runtime.verified_backend.VerifiedReplayBackend") as verified, \
                    patch("gpu_runtime.server.RealSearchBackend") as search, \
                    patch("gpu_runtime.server.RealProverBackend") as real, \
                    patch("gpu_runtime.server.ToyBackend") as toy:
                with self.assertRaises(ValueError):
                    build_backend(build_parser().parse_args(argv))
                for factory in (verified, search, real, toy):
                    factory.assert_not_called()

    def test_invalid_verified_support_rejects_before_base_constructor(self):
        from gpu_runtime.server import build_backend, build_parser
        for support in ("0", "1", "4097"):
            with self.subTest(support=support), patch("gpu_runtime.real_backend.RealProverBackend.__init__") as base:
                args = build_parser().parse_args(["--backend", "verified-replay", "--verified-dataset-root", ".",
                                                  "--verified-max-distance", support])
                with self.assertRaises(ValueError):
                    build_backend(args)
                base.assert_not_called()


if __name__ == "__main__":
    unittest.main()
