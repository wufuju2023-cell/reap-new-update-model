"""Actual tiny autograd/Adam/rollback gates; no claim of a real 7B update."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_runtime.mixed_backend import MixedReplayBackend
from gpu_runtime.mixed_objective import OBJECTIVE_KIND, SFT_PROFILE
from gpu_runtime.search_backend import KLGuardExceeded
from gpu_runtime.verified_backend import VerifiedReplayBackend
from tests import test_verified_backend as fixtures
from tests import test_mixed_objective as data_fixtures

SFT_PIN = "d"*64


def event(version=0):
    return {"kind": OBJECTIVE_KIND, "session_id": "learner", "event_id": f"mixed-{version}",
        "policy_version": version, "samples": [
            {"source": "replay", "dataset_sha256": fixtures.DIGEST, "row": i % 2} for i in range(9)]
            + [{"source": "mathlib_sft", "dataset_sha256": SFT_PIN, "row": 0}]}


def equal_tree(torch, a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(torch, a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_tree(torch, x, y) for x, y in zip(a, b))
    return a == b


class MixedBackendTests(unittest.TestCase):
    def backend(self):
        backend = fixtures.VerifiedBackendTests.backend(self)
        backend.__class__ = MixedReplayBackend
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        backend.mathlib_dataset_root = Path(temporary.name)
        backend._test_sft = {"profile": SFT_PROFILE, "source_kind": "mathlib_sft",
            "source_policy_version": None,
            "source": {"commit": "e"*40, "file": "Fixture.lean", "declaration": "Fixture.human"},
            "rows": [{"row": 0, "state": ["prompt"], "next_state": [],
                "prompt": "prompt", "tactic": "long", "return": -2}]}
        backend._load_mathlib_dataset = lambda pin: copy.deepcopy(backend._test_sft)
        return backend

    def runtime(self, backend):
        return fixtures.VerifiedBackendTests.runtime(self, backend)

    def capture(self, backend, runtime, sid="learner"):
        encoded = backend.export_session(sid)
        payload = backend.torch.load(io.BytesIO(base64.b64decode(encoded["payload"])), weights_only=True)
        return {"logical": runtime.sessions.get(sid).snapshot(), "backend": payload}

    def test_real_joint_step_exact_weighted_losses_provenance_eos_and_isolation(self):
        backend = self.backend(); runtime = self.runtime(backend)
        metadata = runtime.create_session("learner"); runtime.create_session("control")
        torch = backend.torch; backend._activate("learner")
        before = backend.session_fingerprints("learner")
        control = self.capture(backend, runtime, "control")
        host_rng = torch.get_rng_state().clone()
        backend._activate("learner")
        base = {n: p.detach().clone() for n, p in backend.model.named_parameters() if "adapters" not in n}
        expected = {s: {"policy_loss": 0.0, "value_loss": 0.0} for s in ("replay", "mathlib_sft")}
        with torch.no_grad():
            for i in range(10):
                source = "replay" if i < 9 else "mathlib_sft"
                ids = [1, 7] if i < 9 and i % 2 == 0 else [1, 2, 3, 7]
                value_class = i % 2 * 2 if i < 9 else 1
                output = backend.model(torch.tensor([[1, 2, *ids]]))
                logs = output.logits[:, 1:-1].log_softmax(-1)
                expected[source]["policy_loss"] -= 0.1*logs.gather(-1, torch.tensor([[ids]]).reshape(1, len(ids), 1)).sum().item()
                logits = backend.sessions["learner"].value_head(output.hidden_states[-1][:, 1].float())
                expected[source]["value_loss"] += 0.1*backend.functional.cross_entropy(logits, torch.tensor([value_class])).item()
        receipt = runtime.learn("learner", expected_policy_version=0, event=event())
        detail = receipt["detail"]
        self.assertEqual(receipt["policy_version"], 1)
        self.assertEqual(detail["source_counts"], {"replay": 9, "mathlib_sft": 1})
        self.assertEqual(detail["sample_weight"], 0.1)
        self.assertEqual(backend.sessions["learner"].examples_seen, 10)
        self.assertEqual(metadata["adapter_metadata"]["objective"], OBJECTIVE_KIND)
        self.assertEqual(detail["objective"], OBJECTIVE_KIND)
        for source in expected:
            for key, value in expected[source].items():
                self.assertAlmostEqual(detail["source_losses"][source][key], value, places=5)
        for key in ("policy_loss", "value_loss", "kl", "loss"):
            self.assertAlmostEqual(sum(v[key] for v in detail["source_losses"].values()), detail[key], places=6)
        self.assertAlmostEqual(sum(v["weight_sum"] for v in detail["source_losses"].values()), 1.0)
        self.assertEqual([r["source_policy_version"] for r in detail["samples"][:9]], [4, 9]*4+[4])
        human = detail["samples"][-1]
        self.assertEqual(human["mathlib_source"], backend._test_sft["source"])
        self.assertIsNone(human["source_policy_version"])
        for field in ("source_session_id", "source_tree_id", "node_index", "prompt", "tactic"):
            self.assertNotIn(field, human)
        after = backend.session_fingerprints("learner")
        for key in ("adapter", "value_head", "optimizer"):
            self.assertNotEqual(before[key], after[key])
        self.assertTrue(equal_tree(torch, control, self.capture(backend, runtime, "control")))
        self.assertTrue(torch.equal(host_rng, torch.get_rng_state()))
        for name, parameter in backend.model.named_parameters():
            if name in base:
                self.assertTrue(torch.equal(parameter, base[name]))
                self.assertIsNone(parameter.grad)
        duplicate = runtime.learn("learner", expected_policy_version=0, event=event())
        self.assertTrue(duplicate["idempotent"])
        self.assertEqual(after, backend.session_fingerprints("learner"))

    def test_both_sources_value_gradient_reaches_adapter_and_uses_prompt_only(self):
        backend = self.backend(); runtime = self.runtime(backend); runtime.create_session("learner")
        torch = backend.torch; session = backend._activate("learner")
        with torch.no_grad():
            hidden = backend.model(torch.tensor([[1, 2]])).hidden_states[-1][:, -1].clone()
            logits = session.value_head(hidden.float()).clone()
        observed = []
        def observe(module, inputs, output):
            i = len(observed); target = (i % 2)*2 if i < 9 else 1
            value_loss = backend.functional.cross_entropy(output, torch.tensor([target]))
            grads = torch.autograd.grad(value_loss, backend._adapter_parameters("learner"), retain_graph=True)
            observed.append((inputs[0].detach().clone(), output.detach().clone(), [g.detach().clone() for g in grads]))
        handle = session.value_head.register_forward_hook(observe)
        try:
            runtime.learn("learner", expected_policy_version=0, event=event())
        finally:
            handle.remove()
        self.assertEqual(len(observed), 10)
        for actual_hidden, actual_logits, grads in observed:
            torch.testing.assert_close(actual_hidden, hidden, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(actual_logits, logits, rtol=1e-6, atol=1e-7)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)

    def test_full_snapshot_restore_and_old_profile_snapshot_or_release_rejected(self):
        backend = self.backend(); runtime = self.runtime(backend); runtime.create_session("learner")
        before = self.capture(backend, runtime)
        runtime.snapshot("learner", "mixed-initial")
        runtime.learn("learner", expected_policy_version=0, event=event())
        learned = self.capture(backend, runtime)
        self.assertTrue(learned["backend"]["optimizer"]["state"])
        runtime.snapshot("learner", "mixed-learned")
        runtime.restore("learner", "mixed-initial")
        self.assertTrue(equal_tree(backend.torch, before, self.capture(backend, runtime)))
        runtime.restore("learner", "mixed-learned")
        self.assertTrue(equal_tree(backend.torch, learned, self.capture(backend, runtime)))
        mixed = backend.export_session("learner")
        self.assertEqual(mixed["schema_version"], MixedReplayBackend.SNAPSHOT_SCHEMA)
        self.assertIn("mixed_config", mixed); self.assertNotIn("verified_config", mixed)
        old = fixtures.VerifiedBackendTests.backend(self)
        old.create_session("learner"); old_snapshot = old.export_session("learner")
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            backend.import_session("learner", old_snapshot)
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            old.import_session("learner", mixed)
        with self.assertRaisesRegex(ValueError, "objective/encoding mismatch"):
            backend.initialize_from_experience("learner", {"contract": old.experience_contract(),
                "encoding": "torch-save-base64", "payload": "invalid-must-not-read"})
        for changed in ({**mixed, "session_id": "other"},
                        {**mixed, "mixed_config": {**mixed["mixed_config"], "mixture": {}}}):
            with self.assertRaises(ValueError):
                backend.import_session("learner", changed)
        self.assertTrue(equal_tree(backend.torch, learned, self.capture(backend, runtime)))

    def test_actual_tentative_update_rejected_by_real_kl_with_complete_rollback(self):
        backend = self.backend(); backend.max_post_update_kl = 1e-12
        runtime = self.runtime(backend); runtime.create_session("learner"); runtime.create_session("control")
        before = self.capture(backend, runtime); control = self.capture(backend, runtime, "control")
        host_rng = backend.torch.get_rng_state().clone()
        session = backend.sessions["learner"]; actual_step = session.optimizer.step
        with patch.object(session.optimizer, "step", wraps=actual_step) as observed_step:
            with self.assertRaises(KLGuardExceeded) as caught:
                runtime.learn("learner", expected_policy_version=0, event=event())
            self.assertEqual(observed_step.call_count, 1)
        self.assertGreater(caught.exception.detail["post_update_kl"], 1e-12)
        self.assertTrue(equal_tree(backend.torch, before, self.capture(backend, runtime)))
        self.assertTrue(equal_tree(backend.torch, control, self.capture(backend, runtime, "control")))
        self.assertTrue(backend.torch.equal(host_rng, backend.torch.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in backend.model.parameters()))
        self.assertTrue(backend.model.config.use_cache)

    def test_late_failure_rolls_back_populated_adam_rng_buffer_and_receipts(self):
        backend = self.backend(); runtime = self.runtime(backend)
        runtime.create_session("learner"); runtime.create_session("control")
        runtime.learn("learner", expected_policy_version=0, event=event())
        before = self.capture(backend, runtime); control = self.capture(backend, runtime, "control")
        host_rng = backend.torch.get_rng_state().clone()
        session = backend.sessions["learner"]; actual_step = session.optimizer.step
        def broken_step():
            actual_step()
            backend.torch.rand(3)
            with backend.torch.no_grad():
                next(session.value_head.parameters()).fill_(float("nan"))
        with patch.object(session.optimizer, "step", side_effect=broken_step) as observed:
            with self.assertRaises(FloatingPointError):
                runtime.learn("learner", expected_policy_version=1, event=event(1))
            self.assertEqual(observed.call_count, 1)
        self.assertTrue(equal_tree(backend.torch, before, self.capture(backend, runtime)))
        self.assertTrue(equal_tree(backend.torch, control, self.capture(backend, runtime, "control")))
        self.assertTrue(backend.torch.equal(host_rng, backend.torch.get_rng_state()))

    def test_corrupt_sft_bad_ratio_eos_and_overflow_never_step(self):
        for failure in ("corrupt", "ratio", "label", "eos", "sequence", "profile"):
            with self.subTest(failure=failure):
                backend = self.backend(); runtime = self.runtime(backend); runtime.create_session("learner")
                current = event()
                if failure == "corrupt":
                    backend._load_mathlib_dataset = lambda pin: (_ for _ in ()).throw(ValueError("corrupt SFT"))
                elif failure == "ratio":
                    current["samples"][-1] = current["samples"][0]
                elif failure == "label":
                    backend._test_sft["rows"][0]["return"] = -9
                elif failure == "eos":
                    backend.tokenizer.eos_token_id = 1
                elif failure == "sequence":
                    backend.max_sequence_tokens = 3
                else:
                    backend._test_sft["profile"] = "verified-generated-action-negative-longest-branch-v1"
                before = self.capture(backend, runtime)
                with patch.object(backend.sessions["learner"].optimizer, "step") as step:
                    with self.assertRaises(ValueError):
                        runtime.learn("learner", expected_policy_version=0, event=current)
                    step.assert_not_called()
                self.assertTrue(equal_tree(backend.torch, before, self.capture(backend, runtime)))

    def test_fixed_production_loaders_use_distinct_roots_and_pins(self):
        backend = self.backend()
        del backend._load_dataset; del backend._load_mathlib_dataset
        with patch("cpu_runtime.verified_trajectory.load_verified_dataset", return_value=backend._test_dataset) as generated, \
                patch("cpu_runtime.mathlib_trajectory.load_mathlib_dataset", return_value=backend._test_sft) as human:
            prepared = backend._prepare_event("learner", 0, event())
        generated.assert_called_once_with(backend.dataset_root/fixtures.DIGEST, expected_sha256=fixtures.DIGEST)
        human.assert_called_once_with(backend.mathlib_dataset_root/SFT_PIN, expected_sha256=SFT_PIN)
        self.assertEqual(prepared["source_counts"], {"replay": 9, "mathlib_sft": 1})

    @unittest.skipUnless(data_fixtures.REPLAY_DIR.is_dir() and data_fixtures.SFT_DIR.is_dir(),
                         "optional pinned local Lean evidence not present")
    def test_tiny_update_with_actual_strictly_loaded_generated_and_human_bundles(self):
        from cpu_runtime.verified_dataset_store import BUNDLE_FILES as replay_files
        from cpu_runtime.mathlib_trajectory import BUNDLE_FILES as human_files
        backend = self.backend()
        del backend._load_dataset; del backend._load_mathlib_dataset
        before = {}
        for source, pin, root, files in (
            (data_fixtures.REPLAY_DIR, data_fixtures.REPLAY_PIN, backend.dataset_root, replay_files),
            (data_fixtures.SFT_DIR, data_fixtures.SFT_PIN, backend.mathlib_dataset_root, human_files)):
            destination = root/pin; destination.mkdir()
            for name in files:
                raw = (source/name).read_bytes()
                before[str(source/name)] = hashlib.sha256(raw).hexdigest()
                (destination/name).write_bytes(raw)
        runtime = self.runtime(backend); runtime.create_session("learner")
        refs = [{"source": "replay", "dataset_sha256": data_fixtures.REPLAY_PIN, "row": i % 3} for i in range(9)]
        refs.append({"source": "mathlib_sft", "dataset_sha256": data_fixtures.SFT_PIN, "row": 0})
        receipt = runtime.learn("learner", expected_policy_version=0, event={**event(), "samples": refs})
        self.assertEqual(receipt["policy_version"], 1)
        self.assertEqual(receipt["detail"]["source_counts"], {"replay": 9, "mathlib_sft": 1})
        self.assertEqual([r["return"] for r in receipt["detail"]["samples"]], [-3, -2, -1]*3+[-2])
        for p, digest in before.items():
            self.assertEqual(hashlib.sha256(Path(p).read_bytes()).hexdigest(), digest)
        # This uses the tiny synthetic tokenizer/model, not a new real 7B or
        # Lean run. Copying fixed files here is not a production SFT installer.

    def test_missing_mathlib_store_rejected_before_loading_model(self):
        with tempfile.TemporaryDirectory() as root, patch("gpu_runtime.real_backend.RealProverBackend.__init__") as base:
            with self.assertRaisesRegex(ValueError, "Mathlib dataset store"):
                MixedReplayBackend("never-load", dataset_root=root, mathlib_dataset_root=Path(root)/"absent", max_distance=8)
            base.assert_not_called()

    def test_verified_default_contract_bytes_and_receipt_fields_remain_unchanged(self):
        backend = fixtures.VerifiedBackendTests.backend(self)
        digest = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        # Captured from the old implementation before the hook refactor, using
        # this same frozen tiny fixture. Not a hash of the new implementation.
        self.assertEqual(digest(backend._config()), "4b734edecfbaf907b92b92c462e8461fe061b8cdb38150cd40afb783e51e7002")
        self.assertEqual(digest(backend.experience_contract()), "ca1be4e58ef9c87755af71208f00805543d3fd42d6cdff49438d8711ae2b216e")
        runtime = self.runtime(backend); runtime.create_session("learner")
        detail = runtime.learn("learner", expected_policy_version=0, event=fixtures.event())["detail"]
        for field in ("source_counts", "source_losses", "sample_weight", "source_loss_reduction"):
            self.assertNotIn(field, detail)
        self.assertEqual(backend.BACKEND_KIND, "verified-replay")
        self.assertEqual(detail["objective"], "verified_success_replay")


if __name__ == "__main__":
    unittest.main()
