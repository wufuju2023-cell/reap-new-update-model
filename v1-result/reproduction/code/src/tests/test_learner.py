"""Persistent learner mechanism gates with real tiny CPU autograd, not 7B/Lean."""
import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_runtime import GpuRuntime
from gpu_runtime.learner import LearnerCoordinator, publish_checkpoint, _WriterLease
from tests import test_verified_backend as fixtures

PIN = fixtures.DIGEST
NEW_PIN = "d"*64


class LearnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.backend = fixtures.VerifiedBackendTests.backend(self)
        self.backend.model.add_adapter("__bootstrap__", self.backend.lora_config)
        self.backend.model.adapters["__bootstrap__"].requires_grad_(False)
        # Tiny fixture has no PEFT namespace; production mapping is tested below.
        self.backend._checkpoint_adapter_specs = lambda: {name: (tuple(t.shape), t.dtype)
            for name, t in self.backend._adapter_state_dict("__bootstrap__").items()}
        self.runtime = GpuRuntime(backend=self.backend, snapshot_root=self.root/"snapshots",
            learner_release_root=self.root/"learner-store", max_resident_sessions=4)
        self.addCleanup(self.runtime.close)
        # Synthetic descriptors intentionally replace only dataset admission;
        # numeric training, checkpoint validation and parameter import are real.
        def descriptor(backend, pin):
            return {"dataset_sha256": pin, "profile": backend._test_dataset["profile"],
                "replay_receipt_sha256": "e"*64, "trace_sha256": "f"*64,
                "source_session_id": "old-theorem", "theorem_sha256": "c"*64,
                "rows": 2, "source_model_release_sha256": None}
        self.admission = patch("gpu_runtime.learner.describe_dataset", side_effect=descriptor)
        self.admission.start(); self.addCleanup(self.admission.stop)

    def coordinator(self, name="central", **kwargs):
        result = LearnerCoordinator(self.runtime, learner_id=name, dataset_pins=[PIN],
            journal_root=self.root/(name+"-journal"), batch_size=2, **kwargs)
        self.addCleanup(result.close)
        return result

    def fingerprint(self, sid):
        return self.runtime.actor.submit(lambda: self.backend.session_fingerprints(sid))

    def test_continuous_steps_immutable_release_and_fixed_private_actors(self):
        learner = self.coordinator()
        first = learner.train_next(); release = learner.publish()
        old_release = copy.deepcopy(self.runtime.learner_releases.load_release(release["model_release_sha256"]))
        source = self.fingerprint("central")
        actor = self.runtime.create_session("actor-one", theorem_id="1"*64,
            model_release_sha256=release["model_release_sha256"])
        self.assertEqual(actor["role"], "actor")
        self.assertEqual(actor["policy_version"], 0)
        self.assertEqual(actor["lineage"]["source"]["learner_step"], 1)
        self.assertEqual(actor["buffer_metadata"]["pending_event_ids"], [])
        self.assertEqual(actor["event_receipts"], {})
        initial_actor = self.fingerprint("actor-one")
        for component in ("adapter", "value_head"):
            self.assertEqual(source[component], initial_actor[component])
        self.assertNotEqual(source["optimizer"], initial_actor["optimizer"])
        self.assertEqual(self.backend.sessions["actor-one"].optimizer_steps, 0)
        self.assertEqual(self.backend.sessions["actor-one"].examples_seen, 0)
        self.assertNotEqual(self.backend.sessions["central"].rng_seed, self.backend.sessions["actor-one"].rng_seed)
        with self.assertRaisesRegex(ValueError, "fixed-release actor"):
            self.runtime.learn("actor-one", expected_policy_version=0, event=fixtures.event())
        second = learner.train_next(append_dataset_pins=[NEW_PIN]); release2 = learner.publish()
        self.assertEqual(second["step"], 2)
        self.assertNotEqual(release2["model_release_sha256"], release["model_release_sha256"])
        self.assertEqual(initial_actor, self.fingerprint("actor-one"))
        self.assertEqual(old_release, self.runtime.learner_releases.load_release(release["model_release_sha256"]))
        cp = self.runtime.learner_releases.load_checkpoint(second["checkpoint_sha256"])
        self.assertEqual(cp["manifest"]["parent_checkpoint_sha256"], first["checkpoint_sha256"])
        self.assertEqual(cp["data_receipt"]["event"]["samples"], [{"dataset_sha256": NEW_PIN, "row": i} for i in range(2)])
        self.assertFalse(self.runtime.sessions.get("central").completed)
        self.assertIsNone(self.runtime.sessions.get("central").theorem_id)

    def test_restore_full_state_resume_sampler_and_publish_without_live_source(self):
        learner = self.coordinator(); learner.train_next()
        second = learner.train_next(append_dataset_pins=[NEW_PIN])
        before = self.fingerprint("central")
        learner.close(); self.runtime.delete_session("central")
        release = publish_checkpoint(self.runtime, checkpoint_sha256=second["checkpoint_sha256"],
            journal_root=self.root/"offline-publication")
        self.assertEqual(release["source"]["learner_step"], 2)
        restored = LearnerCoordinator.restore(self.runtime, checkpoint_sha256=second["checkpoint_sha256"],
            journal_root=self.root/"restored-journal")
        self.addCleanup(restored.close)
        self.assertEqual(before, self.fingerprint("central"))
        self.assertEqual(restored.sampler_state, {"step": 2, "cursor": 4})
        third = restored.train_next()
        self.assertEqual(third["runtime_receipt"]["detail"]["examples_seen"], 6)
        self.assertEqual(third["step"], 3)
        final = self.fingerprint("central")
        restored.close(); self.runtime.delete_session("central")
        for index in range(2):
            repeated = LearnerCoordinator.restore(self.runtime, checkpoint_sha256=third["checkpoint_sha256"],
                journal_root=self.root/f"restored-again-{index}")
            self.assertEqual(final, self.fingerprint("central"))
            repeated.close(); self.runtime.delete_session("central")

    def test_unknown_after_training_blocks_next_step_and_keeps_durable_intent(self):
        learner = self.coordinator()
        with patch.object(self.runtime.learner_releases, "create_checkpoint", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                learner.train_next()
        after = self.fingerprint("central")
        self.assertEqual(self.runtime.sessions.get("central").policy_version, 1)
        self.assertTrue((learner.journal/"step-00000001"/"runtime-receipt.json").is_file())
        with self.assertRaisesRegex(RuntimeError, "blocked"):
            learner.train_next()
        self.assertEqual(after, self.fingerprint("central"))

    def test_new_run_from_release_resets_private_state_but_not_weights(self):
        learner = self.coordinator(); learner.train_next(); release = learner.publish()
        before = self.fingerprint("central")
        new = self.coordinator("next-central", initial_model_release_sha256=release["model_release_sha256"])
        self.assertEqual(new.sampler_state, {"step": 0, "cursor": 0})
        self.assertEqual(self.fingerprint("next-central")["adapter"], before["adapter"])
        step = new.train_next()
        self.assertEqual(step["step"], 1)
        self.assertEqual(self.fingerprint("central"), before)
        inherited = self.fingerprint("next-central")
        new.close(); self.runtime.delete_session("next-central")
        resumed = LearnerCoordinator.restore(self.runtime, checkpoint_sha256=step["checkpoint_sha256"],
            journal_root=self.root/"inherited-restored")
        self.addCleanup(resumed.close)
        self.assertEqual(inherited, self.fingerprint("next-central"))
        self.assertEqual(resumed.train_next()["step"], 2)

    def test_reconstruction_cannot_repeat_unknown_update_or_restore_old_head(self):
        learner = self.coordinator(); first = learner.train_next()
        with patch.object(self.runtime.learner_releases, "create_checkpoint", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                learner.train_next()
        learner.close(); self.runtime.delete_session("central")
        with patch.object(self.runtime, "create_session", wraps=self.runtime.create_session) as create:
            with self.assertRaisesRegex(RuntimeError, "unresolved"):
                LearnerCoordinator.restore(self.runtime, checkpoint_sha256=first["checkpoint_sha256"],
                    journal_root=self.root/"must-not-resume")
            create.assert_not_called()
        other = self.coordinator("other"); old = other.train_next(); other.train_next()
        other.close(); self.runtime.delete_session("other")
        with self.assertRaisesRegex(ValueError, "durable learner head"):
            LearnerCoordinator.restore(self.runtime, checkpoint_sha256=old["checkpoint_sha256"],
                journal_root=self.root/"old-head-refused")

    def test_roles_release_contract_and_failed_initialization_are_atomic(self):
        learner = self.coordinator(); learner.train_next(); release = learner.publish()
        for kwargs in ({"role": "actor"}, {"role": "learner", "theorem_id": "2"*64},
                       {"role": "unknown"}, {"model_release_sha256": "0"*64, "theorem_id": "2"*64}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, FileNotFoundError)):
                self.runtime.create_session("bad", **kwargs)
            self.assertNotIn("bad", self.backend.sessions)
        with patch.object(self.backend, "initialize_from_experience", side_effect=ValueError("shape")):
            with self.assertRaisesRegex(ValueError, "shape"):
                self.runtime.create_session("bad", theorem_id="2"*64, model_release_sha256=release["model_release_sha256"])
        self.assertNotIn("bad", self.backend.sessions)
        self.runtime.create_session("old-default")
        self.assertNotIn("role", self.runtime.sessions.get("old-default").snapshot())
        with self.assertRaisesRegex(ValueError, "theorem"):
            self.runtime.snapshot("central", "forbidden-proof", for_experience=True)

    def test_extracted_weights_reject_nonfinite_wrong_shape_and_counter(self):
        learner = self.coordinator(); step = learner.train_next()
        cp = self.runtime.learner_releases.load_checkpoint(step["checkpoint_sha256"])
        torch = self.backend.torch
        payload = torch.load(io.BytesIO(base64.b64decode(cp["backend_state"]["payload"])), weights_only=True)
        for change in (lambda p: p["adapter"]["weight"].fill_(float("nan")),
                       lambda p: p["value_head"].update({"2.bias": torch.zeros(7)}),
                       lambda p: p.update(optimizer_steps=2)):
            corrupted = copy.deepcopy(payload); change(corrupted)
            buffer = io.BytesIO(); torch.save(corrupted, buffer)
            snapshot = {**cp["backend_state"], "payload": base64.b64encode(buffer.getvalue()).decode()}
            with self.assertRaises((ValueError, FloatingPointError)):
                self.backend.extract_checkpoint_weights(cp["logical_state"], snapshot)

    def test_new_lora_dtype_comes_from_base_linear_not_bootstrap_autocast(self):
        from gpu_runtime.verified_backend import VerifiedReplayBackend
        torch = self.backend.torch
        base = torch.nn.Linear(4, 3).to(dtype=torch.bfloat16)
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.base_layer = base
            def get_base_layer(self):
                return self.base_layer
        model = torch.nn.Module(); model.add_module("projection", Layer())
        bootstrap = {"projection.lora_A.weight": torch.zeros(2, 4, dtype=torch.float32),
            "projection.lora_B.weight": torch.zeros(3, 2, dtype=torch.float32)}
        with patch.object(self.backend, "model", model), patch.object(self.backend, "_adapter_state_dict", return_value=bootstrap):
            rng = torch.get_rng_state().clone()
            spec = VerifiedReplayBackend._checkpoint_adapter_specs(self.backend)
            self.assertEqual(spec, {"projection.lora_A.weight": ((2, 4), torch.bfloat16),
                "projection.lora_B.weight": ((3, 2), torch.bfloat16)})
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            with patch.object(model.projection, "get_base_layer", return_value=torch.nn.Embedding(3, 4)):
                with self.assertRaisesRegex(ValueError, "ordinary unquantized"):
                    VerifiedReplayBackend._checkpoint_adapter_specs(self.backend)

    def test_checkpoint_bf16_extraction_is_exact_and_other_dtypes_rejected(self):
        from gpu_runtime.verified_backend import VerifiedReplayBackend
        learner = self.coordinator(); step = learner.train_next()
        cp = self.runtime.learner_releases.load_checkpoint(step["checkpoint_sha256"])
        torch = self.backend.torch
        original = torch.load(io.BytesIO(base64.b64decode(cp["backend_state"]["payload"])), weights_only=True)
        base = torch.nn.Linear(4, 4).to(dtype=torch.bfloat16)
        class Layer(torch.nn.Module):
            def get_base_layer(self):
                return base
        model = torch.nn.Module(); model.add_module("projection", Layer())
        bootstrap = {"projection.lora_A.weight": torch.zeros(1, 4), "projection.lora_B.weight": torch.zeros(4, 1)}
        with patch.object(self.backend, "model", model), patch.object(self.backend, "_adapter_state_dict", return_value=bootstrap), \
             patch.object(self.backend, "_checkpoint_adapter_specs", side_effect=lambda: VerifiedReplayBackend._checkpoint_adapter_specs(self.backend)):
            for dtype in (torch.bfloat16, torch.float32, torch.float16):
                payload = copy.deepcopy(original)
                payload["adapter"] = {"projection.lora_A.weight": original["adapter"]["weight"][:1].to(dtype),
                    "projection.lora_B.weight": original["adapter"]["weight"][:, :1].to(dtype)}
                stream = io.BytesIO(); torch.save(payload, stream)
                snapshot = {**cp["backend_state"], "payload": base64.b64encode(stream.getvalue()).decode()}
                if dtype != torch.bfloat16:
                    with self.assertRaisesRegex(ValueError, "shape/dtype"):
                        self.backend.extract_checkpoint_weights(cp["logical_state"], snapshot)
                    continue
                weights = self.backend.extract_checkpoint_weights(cp["logical_state"], snapshot)
                decoded = torch.load(io.BytesIO(base64.b64decode(weights["payload"])), weights_only=True)
                for name, tensor in payload["adapter"].items():
                    self.assertEqual(decoded["adapter"][name].dtype, torch.bfloat16)
                    self.assertTrue(torch.equal(decoded["adapter"][name], tensor))

    def test_writer_lease_prevents_two_owners_and_creation_failure_cleans_only_own_state(self):
        learner = self.coordinator()
        with self.assertRaises(OSError):
            _WriterLease(self.runtime.learner_releases.root, learner.run_sha)
        from gpu_runtime import learner as module
        write = module._write
        def fail_ack(path, value):
            if path.name == "created.json":
                raise OSError("ack unavailable")
            return write(path, value)
        with patch.object(module, "_write", side_effect=fail_ack), self.assertRaisesRegex(OSError, "ack unavailable"):
            self.coordinator("failed")
        self.assertNotIn("failed", self.backend.sessions)
        self.assertIn("central", self.backend.sessions)

    def test_gpu_worker_control_flow_on_tiny_cpu_fixture(self):
        self.worker_fixture()

    def test_final_continuation_failure_cannot_leave_success_report(self):
        self.worker_fixture(fail_continuation=True)

    def test_gpu_worker_resume_never_repeats_first_committed_update(self):
        self.worker_fixture(resume=True)

    def worker_fixture(self, *, fail_continuation=False, resume=False):
        # Hardware and admission are explicitly stubbed: only worker control
        # flow and real tiny gradients/restores are covered by this local test.
        from containers.gpu import smoke_learner_loop as probe
        from gpu_runtime import learner as coordinator_module
        root = self.root/"worker-output"
        resume_args = []
        if resume:
            seed = LearnerCoordinator(self.runtime, learner_id="central-verified", dataset_pins=[PIN],
                journal_root=self.root/"prior-run", batch_size=2)
            first = seed.train_next(); seed.close(); self.runtime.delete_session("central-verified")
            resume_args = ["--resume-checkpoint", first["checkpoint_sha256"],
                "--learner-store", str(self.runtime.learner_releases.root)]
        original_save = probe.save
        def save(output, name, value):
            if fail_continuation and name == "final-continuation.json":
                raise OSError("continuation write unavailable")
            return original_save(output, name, value)
        def command(output, name, deadline):
            ready = json.loads((output/("ready.json" if name == "append-and-train.json" else "ready2.json")).read_bytes())
            sid = "collected" if name == "append-and-train.json" else "new-actor"
            self.runtime.create_session(sid, theorem_id="c"*64,
                model_release_sha256=ready["release"]["model_release_sha256"])
            return ({"dataset_sha256": NEW_PIN, "actor_session_id": sid}
                if name == "append-and-train.json" else {"actor_session_id": sid})
        def new_descriptor(backend, pin):
            result = coordinator_module.describe_dataset(backend, pin)
            result.update(source_session_id="collected", source_model_release_sha256=
                self.runtime.sessions.get("collected").lineage["model_release_sha256"])
            return result
        with patch.object(probe, "VerifiedReplayBackend", return_value=self.backend), \
             patch.object(probe, "GpuRuntime", return_value=self.runtime), \
             patch.object(probe, "load_verified_dataset", return_value=self.backend._test_dataset), \
             patch.object(probe, "describe_dataset", side_effect=new_descriptor), \
             patch.object(probe, "base_fingerprint", return_value={"explicit_tiny_fixture": True}), \
             patch.object(probe, "save", side_effect=save), \
             patch.object(self.runtime, "learn", wraps=self.runtime.learn) as learn, \
             patch.object(probe, "wait_command", side_effect=command), \
             patch.object(probe, "require_amd"):
            result = probe.main(["--model-path", "unused", "--dataset-root", str(self.backend.dataset_root),
                "--dataset-sha256", PIN, "--output-dir", str(root), "--port", "0", *resume_args])
        self.assertEqual(learn.call_count, 1 if resume else 2)
        report = json.loads((root/"report.json").read_bytes())
        if fail_continuation:
            self.assertEqual(result, 1)
            self.assertFalse(report["ok"])
            self.assertEqual(report["error"]["message"], "continuation write unavailable")
            return
        self.assertEqual(result, 0, report.get("error"))
        self.assertTrue(all(report["gates"].values()))
        self.assertEqual(report["second"]["step"], 2)


if __name__ == "__main__":
    unittest.main()
