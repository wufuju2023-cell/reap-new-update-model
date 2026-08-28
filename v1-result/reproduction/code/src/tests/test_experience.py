"""Local mechanism gates only. No 7B weights, network, or GPU required."""
from copy import deepcopy
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_runtime import GpuRuntime, ToyBackend, SnapshotIntegrityError
from gpu_runtime.errors import SessionNotFoundError, SnapshotNotFoundError
from tests.test_gpu_runtime import chat_request


def acceptance(runtime, sid="source", name="candidate", kind="local-mechanism"):
    state, _ = runtime.snapshots.load(sid, name)
    manifest = runtime.snapshots.root / sid / name / "manifest.json"
    return {"completed": True, "passed": True, "kind": kind, "evidence_sha256": "a" * 64,
        "source": {"session_id": sid, "theorem_id": state["theorem_id"], "snapshot": name,
                   "policy_version": state["policy_version"],
                   "snapshot_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   "parent_experience_id": state["lineage"].get("experience_id")}}


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.backend = ToyBackend()
        self.runtime = GpuRuntime(backend=self.backend, snapshot_root=Path(self.tmp.name))
        self.addCleanup(self.runtime.close)

    def source(self):
        r = self.runtime
        r.create_session("source", theorem_id="problem-a")
        r.learn("source", expected_policy_version=0, event={"event_id": "a1", "reward": 1})
        r.snapshot("source", "candidate", for_experience=True)

    def publish(self):
        self.source()
        return self.runtime.publish_experience("source", "candidate", "exp-1", acceptance(self.runtime))

    def test_parameters_inherited_but_version_optimizer_buffer_and_events_are_fresh(self):
        r = self.runtime
        r.create_session("concurrent")
        unchanged = r.inspect_backend("concurrent")
        release = self.publish()
        original = r.experiences.load("exp-1")
        dest = r.create_session("next", theorem_id="problem-b", experience_id="exp-1")
        self.assertEqual(dest["policy_version"], 0)
        self.assertEqual(dest["optimizer_metadata"]["steps"], 0)
        self.assertEqual(dest["event_receipts"], {})
        self.assertEqual(dest["buffer_metadata"], {"events": {}, "pending_event_ids": [], "consumed_event_ids": []})
        self.assertEqual(dest["lineage"]["source"]["policy_version"], 1)
        self.assertEqual(dest["lineage"]["weights_sha256"], release["weights_sha256"])
        self.assertEqual(r.policy("next", chat_request())["choices"][0]["message"]["content"], "trivial")
        self.assertEqual(r.inspect_backend("next")["value_score"], 1)
        r.snapshot("next", "initial")
        r.learn("next", expected_policy_version=0, event={"event_id": "b1", "reward": 1})
        r.restore("next", "initial")
        self.assertEqual(r.sessions.get("next").policy_version, 0)
        self.assertEqual(r.sessions.get("next").lineage, dest["lineage"])
        self.assertEqual(r.inspect_backend("concurrent"), unchanged)
        self.assertEqual(r.experiences.load("exp-1"), original)
        r.create_session("no-source")
        self.assertEqual(r.inspect_backend("no-source"), unchanged)

    def test_source_sealed_snapshot_and_release_never_overwritten(self):
        self.publish()
        r = self.runtime
        with self.assertRaisesRegex(ValueError, "completed"):
            r.learn("source", expected_policy_version=1, event={"event_id": "a2"})
        with self.assertRaisesRegex(ValueError, "completed"):
            r.restore("source", "candidate")
        with self.assertRaises(FileExistsError):
            r.snapshot("source", "candidate")
        with self.assertRaises(FileExistsError):
            r.publish_experience("source", "candidate", "exp-1", acceptance(r))
        r.delete_session("source")
        # Published parameters remain available after the source process/session is gone.
        r.create_session("later", theorem_id="problem-c", experience_id="exp-1")

    def test_content_pins_checked_before_allocation_and_capacity_reservation(self):
        release = self.publish()
        pins = {"experience_weights_sha256": release["weights_sha256"],
                "experience_snapshot_sha256": release["source"]["snapshot_sha256"]}
        before = set(self.runtime._resident_session_ids)
        for field in pins:
            with self.subTest(field=field), patch.object(self.backend, "create_session") as create:
                with self.assertRaisesRegex(ValueError, "pinned release"):
                    self.runtime.create_session("bad", theorem_id="b", experience_id="exp-1",
                                                **{**pins, field: "0" * 64})
                create.assert_not_called()
                self.assertEqual(self.runtime._resident_session_ids, before)
                self.assertNotIn("bad", self.backend._states)
        created = self.runtime.create_session("next", theorem_id="b", experience_id="exp-1", **pins)
        self.assertEqual(created["lineage"]["weights_sha256"], pins["experience_weights_sha256"])

    def test_invalid_or_unpaired_content_pins_reject_before_store_access(self):
        for kwargs in ({"experience_weights_sha256": "a" * 64},
                       {"experience_id": "exp", "experience_weights_sha256": "a" * 64},
                       {"experience_id": "exp", "experience_weights_sha256": "invalid",
                        "experience_snapshot_sha256": "a" * 64}):
            with self.subTest(kwargs=kwargs), patch.object(self.runtime.experiences, "load") as load:
                with self.assertRaises(ValueError):
                    self.runtime.create_session("bad", theorem_id="b", **kwargs)
                load.assert_not_called()

    def test_missing_corrupt_incompatible_and_wrong_identities_rejected_without_creation(self):
        r = self.runtime
        self.publish()
        for kwargs in ({"experience_id": "missing", "theorem_id": "b"},
                       {"experience_id": "exp-1"},
                       {"experience_id": "exp-1", "theorem_id": "problem-a"}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, SnapshotNotFoundError)):
                r.create_session("bad", **kwargs)
            self.assertNotIn("bad", self.backend._states)
        with self.assertRaises(ValueError):
            r.create_session("source", theorem_id="different", experience_id="exp-1")
        with patch.object(self.backend, "experience_contract", return_value={"base": "wrong"}):
            with self.assertRaisesRegex(ValueError, "compatibility"):
                r.create_session("bad", theorem_id="b", experience_id="exp-1")
        path = r.experiences.snapshots.root / "exp-1" / "release" / "backend.json"
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaises(SnapshotIntegrityError):
            r.create_session("bad", theorem_id="b", experience_id="exp-1")
        self.assertNotIn("bad", self.backend._states)

    def test_acceptance_binds_completed_source_and_exact_evidence(self):
        r = self.runtime
        self.source()
        valid = acceptance(r)
        for field, value in (("completed", False), ("passed", False), ("kind", "unverified"),
                             ("source", {}), ("evidence_sha256", "bad")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                r.publish_experience("source", "candidate", "bad", {**valid, field: value})
        self.assertFalse((r.experiences.snapshots.root / "bad").exists())
        r.create_session("old", theorem_id="old-problem")
        r.learn("old", expected_policy_version=0, event={"event_id": "e"})
        r.snapshot("old", "old-format")
        with self.assertRaisesRegex(ValueError, "candidate"):
            r.publish_experience("old", "old-format", "bad", valid)

    def test_initialization_failure_removes_partial_backend_and_allows_clean_retry(self):
        self.publish()
        original = self.backend.initialize_from_experience
        def fail(sid, weights):
            original(sid, weights)
            raise RuntimeError("injected after parameter copy")
        with patch.object(self.backend, "initialize_from_experience", side_effect=fail):
            with self.assertRaises(RuntimeError):
                self.runtime.create_session("next", theorem_id="b", experience_id="exp-1")
        self.assertNotIn("next", self.backend._states)
        with self.assertRaises(SessionNotFoundError):
            self.runtime.sessions.get("next")
        self.runtime.create_session("next", theorem_id="b", experience_id="exp-1")

    def test_disk_failure_does_not_seal_or_publish_partial_release(self):
        r = self.runtime
        r.create_session("source", theorem_id="a")
        r.learn("source", expected_policy_version=0, event={"event_id": "e"})
        with patch("gpu_runtime.snapshot_store._write_durable", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                r.snapshot("source", "candidate", for_experience=True)
        self.assertFalse(r.sessions.get("source").completed)
        self.assertFalse((r.snapshots.root / "source" / "candidate").exists())
        r.snapshot("source", "candidate", for_experience=True)
        with patch("gpu_runtime.snapshot_store._write_durable", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                r.publish_experience("source", "candidate", "bad", acceptance(r))
        self.assertFalse((r.experiences.snapshots.root / "bad" / "release").exists())
        self.assertEqual(list((r.experiences.snapshots.root / "bad").iterdir()), [])

    def test_failed_initialization_cleanup_quarantines_residual_backend(self):
        self.publish()
        with patch.object(self.backend, "initialize_from_experience", side_effect=RuntimeError("copy failed")), \
             patch.object(self.backend, "delete_session", side_effect=RuntimeError("cleanup failed")):
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                self.runtime.create_session("bad", theorem_id="b", experience_id="exp-1")
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            self.runtime.inspect_backend("bad")
        with self.assertRaises(SessionNotFoundError):
            self.runtime.sessions.get("bad")


class TensorExperienceTests(unittest.TestCase):
    def test_exact_lora_namespace_export_and_optimizer_scope(self):
        from types import SimpleNamespace
        from gpu_runtime.real_backend import RealProverBackend
        import torch
        state = {}
        for sid in ("s", "s-control", "xs", "model"):
            for kind in ("lora_A", "lora_B"):
                state[f"base_model.model.layer.q_proj.{kind}.{sid}.weight"] = torch.nn.Parameter(torch.ones(2, 2))
        state["base_model.model.layer.q_proj.base_layer.weight"] = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
        b = RealProverBackend.__new__(RealProverBackend)
        b.model = SimpleNamespace(state_dict=lambda: state, named_parameters=lambda: state.items())
        for sid in ("s", "s-control", "xs", "model"):
            exported = b._adapter_state_dict(sid)
            self.assertEqual(set(exported), {f"base_model.model.layer.q_proj.{kind}.weight" for kind in ("lora_A", "lora_B")})
            selected = b._adapter_parameters(sid)
            self.assertEqual({id(p) for p in selected}, {id(state[f"base_model.model.layer.q_proj.{kind}.{sid}.weight"]) for kind in ("lora_A", "lora_B")})
            for kind in ("lora_A", "lora_B"):
                self.assertIs(exported[f"base_model.model.layer.q_proj.{kind}.weight"], state[f"base_model.model.layer.q_proj.{kind}.{sid}.weight"])
        with self.assertRaises(RuntimeError):
            b._adapter_state_dict("missing")
        self.assertFalse(state["base_model.model.layer.q_proj.base_layer.weight"].requires_grad)

    def test_real_adapter_ids_rejected_before_model_mutation(self):
        from gpu_runtime.real_backend import RealProverBackend
        b = RealProverBackend.__new__(RealProverBackend)
        for sid in ("a.b", "__bootstrap__"):
            with self.subTest(sid=sid), self.assertRaisesRegex(ValueError, "session IDs"):
                b.create_session(sid)

    def test_base_identity_hashes_actual_frozen_tensors_and_tokenizer_not_paths(self):
        from types import SimpleNamespace
        from gpu_runtime.real_backend import RealProverBackend
        b = self.backend()
        torch = b.torch
        del b._experience_base_digest
        b.model = torch.nn.Linear(4, 4).requires_grad_(False)
        b.model.register_parameter("lora_ignored", torch.nn.Parameter(torch.ones(1)))
        b.model.config = SimpleNamespace(to_dict=lambda: {"hidden_size": 4, "_name_or_path": "/old/path"})
        b.tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: "fixed tokenizer"))
        first = RealProverBackend.experience_contract(b)
        del b._experience_base_digest
        b.model.config = SimpleNamespace(to_dict=lambda: {"hidden_size": 4, "_name_or_path": "/new/path"})
        with torch.no_grad():
            b.model.lora_ignored.add_(1)
        self.assertEqual(first, RealProverBackend.experience_contract(b))
        del b._experience_base_digest
        with torch.no_grad():
            b.model.weight.add_(1)
        self.assertNotEqual(first, RealProverBackend.experience_contract(b))

    def test_gpu_acceptance_script_logic_on_cpu_tensors_is_not_gpu_evidence(self):
        from containers.gpu.smoke_experience_gpu import audit
        from tests.test_search_objective import event
        b = self.backend()
        with tempfile.TemporaryDirectory() as root, GpuRuntime(backend=b, snapshot_root=Path(root)) as r:
            r.create_session("search-a", theorem_id="a")
            r.learn("search-a", expected_policy_version=0, event=event())
            r.snapshot("search-a", "candidate", for_experience=True)
            # Synthetic attestation is confined to this temporary unit test.
            r.publish_experience("search-a", "candidate", "exp", acceptance(r, "search-a", kind="independent-lean"))
            def factory(sid, gamma):
                result = event()
                result.update(session_id=sid, tree_id=sid + ".tree0", event_id=sid + ".e0", gamma=gamma)
                return result
            result = audit(r, "exp", "next", "b", event_factory=factory)
            self.assertTrue(all(result["gates"].values()))
            self.assertFalse(result["lean_feedback_ttt_verified"])

    def backend(self):
        from tests.test_search_objective import SearchBackendCPUTests
        backend = SearchBackendCPUTests.make_backend(self)
        # Only identity computation is stubbed. Production transfer/shape checks,
        # optimizer, RNG, search objective, snapshot and runtime code are real.
        backend._experience_base_digest = "b" * 64
        backend.lora_config.lora_alpha = 2
        backend.lora_config.lora_dropout = 0.0
        return backend

    def test_real_tensor_copy_is_exact_and_private_state_is_not_inherited(self):
        from tests.test_search_objective import event
        b = self.backend()
        torch = b.torch
        with tempfile.TemporaryDirectory() as root, GpuRuntime(backend=b, snapshot_root=Path(root)) as r:
            r.create_session("search-a", theorem_id="a")
            r.create_session("other", theorem_id="other")
            other = b.session_fingerprints("other")
            r.learn("search-a", expected_policy_version=0, event=event())
            r.snapshot("search-a", "candidate", for_experience=True)
            r.publish_experience("search-a", "candidate", "exp", acceptance(r, "search-a", kind="independent-lean"))
            source = b.session_fingerprints("search-a")
            r.create_session("target", theorem_id="b", experience_id="exp")
            target = b.session_fingerprints("target")
            for key in ("adapter", "value_head"):
                self.assertEqual(source[key], target[key])
            self.assertEqual(b.sessions["target"].optimizer.state_dict()["state"], {})
            self.assertEqual(b.sessions["target"].optimizer_steps, 0)
            self.assertEqual(b.sessions["target"].examples_seen, 0)
            self.assertNotEqual(b.sessions["search-a"].rng_seed, b.sessions["target"].rng_seed)
            self.assertEqual(b.session_fingerprints("other"), other)
            r.snapshot("target", "inherited-initial")
            target_event = event()
            target_event.update(session_id="target", tree_id="target.tree0", event_id="target-step")
            r.learn("target", expected_policy_version=0, event=target_event)
            self.assertEqual(r.sessions.get("target").policy_version, 1)
            self.assertNotEqual(b.session_fingerprints("target")["adapter"], source["adapter"])
            self.assertEqual(b.session_fingerprints("search-a"), source)
            self.assertEqual(b.session_fingerprints("other"), other)
            r.restore("target", "inherited-initial")
            self.assertEqual(b.session_fingerprints("target"), target)
            original_learn = b.learn
            def fail_after_update(sid, ev):
                original_learn(sid, ev)
                raise RuntimeError("injected inherited update failure")
            with patch.object(b, "learn", side_effect=fail_after_update), self.assertRaises(RuntimeError):
                r.learn("target", expected_policy_version=0, event=target_event)
            self.assertEqual(b.session_fingerprints("target"), target)
            self.assertEqual(r.sessions.get("target").policy_version, 0)
            # Compare destination RNG with its own independent initialization.
            target_rng = b.sessions["target"].cpu_rng_state.clone()
            r.delete_session("target")
            r.create_session("target", theorem_id="b")
            self.assertTrue(torch.equal(target_rng, b.sessions["target"].cpu_rng_state))

    def test_shapes_dtypes_nonfinite_unknown_keys_and_objective_rejected(self):
        from tests.test_search_objective import event
        b = self.backend()
        with tempfile.TemporaryDirectory() as root, GpuRuntime(backend=b, snapshot_root=Path(root)) as r:
            r.create_session("search-a", theorem_id="a")
            r.learn("search-a", expected_policy_version=0, event=event())
            r.snapshot("search-a", "candidate", for_experience=True)
            _, raw = r.snapshots.load("search-a", "candidate")
            for field in ("session_id", "search_config", "experience_contract"):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    b.experience_weights("search-a", {**raw, field: "wrong"})
            weights = b.experience_weights("search-a", raw)
            for fault in ("shape", "dtype", "nan", "extra", "gamma", "base"):
                bad = deepcopy(weights)
                if fault in ("gamma", "base"):
                    if fault == "gamma":
                        bad["contract"]["search_config"]["gamma"] = 0.98
                    else:
                        bad["contract"]["base_sha256"] = "c" * 64
                else:
                    payload = b.torch.load(io.BytesIO(base64.b64decode(bad["payload"])), weights_only=True)
                    key = next(iter(payload["adapter"]))
                    if fault == "shape": payload["adapter"][key] = b.torch.zeros(1)
                    if fault == "dtype": payload["adapter"][key] = payload["adapter"][key].double()
                    if fault == "nan": payload["adapter"][key].fill_(float("nan"))
                    if fault == "extra": payload["optimizer"] = {}
                    buf = io.BytesIO()
                    b.torch.save(payload, buf)
                    bad["payload"] = base64.b64encode(buf.getvalue()).decode()
                r.create_session("target", theorem_id="b")
                before = b.session_fingerprints("target")
                with self.subTest(fault=fault), self.assertRaises((ValueError, FloatingPointError)):
                    b.initialize_from_experience("target", bad)
                self.assertEqual(before, b.session_fingerprints("target"))
                r.delete_session("target")


if __name__ == "__main__":
    unittest.main()
