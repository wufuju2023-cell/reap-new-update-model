"""Same-runtime terminal snapshot reuse mechanisms; no real GPU speed claims."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_runtime import GpuRuntime, ToyBackend
from gpu_runtime.errors import SnapshotIntegrityError
from tests.test_experience import acceptance
from tests.test_gpu_runtime import chat_request


class SnapshotReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.backend = ToyBackend()
        self.runtime = GpuRuntime(backend=self.backend, snapshot_root=self.root, max_resident_sessions=2)
        self.addCleanup(self.runtime.close)
        self.runtime.create_session("done", theorem_id="problem-a")

    def snapshot(self, name="after-online-ttt"):
        return self.runtime.snapshot("done", name)

    def retire(self, name="after-online-ttt", version=0):
        return self.runtime.retire_session("done", name, expected_policy_version=version, reuse_snapshot=True)

    def assert_not_retired(self):
        self.assertIn("done", self.backend._states)
        self.assertNotIn("done", self.runtime._retired_session_ids)
        self.assertFalse((self.root / "_retirements/done").exists())

    def test_reuses_exact_bytes_without_export_or_load_and_preserves_default_path(self):
        path = self.snapshot()
        original = {p.name: p.read_bytes() for p in path.iterdir()}
        revision = self.runtime._snapshot_witnesses["done"]["revision"]
        with patch.object(self.backend, "export_session", side_effect=AssertionError("re-export")), \
                patch.object(self.runtime.snapshots, "create", side_effect=AssertionError("re-snapshot")), \
                patch.object(self.runtime.snapshots, "load", side_effect=AssertionError("materialized backend")):
            receipt = self.retire()
        self.assertEqual(receipt["schema_version"], "reap.gpu.retirement.v2")
        self.assertEqual(receipt["snapshot_mode"], "reuse_verified")
        self.assertEqual(receipt["snapshot_revision"], revision)
        self.assertEqual(receipt["snapshot_sha256"], hashlib.sha256(original["manifest.json"]).hexdigest())
        self.assertEqual(original, {p.name: p.read_bytes() for p in path.iterdir()})
        self.assertEqual(list((self.root / "done").iterdir()), [path])
        self.assertNotIn("done", self.backend._states)
        metrics = self.runtime.actor.metrics()
        self.assertEqual(self.runtime.retirement_receipt("done"), receipt)
        self.assertEqual(metrics, self.runtime.actor.metrics())
        self.runtime.create_session("next")
        default = self.runtime.retire_session("next", "fresh", expected_policy_version=0)
        self.assertEqual(default["schema_version"], "reap.gpu.retirement.v1")
        self.assertNotIn("snapshot_mode", default)

    def test_missing_old_and_restarted_runtime_witnesses_refuse_without_fallback(self):
        with self.assertRaisesRegex(ValueError, "same-runtime witness"):
            self.retire()
        self.assert_not_retired()
        self.snapshot("old")
        self.snapshot()
        with self.assertRaisesRegex(ValueError, "same-runtime witness"):
            self.retire("old")
        replacement = GpuRuntime(backend=self.backend, sessions=self.runtime.sessions, snapshot_root=self.root)
        self.addCleanup(replacement.close)
        with self.assertRaisesRegex(ValueError, "same-runtime witness"):
            replacement.retire_session("done", "after-online-ttt", expected_policy_version=0, reuse_snapshot=True)
        self.assert_not_retired()

    def test_any_same_session_backend_operation_invalidates_even_without_version_change(self):
        operations = {
            "policy": lambda: self.runtime.policy("done", chat_request()),
            "value": lambda: self.runtime.value("done", chat_request()),
            "inspect": lambda: self.runtime.inspect_backend("done"),
            "restore": lambda: self.runtime.restore("done", "after-online-ttt"),
        }
        for label, operation in operations.items():
            with self.subTest(label=label):
                name = "snap-" + label
                # restore uses the separately retained first snapshot.
                if label == "policy":
                    self.snapshot()
                self.snapshot(name)
                operation()
                self.assertEqual(self.runtime.sessions._states["done"].policy_version, 0)
                with self.assertRaisesRegex(ValueError, "same-runtime witness"):
                    self.retire(name)
                self.assert_not_retired()

    def test_failed_backend_call_invalidates_before_invocation(self):
        for label in ("policy", "value", "export_session"):
            with self.subTest(label=label):
                name = "failed-" + label
                self.snapshot(name)
                with patch.object(self.backend, label, side_effect=RuntimeError("backend failed")):
                    with self.assertRaisesRegex(RuntimeError, "backend failed"):
                        if label == "export_session":
                            self.runtime.inspect_backend("done")
                        else:
                            getattr(self.runtime, label)("done", chat_request())
                with self.assertRaisesRegex(ValueError, "same-runtime witness"):
                    self.retire(name)
                self.assert_not_retired()

    def test_learn_and_idempotent_or_stale_learn_invalidate(self):
        event = {"event_id": "one", "reward": 1}
        self.snapshot()
        self.runtime.learn("done", expected_policy_version=0, event=event)
        with self.assertRaisesRegex(ValueError, "same-runtime witness"):
            self.retire(version=1)
        self.snapshot("learned")
        receipt = self.runtime.learn("done", expected_policy_version=0, event=event)
        self.assertTrue(receipt["idempotent"])
        with self.assertRaisesRegex(ValueError, "same-runtime witness"):
            self.retire("learned", version=1)
        self.snapshot("stale")
        with self.assertRaises(Exception):
            self.runtime.learn("done", expected_policy_version=0, event={"event_id": "other", "reward": 1})
        with self.assertRaisesRegex(ValueError, "same-runtime witness"):
            self.retire("stale", version=1)

    def test_other_session_activity_and_readonly_logical_reads_preserve_witness(self):
        self.snapshot()
        self.runtime.create_session("other")
        self.runtime.policy("other", chat_request())
        self.runtime.value("other", chat_request())
        self.runtime.learn("other", expected_policy_version=0, event={"event_id": "one", "reward": 1})
        self.runtime.snapshot("other", "snap")
        with self.runtime.sessions.locked("done") as state:
            self.assertEqual(state.policy_version, 0)
        self.assertEqual(self.retire()["status"], "released")
        self.assertIn("other", self.backend._states)

    def test_failed_learn_rollback_does_not_restore_reuse_witness(self):
        self.snapshot()
        before = self.backend.export_session("done")
        original_learn = self.backend.learn
        def fail_after_mutation(sid, event):
            original_learn(sid, event)
            raise RuntimeError("failure after mutation")
        with patch.object(self.backend, "learn", side_effect=fail_after_mutation):
            with self.assertRaisesRegex(RuntimeError, "failure after mutation"):
                self.runtime.learn("done", expected_policy_version=0, event={"event_id": "failed", "reward": 1})
        self.assertEqual(self.backend.export_session("done"), before)
        self.assertEqual(self.runtime.sessions._states["done"].policy_version, 0)
        with self.assertRaisesRegex(ValueError, "same-runtime witness"):
            self.retire()
        self.assert_not_retired()

    def test_logical_metadata_and_numeric_type_changes_refuse(self):
        for index, replacement in enumerate((True, 0, {"wrong": 1})):
            with self.subTest(replacement=replacement):
                with self.runtime.sessions.locked("done") as state:
                    state.completed = False
                name = "logical-" + str(index)
                self.snapshot(name)
                with self.runtime.sessions.locked("done") as state:
                    state.completed = replacement
                with self.assertRaisesRegex(ValueError, "logical state differs"):
                    self.retire(name)
                self.assert_not_retired()

    def test_all_files_and_manifest_are_hashed_not_just_policy_version(self):
        for index, filename in enumerate(("backend.json", "session.json", "manifest.json")):
            with self.subTest(filename=filename):
                name = "corrupt-" + str(index)
                path = self.snapshot(name) / filename
                raw = path.read_bytes()
                path.write_bytes(raw[:-1] + (b"X" if raw[-1:] != b"X" else b"Y"))
                with patch.object(self.runtime.snapshots, "create", side_effect=AssertionError("fallback")):
                    with self.assertRaises(SnapshotIntegrityError):
                        self.retire(name)
                self.assert_not_retired()

    def test_rewritten_valid_manifest_cannot_replace_trusted_runtime_hash(self):
        path = self.snapshot()
        backend = path / "backend.json"
        backend.write_bytes(backend.read_bytes() + b" ")
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["files"]["backend.json"] = {"bytes": backend.stat().st_size,
            "sha256": hashlib.sha256(backend.read_bytes()).hexdigest()}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(SnapshotIntegrityError, "runtime witness"):
            self.retire()
        self.assert_not_retired()

    def test_verifier_streams_large_backend_payload_without_read_bytes_or_json_decode(self):
        # A large toy export exercises streaming without allocating model tensors.
        original_export = self.backend.export_session
        with patch.object(self.backend, "export_session", side_effect=lambda sid: {
                **original_export(sid), "padding": "z" * (3 * 1024 * 1024)}):
            path = self.snapshot()
        witness = self.runtime._snapshot_witnesses["done"]
        original_open = Path.open
        sizes = []
        class Reader:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            def read(self, size=-1):
                sizes.append(size)
                self.assert_bounded(size)
                return self.stream.read(size)
            @staticmethod
            def assert_bounded(size):
                if not 0 < size <= 1024 * 1024:
                    raise AssertionError("unbounded read")
        def opened(p, *args, **kwargs):
            stream = original_open(p, *args, **kwargs)
            return Reader(stream) if p == path / "backend.json" else stream
        with patch.object(Path, "read_bytes", side_effect=AssertionError("read_bytes")), \
                patch.object(Path, "open", opened):
            logical = self.runtime.snapshots.verify_for_reuse("done", "after-online-ttt",
                expected_manifest_sha256=witness["manifest_sha256"])
        self.assertEqual(logical, witness["logical"])
        self.assertGreaterEqual(len(sizes), 4)

    def test_delete_recreate_cannot_reuse_prior_identity_snapshot(self):
        self.snapshot()
        self.runtime.delete_session("done")
        self.runtime.create_session("done", theorem_id="problem-a")
        with self.assertRaisesRegex(ValueError, "same-runtime witness"):
            self.retire()
        self.assert_not_retired()

    def test_failed_snapshot_attempt_and_publish_attempt_invalidate_previous_witness(self):
        self.snapshot()
        with self.assertRaises(Exception):
            self.snapshot()  # Existing immutable name cannot be overwritten.
        with self.assertRaisesRegex(ValueError, "same-runtime witness"):
            self.retire()
        self.snapshot("later")
        with self.assertRaisesRegex(ValueError, "completed experience candidate"):
            self.runtime.publish_experience("done", "later", "experience", {})
        with self.assertRaisesRegex(ValueError, "same-runtime witness"):
            self.retire("later")

    def test_revision_mismatch_and_pending_events_refuse_reuse(self):
        self.snapshot()
        self.runtime._session_revisions["done"] += 1
        with self.assertRaisesRegex(ValueError, "same-runtime witness"):
            self.retire()
        with self.runtime.sessions.locked("done") as state:
            state.buffer_metadata["pending_event_ids"] = ["unresolved"]
        self.snapshot("pending")
        with self.assertRaisesRegex(ValueError, "pending"):
            self.retire("pending")
        self.assert_not_retired()

    def test_prepared_failure_quarantines_and_never_retries_or_resnapshots(self):
        self.snapshot()
        with patch("gpu_runtime.runtime._write_retirement_record", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                self.retire()
        self.assertIn("done", self.backend._states)
        self.assertIn("done", self.runtime._unusable_sessions)
        with self.assertRaisesRegex(ValueError, "already attempted"):
            self.retire()
        self.assertNotIn("done", self.runtime._snapshot_witnesses)

    def test_reused_sealed_source_can_publish_after_retirement(self):
        self.runtime.learn("done", expected_policy_version=0, event={"event_id": "one", "reward": 1})
        self.runtime.snapshot("done", "candidate", for_experience=True)
        # Acceptance helper only builds local metadata and does not call the backend.
        accepted = acceptance(self.runtime, sid="done", name="candidate")
        self.retire("candidate", version=1)
        release = self.runtime.publish_experience("done", "candidate", "experience", accepted)
        self.assertEqual(release["source"]["snapshot"], "candidate")

    def test_post_delete_receipt_failure_keeps_prepared_v2_and_forbids_retry(self):
        from gpu_runtime import runtime as module
        original_write = module._write_retirement_record
        def write(root, sid, name, record):
            if name == "released.json":
                raise OSError("release acknowledgement cannot be persisted")
            return original_write(root, sid, name, record)
        self.snapshot()
        with patch.object(module, "_write_retirement_record", side_effect=write):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                self.retire()
        self.assertNotIn("done", self.backend._states)
        receipt = self.runtime.retirement_receipt("done")
        self.assertEqual(receipt["schema_version"], "reap.gpu.retirement.v2")
        self.assertEqual(receipt["status"], "prepared")
        with self.assertRaisesRegex(ValueError, "already attempted"):
            self.retire()

    def test_v2_receipt_reader_rejects_mode_revision_and_schema_tampering(self):
        self.snapshot()
        receipt = self.retire()
        path = self.root / "_retirements/done/released.json"
        for key, value in (("snapshot_revision", True), ("snapshot_revision", 0),
                           ("snapshot_mode", "fresh"), ("schema_version", "reap.gpu.retirement.v1")):
            with self.subTest(key=key, value=value):
                path.write_text(json.dumps({**receipt, key: value}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.runtime.retirement_receipt("done")

    def test_reuse_flag_requires_boolean(self):
        self.snapshot()
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                self.runtime.retire_session("done", "after-online-ttt", expected_policy_version=0, reuse_snapshot=value)
        self.assert_not_retired()


class SnapshotReuseProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def make_trained_source(self):
        from tests.test_verified_backend import VerifiedBackendTests, event
        backend = VerifiedBackendTests.backend(self)
        runtime = GpuRuntime(backend=backend, snapshot_root=self.root / "source")
        self.addCleanup(runtime.close)
        runtime.create_session("learner")
        runtime.learn("learner", expected_policy_version=0, event=event())
        path = runtime.snapshot("learner", "after")
        runtime.delete_session("learner")
        return backend, path, hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()

    def test_actual_tiny_trained_snapshot_restore_reuse_and_capacity_without_probe_learn(self):
        from containers.gpu import smoke_snapshot_reuse as probe
        from tests.test_policy_scoring_probe import FakeMeter
        backend, source, pin = self.make_trained_source()
        out = self.root / "probe"
        out.mkdir()
        prepared = probe.prepare_source(source, out, pin, session_id="learner", expected_policy_version=1)
        with GpuRuntime(backend=backend, snapshot_root=out / "snapshots", max_resident_sessions=1) as runtime, \
                patch.object(backend, "learn", side_effect=AssertionError("probe trained")), \
                patch.object(backend, "policy", side_effect=AssertionError("probe generated")):
            result = probe.audit(runtime, backend, prepared, out, meter=FakeMeter())
        self.assertTrue(result["ok"], result)
        self.assertTrue(all(result["gates"].values()))
        self.assertEqual(result["reuse_export_calls"], 0)
        self.assertEqual(result["retirement_receipt"]["policy_version"], 1)
        self.assertEqual(probe.snapshot_pins(source), prepared["files"])
        self.assertTrue((out / "retirement-intent.json").exists())

    def test_probe_rejects_source_pin_or_corruption_before_restore(self):
        from containers.gpu import smoke_snapshot_reuse as probe
        _, source, pin = self.make_trained_source()
        out = self.root / "probe"
        out.mkdir()
        with self.assertRaisesRegex(RuntimeError, "pin mismatch"):
            probe.prepare_source(source, out, "0" * 64, session_id="learner", expected_policy_version=1)
        self.assertFalse((out / "snapshots").exists())
        with (source / "backend.json").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaises(SnapshotIntegrityError):
            probe.prepare_source(source, out, pin, session_id="learner", expected_policy_version=1)
        self.assertFalse((out / "snapshots").exists())

    def test_probe_output_is_exclusive_and_unknown_retire_is_not_retried(self):
        from containers.gpu import smoke_snapshot_reuse as probe
        from tests.test_policy_scoring_probe import FakeMeter
        backend, source, pin = self.make_trained_source()
        out = self.root / "probe"
        out.mkdir()
        prepared = probe.prepare_source(source, out, pin, session_id="learner", expected_policy_version=1)
        with self.assertRaises(FileExistsError):
            probe.prepare_source(source, out, pin, session_id="learner", expected_policy_version=1)
        with GpuRuntime(backend=backend, snapshot_root=out / "snapshots", max_resident_sessions=1) as runtime, \
                patch.object(runtime, "retire_session", side_effect=TimeoutError("unknown")) as retire:
            with self.assertRaisesRegex(TimeoutError, "unknown"):
                probe.audit(runtime, backend, prepared, out, meter=FakeMeter())
            self.assertEqual(retire.call_count, 1)
        self.assertTrue((out / "retirement-intent.json").exists())
        self.assertFalse((out / "retirement-receipt.json").exists())

    def test_probe_main_refuses_cpu_and_persists_failure_without_loading_backend(self):
        from containers.gpu import smoke_snapshot_reuse as probe
        backend, source, pin = self.make_trained_source()
        model = self.root / "model"
        model.mkdir()
        (model / "reap-model-lock.json").write_text(json.dumps({"schema_version": "reap.model-lock.v2",
            "repo": "FrenzyMath/REAL-Prover", "revision": "fe76f68d9a88f342cb7b546307c20292fea9cced",
            "hidden_size": 3584}), encoding="utf-8")
        out = self.root / "main-probe"
        with patch.object(backend.torch.cuda, "is_available", return_value=False), \
                patch.object(probe, "VerifiedReplayBackend") as constructor, patch("builtins.print"):
            code = probe.main(["--model-path", str(model), "--dataset-root", str(backend.dataset_root),
                "--source-snapshot", str(source), "--source-manifest-sha256", pin,
                "--session-id", "learner", "--output-dir", str(out)])
        constructor.assert_not_called()
        self.assertEqual(code, 1)
        report = json.loads((out / "report.json").read_bytes())
        self.assertFalse(report["ok"] or report["real_7B_GPU_gate_passed"])
        self.assertIn("HIP", report["error"]["message"])


if __name__ == "__main__":
    unittest.main()
