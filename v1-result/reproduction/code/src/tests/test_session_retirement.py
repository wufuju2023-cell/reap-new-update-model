"""Local admission/retirement mechanisms; not GPU memory or crash-resume evidence."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_runtime import GpuRuntime, ToyBackend, VersionConflictError
from gpu_runtime.errors import SessionNotFoundError
from tests.test_experience import acceptance
from tests.test_gpu_runtime import chat_request


class SessionRetirementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.backend = ToyBackend()
        self.runtime = GpuRuntime(backend=self.backend, snapshot_root=self.root, max_resident_sessions=3)
        self.addCleanup(self.runtime.close)

    def receipt(self, sid, name):
        return json.loads((self.root / "_retirements" / sid / name).read_bytes())

    def test_capacity_is_atomic_under_concurrent_admission(self):
        def create(index):
            try:
                self.runtime.create_session(f"s{index}")
                return True
            except ValueError as exc:
                self.assertIn("capacity", str(exc))
                return False
        with ThreadPoolExecutor(max_workers=12) as pool:
            self.assertEqual(sum(pool.map(create, range(12))), 3)
        self.assertEqual(len(self.backend._states), 3)

    def test_twenty_tasks_hold_at_most_three_resident_sessions(self):
        peak = 0
        for start in range(0, 20, 3):
            group = [f"task-{index}" for index in range(start, min(start + 3, 20))]
            for sid in group:
                self.runtime.create_session(sid)
                peak = max(peak, len(self.backend._states))
            for sid in group:
                result = self.runtime.retire_session(sid, "final", expected_policy_version=0)
                self.assertEqual(result["status"], "released")
        self.assertEqual(peak, 3)
        self.assertEqual(self.backend._states, {})

    def test_retirement_preserves_snapshot_and_blocks_late_calls_and_identity_reuse(self):
        r = self.runtime
        r.create_session("done", theorem_id="a")
        r.learn("done", expected_policy_version=0, event={"event_id": "one", "reward": 1})
        before = r.inspect_backend("done")
        result = r.retire_session("done", "final", expected_policy_version=1)
        saved, backend = r.snapshots.load("done", "final")
        self.assertEqual(backend, before)
        self.assertEqual(saved["policy_version"], 1)
        self.assertEqual(result, self.receipt("done", "released.json"))
        self.assertEqual(result["snapshot_sha256"], hashlib.sha256((self.root / "done/final/manifest.json").read_bytes()).hexdigest())
        self.assertEqual(self.receipt("done", "prepared.json")["status"], "prepared")
        with self.assertRaises(SessionNotFoundError):
            r.policy("done", chat_request())
        with self.assertRaises(SessionNotFoundError):
            r.learn("done", expected_policy_version=0, event={"event_id": "late"})
        with self.assertRaisesRegex(ValueError, "retired"):
            r.create_session("done")
        with self.assertRaisesRegex(ValueError, "already attempted"):
            r.retire_session("done", "second", expected_policy_version=1)
        self.assertFalse((self.root / "done/second").exists())

    def test_sealed_source_can_be_published_after_retirement(self):
        r = self.runtime
        r.create_session("source", theorem_id="problem-a")
        r.learn("source", expected_policy_version=0, event={"event_id": "source-step", "reward": 1})
        r.snapshot("source", "candidate", for_experience=True)
        accepted = acceptance(r)
        original = {p.name: p.read_bytes() for p in (self.root / "source/candidate").iterdir()}
        r.retire_session("source", "final", expected_policy_version=1)
        r.publish_experience("source", "candidate", "experience", accepted)
        state = r.create_session("target", theorem_id="problem-b", experience_id="experience")
        self.assertEqual(state["policy_version"], 0)
        self.assertTrue(r.inspect_backend("target")["learned"])
        self.assertEqual(original, {p.name: p.read_bytes() for p in (self.root / "source/candidate").iterdir()})

    def test_invalid_version_pending_and_quarantine_do_not_snapshot_or_delete(self):
        r = self.runtime
        r.create_session("active")
        with self.assertRaises(VersionConflictError):
            r.retire_session("active", "final", expected_policy_version=1)
        for invalid in (True, -1, 1.2):
            with self.subTest(version=invalid), self.assertRaises(ValueError):
                r.retire_session("active", "final", expected_policy_version=invalid)
        with r.sessions.locked("active") as state:
            state.buffer_metadata["pending_event_ids"] = ["pending"]
        with self.assertRaisesRegex(ValueError, "pending"):
            r.retire_session("active", "final", expected_policy_version=0)
        r._unusable_sessions.add("active")
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            r.retire_session("active", "final", expected_policy_version=0)
        self.assertIn("active", self.backend._states)
        self.assertFalse((self.root / "active/final").exists())

    def test_snapshot_or_prepared_receipt_failure_never_deletes_live_state(self):
        r = self.runtime
        r.create_session("active")
        before = r.inspect_backend("active")
        with patch("gpu_runtime.snapshot_store._write_durable", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                r.retire_session("active", "snapshot-fail", expected_policy_version=0)
        self.assertEqual(r.inspect_backend("active"), before)
        self.assertNotIn("active", r._retired_session_ids)
        with patch("gpu_runtime.runtime._write_retirement_record", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                r.retire_session("active", "receipt-fail", expected_policy_version=0)
        self.assertEqual(self.backend.export_session("active"), before)
        self.assertEqual(r.sessions.get("active").policy_version, 0)
        self.assertIn("active", r._retired_session_ids)

    def test_failed_backend_deletion_quarantines_without_success_receipt(self):
        r = self.runtime
        r.create_session("failed")
        with patch.object(self.backend, "delete_session", side_effect=RuntimeError("delete failed")):
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                r.retire_session("failed", "final", expected_policy_version=0)
        self.assertIn("failed", self.backend._states)
        self.assertEqual(self.receipt("failed", "prepared.json")["status"], "prepared")
        self.assertFalse((self.root / "_retirements/failed/released.json").exists())
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            r.inspect_backend("failed")
        with self.assertRaisesRegex(ValueError, "retired"):
            r.create_session("failed")
        # Other sessions continue; unresolved residency still consumes its slot.
        r.create_session("other1")
        r.create_session("other2")
        with self.assertRaisesRegex(ValueError, "capacity"):
            r.create_session("over")

    def test_post_delete_receipt_failure_is_unknown_and_never_retried(self):
        from gpu_runtime.runtime import _write_retirement_record
        r = self.runtime
        r.create_session("failed")
        def write(root, sid, name, record):
            if name == "released.json":
                raise OSError("disk full")
            return _write_retirement_record(root, sid, name, record)
        with patch("gpu_runtime.runtime._write_retirement_record", side_effect=write):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                r.retire_session("failed", "final", expected_policy_version=0)
        self.assertNotIn("failed", self.backend._states)
        with self.assertRaisesRegex(ValueError, "already attempted"):
            r.retire_session("failed", "final", expected_policy_version=0)
        self.assertFalse((self.root / "_retirements/failed/released.json").exists())

    def test_failed_creation_cleanup_residue_counts_toward_capacity(self):
        r = self.runtime
        with patch.object(r.sessions, "create", side_effect=RuntimeError("logical store failed")), \
             patch.object(self.backend, "delete_session", side_effect=RuntimeError("cleanup failed")):
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                r.create_session("residue")
        with self.assertRaises(SessionNotFoundError):
            r.sessions.get("residue")
        r.create_session("one")
        r.create_session("two")
        with self.assertRaisesRegex(ValueError, "capacity"):
            r.create_session("over")

    def test_backend_create_failure_cleans_allocation_or_reserves_quarantine_slot(self):
        r = self.runtime
        original = self.backend.create_session
        def leaky_create(sid):
            original(sid)
            raise RuntimeError("failed after allocation")
        with patch.object(self.backend, "create_session", side_effect=leaky_create):
            with self.assertRaisesRegex(RuntimeError, "failed after allocation"):
                r.create_session("cleaned")
            self.assertNotIn("cleaned", self.backend._states)
            self.assertNotIn("cleaned", r._resident_session_ids)
            with patch.object(self.backend, "delete_session", side_effect=RuntimeError("cleanup failed")):
                with self.assertRaisesRegex(RuntimeError, "quarantined"):
                    r.create_session("residue")
        r.create_session("one")
        r.create_session("two")
        with self.assertRaisesRegex(ValueError, "capacity"):
            r.create_session("over")
        self.assertEqual(set(self.backend._states), {"residue", "one", "two"})

    def test_failure_after_prepared_publication_freezes_live_session(self):
        from gpu_runtime.runtime import _write_retirement_record
        r = self.runtime
        r.create_session("active")
        def write_then_fail(*args):
            _write_retirement_record(*args)
            raise OSError("directory fsync failed after publication")
        with patch("gpu_runtime.runtime._write_retirement_record", side_effect=write_then_fail):
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                r.retire_session("active", "final", expected_policy_version=0)
        self.assertIn("active", self.backend._states)
        self.assertEqual(self.receipt("active", "prepared.json")["status"], "prepared")
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            r.learn("active", expected_policy_version=0, event={"event_id": "late"})
        with self.assertRaisesRegex(ValueError, "already attempted"):
            r.retire_session("active", "other", expected_policy_version=0)

    def test_duplicate_creation_never_cleans_an_existing_session(self):
        from gpu_runtime import DuplicateSessionError
        self.runtime.create_session("existing")
        with patch.object(self.backend, "delete_session", side_effect=AssertionError("must not delete existing")):
            with self.assertRaises(DuplicateSessionError):
                self.runtime.create_session("existing")
        self.assertIn("existing", self.backend._states)

    def test_prepopulated_session_store_counts_toward_capacity(self):
        from gpu_runtime.session_store import SessionStore
        backend, sessions = ToyBackend(), SessionStore()
        sessions.create("existing", backend.create_session("existing"))
        with GpuRuntime(backend=backend, sessions=sessions, snapshot_root=self.root / "injected",
                        max_resident_sessions=1) as r:
            with self.assertRaisesRegex(ValueError, "capacity"):
                r.create_session("next")
            r.retire_session("existing", "final", expected_policy_version=0)
            r.create_session("next")

    def test_readonly_receipt_requires_bound_records_and_does_not_queue_work(self):
        r = self.runtime
        with self.assertRaises(SessionNotFoundError):
            r.retirement_receipt("missing")
        self.assertFalse((self.root / "_retirements/missing").exists())
        r.create_session("done")
        released = r.retire_session("done", "final", expected_policy_version=0)
        completed = r.actor.completed
        self.assertEqual(r.retirement_receipt("done"), released)
        self.assertEqual(r.actor.completed, completed)
        path = self.root / "_retirements/done/released.json"
        path.unlink()
        self.assertEqual(r.retirement_receipt("done")["status"], "prepared")
        path.write_text(json.dumps({**released, "policy_version": 1}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            r.retirement_receipt("done")
        path.write_text(json.dumps(released), encoding="utf-8")
        (path.parent / "prepared.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing its prepared"):
            r.retirement_receipt("done")

    def test_receipt_reader_rejects_corruption_and_traversal(self):
        r = self.runtime
        r.create_session("done")
        released = r.retire_session("done", "final", expected_policy_version=0)
        path = self.root / "_retirements/done/released.json"
        for field, value in (("session_id", "other"), ("schema_version", "other"),
                             ("snapshot_sha256", "x" * 64), ("snapshot", "../outside"),
                             ("policy_version", True), ("mutation_retry_allowed", True)):
            path.write_text(json.dumps({**released, field: value}), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(ValueError):
                r.retirement_receipt("done")
        path.write_bytes(b'{"status":"prepared","status":"released"}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            r.retirement_receipt("done")
        path.write_text(json.dumps(released), encoding="utf-8")
        manifest = self.root / "done/final/manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "manifest identity/hash"):
            r.retirement_receipt("done")

    def test_default_capacity_and_legacy_delete_remain_compatible(self):
        with GpuRuntime(backend=ToyBackend(), snapshot_root=self.root / "unbounded") as r:
            for index in range(8):
                r.create_session(f"s{index}")
            r.delete_session("s0")
            r.create_session("s0")
        for cap in (0, -1, True, 2.5):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                GpuRuntime(backend=ToyBackend(), snapshot_root=self.root / "bad", max_resident_sessions=cap)


if __name__ == "__main__":
    unittest.main()
