"""CPU tensor fixtures only: audit integrity and actor-state acceptance gates."""
import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from gpu_runtime.session_store import SessionState
from gpu_runtime.snapshot_store import SnapshotStore
from tools.audit_learner_actor_snapshots import audit, main


class ActorSnapshotAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = SnapshotStore(self.root/"snapshots")
        self.state = {"adapter": {"a": torch.tensor([1.0])}, "value_head": {"w": torch.tensor([2.0])},
            "optimizer": {"state": {}, "param_groups": [{"params": [0, 1], "lr": 0.001}]},
            "optimizer_steps": 0, "examples_seen": 0,
            "rng": {"seed": 123, "device_type": "cuda", "cpu": torch.tensor([1, 2], dtype=torch.uint8),
                    "device": torch.tensor([3, 4], dtype=torch.uint8)}}
        def logical(sid, release):
            return SessionState(session_id=sid, role="actor", theorem_id="c"*64,
                lineage={"model_release_sha256": release, "weights_sha256": "d"*64,
                    "source": {"source_kind": "learner_checkpoint", "learner_id": "central",
                               "learner_step": 1 if release == "a"*64 else 2, "checkpoint_sha256": "e"*64},
                    "reset": ["optimizer", "rng", "buffer", "policy_version", "event_receipts"]},
                optimizer_metadata={"kind": "AdamW", "steps": 0}).snapshot()
        self.old, self.new = logical("actor-one", "a"*64), logical("actor-two", "b"*64)
        self.before = self.snapshot("before", self.old)
        self.after = self.snapshot("after", self.old)
        self.new_path = self.snapshot("initial", self.new)
        self.report = {"schema_version": "reap.learner-loop.gpu-gate.v1", "ok": True, "gates": {"original": True},
            "release1": {"model_release_sha256": "a"*64}, "release2": {"model_release_sha256": "b"*64},
            "second": {"checkpoint_sha256": "e"*64}, "actors": [self.old, self.new]}
        self.final = {"checkpoint_sha256": "e"*64, "release_sha256": "b"*64, "goal_complete": False}
        self.report_path = self.root/"report.json"; self.report_path.write_text(json.dumps(self.report))
        self.final_path = self.root/"final-continuation.json"; self.final_path.write_text(json.dumps(self.final))

    def snapshot(self, name, logical, state=None):
        stream = io.BytesIO(); torch.save(self.state if state is None else state, stream)
        backend = {"schema_version": "reap.gpu.verified-replay-backend.v1", "encoding": "torch-save-base64",
            "session_id": logical["session_id"], "verified_config": {"objective": "verified_success_replay"},
            "payload": base64.b64encode(stream.getvalue()).decode()}
        return self.store.create(logical["session_id"], name, session_state=logical, backend_state=backend)

    def arguments(self, **changes):
        return {"actor_before": self.before, "actor_after": self.after, "new_actor": self.new_path,
                "report": self.report_path, "continuation": self.final_path, **changes}

    def test_full_comparison_and_empty_private_state_pass_without_GPU_calls(self):
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU check forbidden")):
            result = audit(**self.arguments())
        self.assertTrue(result["ok"])
        self.assertTrue(all(result["gates"].values()))
        self.assertTrue(result["CPU_only"])
        self.assertNotIn("torch-save-base64", json.dumps(result))
        self.assertEqual(result["sources"]["new_actor"]["decoded_tensor_count"], 4)

    def test_old_logical_mutation_is_not_hidden_by_parameter_fingerprints(self):
        changed = deepcopy(self.old); changed["reference_metadata"]["unrelated"] = 1
        path = self.snapshot("changed-logical", changed)
        result = audit(**self.arguments(actor_after=path))
        self.assertFalse(result["ok"])
        self.assertFalse(result["gates"]["old_actor_complete_logical_state_equal"])
        self.assertTrue(result["gates"]["old_actor_all_decoded_backend_state_equal"])

    def test_old_rng_counter_and_parameter_changes_all_fail(self):
        for name, mutate in (("rng", lambda s: s["rng"]["device"].fill_(0)),
                             ("counter", lambda s: s.update(examples_seen=1)),
                             ("tensor", lambda s: s["adapter"]["a"].add_(1))):
            state = deepcopy(self.state); mutate(state)
            path = self.snapshot("changed-"+name, self.old, state)
            result = audit(**self.arguments(actor_after=path))
            self.assertFalse(result["gates"]["old_actor_all_decoded_backend_state_equal"])
            self.assertFalse(result["ok"])

    def test_new_actor_nonempty_adam_or_nonzero_counters_fail(self):
        for name, mutate in (("adam", lambda s: s["optimizer"]["state"].update({0: {"step": torch.tensor(1)}})),
                             ("steps", lambda s: s.update(optimizer_steps=1)),
                             ("examples", lambda s: s.update(examples_seen=1)),
                             ("bool", lambda s: s.update(optimizer_steps=False))):
            state = deepcopy(self.state); mutate(state)
            path = self.snapshot("bad-"+name, self.new, state)
            self.assertFalse(audit(**self.arguments(new_actor=path))["ok"])

    def test_new_actor_logical_receipts_or_buffers_fail(self):
        for name, mutate in (("receipt", lambda s: s["event_receipts"].update(old={})),
                             ("buffer", lambda s: s["buffer_metadata"]["consumed_event_ids"].append("old"))):
            logical = deepcopy(self.new); mutate(logical)
            path = self.snapshot("bad-"+name, logical)
            self.assertFalse(audit(**self.arguments(new_actor=path))["ok"])

    def test_error_report_and_missing_or_mismatched_final_continuation_fail(self):
        report = {**self.report, "error": {"type": "OSError"}}
        self.report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "error"): audit(**self.arguments())
        self.report_path.write_text(json.dumps(self.report))
        self.final_path.write_text(json.dumps({**self.final, "release_sha256": "f"*64}))
        with self.assertRaisesRegex(ValueError, "continuation"): audit(**self.arguments())
        self.final_path.unlink()
        with self.assertRaises(ValueError): audit(**self.arguments())

    def test_corrupt_input_reports_failure_and_exclusive_output_is_not_overwritten(self):
        (self.after/"backend.json").write_bytes(b"{}")
        output = self.root/"audit.json"
        args = [item for key, value in {**self.arguments(), "output": output}.items()
                for item in ("--"+key.replace("_", "-"), str(value))]
        self.assertEqual(main(args), 1)
        raw = output.read_bytes()
        self.assertFalse(json.loads(raw)["ok"])
        self.assertNotIn("gates", json.loads(raw))
        with self.assertRaises(FileExistsError): main(args)
        self.assertEqual(output.read_bytes(), raw)

    def test_output_inside_snapshot_is_refused_before_writing(self):
        output = self.before/"audit.json"
        args = [item for key, value in {**self.arguments(), "output": output}.items()
                for item in ("--"+key.replace("_", "-"), str(value))]
        with self.assertRaisesRegex(ValueError, "inside"): main(args)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
