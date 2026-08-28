"""Filesystem/mock-executor checks, not Lean, browser, or GPU execution."""
import hashlib
import json
import sys
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cpu_runtime import online_batch as batch


class OnlineBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "Proof.lean").write_text("example : True := by trivial\n", encoding="utf-8")
        self.manifest = self.root / "batch.jsonl"
        self.output = self.root / "out"
        self.write_manifest("batch-a", "batch-b")
        self.calls = []

    def write_manifest(self, *sessions):
        self.manifest.write_bytes(b"".join(batch.encoded({"session_id": sid, "theorem_file": "Proof.lean"}) for sid in sessions))

    def invoke(self, **kwargs):
        return batch.run_batch(manifest=self.manifest, project_dir=self.project, output_root=self.output,
                               gpu_base_url="http://127.0.0.1:18760", gamma=0.99,
                               executor=kwargs.pop("executor", self.executor), **kwargs)

    def executor(self, **kwargs):
        self.calls.append(kwargs)
        return self.complete(kwargs)

    def complete(self, kwargs, *, updated=True, exhausted=False):
        sid = kwargs["session_id"]
        directory = kwargs["output_root"] / sid
        directory.mkdir()
        checkpoints = directory / "checkpoints"
        checkpoints.mkdir()

        def save(name, value):
            (directory / name).write_bytes(batch.encoded(value))

        source = kwargs["project_dir"] / kwargs["theorem_file"]
        save("session.json", {"session_id": sid, "tree_id": sid + ".tree0", "profile": batch.PROFILE,
            "theorem_file": str(source), "theorem_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "gamma": kwargs["gamma"], "max_updates": kwargs["max_updates"], "total_deadline_seconds": None,
            "policy_base_url": kwargs["gpu_base_url"] + f"/sessions/{sid}/policy/v1",
            "value_base_url": kwargs["gpu_base_url"] + f"/sessions/{sid}/value/v1"})
        save("create-receipt.json", {"session_id": sid, "policy_version": 0})
        for label in ("before", "after"):
            save(f"snapshot-{label}.json", {"session_id": sid, "snapshot": f"{label}-online-ttt"})
        save("result.json", {"schema_version": "reap.training.result.v1", "session_id": sid,
            "solved": not exhausted, "status": "exhausted" if exhausted else "solved", "error": None,
            "proof_script": None if exhausted else "trivial"})
        if exhausted:
            save("progress.jsonl", {"schema_version": "reap.training.progress.v1", "session_id": sid,
                "done": True, "solved": False, "status": "exhausted", "step": 32, "max_steps": 32})
        receipts = []
        if updated:
            event = {"event_id": sid + ".s0.n0", "session_id": sid, "tree_id": sid + ".tree0",
                     "policy_version": 0, "gamma": kwargs["gamma"]}
            receipt = {"event_id": event["event_id"], "policy_version": 1, "applied": True, "idempotent": False}
            save("checkpoints/learn-000000.request.json", event)
            save("checkpoints/learn-000000.receipt.json", receipt)
            receipts.append(receipt)
        report = {"session_id": sid, "tree_id": sid + ".tree0", "profile": batch.PROFILE,
            "returncode": 1 if exhausted else 0, "root_verified": not exhausted,
            "error": None, "total_deadline_seconds": None,
            "status": "lean_failed" if exhausted else ("passed_execution" if updated else "solved_without_online_update"),
            "policy_version": int(updated), "optimizer_updates": int(updated), "learn_receipts": receipts,
            "post_update_generations": [{"tree_id": sid + ".tree0", "policy_version": 1}] if updated else [],
            "online_update_consumed_by_later_generation": updated}
        save("online-result.json", report)
        return report

    def records(self):
        return [json.loads(line) for line in (self.output / "solutions.jsonl").read_bytes().splitlines()]

    def inherited_executor(self, **kwargs):
        report = self.executor(**kwargs)
        experience_id = kwargs.get("experience_id")
        if experience_id is not None:
            directory = kwargs["output_root"] / kwargs["session_id"]
            session = batch.read_json(directory / "session.json")
            session.update(experience_id=experience_id, experience_candidate=False)
            session.update({field: kwargs[field] for field in ("experience_weights_sha256", "experience_snapshot_sha256")})
            (directory / "session.json").write_bytes(batch.encoded(session))
            created = batch.read_json(directory / "create-receipt.json")
            created.update(schema_version="reap.gpu.session.v1", completed=False,
                           optimizer_metadata={"kind": "AdamW", "steps": 0}, event_receipts={},
                           buffer_metadata={"events": {}, "pending_event_ids": [], "consumed_event_ids": []})
            created.update(theorem_id=session["theorem_sha256"], lineage={"experience_id": experience_id,
                "weights_sha256": kwargs["experience_weights_sha256"],
                "reset": ["optimizer", "rng", "buffer", "policy_version", "event_receipts"],
                "source": {"session_id": "source-session", "theorem_id": "source-theorem",
                    "policy_version": 2, "snapshot": "candidate",
                    "snapshot_sha256": kwargs["experience_snapshot_sha256"], "parent_experience_id": None}})
            (directory / "create-receipt.json").write_bytes(batch.encoded(created))
            report["experience_id"] = experience_id
            (directory / "online-result.json").write_bytes(batch.encoded(report))
        return report

    def write_inherited_manifest(self):
        self.manifest.write_bytes(batch.encoded({"session_id": "batch-a", "theorem_file": "Proof.lean",
                                                "experience_id": "accepted-source-01",
                                                "experience_weights_sha256": "a" * 64,
                                                "experience_snapshot_sha256": "b" * 64}) +
                                  batch.encoded({"session_id": "batch-b", "theorem_file": "Proof.lean"}))

    def test_explicit_per_theorem_experience_and_fresh_mix_resume_without_rerun(self):
        self.write_inherited_manifest()
        result = self.invoke(executor=self.inherited_executor, concurrency=2)
        self.assertEqual(result["status"], "completed")
        calls = {c["session_id"]: c for c in self.calls}
        self.assertEqual(calls["batch-a"]["experience_id"], "accepted-source-01")
        self.assertNotIn("experience_id", calls["batch-b"])
        records = {r["session_id"]: r for r in self.records()}
        self.assertEqual(records["batch-a"]["experience_id"], "accepted-source-01")
        self.assertEqual(records["batch-a"]["policy_version"], 1)
        again = self.invoke(executor=lambda **kw: self.fail("completed work rerun"))
        self.assertEqual(again["skipped"], ["batch-a", "batch-b"])

    def test_changed_experience_blocks_resume_before_any_dispatch(self):
        self.write_inherited_manifest()
        self.invoke(executor=self.inherited_executor)
        self.manifest.write_bytes(self.manifest.read_bytes().replace(b"accepted-source-01", b"accepted-source-02"))
        with self.assertRaisesRegex(batch.BatchBlocked, "changed"):
            self.invoke(executor=lambda **kw: self.fail("changed source dispatched"))

    def test_inherited_artifact_mismatches_block_refill(self):
        for artifact in ("session.json", "online-result.json", "create-receipt.json", "theorem_id"):
            with self.subTest(artifact=artifact):
                self.output = self.root / artifact.replace(".", "-")
                self.calls.clear()
                self.write_inherited_manifest()
                def mismatched(**kw):
                    report = self.inherited_executor(**kw)
                    name = "create-receipt.json" if artifact == "theorem_id" else artifact
                    path = kw["output_root"] / kw["session_id"] / name
                    value = batch.read_json(path)
                    if artifact == "create-receipt.json":
                        value["lineage"]["experience_id"] = "wrong-source"
                    elif artifact == "theorem_id":
                        value["theorem_id"] = "source-theorem-not-destination"
                    else:
                        value["experience_id"] = "wrong-source"
                    path.write_bytes(batch.encoded(value))
                    return value if name == "online-result.json" else report
                result = self.invoke(executor=mismatched, concurrency=1)
                self.assertEqual(result["status"], "manual_intervention_required")
                self.assertEqual(result["not_started"], ["batch-b"])

    def test_invalid_release_selectors_rejected_before_dispatch(self):
        for value in (None, "", "../source", "x" * 65, True, {"latest": True}):
            with self.subTest(value=value):
                self.manifest.write_bytes(batch.encoded({"session_id": "batch-a", "theorem_file": "Proof.lean",
                                                        "experience_id": value}))
                with self.assertRaises(batch.BatchBlocked):
                    self.invoke(executor=lambda **kw: self.fail("invalid selector dispatched"))

    def test_missing_source_error_is_not_retried_or_followed_by_more_tasks(self):
        self.write_inherited_manifest()
        def missing(**kw):
            self.calls.append(kw)
            raise FileNotFoundError("release missing at runtime")
        result = self.invoke(executor=missing, concurrency=1)
        self.assertEqual(result["not_started"], ["batch-b"])
        self.assertEqual(len(self.calls), 1)
        with self.assertRaises(batch.BatchBlocked):
            self.invoke(executor=lambda **kw: self.fail("uncertain create retried"))

    def test_pinned_manifest_requires_complete_valid_content_reference(self):
        valid = {"session_id": "batch-a", "theorem_file": "Proof.lean", "experience_id": "release",
                 "experience_weights_sha256": "a" * 64, "experience_snapshot_sha256": "b" * 64}
        for field in ("experience_id", "experience_weights_sha256", "experience_snapshot_sha256"):
            for action in ("missing", "invalid"):
                with self.subTest(field=field, action=action):
                    row = dict(valid)
                    if action == "missing":
                        del row[field]
                    else:
                        row[field] = "../invalid"
                    self.manifest.write_bytes(batch.encoded(row))
                    with self.assertRaises(batch.BatchBlocked):
                        self.invoke(executor=lambda **kw: self.fail("invalid reference dispatched"))
                    self.assertFalse(self.output.exists())

    def test_same_release_name_changed_content_blocks_resume_before_dispatch(self):
        self.write_inherited_manifest()
        self.invoke(executor=self.inherited_executor)
        self.manifest.write_bytes(self.manifest.read_bytes().replace(b"a" * 64, b"c" * 64))
        with self.assertRaisesRegex(batch.BatchBlocked, "changed"):
            self.invoke(executor=lambda **kw: self.fail("changed release content dispatched"))

    def test_inherited_completion_uses_same_full_lineage_and_reset_gate(self):
        for index, path in enumerate(("lineage.source", "lineage.reset", "optimizer_metadata", "buffer_metadata",
                                      "event_receipts", "lineage.weights_sha256")):
            with self.subTest(path=path):
                self.output = self.root / ("lineage-" + str(index))
                self.write_inherited_manifest()
                def invalid(**kwargs):
                    report = self.inherited_executor(**kwargs)
                    receipt_path = kwargs["output_root"] / kwargs["session_id"] / "create-receipt.json"
                    created = batch.read_json(receipt_path)
                    location = created
                    fields = path.split(".")
                    for field in fields[:-1]:
                        location = location[field]
                    location[fields[-1]] = {"unexpected": True}
                    receipt_path.write_bytes(batch.encoded(created))
                    return report
                result = self.invoke(executor=invalid, concurrency=1)
                self.assertEqual(result["status"], "manual_intervention_required")
                self.assertEqual(result["not_started"], ["batch-b"])
                with self.assertRaises(batch.BatchBlocked):
                    self.invoke(executor=lambda **kw: self.fail("bad initialization retried"))

    def retirement_client(self, action=None):
        calls = []

        def retire(sid, name, *, expected_policy_version, reuse_snapshot=False):
            calls.append(sid)
            intent, _, receipt = batch.retirement_paths(self.output, sid)
            self.assertTrue(intent.exists())
            self.assertFalse(receipt.exists())
            value = {"schema_version": "reap.gpu.retirement.v1", "session_id": sid,
                "policy_version": expected_policy_version, "snapshot": name, "snapshot_sha256": "a" * 64,
                "status": "released", "mutation_retry_allowed": False, "tombstone_scope": "current_runtime"}
            if reuse_snapshot:
                self.assertEqual(name, "after-online-ttt")
                self.assertIs(batch.read_json(intent)["reuse_snapshot"], True)
                value.update(schema_version="reap.gpu.retirement.v2", snapshot_mode="reuse_verified", snapshot_revision=1)
            return action(sid, value) if action else value

        return SimpleNamespace(retire_session=retire), calls

    def test_retire_completed_refills_only_after_released_receipt_and_resumes_without_calls(self):
        self.write_manifest(*(f"batch-{i}" for i in range(12)))
        resident, peak = set(), 0
        guard = threading.Lock()
        rendezvous = threading.Barrier(2, timeout=5)

        def execute(**kwargs):
            nonlocal peak
            with guard:
                resident.add(kwargs["session_id"])
                peak = max(peak, len(resident))
                self.assertLessEqual(len(resident), 2)
            if kwargs["session_id"] in ("batch-0", "batch-1"):
                rendezvous.wait()
            return self.complete(kwargs, exhausted=int(kwargs["session_id"].split("-")[1]) % 2 == 1)

        def release(sid, receipt):
            with guard:
                resident.remove(sid)
            return receipt

        client, calls = self.retirement_client(release)
        result = self.invoke(executor=execute, retire_completed=True, retirement_client=client, concurrency=2)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(calls), 12)
        self.assertEqual(len(result["retired"]), 12)
        self.assertEqual(peak, 2)
        self.assertFalse(resident)
        again = self.invoke(executor=lambda **kwargs: self.fail("retired session rerun"),
                            retire_completed=True, retirement_client=client)
        self.assertEqual(len(again["retirement_skipped"]), 12)
        self.assertEqual(len(calls), 12)

    def test_retirement_batch_local_http_runtime_round_trip_holds_capacity(self):
        self._retirement_batch_local_http_runtime_round_trip_holds_capacity(reuse=False)

    def test_reuse_batch_local_http_runtime_round_trip_holds_capacity_without_new_snapshots(self):
        self._retirement_batch_local_http_runtime_round_trip_holds_capacity(reuse=True)

    def _retirement_batch_local_http_runtime_round_trip_holds_capacity(self, *, reuse):
        """Real HTTP/runtime/toy retirement; theorem results remain local fixtures."""
        from http.server import ThreadingHTTPServer
        from cpu_runtime.http_clients import GpuHttpClient
        from gpu_runtime import GpuRuntime, ToyBackend
        from gpu_runtime.server import RuntimeHandler

        self.write_manifest(*(f"http-{i}" for i in range(6)))
        backend = ToyBackend()
        runtime = GpuRuntime(backend=backend, snapshot_root=self.root / "gpu", max_resident_sessions=2)
        handler = type("RetirementHandler", (RuntimeHandler,), {"runtime": runtime, "backend_name": "toy"})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        client = GpuHttpClient(url)

        def execute(**kwargs):
            sid = kwargs["session_id"]
            created = client.create_session(sid)
            before = client.snapshot(sid, "before-online-ttt")
            report = self.complete(kwargs, updated=False, exhausted=sid in ("http-1", "http-3"))
            directory = kwargs["output_root"] / sid
            for name, value in (("create-receipt.json", created), ("snapshot-before.json", before),
                                ("snapshot-after.json", client.snapshot(sid, "after-online-ttt"))):
                (directory / name).write_bytes(batch.encoded(value))
            return report

        try:
            result = batch.run_batch(manifest=self.manifest, project_dir=self.project, output_root=self.output,
                gpu_base_url=url, gamma=0.99, max_updates=0, concurrency=2, executor=execute,
                retire_completed=True, retirement_client=client, reuse_final_snapshot=reuse)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["retired"]), 6)
            self.assertEqual(backend._states, {})
            for sid in result["retired"]:
                receipt = client.retirement_receipt(sid)
                self.assertEqual(receipt["status"], "released")
                self.assertEqual(receipt["schema_version"], "reap.gpu.retirement.v2" if reuse else "reap.gpu.retirement.v1")
                snapshots = {p.name for p in (self.root / "gpu" / sid).iterdir()}
                expected = {"before-online-ttt", "after-online-ttt"}
                self.assertEqual(snapshots, expected if reuse else expected | {batch.RETIREMENT_SNAPSHOT})
            again = batch.run_batch(manifest=self.manifest, project_dir=self.project, output_root=self.output,
                gpu_base_url=url, gamma=0.99, max_updates=0, concurrency=2,
                executor=lambda **kwargs: self.fail("completed session re-dispatched"),
                retire_completed=True, retirement_client=client, reuse_final_snapshot=reuse)
            self.assertEqual(len(again["retirement_skipped"]), 6)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)
            runtime.close()

    def test_reuse_requires_retirement_opt_in_and_is_bound_on_resume(self):
        for invalid in (True, "true", 1):
            with self.subTest(invalid=invalid), self.assertRaises(batch.BatchBlocked):
                self.invoke(reuse_final_snapshot=invalid)
        self.assertFalse(self.output.exists())
        client, calls = self.retirement_client()
        self.invoke(retire_completed=True, reuse_final_snapshot=True, retirement_client=client)
        with self.assertRaisesRegex(batch.BatchBlocked, "config/source"):
            self.invoke(retire_completed=True, reuse_final_snapshot=False, retirement_client=client)
        self.assertEqual(len(calls), 2)
        intent, _, _ = batch.retirement_paths(self.output, "batch-a")
        self.assertEqual(batch.read_json(intent)["schema_version"], "reap.online-batch.retirement-intent.v2")
        self.assertEqual(batch.read_json(intent)["snapshot"], "after-online-ttt")

    def test_reuse_unknown_ack_stops_refill_and_never_retries(self):
        def lost(sid, receipt):
            raise TimeoutError("reuse may have retired resident state")
        client, calls = self.retirement_client(lost)
        result = self.invoke(retire_completed=True, reuse_final_snapshot=True, retirement_client=client, concurrency=1)
        self.assertEqual(result["not_started"], ["batch-b"])
        self.assertEqual(result["status"], "manual_intervention_required")
        with self.assertRaisesRegex(batch.BatchBlocked, "outcome is unknown"):
            self.invoke(retire_completed=True, reuse_final_snapshot=True, retirement_client=client)
        self.assertEqual(calls, ["batch-a"])

    def test_reuse_rejects_downgrade_wrong_snapshot_and_invalid_revision_without_retry(self):
        for index, (field, value) in enumerate((("schema_version", "reap.gpu.retirement.v1"),
                ("snapshot", batch.RETIREMENT_SNAPSHOT), ("snapshot_mode", "fresh"),
                ("snapshot_revision", 0), ("snapshot_revision", True))):
            with self.subTest(field=field, value=value):
                self.output = self.root / ("bad-reuse-" + str(index))
                client, calls = self.retirement_client(lambda sid, receipt: {**receipt, field: value})
                result = self.invoke(retire_completed=True, reuse_final_snapshot=True,
                                     retirement_client=client, concurrency=1)
                self.assertEqual(result["not_started"], ["batch-b"])
                intent, response, receipt = batch.retirement_paths(self.output, "batch-a")
                self.assertTrue(intent.exists() and response.exists())
                self.assertFalse(receipt.exists())
                with self.assertRaisesRegex(batch.BatchBlocked, "outcome is unknown"):
                    self.invoke(retire_completed=True, reuse_final_snapshot=True, retirement_client=client)
                self.assertEqual(calls, ["batch-a"])

    def test_reuse_cli_explicitly_forwards_both_flags(self):
        args = ["online_batch", "--manifest", str(self.manifest), "--project-dir", str(self.project),
                "--output-dir", str(self.output), "--gpu-base-url", "http://127.0.0.1:1234",
                "--gamma", "0.99", "--retire-completed", "--reuse-final-snapshot"]
        with patch.object(sys, "argv", args), patch.object(batch, "run_batch", return_value={"status": "completed"}) as run, \
                patch("builtins.print"):
            self.assertEqual(batch.main(), 0)
        self.assertIs(run.call_args.kwargs["retire_completed"], True)
        self.assertIs(run.call_args.kwargs["reuse_final_snapshot"], True)

    def test_retirement_unknown_ack_stops_refill_and_resume_without_resubmission(self):
        def lost(sid, receipt):
            raise TimeoutError("release may have happened")

        client, calls = self.retirement_client(lost)
        first = self.invoke(retire_completed=True, retirement_client=client, concurrency=1)
        self.assertEqual(first["status"], "manual_intervention_required")
        self.assertEqual(first["not_started"], ["batch-b"])
        self.assertEqual(first["completed"], ["batch-a"])
        self.assertEqual(len(self.records()), 1)  # proof remains valid despite unknown release
        with self.assertRaisesRegex(batch.BatchBlocked, "outcome is unknown"):
            self.invoke(retire_completed=True, retirement_client=client)
        self.assertEqual(calls, ["batch-a"])

    def test_invalid_retirement_receipts_are_preserved_but_never_committed_or_retried(self):
        for key, value in (("status", "prepared"), ("session_id", "wrong"), ("policy_version", 99),
                           ("snapshot_sha256", "bad"), ("mutation_retry_allowed", True)):
            with self.subTest(key=key):
                self.output = self.root / ("invalid-" + key)
                client, calls = self.retirement_client(lambda sid, receipt: {**receipt, key: value})
                result = self.invoke(retire_completed=True, retirement_client=client, concurrency=1)
                self.assertEqual(result["not_started"], ["batch-b"])
                intent, response, receipt = batch.retirement_paths(self.output, "batch-a")
                self.assertTrue(intent.exists() and response.exists())
                self.assertFalse(receipt.exists())
                with self.assertRaisesRegex(batch.BatchBlocked, "outcome is unknown"):
                    self.invoke(retire_completed=True, retirement_client=client)
                self.assertEqual(calls, ["batch-a"])

    def test_retirement_intent_write_failure_dispatches_nothing_and_can_resume_first_submission(self):
        self.write_manifest("batch-a")
        client, calls = self.retirement_client()
        real_publish = batch.publish

        def fail_intent(path, content, **kwargs):
            if path.parent.name == ".batch-retirements" and path.name.endswith(".intent.json"):
                raise OSError("disk full before dispatch")
            return real_publish(path, content, **kwargs)

        with patch.object(batch, "publish", side_effect=fail_intent):
            self.assertEqual(self.invoke(retire_completed=True, retirement_client=client)["status"], "manual_intervention_required")
        self.assertEqual(calls, [])
        again = self.invoke(retire_completed=True, retirement_client=client,
                            executor=lambda **kwargs: self.fail("completed proof rerun"))
        self.assertEqual(again["retired"], ["batch-a"])
        self.assertEqual(calls, ["batch-a"])

    def test_completed_artifact_recovery_with_first_retirement_is_not_no_GPU_calls(self):
        self.write_manifest("batch-a")
        client, calls = self.retirement_client()

        def crash(**kwargs):
            self.complete(kwargs)
            raise RuntimeError("crash before terminal commit or retirement")

        first = self.invoke(executor=crash, retire_completed=True, retirement_client=client)
        self.assertEqual(first["status"], "manual_intervention_required")
        self.assertEqual(calls, [])
        again = self.invoke(retire_completed=True, retirement_client=client,
                            executor=lambda **kwargs: self.fail("completed search rerun"))
        self.assertEqual(again["recovered_without_search_rerun"], ["batch-a"])
        self.assertEqual(again["recovered_without_GPU_calls"], [])
        self.assertEqual(again["retired"], ["batch-a"])
        self.assertEqual(calls, ["batch-a"])

    def test_retirement_receipt_write_failure_blocks_resume_even_with_response(self):
        self.write_manifest("batch-a")
        client, calls = self.retirement_client()
        real_publish = batch.publish

        def fail_receipt(path, content, **kwargs):
            if path.parent.name == ".batch-retirements" and path.name.endswith(".receipt.json"):
                raise OSError("disk full after release")
            return real_publish(path, content, **kwargs)

        with patch.object(batch, "publish", side_effect=fail_receipt):
            self.assertEqual(self.invoke(retire_completed=True, retirement_client=client)["status"], "manual_intervention_required")
        with self.assertRaisesRegex(batch.BatchBlocked, "outcome is unknown"):
            self.invoke(retire_completed=True, retirement_client=client)
        self.assertEqual(calls, ["batch-a"])

    def test_retirement_opt_in_and_receipt_binding_cannot_change_on_resume(self):
        self.write_manifest("batch-a")
        client, calls = self.retirement_client()
        self.invoke(retire_completed=True, retirement_client=client)
        with self.assertRaisesRegex(batch.BatchBlocked, "config/source"):
            self.invoke(retire_completed=False)
        _, _, receipt_path = batch.retirement_paths(self.output, "batch-a")
        receipt = json.loads(receipt_path.read_bytes())
        receipt["intent_sha256"] = "0" * 64
        receipt_path.write_bytes(batch.encoded(receipt))
        with self.assertRaisesRegex(batch.BatchBlocked, "binding mismatch"):
            self.invoke(retire_completed=True, retirement_client=client)
        self.assertEqual(calls, ["batch-a"])

    def test_bounded_concurrent_success_and_verified_resume_skip(self):
        self.write_manifest("batch-a", "batch-b", "batch-c")
        barrier = threading.Barrier(2, timeout=5)
        guard = threading.Lock()
        active = peak = count = 0

        def execute(**kwargs):
            nonlocal active, peak, count
            with guard:
                active += 1
                count += 1
                position = count
                peak = max(peak, active)
            try:
                if position <= 2:
                    barrier.wait()
                return self.executor(**kwargs)
            finally:
                with guard:
                    active -= 1

        result = self.invoke(executor=execute)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(peak, 2)
        self.assertEqual(result["scheduling"]["max_active"], 2)
        self.assertEqual(result["scheduling"]["started"], 3)
        self.assertEqual(result["scheduling"]["finished"], 3)
        self.assertEqual(result["scheduling"]["active"], 0)
        self.assertGreaterEqual(result["scheduling"]["session_execution_seconds_total"], 0)
        self.assertEqual({r["session_id"] for r in self.records()}, {"batch-a", "batch-b", "batch-c"})
        self.assertTrue(all(r["online_execution_gate_passed"] for r in self.records()))
        self.assertIsNone(result["total_deadline_seconds"])
        self.assertTrue(all("deadline_seconds" not in kwargs for kwargs in self.calls))
        original = (self.output / "solutions.jsonl").read_bytes()
        again = self.invoke(executor=lambda **kwargs: self.fail("GPU was rerun"))
        self.assertEqual(again["skipped"], ["batch-a", "batch-b", "batch-c"])
        self.assertEqual(original, (self.output / "solutions.jsonl").read_bytes())
        self.assertEqual(again["scheduling"]["started"], 0)

    def test_exhausted_search_is_terminal_not_a_solution_and_next_theorem_runs(self):
        self.write_manifest("batch-a", "batch-b", "batch-c")

        def execute(**kwargs):
            self.calls.append(kwargs)
            return self.complete(kwargs, updated=kwargs["session_id"] != "batch-c",
                                 exhausted=kwargs["session_id"] != "batch-b")

        result = self.invoke(executor=execute, concurrency=1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["completed"], ["batch-b"])
        self.assertEqual(result["exhausted"], ["batch-a", "batch-c"])
        self.assertEqual(len(self.calls), 3)
        self.assertEqual([r["session_id"] for r in self.records()], ["batch-b"])
        path = self.output / "terminal-unsolved.jsonl"
        raw = path.read_bytes()
        records = [json.loads(line) for line in raw.splitlines()]
        self.assertTrue(all(r["status"] == "search_exhausted" and r["proof_script"] is None for r in records))
        self.assertTrue(all(not r["root_verified"] and not r["online_execution_gate_passed"] for r in records))
        again = self.invoke(executor=lambda **kwargs: self.fail("terminal search was rerun"))
        self.assertEqual(again["skipped_exhausted"], ["batch-a", "batch-c"])
        self.assertEqual(path.read_bytes(), raw)

    def test_no_training_control_accepts_zero_update_budget(self):
        self.write_manifest("batch-a", "batch-b")
        result = self.invoke(max_updates=0, executor=lambda **kwargs:
                            self.complete(kwargs, updated=False, exhausted=kwargs["session_id"] == "batch-b"))
        self.assertEqual(result["completed"], ["batch-a"])
        self.assertEqual(result["exhausted"], ["batch-b"])
        self.assertFalse(self.records()[0]["online_execution_gate_passed"])
        with self.assertRaises(batch.BatchBlocked):
            self.invoke(max_updates=-1)

    def test_actual_online_coordinator_classifies_exhaustion_protocol_without_GPU(self):
        """Real subprocess/coordinator; the subprocess is a protocol fixture, not Lean."""
        self.write_manifest("batch-a")

        class Client:
            def create_session(self, session_id):
                return {"session_id": session_id, "policy_version": 0, "value_metadata": {
                    "objective": "search_visit_backup", "gamma": 0.99,
                    "value_semantics": "reap.search_backup_discounted_return.v1"}}

            def snapshot(self, session_id, name):
                return {"session_id": session_id, "snapshot": name}

        program = """
import json, os, pathlib, sys
p = pathlib.Path(os.environ['REAP_SESSION_DIR'])
sid = os.environ['REAP_SESSION_ID']
(p / 'result.json').write_text(json.dumps({'schema_version': 'reap.training.result.v1',
    'session_id': sid, 'solved': False, 'status': 'exhausted', 'proof_script': None, 'error': None}))
(p / 'progress.jsonl').write_text(json.dumps({'schema_version': 'reap.training.progress.v1',
    'session_id': sid, 'done': True, 'solved': False, 'status': 'exhausted'}) + '\\n')
sys.exit(1)
"""

        def execute(**kwargs):
            return batch.run_online(**kwargs, client=Client(), command=[sys.executable, "-c", program])

        result = self.invoke(max_updates=0, executor=execute)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["exhausted"], ["batch-a"])
        actual = json.loads((self.output / "batch-a/online-result.json").read_bytes())
        self.assertEqual((actual["returncode"], actual["status"]), (1, "lean_failed"))
        self.assertIsNone(actual["error"])
        again = self.invoke(max_updates=0, executor=lambda **kwargs: self.fail("finished subprocess reran"))
        self.assertEqual(again["skipped_exhausted"], ["batch-a"])

    def test_exhausted_result_commit_failure_recovers_without_rerun(self):
        self.write_manifest("batch-a")
        real_publish = batch.publish

        def fail_commit(path, content, **kwargs):
            if path.name == "terminal-unsolved.jsonl":
                raise OSError("simulated unsolved publication failure")
            return real_publish(path, content, **kwargs)

        with patch.object(batch, "publish", side_effect=fail_commit):
            first = self.invoke(executor=lambda **kwargs: self.complete(kwargs, exhausted=True))
        self.assertEqual(first["status"], "manual_intervention_required")
        again = self.invoke(executor=lambda **kwargs: self.fail("exhausted session was rerun"))
        self.assertEqual(again["recovered_without_GPU_calls"], ["batch-a"])
        self.assertTrue((self.output / "terminal-unsolved.jsonl").is_file())
        self.assertFalse((self.output / "solutions.jsonl").exists())

    def test_exhausted_with_unknown_mutation_or_missing_evidence_stops_batch(self):
        for damage in ("unknown-error", "missing-snapshot", "extra-request", "partial-progress", "failed-progress"):
            with self.subTest(damage=damage):
                self.output = self.root / damage

                def execute(**kwargs):
                    report = self.complete(kwargs, exhausted=True)
                    directory = kwargs["output_root"] / kwargs["session_id"]
                    if damage == "unknown-error":
                        report["error"] = {"type": "TimeoutError", "mutation_retry_allowed": False}
                        (directory / "online-result.json").write_bytes(batch.encoded(report))
                    elif damage == "missing-snapshot":
                        (directory / "snapshot-after.json").unlink()
                    elif damage == "extra-request":
                        (directory / "checkpoints/learn-000001.request.json").write_bytes(batch.encoded({"event_id": "unknown"}))
                    elif damage == "partial-progress":
                        path = directory / "progress.jsonl"
                        path.write_bytes(path.read_bytes().rstrip(b"\n"))
                    else:
                        path = directory / "progress.jsonl"
                        progress = json.loads(path.read_bytes())
                        progress["done"] = False
                        path.write_bytes(batch.encoded(progress))
                    return report

                first = self.invoke(executor=execute, concurrency=1)
                self.assertEqual(first["status"], "manual_intervention_required")
                self.assertEqual(first["not_started"], ["batch-b"])
                self.assertFalse((self.output / "terminal-unsolved.jsonl").exists())
                with self.assertRaises(batch.BatchBlocked):
                    self.invoke(executor=lambda **kwargs: self.fail("unsafe mutation retry"))

    def test_tampered_or_partial_terminal_unsolved_record_rejects_resume(self):
        self.write_manifest("batch-a")
        self.invoke(executor=lambda **kwargs: self.complete(kwargs, exhausted=True))
        path = self.output / "terminal-unsolved.jsonl"
        original = path.read_bytes()
        for corrupted in (original + b'{"session_id":', original + original):
            path.write_bytes(corrupted)
            with self.assertRaises(batch.BatchBlocked):
                self.invoke(executor=lambda **kwargs: self.fail("invalid terminal evidence retried"))
            self.assertEqual(path.read_bytes(), corrupted)

    def test_crash_after_completed_result_recovers_without_GPU_or_overwrite(self):
        self.write_manifest("batch-a")

        def crash(**kwargs):
            self.executor(**kwargs)
            raise RuntimeError("crash before batch commit")

        first = self.invoke(executor=crash)
        self.assertEqual(first["status"], "manual_intervention_required")
        self.assertFalse((self.output / "solutions.jsonl").exists())
        saved = (self.output / "batch-a/online-result.json").read_bytes()
        again = self.invoke(executor=lambda **kwargs: self.fail("GPU was rerun"))
        self.assertEqual(again["recovered_without_GPU_calls"], ["batch-a"])
        self.assertEqual(saved, (self.output / "batch-a/online-result.json").read_bytes())
        self.assertEqual(len(self.records()), 1)

    def test_unknown_mutation_blocks_resume_and_does_not_launch_queued_work(self):
        self.write_manifest("batch-a", "batch-b")

        def unknown(**kwargs):
            self.calls.append(kwargs)
            (kwargs["output_root"] / kwargs["session_id"]).mkdir()
            raise TimeoutError("unknown remote mutation")

        first = self.invoke(executor=unknown, concurrency=1)
        self.assertEqual(first["not_started"], ["batch-b"])
        self.assertEqual(len(self.calls), 1)
        with self.assertRaisesRegex(batch.BatchBlocked, "manual intervention"):
            self.invoke(executor=lambda **kwargs: self.fail("unsafe retry"))

    def test_source_profile_config_changes_fail_before_any_GPU_call(self):
        self.invoke()
        with self.assertRaisesRegex(batch.BatchBlocked, "config/source"):
            self.invoke(max_updates=6, executor=lambda **kwargs: self.fail("changed config executed"))
        (self.project / "Proof.lean").write_text("example : True := by decide\n", encoding="utf-8")
        with self.assertRaisesRegex(batch.BatchBlocked, "config/source"):
            self.invoke(executor=lambda **kwargs: self.fail("changed source executed"))

    def test_partial_or_duplicate_solution_line_is_not_success_or_truncated(self):
        self.invoke()
        path = self.output / "solutions.jsonl"
        original = path.read_bytes()
        for corrupted in (original + b'{"session_id":', original + original.splitlines(keepends=True)[0]):
            with self.subTest(corrupted=corrupted[-25:]):
                path.write_bytes(corrupted)
                with self.assertRaises(batch.BatchBlocked):
                    self.invoke(executor=lambda **kwargs: self.fail("invalid solutions retried"))
                self.assertEqual(path.read_bytes(), corrupted)

    def test_mismatched_saved_receipt_and_unknown_extra_request_are_rejected(self):
        self.write_manifest("batch-a")
        self.invoke()
        request = self.output / "batch-a/checkpoints/learn-000001.request.json"
        request.write_bytes(batch.encoded({"event_id": "unknown"}))
        with self.assertRaisesRegex(batch.BatchBlocked, "unresolved or extra"):
            self.invoke(executor=lambda **kwargs: self.fail("unknown mutation reran"))
        request.unlink()
        path = self.output / "batch-a/checkpoints/learn-000000.receipt.json"
        receipt = json.loads(path.read_bytes())
        receipt["policy_version"] = 2
        path.write_bytes(batch.encoded(receipt))
        with self.assertRaisesRegex(batch.BatchBlocked, "receipt/event mismatch"):
            self.invoke(executor=lambda **kwargs: self.fail("wrong receipt reran"))

    def test_solved_without_update_is_recorded_but_not_TTT_gate(self):
        self.write_manifest("batch-a")
        result = self.invoke(executor=lambda **kwargs: self.complete(kwargs, updated=False))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.records()[0]["status"], "solved_without_online_update")
        self.assertFalse(self.records()[0]["online_execution_gate_passed"])
        self.assertFalse(self.records()[0]["independent_GPU_tensor_and_wire_audit_passed"])

    def test_failed_final_Lean_result_is_never_published_as_solution(self):
        self.write_manifest("batch-a")

        def bad(**kwargs):
            result = self.complete(kwargs)
            path = kwargs["output_root"] / kwargs["session_id"] / "result.json"
            final = json.loads(path.read_bytes())
            final["solved"] = False
            path.write_bytes(batch.encoded(final))
            return result

        self.assertEqual(self.invoke(executor=bad)["status"], "manual_intervention_required")
        self.assertFalse((self.output / "solutions.jsonl").exists())

    def test_second_writer_cannot_enter_and_OS_lock_is_reusable(self):
        self.output.mkdir()
        with batch.batch_lock(self.output):
            with self.assertRaisesRegex(batch.BatchBlocked, "another batch writer"):
                self.invoke(executor=lambda **kwargs: self.fail("second writer executed"))
        self.assertEqual(self.invoke()["status"], "completed")

    def test_atomic_append_failure_recovers_completed_result(self):
        self.write_manifest("batch-a")
        real_publish = batch.publish

        def fail_commit(path, content, **kwargs):
            if path.name == "solutions.jsonl":
                raise OSError("simulated publication failure")
            return real_publish(path, content, **kwargs)

        with patch.object(batch, "publish", side_effect=fail_commit):
            self.assertEqual(self.invoke()["status"], "manual_intervention_required")
        self.assertFalse((self.output / "solutions.jsonl").exists())
        result = self.invoke(executor=lambda **kwargs: self.fail("committed online job retried"))
        self.assertEqual(result["recovered_without_GPU_calls"], ["batch-a"])
        self.assertTrue((self.output / "solutions.jsonl").read_bytes().endswith(b"\n"))

    def test_failure_after_replace_cannot_overwrite_a_committed_prefix(self):
        real_publish = batch.publish
        failed = False

        def fail_after_commit(path, content, **kwargs):
            nonlocal failed
            result = real_publish(path, content, **kwargs)
            if path.name == "solutions.jsonl" and not failed:
                failed = True
                raise OSError("simulated directory fsync failure after replace")
            return result

        with patch.object(batch, "publish", side_effect=fail_after_commit):
            first = self.invoke()
        self.assertEqual(first["status"], "manual_intervention_required")
        original = (self.output / "solutions.jsonl").read_bytes()
        self.assertEqual(len(self.records()), 1)
        again = self.invoke(executor=lambda **kwargs: self.fail("finished session was rerun"))
        self.assertEqual(len(again["skipped"]), 1)
        self.assertEqual(len(again["recovered_without_GPU_calls"]), 1)
        self.assertTrue((self.output / "solutions.jsonl").read_bytes().startswith(original))
        self.assertEqual(len(self.records()), 2)

    def test_manifest_duplicate_and_path_traversal_fail_before_start(self):
        for rows in ([{"session_id": "duplicate", "theorem_file": "Proof.lean"}] * 2,
                     [{"session_id": "bad-path", "theorem_file": "../Proof.lean"}]):
            with self.subTest(rows=rows):
                self.manifest.write_bytes(b"".join(batch.encoded(row) for row in rows))
                with self.assertRaises(batch.BatchBlocked):
                    self.invoke(executor=lambda **kwargs: self.fail("bad manifest executed"))
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
