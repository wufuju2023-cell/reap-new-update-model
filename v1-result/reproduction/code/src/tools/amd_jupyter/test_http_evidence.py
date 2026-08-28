"""Synthetic local-only evidence fixtures; not remote/GPU verification."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from tools.amd_jupyter.collect_http_evidence import (
    MAX_STDOUT, PAGE_BYTES, assemble, blob, collect, decode_blob, json_bytes, read_page, sha,
)
from tools.amd_jupyter.audit_online_wire import audit


class HttpEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.jobs = self.root / "jobs"
        self.jobs.mkdir()
        self.sid = "online-20260827-b"
        self.counter = 0

    def job(self, action, body, response, status=200, method="POST"):
        self.counter += 1
        request_id = f"{self.counter:032x}"
        directory = self.jobs / request_id
        directory.mkdir()
        request = {"request_id": request_id, "method": method,
                   "path": "/sessions/" + self.sid + ("/" + action if action else ""),
                   "body": body, "allowed_sessions": [self.sid], "timeout_seconds": 60}
        raw = response if isinstance(response, bytes) else json_bytes(response)
        (directory / "request.json").write_bytes(json_bytes(request))
        (directory / "response.bin").write_bytes(raw)
        (directory / "result.json").write_bytes(json_bytes({"request_id": request_id,
            "request_sha256": sha(json_bytes(request)), "state": "done", "status": status,
            "sha256": sha(raw), "bytes": len(raw), "completed_at": self.counter}))
        return directory

    def collection(self):
        metadata = collect(self.jobs, self.root / "snapshot.json.gz", [self.sid])
        chunks = []
        for offset in range(0, metadata["bytes"], PAGE_BYTES):
            page = read_page(self.root / "snapshot.json.gz", metadata["sha256"], offset)
            encoded = json_bytes(page)
            self.assertLessEqual(len(encoded), MAX_STDOUT)
            path = self.root / f"chunk-{offset:06d}.json"
            path.write_bytes(encoded)
            chunks.append(path)
        output = self.root / "collection.json"
        assemble(chunks, output, metadata["sha256"])
        return output

    def test_exact_bytes_pages_integrity_and_no_original_writes(self):
        response = os.urandom(90000)
        directory = self.job("value/v1/chat/completions", {"messages": [{"role": "user", "content": "∀ x"}]}, response)
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        output = self.collection()
        entry = json.loads(output.read_bytes())["entries"][0]
        self.assertEqual(decode_blob(entry["files"]["response.bin"]), response)
        self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})
        with self.assertRaises(ValueError):
            read_page(self.root / "snapshot.json.gz", "0" * 64, 0)
        with self.assertRaises(ValueError):
            collect(self.jobs, self.root / "snapshot.json.gz", [self.sid])
        with self.assertRaises(ValueError):
            collect(self.jobs, self.jobs / "bad.gz", [self.sid])
        with self.assertRaises(ValueError):
            collect(self.jobs, self.root / "other.gz", ["unrelated"])

    def make_cpu_and_jobs(self):
        cpu = self.root / "cpu"
        (cpu / "checkpoints").mkdir(parents=True)
        def save(name, value): (cpu / name).write_bytes(json_bytes(value))
        tree = self.sid + ".tree0"
        metadata = {"objective": "search_visit_backup", "gamma": 0.99,
                    "value_semantics": "reap.search_backup_discounted_return.v1", "max_distance": 1375.7}
        created = {"session_id": self.sid, "policy_version": 0, "value_metadata": metadata}
        save("session.json", {"session_id": self.sid, "tree_id": tree, "gamma": 0.99})
        save("create-receipt.json", created)
        self.job("", {}, created, 201)
        event = {"event_id": self.sid + ".s0.n0", "session_id": self.sid, "tree_id": tree,
                 "step": 0, "node_index": 0, "policy_version": 0, "gamma": 0.99,
                 "terminal_verified": False, "prompt": "prompt", "backup": {"value_sum": -2, "visits": 1}}
        detail = {"objective": "search_visit_backup", "optimizer_steps": 1,
                  "prompt_sha256": sha(b"prompt"), "search_trace": {k: event[k] for k in
                    ("tree_id", "step", "node_index", "policy_version", "gamma", "terminal_verified")},
                  "training_config": {"value_floor": 1e-6}, "value_target": 0.99,
                  "parameter_diffs": {name: {"before_sha256": "a" * 64, "after_sha256": "b" * 64}
                                      for name in ("adapter", "value_head", "optimizer")}}
        detail.update({key: True for key in ("finite_loss", "finite_gradients", "finite_parameters", "finite_optimizer_state", "base_parameters_frozen")})
        receipt = {"event_id": event["event_id"], "applied": True, "idempotent": False, "policy_version": 1, "detail": detail}
        save("checkpoints/learn-000000.request.json", event)
        save("checkpoints/learn-000000.receipt.json", receipt)
        self.job("learn/v1", {"expected_policy_version": 0, "event": event}, receipt)
        events, walls = [], []
        def emit(kind, step, version, timestamp, **fields):
            events.append({"kind": kind, "step": step, "policy_version": version, "monotonic_ns": timestamp,
                           "sequence": len(events), "session_id": self.sid, "tree_id": tree, **fields})
        for version, prompt, tactic, timestamp in [(0, "prompt", "exact True.intro", 10), (1, "next prompt", "trivial", 50)]:
            emit("selection", version, version, timestamp, node_index=version)
            emit("generation", version, version, timestamp + 10, node_index=version,
                 generation_index=0, prompt=prompt, tactic=tactic, raw_logprob=-1.0, search_value=-2.0)
            response_id = f"synthetic-policy-{version}"
            walls.append({"name": "tactic_gen", "start": timestamp + 1, "stop": timestamp + 9,
                          "extra": {"result": {"id": response_id}}})
            body = {"messages": [{"role": "user", "content": prompt}]}
            self.job("policy/v1/chat/completions", body, {"id": response_id, "policy_version": version,
                "choices": [{"message": {"content": tactic}, "logprobs": {"content": [{"token": tactic, "logprob": -1.0}]}}]})
            self.job("value/v1/chat/completions", body, {"policy_version": version,
                "choices": [{"message": {"content": '{"score":2.0}'}}]})
            if version == 0:
                emit("checkpoint", 0, 0, 30)
                emit("checkpoint_ack", 0, 1, 40)
        (cpu / "observer.jsonl").write_bytes(b"\n".join(json_bytes(e) for e in events) + b"\n")
        (cpu / "wall_clock.jsonl").write_bytes(b"\n".join(json_bytes(e) for e in walls) + b"\n")
        return cpu

    def test_replica_target_port_is_preserved_and_hashed(self):
        directory = self.job("", {}, {}, 201)
        request = json.loads((directory / "request.json").read_bytes())
        request["target_port"] = 8761
        (directory / "request.json").write_bytes(json_bytes(request))
        result = json.loads((directory / "result.json").read_bytes())
        result["request_sha256"] = sha(json_bytes(request))
        (directory / "result.json").write_bytes(json_bytes(result))
        output = self.collection()
        entry = json.loads(output.read_bytes())["entries"][0]
        self.assertEqual(json.loads(decode_blob(entry["files"]["request.json"])), request)

    def test_invalid_replica_target_port_refused_without_output(self):
        directory = self.job("", {}, {}, 201)
        request = json.loads((directory / "request.json").read_bytes())
        for port in (True, 8763, 8761.0, "8761", None):
            with self.subTest(port=port):
                request["target_port"] = port
                (directory / "request.json").write_bytes(json_bytes(request))
                output = self.root / "invalid.gz"
                with self.assertRaises(ValueError):
                    collect(self.jobs, output, [self.sid])
                self.assertFalse(output.exists())

    def test_explicit_new_session_and_experience_export_excludes_other_sessions(self):
        self.job("", {}, {"session_id": self.sid}, 201)
        self.sid = "experience-source-01"
        self.job("experience/v1", {"experience_id": "accepted-source-01"},
                 {"experience_id": "accepted-source-01"}, 201)
        output = self.collection()
        document = json.loads(output.read_bytes())
        self.assertEqual([e["session_id"] for e in document["entries"]], [self.sid])
        self.assertEqual(len(document["excluded"]), 1)

    def test_retirement_submit_and_read_evidence_remain_session_scoped(self):
        self.job("retire/v1", {"name": "final", "expected_policy_version": 1}, {"status": "released"})
        self.job("retire/v1", {}, {"status": "released"}, method="GET")
        selected = self.sid
        self.sid = "unrelated"
        self.job("retire/v1", {}, {"status": "prepared"}, method="GET")
        self.sid = selected
        output = self.collection()
        document = json.loads(output.read_bytes())
        self.assertEqual(len(document["entries"]), 2)
        self.assertEqual(len(document["excluded"]), 1)
        requests = [json.loads(decode_blob(entry["files"]["request.json"])) for entry in document["entries"]]
        self.assertEqual([r["method"] for r in requests], ["POST", "GET"])

    def test_reject_invalid_allowlist_and_unbound_request(self):
        for sessions in ([], ["../escape"], ["bad/id"], [self.sid, self.sid], ["a" * 65], [None]):
            with self.subTest(sessions=sessions), self.assertRaises(ValueError):
                collect(self.jobs, self.root / "bad.gz", sessions)
        directory = self.job("", {}, {}, 201)
        request = json.loads((directory / "request.json").read_bytes())
        request["allowed_sessions"] = ["other"]
        (directory / "request.json").write_bytes(json_bytes(request))
        with self.assertRaisesRegex(ValueError, "not authorized"):
            collect(self.jobs, self.root / "unbound.gz", [self.sid])

    def test_synthetic_wire_links_and_fail_closed_wrong_policy_version(self):
        cpu = self.make_cpu_and_jobs()
        output = self.collection()
        result = audit(output, cpu)
        self.assertEqual(result["status"], "MATCHED_PARTIAL_WIRE")
        self.assertTrue(result["learns"][0]["later_generation_wire_verified"])
        self.assertFalse(result["final_lean_execution_report_present"])
        # Corrupting retained raw bytes is detected before semantic matching.
        bad = json.loads(output.read_bytes())
        bad["entries"][0]["files"]["response.bin"]["sha256"] = "0" * 64
        corrupt = self.root / "corrupt.json"
        corrupt.write_bytes(json_bytes(bad))
        with self.assertRaisesRegex(ValueError, "hash"):
            audit(corrupt, cpu)
        # Even internally consistent retained bytes must fail if the backend
        # version differs from the observer's ACK label.
        bad = json.loads(output.read_bytes())
        for entry in bad["entries"]:
            response = json.loads(decode_blob(entry["files"]["response.bin"]))
            if response.get("id") == "synthetic-policy-1":
                response["policy_version"] = 2
                raw = json_bytes(response)
                result = json.loads(decode_blob(entry["files"]["result.json"]))
                result.update(sha256=sha(raw), bytes=len(raw))
                entry["files"]["response.bin"] = blob(raw)
                entry["files"]["result.json"] = blob(json_bytes(result))
        wrong_version = self.root / "wrong-version.json"
        wrong_version.write_bytes(json_bytes(bad))
        with self.assertRaisesRegex(ValueError, "raw policy version"):
            audit(wrong_version, cpu)


if __name__ == "__main__":
    unittest.main()
