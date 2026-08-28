from __future__ import annotations

from http.server import ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from gpu_runtime import GpuRuntime, ToyBackend
from gpu_runtime.server import RuntimeHandler
from gpu_runtime.snapshot_store import _json_bytes
from cpu_runtime.http_clients import GpuHttpClient


class GpuHttpTests(unittest.TestCase):
    def test_explicit_initialization_contract_is_actor_read_and_receipt_only(self):
        client = GpuHttpClient(self.base)
        contract = {**self.runtime.backend.experience_contract(), "unicode_fixture": "数学", "nested": {"enabled": True}}
        pin = hashlib.sha256(_json_bytes(contract)).hexdigest()
        threads = []
        def actual_contract():
            threads.append(threading.current_thread())
            return contract
        with patch.object(self.runtime.backend, "experience_contract", side_effect=actual_contract), \
                patch.object(self.runtime, "create_session", wraps=self.runtime.create_session) as create:
            created = client.create_session("pinned-contract", theorem_id="problem",
                expected_initialization_contract_sha256=pin)
        self.assertEqual(threads, [self.runtime.actor._thread])
        create.assert_called_once_with("pinned-contract", theorem_id="problem")
        self.assertEqual(created["initialization_contract"], contract)
        self.assertEqual(created["initialization_contract_sha256"], pin)
        self.assertEqual({k: v for k, v in created.items() if k not in
            ("initialization_contract", "initialization_contract_sha256")},
            self.runtime.sessions.get("pinned-contract").snapshot())

    def test_default_create_does_not_request_contract_or_add_receipt_fields(self):
        client = GpuHttpClient(self.base)
        with patch.object(self.runtime.backend, "experience_contract", side_effect=AssertionError("old path must not add a read")):
            created = client.create_session("legacy-default", expected_initialization_contract_sha256=None)
        self.assertNotIn("initialization_contract", created)
        self.assertNotIn("initialization_contract_sha256", created)
        with patch.object(client, "_request", return_value={}) as request:
            client.create_session("plain")
            request.assert_called_once_with("POST", "/sessions/plain", {})
        with patch.object(client, "_request", return_value={}) as request:
            client.create_session("explicit", expected_initialization_contract_sha256="a"*64)
            request.assert_called_once_with("POST", "/sessions/explicit",
                {"expected_initialization_contract_sha256": "a"*64})

    def test_bad_or_mismatched_contract_pin_never_allocates(self):
        for index, pin in enumerate((None, True, 3, "A"*64, "a"*63, "0"*64)):
            sid = f"bad-contract-{index}"
            with self.subTest(pin=pin), patch.object(self.runtime.backend, "create_session", wraps=self.runtime.backend.create_session) as create:
                with self.assertRaises(HTTPError) as error:
                    self.request("POST", "/sessions/"+sid, {"expected_initialization_contract_sha256": pin})
                self.assertEqual(error.exception.code, 400)
                create.assert_not_called()
                self.assertNotIn(sid, self.runtime.backend._states)
                self.assertNotIn(sid, self.runtime._resident_session_ids)

    def test_missing_or_nonfinite_contract_never_allocates_even_with_explicit_pin(self):
        for index, contract in enumerate((None, {}, [], {"invalid": float("nan")}, {"invalid": object()})):
            sid = f"missing-contract-{index}"
            with self.subTest(contract=contract), patch.object(self.runtime.backend, "experience_contract", return_value=contract), \
                    patch.object(self.runtime, "create_session", wraps=self.runtime.create_session) as create:
                with self.assertRaises(HTTPError) as error:
                    self.request("POST", "/sessions/"+sid, {"expected_initialization_contract_sha256": "0"*64})
                self.assertEqual(error.exception.code, 400)
                create.assert_not_called()
        with patch.object(self.runtime.backend, "experience_contract", None), self.assertRaises(HTTPError) as error:
            self.request("POST", "/sessions/no-contract-method", {"expected_initialization_contract_sha256": "0"*64})
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.runtime.backend._states, {})

    def test_expected_contract_does_not_bypass_experience_source_compatibility(self):
        client = GpuHttpClient(self.base)
        client.create_session("source-contract", theorem_id="problem-source")
        self.runtime.learn("source-contract", expected_policy_version=0, event={"event_id": "step", "reward": 1})
        candidate = client.snapshot("source-contract", "candidate", for_experience=True)
        release = client.publish_experience("source-contract", "candidate", "contract-experience",
            {"source": candidate["source"], "completed": True, "passed": True,
             "kind": "local-mechanism", "evidence_sha256": "a"*64})
        pins = {"experience_id": "contract-experience", "experience_weights_sha256": release["weights_sha256"],
                "experience_snapshot_sha256": release["source"]["snapshot_sha256"]}
        contract = self.runtime.backend.experience_contract()
        pin = hashlib.sha256(_json_bytes(contract)).hexdigest()
        created = client.create_session("same-contract", theorem_id="problem-next", **pins,
            expected_initialization_contract_sha256=pin)
        self.assertEqual(created["initialization_contract"], contract)
        self.assertTrue(self.runtime.inspect_backend("same-contract")["learned"])
        different = {**contract, "different-contract": True}
        with patch.object(self.runtime.backend, "experience_contract", return_value=different), \
                patch.object(self.runtime.backend, "create_session", wraps=self.runtime.backend.create_session) as create:
            with self.assertRaisesRegex(RuntimeError, "compatibility mismatch"):
                client.create_session("incompatible-source", theorem_id="problem-other", **pins,
                    expected_initialization_contract_sha256=hashlib.sha256(_json_bytes(different)).hexdigest())
            create.assert_not_called()
        self.assertNotIn("incompatible-source", self.runtime._resident_session_ids)

    def test_explicit_experience_http_round_trip(self):
        client = GpuHttpClient(self.base)
        client.create_session("source", theorem_id="problem-a")
        self.runtime.learn("source", expected_policy_version=0, event={"event_id": "e", "reward": 1})
        candidate = client.snapshot("source", "candidate", for_experience=True)
        acceptance = {"source": candidate["source"], "completed": True, "passed": True,
                      "kind": "local-mechanism", "evidence_sha256": "a" * 64}
        release = client.publish_experience("source", "candidate", "exp", acceptance)
        created = client.create_session("next", theorem_id="problem-b", experience_id="exp")
        self.assertEqual(created["policy_version"], 0)
        self.assertEqual(created["lineage"]["weights_sha256"], release["weights_sha256"])
        self.assertTrue(self.runtime.inspect_backend("next")["learned"])
        pins = {"experience_weights_sha256": release["weights_sha256"],
                "experience_snapshot_sha256": release["source"]["snapshot_sha256"]}
        pinned = client.create_session("pinned", theorem_id="problem-c", experience_id="exp", **pins)
        self.assertEqual(pinned["lineage"], created["lineage"])
        with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
            client.create_session("badpin", theorem_id="problem-d", experience_id="exp",
                                  **{**pins, "experience_weights_sha256": "0" * 64})
        self.assertNotIn("badpin", self.runtime.backend._states)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.runtime = GpuRuntime(
            backend=ToyBackend(),
            snapshot_root=Path(self.temporary.name) / "snapshots",
        )
        RuntimeHandler.runtime = self.runtime
        RuntimeHandler.backend_name = "toy"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RuntimeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.runtime.close()
        self.temporary.cleanup()

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self.base + path,
            method=method,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=3) as response:
            return json.loads(response.read())

    def test_reap_chat_and_learn_routes(self) -> None:
        self.request("POST", "/sessions/http-smoke", {})
        chat = {
            "model": "reap",
            "messages": [{"role": "user", "content": "STATE:\n⊢ True\nTACTIC:"}],
            "n": 1,
            "temperature": 0.99,
            "max_tokens": 32,
            "logprobs": True,
        }
        before = self.request("POST", "/sessions/http-smoke/policy/v1/chat/completions", chat)
        self.assertEqual(before["choices"][0]["message"]["content"], ToyBackend.FAILURE_TACTIC)
        value = self.request("POST", "/sessions/http-smoke/value/v1/chat/completions", chat)
        self.assertEqual(json.loads(value["choices"][0]["message"]["content"]), {"score": 0.0})
        learned = self.request("POST", "/sessions/http-smoke/learn/v1", {
            "expected_policy_version": 0,
            "event": {
                "event_id": "negative-0",
                "state": "⊢ True",
                "tactic": ToyBackend.FAILURE_TACTIC,
                "verdict": "rejected",
                "reward": -1.0,
                "root_verified": False,
            },
        })
        self.assertEqual(learned["policy_version"], 1)
        after = self.request("POST", "/sessions/http-smoke/policy/v1/chat/completions", chat)
        self.assertEqual(after["choices"][0]["message"]["content"], "trivial")

    def test_cpu_client_transport_timeout_is_configurable(self) -> None:
        client = GpuHttpClient(self.base, timeout_seconds=600)
        with patch("cpu_runtime.http_clients.urlopen", wraps=urlopen) as opened:
            created = client.create_session("long-transport")
        self.assertEqual(created["session_id"], "long-transport")
        self.assertEqual(opened.call_args.kwargs["timeout"], 600)
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                GpuHttpClient(self.base, timeout_seconds=timeout)

    def test_health_reads_actor_metrics_without_queueing_behind_work(self) -> None:
        entered, release = threading.Event(), threading.Event()

        def work():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release worker")

        worker = threading.Thread(target=lambda: self.runtime.actor.submit(work))
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            before = self.runtime.actor.metrics()
            result = self.request("GET", "/health")
            self.assertEqual(result["gpu_actor"]["active"], 1)
            self.assertEqual(result["gpu_actor"]["submitted"], before["submitted"])
            self.assertEqual(result["gpu_actor_completed"], result["gpu_actor"]["completed"])
            self.assertEqual(result["gpu_actor_max_active"], 1)
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())

    def test_retire_client_round_trip_releases_capacity_and_reads_without_mutation(self):
        self.runtime.max_resident_sessions = 1
        client = GpuHttpClient(self.base)
        client.create_session("first")
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            client.create_session("second")
        receipt = client.retire_session("first", "final", expected_policy_version=0)
        metrics = self.runtime.actor.metrics()
        self.assertEqual(client.retirement_receipt("first"), receipt)
        self.assertEqual(self.runtime.actor.metrics()["submitted"], metrics["submitted"])
        self.assertEqual(receipt["status"], "released")
        client.create_session("second")
        with self.assertRaisesRegex(RuntimeError, "retired"):
            client.create_session("first")
        with self.assertRaisesRegex(RuntimeError, "404"):
            client.retirement_receipt("unknown")

    def test_retirement_reuse_is_explicit_and_round_trips_v2_without_export(self):
        client = GpuHttpClient(self.base)
        client.create_session("reuse")
        client.snapshot("reuse", "after-online-ttt")
        with patch.object(self.runtime.backend, "export_session", side_effect=AssertionError("re-export")):
            receipt = client.retire_session("reuse", "after-online-ttt", expected_policy_version=0, reuse_snapshot=True)
        self.assertEqual(receipt["schema_version"], "reap.gpu.retirement.v2")
        self.assertEqual(receipt["snapshot_mode"], "reuse_verified")
        self.assertEqual(client.retirement_receipt("reuse"), receipt)
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                client.retire_session("reuse", "after-online-ttt", expected_policy_version=0, reuse_snapshot=value)
        with self.assertRaises(HTTPError) as error:
            self.request("POST", "/sessions/reuse/retire/v1",
                         {"name": "after-online-ttt", "expected_policy_version": 0, "reuse_snapshot": "true"})
        self.assertEqual(error.exception.code, 400)

    def test_retirement_post_rejects_unknown_fields_and_get_preserves_prepared_status(self):
        client = GpuHttpClient(self.base)
        client.create_session("failed")
        with self.assertRaises(HTTPError) as error:
            self.request("POST", "/sessions/failed/retire/v1",
                         {"name": "final", "expected_policy_version": 0, "ignore_pending": True})
        self.assertEqual(error.exception.code, 400)
        with patch.object(self.runtime.backend, "delete_session", side_effect=RuntimeError("delete failed")):
            with self.assertRaisesRegex(RuntimeError, "500"):
                client.retire_session("failed", "final", expected_policy_version=0)
        self.assertEqual(client.retirement_receipt("failed")["status"], "prepared")
        self.assertIn("failed", self.runtime.backend._states)

    def test_capacity_cli_is_explicit_and_validated_before_model_loading(self):
        from gpu_runtime.server import build_parser
        parser = build_parser()
        self.assertIsNone(parser.parse_args([]).max_resident_sessions)
        self.assertEqual(parser.parse_args(["--max-resident-sessions", "2"]).max_resident_sessions, 2)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(["--max-resident-sessions", "0"])


class BackendOptionsTests(unittest.TestCase):
    def test_default_preserves_backend_arguments(self):
        from gpu_runtime.server import build_backend, build_parser
        with patch("gpu_runtime.server.RealSearchBackend") as construct:
            args = build_parser().parse_args(["--backend", "real-search", "--gamma", "0.99"])
            build_backend(args)
            construct.assert_called_once_with(args.model_path, device=args.device, gamma=0.99)

    def test_explicit_options_are_forwarded(self):
        from gpu_runtime.server import build_backend, build_parser
        with patch("gpu_runtime.server.RealSearchBackend") as construct:
            args = build_parser().parse_args(["--backend", "real-search", "--gamma", "0.99",
                "--policy-scoring", "candidate_chunks", "--policy-score-batch-size", "3",
                "--policy-score-token-chunk", "4", "--max-post-update-kl", "2"])
            build_backend(args)
            options = construct.call_args.kwargs
            self.assertEqual(options["max_post_update_kl"], 2)
            self.assertEqual(options["policy_scoring"].candidate_batch_size, 3)
            self.assertEqual(options["policy_scoring"].token_chunk_size, 4)

    def test_invalid_modes_fail_before_model_construction(self):
        from gpu_runtime.server import build_backend, build_parser
        for flags in (["--backend", "toy", "--policy-scoring", "candidate_chunks"],
                      ["--backend", "toy", "--policy-scoring", "tokenwise_deferred"],
                      ["--backend", "real", "--policy-scoring", "tokenwise_deferred", "--policy-score-batch-size", "3"],
                      ["--backend", "real", "--max-post-update-kl", "2"],
                      ["--policy-score-batch-size", "3"],
                      ["--policy-scoring", "candidate_chunks", "--policy-score-token-chunk", "33"]):
            with self.subTest(flags=flags), patch("gpu_runtime.server.RealProverBackend") as real, \
                    patch("gpu_runtime.server.RealSearchBackend") as search:
                with self.assertRaises(ValueError):
                    build_backend(build_parser().parse_args(flags))
                    real.assert_not_called()
                    search.assert_not_called()

    def test_deferred_scoring_is_explicit_and_forwarded_without_chunk_sizes(self):
        from gpu_runtime.server import build_backend, build_parser
        with patch("gpu_runtime.server.RealProverBackend") as construct:
            build_backend(build_parser().parse_args(["--backend", "real", "--policy-scoring", "tokenwise_deferred"]))
            self.assertEqual(construct.call_args.kwargs["policy_scoring"].mode, "tokenwise_deferred")

    def test_kl_threshold_must_be_positive_finite(self):
        from gpu_runtime.server import build_parser
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), patch("sys.stderr"), self.assertRaises(SystemExit):
                build_parser().parse_args(["--max-post-update-kl", value])


if __name__ == "__main__":
    unittest.main()
