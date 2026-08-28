"""Local CPU contract tests; neither remote GPU nor real Lean execution evidence.

The numeric case reuses the tiny Torch fixture.  HTTP cases bind only ephemeral
loopback ports and never construct an OpenCLI transport or open a browser.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from cpu_runtime.http_clients import GpuHttpClient
from cpu_runtime.online_ttt import OnlineCoordinator, validate_created, validate_learn_body
from gpu_runtime.search_objective import (
    MAX_CANDIDATES, VALUE_SEMANTICS, prepare_search_event, value_to_distance,
)


@contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("owned local HTTP server did not stop")


class ObserverFixture:
    """Observer-shaped input, including real created/merged/eval dispositions."""
    def __init__(self, root, learn):
        self.coordinator = OnlineCoordinator(
            "contract-a", "contract-a.tree0", root / "acks", learn, gamma=0.99)
        self.sequence = 0

    def record(self, kind, *, step=0, **fields):
        result = {"schema_version": "reap.training.observer.v1",
                  "session_id": "contract-a", "tree_id": "contract-a.tree0",
                  "sequence": self.sequence, "policy_version": self.coordinator.version,
                  "kind": kind, "step": step, **fields}
        self.sequence += 1
        return result

    def generate(self, tactic, *, node=0, candidate=0, child=1,
                 disposition="created", logprob=-2.0, step=0):
        self.coordinator.accept(self.record("generation", step=step,
            node_index=node, generation_index=step, candidate_index=candidate,
            tactic=tactic, prompt="GOAL ⊢ True\nPROOFSTEP\n", ps="",
            raw_logprob=logprob, goal_state="⊢ True", node_kind="OR",
            state_key=f"state-{node}", search_value=-3.0))
        self.coordinator.accept(self.record("eval", step=step,
            node_index=node, generation_index=step, candidate_index=candidate,
            tactic=tactic, disposition=disposition, child_index=child,
            eval_result={"ok": None} if child is not None else {"error": "rejected"},
            partial_goal=False))

    def checkpoint(self, edges, *, step=0, visits=4, total=-12.0):
        return self.record("checkpoint", step=step, gamma=0.99, root_is_solved=False,
            tree={"root_index": 0, "nodes": [{
                "data": {"toPlay": "OR", "isSolved": False,
                         "numVisit": visits, "valueSum": total},
                "children": [{"childIndex": i + 1, "edge": {
                    "tacticStr": tactic, "numVisit": count, "isFocus": False}}
                    for i, (tactic, count) in enumerate(edges)]}]})

    def target(self, edges, **kwargs):
        self.coordinator.accept(self.record("backup", node_index=0,
                                             step=kwargs.get("step", 0)))
        return self.coordinator._target(self.checkpoint(edges, **kwargs))[0]


class OnlineContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def fixture(self, learn=None):
        if learn is None:
            def learn(*args):
                raise AssertionError("pure extraction test must not submit LEARN")
        return ObserverFixture(self.root, learn)

    def prepare(self, event):
        return prepare_search_event(event, session_id="contract-a",
                                    policy_version=event["policy_version"], gamma=0.99)

    def test_canonical_tree_visits_not_generated_aliases_or_rejected_tactics(self):
        fixture = self.fixture()
        fixture.generate("short", candidate=0, child=1)
        fixture.generate("alias for short", candidate=1, child=1, disposition="merged")
        fixture.generate("invalid", candidate=2, child=None, disposition="eval_rejected")
        fixture.generate("long", candidate=3, child=2)
        prepared = self.prepare(fixture.target([("short", 3), ("long", 1)]))
        self.assertEqual([c["tactic"] for c in prepared["candidates"]], ["short", "long"])
        self.assertEqual([c["target_probability"] for c in prepared["candidates"]], [0.75, 0.25])
        self.assertEqual(prepared["tree_id"], "contract-a.tree0")
        self.assertEqual(prepared["reward"], 0)
        self.assertFalse(prepared["terminal_verified"])
        self.assertAlmostEqual(prepared["value_target"], 0.99 ** 2)
        self.assertEqual(prepared["value_trace"]["source_value_sum"], -12.0)
        self.assertEqual(prepared["value_trace"]["source_visits"], 4)

    def test_existing_and_new_candidates_may_have_different_behavior_versions(self):
        fixture = self.fixture()
        fixture.generate("short", logprob=-10.0)
        # Isolate event extraction after a committed version transition.  The
        # end-to-end numeric test below exercises the actual commit and ACK.
        fixture.coordinator.version = 1
        fixture.generate("long", child=2, step=1, logprob=-0.1)
        prepared = self.prepare(fixture.target([("short", 3), ("long", 1)], step=1))
        self.assertEqual(prepared["policy_version"], 1)
        self.assertEqual([c["behavior_version"] for c in prepared["candidates"]], [0, 1])
        self.assertEqual([c["target_probability"] for c in prepared["candidates"]], [0.75, 0.25])

    def test_merged_canonical_repetition_preserves_first_creator_provenance(self):
        fixture = self.fixture()
        fixture.generate("short", logprob=-10.0)
        fixture.coordinator.version = 1
        fixture.generate("short", step=1, logprob=-0.1, disposition="merged")
        fixture.generate("long", child=2, candidate=1, step=1)
        prepared = self.prepare(fixture.target([("short", 3), ("long", 1)], step=1))
        original = prepared["candidates"][0]
        self.assertEqual(original["behavior_version"], 0)
        self.assertEqual(original["raw_logprob"], -10.0)
        self.assertEqual(original["visits"], 3)
        self.assertFalse(fixture.coordinator.generated)

    def test_backend_candidate_limit_rejects_whole_event_not_visit_truncation(self):
        fixture = self.fixture()
        for index in range(MAX_CANDIDATES + 1):
            fixture.generate(f"tactic-{index}", candidate=index, child=index + 1)
        with self.assertRaisesRegex(ValueError, "candidate"):
            prepared = fixture.target([(f"tactic-{i}", 1) for i in range(MAX_CANDIDATES + 1)])
            self.prepare(prepared)
        self.assertEqual(fixture.coordinator.version, 0)
        self.assertFalse(fixture.coordinator.receipts)

    def test_actual_client_wrapper_has_exact_8192_byte_bridge_boundary(self):
        from tools.amd_jupyter.http_bridge import make_server
        from tools.amd_jupyter.remote_http_job import MAX_BODY, validate_request
        fixture = self.fixture()
        fixture.generate("short")
        event = fixture.target([("short", 1)])
        body = {"expected_policy_version": 0, "event": event}

        def wire_bytes():
            # This is GpuHttpClient's actual serialization, not the helper's
            # compact JSON or the event-only evidence file.
            return json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")

        event["prompt"] += "x" * (MAX_BODY - len(wire_bytes()))
        self.assertEqual(len(wire_bytes()), MAX_BODY)
        validate_learn_body(0, event)
        self.assertLess(len(json.dumps(event, ensure_ascii=False).encode("utf-8")), MAX_BODY)
        accepted = []

        def request(method, path, received):
            validate_request({"request_id": "a" * 32, "method": method, "path": path,
                "body": received, "allowed_sessions": ["contract-a"], "timeout_seconds": 30})
            accepted.append(received)
            return 200, b'{"accepted":true}'

        # Stub only the transport side of the real bridge HTTP handler.
        server = make_server(SimpleNamespace(allowed_sessions=["contract-a"], request=request), 0)
        with serving(server) as url:
            client = GpuHttpClient(url, timeout_seconds=5)
            self.assertTrue(client._request("POST", "/sessions/contract-a/learn/v1", body)["accepted"])
            event["prompt"] += "x"
            self.assertEqual(len(wire_bytes()), MAX_BODY + 1)
            with self.assertRaisesRegex(ValueError, "8192|8 KiB"):
                validate_learn_body(0, event)
            with self.assertRaisesRegex(RuntimeError, "HTTP 400.*8 KiB"):
                client._request("POST", "/sessions/contract-a/learn/v1", body)
        self.assertEqual(len(accepted), 1)

    def test_tiny_real_backend_http_receipt_and_distance_across_online_update(self):
        # Reuse the numerical backend fixture without importing its TestCase
        # into this module (which would duplicate test discovery).
        path = Path(__file__).with_name("test_search_objective.py")
        spec = importlib.util.spec_from_file_location("_online_numeric_fixture", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        backend = module.SearchBackendCPUTests.make_backend(self)
        runtime = module.SearchBackendCPUTests.runtime(self, backend)
        from http.server import ThreadingHTTPServer
        from gpu_runtime.real_backend import RealProverBackend
        from gpu_runtime.schemas import parse_chat_request
        from gpu_runtime.server import RuntimeHandler

        class LocalHandler(RuntimeHandler):
            backend_name = "real-search-tiny-cpu-test"
        LocalHandler.runtime = runtime
        server = ThreadingHTTPServer(("127.0.0.1", 0), LocalHandler)
        with serving(server) as url:
            client = GpuHttpClient(url, timeout_seconds=10)
            created = client.create_session("contract-a")
            validate_created(created, "contract-a", 0.99)
            self.assertEqual(created["value_metadata"]["value_semantics"], VALUE_SEMANTICS)
            self.assertEqual(created["value_metadata"]["gamma"], 0.99)

            def learn(version, event):
                return client._request("POST", "/sessions/contract-a/learn/v1",
                                       {"expected_policy_version": version, "event": event})

            fixture = self.fixture(learn)
            fixture.generate("short")
            fixture.generate("long", candidate=1, child=2)
            fixture.coordinator.accept(fixture.record("backup", node_index=0))
            fixture.coordinator.accept(fixture.checkpoint([("short", 3), ("long", 1)]))
            receipt = fixture.coordinator.receipts[0]
            detail = receipt["detail"]
            self.assertEqual(receipt["policy_version"], 1)
            self.assertEqual(detail["objective"], "search_visit_backup")
            self.assertEqual(detail["optimizer_steps"], 1)
            self.assertTrue(detail["finite_loss"] and detail["finite_gradients"])
            self.assertEqual(detail["search_trace"]["tree_id"], "contract-a.tree0")
            self.assertEqual(detail["search_trace"]["policy_version"], 0)
            for group in ("adapter", "value_head"):
                diff = detail["parameter_diffs"][group]
                self.assertGreater(diff["changed_tensors"], 0)
                self.assertNotEqual(diff["before_sha256"], diff["after_sha256"])
            ack = json.loads((self.root / "acks/checkpoint-000000.ack.json").read_bytes())
            self.assertEqual((ack["status"], ack["policy_version"]), ("continue", 1))

            raw = {"model": "reap", "messages": [{"role": "user", "content": "prompt"}]}
            probability = RealProverBackend.value(backend, "contract-a", parse_chat_request(raw))
            value = client._request("POST", "/sessions/contract-a/value/v1/chat/completions", raw)
            score = json.loads(value["choices"][0]["message"]["content"])["score"]
            self.assertEqual(value["policy_version"], 1)
            self.assertAlmostEqual(score, value_to_distance(probability, 0.99))
            self.assertGreaterEqual(score, 1.0)
            self.assertNotEqual(score, probability)
            fixture.coordinator.accept(fixture.record("checkpoint_ack"))
            fixture.generate("short", node=1, child=3, step=1)
            self.assertEqual(fixture.coordinator.post_update_generations[-1]["tree_id"], "contract-a.tree0")


if __name__ == "__main__":
    unittest.main()
