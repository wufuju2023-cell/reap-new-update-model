"""Local process/control and tiny tensor tests, never evidence of GPU fit/speed."""
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from containers.gpu import smoke_inference_replicas as probe

REQUESTS = [{"prompt": "User: STATE:\n⊢ True\nTACTIC:", "n": 1, "max_tokens": 3, "temperature": .8}]


class FixtureEngine:
    def __init__(self, config):
        self.config, self.ready, self.measured, self.measurements = config, False, False, 0

    def reset(self):
        self.ready = True; self.measured = False
        return {"full_restore_equal": True, "session_id": self.config["session_id"], "policy_version": 0}

    def measure(self):
        if not self.ready:
            raise RuntimeError("missing reset")
        self.ready = False; self.measured = True; self.measurements += 1
        if self.config.get("fail_measure") == self.measurements:
            raise RuntimeError("fixture uncertain measurement, never resend")
        start = time.monotonic_ns()
        rng = random.Random(self.config["session_id"])
        token = str(rng.randrange(100))
        time.sleep(.06)
        end = time.monotonic_ns()
        return {"pid": os.getpid(), "session_id": self.config["session_id"],
                "start_monotonic_ns": start, "end_monotonic_ns": end,
                "synchronized_seconds": (end-start)/1e9, "memory": {"allocated_bytes": 123},
                "outputs": [{"choices": [{"content": token, "tokens": [{"token": token, "logprob": -1.0}]}],
                             "value": 2.0, "policy_version": 0}]}

    def audit(self):
        if not self.measured:
            raise RuntimeError("missing measure")
        self.measured = False
        return {"non_rng_state_unchanged": True, "host_rng_unchanged": True, "policy_version": 0,
                "fingerprints": {"rng": self.config["session_id"], "adapter": "fixed"},
                "logical_state_sha256": "a"*64}

    def close(self):
        if self.config.get("fail_cleanup"):
            raise RuntimeError("fixture cleanup failed")
        return {"closed": True}


def fixture_factory(config, root):
    if config.get("fail_fit") and config["session_id"] == probe.SIDS[1]:
        raise MemoryError("second replica fixture does not fit")
    return FixtureEngine(config), {"pid": os.getpid(), "session_id": config["session_id"],
        "actual_amd_7b": True,  # Simulated readiness only; not reachable from the real CLI.
        "model_release_sha256": probe.SOURCE, "weights_sha256": "a"*64, "contract_sha256": "b"*64,
        "policy_scoring": "tokenwise", "policy_version": 0, "device": "fixture",
        "torch_num_threads": 1, "torch_num_interop_threads": 1,
        "device_visibility": {}, "initial_fingerprints": {"adapter": "same", "value_head": "same"}}


def fixture_worker(root, config):
    raise SystemExit(probe.worker_loop(root, config, factory=fixture_factory))


class FixtureClient:
    def __init__(self, root, config):
        self.root, self.config, self.seq = root, config, 0
        self.process = multiprocessing.get_context("spawn").Process(target=fixture_worker, args=(root, config))
        self.process.start()

    def poll(self):
        return self.process.exitcode

    def wait(self, timeout):
        self.process.join(timeout)
        if self.process.is_alive():
            raise TimeoutError("fixture wait expired")
        return self.process.exitcode

    def terminate(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(5)


def config(**extra):
    return {"model_release_sha256": probe.SOURCE, "requests": REQUESTS, "timeout_seconds": 10,
            "repeats": 2, "atol": .001, "value_atol": .0001, "torch_threads": 1, **extra}


class ReplicaProcessTests(unittest.TestCase):
    def test_two_native_processes_overlap_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = probe.orchestrate(root, config(), launcher=FixtureClient)
            self.assertTrue(result["ok"])
            self.assertTrue(result["two_replicas_fit"])
            self.assertFalse(result["throughput_improvement_claimed"])
            self.assertNotEqual(result["ready"][0]["pid"], result["ready"][1]["pid"])
            for i in range(2):
                commands = [probe.read(p)["action"] for p in sorted((root/f"replica-{i}").glob("command-*.json"))]
                self.assertEqual(commands.count("measure"), 5)  # Warmup + two pairs, no retry.
                self.assertEqual(commands.count("reset"), 5)
                self.assertEqual(commands[-1], "stop")
                self.assertTrue(probe.read(root/f"replica-{i}"/"worker-exit.json")["ok"])
            self.assertTrue(all(r["synchronized_call_overlap_seconds"] > 0 for r in result["rounds"]))

    def test_second_process_fit_failure_never_evicts_and_restarts_first(self):
        made = []
        def launch(root, cfg):
            client = FixtureClient(root, cfg); made.append(client); return client
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "worker (exited|stopped)"):
                probe.orchestrate(root, config(fail_fit=True), launcher=launch)
            self.assertEqual(len(made), 2)
            self.assertTrue(all(c.poll() is not None for c in made))
            self.assertFalse((root/"both-ready.json").exists())
            self.assertEqual(list(root.glob("replica-*/command-*.json")), [])
            self.assertIn("MemoryError", (root/"replica-1/worker-exit.json").read_text())

    def test_measurement_unknown_never_replays_original_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "worker (exited|stopped)"):
                probe.orchestrate(root, config(fail_measure=2), launcher=FixtureClient)
            for folder in root.glob("replica-*"):
                commands = [probe.read(p)["action"] for p in folder.glob("command-*.json")]
                self.assertLessEqual(commands.count("measure"), 2)
            self.assertFalse((root/"round-00.json").exists())

    def test_late_cleanup_failure_cannot_pass_even_after_equal_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "worker (exited|stopped)"):
                probe.orchestrate(root, config(repeats=1, fail_cleanup=True), launcher=FixtureClient)
            self.assertTrue((root/"round-00.json").exists())
            self.assertFalse(probe.read(root/"replica-0/worker-exit.json")["ok"])

    def test_file_wait_timeout_never_publishes_command_or_restarts_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = MagicMock(root=Path(temporary)); client.poll.return_value = None
            with patch.object(probe.time, "monotonic", side_effect=[10, 12]), patch.object(probe.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "unknown outcome"):
                    probe.wait_file(client, "response-0001.json", 1)
            self.assertEqual(list(client.root.iterdir()), [])
            client.terminate.assert_not_called()

    def test_command_and_response_pins_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = MagicMock(root=Path(temporary), seq=0, config={"session_id": probe.SIDS[0]})
            ticket = probe.submit(client, "measure")
            probe.save(client.root, "response-0001.json", {"seq": 1, "action": "measure", "command_sha256": "0"*64, "result": {}})
            with self.assertRaisesRegex(RuntimeError, "bound to original"):
                probe.receive(client, ticket, 1)
            client.seq = 0
            with self.assertRaises(FileExistsError):
                probe.submit(client, "measure")


class ReplicaNumericalTests(unittest.TestCase):
    def test_strict_input_and_tolerance_bounds(self):
        for changes in ({"n": True}, {"n": 3}, {"max_tokens": 33}, {"temperature": float("nan")},
                        {"temperature": 2.1}, {"prompt": ""}, {"extra": 1}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                probe.validate_requests([{**REQUESTS[0], **changes}])
        for args in ((4, 10, .001, .0001), (1, float("inf"), .001, .0001),
                     (1, 10, .00101, .0001), (1, 10, .001, .00011), (True, 10, 0, 0)):
            with self.assertRaises(RuntimeError): probe.validate_bounds(*args)

    def test_exact_text_required_and_nonfinite_or_numeric_error_cannot_pass(self):
        output = [{"choices": [{"content": "exact h", "tokens": [{"token": "exact", "logprob": -1.0}]}],
                   "value": 3.0, "policy_version": 0}]
        self.assertTrue(probe.compare_outputs(output, deepcopy(output), .001, .0001)["ok"])
        for field in ("text", "token", "nan", "version"):
            changed = deepcopy(output)
            if field == "text": changed[0]["choices"][0]["content"] = "different"
            elif field == "token": changed[0]["choices"][0]["tokens"][0]["token"] = "different"
            elif field == "nan": changed[0]["choices"][0]["tokens"][0]["logprob"] = float("nan")
            else: changed[0]["policy_version"] = False
            with self.subTest(field=field), self.assertRaises((RuntimeError, ValueError)):
                probe.compare_outputs(output, changed, .001, .0001)
        changed = deepcopy(output); changed[0]["choices"][0]["tokens"][0]["logprob"] += .01
        self.assertFalse(probe.compare_outputs(output, changed, .001, .0001)["ok"])
        changed = deepcopy(output); changed[0]["value"] += .01
        self.assertFalse(probe.compare_outputs(output, changed, .001, .0001)["ok"])

    def test_actual_tiny_backend_public_policy_value_restore_and_rng_without_probe_learning(self):
        from tests.test_mixed_learner import MixedLearnerTests
        MixedLearnerTests.setUp(self)
        learner = MixedLearnerTests.coordinator(self)
        learner.train_next(); release = learner.publish()  # Fixture source, not probe training.
        learner.close(); self.runtime.delete_session("mixed")
        sid = probe.SIDS[0]
        self.runtime.create_session(sid, role="actor", theorem_id=probe.content_sha256(REQUESTS),
                                    model_release_sha256=release["model_release_sha256"])
        class Meter:
            def synchronize(self): pass
            def reset(self): pass
            def memory(self): return {"allocated_bytes": 0}
        engine = probe.ReplicaEngine(self.runtime, self.backend, sid, release, REQUESTS, Meter())
        with patch.object(self.backend, "learn", wraps=self.backend.learn) as learns:
            engine.reset(); serial = engine.measure(); first = engine.audit()
            engine.reset(); parallel = engine.measure(); second = engine.audit()
            self.assertTrue(probe.compare_outputs(serial["outputs"], parallel["outputs"], 0, 0)["ok"])
            self.assertEqual(first["fingerprints"], second["fingerprints"])
            self.assertEqual(first["logical_state_sha256"], second["logical_state_sha256"])
            learns.assert_not_called()
        self.assertEqual(self.backend.sessions[sid].optimizer_steps, 0)
        self.assertEqual(self.backend.sessions[sid].optimizer.state, {})

    def test_actual_gpu_factory_refuses_cpu_before_model_loading(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch unavailable")
        with patch.object(torch.version, "hip", None), patch("gpu_runtime.mixed_backend.MixedReplayBackend") as backend:
            with self.assertRaisesRegex(RuntimeError, "actual AMD GPU"):
                probe.build_engine({}, Path("unused"))
            backend.assert_not_called()

    def test_late_source_failure_clears_pass_and_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); requests = root/"requests.json"; requests.write_text(json.dumps(REQUESTS))
            argv = ["--model-path", "model", "--source-release-root", "store", "--replay-dataset-root", "replay",
                    "--mathlib-dataset-root", "human", "--requests-json", str(requests), "--output-dir", str(root/"output")]
            source = {"contract": {"base_sha256": probe.BASE}}
            with patch.object(probe, "source_hashes", side_effect=[{"source": "a"}, {"source": "b"}]), \
                    patch.object(probe, "inventory", return_value={}), patch.object(probe, "backend_options"), \
                    patch("gpu_runtime.mixed_learner_store.MixedLearnerStore") as store, \
                    patch.object(probe, "orchestrate", return_value={"ok": True}) as run:
                store.return_value.load_release.return_value = (source, {})
                self.assertEqual(probe.main(argv), 1)
                report = probe.read(root/"output/report.json")
                self.assertFalse(report["ok"]); self.assertFalse(report["actual_gpu_probe_passed"])
                self.assertIn("source code changed", report["error"]["message"])
                with self.assertRaises(FileExistsError): probe.main(argv)
                self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
