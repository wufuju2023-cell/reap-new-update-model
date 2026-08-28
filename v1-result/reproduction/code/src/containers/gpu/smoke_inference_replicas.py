#!/usr/bin/env python3
"""Two independent fixed-release inference processes; never learn or download.

The CLI requires AMD in each child. Injectable engines/launchers below are for
local mechanism tests only, never a CLI CPU fallback. Each child owns its model,
runtime, RNG, snapshot and allocator. A command is published once; an uncertain
result stops the experiment without restarting any child or replaying a command.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from containers.gpu.smoke_continual_mixed_learner import save, inventory, backend_options
from containers.gpu.smoke_gpu import require, equal_tree
from containers.gpu.smoke_mixed_learner import capture, fresh_actor
from containers.gpu.smoke_policy_scoring import CudaMeter, read_json, compare_rows
from gpu_runtime.learner_release_store import content_sha256

SOURCE = "0103155583b260c7863ea58bbf2d3377444c272db0ea4f3a75dd49c40b65c98e"
BASE = "2ff73d37f6f4edad02f5c2e67bdabeeecef97a187b4747834e6acdf648980839"
SIDS = ("inference-replica-0", "inference-replica-1")
SCHEMA = "reap.inference-replicas.v1"
MAX_JSON = 8 * 1024 * 1024


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    value, _ = read_json(path)
    return value


def validate_requests(value):
    require(isinstance(value, list) and 1 <= len(value) <= 2, "one or two fixed prompts required")
    for row in value:
        require(isinstance(row, dict) and set(row) == {"prompt", "n", "max_tokens", "temperature"},
                "each prompt requires exact prompt/n/max_tokens/temperature fields")
        require(isinstance(row["prompt"], str) and 0 < len(row["prompt"].encode()) <= 4096,
                "prompt must contain 1..4096 UTF-8 bytes")
        require(type(row["n"]) is int and 1 <= row["n"] <= 2, "one or two candidates required")
        require(type(row["max_tokens"]) is int and 1 <= row["max_tokens"] <= 32, "1..32 generated tokens required")
        require(type(row["temperature"]) in (int, float) and math.isfinite(row["temperature"])
                and 0 <= row["temperature"] <= 2, "finite temperature in [0,2] required")
    return deepcopy(value)


def validate_bounds(repeats, timeout, atol, value_atol):
    require(type(repeats) is int and 1 <= repeats <= 3, "repeats must be in 1..3")
    require(type(timeout) in (int, float) and math.isfinite(timeout) and 1 <= timeout <= 1800,
            "timeout must be finite in [1,1800] seconds")
    for number, maximum, name in ((atol, .001, "logprob"), (value_atol, .0001, "value")):
        require(type(number) in (int, float) and math.isfinite(number) and 0 <= number <= maximum,
                name + " tolerance may only tighten the fixed gate")


def parameter_versions(backend):
    return [(name, tensor.data_ptr(), tensor._version, str(tensor.dtype), list(tensor.shape))
            for name, tensor in list(backend.model.named_parameters()) + list(backend.model.named_buffers())]


def rng_digest(backend):
    torch = backend.torch
    items = [torch.get_rng_state()]
    if backend.device.type == "cuda":
        items.append(torch.cuda.get_rng_state(backend.device))
    return [digest(t.cpu().numpy().tobytes()) for t in items]


def memory(meter):
    result = meter.memory()
    require(all(type(x) is int and x >= 0 for x in result.values()), "invalid allocator counters")
    return result


def rss_bytes():
    """Linux current RSS, not peak GPU memory. Actual GPU workers run on Linux."""
    path = Path('/proc/self/status')
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1])*1024
    return None


class ReplicaEngine:
    """One initialized fixed actor. Measured calls use unchanged public runtime."""
    def __init__(self, runtime, backend, sid, release, requests, meter):
        self.runtime, self.backend, self.sid = runtime, backend, sid
        self.requests, self.release, self.meter = validate_requests(requests), release, meter
        self.baseline = capture(runtime, backend, sid)
        theorem = content_sha256(requests)  # Synthetic probe identity, not a proved theorem.
        require(fresh_actor(self.baseline, release, theorem), "replica is not a fresh fixed-release actor")
        self.snapshot = runtime.snapshot(sid, "probe-initial")
        self.initial_fingerprints = runtime.actor.submit(lambda: backend.session_fingerprints(sid))
        self.prepared = False
        self.measured = False

    def reset(self):
        require(not self.measured, "must audit the preceding measurement before restore")
        self.runtime.restore(self.sid, "probe-initial")
        restored = capture(self.runtime, self.backend, self.sid)
        require(equal_tree(self.backend.torch, self.baseline, restored), "complete private initial state restore differs")
        self.versions = parameter_versions(self.backend)
        self.host_rng = rng_digest(self.backend)
        self.prepared = True
        return {"full_restore_equal": True, "session_id": self.sid, "policy_version": 0}

    def measure(self):
        require(self.prepared and not self.measured, "measurement requires a fresh complete reset")
        self.prepared = False  # A failed call never receives an implicit retry.
        self.meter.synchronize(); self.meter.reset()
        start = time.monotonic_ns()
        outputs = []
        for request in self.requests:
            raw = {"model": "reap", "messages": [{"role": "user", "content": request["prompt"]}],
                   "n": request["n"], "temperature": request["temperature"],
                   "max_tokens": request["max_tokens"], "logprobs": True}
            policy = self.runtime.policy(self.sid, raw)
            value = self.runtime.value(self.sid, raw)
            require(type(policy["policy_version"]) is int and policy["policy_version"] == 0
                    and type(value["policy_version"]) is int and value["policy_version"] == 0,
                    "fixed actor version changed")
            require(len(policy["choices"]) == request["n"] and len(value["choices"]) == 1, "response count mismatch")
            score = json.loads(value["choices"][0]["message"]["content"])["score"]
            require(type(score) in (int, float) and math.isfinite(score)
                    and 1 <= score <= self.backend.max_distance, "nonfinite/out-of-support value")
            choices = []
            for index, row in enumerate(policy["choices"]):
                require(row["index"] == index and isinstance(row["message"]["content"], str), "choice mapping differs")
                logs = row["logprobs"]["content"]
                require(isinstance(logs, list) and len(logs) <= request["max_tokens"], "invalid token score count")
                compare_rows([logs], [logs], 0, 0)  # Checks every token/logprob for finiteness.
                choices.append({"content": row["message"]["content"], "tokens": logs})
            outputs.append({"choices": choices, "value": score, "policy_version": 0})
        self.meter.synchronize()
        end = time.monotonic_ns()
        self.measured = True
        return {"pid": os.getpid(), "session_id": self.sid, "start_monotonic_ns": start,
                "end_monotonic_ns": end, "synchronized_seconds": (end-start)/1e9,
                "outputs": outputs, "memory": memory(self.meter)}

    def audit(self):
        require(self.measured, "audit requires one completed measurement")
        after = capture(self.runtime, self.backend, self.sid)
        before_private, private = self.baseline["backend"], after["backend"]
        unchanged = equal_tree(self.backend.torch, self.baseline["metadata"], after["metadata"])
        unchanged = unchanged and all(equal_tree(self.backend.torch, before_private[key], private[key])
            for key in before_private if key != "rng")
        require(unchanged and parameter_versions(self.backend) == self.versions, "parameters/private non-RNG state changed")
        require(self.host_rng == rng_digest(self.backend), "inference consumed outside session RNG")
        fingerprints = self.runtime.actor.submit(lambda: self.backend.session_fingerprints(self.sid))
        for key in ("adapter", "value_head", "optimizer"):
            require(fingerprints[key] == self.initial_fingerprints[key], "immutable actor parameters/optimizer changed")
        self.measured = False
        return {"non_rng_state_unchanged": True, "host_rng_unchanged": True, "policy_version": 0,
                "fingerprints": fingerprints, "logical_state_sha256": content_sha256(after["metadata"]),
                "actor_metrics": self.runtime.actor.metrics(), "memory": memory(self.meter),
                "base_check_scope": "parameter/buffer identity/version; no new full frozen-base byte hash"}

    def close(self):
        self.runtime.close()
        return {"closed": True}


class DeviceMeter(CudaMeter):
    def memory(self):
        free, total = self.torch.cuda.mem_get_info(self.device)
        result = {**super().memory(), "device_global_free_bytes": free, "device_global_total_bytes": total,
                  "sample_monotonic_ns": time.monotonic_ns()}
        rss = rss_bytes()
        if rss is not None:
            result['process_rss_bytes'] = rss
        return result


def build_engine(config, worker_root):
    import torch
    from gpu_runtime import GpuRuntime
    from gpu_runtime.mixed_backend import MixedReplayBackend
    from gpu_runtime.mixed_learner_store import MixedLearnerStore
    from gpu_runtime.real_backend import PolicyScoringConfig
    require(bool(torch.version.hip) and torch.cuda.is_available(), "actual AMD GPU required; no CPU fallback")
    require(type(config['torch_threads']) is int and 1 <= config['torch_threads'] <= 8, 'fixed CPU thread bound required')
    torch.set_num_threads(config['torch_threads'])
    torch.set_num_interop_threads(1)
    require(config["model_release_sha256"] == SOURCE, "probe requires the pinned mixed R2")
    release, _ = MixedLearnerStore(Path(config["source_release_root"])).load_release(SOURCE)
    require(release == config["source_release"] and release["contract"]["base_sha256"] == BASE, "source release changed")
    backend = MixedReplayBackend(config["model_path"], dataset_root=config["replay_dataset_root"],
        mathlib_dataset_root=config["mathlib_dataset_root"], device="cuda:0", **backend_options(release))
    require(backend.hidden_size == 3584 and backend.policy_scoring == PolicyScoringConfig(), "actual 7B/default tokenwise required")
    require(backend.experience_contract() == release["contract"], "model/source full contract mismatch")
    require(all(t.device == torch.device("cuda:0") for t in backend.model.parameters()), "model parameters are not all on target GPU")
    for row in config["requests"]:
        require(backend.tokenizer(row["prompt"], return_tensors="pt")["input_ids"].shape[1] + row["max_tokens"] <= 2048,
                "prompt plus generation exceeds bounded probe context")
    runtime = GpuRuntime(backend=backend, snapshot_root=worker_root/"snapshots",
        learner_release_root=Path(config["source_release_root"]), max_resident_sessions=1)
    try:
        sid = config["session_id"]
        created = runtime.create_session(sid, role="actor", theorem_id=content_sha256(config["requests"]),
                                         model_release_sha256=SOURCE)
        save(worker_root, "created.json", created)
        engine = ReplicaEngine(runtime, backend, sid, release, config["requests"], DeviceMeter(torch, backend.device))
        ready = {"pid": os.getpid(), "session_id": sid, "actual_amd_7b": True, "hip": torch.version.hip,
                 "torch_version": torch.__version__, "dependencies": {name: importlib.metadata.version(name)
                    for name in ("transformers", "peft")}, "model_release_sha256": SOURCE,
                 "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
                 "process_rss_bytes": rss_bytes(),
                 "weights_sha256": release["weights_sha256"], "contract_sha256": content_sha256(release["contract"]),
                 "policy_scoring": "tokenwise", "policy_version": 0, "device": str(backend.device),
                 "device_visibility": {name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")},
                 "snapshot_manifest_sha256": digest((engine.snapshot/"manifest.json").read_bytes()),
                 "memory": memory(engine.meter), "initial_fingerprints": engine.initial_fingerprints}
        return engine, ready
    except BaseException:
        runtime.close()
        raise


def worker_loop(root, config, factory=build_engine):
    engine = None
    ok = False
    error = None
    try:
        loading_started = time.monotonic()
        engine, ready = factory(config, root)
        ready['load_initialization_snapshot_seconds'] = time.monotonic()-loading_started
        save(root, "ready.json", ready)
        for seq in range(1, 65):
            command_path = root/f"command-{seq:04}.json"
            deadline = time.monotonic() + config["timeout_seconds"]
            while not command_path.exists():
                require(time.monotonic() < deadline, "worker command wait expired; no automatic restart")
                time.sleep(.01)
            command = read(command_path)
            require(set(command) == {"seq", "action", "session_id"} and type(command["seq"]) is int
                    and command["seq"] == seq and command["session_id"] == config["session_id"], "command identity differs")
            action = command["action"]
            require(action in ("reset", "measure", "audit", "stop"), "unsupported inference-only command")
            operation_started = time.monotonic()
            result = getattr(engine, "close" if action == "stop" else action)()
            result['worker_operation_wall_seconds'] = time.monotonic()-operation_started
            if action == "stop":
                engine = None
            save(root, f"response-{seq:04}.json", {"seq": seq, "action": action,
                "command_sha256": digest(command_path.read_bytes()), "result": result})
            if action == "stop":
                ok = True
                break
        require(ok, "worker command bound exhausted")
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "mutation_retry_allowed": False}
    finally:
        if engine is not None:
            try:
                engine.close()
            except BaseException as exc:
                ok = False
                error = {"type": type(exc).__name__, "message": str(exc), "during": "cleanup", "mutation_retry_allowed": False}
        save(root, "worker-exit.json", {"ok": ok, "error": error, "pid": os.getpid(), "retry_allowed": False})
    return 0 if ok else 1


class ProcessClient:
    def __init__(self, root, config):
        self.root, self.config, self.seq = root, config, 0
        save(root, "config.json", config)
        self.log = (root/"process.log").open("xb")
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
        self.process = subprocess.Popen([sys.executable, "-B", "-m", "containers.gpu.smoke_inference_replicas",
            "--worker-root", str(root)], stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
            env=environment, start_new_session=True)

    def poll(self):
        return self.process.poll()

    def wait(self, timeout):
        result = self.process.wait(timeout=timeout)
        self.log.close()
        return result

    def terminate(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=5)
        self.log.close()


def wait_file(client, name, timeout):
    deadline = time.monotonic() + timeout
    while not (client.root/name).exists():
        require(client.poll() is None, f"worker exited before {name}; no retry")
        require(not (client.root/"worker-exit.json").exists(), f"worker stopped before {name}; no retry")
        require(time.monotonic() < deadline, f"unknown outcome waiting for {name}; do not resubmit")
        time.sleep(.01)
    return read(client.root/name)


def submit(client, action):
    client.seq += 1
    name = f"command-{client.seq:04}.json"
    record = {"seq": client.seq, "action": action, "session_id": client.config["session_id"]}
    pin = save(client.root, name, record)["sha256"]
    return {"seq": client.seq, "action": action, "sha256": pin}


def receive(client, ticket, timeout):
    record = wait_file(client, f"response-{ticket['seq']:04}.json", timeout)
    require(set(record) == {"seq", "action", "command_sha256", "result"}
            and record["seq"] == ticket["seq"] and record["action"] == ticket["action"]
            and record["command_sha256"] == ticket["sha256"], "response is not bound to original command")
    return record["result"]


def compare_outputs(reference, actual, atol, value_atol):
    require(len(reference) == len(actual), "request count differs")
    scores = []
    maximum_value = 0.0
    for a, b in zip(reference, actual):
        require(type(a["policy_version"]) is int and a["policy_version"] == 0
                and type(b["policy_version"]) is int and b["policy_version"] == 0, "fixed versions differ")
        require(len(a["choices"]) == len(b["choices"]), "candidate count differs")
        for left, right in zip(a["choices"], b["choices"]):
            require(left["content"] == right["content"], "generated tactic differs; no numeric tolerance for tokens")
            scores.append(compare_rows([left["tokens"]], [right["tokens"]], atol, 0))
        require(all(type(x) in (int, float) and math.isfinite(x) for x in (a["value"], b["value"])), "nonfinite value")
        maximum_value = max(maximum_value, abs(a["value"]-b["value"]))
    return {"ok": all(row["ok"] for row in scores) and maximum_value <= value_atol,
            "logprob_comparisons": scores, "maximum_value_error": maximum_value}


def summarize_round(serial, parallel, audits_serial, audits_parallel, *, atol, value_atol):
    require(len(serial) == len(parallel) == len(audits_serial) == len(audits_parallel) == 2, "exactly two replicas required")
    comparisons = []
    for i in range(2):
        require(serial[i]["session_id"] == parallel[i]["session_id"] == SIDS[i], "replica/session crossed")
        require(serial[i]["pid"] == parallel[i]["pid"], "worker was replaced")
        for audit in (audits_serial[i], audits_parallel[i]):
            require(audit["non_rng_state_unchanged"] is True and audit["host_rng_unchanged"] is True
                    and type(audit["policy_version"]) is int and audit["policy_version"] == 0, "worker state gate failed")
        require(audits_serial[i]["fingerprints"] == audits_parallel[i]["fingerprints"]
                and audits_serial[i]["logical_state_sha256"] == audits_parallel[i]["logical_state_sha256"],
                "serial/parallel RNG or complete private state differs")
        comparisons.append(compare_outputs(serial[i]["outputs"], parallel[i]["outputs"], atol, value_atol))
    require(serial[0]["pid"] != serial[1]["pid"], "replicas are not independent processes")
    for record in [*serial, *parallel]:
        start, end = record["start_monotonic_ns"], record["end_monotonic_ns"]
        require(type(start) is int and type(end) is int and 0 < start < end, "invalid monotonic interval")
        require(math.isclose(record["synchronized_seconds"], (end-start)/1e9, rel_tol=1e-12), "duration differs from interval")
    require(serial[0]["end_monotonic_ns"] <= serial[1]["start_monotonic_ns"], "serial reference overlapped")
    overlap = max(0, min(r["end_monotonic_ns"] for r in parallel)-max(r["start_monotonic_ns"] for r in parallel))/1e9
    serial_seconds = sum(r["synchronized_seconds"] for r in serial)
    parallel_span = (max(r["end_monotonic_ns"] for r in parallel)-min(r["start_monotonic_ns"] for r in parallel))/1e9
    return {"numerical_and_state_passed": all(c["ok"] for c in comparisons), "comparisons": comparisons,
            "serial_inference_seconds_sum": serial_seconds, "parallel_inference_seconds_span": parallel_span,
            "synchronized_call_overlap_seconds": overlap, "calls_overlapped": overlap > 0,
            "serial_over_parallel_span_ratio": serial_seconds/parallel_span}


def orchestrate(root, config, launcher=ProcessClient):
    clients, ready, rounds = [], [], []
    timeout = config["timeout_seconds"]
    validate_bounds(config["repeats"], timeout, config["atol"], config["value_atol"])
    try:
        # Fit naturally: the second actual load must succeed while the first is
        # resident. Never evict the first, shrink precision or retry an OOM.
        for index, sid in enumerate(SIDS):
            directory = root/f"replica-{index}"; directory.mkdir(exist_ok=False)
            child = {**config, "session_id": sid}
            clients.append(launcher(directory, child))
            item = wait_file(clients[-1], "ready.json", timeout)
            require(item["session_id"] == sid and item["actual_amd_7b"] is True
                    and item["model_release_sha256"] == config["model_release_sha256"]
                    and item["policy_scoring"] == "tokenwise" and type(item["policy_version"]) is int
                    and item["policy_version"] == 0, "worker readiness/profile gate failed")
            ready.append(item)
        require(ready[0]["pid"] != ready[1]["pid"], "second process did not load independently")
        for key in ("weights_sha256", "contract_sha256", "device", "device_visibility",
                    "torch_num_threads", "torch_num_interop_threads"):
            require(ready[0][key] == ready[1][key], "replica source/device differs")
        for key in ("adapter", "value_head"):
            require(ready[0]["initial_fingerprints"][key] == ready[1]["initial_fingerprints"][key], "initial release parameters differ")
        save(root, "both-ready.json", {"replicas": ready, "both_loaded_simultaneously": True})

        def call_all(action):
            tickets = [submit(c, action) for c in clients]
            return [receive(c, t, timeout) for c, t in zip(clients, tickets)]

        # One real warmup per child, serial; restore it before any measured pair.
        warmups = []
        for client in clients:
            receive(client, submit(client, "reset"), timeout)
            warmups.append(receive(client, submit(client, "measure"), timeout))
            receive(client, submit(client, "audit"), timeout)
        save(root, "warmups.json", {"excluded_from_measurement": True, "records": warmups})
        for iteration in range(config["repeats"]):
            measured, audits, parent_times, reset_records = {}, {}, {}, {}
            # Alternate mode order to expose warm-cache/order effects.
            for mode in (("serial", "parallel") if iteration % 2 == 0 else ("parallel", "serial")):
                resets = call_all("reset")
                require(all(r["full_restore_equal"] is True for r in resets), "restore failed")
                reset_records[mode] = resets
                start = time.monotonic()
                if mode == "serial":
                    measured[mode] = [receive(c, submit(c, "measure"), timeout) for c in clients]
                else:
                    measured[mode] = call_all("measure")
                parent_times[mode] = time.monotonic()-start
                require(all(row['pid'] == ready[i]['pid'] for i, row in enumerate(measured[mode])), 'measuring worker PID changed')
                audits[mode] = call_all("audit")  # Heavy capture outside both timing modes.
            summary = summarize_round(measured["serial"], measured["parallel"], audits["serial"], audits["parallel"],
                                      atol=config["atol"], value_atol=config["value_atol"])
            record = {"iteration": iteration, **summary, "parent_dispatch_to_receipts_seconds": parent_times,
                      "measurements": measured, "audits": audits, "restores_outside_timing": reset_records}
            save(root, f"round-{iteration:02}.json", record); rounds.append(record)
            require(summary["numerical_and_state_passed"], "serial/parallel output or state comparison failed; no retry")
        call_all("stop")
        for client in clients:
            require(client.wait(timeout) == 0, "worker process did not exit successfully")
            final = read(client.root/"worker-exit.json")
            require(final["ok"] is True and final["error"] is None, "worker finalization failed")
        return {"ok": all(r["calls_overlapped"] for r in rounds), "two_replicas_fit": True,
                "numerical_and_state_passed": True, "synchronized_calls_overlapped_each_round": all(r["calls_overlapped"] for r in rounds),
                "ready": ready, "rounds": rounds,
                "median_serial_over_parallel_span_ratio": statistics.median(r["serial_over_parallel_span_ratio"] for r in rounds),
                "median_parent_serial_over_parallel_ratio": statistics.median(
                    r["parent_dispatch_to_receipts_seconds"]["serial"]/r["parent_dispatch_to_receipts_seconds"]["parallel"] for r in rounds),
                "throughput_improvement_claimed": False}
    finally:
        for client in clients:
            client.terminate()  # Only this parent's exact-created processes; never retries.


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = set((root/"gpu_runtime").glob("*.py"))
    paths.update((root/"cpu_runtime").glob("*.py"))
    paths.update(root/"containers/gpu"/n for n in ("smoke_inference_replicas.py", "smoke_continual_mixed_learner.py",
                 "smoke_mixed_learner.py", "smoke_policy_scoring.py", "smoke_gpu.py", "smoke_kl_guard.py", "smoke_search_gpu.py"))
    return {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in sorted(paths)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    for name in ("model-path", "source-release-root", "replay-dataset-root", "mathlib-dataset-root", "requests-json", "output-dir"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--model-release-sha256", default=SOURCE)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--atol", type=float, default=.001)
    parser.add_argument("--value-atol", type=float, default=.0001)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.worker_root is not None:
        return worker_loop(args.worker_root, read(args.worker_root/"config.json"))
    require(all(getattr(args, name) is not None for name in ("model_path", "source_release_root", "replay_dataset_root",
        "mathlib_dataset_root", "requests_json", "output_dir")), "all explicit paths required")
    validate_bounds(args.repeats, args.timeout_seconds, args.atol, args.value_atol)
    require(1 <= args.torch_threads <= 8, 'torch threads must be in 1..8 and stay fixed across modes')
    require(args.model_release_sha256 == SOURCE, "only the fixed mixed R2 source is supported")
    requests, input_pin = read_json(args.requests_json); requests = validate_requests(requests)
    root = args.output_dir.absolute(); root.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": SCHEMA, "ok": False, "actual_gpu_probe_passed": False,
              "scope": "two independent fixed-release collector inference processes; no learner or Lean search",
              "model_release_sha256": SOURCE, "base_tokenizer_sha256": BASE, "requests_sha256": input_pin,
              "training_updates": 0, "automatic_retry_allowed": False, "automatic_resume": False,
              "timing_scope": "synchronized host inference windows; parent adds bounded file IPC; excludes model load, restore, warmup, state audit",
              "overlap_scope": "same-host monotonic GPU-synchronized call windows, not a kernel overlap trace",
              "memory_scope": "per-process allocator counters plus timestamped device-global readings; never sum device-global free/total",
              "theorem_identity_scope": "synthetic prompt-set identifier only; no mathematical verification claimed"}
    implementation = original = None
    try:
        from gpu_runtime.mixed_learner_store import MixedLearnerStore
        implementation = source_hashes(); save(root, "source-before.json", implementation)
        original = inventory(args.source_release_root)
        save(root, "source-release-files-before.json", original)
        source, _ = MixedLearnerStore(args.source_release_root).load_release(SOURCE)
        require(source["contract"]["base_sha256"] == BASE, "source base/tokenizer differs")
        backend_options(source)  # Exact source config, never caller-adjusted KL/dtype.
        config = {"model_path": str(args.model_path.absolute()), "source_release_root": str(args.source_release_root.absolute()),
            "replay_dataset_root": str(args.replay_dataset_root.absolute()), "mathlib_dataset_root": str(args.mathlib_dataset_root.absolute()),
            "source_release": source, "model_release_sha256": SOURCE, "requests": requests,
            "torch_threads": args.torch_threads,
            "timeout_seconds": args.timeout_seconds, "repeats": args.repeats, "atol": args.atol, "value_atol": args.value_atol}
        save(root, "input.json", config)
        report.update(orchestrate(root, config))
        require(source_hashes() == implementation, "source code changed during probe")
        require(inventory(args.source_release_root, original.keys()) == original, "old release store files changed")
        report["source_code_unchanged"] = report["old_source_files_unchanged"] = True
        report["actual_gpu_probe_passed"] = report["ok"] is True
    except BaseException as exc:
        report["ok"] = report["actual_gpu_probe_passed"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "mutation_retry_allowed": False}
    # A publication/fsync error still causes a nonzero outer process exit. The
    # caller must require that actual exit, never report.ok alone.
    save(root, "report.json", report)
    print(json.dumps({"ok": report["ok"], "output_dir": str(root), "automatic_retry_allowed": False}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
