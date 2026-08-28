#!/usr/bin/env python3
"""WSL-only A/async-worker/host-toy integration. NO Edge, remote, or GPU use.

Retains this run's stopped containers, named output volumes, HTTP job files,
snapshots, Lean results, journals, and inspect evidence in a NEW output directory.
This exercises real Lean and transport with ToyBackend, not neural GPU TTT.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.request import ProxyHandler, Request, build_opener
import uuid

try:
    from .http_bridge import Bridge, make_server
    from .remote_http_job import json_bytes, poll_job, submit_job
except ImportError:
    from http_bridge import Bridge, make_server
    from remote_http_job import json_bytes, poll_job, submit_job


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save_json(path, value):
    path.write_bytes(json_bytes(value) + b"\n")


def podman(*args, timeout=60, check=True):
    result = subprocess.run(["podman", *args], capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"podman {args[0]} failed: {result.stderr.strip()}")
    return result


def stop_owned_process(process):
    if process is None or process.poll() is not None:
        return
    require(os.getpgid(process.pid) == process.pid, "refusing to stop a process without its own group")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


class LocalJobTransport:
    """Same durable submit/poll helper, real detached workers, no browser hop."""
    def __init__(self, root):
        self.root = root
        self.workers = []
        self.lock = threading.Lock()

    def launch(self, command):
        request_id = command[-1]
        with (self.root / request_id / "worker.log").open("ab") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        self.workers.append(process)

    def __call__(self, action, request=None, *, request_id="", offset=0):
        with self.lock:
            if action == "submit":
                return submit_job(self.root, request, launcher=self.launch)
            require(action == "poll", "unexpected local transport operation")
            return poll_job(self.root, request_id, offset)

    def close(self):
        with self.lock:
            for worker in self.workers:
                stop_owned_process(worker)


def http(port, method, path, body=None):
    request = Request(f"http://127.0.0.1:{port}" + path, method=method,
                      data=None if body is None else json_bytes(body), headers={"Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(request, timeout=30) as response:
        return json.loads(response.read())


def inspect_owned(name, run_id):
    result = podman("inspect", name, check=False)
    if result.returncode:
        return None
    info = json.loads(result.stdout)[0]
    require(info["Config"]["Labels"].get("reap.local-bridge-run") == run_id, "refusing to touch another run's container")
    return info


def copy_output(name, destination):
    # Podman 4.9 cp must receive an existing destination for contents semantics.
    target = destination / "out"
    target.mkdir(exist_ok=True)
    podman("cp", name + ":/workspace/out/.", str(target))


def audit_session(output, session_id, inspect):
    require(inspect["State"]["ExitCode"] == 0, f"container failed: {session_id}")
    mounts = inspect.get("Mounts", [])
    require(len(mounts) == 1 and mounts[0]["Type"] == "volume" and mounts[0]["Destination"] == "/workspace/out", "unexpected source/data mounts")
    require(not inspect["HostConfig"].get("Devices") and not inspect["HostConfig"].get("Privileged"), "unexpected GPU devices/privileges")
    session_root = output / session_id / "out" / session_id
    records = [json.loads(line) for line in (session_root / "ttt-journal.jsonl").read_text().splitlines()]
    acts = [item for item in records if item["kind"] == "act_finished"]
    starts = [item for item in records if item["kind"] == "act_started"]
    learns = [item for item in records if item["kind"] == "learn_finished"]
    finished = [item for item in records if item["kind"] == "session_finished"][-1]
    require(len(acts) == 2 and not acts[0]["root_verified"] and acts[1]["root_verified"], "expected failed ACT then successful fresh-root ACT")
    require(len(learns) == 1 and learns[0]["policy_version"] > learns[0]["previous_policy_version"], "missing version-advancing toy learn")
    require(all(item["restart_from_root"] for item in starts) and starts[1]["policy_version"] == learns[0]["policy_version"], "fresh-root ACT did not consume new version")
    require(finished["status"] == "solved" and finished["root_verified"], "final proof not verified")
    for segment in range(2):
        result = json.loads((session_root / f"segment-{segment:03d}" / session_id / "result.json").read_text())
        require(bool(result["solved"]) == (segment == 1), "Lean result disagrees with journal")
    return {"session_id": session_id, "policy_version": finished["policy_version"],
            "segments": len(acts), "learn_steps": len(learns), "real_lean_verified": True,
            "journal": str(session_root / "ttt-journal.jsonl")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image", default="localhost/reap-cpu:v1-bridge")
    args = parser.parse_args()
    require(os.name == "posix" and "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower(), "run this local-only script inside WSL")
    output = args.output_dir.resolve()
    require(not output.exists(), "output directory must be new; evidence is never overwritten")
    image = json.loads(podman("image", "inspect", args.image).stdout)[0]
    # Check only availability; never stop an unrelated listener.
    for port in (8760, 18760):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    output.mkdir(parents=True)
    save_json(output / "image-inspect.json", image)
    run_id = "local-bridge-" + datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    sessions = [run_id + "-a", run_id + "-b"]
    names = []
    repo = Path(__file__).resolve().parents[2]
    transport = LocalJobTransport(output / "http-jobs")
    b_process = None
    server = thread = None
    report = {"ok": False, "gate": "local_CPU_containers_async_workers_host_ToyBackend",
              "browser_used": False, "remote_used": False, "gpu_verified": False,
              "neural_optimizer_verified": False, "run_id": run_id, "input_image": args.image,
              "image_id": image["Id"], "started_at": datetime.now(timezone.utc).isoformat(), "sessions": sessions}
    started = time.monotonic()
    try:
        with (output / "toy-server.log").open("wb") as log:
            b_process = subprocess.Popen([sys.executable, "-B", "-m", "gpu_runtime.server", "--backend", "toy",
                                          "--host", "127.0.0.1", "--port", "8760", "--snapshot-root", str(output / "toy-snapshots")],
                                         cwd=repo, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            require(b_process.poll() is None, "owned toy server exited; inspect toy-server.log")
            if '"ready": true' in (output / "toy-server.log").read_text():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("owned toy server did not become ready")
        require(http(8760, "GET", "/health")["backend"] == "toy", "expected owned ToyBackend")
        server = make_server(Bridge(transport, sessions, timeout_seconds=600, poll_interval=0.05), 18760)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        require(http(18760, "GET", "/health")["backend"] == "toy", "asynchronous worker health round trip failed")
        for session_id in sessions:
            name, volume = session_id, session_id + "-out"
            names.append(name)
            command = ["run", "-d", "--name", name, "--label", "reap.local-bridge-run=" + run_id,
                       "--network=slirp4netns:allow_host_loopback=true", "-v", volume + ":/workspace/out",
                       "--entrypoint", "python3", image["Id"], "-m", "cpu_runtime.run_ttt",
                       "--session-id", session_id, "--gpu-base-url", "http://10.0.2.2:18760",
                       "--project-dir", "/opt/reap-runtime", "--theorem-file", "Smoke.lean",
                       "--output-dir", "/workspace/out", "--http-timeout-seconds", "720",
                       "--segment-timeout-seconds", "900"]
            podman(*command)
            save_json(output / (session_id + "-command.json"), ["podman", *command])
        overlap = False
        last_progress = 0
        while True:
            infos = [inspect_owned(name, run_id) for name in names]
            require(all(info is not None for info in infos), "owned container disappeared")
            running = [info["State"]["Running"] for info in infos]
            if all(running) and not overlap:
                tops = [podman("top", name, "pid", "hpid", "args", check=False).stdout for name in names]
                if all("lake env lean" in top or "/lean " in top for top in tops):
                    overlap = True
                    (output / "concurrent-lean-processes.log").write_text("\n\n".join(tops))
            if not any(running):
                break
            if time.monotonic() - last_progress > 10:
                print(json_bytes({"local_only": True, "elapsed_seconds": round(time.monotonic() - started, 1),
                                  "containers_running": sum(running), "lean_overlap_seen": overlap}).decode(), flush=True)
                last_progress = time.monotonic()
            time.sleep(1)
        require(overlap, "no observed concurrent Lean processes; do not claim concurrency")
        summaries = []
        for session_id, info in zip(sessions, infos):
            destination = output / session_id
            destination.mkdir()
            (destination / "container.log").write_text(podman("logs", session_id).stdout)
            save_json(destination / "container-inspect.json", info)
            copy_output(session_id, destination)
            summaries.append(audit_session(output, session_id, info))
            saved = http(18760, "POST", f"/sessions/{session_id}/snapshot/v1", {"name": "local-final"})
            restored = http(18760, "POST", f"/sessions/{session_id}/restore/v1", {"name": "local-final"})
            require(restored["policy_version"] == summaries[-1]["policy_version"], "snapshot restore version mismatch")
            save_json(destination / "snapshot-restore.json", {"snapshot": saved, "restored": restored})
            require(http(18760, "DELETE", f"/sessions/{session_id}")["deleted"], "session cleanup route failed")
        jobs = []
        for request_path in sorted((output / "http-jobs").glob("*/request.json")):
            request = json.loads(request_path.read_bytes())
            result = json.loads(request_path.with_name("result.json").read_bytes())
            require(result["state"] == "done" and 200 <= result["status"] < 300, "HTTP job did not complete successfully")
            jobs.append({"request_id": request["request_id"], "method": request["method"], "path": request["path"],
                         "status": result["status"], "response_bytes": result["bytes"]})
        save_json(output / "http-jobs-summary.json", jobs)
        report.update(ok=True, sessions=summaries, concurrent_lean_processes=True, http_jobs=len(jobs),
                      backend="ToyBackend", actual_neural_parameter_updates=0,
                      preserved_containers=names, preserved_named_volumes=[name + "-out" for name in names])
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        cleanup_errors = []
        for name in names:
            try:
                info = inspect_owned(name, run_id)
                if info and info["State"]["Running"]:
                    podman("stop", "--time", "5", name)
                if info:
                    destination = output / name
                    destination.mkdir(exist_ok=True)
                    save_json(destination / "container-inspect-final.json", inspect_owned(name, run_id))
                    logs = podman("logs", name, check=False)
                    (destination / "container-final.log").write_text(logs.stdout + logs.stderr)
                    if not (destination / "out").exists():
                        copy_output(name, destination)
            except Exception as exc:
                cleanup_errors.append(f"{name}: {exc}")
        if server:
            server.shutdown()
            server.server_close()
        if thread:
            thread.join(timeout=5)
        for label, close in (("workers", transport.close), ("toy-server", lambda: stop_owned_process(b_process))):
            try:
                close()
            except Exception as exc:
                cleanup_errors.append(f"{label}: {exc}")
        report.update(wall_seconds=round(time.monotonic() - started, 3), cleanup_errors=cleanup_errors,
                      owned_services_stopped=(not thread or not thread.is_alive()) and
                      all(worker.poll() is not None for worker in transport.workers) and
                      (b_process is None or b_process.poll() is not None))
        if cleanup_errors:
            report["ok"] = False
        save_json(output / "report.json", report)
        print(json_bytes(report).decode(), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
