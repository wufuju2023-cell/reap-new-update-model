#!/usr/bin/env python3
"""Extract the actual online result and independently check it in offline Lean.

This is a reproduction helper, not a search or training entry point. It never
retries a container, rewrites an old output, or treats historical receipts as a
new proof check. Intended runtime: Linux/WSL with local rootless Podman.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import textwrap
import time


THEOREMS = {
    "01-CoupledOddSquare.lean": "MultiroundCandidates.CoupledOddSquare.coupled_odd_square",
    "02-AffineAccumulator.lean": "MultiroundCandidates.AffineAccumulator.affine_accumulator",
    "03-ScaledTriangular.lean": "MultiroundCandidates.ScaledTriangular.scaled_triangular",
    "04-DifferenceInvariant.lean": "MultiroundCandidates.DifferenceInvariant.difference_invariant",
    "05-CubeAccumulator.lean": "MultiroundCandidates.CubeAccumulator.cube_accumulator",
}
SCHEMA = "reap.reproduction.independent-lean.v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def prepare_proof(run_root, session_id, input_file, theorem):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", session_id), "invalid session ID")
    require(THEOREMS.get(input_file) == theorem, "input/declaration mapping mismatch")
    source_path = run_root / "inputs" / input_file
    session = run_root / "outputs" / session_id
    raw = source_path.read_bytes()
    metadata = read_json(session / "session.json")
    result = read_json(session / "result.json")
    online = read_json(session / "online-result.json")
    require(metadata.get("session_id") == session_id, "source session mismatch")
    require(metadata.get("theorem_sha256") == sha(raw), "input differs from searched theorem")
    require(result.get("session_id") == session_id and result.get("solved") is True,
            "no complete solved result")
    require(online.get("session_id") == session_id and online.get("root_verified") is True
            and online.get("returncode") == 0 and online.get("error") is None,
            "search did not end with a known verified root")
    proof = result.get("proof_script")
    require(isinstance(proof, str) and proof.strip(), "missing generated proof")
    marker = "  reapTrainingMCTS"
    text = raw.decode("utf-8")
    require(text.count(marker) == 1, "expected exactly one search entry")
    replacement = text.replace(marker, textwrap.indent(proof, "  "))
    require("reapTrainingMCTS" not in replacement, "proof still invokes search")
    encoded = (replacement + "\n#print axioms " + theorem + "\n").encode("utf-8")
    inputs = {"source": source_path, "session": session / "session.json",
              "result": session / "result.json", "online_result": session / "online-result.json"}
    binding = {name: {"path": str(path.relative_to(run_root)), "sha256": sha(path.read_bytes())}
               for name, path in inputs.items()}
    require(binding["source"]["sha256"] == sha(raw), "input changed while extracting proof")
    return encoded, online, binding


def command(args, timeout=600):
    return subprocess.run(args, capture_output=True, timeout=timeout, check=False)


def verify(run_root, session_id, input_file, theorem, cpu_image, *, runner=command):
    run_root = Path(run_root).resolve()
    require(re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", cpu_image), "use an inspected immutable image ID")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", session_id), "invalid session ID")
    destination = run_root / "proof-check" / session_id
    destination.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": SCHEMA, "passed": False, "session_id": session_id,
               "theorem": theorem, "requested_image": cpu_image}
    started = time.monotonic()
    try:
        proof, online, binding = prepare_proof(run_root, session_id, input_file, theorem)
        (destination / "proof.lean").write_bytes(proof)
        receipt.update(theorem_sha256=binding["source"]["sha256"], proof_sha256=sha(proof),
                       proof_file="proof.lean", input_files=binding,
                       online_status=online.get("status"), online_updates=online.get("optimizer_updates"))
        name = "reap-proof-" + session_id
        exists = runner(["podman", "container", "exists", name], timeout=30)
        require(exists.returncode == 1, "container already exists or Podman check failed")
        argv = ["podman", "run", "--name", name, "--pull", "never", "--network", "none",
                "--userns", "keep-id:uid=10001,gid=10001", "--user", "10001:10001",
                "--volume", str(destination) + ":/proof:ro", "--workdir", "/opt/reap-runtime",
                "--entrypoint", "bash", cpu_image, "-lc", "lake env lean /proof/proof.lean"]
        receipt["command"] = argv
        write_json(destination / "launch-intent.json", {"command": argv, "mutation_retry_allowed": False})
        result = runner(argv)
        (destination / "stdout.log").write_bytes(result.stdout)
        (destination / "stderr.log").write_bytes(result.stderr)
        receipt["lean_returncode"] = result.returncode
        inspected = runner(["podman", "inspect", name], timeout=30)
        (destination / "inspect.stdout.json").write_bytes(inspected.stdout)
        (destination / "inspect.stderr.log").write_bytes(inspected.stderr)
        require(inspected.returncode == 0, "could not confirm final container state")
        info, = json.loads(inspected.stdout)
        receipt.update(container_exit_code=info["State"]["ExitCode"],
                       container_running=info["State"]["Running"], image=info["Image"],
                       network=info["HostConfig"]["NetworkMode"])
        require(receipt["container_running"] is False, "container remains active; do not relaunch")
        require(receipt["lean_returncode"] == receipt["container_exit_code"] == 0,
                "independent Lean failed")
        require(receipt["network"] == "none", "proof check was not offline")
        require(receipt["image"].removeprefix("sha256:") == cpu_image.removeprefix("sha256:"),
                "actual image differs from requested ID")
        stdout = result.stdout.decode("utf-8", errors="strict")
        stderr = result.stderr.decode("utf-8", errors="strict")
        require("sorryAx" not in stdout + stderr, "proof depends on sorryAx")
        matches = re.findall(re.escape("'" + theorem + "' depends on axioms: ") + r"\[([^\]]*)\]", stdout)
        axiom_free = re.findall(r"^" + re.escape("'" + theorem + "' does not depend on any axioms")
                                + r"[ \t]*\r?$", stdout, re.MULTILINE)
        require(len(matches) + len(axiom_free) == 1, "missing or ambiguous declaration axiom check")
        axioms = [] if axiom_free else [item.strip() for item in matches[0].split(",") if item.strip()]
        require(set(axioms) <= {"propext", "Classical.choice", "Quot.sound"}, "unexpected proof axiom")
        require(all(sha((run_root / item["path"]).read_bytes()) == item["sha256"]
                    for item in binding.values()), "input changed during verification")
        receipt.update(passed=True, axioms=axioms)
    except BaseException as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            for name, data in (("stdout.log", exc.stdout), ("stderr.log", exc.stderr)):
                if data is not None and not (destination / name).exists():
                    (destination / name).write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        receipt["error"] = {"kind": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["files"] = {path.name: {"sha256": sha(path.read_bytes()), "size": path.stat().st_size}
                            for path in sorted(destination.iterdir()) if path.is_file()}
        write_json(destination / "receipt.json", receipt)
    # Only written after every real process / identity / axiom gate succeeded.
    accepted = dict(receipt)
    accepted["verification_receipt_sha256"] = sha((destination / "receipt.json").read_bytes())
    write_json(destination / "accepted.json", accepted)
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--input-file", required=True, choices=tuple(THEOREMS))
    parser.add_argument("--theorem", required=True)
    parser.add_argument("--cpu-image", required=True)
    args = parser.parse_args()
    result = verify(**vars(args))
    print(json.dumps({"passed": result["passed"], "session_id": result["session_id"],
                      "axioms": result["axioms"], "proof_sha256": result["proof_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
