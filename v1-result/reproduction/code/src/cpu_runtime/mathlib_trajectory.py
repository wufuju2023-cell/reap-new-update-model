"""Pinned human Mathlib tactic traces; no generated-actor events or relabeling.

The first profile deliberately admits only top-level, single-line tactics with
one live goal throughout. Unsupported branching is rejected, never given a
made-up value target. Run export inside the fixed offline Lean CPU environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time

from .verified_dataset_store import safe_directory
from .verified_trajectory import ALLOWED_AXIOMS, encode, sha, write_new

MATHLIB_COMMIT = "5352afccd6866369be9de43f5b7ec47203555f44"
SOURCE_FILE = "Mathlib/Logic/ExistsUnique.lean"
SOURCE_SHA256 = "2f5b7ae2ea952e0d924c90f47c874c21acb88e971a81df15eb6ac69b35a8eb3e"
DECLARATIONS = ("ExistsUnique.elim₂", "ExistsUnique.intro₂", "ExistsUnique.unique₂")
PROFILE = "mathlib_sft_linear_negative_remaining_actions_v1"
SCHEMA = "reap.mathlib-sft.dataset.v1"
MODULE = Path(__file__).resolve().parents[1] / "containers/cpu/mathlib-replay/MathlibReplay.lean"
CONTEXT = "namespace MathlibSFTGate\nvariable {α : Sort*}\n"
BASE_FILES = {"source.lean", "LICENSE", "source.json", "plan.json", "run.json", "MathlibReplay.lean", "extractor.py",
              "original.lean", "capture.lean", "replay.lean"}
PROCESS_FILES = {f"{stage}.{suffix}" for stage in ("original", "capture", "replay")
                 for suffix in ("stdout", "stderr", "receipt.json")}
BUNDLE_FILES = BASE_FILES | PROCESS_FILES | {"capture-trace.json", "replay-trace.json", "dataset.json"}


class MathlibTrajectoryRejected(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MathlibTrajectoryRejected(message)


def checked_axioms(stdout: str, theorem: str) -> list[str]:
    # The original declarations contain Unicode subscripts; do not relax the
    # generated-trajectory loader's separate identifier contract.
    require(theorem in {"MathlibSFTGate." + name for name in DECLARATIONS}, "unsupported replay theorem")
    matches = re.findall(r"^'" + re.escape(theorem) + r"' depends on axioms: \[([^\]]*)\]$", stdout, re.MULTILINE)
    matches.extend("" for _ in re.findall(r"^'" + re.escape(theorem) + r"' does not depend on any axioms$", stdout, re.MULTILINE))
    require(len(matches) == 1 and "sorryAx" not in stdout, "missing/ambiguous/unsafe axiom report")
    axioms = [value.strip() for value in matches[0].split(",") if value.strip()]
    require(set(axioms) <= ALLOWED_AXIOMS and len(set(axioms)) == len(axioms), "unsupported axiom")
    return axioms


def extract_declaration(source: bytes, declaration: str) -> dict:
    """Extract byte spans, preserving the original declaration/proof verbatim."""
    require(declaration in DECLARATIONS, "declaration outside the explicit first profile")
    prefix = ("theorem " + declaration + " ").encode()
    require(source.count(prefix) == 1, "missing or ambiguous theorem source")
    start = source.index(prefix)
    marker = source.index(b" := by\n", start)
    require(b"\n\n" not in source[start:marker], "unsupported declaration header")
    proof_start = marker + len(b" := by\n")
    end = source.find(b"\n\n", proof_start)
    require(end >= proof_start, "missing exact proof boundary")
    end += 1  # Include the last tactic newline, not the following blank line.
    proof = source[proof_start:end]
    actions = []
    offset = proof_start
    for raw in proof.splitlines(keepends=True):
        require(raw.endswith(b"\n") and raw.startswith(b"  ") and not raw.startswith(b"   "),
                "only two-space top-level single-line tactics are supported")
        tactic = raw[2:-1].decode("utf-8")
        require(tactic.startswith(("simp only ", "apply ", "exact ")) and
                not any(token in tactic for token in (";", "<;>", "sorry", "admit", "--", "/-")),
                "unsupported tactic boundary or unsafe proof")
        actions.append({"tactic": tactic, "byte_start": offset + 2, "byte_end": offset + len(raw) - 1,
                        "sha256": sha(raw[2:-1])})
        offset += len(raw)
    require(1 <= len(actions) <= 16 and offset == end, "unsupported action count or source span")
    return {"declaration": declaration, "byte_start": start, "byte_end": end,
            "proof_byte_start": proof_start, "proof_byte_end": end,
            "declaration_sha256": sha(source[start:end]), "proof_sha256": sha(proof),
            "actions": actions}


def prompt_for_state(state: str) -> str:
    # Matches pinned Reap mkPrompt(state, #[]); Lean emits the authoritative
    # prompt and this independent loader comparison catches accidental drift.
    return ("User: Please generate a tactic in lean4 to solve the state.\n"
            "Here're some theorems that may be helpful:\n\nSTATE:\n" + state + "\nTACTIC:\n\nAssistant:")


def validate_trace(trace: dict, selection: dict) -> None:
    actions = selection["actions"]
    require(trace.get("schema_version") == "reap.mathlib-sft.linear-trace.v1" and
            trace.get("complete") is True and trace.get("original_declaration") == selection["declaration"] and
            trace.get("theorem") == "MathlibSFTGate." + selection["declaration"], "trace identity mismatch")
    require(type(trace.get("root_return")) is int and trace["root_return"] == -len(actions),
            "unsupported root return")
    rows = trace.get("rows")
    require(isinstance(rows, list) and len(rows) == len(actions), "missing trace rows")
    for index, (row, action) in enumerate(zip(rows, actions)):
        require(set(row) == {"row", "state", "tactic", "next_state", "prompt", "return"}, "unexpected row fields")
        require(type(row["row"]) is int and row["row"] == index and row["tactic"] == action["tactic"],
                "trace tactic/source mismatch")
        require(isinstance(row["state"], list) and len(row["state"]) == 1 and
                isinstance(row["state"][0], str) and bool(row["state"][0]), "linear state required")
        require(row["prompt"] == prompt_for_state(row["state"][0]), "prompt does not match fixed empty-premise template")
        require(type(row["return"]) is int and row["return"] == index - len(actions), "wrong actual remaining-action return")
        if index + 1 < len(actions):
            require(row["next_state"] == rows[index + 1]["state"], "disconnected state transition")
        else:
            require(row["next_state"] == [], "nonterminal last action")


def program(source: bytes, selection: dict, module: bytes, output: Path, stage: str) -> bytes:
    declaration = selection["declaration"]
    header = source[selection["byte_start"]:selection["proof_byte_start"]].decode()
    if stage == "original":
        body = source[selection["proof_byte_start"]:selection["proof_byte_end"]].decode()
    else:
        expected = str(output / "capture-trace.json") if stage == "replay" else ""
        body = "  mathlibSFTReplay " + " ".join(json.dumps(x) for x in
            (str(output / "plan.json"), str(output / (stage + "-trace.json")), expected)) + "\n"
    candidate = "MathlibSFTGate." + declaration
    # The equality checks elaborated statement identity by type-checking both
    # proof terms. Its use of the original theorem is outside the replay proof.
    suffix = (f"end MathlibSFTGate\n#mathlibSFTNoSelf {declaration} {candidate}\n"
              f"example : @{declaration} = @{candidate} := by rfl\n#print axioms {candidate}\n")
    return ("import ReapRuntime\n" + module.decode() + "\n" + CONTEXT + header + body + suffix).encode()


def export_mathlib(*, mathlib_root: Path, declaration: str, output: Path, lean_project: Path,
                   replay_module: Path = MODULE, timeout: float = 180) -> dict:
    require(math.isfinite(timeout) and timeout > 0, "positive finite timeout required")
    commit = subprocess.check_output(["git", "-C", str(mathlib_root), "rev-parse", "HEAD"], text=True).strip()
    require(commit == MATHLIB_COMMIT, "Mathlib commit mismatch")
    with safe_directory(mathlib_root / "Mathlib/Logic") as directory:
        source = directory.read("ExistsUnique.lean")
    pinned = subprocess.check_output(["git", "-C", str(mathlib_root), "show", f"{commit}:{SOURCE_FILE}"])
    require(source == pinned and sha(source) == SOURCE_SHA256, "working source differs from pinned Mathlib bytes")
    with safe_directory(mathlib_root) as directory:
        license_text = directory.read("LICENSE")
    selection = extract_declaration(source, declaration)
    module = replay_module.read_bytes()
    output = output.absolute()
    require(not output.exists() and not any(p.is_symlink() for p in (output, *output.parents)),
            "output must be a new directory without links")
    with safe_directory(output, create=True):
        pass
    source_info = {"repository": "https://github.com/leanprover-community/mathlib4", "commit": commit,
                   "file": SOURCE_FILE, "source_sha256": sha(source), "source_git_blob_sha1":
                   hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest(),
                   "selection": selection, "ambient_context": CONTEXT,
                   "source_kind": "mathlib_sft", "generator_events": "none",
                   "extractor_sha256": sha(Path(__file__).read_bytes())}
    plan = {"schema_version": "reap.mathlib-sft.linear-plan.v1", "declaration": declaration,
            "actions": [row["tactic"] for row in selection["actions"]]}
    files = {"source.lean": source, "LICENSE": license_text, "source.json": encode(source_info),
             "plan.json": encode(plan), "run.json": encode({"output": str(output), "lean_project": str(lean_project)}),
             "MathlibReplay.lean": module, "extractor.py": Path(__file__).read_bytes()}
    files.update({stage + ".lean": program(source, selection, module, output, stage)
                  for stage in ("original", "capture", "replay")})
    for name, content in files.items():
        write_new(output / name, content)
    initial_hashes = {name: sha(content) for name, content in files.items()}
    for stage in ("original", "capture", "replay"):
        command = ["lake", "env", "lean", str(output / (stage + ".lean"))]
        started = time.perf_counter()
        result = subprocess.run(command, cwd=lean_project, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        elapsed = time.perf_counter() - started
        write_new(output / (stage + ".stdout"), result.stdout)
        write_new(output / (stage + ".stderr"), result.stderr)
        receipt = {"stage": stage, "command": command, "cwd": str(lean_project), "returncode": result.returncode,
                   "elapsed_seconds": elapsed, "stdout_sha256": sha(result.stdout), "stderr_sha256": sha(result.stderr),
                   "inputs_sha256": initial_hashes}
        write_new(output / (stage + ".receipt.json"), encode(receipt))
        require(result.returncode == 0, f"{stage} Lean failed; no dataset published")
        checked_axioms(result.stdout.decode(), "MathlibSFTGate." + declaration)
    trace = json.loads((output / "replay-trace.json").read_bytes())
    validate_trace(trace, selection)
    require(trace == json.loads((output / "capture-trace.json").read_bytes()), "capture/replay mismatch")
    require(all(sha((output / name).read_bytes()) == digest for name, digest in initial_hashes.items()),
            "source/driver changed during Lean execution")
    inputs = {name: sha((output / name).read_bytes()) for name in sorted(BUNDLE_FILES - {"dataset.json"})}
    dataset = {"schema_version": SCHEMA, "profile": PROFILE, "source_kind": "mathlib_sft",
               "source": source_info, "inputs_sha256": inputs, "rows": trace["rows"],
               "root_return": trace["root_return"], "axioms": checked_axioms(
                   (output / "replay.stdout").read_text(encoding="utf-8"), "MathlibSFTGate." + declaration),
               "action_unit": "one original top-level single-line tactic; linear-only; branch rejected",
               "prompt_origin": "pinned Reap mkPrompt(actual_state, empty premises)",
               "source_policy_version": None, "training_integration": "none"}
    encoded = encode(dataset)
    # Validate the complete prospective bundle before advertising dataset.json.
    bundle = {name: (output / name).read_bytes() for name in BUNDLE_FILES - {"dataset.json"}}
    _validate_bundle({**bundle, "dataset.json": encoded}, sha(encoded))
    with (output / "dataset.prepared.json").open("xb") as handle:
        handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
    os.link(output / "dataset.prepared.json", output / "dataset.json")
    return load_mathlib_dataset(output, expected_sha256=sha((output / "dataset.json").read_bytes()))


def load_mathlib_dataset(directory: Path, *, expected_sha256: str) -> dict:
    require(isinstance(expected_sha256, str) and bool(re.fullmatch(r"[a-f0-9]{64}", expected_sha256)), "canonical dataset pin required")
    with safe_directory(directory) as opened:
        bundle = {name: opened.read(name) for name in sorted(BUNDLE_FILES)}
    return _validate_bundle(bundle, expected_sha256)


def _validate_bundle(bundle: dict[str, bytes], expected_sha256: str) -> dict:
    require(sha(bundle["dataset.json"]) == expected_sha256, "dataset pin mismatch")
    dataset = json.loads(bundle["dataset.json"])
    require(isinstance(dataset, dict) and set(dataset) == {"schema_version", "profile", "source_kind", "source",
            "inputs_sha256", "rows", "root_return", "axioms", "action_unit", "prompt_origin",
            "source_policy_version", "training_integration"}, "unexpected dataset fields")
    require(dataset.get("schema_version") == SCHEMA and dataset.get("profile") == PROFILE and
            dataset.get("source_kind") == "mathlib_sft" and dataset.get("source_policy_version") is None,
            "invalid human SFT profile or invented actor version")
    hashes = dataset.get("inputs_sha256")
    require(isinstance(hashes, dict) and set(hashes) == BUNDLE_FILES - {"dataset.json"} and
            all(sha(bundle[name]) == digest for name, digest in hashes.items()), "incomplete or corrupt human evidence")
    source = json.loads(bundle["source.json"])
    require(isinstance(source, dict) and set(source) == {"repository", "commit", "file", "source_sha256",
            "source_git_blob_sha1", "selection", "ambient_context", "source_kind", "generator_events", "extractor_sha256"},
            "unexpected source fields")
    require(source == dataset.get("source") and source.get("commit") == MATHLIB_COMMIT and
            source.get("repository") == "https://github.com/leanprover-community/mathlib4" and
            source.get("file") == SOURCE_FILE and source.get("source_sha256") == SOURCE_SHA256 == sha(bundle["source.lean"]),
            "Mathlib source identity mismatch")
    require(source.get("source_git_blob_sha1") == hashlib.sha1(b"blob " + str(len(bundle["source.lean"])).encode() +
            b"\0" + bundle["source.lean"]).hexdigest() and source.get("extractor_sha256") == sha(bundle["extractor.py"]),
            "Mathlib blob or extractor identity mismatch")
    selection = extract_declaration(bundle["source.lean"], source["selection"]["declaration"])
    require(selection == source["selection"] and source.get("ambient_context") == CONTEXT and
            source.get("source_kind") == "mathlib_sft" and source.get("generator_events") == "none", "source span/context mismatch")
    plan = {"schema_version": "reap.mathlib-sft.linear-plan.v1", "declaration": selection["declaration"],
            "actions": [row["tactic"] for row in selection["actions"]]}
    require(json.loads(bundle["plan.json"]) == plan, "plan differs from original human proof")
    config = json.loads(bundle["run.json"])
    require(isinstance(config, dict) and set(config) == {"output", "lean_project"} and
            all(isinstance(config[k], str) and config[k].startswith("/") and
                ".." not in PurePosixPath(config[k]).parts for k in config), "fixed Linux replay paths required")
    # Rebuild the recorded Linux command on Windows without host separators.
    recorded_output = PurePosixPath(config["output"])
    for stage in ("original", "capture", "replay"):
        require(bundle[stage + ".lean"] == program(bundle["source.lean"], selection, bundle["MathlibReplay.lean"],
                recorded_output, stage), "driver differs from pinned source and replay module")
        receipt = json.loads(bundle[stage + ".receipt.json"])
        require(receipt.get("stage") == stage and type(receipt.get("returncode")) is int and receipt["returncode"] == 0 and
                receipt.get("command") == ["lake", "env", "lean", str(recorded_output / (stage + ".lean"))] and
                receipt.get("cwd") == config["lean_project"] and
                receipt.get("inputs_sha256") == {name: hashes[name] for name in BASE_FILES} and
                receipt.get("stdout_sha256") == hashes[stage + ".stdout"] and
                receipt.get("stderr_sha256") == hashes[stage + ".stderr"], "Lean process evidence mismatch")
        axioms = checked_axioms(bundle[stage + ".stdout"].decode(), "MathlibSFTGate." + selection["declaration"])
        if stage == "replay":
            require(axioms == dataset.get("axioms"), "axiom identity mismatch")
    trace = json.loads(bundle["replay-trace.json"])
    validate_trace(trace, selection)
    require(trace == json.loads(bundle["capture-trace.json"]) and dataset.get("rows") == trace["rows"] and
            dataset.get("root_return") == trace["root_return"], "published rows differ from independent trace")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mathlib-root", type=Path, required=True)
    parser.add_argument("--declaration", choices=DECLARATIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lean-project", type=Path, required=True)
    args = parser.parse_args()
    dataset = export_mathlib(**vars(args))
    print(json.dumps({"profile": PROFILE, "rows": len(dataset["rows"]), "root_return": dataset["root_return"]}))


if __name__ == "__main__":
    main()
