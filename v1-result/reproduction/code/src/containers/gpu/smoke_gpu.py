#!/usr/bin/env python3
"""Two-session REAL-Prover GPU numerics/isolation gate, NOT a Lean-feedback TTT run.

Three synthetic negative training events exercise real ROCm parameter updates.
The independent end-to-end TTT gate must still obtain actual Lean feedback.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import time

from gpu_runtime import GpuRuntime, RealProverBackend


# Exactly Reap.Tactic.Generator.mkPrompt with no premises. The GPU image must
# remain independent of cpu_runtime, so do not import the CPU prompt helper.
STATE = "⊢ True"
PROMPT = (
    "User: Please generate a tactic in lean4 to solve the state.\n"
    "Here're some theorems that may be helpful:\n"
    "\nSTATE:\n" + STATE + "\nTACTIC:\n\nAssistant:"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_tree(torch, value, label: str) -> None:
    if torch.is_tensor(value):
        require(bool(torch.isfinite(value).all().item()), f"non-finite tensor: {label}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(torch, item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_tree(torch, item, f"{label}.{index}")
    elif isinstance(value, float):
        require(math.isfinite(value), f"non-finite scalar: {label}")


def equal_tree(torch, left, right) -> bool:
    """Compare decoded state, never torch.save ZIP bytes or their metadata."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right)
                and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(torch, left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_tree(torch, a, b) for a, b in zip(left, right))
    return left == right


def tensor_leaves(torch, value, path=()) -> dict:
    if torch.is_tensor(value):
        return {path: value}
    leaves = {}
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (list, tuple)) else ()
    for key, item in items:
        leaves.update(tensor_leaves(torch, item, path + (key,)))
    return leaves


def tensor_delta(torch, before, after) -> dict:
    left, right = tensor_leaves(torch, before), tensor_leaves(torch, after)
    common = left.keys() & right.keys()
    return {
        "changed_tensors": sum(not equal_tree(torch, left[key], right[key]) for key in common),
        "added_tensors": len(right.keys() - left.keys()),
        "removed_tensors": len(left.keys() - right.keys()),
        "before_tensors": len(left), "after_tensors": len(right),
    }


def capture(runtime: GpuRuntime, backend: RealProverBackend, session_id: str) -> dict:
    raw = runtime.inspect_backend(session_id)
    require(raw.get("schema_version") == "reap.gpu.real-prover-backend.v1", "unexpected backend snapshot schema")
    require(raw.get("encoding") == "torch-save-base64", "unexpected backend snapshot encoding")
    state = backend.torch.load(
        io.BytesIO(base64.b64decode(raw["payload"], validate=True)),
        map_location="cpu", weights_only=True,
    )
    del raw
    require(isinstance(state, dict), "decoded backend snapshot is not a dictionary")
    for key in ("adapter", "value_head", "optimizer", "optimizer_steps", "examples_seen"):
        require(key in state, f"missing backend snapshot component: {key}")
    finite_tree(backend.torch, state, session_id)
    metadata = runtime.actor.submit(lambda: runtime.sessions.get(session_id).snapshot())
    return {"backend": state, "metadata": metadata}


def tensor_chunks(tensor, max_elements: int):
    """Bound each host copy even for a noncontiguous parameter/buffer."""
    if tensor.numel() <= max_elements:
        yield tensor.detach().to(device="cpu").contiguous()
    elif tensor.is_contiguous():
        flat = tensor.detach().view(-1)
        for offset in range(0, flat.numel(), max_elements):
            yield flat[offset:offset + max_elements].to(device="cpu").contiguous()
    else:
        # Split views without first flattening/copying a whole large GPU tensor.
        dimension = max(range(tensor.ndim), key=lambda index: tensor.shape[index])
        midpoint = tensor.shape[dimension] // 2
        yield from tensor_chunks(tensor.narrow(dimension, 0, midpoint), max_elements)
        yield from tensor_chunks(tensor.narrow(dimension, midpoint, tensor.shape[dimension] - midpoint), max_elements)


def base_fingerprint(backend: RealProverBackend, chunk_bytes: int) -> dict:
    """Hash every frozen non-LoRA parameter and buffer with bounded CPU copies."""
    torch = backend.torch
    digest = hashlib.sha256()
    counts = {"parameter_tensors": 0, "buffer_tensors": 0, "elements": 0, "bytes": 0}
    # This backend uses ordinary LoRA A/B only; any other trainable model
    # parameter fails the frozen-base assertion instead of being silently omitted.
    for kind, tensors in (("parameter", backend.model.named_parameters()), ("buffer", backend.model.named_buffers())):
        for name, tensor in tensors:
            if ".lora_A." in name or ".lora_B." in name:
                continue
            if kind == "parameter":
                require(not tensor.requires_grad, f"base parameter is trainable: {name}")
            header = json.dumps([kind, name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            max_elements = max(1, chunk_bytes // tensor.element_size())
            for chunk in tensor_chunks(tensor, max_elements):
                # Reinterpret bytes to support BF16 without a NumPy BF16 cast.
                digest.update(chunk.reshape(-1).view(torch.uint8).numpy().tobytes())
            counts[kind + "_tensors"] += 1
            counts["elements"] += tensor.numel()
            counts["bytes"] += tensor.numel() * tensor.element_size()
    require(counts["parameter_tensors"] > 0, "no frozen base parameters fingerprinted")
    return {"sha256": digest.hexdigest(), "coverage": "all_non_LoRA_model_parameters_and_buffers",
            "base_parameters_require_grad": False, "cpu_chunk_bytes": chunk_bytes, **counts}


def warm(runtime: GpuRuntime, session_id: str, request: dict) -> dict:
    policy, value = runtime.policy(session_id, request), runtime.value(session_id, request)
    choices = policy.get("choices", [])
    require(len(choices) == 1, f"unexpected policy choice count: {session_id}")
    content = choices[0]["message"]["content"]
    require(isinstance(content, str) and bool(content.strip()), f"empty policy candidate: {session_id}")
    tokens = choices[0].get("logprobs", {}).get("content", [])
    require(bool(tokens), f"missing token logprobs: {session_id}")
    require(all(math.isfinite(float(token["logprob"])) for token in tokens), f"non-finite logprobs: {session_id}")
    score = float(json.loads(value["choices"][0]["message"]["content"])["score"])
    require(math.isfinite(score) and -1 <= score <= 1, f"invalid value score: {session_id}")
    require(policy["policy_version"] == value["policy_version"], f"policy/value version mismatch: {session_id}")
    return {"candidate": content, "token_logprob_count": len(tokens), "value_score": score,
            "policy_version": policy["policy_version"]}


def learn(runtime: GpuRuntime, backend: RealProverBackend, session_id: str, version: int, event_id: str) -> dict:
    response = runtime.learn(session_id, expected_policy_version=version, event={
        "event_id": event_id, "state": STATE, "prompt": PROMPT,
        "tactic": "fail_if_success trivial", "verdict": "synthetic_rejected",
        "reward": -1.0, "terminal_verified": False,
        "source": "synthetic_gpu_numerics_gate_NOT_Lean_feedback",
    })
    require(response["applied"] and response["policy_version"] == version + 1, "learn did not advance exactly one version")
    finite_tree(backend.torch, response, "learn_response")
    return response


def require_update(torch, before: dict, after: dict) -> dict:
    deltas = {}
    for name in ("adapter", "value_head", "optimizer"):
        delta = tensor_delta(torch, before["backend"][name], after["backend"][name])
        require(delta["changed_tensors"] + delta["added_tensors"] > 0, f"learn changed no {name} tensors")
        require(delta["removed_tensors"] == 0, f"learn unexpectedly removed {name} tensors")
        if name != "optimizer":
            require(delta["added_tensors"] == 0 and delta["changed_tensors"] > 0,
                    f"learn did not modify the existing {name} tensors")
        deltas[name] = delta
    for name in ("optimizer_steps", "examples_seen"):
        require(after["backend"][name] == before["backend"][name] + 1, f"invalid {name} transition")
    require(after["metadata"]["policy_version"] == before["metadata"]["policy_version"] + 1, "invalid exported version transition")
    return deltas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/opt/models/REAL-Prover")
    parser.add_argument("--snapshot-root", type=Path, default=Path("/workspace/out/smoke-snapshots"))
    args = parser.parse_args()
    started = time.perf_counter()
    # Fail before loading 7B weights if this is not the required AMD/ROCm host.
    import torch
    require(bool(torch.version.hip), "ROCm/HIP build required; a CUDA-only GPU does not satisfy this gate")
    require(torch.cuda.is_available(), "no usable ROCm GPU")
    backend = RealProverBackend(args.model_path)
    session_a, session_b = "gpu-smoke-a", "gpu-smoke-b"
    with GpuRuntime(backend=backend, snapshot_root=args.snapshot_root) as runtime:
        runtime.create_session(session_a)
        runtime.create_session(session_b)
        equivalence = {}
        for session_id in (session_a, session_b):
            error = runtime.actor.submit(lambda sid=session_id: backend.initial_equivalence_error(sid, PROMPT))
            require(math.isfinite(error) and error <= 1e-6, f"fresh adapter is not finite/base-equivalent: {session_id}")
            equivalence[session_id] = error
        request = {"model": "REAL-Prover", "messages": [{"role": "user", "content": PROMPT}],
                   "n": 1, "temperature": 0.0, "max_tokens": 8, "logprobs": True}
        warm_results = {sid: warm(runtime, sid, request) for sid in (session_a, session_b)}
        require(all(item["policy_version"] == 0 for item in warm_results.values()), "fresh inference version is not zero")
        base_before = runtime.actor.submit(lambda: base_fingerprint(backend, 8 * 1024 * 1024))
        a_zero, b_zero = capture(runtime, backend, session_a), capture(runtime, backend, session_b)

        # Populate B's AdamW state so its isolation check covers real momentum
        # tensors, not merely an unchanged empty optimizer dictionary.
        b_learn = learn(runtime, backend, session_b, 0, "gpu-smoke-b-synthetic-0")
        b_baseline = capture(runtime, backend, session_b)
        b_delta = require_update(torch, b_zero, b_baseline)
        del b_zero
        require(equal_tree(torch, a_zero, capture(runtime, backend, session_a)), "B learn mutated A")

        first_learn = learn(runtime, backend, session_a, 0, "gpu-smoke-a-synthetic-0")
        a_snapshot = capture(runtime, backend, session_a)
        first_delta = require_update(torch, a_zero, a_snapshot)
        del a_zero
        require(equal_tree(torch, b_baseline, capture(runtime, backend, session_b)), "A learn mutated B")
        snapshot_name = "after-step-1-" + str(time.time_ns())
        snapshot = runtime.snapshot(session_a, snapshot_name)
        second_learn = learn(runtime, backend, session_a, 1, "gpu-smoke-a-synthetic-1")
        a_second = capture(runtime, backend, session_a)
        second_delta = require_update(torch, a_snapshot, a_second)
        del a_second
        require(equal_tree(torch, b_baseline, capture(runtime, backend, session_b)), "second A learn mutated B")
        restored = runtime.restore(session_a, snapshot_name)
        require(restored["policy_version"] == 1, "restore did not recover snapshot policy version")
        require(equal_tree(torch, a_snapshot, capture(runtime, backend, session_a)), "restore did not exactly recover A tensors/optimizer/metadata")
        require(equal_tree(torch, b_baseline, capture(runtime, backend, session_b)), "A restore mutated B")
        post_restore_warm = {sid: warm(runtime, sid, request) for sid in (session_a, session_b)}
        require(all(item["policy_version"] == 1 for item in post_restore_warm.values()), "restored inference version mismatch")
        for session_id, baseline in ((session_a, a_snapshot), (session_b, b_baseline)):
            require(equal_tree(torch, baseline, capture(runtime, backend, session_id)), "warm inference changed session state")
        base_after = runtime.actor.submit(lambda: base_fingerprint(backend, 8 * 1024 * 1024))
        require(base_before == base_after, "frozen base tensor fingerprint changed")

        report = {
            "ok": True, "gate": "real_ROCm_GPU_numerics_and_two_session_isolation",
            "ttt_verified": False, "lean_feedback_verified": False,
            "training_data": "three_synthetic_negative_events_NOT_Lean_feedback",
            "torch_hip": torch.version.hip, "device": torch.cuda.get_device_name(0),
            "hidden_size": backend.hidden_size, "coexisting_sessions": 2,
            "fresh_lora_max_logit_delta": equivalence, "warm": warm_results,
            "synthetic_b_learn": b_learn, "b_parameter_deltas": b_delta,
            "a_learns": [first_learn, second_learn], "a_parameter_deltas": [first_delta, second_delta],
            "isolation": {"b_learn_preserved_a_exactly": True, "a_learns_and_restore_preserved_b_exactly": True,
                          "scope": "decoded_adapter_value_optimizer_counters_and_full_session_metadata"},
            "snapshot": str(snapshot), "restore_exact": True, "restored_policy_version": restored["policy_version"],
            "post_restore_warm": post_restore_warm,
            "frozen_base_before": base_before, "frozen_base_after": base_after,
            "wall_seconds": time.perf_counter() - started,
        }
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
