#!/usr/bin/env python3
"""Offline fixed-token raw-logprob comparison; no generation or training.

Input JSON: {"prompt_ids": [1, 2], "attention_mask": [1, 1],
"candidates": [[3, 4], [5]]}. Candidates must already end at their first EOS.
Run in an otherwise idle process/device; timings include host scoring work and
GPU synchronization, exclude model load and separately reported warmups.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import time

from gpu_runtime.actor import GpuActor
from gpu_runtime.real_backend import (MAX_SCORING_CANDIDATES,
    MAX_SCORING_SEQUENCE_TOKENS, PolicyScoringConfig, RealProverBackend)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_nonnegative(value):
    number = float(value)
    require(math.isfinite(number) and number >= 0, "tolerance must be finite and nonnegative")
    return number


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    raw = Path(path).read_bytes()
    require(len(raw) <= 2 * 1024 * 1024, "JSON input exceeds 2 MiB")
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, "nonfinite JSON")), hashlib.sha256(raw).hexdigest()


def validate_inputs(data, vocab_size=None, eos_ids=()):
    require(isinstance(data, dict) and set(data) == {"prompt_ids", "attention_mask", "candidates"},
            "input requires exactly prompt_ids, attention_mask, candidates")
    prompt, mask, candidates = data["prompt_ids"], data["attention_mask"], data["candidates"]
    require(isinstance(prompt, list) and bool(prompt), "nonempty prompt required")
    require(isinstance(mask, list) and len(mask) == len(prompt)
            and all(type(x) is int and x in (0, 1) for x in mask)
            and mask[-1] == 1 and mask == sorted(mask), "binary left-padding mask required")
    require(isinstance(candidates, list) and 1 <= len(candidates) <= MAX_SCORING_CANDIDATES
            and all(isinstance(row, list) and row for row in candidates), "1..64 nonempty candidates required")
    require(len(prompt) + max(map(len, candidates)) <= MAX_SCORING_SEQUENCE_TOKENS,
            "prompt plus longest candidate exceeds 4096 tokens")
    for row in [prompt, *candidates]:
        require(all(type(token) is int and token >= 0 and (vocab_size is None or token < vocab_size)
                    for token in row), "invalid/out-of-vocabulary token ID")
    for row in candidates:
        require(not any(token in eos_ids for token in row[:-1]), "candidate contains tokens after first EOS")
    return data


def compare_rows(reference, actual, atol, rtol):
    require(len(reference) == len(actual), "candidate count mismatch")
    maximum, failures, tokens = 0.0, [], 0
    for row_index, (left, right) in enumerate(zip(reference, actual)):
        require(len(left) == len(right), "candidate token count mismatch")
        for token_index, (a, b) in enumerate(zip(left, right)):
            require(a["token"] == b["token"], "decoded token mismatch")
            x, y = float(a["logprob"]), float(b["logprob"])
            require(math.isfinite(x) and math.isfinite(y), "nonfinite logprob")
            delta = abs(x - y)
            maximum = max(maximum, delta)
            tokens += 1
            if delta > atol + rtol * abs(x):
                failures.append({"candidate": row_index, "token_index": token_index,
                                 "reference": x, "actual": y, "absolute_error": delta})
    return {"ok": not failures, "tokens": tokens, "maximum_absolute_error": maximum,
            "mismatch_count": len(failures), "first_mismatches": failures[:20]}


class CudaMeter:
    def __init__(self, torch, device):
        self.torch, self.device = torch, device

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def reset(self):
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def memory(self):
        cuda, device = self.torch.cuda, self.device
        return {"allocated_bytes": cuda.memory_allocated(device),
                "peak_allocated_bytes": cuda.max_memory_allocated(device),
                "reserved_bytes": cuda.memory_reserved(device),
                "peak_reserved_bytes": cuda.max_memory_reserved(device)}


def state_guard(backend, session_id):
    """Cheap mutation checks, explicitly not a fresh full frozen-base hash."""
    torch, session = backend.torch, backend._session(session_id)
    tensors = list(backend.model.named_parameters()) + list(backend.model.named_buffers())
    if session.value_head is not None:
        tensors += [("value." + name, value) for name, value in session.value_head.state_dict().items()]
    versions = [(name, tensor.data_ptr(), tensor._version, tuple(tensor.shape), str(tensor.dtype))
                for name, tensor in tensors]
    digest = hashlib.sha256()
    for name, tensor in tensors:
        if tensor.requires_grad or name.startswith("value."):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    rng = [torch.get_rng_state().clone(), session.cpu_rng_state.clone()]
    if backend.device.type == "cuda":
        rng += [torch.cuda.get_rng_state(backend.device).clone(), session.device_rng_state.clone()]
    return {"versions": versions, "trainable_sha256": digest.hexdigest(), "rng": rng,
            "optimizer_steps": session.optimizer_steps, "examples_seen": session.examples_seen,
            "optimizer_entries": len(session.optimizer.state) if session.optimizer is not None else 0}


def audit(backend, session_id, inputs, config, *, atol, rtol=0.0, repeats=3, warmups=1, meter=None):
    """Call through the actor. An injected CPU meter is only for mechanism tests."""
    atol, rtol = finite_nonnegative(atol), finite_nonnegative(rtol)
    require(type(repeats) is int and 1 <= repeats <= 10, "repeats must be in 1..10")
    require(type(warmups) is int and 1 <= warmups <= 3, "warmups must be in 1..3")
    require(isinstance(config, PolicyScoringConfig) and config.mode in ("candidate_chunks", "tokenwise_deferred"),
            "explicit comparison config required")
    comparison_mode = config.mode
    eos = backend.tokenizer.eos_token_id
    eos = eos if isinstance(eos, (list, tuple)) else [eos]
    validate_inputs(inputs, backend.model.config.vocab_size, eos)
    torch = backend.torch
    encoded = {key: torch.tensor([inputs[source]], device=backend.device, dtype=torch.long)
               for key, source in (("input_ids", "prompt_ids"), ("attention_mask", "attention_mask"))}
    candidates = inputs["candidates"]
    session = backend._activate(session_id)
    backend.model.eval()
    meter = meter or CudaMeter(torch, backend.device)
    before = state_guard(backend, session_id)
    records, warmup_records, comparisons = [], [], []
    first_reference = None

    def measure(mode):
        meter.synchronize()
        meter.reset()
        baseline = meter.memory()["allocated_bytes"]
        start = time.perf_counter()
        with backend._session_rng(session), torch.no_grad():
            if mode == "tokenwise":
                rows = [backend._score_generated_tokens(encoded, row) for row in candidates]
            elif mode == "tokenwise_deferred":
                rows = backend._score_generated_candidates_deferred(encoded, candidates)
            else:
                rows = backend._score_generated_candidates(encoded, candidates, config)
        meter.synchronize()
        elapsed = time.perf_counter() - start
        require(math.isfinite(elapsed) and elapsed > 0, "invalid elapsed time")
        return {"mode": mode, "host_synchronized_seconds": elapsed, "rows": rows,
                "baseline_allocated_bytes": baseline, **meter.memory()}

    for _ in range(warmups):
        for mode in ("tokenwise", comparison_mode):
            warmup_records.append(measure(mode))
    for iteration in range(repeats):
        order = ("tokenwise", comparison_mode) if iteration % 2 == 0 else (comparison_mode, "tokenwise")
        pair = {}
        for mode in order:
            record = measure(mode)
            record["iteration"] = iteration
            records.append(record)
            pair[mode] = record["rows"]
        first_reference = first_reference or pair["tokenwise"]
        comparisons.append({"iteration": iteration,
                            **compare_rows(pair["tokenwise"], pair[comparison_mode], atol, rtol)})
        comparisons.append({"iteration": iteration, "check": "reference_repeatability",
                            **compare_rows(first_reference, pair["tokenwise"], atol, rtol)})
    after = state_guard(backend, session_id)
    unchanged_rng = all(torch.equal(a, b) for a, b in zip(before.pop("rng"), after.pop("rng")))
    unchanged_state = before == after
    medians = {mode: statistics.median(record["host_synchronized_seconds"] for record in records
                                     if record["mode"] == mode) for mode in ("tokenwise", comparison_mode)}
    ratio = medians["tokenwise"] / medians[comparison_mode]
    return {"ok": unchanged_rng and unchanged_state and all(item["ok"] for item in comparisons),
            "session_id": session_id, "inputs": inputs, "config": asdict(config),
            "atol": atol, "rtol": rtol, "warmups_per_mode": warmups, "repeats": repeats,
            "warmups_excluded_from_summary": True, "warmups": warmup_records, "measurements": records,
            "comparisons": comparisons, "median_host_synchronized_seconds": medians,
            "comparison_mode": comparison_mode, "median_speed_ratio_tokenwise_over_comparison": ratio,
            **({"median_speed_ratio_tokenwise_over_chunks": ratio} if comparison_mode == "candidate_chunks" else {}),
            "unchanged_rng": unchanged_rng, "unchanged_state": unchanged_state,
            "state_before": before, "state_after": after,
            "state_check_scope": "parameter/buffer identity+version; trainable/value bytes; RNG; optimizer counters",
            "full_frozen_base_content_hash_checked": False,
            "no_generation": True, "no_training": True, "end_to_end_TTT_speedup_verified": False,
            "timing_scope": "same process; synchronized host scoring wall time incl decoding and transfers; no load/warmup",
            "memory_scope": "process allocator peaks; reserved cache retained between paths; not device-wide utilization"}


def audit_matrix(backend, session_id, inputs, *, atol, rtol=0.0, repeats=3, warmups=1, meter=None):
    """Separate helper logic, token chunking and candidate batching effects.

    Uses the same already loaded model and same session for all variants. The
    original 2x8 failure is retained separately; this does not silently replace
    its acceptance result or make a tolerance/production-config choice.
    """
    variants, cross_references, anchor = {}, [], None
    for label, batch, chunk in (("batch1_chunk1", 1, 1), ("batch1_chunk8", 1, 8), ("batch2_chunk1", 2, 1)):
        result = audit(backend, session_id, inputs, PolicyScoringConfig("candidate_chunks", batch, chunk),
                       atol=atol, rtol=rtol, repeats=repeats, warmups=warmups, meter=meter)
        variants[label] = result
        reference = next(row["rows"] for row in result["measurements"] if row["mode"] == "tokenwise")
        anchor = anchor if anchor is not None else reference
        cross_references.append({"variant": label, **compare_rows(anchor, reference, atol, rtol)})
        # Never continue into another variant after an unexpected mutation.
        if not result["unchanged_rng"] or not result["unchanged_state"]:
            break
    return {"ok": len(variants) == 3 and all(result["ok"] for result in variants.values())
                  and all(result["ok"] for result in cross_references),
            "diagnostic_matrix": True, "session_id": session_id, "variants": variants,
            "cross_variant_tokenwise_references": cross_references,
            "same_loaded_model_and_session": True,
            "production_mode_changed": False, "tolerance_changed": False,
            "no_generation": True, "no_training": True, "end_to_end_TTT_speedup_verified": False,
            "interpretation": {"batch1_chunk1": "helper-equivalence control at legacy forward shapes",
                "batch1_chunk8": "token-chunk/sequence-shape effect with one candidate",
                "batch2_chunk1": "candidate-batch shape effect without token chunking"}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--input-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--session-id", default="policy-scoring-probe")
    p.add_argument("--atol", type=finite_nonnegative, required=True)
    p.add_argument("--rtol", type=finite_nonnegative, default=0.0)
    p.add_argument("--candidate-batch-size", type=int, default=2)
    p.add_argument("--token-chunk-size", type=int, default=8)
    p.add_argument("--comparison-mode", choices=("candidate_chunks", "tokenwise_deferred"), default="candidate_chunks")
    p.add_argument("--repeats", type=int, choices=range(1, 11), default=3)
    p.add_argument("--warmups", type=int, choices=range(1, 4), default=1)
    p.add_argument("--diagnostic-matrix", action="store_true",
                   help="same-load 1x1, 1x8, 2x1 diagnostics; does not change the default single comparison")
    args = p.parse_args()
    require(not args.diagnostic_matrix or args.comparison_mode == "candidate_chunks",
            "diagnostic-matrix is specifically for candidate_chunks shapes")
    require(args.comparison_mode == "candidate_chunks" or (args.candidate_batch_size == 2 and args.token_chunk_size == 8),
            "chunk size options are only valid for candidate_chunks")
    config = PolicyScoringConfig(args.comparison_mode, args.candidate_batch_size, args.token_chunk_size)
    inputs, input_sha = read_json(args.input_json)
    validate_inputs(inputs)
    require(args.model_path.is_dir(), "existing local model required; never download")
    lock, lock_sha = read_json(args.model_path / "reap-model-lock.json")
    require(lock.get("schema_version") == "reap.model-lock.v2"
            and lock.get("repo") == "FrenzyMath/REAL-Prover"
            and lock.get("revision") == "fe76f68d9a88f342cb7b546307c20292fea9cced"
            and lock.get("hidden_size") == 3584, "pinned REAL-Prover model lock mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "reap.policy-scoring.gpu-probe.v1", "ok": False,
              "input_sha256": input_sha, "model_lock_sha256": lock_sha,
              "model_revision": lock["revision"], "model_lock_is_prior_verification_only": True}
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        require(bool(torch.version.hip) and torch.cuda.is_available(), "usable AMD ROCm/HIP required")
        backend = RealProverBackend(str(args.model_path))
        require(backend.device.type == "cuda" and backend.hidden_size == 3584, "real AMD 7B backend required")
        require(all(tensor.device == backend.device for tensor in backend.model.parameters()), "model device mismatch")
        report.update(gpu_name=torch.cuda.get_device_name(backend.device), hip=torch.version.hip,
                      torch_version=torch.__version__, device=str(backend.device),
                      dependencies={name: importlib.metadata.version(name) for name in ("transformers", "peft")},
                      model_execution={"attention_implementation": getattr(backend.model.config, "_attn_implementation", None),
                          "parameter_dtypes": sorted({str(tensor.dtype) for tensor in backend.model.parameters()}),
                          "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                          "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
                          "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled()})
        root = Path(__file__).resolve().parents[2]
        report["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in ("containers/gpu/smoke_policy_scoring.py", "gpu_runtime/real_backend.py",
                                               "gpu_runtime/actor.py", "containers/gpu/requirements-gpu.lock")}
        actor = GpuActor()
        try:
            actor.submit(lambda: backend.create_session(args.session_id))
            options = {"atol": args.atol, "rtol": args.rtol, "repeats": args.repeats, "warmups": args.warmups}
            report.update(actor.submit(lambda: audit_matrix(backend, args.session_id, inputs, **options)
                if args.diagnostic_matrix else audit(backend, args.session_id, inputs, config, **options)))
        finally:
            actor.close()
    except Exception as exc:
        report.update(ok=False, error_type=type(exc).__name__, error=str(exc))
    with (args.output_dir / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"ok": report["ok"], "report": str(args.output_dir / "report.json")}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
