"""Train only the cached-feature 64-bin REAL7B value head."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path

BASE_FULL_ROWS = 80_000
FULL_EXTENSION_ROWS = 205_628
EXTENSION_FEATURE_SCHEMA = "new_value_head.feature-extension-manifest.v3"
EXTENSION_SHARD_SCHEMA = "new_value_head.feature-extension-shard.v3"


def load_features(torch, directory: Path, manifest_name: str):
    manifest_path = directory / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "new_value_head.feature-manifest.v2":
        raise ValueError(f"invalid feature manifest: {manifest_path}")
    features, labels, ids, roots, families, states, depths = [], [], [], [], [], [], []
    expected_start = 0
    for receipt in manifest.get("shards", []):
        path = directory / receipt["path"]
        parts = path.stem.split("-")
        if len(parts) != 3 or int(parts[1]) != expected_start or int(parts[2])-int(parts[1]) != receipt["rows"]:
            raise ValueError(f"gapped or malformed feature shard: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError(f"feature shard hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload.get("schema_version") != "new_value_head.feature-shard.v2"
                or payload.get("hidden_size") != 3584
                or payload.get("source_sha256") != manifest.get("source_sha256")
                or payload.get("dataset_manifest_sha256") != manifest.get("dataset_manifest_sha256")
                or payload.get("model_fingerprint_sha256") != manifest["model_fingerprint"]["sha256"]):
            raise ValueError(f"invalid feature shard: {path}")
        x, y, d = payload["features"], payload["labels"], payload["depths"]
        if (x.shape != (receipt["rows"], 3584) or x.dtype != torch.float16 or not bool(torch.isfinite(x).all())
                or y.shape != (receipt["rows"],) or d.shape != (receipt["rows"],)
                or bool(((y < 0) | (y > 63)).any()) or not torch.equal(y + 1, d)):
            raise ValueError(f"invalid feature tensor: {path}")
        vectors = (payload.get("sample_ids"), payload.get("root_ids"), payload.get("family_ids"),
                   payload.get("state_sha256s"))
        if any(not isinstance(values, list) or len(values) != receipt["rows"]
               or any(not isinstance(value, str) or not value for value in values) for values in vectors):
            raise ValueError(f"invalid feature identities: {path}")
        features.append(x); labels.append(y); depths.append(d)
        ids.extend(vectors[0]); roots.extend(vectors[1]); families.extend(vectors[2]); states.extend(vectors[3])
        expected_start = int(parts[2])
    if not features or expected_start != manifest.get("rows"):
        raise ValueError(f"no feature shards in {directory}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate sample ids in {directory}")
    return (torch.cat(features), torch.cat(labels), ids, roots, families, states, torch.cat(depths),
            manifest["model_fingerprint"]["sha256"], manifest["dataset_manifest_sha256"])


def load_extension_features(torch, directory: Path, base_feature_manifest: Path):
    """Load only [80k,205628), binding it to the immutable 80k feature manifest."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != EXTENSION_FEATURE_SCHEMA
            or manifest.get("range_start_inclusive") != BASE_FULL_ROWS
            or manifest.get("range_end_exclusive") != FULL_EXTENSION_ROWS
            or manifest.get("rows") != FULL_EXTENSION_ROWS-BASE_FULL_ROWS
            or manifest.get("rows_per_shard") != 2_000):
        raise ValueError(f"invalid full extension feature manifest: {manifest_path}")
    base_sha = hashlib.sha256(base_feature_manifest.read_bytes()).hexdigest()
    base_receipts = manifest.get("base_feature_manifests")
    if (not isinstance(base_receipts, list)
            or not any(item.get("sha256") == base_sha and item.get("rows") == BASE_FULL_ROWS
                       for item in base_receipts if isinstance(item, dict))):
        raise ValueError("extension does not bind the supplied frozen 80k feature manifest")
    features, labels, ids, roots, families, states, depths = [], [], [], [], [], [], []
    expected_start = BASE_FULL_ROWS
    for receipt in manifest.get("shards", []):
        path = directory / receipt["path"]
        if (receipt.get("global_start") != expected_start
                or receipt.get("global_end") != expected_start + receipt.get("rows", -1)
                or receipt["rows"] <= 0 or receipt["rows"] > 2_000):
            raise ValueError(f"gapped or malformed extension feature shard: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError(f"extension feature shard hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload.get("schema_version") != EXTENSION_SHARD_SCHEMA
                or payload.get("global_start") != expected_start
                or payload.get("global_end") != receipt["global_end"]
                or payload.get("source_sha256") != manifest.get("source_sha256")
                or payload.get("dataset_extension_manifest_sha256")
                    != manifest.get("dataset_extension_manifest_sha256")
                or payload.get("base_dataset_manifest_sha256")
                    != manifest.get("base_dataset_manifest_sha256")
                or payload.get("model_fingerprint_sha256") != manifest["model_fingerprint"]["sha256"]
                or payload.get("hidden_size") != 3584):
            raise ValueError(f"invalid extension feature shard: {path}")
        x, y, d = payload["features"], payload["labels"], payload["depths"]
        rows = receipt["rows"]
        if (x.shape != (rows, 3584) or x.dtype != torch.float16 or not bool(torch.isfinite(x).all())
                or y.shape != (rows,) or d.shape != (rows,) or bool(((y < 0) | (y > 63)).any())
                or not torch.equal(y+1, d)):
            raise ValueError(f"invalid extension feature tensor: {path}")
        vectors = (payload.get("sample_ids"), payload.get("root_ids"),
                   payload.get("family_ids"), payload.get("state_sha256s"))
        if any(not isinstance(values, list) or len(values) != rows
               or any(not isinstance(value, str) or not value for value in values) for values in vectors):
            raise ValueError(f"invalid extension feature identities: {path}")
        features.append(x); labels.append(y); depths.append(d)
        ids.extend(vectors[0]); roots.extend(vectors[1]); families.extend(vectors[2]); states.extend(vectors[3])
        expected_start = receipt["global_end"]
    if not features or expected_start != FULL_EXTENSION_ROWS:
        raise ValueError("extension features do not reach the fixed eligible-train end")
    if len(ids) != len(set(ids)) or len(states) != len(set(states)):
        raise ValueError("duplicate sample/state identities inside extension")
    provenance = {"extension_feature_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  "dataset_extension_manifest_sha256": manifest["dataset_extension_manifest_sha256"],
                  "base_feature_manifest_sha256": base_sha,
                  "base_dataset_manifest_sha256": manifest["base_dataset_manifest_sha256"]}
    return (torch.cat(features), torch.cat(labels), ids, roots, families, states, torch.cat(depths),
            manifest["model_fingerprint"]["sha256"], provenance)


def assert_base_extension_disjoint(base_ids, base_states, extra_ids, extra_states):
    if set(base_ids) & set(extra_ids) or set(base_states) & set(extra_states):
        raise ValueError("sample/state overlap between frozen 80k and extension")


def assert_split_isolation(train_vectors, validation_vectors, test_vectors):
    """Reject sample, root, theorem-family, or state leakage across all splits."""
    if not (len(train_vectors) == len(validation_vectors) == len(test_vectors) == 4):
        raise ValueError("split isolation requires sample/root/family/state vectors")
    for train, validation, test in zip(train_vectors, validation_vectors, test_vectors):
        if (set(train) & (set(validation) | set(test)) or set(validation) & set(test)):
            raise ValueError("sample/root/theorem-family/state leakage between splits")


def metrics(torch, logits, labels):
    probabilities = logits.softmax(-1)
    support = torch.arange(1, 65, device=logits.device, dtype=probabilities.dtype)
    expected = (probabilities * support).sum(-1)
    actual = labels.to(expected.dtype) + 1
    centered_x, centered_y = expected-expected.mean(), actual-actual.mean()
    denominator = (centered_x.square().sum()*centered_y.square().sum()).sqrt()
    correlation = float((centered_x*centered_y).sum()/denominator) if float(denominator) else 0.0
    argmax_depth = logits.argmax(-1).to(actual.dtype) + 1
    result = {"nll": float(torch.nn.functional.cross_entropy(logits, labels)),
            "accuracy": float((logits.argmax(-1) == labels).float().mean()),
            "expected_clipped_depth_mae": float((expected-actual).abs().mean()),
            "expected_clipped_depth_within_two": float(((expected-actual).abs() <= 2).float().mean()),
            "argmax_within_two_bins": float(((argmax_depth-actual).abs() <= 2).float().mean()),
            "pearson_expected_vs_clipped_depth": correlation}
    buckets = {}
    for low, high in ((1, 4), (5, 8), (9, 16), (17, 32), (33, 64)):
        mask = (actual >= low) & (actual <= high)
        count = int(mask.sum())
        buckets[f"{low}-{high}"] = {"count": count}
        if count:
            delta = (expected[mask]-actual[mask]).abs()
            buckets[f"{low}-{high}"].update({"expected_clipped_depth_mae": float(delta.mean()),
                "accuracy": float((logits[mask].argmax(-1) == labels[mask]).float().mean()),
                "expected_clipped_depth_within_two": float((delta <= 2).float().mean()),
                "argmax_within_two_bins": float(((argmax_depth[mask]-actual[mask]).abs() <= 2).float().mean())})
    result["depth_buckets"] = buckets
    return result


def within_root_order_accuracy(torch, logits, labels, roots):
    """Measure whether states from the same proof tree are ordered by known depth."""
    probabilities = logits.softmax(-1)
    support = torch.arange(1, 65, dtype=probabilities.dtype)
    expected = (probabilities.cpu() * support).sum(-1)
    groups = {}
    for index, root in enumerate(roots):
        groups.setdefault(root, []).append(index)
    correct = ties = pairs = 0
    for indices in groups.values():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1:]:
                actual_delta = int(labels[left]) - int(labels[right])
                if actual_delta == 0:
                    continue
                predicted_delta = float(expected[left] - expected[right])
                pairs += 1
                if predicted_delta == 0:
                    ties += 1
                elif (predicted_delta > 0) == (actual_delta > 0):
                    correct += 1
    return {"comparable_pairs": pairs, "correct": correct, "ties": ties,
            "accuracy_ties_half": (correct + 0.5 * ties) / pairs if pairs else None}


def evaluate(torch, head, x, y, roots, device, batch_size):
    rows = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            rows.append(head(x[start:start+batch_size].to(device=device, dtype=torch.float32)).cpu())
    logits = torch.cat(rows)
    return {**metrics(torch, logits, y), "within_root_order": within_root_order_accuracy(
        torch, logits, y, roots)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train-limit", type=int, choices=(20_000, 40_000, 60_000, 80_000))
    mode.add_argument("--train-extension-features", type=Path,
                      help="v3 [80000,205628) feature directory; trains an isolated full-scale artifact")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()
    import torch
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite training output: {args.output}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    full_scale = args.train_extension_features is not None
    base_manifest_name = "boundary-080000.json" if full_scale else f"boundary-{args.train_limit:06d}.json"
    base_manifest_path = args.train_features / base_manifest_name
    x_train, y_train, train_ids, train_roots, train_families, train_states, train_depths, train_fp, train_data = load_features(
        torch, args.train_features, base_manifest_name)
    extension_provenance = None
    if full_scale:
        extension = load_extension_features(torch, args.train_extension_features, base_manifest_path)
        x_extra, y_extra, extra_ids, extra_roots, extra_families, extra_states, extra_depths, extra_fp, extension_provenance = extension
        if extra_fp != train_fp or extension_provenance["base_dataset_manifest_sha256"] != train_data:
            raise SystemExit("base/extension model or dataset provenance differs")
        try:
            assert_base_extension_disjoint(train_ids, train_states, extra_ids, extra_states)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        x_train = torch.cat((x_train, x_extra)); y_train = torch.cat((y_train, y_extra))
        train_depths = torch.cat((train_depths, extra_depths))
        train_ids += extra_ids; train_roots += extra_roots
        train_families += extra_families; train_states += extra_states
    x_val, y_val, val_ids, val_roots, val_families, val_states, _, val_fp, val_data = load_features(
        torch, args.validation_features, "manifest.json")
    x_test, y_test, test_ids, test_roots, test_families, test_states, _, test_fp, test_data = load_features(
        torch, args.test_features, "manifest.json")
    expected_train_rows = FULL_EXTENSION_ROWS if full_scale else args.train_limit
    if len(x_train) != expected_train_rows:
        raise SystemExit(f"train manifests have {len(x_train)} rows, expected {expected_train_rows}")
    if len(x_val) != 8000 or len(x_test) != 8000:
        raise SystemExit("validation and test must each contain exactly 8000 rows")
    if len({train_fp, val_fp, test_fp}) != 1 or len({train_data, val_data, test_data}) != 1:
        raise SystemExit("model or prepared-dataset fingerprint differs between feature splits")
    try:
        assert_split_isolation((train_ids, train_roots, train_families, train_states),
            (val_ids, val_roots, val_families, val_states),
            (test_ids, test_roots, test_families, test_states))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    device = torch.device(args.device)
    head = torch.nn.Sequential(torch.nn.Linear(3584, 256), torch.nn.SiLU(), torch.nn.Linear(256, 64)).to(device)
    if sum(parameter.numel() for parameter in head.parameters()) != 934_208:
        raise RuntimeError("unexpected value-head parameter count")
    initial_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    initial_digest = hashlib.sha256(b"".join(value.numpy().tobytes() for value in initial_state.values())).hexdigest()
    initial_metrics = {"validation": evaluate(torch, head, x_val, y_val, val_roots, device, args.batch_size),
                       "test": evaluate(torch, head, x_test, y_test, test_roots, device, args.batch_size)}
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    best = None; best_nll = math.inf; stale = 0; history = []
    for epoch in range(args.epochs):
        head.train(); order = torch.randperm(len(x_train), generator=generator); loss_sum = 0.0
        for start in range(0, len(order), args.batch_size):
            index = order[start:start+args.batch_size]
            features = x_train[index].to(device=device, dtype=torch.float32)
            labels = y_train[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite value-head loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); loss_sum += float(loss)*len(index)
        validation = evaluate(torch, head, x_val, y_val, val_roots, device, args.batch_size)
        history.append({"epoch": epoch+1, "train_nll": loss_sum/len(x_train), "validation": validation})
        print(json.dumps(history[-1]), flush=True)
        if validation["nll"] < best_nll:
            best_nll = validation["nll"]
            best = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best is None:
        raise RuntimeError("training produced no checkpoint")
    head.load_state_dict(best)
    report = {"schema_version": ("new_value_head.full-training-report.v2" if full_scale
                                  else "new_value_head.training-report.v1"), "train_rows": len(x_train),
              "validation_rows": len(x_val), "test_rows": len(x_test), "seed": args.seed,
              "architecture": "linear-3584-silu-256-linear-64", "parameters": 934_208,
              "initial_head_sha256": initial_digest, "feature_model_fingerprint_sha256": train_fp,
              "dataset_manifest_sha256": train_data,
              "initial_random_head": initial_metrics,
              "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate,
                            "weight_decay": args.weight_decay},
              "objective": "unweighted-categorical-cross-entropy-depth-1-to-64",
              "validation": evaluate(torch, head, x_val, y_val, val_roots, device, args.batch_size),
              "test": evaluate(torch, head, x_test, y_test, test_roots, device, args.batch_size), "history": history}
    if full_scale:
        report["extension_provenance"] = extension_provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".tmp-{args.output.name}-", dir=args.output.parent))
    checkpoint = staging / "value-head.pt"
    initial_checkpoint = staging / "value-head-initial.pt"
    common = {"schema_version": ("new_value_head.categorical-head.v3" if full_scale
                                  else "new_value_head.categorical-head.v2"),
                "hidden_size": 3584, "classes": 64, "train_rows": len(x_train), "seed": args.seed,
                "initial_head_sha256": initial_digest,
                "feature_model_fingerprint_sha256": train_fp,
                "dataset_manifest_sha256": train_data,
                "decode": "softmax-temperature-1-expected-distance-1-to-64-bin64-saturates",
                "online_target": "clipped-distance-two-hot-categorical-cross-entropy",
                "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate,
                              "weight_decay": args.weight_decay}}
    if full_scale:
        common["profile"] = "full-eligible-train-205628-isolated-v1"
        common["extension_provenance"] = extension_provenance
    torch.save({**common, "artifact_role": "pretrained", "state_dict": best}, checkpoint)
    torch.save({**common, "artifact_role": "matched-random-initial", "state_dict": initial_state}, initial_checkpoint)
    report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report["initial_checkpoint_sha256"] = hashlib.sha256(initial_checkpoint.read_bytes()).hexdigest()
    report_path = staging / "report.json"
    report_payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)+"\n"
    report_path.write_text(report_payload, encoding="utf-8")
    (staging / "COMPLETE").write_text(report["checkpoint_sha256"] + "\n", encoding="utf-8")
    try:
        os.replace(staging, args.output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
