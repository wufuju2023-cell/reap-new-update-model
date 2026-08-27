#!/usr/bin/env python3
"""Remote-builder-only existing-model context with an audited delivery recipe."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.amd_jupyter import prepare_gpu_build_context as original

RECIPE = "containers/gpu/Containerfile"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def verify_context(output):
    output = Path(output)
    original.reject_link_components(output)
    metadata = json.loads((output / original.CONTEXT_MANIFEST).read_bytes())
    expected = set(metadata["files"]) | {original.CONTEXT_MANIFEST}
    actual = set()
    for path in output.rglob("*"):
        original.reject_link_components(path)
        if path.is_file():
            actual.add(path.relative_to(output).as_posix())
    original.require(actual == expected, "context contains missing or unexpected files")
    for name, item in metadata["files"].items():
        hashes = file_hash(original.local_regular(output, name))
        original.require(all(hashes[key] == item[key] for key in hashes), "context checksum mismatch: " + name)
    recipe = metadata["delivery_recipe"]
    original.require(recipe["sha256"] == metadata["files"][RECIPE]["sha256"], "recipe identity mismatch")
    original.require(metadata["model"]["manifest_sha256"] == file_hash(output / "model-manifest.json")["sha256"], "official manifest changed")
    return metadata


def prepare(repo_root, model_dir, manifest, manifest_sha256, output, recipe):
    repo_root, output, recipe = Path(repo_root), Path(output), Path(recipe)
    original.reject_link_components(recipe)
    original.require(recipe.is_file() and recipe.stat().st_size < 2 * 1024 * 1024, "recipe missing/oversize")
    recipe_bytes = recipe.read_bytes()
    text = recipe_bytes.decode("utf-8")
    original.require(text.count('CMD ["--backend", "real-search", "--gamma", "0.99"') == 2,
                     "both release targets must default to real-search gamma 0.99")
    report = original.prepare_context(repo_root, model_dir, manifest, manifest_sha256, output)
    manifest_path = output / original.CONTEXT_MANIFEST
    previous_bytes = manifest_path.read_bytes()
    metadata = json.loads(previous_bytes)
    old_models = copy.deepcopy(metadata["model"])
    old_model_files = {name: copy.deepcopy(item) for name, item in metadata["files"].items()
                       if name.startswith("model/") or name == "model-manifest.json"}
    old_recipe = metadata["files"][RECIPE]["sha256"]
    original.require(recipe.read_bytes() == recipe_bytes, "delivery recipe changed during preparation")
    destination = output / RECIPE
    temporary = destination.with_name(destination.name + ".delivery-new")
    original.write_new(temporary, recipe_bytes)
    os.replace(temporary, destination)  # Only this wrapper's newly created context.
    hashes = {"size": len(recipe_bytes), "sha256": hashlib.sha256(recipe_bytes).hexdigest()}
    metadata["files"][RECIPE] = {**hashes, "kind": "delivery_recipe"}
    metadata["source"]["files"][RECIPE] = hashes
    metadata["delivery_recipe"] = {
        "source": "v1-result/docker/train.Dockerfile", **hashes,
        "superseded_original_recipe_sha256": old_recipe,
        "original_context_manifest_sha256": hashlib.sha256(previous_bytes).hexdigest(),
        "original_context_manifest": json.loads(previous_bytes),
        "note": "Original source git identity is retained; only the context recipe is replaced. Model entries are unchanged."
    }
    original.require(metadata["model"] == old_models and all(metadata["files"][name] == item
        for name, item in old_model_files.items()), "model provenance must not change")
    replacement = manifest_path.with_name(manifest_path.name + ".delivery-new")
    original.write_new(replacement, original.json_bytes(metadata))
    os.replace(replacement, manifest_path)
    verify_context(output)
    return {**report, "context_manifest_sha256": file_hash(manifest_path)["sha256"],
            "recipe_sha256": hashes["sha256"], "all_context_hashes_verified": True,
            "model_manifest_and_entries_preserved": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=Path(__file__).with_name("train.Dockerfile"))
    parser.add_argument("--authorized-remote-builder", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.authorized_remote_builder:
        parser.error("requires a user-authorized remote builder; do not copy model weights to the local PC")
    if args.verify_only:
        verify_context(args.output)
        print(json.dumps({"verified": True, "image_built": False, "GPU_verified": False}))
        return 0
    if not all((args.model_dir, args.manifest, args.manifest_sha256)):
        parser.error("model-dir, manifest and manifest-sha256 are required")
    print(json.dumps(prepare(args.repo_root, args.model_dir, args.manifest,
                             args.manifest_sha256, args.output, args.recipe), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
