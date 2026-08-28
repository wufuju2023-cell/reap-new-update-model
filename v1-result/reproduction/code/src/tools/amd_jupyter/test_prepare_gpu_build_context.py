from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from containers.gpu.download_model import normalize_official_metadata
from tools.amd_jupyter import prepare_gpu_build_context as prepare


class PrepareGpuBuildContextTests(unittest.TestCase):
    """Tiny fabricated files validate packaging mechanics, never a real 7B image."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.model = self.root / "model"
        self.model.mkdir()
        actual_repo = Path(__file__).resolve().parents[2]
        for name in prepare.FIXED_SOURCES:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(actual_repo / name, path)
        (self.source / "gpu_runtime").mkdir()
        (self.source / "gpu_runtime/__init__.py").write_text("# fixture\n", encoding="utf-8")
        (self.source / "gpu_runtime/server.py").write_text("# fixture server\n", encoding="utf-8")
        self.shard = "model-00001-of-00001.safetensors"
        encoded = json.dumps({"weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
        encoded += b" " * (-len(encoded) % 8)
        (self.model / self.shard).write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"\x00\x00")
        (self.model / "config.json").write_text('{"hidden_size":3584,"model_type":"qwen2"}', encoding="utf-8")
        (self.model / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 2}, "weight_map": {"weight": self.shard}}), encoding="utf-8")
        (self.model / ".gitattributes").write_text("*.safetensors filter=lfs\n", encoding="utf-8")
        siblings = []
        for path in self.model.iterdir():
            content = path.read_bytes()
            entry = {"rfilename": path.name, "size": len(content),
                     "blobId": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()}
            if path.name == self.shard:
                entry["lfs"] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            siblings.append(entry)
        self.raw_manifest = normalize_official_metadata(
            {"id": prepare.REPO, "sha": prepare.REVISION, "private": False, "gated": False, "siblings": siblings},
            prepare.REPO, prepare.REVISION)
        self.manifest = self.root / "official-fixture-manifest.json"
        self.save_manifest()
        self.output = self.root / "context"

    def save_manifest(self):
        self.manifest.write_bytes(prepare.json_bytes(self.raw_manifest))
        self.checksum = hashlib.sha256(self.manifest.read_bytes()).hexdigest()

    def build_context(self):
        return prepare.prepare_context(self.source, self.model, self.manifest, self.checksum, self.output)

    def test_minimal_context_hashes_model_sources_and_excludes_secrets_and_A(self):
        sentinel = b"PRIVATE_SECRET_NOT_TO_BE_PACKAGED"
        for root, name in ((self.source, ".downloads/download-urls.json"), (self.source, ".git/config"),
                           (self.source, "cpu_runtime/private.py"), (self.source, "gpu_runtime/.secret.py"),
                           (self.model, ".cache/token"), (self.model, "download-urls.json"),
                           (self.model, "reap-model-lock.json")):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(sentinel)
        source_before = {path.relative_to(self.model).as_posix(): path.read_bytes()
                         for path in self.model.rglob("*") if path.is_file()}
        report = self.build_context()
        self.assertFalse(report["image_built"])
        self.assertFalse(report["GPU_verified"])
        self.assertEqual(report["model_files"], 4)
        metadata = json.loads((self.output / prepare.CONTEXT_MANIFEST).read_text())
        expected_source = set(prepare.FIXED_SOURCES) | {"gpu_runtime/__init__.py", "gpu_runtime/server.py"}
        self.assertEqual(set(metadata["source"]["files"]), expected_source)
        self.assertEqual(set(metadata["model"]["source_files"]), set(self.raw_manifest["files"]))
        self.assertEqual(metadata["model"]["tensor_count"], 1)
        for name, entry in metadata["files"].items():
            content = (self.output / name).read_bytes()
            self.assertEqual(entry["size"], len(content))
            self.assertEqual(entry["sha256"], hashlib.sha256(content).hexdigest())
            self.assertNotIn(sentinel, content)
        actual = {path.relative_to(self.output).as_posix() for path in self.output.rglob("*") if path.is_file()}
        self.assertEqual(actual, set(metadata["files"]) | {prepare.CONTEXT_MANIFEST})
        self.assertEqual(source_before, {path.relative_to(self.model).as_posix(): path.read_bytes()
                                        for path in self.model.rglob("*") if path.is_file()})
        ignore = (self.output / ".dockerignore").read_text().splitlines()
        self.assertEqual(ignore[0], "**")
        self.assertIn("!model/" + self.shard, ignore)
        self.assertNotIn("!model/**", ignore)
        self.assertNotIn("!model/download-urls.json", ignore)

    def test_existing_context_is_preserved(self):
        self.output.mkdir()
        original = self.output / "keep"
        original.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            self.build_context()
        self.assertEqual(list(self.output.iterdir()), [original])
        self.assertEqual(original.read_bytes(), b"keep")

    def test_admitted_replay_and_mathlib_loaders_import_from_isolated_minimal_context(self):
        self.build_context()
        code = "import sys; sys.path.insert(0, sys.argv[1]); from cpu_runtime.verified_trajectory import load_verified_dataset; from cpu_runtime.verified_dataset_store import install_verified_dataset; from cpu_runtime.mathlib_trajectory import load_mathlib_dataset"
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(self.output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_manifest_hash_and_revision_mismatches_fail_before_copy(self):
        self.checksum = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.build_context()
        self.assertFalse(self.output.exists())
        self.raw_manifest["revision"] = "a" * 40
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "fixed official REAL-Prover"):
            self.build_context()
        self.assertFalse(self.output.exists())

    def test_manifest_path_traversal_is_rejected(self):
        self.raw_manifest["files"]["../private"] = self.raw_manifest["files"]["config.json"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.build_context()
        self.assertFalse(self.output.exists())

    def test_model_symlink_is_rejected(self):
        path = self.model / self.shard
        original = path.read_bytes()
        path.unlink()
        external = self.root / "external.safetensors"
        external.write_bytes(original)
        try:
            path.symlink_to(external)
        except (OSError, NotImplementedError):
            self.skipTest("host does not permit symbolic links")
        with self.assertRaisesRegex(ValueError, "symlink input"):
            self.build_context()
        self.assertFalse(self.output.exists())

    def test_symlinked_source_model_and_output_ancestors_are_rejected(self):
        alias = self.root / "alias"
        try:
            alias.symlink_to(self.root, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("host does not permit symbolic links")
        for changed in ("source", "model", "manifest", "output"):
            with self.subTest(changed=changed):
                values = {"source": self.source, "model": self.model,
                          "manifest": self.manifest, "output": self.output}
                values[changed] = alias / values[changed].name
                with self.assertRaisesRegex(ValueError, "symlink input/output"):
                    prepare.prepare_context(values["source"], values["model"], values["manifest"],
                                            self.checksum, values["output"])
                self.assertFalse(self.output.exists())

    def test_same_size_model_corruption_leaves_no_success_manifest(self):
        path = self.model / self.shard
        content = path.read_bytes()
        path.write_bytes(content[:-1] + b"\x01")
        with self.assertRaisesRegex(ValueError, "LFS SHA256 mismatch"):
            self.build_context()
        self.assertFalse((self.output / prepare.CONTEXT_MANIFEST).exists())
        self.assertFalse((self.output / "model/reap-model-lock.json").exists())

    def test_malformed_tensor_header_is_rejected_after_matching_file_hash(self):
        path = self.model / self.shard
        content = b"\xff" * len(path.read_bytes())
        path.write_bytes(content)
        self.raw_manifest["files"][self.shard]["lfs_sha256"] = hashlib.sha256(content).hexdigest()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "safetensors header length"):
            self.build_context()
        self.assertFalse((self.output / prepare.CONTEXT_MANIFEST).exists())

    def test_model_copy_cannot_be_nested_inside_original_model_directory(self):
        self.output = self.model / "context"
        with self.assertRaisesRegex(ValueError, "must not be inside"):
            self.build_context()
        self.assertFalse(self.output.exists())

    def test_dependency_lock_rejects_torch_and_unhashed_pins(self):
        path = self.source / "containers/gpu/requirements-gpu-hashed.lock"
        baseline = path.read_bytes()
        for extra in (b"torch==2.11.0 --hash=sha256:" + b"a" * 64 + b"\n", b"requests==2.34.2\n"):
            with self.subTest(extra=extra):
                path.write_bytes(baseline + extra)
                with self.assertRaises(ValueError):
                    self.build_context()
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
