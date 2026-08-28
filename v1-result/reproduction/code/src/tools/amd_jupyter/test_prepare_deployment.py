from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.amd_jupyter import prepare_deployment as prepare


class PrepareDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        for name in (*prepare.FIXED_FILES, "gpu_runtime/__init__.py", "gpu_runtime/server.py"):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# local fixture\n", encoding="utf-8")
        (self.repo / "containers/gpu/requirements-gpu.lock").write_text(
            "transformers==4.57.1\npeft==0.17.1\naccelerate==1.10.1\nhuggingface-hub==0.34.4\nsafetensors==0.6.2\n",
            encoding="utf-8")
        manifest = {
            "schema_version": "reap.official-model-manifest.v1", "repo": prepare.MODEL_REPO,
            "revision": prepare.MODEL_REVISION,
            "source_api": f"https://huggingface.co/api/models/{prepare.MODEL_REPO}/revision/{prepare.MODEL_REVISION}?blobs=true",
            "files": {"config.json": {"size": 2, "git_blob_sha1": hashlib.sha1(b"blob 2\0{}").hexdigest()}},
        }
        self.manifest = self.root / "trusted-model-manifest.json"
        self.manifest.write_bytes(prepare.canonical_json(manifest))
        self.manifest_sha256 = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.output = self.root / "output"

    def build(self, **kwargs):
        return prepare.build_deployment(self.repo, self.output, self.manifest, self.manifest_sha256, **kwargs)

    def test_archive_extracts_to_only_safe_whitelist_with_matching_hashes(self):
        sentinel = b"PRIVATE_SENTINEL_MUST_NEVER_ENTER_ARCHIVE"
        for name in (".git/config", ".downloads/token.txt", ".env", "credentials.json",
                     "gpu_runtime/__pycache__/secret.pyc", "gpu_runtime/nested/secret.py",
                     "gpu_runtime/.secret.py", "containers/gpu/model.safetensors"):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(sentinel)
        report = self.build()
        self.assertFalse(report["ttt_verified"])
        metadata = json.loads((self.output / prepare.DEPLOYMENT_MANIFEST).read_text())
        self.assertFalse(metadata["model"]["weights_included"])
        self.assertIsNone(metadata["source"]["dirty"])
        expected = set(prepare.FIXED_FILES) | {"gpu_runtime/__init__.py", "gpu_runtime/server.py",
                                             prepare.MODEL_MANIFEST, "setup_venv.sh"}
        self.assertEqual(set(metadata["files"]), expected)
        extracted = self.root / "extracted"
        with tarfile.open(self.output / prepare.ARCHIVE_NAME, "r:gz") as archive:
            self.assertEqual(set(archive.getnames()), expected | {prepare.DEPLOYMENT_MANIFEST})
            for member in archive.getmembers():
                self.assertTrue(member.isfile())
                prepare.safe_relative(member.name)
                self.assertNotIn(sentinel, archive.extractfile(member).read())
            archive.extractall(extracted, filter="data")
        for name, entry in metadata["files"].items():
            content = (extracted / name).read_bytes()
            self.assertEqual(len(content), entry["size"])
            self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])
        self.assertEqual((extracted / prepare.DEPLOYMENT_MANIFEST).read_bytes(),
                         (self.output / prepare.DEPLOYMENT_MANIFEST).read_bytes())
        for line in (self.output / "manifest.sha256").read_text().splitlines():
            expected_hash, name = line.split("  ", 1)
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), expected_hash)

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            self.build()
        self.assertEqual(list(self.output.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_manifest_hash_failure_creates_no_output(self):
        self.manifest_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_explicit_bridge_missing_fails_instead_of_silently_omitting(self):
        with self.assertRaisesRegex(ValueError, "missing required deployment file"):
            self.build(include_remote_http_job=True)
        self.assertFalse(self.output.exists())
        (self.repo / prepare.REMOTE_HTTP_JOB).write_text("# bridge fixture\n", encoding="utf-8")
        self.build(include_remote_http_job=True)
        metadata = json.loads((self.output / prepare.DEPLOYMENT_MANIFEST).read_text())
        self.assertTrue(metadata["remote_http_job_included"])
        self.assertIn(prepare.REMOTE_HTTP_JOB, metadata["files"])

    def test_torch_or_unpinned_requirements_are_rejected(self):
        path = self.repo / "containers/gpu/requirements-gpu.lock"
        baseline = path.read_text()
        for extra in ("torch==2.11.0\n", "requests\n", "--index-url https://untrusted.invalid\n"):
            with self.subTest(extra=extra):
                path.write_text(baseline + extra, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.build()
                self.assertFalse(self.output.exists())

    def test_explicit_remote_lock_missing_fails_and_valid_lock_is_selected(self):
        with self.assertRaisesRegex(ValueError, "missing required deployment file"):
            self.build(include_remote_requirements=True)
        self.assertFalse(self.output.exists())
        baseline = (self.repo / "containers/gpu/requirements-gpu.lock").read_text()
        (self.repo / prepare.REMOTE_REQUIREMENTS).write_text(baseline + "packaging==25.0\n", encoding="utf-8")
        self.build(include_remote_requirements=True)
        metadata = json.loads((self.output / prepare.DEPLOYMENT_MANIFEST).read_text())
        self.assertEqual(metadata["setup"]["requirements"], prepare.REMOTE_REQUIREMENTS)
        with tarfile.open(self.output / prepare.ARCHIVE_NAME, "r:gz") as archive:
            setup = archive.extractfile("setup_venv.sh").read().decode()
        self.assertIn("requirements=" + prepare.REMOTE_REQUIREMENTS, setup)
        self.assertNotIn("__REQUIREMENTS_LOCK__", setup)

    def test_remote_lock_cannot_change_top_level_pins_or_replace_torch(self):
        baseline = (self.repo / "containers/gpu/requirements-gpu.lock").read_text()
        for changed in (baseline.replace("peft==0.17.1", "peft==0.18.0"), baseline + "torch==2.11.0\n"):
            with self.subTest(changed=changed):
                (self.repo / prepare.REMOTE_REQUIREMENTS).write_text(changed, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.build(include_remote_requirements=True)
                self.assertFalse(self.output.exists())

    def test_symlink_source_escape_is_rejected(self):
        path = self.repo / "gpu_runtime/escape.py"
        outside = self.root / "credential.txt"
        outside.write_text("private", encoding="utf-8")
        try:
            path.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("host does not permit symbolic links")
        with self.assertRaisesRegex(ValueError, "symlink deployment source"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_oversize_source_is_rejected_before_packaging(self):
        (self.repo / "gpu_runtime/oversize.py").write_bytes(b"x" * (prepare.MAX_FILE_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "oversize deployment source"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_path_traversal_member_names_are_rejected(self):
        for name in ("../escape.py", "/absolute.py", "x/../../escape.py", "x\\escape.py", "C:/escape.py", "x/./file.py"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                prepare.safe_relative(name)


if __name__ == "__main__":
    unittest.main()
