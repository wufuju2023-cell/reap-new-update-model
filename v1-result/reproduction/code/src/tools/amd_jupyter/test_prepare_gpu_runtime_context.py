from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from tools.amd_jupyter import prepare_gpu_runtime_context as prepare


class RuntimeContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo, self.wheels, self.out = (self.root / name for name in ("repo", "wheels", "out"))
        self.wheels.mkdir()
        actual = Path(__file__).resolve().parents[2]
        for name in prepare.FIXED:
            destination = self.repo / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(actual / name, destination)
        (self.repo / "gpu_runtime").mkdir()
        (self.repo / "gpu_runtime/__init__.py").write_text("# fixture\n")
        (self.repo / "gpu_runtime/server.py").write_text("from cpu_runtime.verified_dataset_store import safe_directory\n")
        lock = self.repo / prepare.FIXED[1]
        lines = []
        for name, pin in prepare.validate_hash_lock(lock.read_bytes()).items():
            content = ("tiny mechanism fixture " + name).encode()
            (self.wheels / (name + ".whl")).write_bytes(content)
            lines.append(f"{name}=={pin['version']} --hash=sha256:{hashlib.sha256(content).hexdigest()}")
        lock.write_text("\n".join(lines) + "\n")

    def run_prepare(self):
        return prepare.prepare_context(self.repo, self.wheels, self.out)

    def test_exact_hashes_allowlist_no_model_or_secrets(self):
        (self.repo / "secrets.txt").write_text("secret sentinel")
        (self.wheels / "token.json").write_text("secret sentinel")
        (self.repo / "gpu_runtime/.private.py").write_text("secret sentinel")
        result = self.run_prepare()
        self.assertEqual(result["model_files"], 0)
        self.assertFalse(result["image_built"])
        self.assertFalse(result["gpu_verified"])
        metadata = json.loads((self.out / prepare.MANIFEST).read_text())
        names = {p.relative_to(self.out).as_posix() for p in self.out.rglob("*") if p.is_file()}
        self.assertEqual(names, set(metadata["files"]) | {prepare.MANIFEST})
        self.assertEqual(sum(v["kind"] == "wheel" for v in metadata["files"].values()), 22)
        for name, info in metadata["files"].items():
            raw = (self.out / name).read_bytes()
            self.assertEqual(len(raw), info["size"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), info["sha256"])
            self.assertNotIn(b"secret sentinel", raw)
        self.assertEqual((self.out / ".containerignore").read_text().splitlines()[0], "**")

    def test_existing_output_never_reused(self):
        self.out.mkdir()
        (self.out / "keep").write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "exists"):
            self.run_prepare()
        self.assertEqual((self.out / "keep").read_bytes(), b"keep")

    def test_missing_corrupt_unexpected_and_duplicate_wheels_rejected(self):
        first = next(self.wheels.glob("*.whl"))
        original = first.read_bytes()
        for mode in ("missing", "corrupt", "unexpected", "duplicate"):
            with self.subTest(mode=mode):
                extra = self.wheels / "extra.whl"
                if mode == "missing":
                    first.unlink()
                elif mode == "corrupt":
                    first.write_bytes(b"corrupt")
                else:
                    extra.write_bytes(original if mode == "duplicate" else b"unexpected")
                with self.assertRaises(ValueError):
                    self.run_prepare()
                self.assertFalse(self.out.exists())
                first.write_bytes(original)
                if extra.exists():
                    extra.unlink()

    def test_undeclared_delayed_project_import_rejected(self):
        (self.repo / "gpu_runtime/server.py").write_text("def f():\n    from cpu_runtime.online_ttt import main\n")
        with self.assertRaisesRegex(ValueError, "undeclared project import"):
            self.run_prepare()
        self.assertFalse(self.out.exists())

    def test_model_copy_in_recipe_rejected(self):
        with (self.repo / prepare.FIXED[0]).open("a") as file:
            file.write("\nCOPY model /model\n")
        with self.assertRaisesRegex(ValueError, "must not include model"):
            self.run_prepare()

    def test_wheel_symlink_rejected(self):
        first = next(self.wheels.glob("*.whl"))
        external = self.root / "external"
        external.write_bytes(first.read_bytes())
        first.unlink()
        try:
            first.symlink_to(external)
        except OSError:
            self.skipTest("symlink unavailable")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_prepare()

    def test_actual_sources_have_closed_static_project_imports(self):
        actual = Path(__file__).resolve().parents[2]
        names = [p.relative_to(actual).as_posix() for p in (actual / "gpu_runtime").glob("*.py")]
        names += list(prepare.FIXED)
        prepare.check_imports({name: actual / name for name in names})


if __name__ == "__main__":
    unittest.main()
