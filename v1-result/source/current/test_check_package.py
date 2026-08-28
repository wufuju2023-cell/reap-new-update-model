"""Synthetic integrity fixtures; no Lean, GPU, network or existing package writes."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location("package_checker", Path(__file__).with_name("check_package.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write("README.md", b"# Fixture\n[page](docs/current/page.md)\n")
        self.write("docs/current/page.md", b"fixture\n")
        self.archive("source/current/manifest.json", "source/current/source.zip", {"module.py": b"pass\n"})
        for i, (manifest, archive) in enumerate(M.LEGACY):
            self.archive(manifest, archive, {"a.json": b'{"valid":true}'}, list_form=i == 3, history=i == 4)
        self.archive("evidence/current/stage/manifest.json", "evidence/current/stage/raw.zip", {"report.json": b'{"ok":true}'}, loose=True)

    def write(self, name, raw):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)

    def archive(self, manifest, archive, values, list_form=False, history=False, loose=False):
        output = io.BytesIO()
        if archive.endswith(".zip"):
            with zipfile.ZipFile(output, "w") as z:
                for name, raw in values.items():
                    z.writestr(name, raw)
        else:
            with tarfile.open(fileobj=output, mode="w:gz") as tar:
                for name, raw in values.items():
                    item = tarfile.TarInfo(name); item.size = len(raw)
                    tar.addfile(item, io.BytesIO(raw))
        raw = output.getvalue()
        self.write(archive, raw)
        files = {name: {"bytes": len(value), "sha256": M.digest(value)} for name, value in values.items()}
        meta = {"archive": {"bytes": len(raw), "sha256": M.digest(raw)}, "files": files}
        if list_form:
            meta["files"] = [{"path": k, **v} for k, v in files.items()]
        if history:
            meta.pop("archive"); meta.update(archive_bytes=len(raw), archive_sha256=M.digest(raw))
        if loose:
            meta["loose_files"] = {k: v["sha256"] for k, v in files.items()}
            for name, value in values.items():
                self.write(str(Path(manifest).parent / "files" / name), value)
        self.write(manifest, json.dumps(meta).encode())

    def test_refresh_and_default_read_only(self):
        result = M.check_package(self.root, refresh=True)
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(result["archives"]), 7)
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertTrue(M.check_package(self.root)["ok"])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_stale_manifest_and_no_implicit_refresh(self):
        self.assertTrue(M.check_package(self.root, refresh=True)["ok"])
        old = (self.root / "package-manifest.json").read_bytes()
        self.write("extra.txt", b"new")
        self.assertFalse(M.check_package(self.root)["ok"])
        self.assertEqual((self.root / "package-manifest.json").read_bytes(), old)

    def test_each_archive_integrity_is_checked(self):
        names = [x[1] for x in M.LEGACY] + ["source/current/source.zip", "evidence/current/stage/raw.zip"]
        for name in names:
            with self.subTest(name=name):
                p = self.root / name; raw = p.read_bytes(); p.write_bytes(raw + b"bad")
                self.assertFalse(M.check_package(self.root, refresh=True)["ok"])
                self.assertFalse((self.root / "package-manifest.json").exists())
                p.write_bytes(raw)

    def test_loose_copy_and_extra_files_rejected(self):
        p = self.root / "evidence/current/stage/files/report.json"
        raw = p.read_bytes(); p.write_bytes(b'{"ok":false}')
        self.assertFalse(M.check_package(self.root, refresh=True)["ok"])
        p.write_bytes(raw); self.write("evidence/current/stage/files/extra.txt", b"x")
        self.assertFalse(M.check_package(self.root, refresh=True)["ok"])

    def test_current_escape_rejected_historical_reported(self):
        self.write("docs/old.md", b"[project](../../missing-project.md)\n")
        result = M.check_package(self.root, refresh=True)
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(result["markdown"]["historical_project_links"]), 1)
        self.write("docs/current/page.md", b"[escape](../../../outside.md)\n")
        self.assertFalse(M.check_package(self.root, refresh=True)["ok"])

    def test_missing_and_dangerous_links(self):
        for target in ("missing.md", "javascript:alert", "file:///tmp/x", "//host/path", "/absolute/path"):
            with self.subTest(target=target):
                self.write("docs/current/page.md", ("[x](" + target + ")\n").encode())
                self.assertFalse(M.check_package(self.root, refresh=True)["ok"])

    def test_fenced_examples_not_links(self):
        self.write("docs/current/page.md", b'```text\n[x](missing.md)\n```\n')
        self.assertTrue(M.check_package(self.root, refresh=True)["ok"])

    def test_json_duplicate_nan_and_invalid_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e9999}', b'{'):
            self.write("test.json", raw)
            self.assertFalse(M.check_package(self.root, refresh=True)["ok"])

    def test_json_inside_archive_checked(self):
        self.archive("source/current/manifest.json", "source/current/source.zip", {"bad.json": b'{bad'})
        self.assertFalse(M.check_package(self.root, refresh=True)["ok"])

    def test_unsafe_archive_members_and_pyc(self):
        for name in ("../outside", "module.pyc", "dir/__pycache__/x", "AUX.txt"):
            self.archive("source/current/manifest.json", "source/current/source.zip", {name: b"x"})
            self.assertFalse(M.check_package(self.root, refresh=True)["ok"])

    def test_manifest_duplicates_and_aliases(self):
        row = {"path": "a", "bytes": 1, "sha256": "0" * 64}
        with self.assertRaises(ValueError): M.entries([row, row])
        with self.assertRaises(ValueError): M.entries({"a": row, "A": row})
        with self.assertRaises(ValueError): M.entries({"a": row, "a/b": row})

    def test_tar_link_refused_even_with_matching_manifest(self):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as tar:
            item = tarfile.TarInfo("a"); item.type = tarfile.SYMTYPE; item.linkname = "../outside"
            tar.addfile(item)
        raw = output.getvalue()
        manifest = {"archive": {"bytes": len(raw), "sha256": M.digest(raw)},
                    "files": {"a": {"bytes": 0, "sha256": M.digest(b"")}}}
        with self.assertRaises(ValueError): M.archive_values(manifest, raw, "tar")

    def test_symlink_rejected(self):
        link = self.root / "linked"
        try:
            link.symlink_to(self.root / "README.md")
        except OSError:
            self.skipTest("symlink creation unavailable")
        with self.assertRaises(ValueError): M.check_package(self.root, refresh=True)

    def test_loose_pyc_rejected(self):
        self.write("test.pyc", b"compiled")
        with self.assertRaises(ValueError): M.check_package(self.root, refresh=True)

    def test_bash_status_and_no_execution_default(self):
        self.write("docs/current/page.md", b"```bash\necho hi\n```\n")
        with patch.object(M.subprocess, "run") as run:
            result = M.check_package(self.root, refresh=True)
            run.assert_not_called()
            self.assertEqual(result["markdown"]["bash"]["status"], "not_requested")
        with patch.object(M.shutil, "which", return_value=None):
            result = M.check_package(self.root, refresh=True, bash=True)
            self.assertEqual(result["markdown"]["bash"]["status"], "unavailable")

    def test_refresh_fail_preserves_previous_manifest(self):
        self.assertTrue(M.check_package(self.root, refresh=True)["ok"])
        before = (self.root / "package-manifest.json").read_bytes()
        self.write("bad.json", b"{")
        result = M.check_package(self.root, refresh=True)
        self.assertFalse(result["ok"])
        self.assertFalse(result["refreshed"])
        self.assertEqual((self.root / "package-manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
