import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cpu_runtime import verified_dataset_store as store
from cpu_runtime.verified_trajectory import (
    TrajectoryRejected, encode, export_verified, load_verified_dataset, plan_success_path, sha,
)
from tests import test_verified_trajectory as fixtures


class DatasetStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        args = fixtures.PublicationTests().setup_inputs(self.root)
        candidate = plan_success_path(*fixtures.fixture())
        trace = {"schema_version": "reap.verified-replay.trace.v1", "complete": True,
                 "theorem": "x", "root_return": -2, "rows": [{k: row[k] for k in
                 ("node_index", "state", "next_state", "tactic", "return")} for row in candidate["rows"]]}
        def run(*unused, **kwargs):
            (args["output"] / "trace.json").write_bytes(encode(trace))
            return subprocess.CompletedProcess([], 0, b"'x' depends on axioms: []\n", b"")
        with patch("cpu_runtime.verified_trajectory.subprocess.run", side_effect=run):
            self.dataset = export_verified(**args)
        self.source = args["output"]
        self.pin = sha((self.source / "dataset.json").read_bytes())
        self.registry = self.root / "registry"

    def install(self):
        return store.install_verified_dataset(self.source, self.registry, expected_sha256=self.pin)

    def directory_link(self, link, target):
        if os.name == "nt":
            result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode, 0, result.stdout.decode(errors="replace"))
        else:
            link.symlink_to(target, target_is_directory=True)

    def file_link(self, link, target):
        try:
            link.symlink_to(target)
        except OSError as error:
            if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                self.skipTest("Windows file symlink privilege unavailable; junction tests still run")
            raise

    def test_install_copies_only_named_files_and_detaches_source(self):
        (self.source / "extra-secret.txt").write_text("never copied")
        result = self.install()
        target = Path(result["path"])
        self.assertEqual(result["status"], "installed")
        self.assertEqual(set(p.name for p in target.iterdir()), store.BUNDLE_FILES)
        self.assertEqual(load_verified_dataset(target, expected_sha256=self.pin), self.dataset)
        (self.source / "trace.json").write_text("source modified later")
        self.assertEqual(load_verified_dataset(target, expected_sha256=self.pin), self.dataset)

    def test_existing_valid_directory_only_reverified_no_writes(self):
        self.install(); target = self.registry / self.pin
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in target.iterdir()}
        with patch.object(store._Directory, "write_new", side_effect=AssertionError("must not write")), \
             patch.object(store, "_publish_noreplace", side_effect=AssertionError("must not publish")):
            self.assertEqual(self.install()["status"], "existing")
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in target.iterdir()})

    def test_pin_required_before_any_destination_side_effect(self):
        for pin in (None, "", "A" * 64, "a" * 63, "../other", 4):
            with self.subTest(pin=pin), self.assertRaises(store.DatasetStoreError):
                store.install_verified_dataset(self.source, self.registry, expected_sha256=pin)
            self.assertFalse(self.registry.exists())

    def test_wrong_pin_or_corrupt_source_rejected_before_registry_creation(self):
        with self.assertRaises(TrajectoryRejected):
            store.install_verified_dataset(self.source, self.registry, expected_sha256="f" * 64)
        self.assertFalse(self.registry.exists())
        (self.source / "trace.json").write_text("corrupt")
        with self.assertRaises(TrajectoryRejected): self.install()
        self.assertFalse(self.registry.exists())

    def test_existing_empty_directory_is_not_replaced(self):
        target = self.registry / self.pin; target.mkdir(parents=True)
        with self.assertRaises(store.DatasetStoreError): self.install()
        self.assertEqual(list(target.iterdir()), [])
        self.assertEqual(list(self.registry.iterdir()), [target])

    def test_existing_corrupt_directory_is_never_repaired(self):
        self.install(); target = self.registry / self.pin
        (target / "trace.json").write_text("retain corrupt evidence")
        with self.assertRaises(TrajectoryRejected): self.install()
        self.assertEqual((target / "trace.json").read_text(), "retain corrupt evidence")

    def test_copy_error_leaves_only_unpublished_staging(self):
        real = store._Directory.write_new
        def fail(directory, name, data):
            if name == "plan.json": raise OSError("injected disk failure")
            return real(directory, name, data)
        with patch.object(store._Directory, "write_new", fail), self.assertRaises(OSError): self.install()
        self.assertFalse((self.registry / self.pin).exists())
        entries = list(self.registry.iterdir())
        self.assertEqual(len(entries), 1); self.assertTrue(entries[0].name.startswith(".staging-"))
        self.assertEqual(load_verified_dataset(self.source, expected_sha256=self.pin), self.dataset)

    def test_corrupted_copy_fails_second_full_validation(self):
        real = store._Directory.write_new
        def mutate(directory, name, data):
            return real(directory, name, data + (b" " if name == "trace.json" else b""))
        with patch.object(store._Directory, "write_new", mutate), self.assertRaises(TrajectoryRejected): self.install()
        self.assertFalse((self.registry / self.pin).exists())

    def test_atomic_publish_refuses_a_concurrent_empty_destination(self):
        real = store._publish_noreplace
        for empty in (True, False):
            with self.subTest(empty=empty):
                self.registry = self.root / ("registry-empty" if empty else "registry-nonempty")
                def collide(root, staging, digest):
                    (root.path / digest).mkdir()
                    if not empty:
                        (root.path / digest / "owned-by-other-process").write_text("keep")
                    return real(root, staging, digest)
                with patch.object(store, "_publish_noreplace", collide), self.assertRaises(OSError): self.install()
                self.assertEqual([p.name for p in (self.registry / self.pin).iterdir()],
                                 [] if empty else ["owned-by-other-process"])

    def test_publication_unknown_never_claims_success_or_blindly_retries(self):
        real = store._publish_noreplace
        def ambiguous(root, staging, digest):
            real(root, staging, digest)
            raise OSError("lost completion after atomic rename")
        with patch.object(store, "_publish_noreplace", ambiguous) as publication, self.assertRaises(OSError):
            self.install()
        self.assertEqual(load_verified_dataset(self.registry / self.pin, expected_sha256=self.pin), self.dataset)
        # Explicit caller recovery may only validate existing content.
        with patch.object(store, "_publish_noreplace", side_effect=AssertionError("retry")):
            self.assertEqual(self.install()["status"], "existing")

    def test_source_directory_junction_or_symlink_is_rejected(self):
        alias = self.root / "alias"; self.directory_link(alias, self.source)
        with self.assertRaises((OSError, store.DatasetStoreError)):
            store.install_verified_dataset(alias, self.registry, expected_sha256=self.pin)
        self.assertFalse(self.registry.exists())

    def test_source_ancestor_junction_or_symlink_is_rejected(self):
        parent = self.root / "alias-parent"; self.directory_link(parent, self.root)
        with self.assertRaises((OSError, store.DatasetStoreError)):
            load_verified_dataset(parent / self.source.name, expected_sha256=self.pin)

    def test_source_required_file_symlink_is_rejected_without_following(self):
        outside = self.root / "outside-trace.json"; shutil.copyfile(self.source / "trace.json", outside)
        (self.source / "trace.json").unlink(); self.file_link(self.source / "trace.json", outside)
        with self.assertRaises((OSError, store.DatasetStoreError)): self.install()
        self.assertFalse(self.registry.exists())

    def test_registry_root_link_does_not_write_into_target(self):
        outside = self.root / "outside"; outside.mkdir(); self.directory_link(self.registry, outside)
        with self.assertRaises((OSError, store.DatasetStoreError)): self.install()
        self.assertEqual(list(outside.iterdir()), [])

    def test_existing_digest_link_is_rejected_even_if_target_is_valid(self):
        self.registry.mkdir(); self.directory_link(self.registry / self.pin, self.source)
        with self.assertRaises((OSError, store.DatasetStoreError)): self.install()
        self.assertEqual(load_verified_dataset(self.source, expected_sha256=self.pin), self.dataset)

    @unittest.skipIf(os.name == "nt", "POSIX FIFO boundary; Windows reparse boundary tested separately")
    def test_required_fifo_is_rejected_without_blocking(self):
        (self.source / "trace.json").unlink(); os.mkfifo(self.source / "trace.json")
        with self.assertRaises(store.DatasetStoreError): load_verified_dataset(self.source, expected_sha256=self.pin)

    def test_parent_traversal_and_alternate_stream_names_rejected(self):
        with self.assertRaises(store.DatasetStoreError):
            store.install_verified_dataset(self.source, self.root / "x" / ".." / "registry", expected_sha256=self.pin)
        for name in ("../trace.json", "trace.json:stream", "a/b", "a\\b"):
            with self.assertRaises(store.DatasetStoreError): store._name(name)

    def test_nested_root_creation_is_safe_and_repeatable(self):
        self.registry = self.root / "nested" / "new" / "registry"
        self.assertEqual(self.install()["status"], "installed")
        self.assertEqual(self.install()["status"], "existing")


if __name__ == "__main__":
    unittest.main()
