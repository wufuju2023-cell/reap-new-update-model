"""Tiny model fixtures test packaging only; no real weights, image build or GPU."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from tools.amd_jupyter.test_prepare_gpu_build_context import PrepareGpuBuildContextTests
spec = importlib.util.spec_from_file_location("delivery_context", HERE / "prepare_context.py")
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)


class DeliveryContextTests(unittest.TestCase):
    def setUp(self):
        self.fixture = PrepareGpuBuildContextTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def build(self):
        f = self.fixture
        return delivery.prepare(f.source, f.model, f.manifest, f.checksum, f.output, HERE / "train.Dockerfile")

    def test_recipe_replacement_preserves_all_model_evidence(self):
        report = self.build()
        f = self.fixture
        metadata = delivery.verify_context(f.output)
        original = metadata["delivery_recipe"]["original_context_manifest"]
        self.assertEqual(metadata["model"], original["model"])
        for name, value in original["files"].items():
            if name != delivery.RECIPE:
                self.assertEqual(metadata["files"][name], value)
        self.assertEqual((f.output / delivery.RECIPE).read_bytes(), (HERE / "train.Dockerfile").read_bytes())
        self.assertNotEqual(metadata["files"][delivery.RECIPE], original["files"][delivery.RECIPE])
        self.assertTrue(report["all_context_hashes_verified"])
        self.assertFalse(report["image_built"])
        self.assertFalse(report["GPU_verified"])

    def test_model_tamper_and_unlisted_file_are_rejected(self):
        self.build()
        f = self.fixture
        extra = f.output / "private-unlisted.txt"
        extra.write_text("fixture")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            delivery.verify_context(f.output)
        extra.unlink()
        model = f.output / "model" / f.shard
        data = model.read_bytes()
        model.write_bytes(data[:-1] + b"\x01")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            delivery.verify_context(f.output)

    def test_existing_context_not_overwritten(self):
        self.build()
        before = (self.fixture.output / delivery.original.CONTEXT_MANIFEST).read_bytes()
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            self.build()
        self.assertEqual(before, (self.fixture.output / delivery.original.CONTEXT_MANIFEST).read_bytes())

    def test_non_search_recipe_rejected_before_context_created(self):
        f = self.fixture
        with self.assertRaisesRegex(ValueError, "real-search"):
            delivery.prepare(f.source, f.model, f.manifest, f.checksum, f.output,
                             f.source / "containers/gpu/Containerfile")
        self.assertFalse(f.output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
