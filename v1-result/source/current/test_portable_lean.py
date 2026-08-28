"""Offline preparation/command tests, with explicit mocked mutation failures."""
import importlib.util
import argparse
import ast
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("portable_lean", Path(__file__).with_name("portable_lean.py"))
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)
SOURCE = Path(os.environ.get("REAP_PORTABLE_TEST_SOURCE", str(M.HERE.parents[2]))).resolve()


@unittest.skipUnless((SOURCE / "cpu_runtime/replica_collector.py").is_file(), "set REAP_PORTABLE_TEST_SOURCE to extracted source")
class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.options = dict(source_root=SOURCE, image_id="d2070a64912f1a7c66ec9e4bb92e66004e8adaa58887e7be333f0bf8e3d5f3b7",
            endpoints=["http://127.0.0.1:18767", "http://127.0.0.1:18768"], deployments=["portable-one", "portable-two"],
            release="0103155583b260c7863ea58bbf2d3377444c272db0ea4f3a75dd49c40b65c98e", project_dir="/opt/reap-runtime")

    def prepare(self, name="bundle", **changes):
        return M.prepare(output=self.root / name, **{**self.options, **changes})

    def test_prepare_and_real_offline_check(self):
        result = self.prepare()
        self.assertFalse(result["new_GPU_or_Lean_run"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(M.execute(self.root / "bundle"), 0)
        self.assertFalse((self.root / "bundle/run-intent.json").exists())

    def test_new_run_ids_produce_new_session_ids(self):
        a, b = self.prepare("a"), self.prepare("b")
        self.assertNotEqual(a["run_sha256"], b["run_sha256"])
        self.assertTrue({x["attempt_id"] for x in a["attempts"]}.isdisjoint(x["attempt_id"] for x in b["attempts"]))
        self.assertEqual([x["polarity"] for x in a["attempts"]], ["disprove", "prove"])

    def test_existing_output_never_overwritten(self):
        self.prepare()
        before = (self.root / "bundle/manifest.json").read_bytes()
        with self.assertRaises(ValueError): self.prepare()
        self.assertEqual(before, (self.root / "bundle/manifest.json").read_bytes())

    def test_invalid_image_release_project_and_run_id(self):
        for change in ({"image_id": "f"*64}, {"release": "bad"}, {"project_dir": "/a/../b"}, {"run_id": "../bad"}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError): self.prepare(**change)
                self.assertFalse((self.root / "bundle").exists())

    def test_credentials_and_same_endpoint_rejected(self):
        for endpoints in (["http://u:p@host:123", "http://host:124"], ["http://host:123"]*2):
            with self.assertRaises(ValueError): self.prepare(endpoints=endpoints)
            self.assertFalse((self.root / "bundle").exists())

    def test_tampered_bundle_refuses_subprocess(self):
        self.prepare(); (self.root / "bundle/campaign.py").write_bytes(b"changed")
        with patch.object(M.subprocess, "call") as call:
            with self.assertRaises(ValueError): M.execute(self.root / "bundle")
            call.assert_not_called()

    def test_unknown_launch_retains_intent_and_refuses_second(self):
        self.prepare()
        with patch.object(M, "validate_run_platform"), patch.object(M.subprocess, "call", side_effect=OSError("unknown")) as call:
            with self.assertRaises(OSError): M.execute(self.root / "bundle", run=True)
            self.assertTrue((self.root / "bundle/run-intent.json").exists())
            self.assertFalse((self.root / "bundle/run-exit.json").exists())
            with self.assertRaises(ValueError): M.execute(self.root / "bundle", run=True)
            self.assertEqual(call.call_count, 1)

    def test_existing_native_output_refuses_start(self):
        self.prepare(); (self.root / "bundle/native").mkdir()
        with patch.object(M, "validate_run_platform"), patch.object(M.subprocess, "call") as call:
            with self.assertRaises(ValueError): M.execute(self.root / "bundle", run=True)
            call.assert_not_called()

    def test_container_commands_fixed_image_and_separate_networks(self):
        self.prepare(project_dir="/custom/lean-project")
        path = self.root / "bundle/campaign.py"
        spec = importlib.util.spec_from_file_location("portable_campaign_fixture", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        plan = json.loads((path.parent / "plan.json").read_bytes())
        for phase in ("preflight", "search", "verify"):
            cmd = module.command_for(plan, phase, "test")
            self.assertEqual(cmd[cmd.index("--network")+1], "host" if phase == "search" else "none")
            self.assertEqual(cmd[cmd.index("--workdir")+1], "/custom/lean-project")
            self.assertIn(plan["image_id"], cmd)
            self.assertIn("--pull=never", cmd)
            self.assertIn("PYTHONOPTIMIZE=", cmd)
            self.assertNotIn("--privileged", cmd)
            self.assertNotIn("--global", cmd)
            self.assertIn("GIT_CONFIG_VALUE_0=/custom/lean-project/.lake/packages/Qq", cmd)
        self.assertTrue(all("/home/" not in str(x) for x in module.command_for(plan, "verify", "test")))

    def rebind_fixture(self):
        result = self.prepare(image_id="f"*64, rebind_cpu_image=True)
        folder = self.root / "bundle"
        plan = json.loads((folder / "plan.json").read_bytes())
        spec = importlib.util.spec_from_file_location("binding_fixture", folder / "binding.py")
        binding = importlib.util.module_from_spec(spec); spec.loader.exec_module(binding)
        declarations = (folder / "inputs/declarations.lean").read_bytes()
        return result, folder, plan, binding, declarations

    def synthetic_receipts(self, plan, binding, declarations):
        records = {}
        for case in plan["cases"]:
            prepared = binding.prepared_for(plan, declarations, case)
            receipt = {"schema_version": "reap.closed-prop.preflight.v1", "returncode": 0,
                "proof_verified": False, "scope": "closed_propositions_and_search_options_only",
                "environment_sha256": plan["curriculum"]["environment_sha256"],
                "problem_sha256": prepared["problem"]["problem_sha256"],
                "attempted_prop_sha256": prepared["attempted"]["attempted_prop_sha256"],
                "execution_source_sha256": prepared["execution_source_sha256"],
                "preflight_source_sha256": prepared["preflight_source_sha256"],
                "command": ["synthetic-test-only", case["problem_id"]],
                "stdout_sha256": M.digest(b""), "stderr_sha256": M.digest(b"")}
            raw = M.canonical(receipt)
            records[case["problem_id"]] = (raw, {**case["prepared_descriptor"], "compilation_receipt_sha256": M.digest(raw)})
        return records

    def test_rebind_pending_has_no_scheduler_or_reused_receipt(self):
        result, folder, plan, binding, declarations = self.rebind_fixture()
        self.assertIsNone(result["run_sha256"])
        self.assertEqual(result["attempts"], [])
        old = json.loads((folder / "provenance/original-plan-template.json").read_bytes())
        self.assertNotEqual(plan["curriculum"]["environment_sha256"], old["curriculum"]["environment_sha256"])
        for problem, previous in zip(plan["curriculum"]["problems"], old["curriculum"]["problems"]):
            self.assertIsNone(problem["compilation_receipt_sha256"])
            self.assertNotEqual(problem["problem_sha256"], previous["problem_sha256"])
        for case in plan["cases"]:
            label = case["problem_id"]
            self.assertNotEqual((folder / f"inputs/{label}/attempt.lean").read_bytes(),
                                (folder / f"provenance/original-inputs/{label}/attempt.lean").read_bytes())
            self.assertFalse((folder / f"inputs/{label}/receipt.json").exists())
        self.assertEqual(M.execute(folder), 0)

    def test_rebind_binds_only_both_successful_new_receipts(self):
        _, folder, plan, binding, declarations = self.rebind_fixture()
        records = self.synthetic_receipts(plan, binding, declarations)
        bound = binding.finalize(plan, declarations, records)
        self.assertEqual(bound["cpu_rebinding"]["state"], "actual_preflights_bound")
        self.assertEqual(len(bound["expected_attempts"]), 2)
        changed = dict(records)
        label = next(iter(changed)); raw, descriptor = changed[label]
        receipt = json.loads(raw); receipt["command"] = ["other-synthetic-receipt"]
        new_raw = M.canonical(receipt)
        changed[label] = new_raw, {**descriptor, "compilation_receipt_sha256": M.digest(new_raw)}
        self.assertNotEqual(bound["run_sha256"], binding.finalize(plan, declarations, changed)["run_sha256"])
        self.assertIsNone(plan["run_sha256"])
        self.assertFalse((folder / "runtime-plan.json").exists())

    def test_rebind_failure_missing_or_old_receipt_blocks_before_scheduler(self):
        _, folder, plan, binding, declarations = self.rebind_fixture()
        records = self.synthetic_receipts(plan, binding, declarations)
        label = next(iter(records)); raw, descriptor = records[label]
        failed = json.loads(raw); failed["returncode"] = 1; bad = M.canonical(failed)
        variants = [{}, {**records, label: (bad, {**descriptor, "compilation_receipt_sha256": M.digest(bad)})}]
        old = (folder / f"provenance/original-inputs/{label}/receipt.json").read_bytes()
        variants.append({**records, label: (old, {**descriptor, "compilation_receipt_sha256": M.digest(old)})})
        for variant in variants:
            with patch.object(binding.mm.Matchmaker, "create") as create:
                with self.assertRaises(ValueError): binding.finalize(plan, declarations, variant)
                create.assert_not_called()

    def test_rebind_search_without_runtime_binding_refused(self):
        _, folder, _, _, _ = self.rebind_fixture()
        spec = importlib.util.spec_from_file_location("rebind_container_fixture", folder / "container_job.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.DATA = self.root / "empty-volume"
        with patch.object(module, "run_collector") as collector:
            with self.assertRaises(FileNotFoundError): module.check_bundle("search")
            collector.assert_not_called()

    def test_documented_two_server_cli_parses_without_loading_backend(self):
        tree = ast.parse((SOURCE / "gpu_runtime/server.py").read_text(encoding="utf8"))
        function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "build_parser")
        namespace = {"argparse": argparse, "math": math, "Path": Path, "__doc__": "offline parser check"}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "server-parser-only", "exec"), namespace)
        for port in (8761, 8762):
            args = namespace["build_parser"]().parse_args([
                "--host", "127.0.0.1", "--port", str(port), "--backend", "mixed-replay",
                "--model-path", "/existing/model", "--device", "cuda:0",
                "--verified-dataset-root", "/existing/replay", "--mathlib-dataset-root", "/existing/human",
                "--verified-max-distance", "8", "--max-post-update-kl", "100",
                "--policy-scoring", "tokenwise", "--max-resident-sessions", "1",
                "--learner-release-root", "/existing/learner-store", "--snapshot-root", f"/new/replica-{port}"])
            self.assertEqual(args.port, port)
            self.assertEqual(args.max_resident_sessions, 1)


if __name__ == "__main__":
    unittest.main()
