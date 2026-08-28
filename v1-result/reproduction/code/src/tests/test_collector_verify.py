"""Utility mechanism tests. Lean subprocesses are mocked, not proof evidence."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cpu_runtime.collector_verify import verify_collected
from cpu_runtime.verified_collector import MIXED_OBJECTIVE, run_collector
from cpu_runtime.verified_dataset_store import BUNDLE_FILES, install_verified_dataset
from cpu_runtime.verified_trajectory import encode, load_verified_dataset, plan_success_path, sha
from tests.test_verified_collector import PIN, created, frames, mixed_created


class CollectorVerifyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/"input.lean"
        self.source.write_bytes(b"import ReapRuntime\ntheorem sample : True := by\n  reapTrainingMCTS\n")
        self.module = Path(__file__).resolve().parents[1]/"containers/cpu/verified-replay/VerifiedReplay.lean"
        self.client = Mock()
        self.make_candidate()

    def make_candidate(self, mixed=False, output="collected"):
        self.client.create_session.side_effect = lambda sid, **kw: (mixed_created if mixed else created)(sid, kw["theorem_id"])
        def start(*args, env, **kwargs):
            folder = Path(env["REAP_SESSION_DIR"])
            tree, events = frames()
            (folder/"observer.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
            (folder/"raw_tree.json").write_bytes(encode(tree))
            (folder/"result.json").write_bytes(encode({"schema_version": "reap.training.result.v1",
                "session_id": "test", "solved": True, "status": "solved", "error": None,
                "proof_script": "step\nfinish"}))
            return SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)
        options = {"learner_objective": MIXED_OBJECTIVE} if mixed else {}
        with patch("cpu_runtime.verified_collector.subprocess.Popen", side_effect=start):
            result = run_collector(session_id="test", project_dir=self.root, theorem_file="input.lean",
                theorem="sample", output_root=self.root/output, gpu_base_url="http://unused",
                model_release_sha256=PIN, puct_value_gamma=.9, client=self.client, command=["fixture"], **options)
        self.assertEqual(result["status"], "solved_pending_independent_verification", result)
        self.session = self.root/output/"test"
        self.options = dict(session_dir=self.session, output_dir=self.root/(output+"-verified"),
            project_dir=self.root, dataset_store=self.root/(output+"-registry"),
            expected_release_sha256=PIN, expected_execution_source_sha256=sha(self.source.read_bytes()),
            image_sha256="a"*64, network="none", expected_theorem="sample", replay_module=self.module)

    def replace_json(self, name, edit):
        path = self.session/name
        value = json.loads(path.read_bytes()); edit(value); path.write_bytes(encode(value))

    def lean(self, command, **kwargs):
        self.assertEqual(command[:3], ["lake", "env", "lean"])
        self.assertEqual(kwargs["cwd"], self.root)
        path = Path(command[-1])
        if path.name == "replay.lean":
            tree = json.loads((path.parent/"raw_tree.json").read_bytes())
            session = json.loads((path.parent/"session.json").read_bytes())
            events = [json.loads(line) for line in (path.parent/"observer.jsonl").read_bytes().splitlines()]
            candidate = plan_success_path(tree, events, session)
            trace = {"schema_version": "reap.verified-replay.trace.v1", "complete": True,
                "theorem": "sample", "root_return": candidate["root_return"],
                "rows": [{k: row[k] for k in ("node_index", "state", "next_state", "tactic", "return")}
                         for row in candidate["rows"]]}
            (path.parent/"trace.json").write_bytes(encode(trace))
        else:
            self.assertEqual(path.name, "proof.lean")
        return subprocess.CompletedProcess(command, 0, b"'sample' depends on axioms: [propext]\n", b"")

    def call(self, lean=None, **changes):
        with patch("cpu_runtime.collector_verify.subprocess.run", side_effect=lean or self.lean) as process:
            result = verify_collected(**{**self.options, **changes})
        return result, process

    def test_mock_lean_two_processes_strict_real_export_and_install(self):
        original = {p.name: p.read_bytes() for p in self.session.iterdir() if p.is_file()}
        result, run = self.call()
        self.assertEqual(run.call_count, 2)
        self.assertEqual([Path(c.args[0][-1]).name for c in run.call_args_list], ["proof.lean", "replay.lean"])
        self.assertTrue(result["verified"])
        self.assertEqual(result["root_return"], -2)
        self.assertEqual(result["rows"], 2)
        self.assertFalse(result["new_search"])
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertEqual([c[0] for c in self.client.method_calls], ["create_session"])
        for label in ("proof", "proof_receipt", "dataset_receipt", "verification_receipt"):
            self.assertEqual(sha(Path(result[label+"_path"]).read_bytes()), result[label+"_sha256"])
        destination = Path(result["dataset_path"])
        self.assertEqual({p.name for p in destination.iterdir()}, BUNDLE_FILES)
        dataset = load_verified_dataset(destination, expected_sha256=result["dataset_sha256"])
        self.assertEqual(dataset["session_id"], "test")
        self.assertEqual({p.name: p.read_bytes() for p in self.session.iterdir() if p.is_file()}, original)

    def test_explicit_mixed_profile_keeps_original_actor_provenance(self):
        self.make_candidate(mixed=True, output="mixed")
        result, _ = self.call()
        session = json.loads((Path(result["dataset_path"])/"session.json").read_bytes())
        self.assertEqual(session["learner_objective"], MIXED_OBJECTIVE)
        self.assertEqual(session["lineage"]["model_release_sha256"], PIN)
        self.assertEqual(session["policy_version"], 0)

    def test_wrong_pins_theorem_or_terminal_fields_rejected_before_lean(self):
        cases = [({"expected_release_sha256": "b"*64}, None),
                 ({"expected_execution_source_sha256": "b"*64}, None),
                 ({"expected_theorem": "other"}, None)]
        for key, value in (("status", "exhausted"), ("optimizer_updates", 1),
                           ("optimizer_updates", False), ("policy_version", 1), ("policy_version", False),
                           ("independent_verified", True), ("returncode", 1)):
            cases.append(({}, (key, value)))
        report = (self.session/"collector-result.json").read_bytes()
        for index, (options, edit) in enumerate(cases):
            with self.subTest(index=index):
                (self.session/"collector-result.json").write_bytes(report)
                if edit: self.replace_json("collector-result.json", lambda v: v.update({edit[0]: edit[1]}))
                with patch("cpu_runtime.collector_verify.subprocess.run") as run:
                    with self.assertRaises(ValueError):
                        verify_collected(**{**self.options, **options, "output_dir": self.root/f"reject-{index}"})
                    run.assert_not_called()
                self.assertTrue((self.root/f"reject-{index}"/"failed.json").exists())
        (self.session/"collector-result.json").write_bytes(report)

    def test_changed_proof_candidate_run_or_actor_receipt_rejected(self):
        cases = [("proof.lean", None), ("run.lean", None),
                 ("proof-candidate.json", lambda x: x.update(candidate_rows=99)),
                 ("create-receipt.json", lambda x: x["lineage"].update(model_release_sha256="b"*64)),
                 ("process.json", lambda x: x.update(run_source_sha256="b"*64))]
        for index, (name, edit) in enumerate(cases):
            original = (self.session/name).read_bytes()
            with self.subTest(name=name):
                if edit: self.replace_json(name, edit)
                else: (self.session/name).write_bytes(original+b"-- changed\n")
                with patch("cpu_runtime.collector_verify.subprocess.run") as run:
                    with self.assertRaises(ValueError):
                        verify_collected(**{**self.options, "output_dir": self.root/f"wrong-{index}"})
                    run.assert_not_called()
                (self.session/name).write_bytes(original)

    def test_v0_report_cannot_hide_changed_observer_version_or_learning(self):
        original = (self.session/"observer.jsonl").read_bytes()
        for index, change in enumerate(({"policy_version": 1}, {"kind": "learn"}, {"sequence": 99})):
            events = [json.loads(line) for line in original.splitlines()]
            events[1].update(change)
            (self.session/"observer.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
            with patch("cpu_runtime.collector_verify.subprocess.run") as run:
                with self.assertRaises(ValueError):
                    verify_collected(**{**self.options, "output_dir": self.root/f"observer-{index}"})
                run.assert_not_called()

    def test_kernel_failure_and_bad_axioms_never_run_replay_or_install(self):
        for index, (code, stdout) in enumerate(((1, b"failed"), (0, b"'sample' depends on axioms: [sorryAx]\n"),
                                              (0, b"'other' depends on axioms: []\n"))):
            out = self.root/f"kernel-{index}"
            with patch("cpu_runtime.collector_verify.subprocess.run", return_value=
                    subprocess.CompletedProcess([], code, stdout, b"stderr")) as run:
                with self.assertRaises(ValueError): verify_collected(**{**self.options, "output_dir": out})
            self.assertEqual(run.call_count, 1)
            self.assertTrue((out/"proof-receipt.json").exists())
            self.assertEqual((out/"proof.stdout").read_bytes(), stdout)
            self.assertFalse((out/"dataset").exists())
            self.assertFalse(self.options["dataset_store"].exists())

    def test_replay_failure_or_missing_trace_never_installs(self):
        for index in range(2):
            def run(command, **kwargs):
                if Path(command[-1]).name == "replay.lean":
                    return subprocess.CompletedProcess(command, 1 if index == 0 else 0,
                        b"'sample' depends on axioms: []\n", b"")
                return self.lean(command, **kwargs)
            out = self.root/f"replay-{index}"
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.call(lean=run, output_dir=out)
            self.assertTrue((out/"dataset/failed.json").exists())
            self.assertFalse((out/"dataset/dataset.json").exists())
            self.assertFalse(self.options["dataset_store"].exists())

    def test_timeout_preserves_partial_streams_no_retry(self):
        with patch("cpu_runtime.collector_verify.subprocess.run", side_effect=
                subprocess.TimeoutExpired(["lake"], 180, output=b"partial", stderr=b"error")) as run:
            with self.assertRaises(subprocess.TimeoutExpired): verify_collected(**self.options)
        self.assertEqual(run.call_count, 1)
        out = self.options["output_dir"]
        self.assertEqual((out/"proof.stdout").read_bytes(), b"partial")
        self.assertFalse((out/"verification-receipt.json").exists())
        with patch("cpu_runtime.collector_verify.subprocess.run") as run:
            with self.assertRaises(FileExistsError): verify_collected(**self.options)
            run.assert_not_called()

    def test_input_mutation_after_kernel_is_rejected_before_replay(self):
        def run(command, **kwargs):
            result = self.lean(command, **kwargs)
            (self.session/"source.lean").write_bytes(b"changed")
            return result
        with self.assertRaisesRegex(ValueError, "input changed"):
            self.call(lean=run)
        self.assertFalse((self.options["output_dir"]/"dataset").exists())

    def test_install_failure_preserves_accepted_export_and_marks_unknown(self):
        with patch("cpu_runtime.collector_verify.install_verified_dataset", side_effect=OSError("volume failure")):
            with self.assertRaises(OSError): self.call()
        out = self.options["output_dir"]
        self.assertTrue((out/"dataset/dataset.json").exists())
        self.assertTrue(json.loads((out/"failed.json").read_bytes())["registry_may_already_be_published"])
        self.assertFalse((out/"verification-receipt.json").exists())

    def test_ambiguous_install_never_deletes_or_resubmits_published_bundle(self):
        published = []
        def uncertain(*args, **kwargs):
            published.append(install_verified_dataset(*args, **kwargs))
            raise OSError("reply lost after publication")
        with patch("cpu_runtime.collector_verify.install_verified_dataset", side_effect=uncertain) as install:
            with self.assertRaises(OSError): self.call()
            self.assertEqual(install.call_count, 1)
        item = published[0]
        self.assertTrue(Path(item["path"]).is_dir())
        self.assertEqual(load_verified_dataset(Path(item["path"]), expected_sha256=item["dataset_sha256"])["root_return"], -2)
        self.assertFalse((self.options["output_dir"]/"verification-receipt.json").exists())

    def test_invalid_environment_alternate_lean_or_overlapping_paths_rejected(self):
        for change in ({"network": "host"}, {"image_sha256": "unknown"}, {"lean_bin": "custom-lake"},
                       {"output_dir": self.session/"verify"}, {"dataset_store": self.options["output_dir"]/"store"}):
            with self.subTest(change=change), patch("cpu_runtime.collector_verify.subprocess.run") as run:
                with self.assertRaises(ValueError): verify_collected(**{**self.options, **change})
                run.assert_not_called()
        self.assertFalse(self.options["output_dir"].exists())

    def test_linked_input_file_is_refused(self):
        path = self.session/"proof.lean"
        real = self.root/"proof-original.lean"
        path.replace(real)
        try: path.symlink_to(real)
        except OSError:
            real.replace(path)
            self.skipTest("host does not grant file symlink creation")
        with patch("cpu_runtime.collector_verify.subprocess.run") as run:
            with self.assertRaises((ValueError, OSError)): verify_collected(**self.options)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
