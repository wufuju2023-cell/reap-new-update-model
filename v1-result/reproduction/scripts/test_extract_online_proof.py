"""Local mechanism tests. The fake runner does not execute Lean or a container."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("proof_helper", Path(__file__).with_name("extract_online_proof.py"))
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
IMAGE = "a" * 64
SID = "fresh-source-01"
INPUT = "01-CoupledOddSquare.lean"
THEOREM = helper.THEOREMS[INPUT]


class ProofCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "inputs").mkdir()
        self.session = self.root / "outputs" / SID
        self.session.mkdir(parents=True)
        raw = ("import ReapRuntime\nnamespace MultiroundCandidates.CoupledOddSquare\n"
               "theorem coupled_odd_square : True := by\n  reapTrainingMCTS\n"
               "end MultiroundCandidates.CoupledOddSquare\n").encode()
        (self.root / "inputs" / INPUT).write_bytes(raw)
        self.put("session.json", {"session_id": SID, "theorem_sha256": helper.sha(raw)})
        self.put("result.json", {"session_id": SID, "solved": True, "proof_script": "trivial"})
        self.put("online-result.json", {"session_id": SID, "root_verified": True, "returncode": 0,
                                       "error": None, "status": "passed_execution", "optimizer_updates": 1})
        self.calls = []
        self.exit_code = 0
        self.running = False
        self.network = "none"
        self.image = IMAGE
        self.axioms = "[propext]"
        self.inspect_failure = False

    def put(self, name, value):
        (self.session / name).write_text(json.dumps(value), encoding="utf-8")

    def runner(self, argv, timeout=600):
        self.calls.append(argv)
        if argv[1:3] == ["container", "exists"]:
            return subprocess.CompletedProcess(argv, 1, b"", b"")
        if argv[1] == "run":
            text = f"'{THEOREM}' depends on axioms: {self.axioms}\n".encode()
            return subprocess.CompletedProcess(argv, self.exit_code, text, b"")
        if argv[1] == "inspect":
            info = [{"Image": self.image, "State": {"Running": self.running, "ExitCode": self.exit_code},
                     "HostConfig": {"NetworkMode": self.network}}]
            return subprocess.CompletedProcess(argv, int(self.inspect_failure), json.dumps(info).encode(), b"")
        raise AssertionError(argv)

    def run_check(self):
        return helper.verify(self.root, SID, INPUT, THEOREM, IMAGE, runner=self.runner)

    def Rex(self, message):
        return self.assertRaisesRegex(ValueError, message)

    def test_success_binds_actual_proof_and_offline_command(self):
        value = self.run_check()
        out = self.root / "proof-check" / SID
        self.assertTrue(value["passed"])
        self.assertEqual(value["schema_version"], helper.SCHEMA)
        self.assertEqual(value["axioms"], ["propext"])
        self.assertEqual(value["proof_sha256"], helper.sha((out / "proof.lean").read_bytes()))
        self.assertNotIn("reapTrainingMCTS", (out / "proof.lean").read_text())
        self.assertTrue((out / "accepted.json").is_file())
        self.assertEqual(self.calls[1][self.calls[1].index("--network") + 1], "none")
        self.assertNotIn("--rm", self.calls[1])

    def test_existing_output_cannot_repeat(self):
        self.run_check()
        count = len(self.calls)
        with self.assertRaises(FileExistsError):
            self.run_check()
        self.assertEqual(len(self.calls), count)

    def test_changed_theorem_refused_before_container(self):
        (self.root / "inputs" / INPUT).write_text("changed", encoding="utf-8")
        with self.Rex("differs"):
            self.run_check()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / "proof-check" / SID / "accepted.json").exists())

    def test_unsolved_refused(self):
        self.put("result.json", {"session_id": SID, "solved": False})
        with self.Rex("solved"):
            self.run_check()
        self.assertEqual(self.calls, [])

    def test_unknown_online_refused(self):
        self.put("online-result.json", {"session_id": SID, "root_verified": True,
                                       "returncode": 0, "error": "unknown learn"})
        with self.Rex("known verified"):
            self.run_check()
        self.assertEqual(self.calls, [])

    def test_declaration_binding(self):
        with self.Rex("mapping"):
            helper.verify(self.root, SID, INPUT, "Wrong.name", IMAGE, runner=self.runner)
        self.assertEqual(self.calls, [])

    def test_leftover_search_refused(self):
        self.put("result.json", {"session_id": SID, "solved": True, "proof_script": "reapTrainingMCTS"})
        with self.Rex("invokes search"):
            self.run_check()

    def test_nonzero_exit_keeps_failure_receipt(self):
        self.exit_code = 1
        with self.Rex("Lean failed"):
            self.run_check()
        out = self.root / "proof-check" / SID
        self.assertFalse(json.loads((out / "receipt.json").read_bytes())["passed"])
        self.assertFalse((out / "accepted.json").exists())

    def test_active_container_refused(self):
        self.running = True
        with self.Rex("active"):
            self.run_check()

    def test_wrong_network_refused(self):
        self.network = "host"
        with self.Rex("offline"):
            self.run_check()

    def test_wrong_image_refused(self):
        self.image = "b" * 64
        with self.Rex("image differs"):
            self.run_check()

    def test_sorry_ax_refused(self):
        self.axioms = "[propext, sorryAx]"
        with self.Rex("sorryAx"):
            self.run_check()

    def test_unexpected_axiom_refused(self):
        self.axioms = "[untrustedAxiom]"
        with self.Rex("unexpected"):
            self.run_check()

    def test_unconfirmed_exit_not_success(self):
        self.inspect_failure = True
        with self.Rex("final container"):
            self.run_check()

    def test_timeout_keeps_partial_output_without_retry(self):
        def timeout_runner(argv, timeout=600):
            if argv[1] == "run":
                self.calls.append(argv)
                raise subprocess.TimeoutExpired(argv, timeout, output=b"partial", stderr=b"pending")
            return self.runner(argv, timeout=timeout)
        with self.assertRaises(subprocess.TimeoutExpired):
            helper.verify(self.root, SID, INPUT, THEOREM, IMAGE, runner=timeout_runner)
        out = self.root / "proof-check" / SID
        self.assertEqual((out / "stdout.log").read_bytes(), b"partial")
        self.assertFalse((out / "accepted.json").exists())
        self.assertEqual(sum(argv[1] == "run" for argv in self.calls), 1)

    def test_existing_container_refuses_launch(self):
        def existing(argv, timeout=600):
            self.calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        with self.Rex("container already exists"):
            helper.verify(self.root, SID, INPUT, THEOREM, IMAGE, runner=existing)
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
