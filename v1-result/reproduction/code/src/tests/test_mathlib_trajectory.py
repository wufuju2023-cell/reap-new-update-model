"""Protocol/parser tests only; real Mathlib/Lean acceptance is separate."""
import copy
from pathlib import Path, PurePosixPath
import tempfile
import unittest

from cpu_runtime import mathlib_trajectory as m


SOURCE = ("-- Unicode prefix α keeps byte offsets distinct from character indices\n"
          "theorem ExistsUnique.intro₂ {p : Prop} (h : p) : p := by\n"
          "  simp only []\n"
          "  exact h\n\n").encode()


def trace():
    rows = []
    for index, (state, tactic, after) in enumerate([
        ("p : Prop\nh : p\n⊢ p", "simp only []", ["p : Prop\nh : p\n⊢ p"]),
        ("p : Prop\nh : p\n⊢ p", "exact h", []),
    ]):
        rows.append({"row": index, "state": [state], "tactic": tactic,
                     "next_state": after, "prompt": m.prompt_for_state(state), "return": index - 2})
    return {"schema_version": "reap.mathlib-sft.linear-trace.v1", "complete": True,
            "theorem": "MathlibSFTGate.ExistsUnique.intro₂", "original_declaration": "ExistsUnique.intro₂",
            "root_return": -2, "rows": rows}


class MathlibTrajectoryTests(unittest.TestCase):
    def selection(self):
        return m.extract_declaration(SOURCE, "ExistsUnique.intro₂")

    def test_original_utf8_byte_spans_and_tactics_are_exact(self):
        selection = self.selection()
        self.assertEqual(SOURCE[selection["byte_start"]:selection["proof_byte_start"]].decode(),
                         "theorem ExistsUnique.intro₂ {p : Prop} (h : p) : p := by\n")
        self.assertEqual(selection["proof_sha256"], m.sha(b"  simp only []\n  exact h\n"))
        for row in selection["actions"]:
            original = SOURCE[row["byte_start"]:row["byte_end"]]
            self.assertEqual(original.decode(), row["tactic"])
            self.assertEqual(m.sha(original), row["sha256"])

    def test_unknown_duplicate_or_missing_declaration_is_rejected(self):
        for source, name in ((SOURCE, "model_generated"), (SOURCE + SOURCE, "ExistsUnique.intro₂"),
                             (b"", "ExistsUnique.intro₂")):
            with self.subTest(name=name), self.assertRaises(m.MathlibTrajectoryRejected):
                m.extract_declaration(source, name)

    def test_unsupported_splitting_and_unsafe_actions_rejected(self):
        for tactic in ("   exact h", "  exact h; exact h", "  all_goals exact h", "  sorry",
                       "  exact h -- comment", "  · exact h"):
            source = SOURCE.replace(b"  exact h", tactic.encode())
            with self.subTest(tactic=tactic), self.assertRaises(m.MathlibTrajectoryRejected):
                m.extract_declaration(source, "ExistsUnique.intro₂")

    def test_complete_linear_trace_has_negative_actual_remaining_counts(self):
        m.validate_trace(trace(), self.selection())
        self.assertEqual([row["return"] for row in trace()["rows"]], [-2, -1])

    def test_no_branch_fake_terminal_zero_or_generated_identity(self):
        mutations = [lambda t: t["rows"][0].update(state=["one", "two"]),
                     lambda t: t["rows"][-1].update(next_state=["unfinished"]),
                     lambda t: t["rows"][0].update(next_state=[]),
                     lambda t: t["rows"][0].update(return_value=0),
                     lambda t: t["rows"][0].update({"return": 0}),
                     lambda t: t["rows"][0].update(policy_version=0),
                     lambda t: t.update(root_return=True),
                     lambda t: t.update(complete=False)]
        for mutate in mutations:
            item = copy.deepcopy(trace()); mutate(item)
            with self.assertRaises(m.MathlibTrajectoryRejected):
                m.validate_trace(item, self.selection())

    def test_prompt_and_tactic_cannot_be_substituted(self):
        for field, value in (("prompt", "invented model event"), ("tactic", "exact True.intro")):
            item = trace(); item["rows"][0][field] = value
            with self.assertRaises(m.MathlibTrajectoryRejected):
                m.validate_trace(item, self.selection())

    def test_unicode_axiom_name_is_exact_and_unsafe_axioms_rejected(self):
        name = "MathlibSFTGate.ExistsUnique.intro₂"
        good = f"'{name}' depends on axioms: [propext, Classical.choice, Quot.sound]\n"
        self.assertEqual(m.checked_axioms(good, name), ["propext", "Classical.choice", "Quot.sound"])
        for stdout in (good + good, good.replace(name, "wrong"), good.replace("propext", "sorryAx")):
            with self.assertRaises(m.MathlibTrajectoryRejected):
                m.checked_axioms(stdout, name)

    def test_loader_requires_external_pin_before_any_file_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            for pin in (None, "", "A" * 64, "../outside"):
                with self.subTest(pin=pin), self.assertRaises(m.MathlibTrajectoryRejected):
                    m.load_mathlib_dataset(Path(directory), expected_sha256=pin)

    def test_linux_driver_paths_are_preserved_by_cross_platform_reader(self):
        script = m.program(SOURCE, self.selection(), b"", PurePosixPath("/evidence/sft"), "replay").decode()
        self.assertIn('"/evidence/sft/plan.json" "/evidence/sft/replay-trace.json" "/evidence/sft/capture-trace.json"', script)
        self.assertNotIn("\\\\", script)


if __name__ == "__main__":
    unittest.main()
