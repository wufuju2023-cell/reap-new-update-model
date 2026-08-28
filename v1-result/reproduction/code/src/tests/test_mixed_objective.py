"""Pure fixed-ratio contracts; only the final class loads actual Lean evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import unittest

from gpu_runtime.mixed_objective import (
    OBJECTIVE_KIND, REPLAY_PROFILE, SAMPLE_WEIGHT, SFT_PROFILE, SOURCE_COUNTS,
    TOKENIZATION, make_mixed_sampler, next_mixed_batch, prepare_mixed_event,
)
from gpu_runtime.verified_objective import VALUE_SEMANTICS


class MixedObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.replay_pin, self.sft_pin = "a"*64, "b"*64
        self.replay = {"profile": REPLAY_PROFILE, "session_id": "source-actor",
            "tree_id": "source-actor.tree0", "theorem_sha256": "c"*64,
            "rows": [{"prompt": "replay state", "tactic": "intro n", "return": -4,
                      "policy_version": 17, "node_index": 0},
                     {"prompt": "replay next", "tactic": "rfl", "return": -1,
                      "policy_version": 17, "node_index": 1}]}
        self.sft = {"profile": SFT_PROFILE, "source_kind": "mathlib_sft",
            "source_policy_version": None,
            "source": {"commit": "d"*40, "file": "fixture.lean", "declaration": "Fixture.human"},
            "rows": [{"row": 0, "state": ["sft state"], "next_state": ["sft next"],
                      "prompt": "sft state", "tactic": "apply h", "return": -2},
                     {"row": 1, "state": ["sft next"], "next_state": [],
                      "prompt": "sft next", "tactic": "exact h", "return": -1}]}
        self.loads = []
        self.event = {"kind": OBJECTIVE_KIND, "session_id": "learner", "event_id": "mixed-step1",
            "policy_version": 0, "samples": [
                {"source": "replay", "dataset_sha256": self.replay_pin, "row": i % 2} for i in range(9)]
                + [{"source": "mathlib_sft", "dataset_sha256": self.sft_pin, "row": 0}]}

    def load_replay(self, digest):
        self.assertEqual(digest, self.replay_pin)
        self.loads.append(("replay", digest))
        return self.replay

    def load_sft(self, digest):
        self.assertEqual(digest, self.sft_pin)
        self.loads.append(("mathlib_sft", digest))
        return self.sft

    def prepare(self, event=None):
        return prepare_mixed_event(self.event if event is None else event,
            session_id="learner", policy_version=0, max_distance=8,
            load_replay=self.load_replay, load_mathlib_sft=self.load_sft)

    def sampler(self, **overrides):
        args = {"replay_pins": [self.replay_pin], "mathlib_sft_pins": [self.sft_pin],
            "seed": 71, "max_distance": 8, "load_replay": self.load_replay, "load_mathlib_sft": self.load_sft}
        return make_mixed_sampler(**{**args, **overrides})

    def test_exact_ratio_labels_order_equal_weight_and_no_fake_actor(self):
        before = deepcopy((self.event, self.replay, self.sft))
        prepared = self.prepare()
        self.assertEqual(prepared["source_counts"], SOURCE_COUNTS)
        self.assertEqual(prepared["sample_weight"], SAMPLE_WEIGHT)
        self.assertEqual(prepared["sample_weight"], 0.1)
        self.assertEqual(prepared["value_semantics"], VALUE_SEMANTICS)
        self.assertEqual(prepared["tokenization"], TOKENIZATION)
        self.assertEqual([row["value_class"] for row in prepared["samples"]], [3, 0]*4+[3, 1])
        self.assertEqual(prepared["samples"][0]["source_policy_version"], 17)
        human = prepared["samples"][-1]
        self.assertIsNone(human["source_policy_version"])
        for field in ("source_session_id", "source_tree_id", "node_index", "generation_sequence", "policy_version"):
            self.assertNotIn(field, human)
        self.assertEqual(human["mathlib_source"], self.sft["source"])
        self.assertEqual(self.loads, [("replay", self.replay_pin), ("mathlib_sft", self.sft_pin)])
        self.assertEqual((self.event, self.replay, self.sft), before)
        human["mathlib_source"]["file"] = "cannot-mutate-loader"
        self.assertEqual(self.sft["source"], before[2]["source"])

    def test_input_order_is_preserved_even_if_sft_first(self):
        event = deepcopy(self.event)
        event["samples"] = event["samples"][-1:]+event["samples"][:-1]
        prepared = self.prepare(event)
        self.assertEqual([r["source"] for r in prepared["samples"]], ["mathlib_sft"]+["replay"]*9)

    def test_bad_ratio_and_extra_fields_rejected_before_any_loading(self):
        cases = []
        for field in ("prompt", "tactic", "return", "reward", "source_counts", "sample_weight"):
            cases.append({**self.event, field: 1})
            e = deepcopy(self.event); e["samples"][-1][field] = 1; cases.append(e)
        for count in (0, 9, 11, 32):
            cases.append({**self.event, "samples": [self.event["samples"][0]]*count})
        for sft_count in (0, 2, 10):
            cases.append({**self.event, "samples": [self.event["samples"][0]]*(10-sft_count)
                          + [self.event["samples"][-1]]*sft_count})
        for e in cases:
            with self.subTest(event=e), self.assertRaises(ValueError):
                self.prepare(e)
        self.assertEqual(self.loads, [])

    def test_bad_identity_version_pin_source_row_rejected(self):
        for field, value in (("session_id", "actor"), ("event_id", "../unsafe"),
                             ("policy_version", True), ("policy_version", 1),
                             ("kind", "verified_success_replay")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare({**self.event, field: value})
        for field, value in (("source", "human"), ("source", []), ("source", None),
                             ("dataset_sha256", "A"*64), ("dataset_sha256", "../file"),
                             ("row", True), ("row", -1), ("row", 2.0), ("row", 2)):
            e = deepcopy(self.event); e["samples"][-1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.prepare(e)

    def test_source_profiles_cannot_be_swapped_or_silently_fallback(self):
        for source in ("replay", "sft"):
            dataset = getattr(self, source); old = dataset["profile"]
            dataset["profile"] = SFT_PROFILE if source == "replay" else REPLAY_PROFILE
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "profile"):
                self.prepare()
            dataset["profile"] = old
        def corrupt(_):
            raise ValueError("real loader refused missing/corrupt bundle")
        before = deepcopy(self.event)
        with self.assertRaisesRegex(ValueError, "corrupt bundle"):
            prepare_mixed_event(self.event, session_id="learner", policy_version=0, max_distance=8,
                load_replay=self.load_replay, load_mathlib_sft=corrupt)
        self.assertEqual(self.event, before)

    def test_sft_cannot_invent_generated_fields_and_labels_never_clipped(self):
        for key, value in (("source_policy_version", 0), ("source_kind", "replay")):
            old = self.sft[key]; self.sft[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare()
            self.sft[key] = old
        for key in ("generation_sequence", "policy_version", "session_id"):
            self.sft["rows"][0][key] = 0
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "generated actor"):
                self.prepare()
            del self.sft["rows"][0][key]
        for source in ("replay", "sft"):
            row = getattr(self, source)["rows"][0]; old = row["return"]
            for value in (0, -9, -1.0, True, None):
                row["return"] = value
                with self.subTest(source=source, value=value), self.assertRaisesRegex(ValueError, "never clip"):
                    self.prepare()
            row["return"] = old

    def test_malformed_rows_text_and_human_source_fail_closed(self):
        for source in ("replay", "sft"):
            dataset = getattr(self, source); old = dataset["rows"]
            for rows in ([], None, [None]):
                dataset["rows"] = rows
                with self.subTest(source=source, rows=rows), self.assertRaises(ValueError):
                    self.prepare()
            dataset["rows"] = old
            for field in ("prompt", "tactic"):
                value = old[0][field]; old[0][field] = " "
                with self.subTest(source=source, field=field), self.assertRaises(ValueError):
                    self.prepare()
                old[0][field] = value
        self.sft["source"] = None
        with self.assertRaisesRegex(ValueError, "source provenance"):
            self.prepare()

    def test_two_cursors_restore_deterministically_without_input_or_rng_mutation(self):
        config, state = self.sampler()
        before = deepcopy((config, state)); rng = random.getstate()
        batch1, after1 = next_mixed_batch(config, state)
        self.assertEqual(next_mixed_batch(config, state), (batch1, after1))
        self.assertEqual((config, state), before)
        batch2, after2 = next_mixed_batch(config, after1)
        restored = json.loads(json.dumps({"config": config, "state": after1}))
        self.assertEqual(next_mixed_batch(restored["config"], restored["state"]), (batch2, after2))
        self.assertEqual((after2["step"], after2["replay_cursor"], after2["mathlib_sft_cursor"]), (2, 18, 2))
        self.assertEqual(random.getstate(), rng)
        for batch in (batch1, batch2):
            e = {**self.event, "samples": batch}
            self.assertEqual(self.prepare(e)["source_counts"], SOURCE_COUNTS)
        self.assertNotEqual(batch1[-1]["row"], batch2[-1]["row"])

    def test_seeded_offsets_and_multiple_dataset_boundaries_are_reproducible(self):
        second_pin = "e"*64
        other = deepcopy(self.replay); other["rows"] = other["rows"][:1]
        def loader(pin):
            return {self.replay_pin: self.replay, second_pin: other}[pin]
        config, state = self.sampler(replay_pins=[self.replay_pin, second_pin], load_replay=loader)
        batch, _ = next_mixed_batch(config, state)
        self.assertEqual({(r["dataset_sha256"], r["row"]) for r in batch[:9]},
            {(self.replay_pin, 0), (self.replay_pin, 1), (second_pin, 0)})
        sequences = set()
        for seed in range(8):
            c, s = self.sampler(seed=seed)
            sequences.add(json.dumps(next_mixed_batch(c, s)[0], sort_keys=True))
        self.assertGreater(len(sequences), 1)

    def test_empty_duplicate_catalogs_rejected_before_load_and_all_rows_admitted(self):
        for overrides in ({"replay_pins": []}, {"mathlib_sft_pins": []},
                          {"mathlib_sft_pins": (self.sft_pin,)},
                          {"replay_pins": [self.replay_pin]*2},
                          {"mathlib_sft_pins": [self.replay_pin]}, {"seed": True}, {"max_distance": 1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.sampler(**overrides)
        self.assertEqual(self.loads, [])
        # An invalid row that is not the first selected SFT row still blocks admission.
        self.sft["rows"][1]["return"] = -9
        with self.assertRaisesRegex(ValueError, "never clip"):
            self.sampler()

    def test_restore_rejects_catalog_seed_support_and_cursor_changes(self):
        config, state = self.sampler()
        _, state = next_mixed_batch(config, state)
        changes = [{**config, "seed": 99}, {**config, "max_distance": 16}]
        c = deepcopy(config); c["catalog"]["replay"][0]["rows"] += 1; changes.append(c)
        c = deepcopy(config); c["catalog"]["replay"][0]["dataset_sha256"] = "e"*64; changes.append(c)
        for c in changes:
            with self.subTest(config=c), self.assertRaisesRegex(ValueError, "different fixed"):
                next_mixed_batch(c, state)
        for field, value in (("step", 0), ("step", True), ("replay_cursor", 10),
                             ("mathlib_sft_cursor", 0), ("schema_version", "old"),
                             ("config_sha256", "f"*64), ("untrusted", 1)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                next_mixed_batch(config, {**state, field: value})

    def test_admission_checks_replay_rows_beyond_single_validator_chunk(self):
        self.replay["rows"] = [deepcopy(self.replay["rows"][0]) for _ in range(33)]
        self.replay["rows"][-1]["return"] = -9
        with self.assertRaisesRegex(ValueError, "never clip"):
            self.sampler()
        self.replay["rows"][-1]["return"] = -1
        config, state = self.sampler()
        self.assertEqual(config["catalog"]["replay"][0]["rows"], 33)
        _, after = next_mixed_batch(config, state)
        self.assertEqual(after["replay_cursor"], 9)

    def test_invalid_config_and_integer_overflow_rejected_without_state_mutation(self):
        config, state = self.sampler()
        cases = [{**config, "kind": "random-global-rng"}, {**config, "seed": True},
                 {**config, "objective": "verified_success_replay"}, {**config, "batch_size": 10}]
        for field, value in (("rows", 0), ("rows", True), ("profile", SFT_PROFILE),
                             ("dataset_sha256", "UPPER")):
            c = deepcopy(config); c["catalog"]["replay"][0][field] = value; cases.append(c)
        for c in cases:
            with self.subTest(config=c), self.assertRaises(ValueError):
                next_mixed_batch(c, state)
        before = deepcopy(state)
        for field in ("step", "replay_cursor", "mathlib_sft_cursor"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                next_mixed_batch(config, {**state, field: 2**53})
        self.assertEqual(state, before)


ROOT = Path(__file__).resolve().parents[1]
REPLAY_DIR = ROOT/".downloads/verified-trajectory-local-20260828/final-v3/03"
SFT_DIR = ROOT/".downloads/mathlib-sft-local-20260828/final/intro"
REPLAY_PIN = "7975e11a24d74f326387e82fd4b8f9fc2489d1d570e44c17c5f8a03e1b9555ff"
SFT_PIN = "98c67e8f220e8f5a9227ce010a95792f838b10a9cee6f15a98f1eae695d788cb"


@unittest.skipUnless(REPLAY_DIR.is_dir() and SFT_DIR.is_dir(), "optional local pinned Lean evidence not present")
class MixedRealEvidenceTests(unittest.TestCase):
    def test_two_actual_strict_loaders_resolve_fixed_batch_without_new_lean_or_gpu(self):
        from cpu_runtime.verified_trajectory import load_verified_dataset
        from cpu_runtime.mathlib_trajectory import load_mathlib_dataset
        def hashes():
            return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for directory in (REPLAY_DIR, SFT_DIR) for p in directory.iterdir() if p.is_file()}
        before = hashes()
        replay = lambda pin: load_verified_dataset(REPLAY_DIR, expected_sha256=pin)
        sft = lambda pin: load_mathlib_dataset(SFT_DIR, expected_sha256=pin)
        config, state = make_mixed_sampler(replay_pins=[REPLAY_PIN], mathlib_sft_pins=[SFT_PIN],
            seed=17, max_distance=64, load_replay=replay, load_mathlib_sft=sft)
        refs, after = next_mixed_batch(config, state)
        event = {"kind": OBJECTIVE_KIND, "event_id": "local-real-mixed1", "session_id": "local-learner",
            "policy_version": 0, "samples": refs}
        result = prepare_mixed_event(event, session_id="local-learner", policy_version=0, max_distance=64,
            load_replay=replay, load_mathlib_sft=sft)
        self.assertEqual(result["source_counts"], {"replay": 9, "mathlib_sft": 1})
        self.assertEqual([r["return"] for r in result["samples"][:9]].count(-3), 3)
        self.assertEqual([r["return"] for r in result["samples"][:9]].count(-2), 3)
        self.assertEqual([r["return"] for r in result["samples"][:9]].count(-1), 3)
        self.assertIn(result["samples"][-1]["return"], (-2, -1))
        self.assertIsNone(result["samples"][-1]["source_policy_version"])
        self.assertEqual(after["step"], 1)
        self.assertEqual(before, hashes())
        with self.assertRaises(ValueError):
            prepare_mixed_event(event, session_id="local-learner", policy_version=0, max_distance=64,
                load_replay=sft, load_mathlib_sft=replay)


if __name__ == "__main__":
    unittest.main()
