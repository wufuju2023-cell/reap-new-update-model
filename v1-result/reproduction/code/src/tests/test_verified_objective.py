"""Pure replay contracts; fixture loader does not claim Lean verification."""
from copy import deepcopy
import unittest

from gpu_runtime.verified_objective import DATA_PROFILE, prepare_verified_event, validate_support


class VerifiedObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.digest = "a"*64
        self.dataset = {"profile": DATA_PROFILE, "session_id": "actor", "tree_id": "actor.tree0",
            "theorem_sha256": "b"*64, "rows": [{"prompt": "state", "tactic": "intro n",
                "return": -4, "node_index": 0, "policy_version": 17},
                {"prompt": "next state", "tactic": "rfl", "return": -1,
                 "node_index": 1, "policy_version": 17}]}
        self.event = {"kind": "verified_success_replay", "session_id": "learner", "event_id": "learner.batch0",
            "policy_version": 0, "samples": [{"dataset_sha256": self.digest, "row": 0},
                                             {"dataset_sha256": self.digest, "row": 1}]}
        self.loads = []

    def prepare(self, event=None):
        def fixture_load(digest):
            self.loads.append(digest)
            return deepcopy(self.dataset)
        return prepare_verified_event(self.event if event is None else event,
            session_id="learner", policy_version=0, max_distance=8, load_dataset=fixture_load)

    def test_verified_rows_define_labels_and_actor_version_is_not_learner_version(self):
        old = deepcopy(self.event)
        result = self.prepare()
        self.assertEqual([s["value_class"] for s in result["samples"]], [3, 0])
        self.assertEqual([s["return"] for s in result["samples"]], [-4, -1])
        self.assertEqual(result["sample_weight"], 0.5)
        self.assertEqual(result["samples"][0]["source_policy_version"], 17)
        self.assertEqual(self.loads, [self.digest])
        self.assertEqual(self.event, old)

    def test_client_cannot_supply_labels_or_training_text(self):
        for field, value in (("reward", 1), ("return", -1), ("prompt", "forged"),
                             ("terminal_verified", True), ("backup", {})):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "client labels"):
                self.prepare({**self.event, field: value})
        self.assertEqual(self.loads, [])

    def test_bad_identity_pin_index_and_version_fail_closed(self):
        for field, value in (("session_id", "actor"), ("policy_version", True),
                             ("policy_version", 1), ("event_id", "../escape"), ("kind", "search_visit_backup")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare({**self.event, field: value})
        for ref in ({"dataset_sha256": "A"*64, "row": 0}, {"dataset_sha256": self.digest, "row": -1},
                    {"dataset_sha256": self.digest, "row": True}, {"dataset_sha256": self.digest, "row": 2},
                    {"dataset_sha256": self.digest, "row": 0, "return": -1}):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.prepare({**self.event, "samples": [ref]})

    def test_invalid_verified_label_never_clips_or_uses_search_backup(self):
        for value in (0, -9, -1.0, True, None):
            self.dataset["rows"][0]["return"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "never clip"):
                self.prepare()
        self.dataset["rows"][0]["return"] = -4
        self.dataset["profile"] = "search_visit_backup"
        with self.assertRaisesRegex(ValueError, "profile mismatch"):
            self.prepare()

    def test_bounded_batch_support_and_explicit_replacement(self):
        self.event["samples"] = [self.event["samples"][0]]*32
        result = self.prepare()
        self.assertEqual(len(result["samples"]), 32)
        self.assertEqual(result["sample_weight"], 1/32)
        for count in (0, 33):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.prepare({**self.event, "samples": self.event["samples"][:1]*count})
        for support in (1, 4097, True, 8.0):
            with self.subTest(support=support), self.assertRaises(ValueError):
                validate_support(support)

    def test_lean_loader_rejection_propagates(self):
        def rejected(digest):
            raise ValueError("replay receipt corrupted")
        with self.assertRaisesRegex(ValueError, "receipt corrupted"):
            prepare_verified_event(self.event, session_id="learner", policy_version=0,
                                   max_distance=8, load_dataset=rejected)


if __name__ == "__main__":
    unittest.main()
