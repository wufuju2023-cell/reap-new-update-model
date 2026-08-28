from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from cpu_runtime import matchmaker as mm
from cpu_runtime import closed_problem


def pin(text):
    return mm.content_sha256(text.encode())


RELEASE = pin("release-1")


def config(**updates):
    return {"history_window": 4, "trust_count": 2, "mastery_count": 2,
        "base_steps": 8, "budget_multiplier": 2, "cap_steps": 32,
        "max_attempts": 12, "total_steps": 256, "seed": 9,
        "max_inflight": 2, "max_wait_dispatches": 3, **updates}


def curriculum(families=(2, 2)):
    environment = pin("fixed-environment")
    problems = []
    for family, size in enumerate(families):
        parent = None
        for number in range(size):
            declarations = pin(f"declarations-{family}-{number}")
            name = f"Family{family}.prop{number}"
            identity = mm.problem_pin(environment, declarations, name)
            problems.append({"problem_id": f"p{family}_{number}", "family_id": f"f{family}",
                "is_target": number == 0, "problem_sha256": identity, "declarations_sha256": declarations,
                "closed_prop_name": name, "compilation_receipt_sha256": pin(f"compiled-{family}-{number}"),
                "parent_problem_sha256": parent, "transformation_sha256": pin(f"transform-{number}") if number else None})
            if number == 0:
                parent = identity
    return {"schema_version": "reap.matchmaker.curriculum.v1", "environment_sha256": environment,
            "wrapper_version": mm.WRAPPER_VERSION, "problems": problems}


def source(proposal):
    return {"execution_source_sha256": pin("execution-"+proposal["attempted_prop_sha256"]+str(proposal["budget_steps"])),
        "compilation_receipt_sha256": pin("compile-"+proposal["attempt_id"]),
        "attempted_prop_sha256": proposal["attempted_prop_sha256"], "budget_steps": proposal["budget_steps"]}


def terminal(intent, outcome="exhausted"):
    proposal = intent["proposal"]
    if outcome == "accepted":
        evidence = {"verdict": {"prove": "proved", "disprove": "disproved"}[proposal["polarity"]],
            **{key: proposal[key] for key in ("problem_sha256", "attempted_prop_sha256", "model_release_sha256")},
            "execution_source_sha256": intent["attempt_source"]["execution_source_sha256"],
            "proof_sha256": pin("proof"), "verification_receipt_sha256": pin("verified"),
            "dataset_sha256": pin("dataset"), "dataset_receipt_sha256": pin("dataset-receipt"), "actual_steps": 2}
    elif outcome == "unknown":
        evidence = {"diagnostic_sha256": pin("unknown"), "reason": "transport_unknown"}
    else:
        evidence = {"terminal_receipt_sha256": pin("exhausted"), "actual_steps": proposal["budget_steps"]}
    return {"schema_version": "reap.matchmaker.result.v1", "attempt_id": proposal["attempt_id"],
            "intent_sha256": intent["intent_sha256"], "outcome": outcome, "evidence": evidence}


def reserve(scheduler, release=RELEASE):
    proposal = scheduler.plan_next(release)
    if proposal is None:
        return None
    return scheduler.reserve(proposal, source(proposal))


def finish(scheduler, intent, outcome="exhausted"):
    result = terminal(intent, outcome)
    return scheduler.record_result(result["attempt_id"], result)


class MatchmakerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def create(self, name="run", families=(2, 2), **updates):
        scheduler = mm.Matchmaker.create(self.root/name, curriculum(families), config(**updates))
        self.addCleanup(scheduler.close)
        return scheduler

    def reopen(self, name, run_pin):
        scheduler = mm.Matchmaker.open(self.root/name, run_pin)
        self.addCleanup(scheduler.close)
        return scheduler

    def test_closed_problem_identity_compatibility_and_budget_separation(self):
        data = b"import ReapRuntime\ndef Foo.target : Prop := True\n"
        prepared = closed_problem.prepare(environment_sha256=pin("env"), declarations=data,
                                         closed_prop_name="Foo.target", polarity="disprove", budget_steps=8)
        identity = prepared["problem"]
        self.assertEqual(identity["problem_sha256"], mm.problem_pin(pin("env"), pin(data.decode()), "Foo.target"))
        self.assertEqual(prepared["attempted"]["attempted_prop_sha256"], mm.attempted_prop_pin(identity["problem_sha256"], "disprove"))
        other = closed_problem.prepare(environment_sha256=pin("env"), declarations=data,
                                       closed_prop_name="Foo.target", polarity="disprove", budget_steps=16)
        self.assertEqual(prepared["problem"], other["problem"])
        self.assertEqual(prepared["attempted"], other["attempted"])
        self.assertNotEqual(prepared["execution_source_sha256"], other["execution_source_sha256"])

    def test_fixed_curriculum_strict_identity_and_parent_validation(self):
        mutations = [lambda c: c.update(extra=1), lambda c: c.update(wrapper_version="other"),
            lambda c: c["problems"][0].update(is_target=1), lambda c: c["problems"][0].update(problem_sha256=pin("wrong")),
            lambda c: c["problems"][1].update(parent_problem_sha256=pin("wrong")),
            lambda c: c["problems"][0].update(closed_prop_name="unqualified"),
            lambda c: c["problems"][1].update(compilation_receipt_sha256="ABC"),
            lambda c: c["problems"].append(deepcopy(c["problems"][0])),
            lambda c: c["problems"][0].update(transformation_sha256=pin("extra")),
            lambda c: c["problems"].pop(0)]
        for change in mutations:
            with self.subTest(change=change):
                value = curriculum(); change(value)
                with self.assertRaises(mm.MatchmakerError):
                    mm.Matchmaker.create(self.root/"invalid", value, config())
                self.assertFalse((self.root/"invalid").exists())

    def test_config_rejects_noninteger_nonfinite_and_out_of_range(self):
        for update in ({"seed": True}, {"seed": -1}, {"seed": 2**63}, {"total_steps": float("inf")},
                       {"max_attempts": 0}, {"max_inflight": 5}, {"history_window": 1}, {"base_steps": 33}):
            with self.subTest(update=update), self.assertRaises(mm.MatchmakerError):
                mm.validate_config(config(**update))
        mm.validate_config(config(seed=2**63-1))

    def test_deterministic_sequence_fairness_and_finite_budgets(self):
        left = self.create("left"); right = self.create("right")
        self.assertEqual(left.run_sha256, right.run_sha256)
        observed = []
        for _ in range(8):
            a, b = reserve(left), reserve(right)
            self.assertEqual(a, b)
            observed.append(a["proposal"]["problem_id"])
            finish(left, a); finish(right, b)
        self.assertEqual(len(set(observed[:4])), 4)
        self.assertEqual(observed[:4], observed[4:])
        self.assertEqual(left.status(), right.status())
        self.assertEqual(left.status()["reserved_steps"], 4*8+4*16)
        self.assertTrue(all(item["known_results"] == 2 for item in left.status()["problems"].values()))

    def test_per_problem_inflight_and_global_slot_limit(self):
        scheduler = self.create(max_inflight=2)
        first, second = reserve(scheduler), reserve(scheduler)
        self.assertNotEqual(first["proposal"]["problem_id"], second["proposal"]["problem_id"])
        self.assertIsNone(scheduler.plan_next(RELEASE))
        finish(scheduler, second)
        third = reserve(scheduler)
        self.assertNotEqual(first["proposal"]["problem_id"], third["proposal"]["problem_id"])
        self.assertEqual(len(scheduler.status()["pending_attempts"]), 2)

    def test_release_only_changes_new_attempt_and_proposals_are_snapshots(self):
        scheduler = self.create()
        first = reserve(scheduler)
        second = reserve(scheduler, pin("release-2"))
        self.assertEqual(first["proposal"]["model_release_sha256"], RELEASE)
        self.assertEqual(second["proposal"]["model_release_sha256"], pin("release-2"))
        first["proposal"]["model_release_sha256"] = pin("tampered")
        saved = scheduler.status()["attempts"][first["proposal"]["attempt_id"]]["intent"]
        self.assertEqual(saved["proposal"]["model_release_sha256"], RELEASE)

    def test_stale_altered_uncompiled_source_cannot_reserve(self):
        scheduler = self.create()
        proposal = scheduler.plan_next(RELEASE)
        for key, value in (("budget_steps", True), ("budget_steps", 16), ("attempted_prop_sha256", pin("other"))):
            candidate = deepcopy(proposal); candidate[key] = value
            with self.assertRaises(mm.MatchmakerError):
                scheduler.reserve(candidate, source(proposal))
        bad = source(proposal); bad["budget_steps"] = 16
        with self.assertRaises(mm.MatchmakerError):
            scheduler.reserve(proposal, bad)
        self.assertEqual(scheduler.status()["event_count"], 0)
        first = scheduler.reserve(proposal, source(proposal))
        with self.assertRaises(mm.MatchmakerError):
            scheduler.reserve(proposal, source(proposal))
        finish(scheduler, first)

    def test_no_refunds_and_capped_failure_budget(self):
        scheduler = self.create(families=(1,), total_steps=100, max_attempts=10)
        budgets = []
        for _ in range(4):
            intent = reserve(scheduler); budgets.append(intent["proposal"]["budget_steps"])
            result = terminal(intent); result["evidence"]["actual_steps"] = 0
            scheduler.record_result(result["attempt_id"], result)
        self.assertEqual(budgets, [8, 16, 32, 32])
        self.assertEqual(scheduler.status()["reserved_steps"], 88)
        self.assertIsNone(reserve(scheduler))  # remaining 12 cannot cover the next full 32-step budget.

    def test_recent_mixed_mastery_priorities_and_starvation_guard(self):
        scheduler = self.create(families=(2,), seed=0, max_attempts=18, total_steps=1000)
        decisions, variant_history = [], []
        for _ in range(16):
            intent = reserve(scheduler); proposal = intent["proposal"]
            decisions.append((proposal["problem_id"], proposal["priority_reason"]))
            outcome = "accepted" if proposal["problem_id"] == "p0_1" and proposal["polarity"] == "prove" else "exhausted"
            finish(scheduler, intent, outcome)
            if proposal["problem_id"] == "p0_1":
                variant_history.append(outcome)
                expected_budget = min(32, 8 * 2**variant_history[-4:].count("exhausted"))
                self.assertEqual(scheduler.status()["problems"]["p0_1"]["next_budget_steps"], expected_budget)
        self.assertIn(("p0_1", "mixed"), decisions)
        self.assertIn(("p0_1", "mastered"), decisions)
        self.assertIn(("p0_0", "fairness"), decisions)
        target_indices = [i for i, item in enumerate(decisions) if item[0] == "p0_0"]
        self.assertTrue(all(b-a <= 4 for a, b in zip(target_indices, target_indices[1:])))

    def test_concurrent_callers_cannot_reserve_same_proposal_twice(self):
        scheduler = self.create()
        barrier = threading.Barrier(2)
        def racing_reserve():
            proposal = scheduler.plan_next(RELEASE)
            barrier.wait(timeout=5)
            try:
                return scheduler.reserve(proposal, source(proposal))
            except mm.MatchmakerError:
                return None
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(racing_reserve) for _ in range(2)]
            results = [future.result(timeout=5) for future in futures]
        self.assertEqual(sum(result is not None for result in results), 1)
        self.assertEqual(scheduler.status()["reserved_steps"], 8)

    def test_max_attempts_stops_even_when_budget_remains(self):
        scheduler = self.create(families=(1,), max_attempts=2)
        for _ in range(2):
            finish(scheduler, reserve(scheduler))
        self.assertIsNone(reserve(scheduler))
        self.assertGreater(scheduler.status()["remaining_steps"], 0)

    def test_target_accepted_proof_or_disproof_stops_family_not_other_family(self):
        polarities = set()
        for seed in range(20):
            scheduler = self.create(f"run{seed}", seed=seed)
            for _ in range(4):
                intent = reserve(scheduler)
                if intent["proposal"]["problem_id"] == "p0_0":
                    polarities.add(intent["proposal"]["polarity"])
                    finish(scheduler, intent, "accepted")
                    break
                finish(scheduler, intent)
            self.assertIn("f0", scheduler.status()["terminal_families"])
            next_attempt = reserve(scheduler)
            self.assertEqual(next_attempt["proposal"]["family_id"], "f1")
            if len(polarities) == 2:
                break
        self.assertEqual(polarities, {"prove", "disprove"})

    def test_target_stop_does_not_cancel_already_inflight_variant(self):
        scheduler = self.create(families=(2,))
        both = [reserve(scheduler), reserve(scheduler)]
        target = next(i for i in both if i["proposal"]["problem_id"] == "p0_0")
        variant = next(i for i in both if i is not target)
        finish(scheduler, target, "accepted")
        self.assertIsNone(reserve(scheduler))
        finish(scheduler, variant)
        self.assertEqual(scheduler.status()["pending_attempts"], [])
        self.assertIsNone(reserve(scheduler))

    def test_disproved_variant_excluded_without_stopping_target(self):
        for seed in range(50):
            scheduler = self.create(f"run{seed}", families=(2,), seed=seed)
            both = [reserve(scheduler), reserve(scheduler)]
            variant = next(i for i in both if i["proposal"]["problem_id"] == "p0_1")
            if variant["proposal"]["polarity"] != "disprove":
                continue
            finish(scheduler, variant, "accepted")
            target = next(i for i in both if i is not variant); finish(scheduler, target)
            self.assertEqual(scheduler.status()["terminal_families"], {})
            self.assertTrue(scheduler.status()["problems"]["p0_1"]["disproved"])
            self.assertEqual(reserve(scheduler)["proposal"]["problem_id"], "p0_0")
            return
        self.fail("fixture did not cover disproof")

    def test_unknown_absorbing_blocks_does_not_count_failure(self):
        scheduler = self.create()
        first, second = reserve(scheduler), reserve(scheduler)
        finish(scheduler, first, "unknown")
        finish(scheduler, second)
        status = scheduler.status(); pid = first["proposal"]["problem_id"]
        self.assertTrue(status["blocked"])
        self.assertEqual(status["problems"][pid]["known_results"], 0)
        self.assertEqual(status["problems"][pid]["next_budget_steps"], 8)
        with self.assertRaises(mm.MatchmakerBlocked):
            reserve(scheduler)
        with self.assertRaises(mm.MatchmakerError):
            finish(scheduler, first)
        saved_pin = scheduler.run_sha256; scheduler.close()
        restored = self.reopen("run", saved_pin)
        with self.assertRaises(mm.MatchmakerBlocked):
            reserve(restored)

    def test_recovered_intent_never_automatically_resubmitted(self):
        scheduler = self.create()
        first, second = reserve(scheduler), reserve(scheduler)
        run_pin = scheduler.run_sha256; scheduler.close()
        restored = self.reopen("run", run_pin)
        self.assertEqual(len(restored.status()["recovered_pending"]), 2)
        with self.assertRaises(mm.MatchmakerBlocked):
            reserve(restored)
        finish(restored, first)
        with self.assertRaises(mm.MatchmakerBlocked):
            reserve(restored)
        finish(restored, second)
        self.assertFalse(restored.status()["blocked"])
        self.assertEqual(reserve(restored)["proposal"]["attempt_index"], 3)

    def test_result_identity_and_verification_evidence_mandatory(self):
        scheduler = self.create()
        intent = reserve(scheduler); good = terminal(intent, "accepted")
        mutations = [lambda r: r.update(outcome="solved_pending"), lambda r: r.update(intent_sha256=pin("wrong")),
            lambda r: r.update(attempt_id="other"), lambda r: r["evidence"].update(verdict="wrong"),
            lambda r: r["evidence"].update(problem_sha256=pin("wrong")),
            lambda r: r["evidence"].update(attempted_prop_sha256=pin("wrong")),
            lambda r: r["evidence"].update(execution_source_sha256=pin("wrong")),
            lambda r: r["evidence"].update(model_release_sha256=pin("wrong")),
            lambda r: r["evidence"].update(proof_sha256=None), lambda r: r["evidence"].pop("dataset_receipt_sha256"),
            lambda r: r["evidence"].update(actual_steps=True), lambda r: r["evidence"].update(actual_steps=9)]
        for change in mutations:
            with self.subTest(change=change):
                bad = deepcopy(good); change(bad)
                with self.assertRaises(mm.MatchmakerError):
                    scheduler.record_result(intent["proposal"]["attempt_id"], bad)
                self.assertEqual(scheduler.status()["event_count"], 1)
        finish(scheduler, intent, "accepted")

    def test_duplicate_result_is_read_only_and_results_never_rewrite_prefix(self):
        scheduler = self.create()
        first = reserve(scheduler); event_path = self.root/"run/events/00000001.json"
        before = event_path.read_bytes(); info = event_path.stat()
        result = terminal(first)
        receipt = scheduler.record_result(result["attempt_id"], result)
        files = sorted((self.root/"run/events").iterdir())
        with patch.object(mm, "_publish", side_effect=AssertionError("must not write duplicate")):
            self.assertEqual(scheduler.record_result(result["attempt_id"], deepcopy(result)), receipt)
        self.assertEqual(files, sorted((self.root/"run/events").iterdir()))
        for _ in range(3):
            finish(scheduler, reserve(scheduler))
        self.assertEqual(event_path.read_bytes(), before)
        self.assertEqual(event_path.stat().st_mtime_ns, info.st_mtime_ns)
        self.assertEqual(len(list((self.root/"run/events").glob("*.json"))), 8)

    def test_out_of_order_completion_recovery_preserves_exact_history(self):
        scheduler = self.create()
        first, second = reserve(scheduler), reserve(scheduler)
        finish(scheduler, second); third = reserve(scheduler)
        finish(scheduler, first); finish(scheduler, third)
        before = scheduler.status(); expected = scheduler.plan_next(RELEASE); run_pin = scheduler.run_sha256
        scheduler.close(); restored = self.reopen("run", run_pin)
        self.assertEqual(restored.status(), before)
        self.assertEqual(restored.plan_next(RELEASE), expected)

    def test_single_writer_lease_and_existing_root_refused(self):
        scheduler = self.create()
        with self.assertRaises((OSError, mm.MatchmakerBlocked)):
            mm.Matchmaker.open(self.root/"run", scheduler.run_sha256)
        with self.assertRaises(FileExistsError):
            mm.Matchmaker.create(self.root/"run", curriculum(), config())
        scheduler.close()
        self.reopen("run", scheduler.run_sha256)

    def test_write_failure_never_returns_intent_or_retries_in_same_process(self):
        scheduler = self.create()
        proposal = scheduler.plan_next(RELEASE)
        with patch.object(mm.os, "link", side_effect=OSError("unknown publication")):
            with self.assertRaises(OSError):
                scheduler.reserve(proposal, source(proposal))
        with self.assertRaises(mm.MatchmakerBlocked):
            reserve(scheduler)
        run_pin = scheduler.run_sha256; scheduler.close()
        # Orphan staging is ambiguous on recovery; never silently discard it.
        with self.assertRaises((OSError, mm.MatchmakerError)):
            self.reopen("run", run_pin)

    def test_post_publication_error_recovery_sees_pending_intent(self):
        scheduler = self.create()
        real = mm._publish
        def published_then_failed(directory, name, value):
            real(directory, name, value)
            raise OSError("directory durability result unknown")
        with patch.object(mm, "_publish", side_effect=published_then_failed), self.assertRaises(OSError):
            reserve(scheduler)
        run_pin = scheduler.run_sha256; scheduler.close()
        restored = self.reopen("run", run_pin)
        self.assertEqual(restored.status()["attempt_count"], 1)
        with self.assertRaises(mm.MatchmakerBlocked):
            reserve(restored)

    def test_no_replace_even_when_target_appears_at_commit(self):
        scheduler = self.create()
        path = self.root/"run/events/00000001.json"
        path.write_bytes(b"existing incomplete evidence")
        with self.assertRaises(FileExistsError):
            reserve(scheduler)
        self.assertEqual(path.read_bytes(), b"existing incomplete evidence")
        run_pin = scheduler.run_sha256; scheduler.close()
        with self.assertRaises(mm.MatchmakerError):
            self.reopen("run", run_pin)

    def test_corruption_noncanonical_missing_event_and_wrong_run_refused(self):
        for case in ("noncanonical", "chain", "gap", "run", "tail"):
            scheduler = self.create(case); finish(scheduler, reserve(scheduler))
            run_pin = scheduler.run_sha256; scheduler.close()
            event = self.root/case/"events/00000001.json"
            if case == "noncanonical":
                event.write_bytes(event.read_bytes()+b"\n")
            elif case == "chain":
                value = json.loads(event.read_bytes()); value["previous_event_sha256"] = pin("wrong")
                event.write_bytes(mm.canonical_bytes(value))
            elif case == "gap":
                event.unlink()
            elif case == "tail":
                (self.root/case/"events/00000002.json").unlink()
            else:
                run_pin = pin("wrong")
            with self.subTest(case=case), self.assertRaises((OSError, mm.MatchmakerError)):
                self.reopen(case, run_pin)

    def test_symlink_root_or_event_rejected(self):
        scheduler = self.create(); finish(scheduler, reserve(scheduler))
        run_pin = scheduler.run_sha256; scheduler.close()
        alias = self.root/"alias"
        try:
            alias.symlink_to(self.root/"run", target_is_directory=True)
        except OSError:
            self.skipTest("host cannot create symlink without privilege")
        with self.assertRaises((OSError, ValueError)):
            mm.Matchmaker.open(alias, run_pin)
        event = self.root/"run/events/00000001.json"
        content = event.read_bytes(); event.unlink()
        foreign = self.root/"foreign.json"; foreign.write_bytes(content)
        event.symlink_to(foreign)
        with self.assertRaises((OSError, ValueError)):
            mm.Matchmaker.open(self.root/"run", run_pin)

    def test_closed_instance_refuses_operations(self):
        scheduler = self.create(); scheduler.close()
        for method in (lambda: scheduler.status(), lambda: reserve(scheduler)):
            with self.assertRaises(mm.MatchmakerError):
                method()


if __name__ == "__main__":
    unittest.main()
