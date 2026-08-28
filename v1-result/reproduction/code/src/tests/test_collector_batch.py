from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from cpu_runtime import collector_batch as cb
from cpu_runtime import matchmaker as mm
from tests.test_matchmaker import RELEASE, config, curriculum, pin, source, terminal


def search_response(intent, outcome="exhausted"):
    proposal = intent["proposal"]
    return {"schema_version": "reap.collector-batch.search.v1", "attempt_id": proposal["attempt_id"],
        "intent_sha256": intent["intent_sha256"], "outcome": outcome, "artifacts_complete": True,
        "artifacts_sha256": pin("artifacts-"+proposal["attempt_id"]),
        "terminal_receipt_sha256": pin("terminal-"+proposal["attempt_id"]),
        "actual_steps": proposal["budget_steps"], "policy_version": 0,
        "raw": {"fixture_only": True, "real_Lean_or_GPU": False, "outcome": outcome}}


def retirement(intent, response):
    return {"schema_version": "reap.gpu.retirement.v1", "session_id": intent["proposal"]["attempt_id"],
        "policy_version": 0, "snapshot": cb.RETIRE_SNAPSHOT, "snapshot_sha256": pin("snapshot"),
        "status": "released", "mutation_retry_allowed": False, "tombstone_scope": "current_runtime"}


def verified(intent, response):
    result = terminal(intent, "accepted")
    result["evidence"]["actual_steps"] = response["actual_steps"]
    return result


class CollectorBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def scheduler(self, name="scheduler", **updates):
        scheduler = mm.Matchmaker.create(self.root/name, curriculum(), config(max_inflight=4, **updates))
        self.addCleanup(scheduler.close)
        return scheduler

    def run_batch(self, scheduler, **callbacks):
        defaults = dict(prepare_source=source, provide_release=lambda: RELEASE,
                        search=search_response, retire=retirement, verify=verified)
        defaults.update(callbacks)
        return cb.run_collector_batch(scheduler, self.root/"batch", **defaults)

    def test_twelve_attempts_small_global_budget_and_complete_immutable_records(self):
        scheduler = self.scheduler(base_steps=1, cap_steps=2, total_steps=24, max_attempts=12)
        report = self.run_batch(scheduler)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["exhausted"]), 12)
        self.assertEqual(scheduler.status()["reserved_steps"], 20)
        self.assertFalse(report["independent_GPU_or_Lean_verification_performed_by_executor"])
        for aid in report["started"]:
            directory = self.root/"batch/attempts"/aid
            self.assertEqual(json.loads((directory/"retirement-response.json").read_bytes())["status"], "released")
            self.assertEqual(json.loads((directory/"terminal-result.json").read_bytes())["outcome"], "exhausted")
            self.assertFalse((directory/"validation-response.json").exists())
        self.assertLessEqual(report["metrics"]["max_search_stage_occupancy"], 2)

    def test_reservation_and_local_intent_exist_before_callback(self):
        scheduler = self.scheduler(max_attempts=3)
        def checked_search(intent):
            aid = intent["proposal"]["attempt_id"]
            self.assertEqual(scheduler.status()["attempts"][aid]["intent"], intent)
            self.assertEqual(json.loads((self.root/"batch/attempts"/aid/"intent.json").read_bytes()), intent)
            self.assertIsNone(scheduler.status()["attempts"][aid]["result"])
            return search_response(intent)
        report = self.run_batch(scheduler, search=checked_search)
        self.assertEqual(len(report["exhausted"]), 3)

    def test_gpu_search_slot_held_until_retirement_confirmation(self):
        scheduler = self.scheduler(max_attempts=3)
        retired = threading.Event(); calls = []
        def search(intent):
            calls.append(intent["proposal"]["attempt_index"])
            if len(calls) > 1:
                self.assertTrue(retired.is_set())
            return search_response(intent)
        def retire(intent, result):
            self.assertEqual(len(calls), intent["proposal"]["attempt_index"])
            time.sleep(0.01); retired.set()
            return retirement(intent, result)
        report = self.run_batch(scheduler, search_workers=1, search=search, retire=retire)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(calls, [1, 2, 3])

    def test_search_overlaps_independent_verification_and_keeps_release_pins(self):
        scheduler = self.scheduler(max_attempts=6, seed=0)
        verifying = threading.Event(); overlap = threading.Event(); selected = []
        def release():
            value = pin(f"release-{len(selected)}"); selected.append(value); return value
        def search(intent):
            if verifying.is_set():
                overlap.set()
            p = intent["proposal"]
            # A real successful variant in this simulated callback contract;
            # all target attempts remain known exhausted, so the run continues.
            return search_response(intent, "solved_candidate" if p["problem_id"] == "p0_1" else "exhausted")
        def verify(intent, response):
            verifying.set()
            self.assertTrue(overlap.wait(3), "new search did not overlap verification")
            return verified(intent, response)
        report = self.run_batch(scheduler, provide_release=release, search=search, verify=verify)
        self.assertEqual(report["status"], "completed")
        self.assertTrue(overlap.is_set())
        self.assertTrue(report["accepted"])
        actual = [scheduler.status()["attempts"][aid]["intent"]["proposal"]["model_release_sha256"] for aid in report["started"]]
        # The provider is read-only and may be queried when no proposal remains.
        self.assertEqual(len(set(actual)), len(actual))
        self.assertEqual(actual, selected[:len(actual)])

    def test_pending_cap_and_completed_search_handoff_backpressure(self):
        scheduler = self.scheduler(max_attempts=4)
        barrier = threading.Barrier(2)
        def search(intent):
            if intent["proposal"]["attempt_index"] <= 2:
                barrier.wait(timeout=3)
            return search_response(intent, "solved_candidate")
        def verify(intent, response):
            time.sleep(0.05)
            return verified(intent, response)
        report = self.run_batch(scheduler, search=search, verify=verify, max_pending_validation=1)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["metrics"]["max_validation_stage_occupancy"], 1)
        self.assertEqual(report["metrics"]["max_validation_callbacks_active"], 1)
        self.assertTrue(report["metrics"]["handoff_backpressure_observed"])
        self.assertLessEqual(report["metrics"]["max_search_stage_occupancy"], 2)

    def test_search_unknown_stops_new_work_and_existing_work_drains(self):
        scheduler = self.scheduler(max_attempts=12)
        both = threading.Barrier(2); retired = []
        def search(intent):
            both.wait(timeout=3)
            if intent["proposal"]["attempt_index"] == 1:
                raise RuntimeError("SENSITIVE-MUST-NOT-BE-LOGGED")
            time.sleep(0.03)
            return search_response(intent)
        def retire(intent, response):
            retired.append(intent["proposal"]["attempt_id"])
            return retirement(intent, response)
        report = self.run_batch(scheduler, search=search, retire=retire)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(len(report["started"]), 2)
        self.assertEqual(len(report["unknown"]), 1)
        self.assertEqual(len(report["exhausted"]), 1)
        self.assertEqual(retired, report["exhausted"])
        raw = b"".join(path.read_bytes() for path in (self.root/"batch").rglob("*.json"))
        self.assertNotIn(b"SENSITIVE-MUST-NOT-BE-LOGGED", raw)
        state = scheduler.status()
        failed_id = report["unknown"][0]; pid = state["attempts"][failed_id]["intent"]["proposal"]["problem_id"]
        self.assertEqual(state["problems"][pid]["known_results"], 0)

    def test_default_two_pending_slots_are_bounded_under_fast_completed_searches(self):
        scheduler = self.scheduler(max_attempts=4)
        def verify(intent, response):
            time.sleep(0.1)
            return verified(intent, response)
        report = self.run_batch(scheduler, search=lambda intent: search_response(intent, "solved_candidate"), verify=verify)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["metrics"]["max_validation_stage_occupancy"], 2)
        self.assertTrue(report["metrics"]["handoff_backpressure_observed"])
        self.assertLessEqual(len(report["started"]), 4)
        self.assertEqual(report["scheduler"]["pending_attempts"], [])

    def test_retirement_unknown_is_not_retried_and_does_not_call_verifier(self):
        scheduler = self.scheduler(max_attempts=4)
        calls = []
        def retire(intent, response):
            calls.append(intent["proposal"]["attempt_id"])
            self.assertTrue((self.root/"batch/attempts"/calls[-1]/"retirement-intent.json").exists())
            raise TimeoutError("ACK missing")
        def verify(*args):
            self.fail("unreleased search must not enter verification")
        report = self.run_batch(scheduler, search_workers=1,
            search=lambda intent: search_response(intent, "solved_candidate"), retire=retire, verify=verify)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["accepted"], [])
        self.assertEqual(scheduler.status()["attempts"][calls[0]]["result"]["evidence"]["reason"], "transport_unknown")
        with self.assertRaises(cb.CollectorBatchBlocked):
            self.run_batch(scheduler, search=lambda *args: self.fail("must not rerun"))

    def test_wrong_retirement_identity_and_prepared_receipt_never_release_slot(self):
        for key, value in (("status", "prepared"), ("session_id", "wrong"), ("policy_version", True),
                           ("snapshot", "after"), ("schema_version", "reap.gpu.retirement.v2")):
            with self.subTest(key=key):
                scheduler = self.scheduler(name="s"+key, max_attempts=1)
                def retire(intent, response):
                    receipt = retirement(intent, response); receipt[key] = value; return receipt
                output = self.root/("batch-"+key)
                report = cb.run_collector_batch(scheduler, output, prepare_source=source, provide_release=lambda: RELEASE,
                    search=search_response, retire=retire, verify=verified)
                self.assertEqual(report["status"], "blocked")
                self.assertEqual(len(report["unknown"]), 1)

    def test_incomplete_or_mismatched_terminal_never_calls_retire(self):
        for field, value in (("artifacts_complete", False), ("policy_version", 1), ("attempt_id", "wrong"),
                             ("actual_steps", True), ("outcome", "unknown")):
            with self.subTest(field=field):
                scheduler = self.scheduler(name="s"+field, max_attempts=1)
                def search(intent):
                    result = search_response(intent); result[field] = value; return result
                report = cb.run_collector_batch(scheduler, self.root/("b-"+field), prepare_source=source,
                    provide_release=lambda: RELEASE, search=search, retire=lambda *args: self.fail("must not retire unknown"), verify=verified)
                self.assertEqual(report["status"], "blocked")

    def test_invalid_verifier_claim_not_accepted_or_reclassified_as_exhaustion(self):
        for defect in ("wrong_dataset_pin", "wrong_source", "wrong_steps", "exhausted"):
            scheduler = self.scheduler(name="s-"+defect, max_attempts=1)
            def verify(intent, response):
                result = verified(intent, response)
                if defect == "wrong_dataset_pin":
                    result["evidence"]["dataset_sha256"] = None
                elif defect == "wrong_source":
                    result["evidence"]["execution_source_sha256"] = pin("wrong")
                elif defect == "wrong_steps":
                    result["evidence"]["actual_steps"] += 1
                else:
                    result = terminal(intent)
                return result
            report = cb.run_collector_batch(scheduler, self.root/("b-"+defect), prepare_source=source, provide_release=lambda: RELEASE,
                search=lambda intent: search_response(intent, "solved_candidate"), retire=retirement, verify=verify)
            self.assertEqual(report["accepted"], [])
            self.assertEqual(report["exhausted"], [])
            self.assertEqual(len(report["unknown"]), 1)

    def test_completed_resume_skips_all_callbacks_and_checks_original_receipts(self):
        scheduler = self.scheduler(max_attempts=3)
        first = self.run_batch(scheduler)
        before = {p.relative_to(self.root/"batch").as_posix(): p.read_bytes()
                  for p in (self.root/"batch").rglob("*") if p.is_file()}
        def no_work(*args):
            self.fail("completed callbacks must not be repeated")
        second = self.run_batch(scheduler, prepare_source=no_work, search=no_work, retire=no_work, verify=no_work)
        self.assertEqual(second["started"], [])
        self.assertEqual(second["skipped_completed"], first["started"])
        after = {p.relative_to(self.root/"batch").as_posix(): p.read_bytes()
                 for p in (self.root/"batch").rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        path = self.root/"batch/attempts"/first["started"][0]/"retirement-response.json"
        value = json.loads(path.read_bytes()); value["status"] = "prepared"
        path.write_bytes(mm.canonical_bytes(value))
        with self.assertRaises(mm.MatchmakerError):
            self.run_batch(scheduler, search=no_work)

    def test_pending_restart_refuses_every_callback_before_touching_output(self):
        scheduler = self.scheduler()
        proposal = scheduler.plan_next(RELEASE); scheduler.reserve(proposal, source(proposal))
        run_pin = scheduler.run_sha256; scheduler.close()
        restored = mm.Matchmaker.open(self.root/"scheduler", run_pin); self.addCleanup(restored.close)
        def no_work(*args):
            self.fail("pending intent must not invoke callbacks")
        with self.assertRaises(cb.CollectorBatchBlocked):
            self.run_batch(restored, prepare_source=no_work, provide_release=no_work, search=no_work, retire=no_work, verify=no_work)
        self.assertFalse((self.root/"batch").exists())

    def test_resume_rejects_bool_coercion_in_immutable_intent(self):
        scheduler = self.scheduler(base_steps=1, cap_steps=1, max_attempts=1)
        report = self.run_batch(scheduler)
        path = self.root/"batch/attempts"/report["started"][0]/"intent.json"
        value = json.loads(path.read_bytes()); value["proposal"]["budget_steps"] = True
        path.write_bytes(mm.canonical_bytes(value))
        with self.assertRaises(mm.MatchmakerError):
            self.run_batch(scheduler)

    def test_oversize_callback_output_fails_closed_before_retirement(self):
        scheduler = self.scheduler(max_attempts=1)
        def search(intent):
            result = search_response(intent); result["raw"]["oversized"] = "x" * (1024*1024)
            return result
        report = self.run_batch(scheduler, search=search, retire=lambda *args: self.fail("must not retire without durable terminal"))
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(len(report["unknown"]), 1)

    def test_prepare_failure_does_not_reserve_and_prevents_automatic_resume(self):
        scheduler = self.scheduler()
        def prepare(*args):
            raise RuntimeError("compile failed")
        report = self.run_batch(scheduler, prepare_source=prepare)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(scheduler.status()["attempt_count"], 0)
        with self.assertRaises(mm.MatchmakerError):
            self.run_batch(scheduler)

    def test_storage_failure_after_search_retains_pending_and_never_retire_retries(self):
        scheduler = self.scheduler(max_attempts=1)
        real = cb._Output.write
        def broken(output, aid, name, value):
            if name in {"search-response.json", "diagnostic.json"}:
                raise OSError("disk failure")
            return real(output, aid, name, value)
        with patch.object(cb._Output, "write", broken):
            report = self.run_batch(scheduler, retire=lambda *args: self.fail("unknown search must not retire"))
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(len(scheduler.status()["pending_attempts"]), 1)
        with self.assertRaises(cb.CollectorBatchBlocked):
            self.run_batch(scheduler)

    def test_unknown_verifier_result_is_preserved_and_stops_dispatch(self):
        scheduler = self.scheduler(max_attempts=1)
        report = self.run_batch(scheduler, search=lambda intent: search_response(intent, "solved_candidate"),
                                verify=lambda intent, response: terminal(intent, "unknown"))
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(len(report["unknown"]), 1)

    def test_invalid_worker_bounds_refused_before_callbacks(self):
        scheduler = self.scheduler()
        for kwargs in ({"search_workers": 3}, {"search_workers": True}, {"validation_workers": 2}, {"max_pending_validation": 3}):
            with self.assertRaises(mm.MatchmakerError):
                self.run_batch(scheduler, **kwargs)
        self.assertFalse((self.root/"batch").exists())


if __name__ == "__main__":
    unittest.main()
