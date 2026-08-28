"""Actor scheduling/lifecycle tests; host threads and clocks, not GPU kernels."""
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import patch

from gpu_runtime.actor import GpuActor


class GpuActorTests(unittest.TestCase):
    def test_fifo_wait_and_execution_metrics_while_busy(self):
        actor = GpuActor()
        self.addCleanup(actor.close)
        entered, release, accepted = threading.Event(), threading.Event(), threading.Event()
        clock = [10.0]
        order = []
        original_put = actor._queue.put

        def put(item):
            original_put(item)
            if item is not None and item.submitted_at == 12:
                accepted.set()

        def first():
            order.append("first")
            entered.set()
            self.assertTrue(release.wait(5))
            return "a"

        def second():
            order.append("second")
            clock[0] = 18.0
            return "b"

        with patch("gpu_runtime.actor.monotonic", side_effect=lambda: clock[0]), \
                patch.object(actor._queue, "put", side_effect=put), ThreadPoolExecutor(2) as pool:
            one = pool.submit(actor.submit, first)
            try:
                self.assertTrue(entered.wait(5))
                clock[0] = 12.0
                two = pool.submit(actor.submit, second)
                self.assertTrue(accepted.wait(5))
                live = actor.metrics()
                self.assertEqual((live["active"], live["queued"], live["submitted"]), (1, 1, 2))
                self.assertEqual(live["active_execution_seconds"], 2)
                self.assertEqual(actor.metrics(), live)  # observation does not enqueue work
                clock[0] = 15.0
            finally:
                release.set()
            self.assertEqual((one.result(5), two.result(5)), ("a", "b"))
        result = actor.metrics()
        self.assertEqual(order, ["first", "second"])
        self.assertEqual((result["completed"], result["failed"], result["queued"], result["active"]), (2, 0, 0, 0))
        self.assertEqual(result["queue_wait_seconds_total"], 3.0)
        self.assertEqual(result["queue_wait_seconds_max"], 3.0)
        self.assertEqual(result["execution_seconds_total"], 8.0)
        self.assertEqual(result["execution_seconds_max"], 5.0)
        self.assertEqual(result["max_active"], 1)

    def test_error_does_not_block_next_call_and_counts_completed_failure(self):
        with GpuActor() as actor:
            error = ValueError("backend failed")

            def fail():
                raise error

            with self.assertRaises(ValueError) as raised:
                actor.submit(fail)
            self.assertIs(raised.exception, error)
            self.assertEqual(actor.submit(lambda: 42), 42)
            metrics = actor.metrics()
            self.assertEqual((metrics["submitted"], metrics["completed"], metrics["failed"]), (2, 2, 1))

    def test_close_drains_accepted_work_and_rejects_new_requests(self):
        actor = GpuActor()
        self.addCleanup(actor.close)
        entered, release, sentinel = threading.Event(), threading.Event(), threading.Event()
        original_put = actor._queue.put

        def put(item):
            original_put(item)
            if item is None:
                sentinel.set()

        def work():
            entered.set()
            self.assertTrue(release.wait(5))
            return "receipt"

        with patch.object(actor._queue, "put", side_effect=put), ThreadPoolExecutor(3) as pool:
            result = pool.submit(actor.submit, work)
            try:
                self.assertTrue(entered.wait(5))
                close_one = pool.submit(actor.close)
                self.assertTrue(sentinel.wait(5))
                close_two = pool.submit(actor.close)
                self.assertFalse(close_one.done())
                self.assertFalse(close_two.done())
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    actor.submit(lambda: self.fail("post-close mutation executed"))
            finally:
                release.set()
            self.assertEqual(result.result(5), "receipt")
            close_one.result(5)
            close_two.result(5)
        self.assertFalse(actor.metrics()["accepting"])
        self.assertEqual(actor.metrics()["completed"], 1)

    def test_recursive_submit_or_close_fails_without_deadlock(self):
        with GpuActor() as actor:
            with self.assertRaisesRegex(RuntimeError, "recursively"):
                actor.submit(lambda: actor.submit(lambda: None))
            with self.assertRaisesRegex(RuntimeError, "close from"):
                actor.submit(actor.close)
            self.assertEqual(actor.submit(lambda: "healthy"), "healthy")


if __name__ == "__main__":
    unittest.main()
