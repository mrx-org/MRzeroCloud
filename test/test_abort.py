"""Tests for simulation abort handling."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from MRzeroCloud import api
from MRzeroCloud.api._tools import abort_context, call_tool, stop_simulation
from MRzeroCloud.exceptions import SimulationAborted
from MRzeroCloud.util import simulate


class AbortFlagTests(unittest.TestCase):
    def test_on_message_returns_false_when_job_aborted(self):
        abort = __import__("threading").Event()
        with abort_context(abort):
            from MRzeroCloud.api._tools import _resolve_on_message

            on_message = _resolve_on_message()
            self.assertTrue(on_message("working"))
            abort.set()
            self.assertFalse(on_message("still working"))

    def test_call_tool_raises_simulation_aborted(self):
        abort = threading.Event()

        def slow_call(url, kwargs, on_message):
            self.assertTrue(on_message("step 1"))
            abort.set()
            self.assertFalse(on_message("step 2"))
            raise RuntimeError("ToolCallError: client requested abort in on_message")

        with patch("MRzeroCloud.api._tools.toolapi.call", side_effect=slow_call):
            with abort_context(abort):
                with self.assertRaises(SimulationAborted):
                    call_tool("conseq", seq_file="# dummy")

    def test_job_stop_method(self):
        from MRzeroCloud.simulation import SimulationJob

        job = SimulationJob()
        self.assertFalse(job.abort.is_set())
        job.stop()
        self.assertTrue(job.abort.is_set())

    def test_job_stop_aborts_background_simulation(self):
        api.configure(verbose=False)

        def slow_call(url, kwargs, on_message):
            if not on_message("start"):
                raise RuntimeError("ToolCallError: OnMessageAbort")
            time.sleep(0.15)
            if not on_message("middle"):
                raise RuntimeError("ToolCallError: OnMessageAbort")
            time.sleep(0.15)
            on_message("done")
            return {"Ok": {"signal": {"TypedList": {"Complex": {"real": [1.0], "imag": [0.0]}}}}}

        with patch("MRzeroCloud.api._tools.toolapi.call", side_effect=slow_call):
            with patch("MRzeroCloud.api.pipeline.load_seq", return_value={"sequence": True}):
                with patch("MRzeroCloud.util.load_phantom", return_value={"phantom": True}):
                    with patch(
                        "MRzeroCloud.api.pipeline.trajex",
                        return_value=__import__("numpy").zeros((4, 3), dtype=float),
                    ):
                        job = simulate.start("gre.seq", backend="mr0sim")
                        time.sleep(0.05)
                        stop_simulation(job)
                        with self.assertRaises(SimulationAborted):
                            job.result(timeout=2.0)


class ThreadedStopTests(unittest.TestCase):
    def test_stop_from_main_thread_while_job_runs(self):
        api.configure(verbose=False)

        def slow_call(url, kwargs, on_message):
            for step in ("a", "b", "c", "d"):
                if not on_message(step):
                    raise RuntimeError("ToolCallError: OnMessageAbort")
                time.sleep(0.02)
            return {"Ok": True}

        with patch("MRzeroCloud.api._tools.toolapi.call", side_effect=slow_call):
            with patch("MRzeroCloud.api.pipeline.load_seq", return_value={"sequence": True}):
                with patch("MRzeroCloud.util.load_phantom", return_value={"phantom": True}):
                    with patch(
                        "MRzeroCloud.api.pipeline.trajex",
                        return_value=__import__("numpy").zeros((4, 3), dtype=float),
                    ):
                        with patch(
                            "MRzeroCloud.api._convert.signal_to_complex_np",
                            return_value=__import__("numpy").array([1 + 0j], dtype=complex),
                        ):
                            job = simulate.start("gre.seq", backend="mr0sim")
                            time.sleep(0.05)
                            job.stop()
                            with self.assertRaises(SimulationAborted):
                                job.result(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
