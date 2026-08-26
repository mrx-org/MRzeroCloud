"""Tests for simulation abort handling."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from MRzeroCloud import api
from MRzeroCloud.api._tools import abort_context, stop_simulation
from MRzeroCloud.exceptions import SimulationAborted
from MRzeroCloud.util import simulate


def _slow_modal(*_args, **_kwargs):
    from MRzeroCloud.api._tools import _resolve_on_message

    on_message = _resolve_on_message()
    if not on_message("start"):
        raise SimulationAborted("modal simulation aborted by client")
    time.sleep(0.15)
    if not on_message("middle"):
        raise SimulationAborted("modal simulation aborted by client")
    time.sleep(0.15)
    on_message("done")
    return np.array([1 + 0j], dtype=np.complex64), np.zeros((1, 3), dtype=np.float32)


class BackendTests(unittest.TestCase):
    def test_fly_backend_rejected(self):
        with self.assertRaises(ValueError):
            simulate("gre.seq", backend="mr0sim")


class AbortFlagTests(unittest.TestCase):
    def test_on_message_returns_false_when_job_aborted(self):
        abort = threading.Event()
        with abort_context(abort):
            from MRzeroCloud.api._tools import _resolve_on_message

            on_message = _resolve_on_message()
            self.assertTrue(on_message("working"))
            abort.set()
            self.assertFalse(on_message("still working"))

    def test_job_stop_method(self):
        from MRzeroCloud.simulation import SimulationJob

        job = SimulationJob()
        self.assertFalse(job.abort.is_set())
        job.stop()
        self.assertTrue(job.abort.is_set())

    def test_job_stop_aborts_background_simulation(self):
        api.configure(verbose=False)
        with patch("MRzeroCloud.api.pipeline.run", side_effect=_slow_modal):
            job = simulate.start("gre.seq")
            time.sleep(0.05)
            stop_simulation(job)
            with self.assertRaises(SimulationAborted):
                job.result(timeout=2.0)


class ThreadedStopTests(unittest.TestCase):
    def test_stop_from_main_thread_while_job_runs(self):
        api.configure(verbose=False)

        def slow_steps(*_args, **_kwargs):
            from MRzeroCloud.api._tools import _resolve_on_message

            on_message = _resolve_on_message()
            for step in ("a", "b", "c", "d"):
                if not on_message(step):
                    raise SimulationAborted("modal simulation aborted by client")
                time.sleep(0.02)
            return np.array([1 + 0j], dtype=np.complex64), np.zeros((1, 3), dtype=np.float32)

        with patch("MRzeroCloud.api.pipeline.run", side_effect=slow_steps):
            job = simulate.start("gre.seq")
            time.sleep(0.05)
            job.stop()
            with self.assertRaises(SimulationAborted):
                job.result(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
