"""Background cloud simulation jobs."""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np


class SimulationJob:
    """Handle for a non-blocking :func:`~MRzeroCloud.util.simulate` run."""

    def __init__(self) -> None:
        self._abort = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._result: tuple[np.ndarray, np.ndarray] | None = None
        self._error: BaseException | None = None

    @property
    def abort(self) -> threading.Event:
        return self._abort

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def stop(self) -> None:
        """Request abort on the next modal progress message."""
        self._abort.set()

    def cancel_after(self, seconds: float) -> None:
        """Call :meth:`stop` after ``seconds`` (runs on a background timer)."""
        def _timer() -> None:
            if self._done.wait(timeout=seconds):
                return
            self.stop()

        threading.Thread(target=_timer, daemon=True).start()

    def result(self, timeout: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Wait for completion and return ``(signal, ktraj)``."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if not self._done.wait(timeout=timeout):
            raise TimeoutError("simulation did not finish before timeout")
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    @classmethod
    def launch(
        cls,
        target: Callable[..., tuple[np.ndarray, np.ndarray]],
        *args,
        **kwargs,
    ) -> SimulationJob:
        from .api._tools import abort_context

        job = cls()

        def worker() -> None:
            try:
                with abort_context(job._abort):
                    job._result = target(*args, **kwargs)
            except BaseException as exc:
                job._error = exc
            finally:
                job._done.set()

        job._thread = threading.Thread(target=worker, daemon=True)
        job._thread.start()
        return job
