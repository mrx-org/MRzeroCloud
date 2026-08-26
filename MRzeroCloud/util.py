"""MRzeroCore-compatible utility functions for cloud simulation."""

from __future__ import annotations

import asyncio
import threading

import numpy as np

from .api import pipeline
from .api._config import optional_config
from .api._tools import abort_context, stop_simulation
from .simulation import SimulationJob


def _simulate_impl(
    seq,
    phantom=None,
    *,
    accuracy: float = 1e-5,
    noise_level: float | None = None,
    backend: str | None = None,
    config: dict | None = None,
    t1: float = 1.0,
    t2: float = 0.1,
    min_mag: float = 1e-3,
    use_gpu: bool | None = None,
    exact_trajectories: bool | None = None,
    worker: str | None = None,
):
    del phantom, t1, t2, min_mag
    config = config if config is not None else optional_config()

    if backend is None:
        backend = "modal"
    if str(backend).lower() != "modal":
        raise ValueError(
            f"Unknown simulation backend {backend!r}; only 'modal' is supported"
        )

    kdata, ktraj = pipeline.run(
        seq,
        config=config,
        accuracy=accuracy,
        use_gpu=use_gpu,
        exact_trajectories=exact_trajectories,
        worker=worker,
    )

    if noise_level:
        noise = noise_level * (
            np.random.randn(*kdata.shape) + 1j * np.random.randn(*kdata.shape)
        )
        kdata = kdata + noise.astype(np.complex64)

    return kdata.astype(np.complex64), ktraj.astype(np.float32)


class _SimulateAPI:
    """Callable API with blocking ``simulate(seq)`` and ``simulate.start(seq)``."""

    def __call__(
        self,
        seq,
        phantom=None,
        *,
        accuracy: float = 1e-5,
        noise_level: float | None = None,
        backend: str | None = None,
        config: dict | None = None,
        t1: float = 1.0,
        t2: float = 0.1,
        min_mag: float = 1e-3,
        use_gpu: bool | None = None,
        exact_trajectories: bool | None = None,
        worker: str | None = None,
    ):
        """Simulate a Pulseq sequence over modal HTTP (blocking).

        Parameters
        ----------
        backend:
            Only ``"modal"`` is supported (the default).
        worker:
            Modal worker tier (``cpu``, ``t4``, ``a10g``, ``a100``).
        use_gpu:
            When ``worker`` is omitted, maps to a GPU tier.
        exact_trajectories:
            Use exact k-space trajectories (default ``True``).

        Returns
        -------
        (np.ndarray, np.ndarray)
            Complex ADC signal and k-space trajectory (``complex64``, ``float32``).
        """
        abort = threading.Event()
        with abort_context(abort):
            return _simulate_impl(
                seq,
                phantom,
                accuracy=accuracy,
                noise_level=noise_level,
                backend=backend,
                config=config,
                t1=t1,
                t2=t2,
                min_mag=min_mag,
                use_gpu=use_gpu,
                exact_trajectories=exact_trajectories,
                worker=worker,
            )

    def start(
        self,
        seq,
        phantom=None,
        *,
        accuracy: float = 1e-5,
        noise_level: float | None = None,
        backend: str | None = None,
        config: dict | None = None,
        t1: float = 1.0,
        t2: float = 0.1,
        min_mag: float = 1e-3,
        use_gpu: bool | None = None,
        exact_trajectories: bool | None = None,
        worker: str | None = None,
    ) -> SimulationJob:
        """Start a non-blocking cloud simulation.

        Returns
        -------
        SimulationJob
            Call :meth:`SimulationJob.stop` or :func:`stop_simulation` to cancel,
            then :meth:`SimulationJob.result` for ``(signal, ktraj)``.
        """
        return SimulationJob.launch(
            _simulate_impl,
            seq,
            phantom,
            accuracy=accuracy,
            noise_level=noise_level,
            backend=backend,
            config=config,
            t1=t1,
            t2=t2,
            min_mag=min_mag,
            use_gpu=use_gpu,
            exact_trajectories=exact_trajectories,
            worker=worker,
        )


simulate = _SimulateAPI()


async def simulate_async(
    seq,
    phantom=None,
    *,
    accuracy: float = 1e-5,
    noise_level: float | None = None,
    backend: str | None = None,
    config: dict | None = None,
    t1: float = 1.0,
    t2: float = 0.1,
    min_mag: float = 1e-3,
    use_gpu: bool | None = None,
    exact_trajectories: bool | None = None,
    worker: str | None = None,
):
    """Await a blocking simulation (runs :meth:`simulate.start` in a worker thread)."""
    job = simulate.start(
        seq,
        phantom,
        accuracy=accuracy,
        noise_level=noise_level,
        backend=backend,
        config=config,
        t1=t1,
        t2=t2,
        min_mag=min_mag,
        use_gpu=use_gpu,
        exact_trajectories=exact_trajectories,
        worker=worker,
    )
    return await asyncio.to_thread(job.result)


def imshow(data, *args, **kwargs):
    """Display image data with matplotlib (transposed, origin lower)."""
    import matplotlib.pyplot as plt

    kwargs.setdefault("origin", "lower")
    arr = np.squeeze(np.asarray(data))
    if arr.ndim == 2:
        arr = arr.T
    return plt.imshow(arr, *args, **kwargs)
