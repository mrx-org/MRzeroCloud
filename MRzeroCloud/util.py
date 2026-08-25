"""MRzeroCore-compatible utility functions for cloud simulation."""

from __future__ import annotations

import asyncio
import threading
import warnings
from typing import Sequence

import numpy as np

from .api import pipeline
from .api._config import (
    DEFAULT_BACKEND,
    backend_from_config,
    default_config,
    default_modal_config,
    optional_config,
    phantom_id_from_config,
    phantomlib_params_from_config,
)
from .api._tools import abort_context, stop_simulation
from .simulation import SimulationJob

_accuracy_warned = False


def _normalize_affine(affine):
    if affine is None:
        return None
    if isinstance(affine[0], (list, tuple)):
        return [[float(v) for v in row] for row in affine]
    flat = [float(v) for v in affine]
    if len(flat) >= 12:
        return [
            flat[0:4],
            flat[4:8],
            flat[8:12],
        ]
    raise ValueError("affine must be 3x4 matrix or flat 12-element list")


def load_phantom(
    id=None,
    *,
    affine: Sequence[float] | Sequence[Sequence[float]] | None = None,
    res: Sequence[int] | None = None,
    config: dict | None = None,
):
    """Load a BrainWeb phantom via phantomlib.

    Parameters
    ----------
    id:
        BrainWeb subject id (``int``) or ``phantomlib_subject4_...`` string.
    affine:
        3×4 FOV/slice placement matrix, or flat 12-element list.
    res:
        Phantom grid size ``(res_x, res_y, res_z)``.
    config:
        Optional flat config dict from :func:`mr0.api.load_config`.

    Returns
    -------
    object
        Cloud phantom handle for :func:`simulate`.
    """
    if config is not None and id is None:
        id = config.get("phantom")
    if config is not None:
        params = phantomlib_params_from_config(config)
    else:
        params = phantomlib_params_from_config(optional_config())
    if id is None:
        id = phantom_id_from_config(config)
    if affine is None:
        affine = params["affine"]
    else:
        affine = _normalize_affine(affine)
    if res is None:
        res = (params["res_x"], params["res_y"], params["res_z"])
    return pipeline.load_phantom_toolapi(id, affine, res)


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
    global _accuracy_warned
    config = config if config is not None else optional_config()

    if backend is None:
        backend = backend_from_config(config) if config is not None else DEFAULT_BACKEND

    is_modal = str(backend).lower() == "modal"

    if config is None:
        seq_def = seq if hasattr(seq, "get_definition") else None
        config = default_modal_config(seq_def) if is_modal else default_config(seq_def)

    if accuracy != 1e-5 and not _accuracy_warned and not is_modal:
        warnings.warn(
            "MRzeroCloud simulate() accepts accuracy for API parity; "
            "Fly simulation tools do not expose this parameter yet.",
            stacklevel=3,
        )
        _accuracy_warned = True

    if is_modal:
        kdata, ktraj = pipeline.run(
            backend,
            None,
            None,
            config=config,
            seq=seq,
            accuracy=accuracy,
            t1=t1,
            t2=t2,
            min_mag=min_mag,
            use_gpu=use_gpu,
            exact_trajectories=exact_trajectories,
            worker=worker,
        )
    else:
        if phantom is None:
            phantom = load_phantom(config=config)
        seq0 = pipeline.load_seq(seq)
        kdata, ktraj = pipeline.run(
            backend,
            seq0,
            phantom,
            config=config,
            t1=t1,
            t2=t2,
            min_mag=min_mag,
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
        """Simulate a Pulseq sequence in the cloud (blocking).

        Parameters
        ----------
        backend:
            ``"mr0sim"`` (default Fly ToolAPI chain) or ``"modal"`` (HTTP job API).
        worker:
            Modal worker tier for ``backend="modal"`` (``cpu``, ``t4``, ``a10g``, ``a100``).
        use_gpu:
            When ``worker`` is omitted, maps to a GPU tier on the modal backend.
        exact_trajectories:
            Use exact k-space trajectories on the modal backend (default ``True``).

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
