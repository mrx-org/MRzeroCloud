"""Modal HTTP simulation pipeline."""

from __future__ import annotations

import numpy as np

from ._config import (
    DEFAULT_MODAL_WORKER,
    modal_exact_trajectories_from_config,
    modal_use_gpu_from_config,
    modal_worker_from_config,
    optional_config,
)
from ._modal_http import run_modal_http


def run(
    seq,
    *,
    config: dict | None = None,
    accuracy: float = 1e-5,
    use_gpu: bool | None = None,
    exact_trajectories: bool | None = None,
    worker: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate k-space data and return ``(kdata, ktraj)``."""
    cfg = config or optional_config() or {}
    if exact_trajectories is None:
        exact_trajectories = modal_exact_trajectories_from_config(cfg)
    if not worker and use_gpu is not None:
        worker = DEFAULT_MODAL_WORKER if use_gpu else "cpu"
    if not worker:
        worker = modal_worker_from_config(cfg)
    if use_gpu is None:
        use_gpu = modal_use_gpu_from_config(cfg)
    if not worker:
        worker = DEFAULT_MODAL_WORKER if use_gpu else "cpu"
    return run_modal_http(
        seq,
        config=config,
        accuracy=accuracy,
        use_gpu=use_gpu,
        exact_trajectories=exact_trajectories,
        worker=worker,
    )
