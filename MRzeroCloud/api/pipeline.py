"""Internal ToolAPI simulation pipeline."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ._config import (
    DEFAULT_MODAL_WORKER,
    backend_from_config,
    modal_exact_trajectories_from_config,
    modal_use_gpu_from_config,
    modal_worker_from_config,
    optional_config,
)
from ._convert import (
    parse_phantom_id,
    sequence_for_trajex,
    seq_to_text,
    signal_to_complex_np,
    to_py,
    trajectory_from_trajex_result,
)
from ._modal_http import run_modal_http
from ._tools import SIM_BACKENDS, assert_tool_ok, call_tool


def load_seq(seq, *, exact_trajectory: bool = False):
    """Load a Pulseq sequence via conseq."""
    seq_text = seq_to_text(seq)
    return assert_tool_ok(
        "conseq",
        call_tool("conseq", seq_file=seq_text, exact_trajectory=exact_trajectory),
    )


def load_phantom_toolapi(
    subject,
    affine: Sequence[Sequence[float]],
    res: Sequence[int],
):
    """Load a BrainWeb phantom via phantomlib."""
    res_x, res_y, res_z = (int(res[0]), int(res[1]), int(res[2]))
    result = assert_tool_ok(
        "phantomlib",
        call_tool(
            "phantomlib",
            subject=parse_phantom_id(subject),
            res_x=res_x,
            res_y=res_y,
            res_z=res_z,
            affine=[[float(v) for v in row] for row in affine],
        ),
    )
    result = to_py(result)
    if isinstance(result, dict) and result.get("phantom") is not None:
        return result["phantom"]
    if isinstance(result, dict) and result.get("SegmentedPhantom") is not None:
        return result["SegmentedPhantom"]
    return result


def trajex(sequence, *, t1: float = 1.0, t2: float = 0.1, min_mag: float = 1e-3) -> np.ndarray:
    """Compute k-space trajectory for a conseq sequence."""
    result = assert_tool_ok(
        "trajex",
        call_tool(
            "trajex",
            sequence=sequence_for_trajex(sequence),
            t1=t1,
            t2=t2,
            min_mag=min_mag,
        ),
    )
    trajectory = trajectory_from_trajex_result(result)
    if trajectory is None or trajectory.size == 0:
        raise RuntimeError("trajex returned no usable trajectory")
    return trajectory


def run(
    backend: str | None,
    sequence,
    phantom,
    *,
    config: dict | None = None,
    t1: float = 1.0,
    t2: float = 0.1,
    min_mag: float = 1e-3,
    seq=None,
    accuracy: float = 1e-5,
    use_gpu: bool | None = None,
    exact_trajectories: bool | None = None,
    worker: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate k-space data and return ``(kdata, ktraj)``."""
    if backend is None:
        backend = backend_from_config(config or optional_config())
    backend_key = SIM_BACKENDS.get(backend.lower())
    if backend_key is None:
        raise ValueError(f"Unknown simulation backend: {backend!r}")

    if backend_key == "modal":
        cfg = config or optional_config() or {}
        if exact_trajectories is None:
            exact_trajectories = modal_exact_trajectories_from_config(cfg)
        # Tier precedence: worker arg, legacy use_gpu arg, config worker, config use_gpu.
        if not worker and use_gpu is not None:
            worker = DEFAULT_MODAL_WORKER if use_gpu else "cpu"
        if not worker:
            worker = modal_worker_from_config(cfg)
        if use_gpu is None:
            use_gpu = modal_use_gpu_from_config(cfg)
        if not worker:
            worker = DEFAULT_MODAL_WORKER if use_gpu else "cpu"
        return run_modal_http(
            seq if seq is not None else sequence,
            config=config,
            accuracy=accuracy,
            use_gpu=use_gpu,
            exact_trajectories=exact_trajectories,
            worker=worker,
        )

    ktraj = trajex(sequence, t1=t1, t2=t2, min_mag=min_mag)
    signal_result = assert_tool_ok(
        backend_key,
        call_tool(backend_key, sequence=sequence, phantom=phantom),
    )
    kdata = signal_to_complex_np(signal_result)
    if kdata.size == 0:
        raise RuntimeError(f"{backend_key} returned no signal")
    return kdata, ktraj
