"""PyNUFFT reconstruction compatible with MRzeroCore-style workflows."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .api._config import optional_config, recon_matrix_from_config
from .api._recon import (
    per_axis_cartesian_and_offsets,
    recon_cartesian_fft,
    recon_nufft_adjoint,
)


def _to_numpy_complex(signal) -> np.ndarray:
    signal = np.asarray(signal)
    if np.iscomplexobj(signal):
        return signal.astype(np.complex64).ravel()
    if signal.ndim >= 2 and signal.shape[-1] >= 2:
        return (signal[..., 0] + 1j * signal[..., 1]).astype(np.complex64).ravel()
    return signal.astype(np.complex64).ravel()


def _to_numpy_traj(kspace) -> np.ndarray:
    kspace = np.asarray(kspace, dtype=np.float64)
    if kspace.ndim != 2:
        raise ValueError("kspace trajectory must have shape (N, 2|3)")
    if kspace.shape[1] == 2:
        kspace = np.column_stack([kspace, np.zeros(kspace.shape[0], dtype=np.float64)])
    return kspace


def _normalize_resolution(resolution) -> tuple[int, int, int]:
    if resolution is None:
        config = optional_config()
        if config is not None:
            return recon_matrix_from_config(config)
        raise ValueError("resolution is required when no AnyField metadata is registered")
    if isinstance(resolution, (int, float)):
        side = int(resolution)
        return (side, side, 1)
    resolution = tuple(int(v) for v in resolution)
    if len(resolution) == 2:
        return (resolution[0], resolution[1], 1)
    return (resolution[0], resolution[1], resolution[2])


def _normalize_fov(FOV, resolution: tuple[int, int, int]) -> tuple[float, float, float]:
    if FOV is None:
        return (0.22, 0.22, 0.003)
    if isinstance(FOV, (int, float)):
        side = float(FOV)
        return (side, side, resolution[2] / max(resolution[0], 1) * side)
    fov = tuple(float(v) for v in FOV)
    if len(fov) == 2:
        return (fov[0], fov[1], 0.003)
    return (fov[0], fov[1], fov[2])


def reco_pynufft(
    signal,
    kspace,
    resolution: Sequence[int] | int | None = None,
    FOV: Sequence[float] | float | None = None,
    apply_dcf: bool = True,
) -> np.ndarray:
    """Adjoint NUFFT reconstruction from k-space samples and trajectory.

    Cartesian trajectories use scatter into centered k-space + ``ifftn``.
    Non-Cartesian trajectories use PyNUFFT adjoint with optional Pipe-Menon
    density compensation (``apply_dcf=True`` by default).

    Returns
    -------
    np.ndarray
        Complex reconstructed image (``complex64``).
    """
    from pynufft import NUFFT

    signal_np = _to_numpy_complex(signal)
    ktraj = _to_numpy_traj(kspace)
    nx, ny, nz = _normalize_resolution(resolution)
    fov_x, fov_y, fov_z = _normalize_fov(FOV, (nx, ny, nz))
    nz_use = max(int(nz), 1)

    kmax_x = nx / (2.0 * fov_x)
    kmax_y = ny / (2.0 * fov_y)
    kmax_z = nz_use / (2.0 * fov_z) if fov_z > 1e-30 else 1.0

    axis_ok, offsets_xyz = per_axis_cartesian_and_offsets(
        ktraj, fov_x, fov_y, fov_z, nx, ny, nz_use
    )

    if all(axis_ok):
        image = recon_cartesian_fft(
            signal_np,
            (nx, ny, nz_use),
            tr=ktraj,
            fov_x_m=fov_x,
            fov_y_m=fov_y,
            fov_z_m=fov_z,
            offset_n=offsets_xyz,
        )
        if image.ndim == 2:
            image = image[..., np.newaxis]
        return np.asarray(image, dtype=np.complex64)

    if not all(axis_ok) and kmax_x > 1e-30 and kmax_y > 1e-30 and np.abs(ktraj[:, :2]).max() > 1e-18:
        n_full = min(int(signal_np.size), int(ktraj.shape[0]))
        return recon_nufft_adjoint(
            signal_np[:n_full],
            ktraj[:n_full],
            nx,
            ny,
            nz_use,
            kmax_x,
            kmax_y,
            kmax_z,
            apply_dcf=apply_dcf,
        )

    # Synthetic ω grid fallback (no trajectory or zero kxy).
    kxy = ktraj[:, :2]
    kz_col = ktraj[:, 2]
    use_2d_plan = int(nz_use) == 1 and not np.any(np.abs(kz_col) > 1e-18)

    if kmax_x > 1e-30 and kmax_y > 1e-30 and np.abs(kxy).max() > 1e-18:
        oz = np.zeros(kxy.shape[0], dtype=np.float64)
        if kmax_z > 1e-30:
            oz = (kz_col / kmax_z) * np.pi
        om2 = np.stack(
            [
                (kxy[:, 0] / kmax_x) * np.pi,
                (kxy[:, 1] / kmax_y) * np.pi,
            ],
            axis=-1,
        )
        om = np.column_stack([om2[:, 0], om2[:, 1], oz])
    else:
        kx = np.linspace(-np.pi, np.pi, nx, endpoint=False)
        ky = np.linspace(-np.pi, np.pi, ny, endpoint=False)
        kxg, kyg = np.meshgrid(kx, ky, indexing="xy")
        om = np.stack([kxg.ravel(), kyg.ravel(), np.zeros_like(kxg.ravel())], axis=-1)

    n = min(int(signal_np.size), int(om.shape[0]))
    signal_n = signal_np[:n]
    om_n = om[:n]

    nufft = NUFFT()
    if use_2d_plan:
        nufft.plan(om_n[:, :2], (nx, ny), (2 * nx, 2 * ny), (6, 6))
        reco = np.asarray(nufft.adjoint(signal_n), dtype=np.complex64).reshape(nx, ny, 1)
    else:
        kz_plan = max(2 * nz_use, 4)
        nufft.plan(om_n, (nx, ny, nz_use), (2 * nx, 2 * ny, kz_plan), (4, 4, 4))
        reco = np.asarray(nufft.adjoint(signal_n), dtype=np.complex64).reshape(nx, ny, nz_use)

    return reco
