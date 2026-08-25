"""Reconstruction helpers (Cartesian scatter + Pipe-Menon DCF for NUFFT)."""

from __future__ import annotations

import numpy as np

_GRID_TOL = 1e-3
_DCF_ITERS = 20
_DCF_EPS = 1e-6


def _median_offset_n(d: np.ndarray) -> float:
    if d.size == 0:
        return 0.0
    return float(np.median(d - np.round(d)))


def _centered_int_range(n_len: int) -> tuple[int, int]:
    if n_len % 2 == 0:
        return -n_len // 2, n_len // 2
    return -(n_len // 2), n_len // 2


def _axis_cartesian_after_offset(
    k_axis: np.ndarray,
    fov_m: float,
    tol: float,
) -> tuple[bool, float]:
    if k_axis.size == 0:
        return True, 0.0
    d = k_axis.astype(np.float64, copy=False) * fov_m
    offset = _median_offset_n(d)
    d_adj = d - offset
    r = np.round(d_adj)
    if not np.all(np.abs(d_adj - r) <= tol):
        return False, offset
    return True, offset


def _axis_int_n(k: np.ndarray, fov_m: float, offset_n: float) -> np.ndarray:
    d_adj = k.astype(np.float64) * fov_m - offset_n
    return np.rint(d_adj).astype(np.int64)


def _in_recon_index_range(n: np.ndarray, n_len: int) -> np.ndarray:
    lo, hi = _centered_int_range(n_len)
    return (n >= lo) & (n <= hi)


def per_axis_cartesian_and_offsets(
    tr: np.ndarray,
    fov_x_m: float,
    fov_y_m: float,
    fov_z_m: float,
    nx: int,
    ny: int,
    nz: int,
    tol: float = _GRID_TOL,
) -> tuple[tuple[bool, bool, bool], tuple[float, float, float]]:
    if tr.ndim != 2 or tr.shape[1] < 2:
        return (False, False, False), (0.0, 0.0, 0.0)
    kx = tr[:, 0]
    ky = tr[:, 1]
    kz = tr[:, 2] if tr.shape[1] >= 3 else np.zeros(tr.shape[0], dtype=np.float64)
    okx, ox = _axis_cartesian_after_offset(kx, fov_x_m, tol)
    oky, oy = _axis_cartesian_after_offset(ky, fov_y_m, tol)
    okz, oz = _axis_cartesian_after_offset(kz, fov_z_m, tol)
    return (okx, oky, okz), (ox, oy, oz)


def is_cartesian_trajectory(
    ktraj: np.ndarray,
    nx: int,
    ny: int,
    nz: int = 1,
    fov_x_m: float | None = None,
    fov_y_m: float | None = None,
    fov_z_m: float | None = None,
    tol: float = _GRID_TOL,
) -> bool:
    """True if samples lie on a centered Cartesian grid (integer ``k * FOV`` spacing)."""
    if ktraj is None or ktraj.size == 0:
        return True
    ktraj = np.asarray(ktraj, dtype=np.float64)
    if ktraj.ndim != 2 or ktraj.shape[1] < 2:
        return False
    if fov_x_m is None or fov_y_m is None or fov_z_m is None:
        ok, _ = per_axis_cartesian_and_offsets(
            ktraj, 1.0, 1.0, 1.0, nx, ny, nz, tol
        )
        return bool(all(ok))
    ok, _ = per_axis_cartesian_and_offsets(
        ktraj, fov_x_m, fov_y_m, fov_z_m, nx, ny, nz, tol
    )
    return bool(all(ok))


def scatter_cartesian_kspace(
    signal_1d: np.ndarray,
    tr: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    fov_x_m: float,
    fov_y_m: float,
    fov_z_m: float,
    offset_n: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Scatter ``signal_1d`` onto recon-grid k-space; crop samples outside recon range."""
    ox, oy, oz = offset_n if offset_n is not None else (0.0, 0.0, 0.0)
    ksp = np.zeros((nx, ny, nz), dtype=np.complex128)
    kx = tr[:, 0].astype(np.float64, copy=False)
    ky = tr[:, 1].astype(np.float64, copy=False)
    kz = tr[:, 2].astype(np.float64, copy=False) if tr.shape[1] >= 3 else np.zeros(tr.shape[0], dtype=np.float64)

    n_samp = min(int(signal_1d.shape[0]), int(tr.shape[0]))
    nx_n = _axis_int_n(kx[:n_samp], fov_x_m, ox)
    ny_n = _axis_int_n(ky[:n_samp], fov_y_m, oy)
    nz_n = _axis_int_n(kz[:n_samp], fov_z_m, oz)
    keep = (
        _in_recon_index_range(nx_n, nx)
        & _in_recon_index_range(ny_n, ny)
        & _in_recon_index_range(nz_n, nz)
    )
    if int(np.count_nonzero(keep)) == 0:
        return ksp.astype(np.complex64)

    ix = (nx_n[keep] % nx).astype(np.intp)
    iy = (ny_n[keep] % ny).astype(np.intp)
    iz = (nz_n[keep] % nz).astype(np.intp)
    sig = signal_1d[:n_samp][keep].astype(np.complex128, copy=False)
    np.add.at(ksp, (ix, iy, iz), sig)
    return ksp.astype(np.complex64)


def recon_cartesian_fft(
    signal: np.ndarray,
    matrix,
    tr: np.ndarray | None = None,
    fov_x_m: float | None = None,
    fov_y_m: float | None = None,
    fov_z_m: float | None = None,
    offset_n: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Scatter onto centered Cartesian k-space bins, then ``ifftn`` + ``fftshift``."""
    nx, ny, nz = (int(matrix[0]), int(matrix[1]), int(matrix[2]) if len(matrix) > 2 else 1)
    signal = np.asarray(signal).ravel()

    if tr is not None and fov_x_m is not None and fov_y_m is not None and fov_z_m is not None:
        ksp = scatter_cartesian_kspace(
            signal,
            np.asarray(tr, dtype=np.float64),
            nx,
            ny,
            nz,
            fov_x_m,
            fov_y_m,
            fov_z_m,
            offset_n=offset_n,
        )
    else:
        nread, nphase, nslice = nx, ny, nz
        if nslice != 1:
            raise ValueError(f"cartesian_fft recon expects a single slice, got matrix={matrix}")
        n = nread * nphase
        if signal.size < n:
            raise ValueError(f"Signal has {signal.size} samples, need at least {n}")
        ksp = np.asarray(signal[:n], dtype=np.complex64).reshape(nphase, nread)
        if nz == 1:
            ksp = ksp[..., np.newaxis]

    reco = np.fft.ifftn(ksp, norm="ortho")
    reco = np.fft.fftshift(reco, axes=(0, 1, 2))
    return reco.astype(np.complex64)


def compute_pipe_menon_dcf(nufft, n_samples: int, n_iter: int = _DCF_ITERS) -> np.ndarray:
    """Fixed-point Pipe-Menon DCF using PyNUFFT interpolation operators."""
    if n_samples < 1:
        raise ValueError("Cannot compute DCF for empty trajectory.")
    try:
        y2k = nufft._y2k_cpu
        k2y = nufft._k2y_cpu
    except AttributeError as exc:
        raise RuntimeError("PyNUFFT interpolation operators unavailable for DCF.") from exc

    w = np.ones(int(n_samples), dtype=np.complex64)
    for _ in range(int(n_iter)):
        gridded = y2k(w)
        back = k2y(gridded)
        w = w / np.maximum(np.abs(back), _DCF_EPS)
    return np.abs(w).astype(np.float32)


def recon_nufft_adjoint(
    signal_1d: np.ndarray,
    tr: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    kmax_x: float,
    kmax_y: float,
    kmax_z: float,
    apply_dcf: bool = True,
) -> np.ndarray:
    """PyNUFFT adjoint using a 2D plan for singleton-z and 3D otherwise."""
    from pynufft import NUFFT

    kxy = tr[:, :2]
    kz_col = tr[:, 2].astype(np.float64) if tr.shape[1] >= 3 else None
    oz = np.zeros(kxy.shape[0], dtype=np.float64)
    if kz_col is not None and kmax_z > 1e-30:
        oz = (kz_col / kmax_z) * np.pi
    om2 = np.stack(
        [
            (kxy[:, 0] / kmax_x) * np.pi,
            (kxy[:, 1] / kmax_y) * np.pi,
        ],
        axis=-1,
    )
    om = np.column_stack([om2[:, 0], om2[:, 1], oz])
    n = min(signal_1d.size, om.shape[0])
    signal_n = signal_1d[:n].astype(np.complex64, copy=False)
    om_n = om[:n]
    a = NUFFT()
    use_2d_plan = int(nz) == 1 and (
        kz_col is None or not np.any(np.abs(kz_col[:n]) > 1e-18)
    )
    if use_2d_plan:
        a.plan(om_n[:, :2], (nx, ny), (2 * nx, 2 * ny), (6, 6))
    else:
        kz_plan = max(2 * nz, 4)
        a.plan(om_n, (nx, ny, nz), (2 * nx, 2 * ny, kz_plan), (4, 4, 4))
    if apply_dcf:
        dcf = compute_pipe_menon_dcf(a, n)
        signal_n = signal_n * dcf.astype(np.complex64, copy=False)
    reco = a.adjoint(signal_n)
    if use_2d_plan:
        reco = np.asarray(reco).reshape(nx, ny, 1)
    else:
        reco = np.asarray(reco).reshape(nx, ny, nz)
    return reco.astype(np.complex64)
