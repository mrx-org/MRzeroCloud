"""AnyField protocol metadata helpers."""

from __future__ import annotations

import json
from typing import Any

_registered_metadata: str | dict[str, Any] | None = None

_DEFAULT_AFFINE = [
    [2.5, 0.0, 0.0, -80.0],
    [0.0, 2.5, 0.0, -80.0],
    [0.0, 0.0, 2.5, 0.0],
]
_DEFAULT_PHANTOM_MATRIX = [64, 64, 1]
_DEFAULT_SUBJECT = 4
_DEFAULT_BACKEND = "modal"

# Modal HTTP defaults: cached bifti phantom on its native NIfTI grid. The
# gateway reslices server-side and requires res+affine on every bifti job.
_DEFAULT_MODAL_PHANTOM = "user/numerical_brain_cropped_bifti"
_DEFAULT_MODAL_RES = [141, 161, 1]
_DEFAULT_MODAL_AFFINE = [
    [1.418, 0.0, 0.0, -100.0],
    [0.0, 1.242, 0.0, -100.0],
    [0.0, 0.0, 8.0, -4.0],
]
_DEFAULT_MODAL_WORKER = "t4"

DEFAULT_BACKEND = _DEFAULT_BACKEND
DEFAULT_MODAL_WORKER = _DEFAULT_MODAL_WORKER


def register_metadata(source: str | dict[str, Any]) -> None:
    """Register protocol metadata for :func:`load_config` with no arguments."""
    global _registered_metadata
    _registered_metadata = source


def parse_metadata(source: str | dict[str, Any]) -> dict[str, Any]:
    """Parse AnyField metadata from a JSON string or dict."""
    if isinstance(source, dict):
        return source
    return json.loads(source)


def optional_config() -> dict[str, Any] | None:
    """Return registered metadata, or ``None`` when nothing was registered."""
    if _registered_metadata is None:
        return None
    return load_config(_registered_metadata)


def _affine_from_flat(flat) -> list[list[float]]:
    if flat and len(flat) >= 12:
        return [
            [float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])],
            [float(flat[4]), float(flat[5]), float(flat[6]), float(flat[7])],
            [float(flat[8]), float(flat[9]), float(flat[10]), float(flat[11])],
        ]
    return [row[:] for row in _DEFAULT_AFFINE]


def _seq_matrix(seq) -> list[int] | None:
    """Single-slice ``[nx, ny, 1]`` from a Pulseq ``matrix`` definition."""
    if seq is None or not hasattr(seq, "get_definition"):
        return None
    try:
        matrix = seq.get_definition("matrix")
    except Exception:
        return None
    if matrix and len(matrix) >= 2:
        return [int(matrix[0]), int(matrix[1]), 1]
    return None


def default_config(seq=None) -> dict[str, Any]:
    """Fly ToolAPI defaults: BrainWeb subject 4, single axial (transversal) slice."""
    res = _seq_matrix(seq) or list(_DEFAULT_PHANTOM_MATRIX)  # [nx, ny, 1]
    recon_matrix = res[:]
    assert res[2] == 1, "default phantom must be a single 2D slice (res_z=1)"
    return {
        "phantom": _DEFAULT_SUBJECT,
        "res": res,
        "affine": [row[:] for row in _DEFAULT_AFFINE],
        "backend": "mr0sim",
        "recon_matrix": recon_matrix,
    }


def default_modal_config(seq=None) -> dict[str, Any]:
    """Standalone defaults for the ``modal`` backend (matches MATLAB ``mr0.defaultConfig``).

    ``res``/``affine`` describe the cached phantom's native grid, so no FOV
    change is requested. ``recon_matrix`` still follows the sequence.
    """
    return {
        "phantom_bifti": _DEFAULT_MODAL_PHANTOM,
        "res": list(_DEFAULT_MODAL_RES),
        "affine": [row[:] for row in _DEFAULT_MODAL_AFFINE],
        "backend": "modal",
        "worker": _DEFAULT_MODAL_WORKER,
        "exact_trajectories": True,
        "recon_matrix": _seq_matrix(seq) or list(_DEFAULT_PHANTOM_MATRIX),
    }


def load_config(source: str | dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse AnyField metadata into a flat config dict.

    Returns keys such as ``affine``, ``res``, ``backend``, ``recon_matrix``,
    and ``phantom``.
    """
    if source is None:
        if _registered_metadata is None:
            raise ValueError("pass metadata to load_config(...)")
        source = _registered_metadata

    raw = parse_metadata(source)
    if "phantom_bifti" in raw or "affine" in raw or "res" in raw:
        config = dict(raw)
        has_affine = config.get("affine") is not None
        has_res = config.get("res") is not None
        if has_res != has_affine:
            raise ValueError(
                "config must set both 'res' and 'affine' (or neither); "
                "the gateway reslices bifti phantoms server-side"
            )
        if not has_res:
            config["res"] = list(_DEFAULT_MODAL_RES)
            config["affine"] = [row[:] for row in _DEFAULT_MODAL_AFFINE]
        if not config.get("phantom_bifti"):
            config["phantom_bifti"] = _DEFAULT_MODAL_PHANTOM
        return config

    sim = raw.get("simulation") or {}
    recon = raw.get("recon") or {}
    matrix = sim.get("phantom_matrix") or _DEFAULT_PHANTOM_MATRIX
    recon_matrix = recon.get("matrix") or matrix
    phantom = sim.get("phantom", _DEFAULT_SUBJECT)
    phantom_bifti = sim.get("phantom_bifti")
    if not phantom_bifti:
        phantom_bifti = phantom if isinstance(phantom, str) else _DEFAULT_MODAL_PHANTOM
    return {
        "affine": _affine_from_flat(sim.get("phantom_fov_affine")),
        "res": [int(matrix[0]), int(matrix[1]), int(matrix[2])],
        "backend": sim.get("backend", _DEFAULT_BACKEND),
        "recon_matrix": [
            int(recon_matrix[0]),
            int(recon_matrix[1]),
            int(recon_matrix[2]) if len(recon_matrix) > 2 else 1,
        ],
        "phantom": phantom,
        "worker": sim.get("worker"),
        "use_gpu": sim.get("use_gpu", True),
        "exact_trajectories": sim.get("exact_trajectories", True),
        "phantom_bifti": phantom_bifti,
    }


def phantomlib_params_from_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract phantomlib ``affine`` and resolution from a config dict."""
    if config is None:
        config = optional_config()
    if config is None:
        config = default_config()
    elif "affine" not in config or "res" not in config:
        config = load_config(config)
    res = config["res"]
    return {
        "affine": config["affine"],
        "res_x": int(res[0]),
        "res_y": int(res[1]),
        "res_z": int(res[2]),
    }


def phantom_id_from_config(config: dict[str, Any] | None = None):
    if config is None:
        config = optional_config() or {}
    if "phantom" in config:
        return config["phantom"]
    return load_config(config)["phantom"]


def backend_from_config(config: dict[str, Any] | None = None) -> str:
    if config is None:
        config = optional_config() or {}
    if "backend" in config:
        return config["backend"]
    return load_config(config)["backend"]


def modal_worker_from_config(config: dict[str, Any] | None = None) -> str | None:
    if config is None:
        config = optional_config() or {}
    worker = config.get("worker")
    return str(worker) if worker else None


def modal_use_gpu_from_config(config: dict[str, Any] | None = None) -> bool:
    if config is None:
        config = optional_config() or {}
    if "use_gpu" in config:
        return bool(config["use_gpu"])
    return True


def modal_exact_trajectories_from_config(config: dict[str, Any] | None = None) -> bool:
    if config is None:
        config = optional_config() or {}
    if "exact_trajectories" in config:
        return bool(config["exact_trajectories"])
    return True


def default_phantom_res(config: dict[str, Any] | None = None) -> tuple[int, int, int]:
    params = phantomlib_params_from_config(config)
    return (params["res_x"], params["res_y"], params["res_z"])


def recon_matrix_from_config(config: dict[str, Any] | None = None) -> tuple[int, int, int]:
    if config is None:
        config = optional_config() or {}
    if "recon_matrix" in config:
        matrix = config["recon_matrix"]
    else:
        matrix = load_config(config)["recon_matrix"]
    return (int(matrix[0]), int(matrix[1]), int(matrix[2]))
