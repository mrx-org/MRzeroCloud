"""Compare every sequence in ``test/test_seqs`` on local MRzeroCore and MCP.

One row per sequence, six panels: local magnitude, MCP magnitude, magnitude
difference, local phase, MCP phase, phase difference. The phantom is the same
BrainWeb slice as ``examples/load_gre_sim_recon_modal_vs_mcp.py``. Recon FOV
and matrix come from each ``.seq`` file.

The figure is written to ``test/results/local_vs_mcp.png``.
"""

from __future__ import annotations

import importlib.util
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import MRzeroCloud as mr0

ROOT = Path(__file__).resolve().parents[1]
SEQ_DIR = Path(__file__).resolve().parent / "test_seqs"
OUT_PATH = Path(__file__).resolve().parent / "results" / "local_vs_mcp.png"
EXAMPLE_PATH = ROOT / "examples" / "load_gre_sim_recon_modal_vs_mcp.py"

COLUMNS = (
    ("local mag", "gray", False),
    ("mcp mag", "gray", False),
    ("mag diff", "coolwarm", True),
    ("local phase", "twilight", False),
    ("mcp phase", "twilight", False),
    ("phase diff", "coolwarm", True),
)


def _example():
    spec = importlib.util.spec_from_file_location("gre_compare", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _defs(seq_path: Path) -> dict[str, str]:
    text = seq_path.read_text(encoding="utf-8", errors="replace")
    if "[DEFINITIONS]" not in text:
        return {}
    block = text.split("[DEFINITIONS]", 1)[1].split("[", 1)[0]
    out = {}
    for line in block.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0].lower()] = " ".join(parts[1:])
    return out


def _grid(seq_path: Path) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
    defs = _defs(seq_path)
    fov_vals = [float(v) for v in defs["fov"].split()]
    fov = (fov_vals[0], fov_vals[1], fov_vals[2] if len(fov_vals) > 2 else 0.003)
    matrix_key = "reconmatrix" if "reconmatrix" in defs else "matrix" if "matrix" in defs else None
    if matrix_key:
        values = [int(float(v)) for v in defs[matrix_key].split()]
        res = (values[0], values[1], values[2] if len(values) > 2 else 1)
        return fov, res
    adc_block = seq_path.read_text(encoding="utf-8", errors="replace").split("[ADC]", 1)[1].split("[", 1)[0]
    adc = None
    for line in adc_block.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "1" and parts[1].isdigit():
            adc = int(parts[1])
            break
    over = int(float(defs.get("readoutoversamplingfactor", "1")))
    n = max((adc or 64) // over, 1)
    return fov, (n, n, 1)


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _plane(image: np.ndarray) -> np.ndarray:
    """2D complex plane. A volume keeps the partition with the most signal."""
    arr = np.squeeze(np.asarray(image))
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"expected a 2D or 3D image, got {arr.shape}")
    energy = np.mean(np.abs(arr), axis=(0, 1))
    return arr[:, :, int(np.argmax(energy))]


def _reconstruct(signal, ktraj, fov, res) -> np.ndarray:
    image = mr0.reco_pynufft(
        _numpy(signal).reshape(-1),
        _numpy(ktraj),
        resolution=res,
        FOV=fov,
    )
    return _plane(image)


def _simulate_local(seq_path: Path, phantom, accuracy: float):
    import MRzeroCore as mr0_local

    started = time.perf_counter()
    signal, ktraj = mr0_local.util.simulate(str(seq_path), phantom, accuracy=accuracy)
    elapsed = time.perf_counter() - started
    print(f"  local {elapsed:.1f}s  signal {_numpy(signal).reshape(-1).shape}")
    return signal, ktraj


def _simulate_mcp(seq_path: Path, config: dict, min_mag: float):
    started = time.perf_counter()
    signal, ktraj = mr0.simulate(str(seq_path), config=config, min_mag=min_mag)
    elapsed = time.perf_counter() - started
    print(f"  mcp   {elapsed:.1f}s  signal {tuple(signal.shape)}")
    return signal, ktraj


def _panels(local: np.ndarray, mcp: np.ndarray):
    peak = max(float(np.abs(local).max()), float(np.abs(mcp).max()), 1e-12)
    mag_local = np.abs(local) / peak
    mag_mcp = np.abs(mcp) / peak
    phase_local = np.angle(local)
    phase_mcp = np.angle(mcp)
    phase_diff = np.angle(np.exp(1j * (phase_local - phase_mcp)))
    return (
        mag_local,
        mag_mcp,
        mag_local - mag_mcp,
        phase_local,
        phase_mcp,
        phase_diff,
    )


def _run_one(seq_path: Path, phantom, mcp_config: dict, accuracy: float) -> dict:
    print(seq_path.name)
    fov, res = _grid(seq_path)
    print(f"  fov {fov}  res {res}")
    row = {"name": seq_path.stem, "error": None, "panels": None}
    try:
        signal_l, ktraj_l = _simulate_local(seq_path, phantom, accuracy)
        local = _reconstruct(signal_l, ktraj_l, fov, res)
        signal_m, ktraj_m = _simulate_mcp(seq_path, mcp_config, accuracy)
        remote = _reconstruct(signal_m, ktraj_m, fov, res)
        if local.shape != remote.shape:
            raise RuntimeError(f"recon shapes differ: local {local.shape}, mcp {remote.shape}")
        row["panels"] = _panels(local, remote)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  failed: {row['error']}")
        traceback.print_exc()
    return row


def _draw(rows: list[dict]) -> None:
    n = len(rows)
    fig, axes = plt.subplots(
        n,
        len(COLUMNS),
        figsize=(2.3 * len(COLUMNS), 2.15 * n),
        squeeze=False,
    )
    for col, (title, _cmap, _symmetric) in enumerate(COLUMNS):
        axes[0, col].set_title(title)
    for row_i, row in enumerate(rows):
        axes[row_i, 0].set_ylabel(row["name"], fontsize=8)
        if row["panels"] is None:
            for col in range(len(COLUMNS)):
                ax = axes[row_i, col]
                ax.axis("off")
                if col == 0:
                    ax.text(
                        0.0,
                        0.5,
                        f"{row['name']}\n{row['error']}",
                        fontsize=7,
                        va="center",
                        wrap=True,
                    )
            continue
        for col, (data, (_title, cmap, symmetric)) in enumerate(zip(row["panels"], COLUMNS)):
            ax = axes[row_i, col]
            plt.sca(ax)
            im = mr0.imshow(data, cmap=cmap)
            if symmetric:
                limit = float(np.max(np.abs(data)))
                if "phase" in _title:
                    limit = np.pi
                im.set_clim(-limit, limit)
            elif "phase" in _title:
                im.set_clim(-np.pi, np.pi)
            else:
                im.set_clim(0.0, 1.0)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"wrote {OUT_PATH}")
    plt.show()


def main() -> None:
    seqs = sorted(SEQ_DIR.glob("*.seq"), key=lambda path: path.name.lower())
    if not seqs:
        raise FileNotFoundError(f"No .seq files in {SEQ_DIR}")

    example = _example()
    example._ensure_mcp_token()
    _modal, mcp_config = example._configs()
    phantom = example._local_phantom()

    rows = [
        _run_one(seq_path, phantom, mcp_config, example.PRECISION)
        for seq_path in seqs
    ]
    _draw(rows)


if __name__ == "__main__":
    main()
