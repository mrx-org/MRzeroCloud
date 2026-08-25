"""Save and display reconstructed images."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_last_save_paths: dict[str, str | None] = {"png": None, "npy": None}


def save(
    image: np.ndarray,
    *,
    path: str | Path | None = None,
    png_path: str | Path | None = None,
    npy_path: str | Path | None = None,
) -> dict:
    """Save reconstruction as PNG (magnitude) and/or NumPy array."""
    global _last_save_paths

    image = np.asarray(image)
    if png_path is None and path is not None:
        png_path = Path(path)
        if png_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            png_path = png_path.with_suffix(".png")
    if npy_path is None and path is not None:
        npy_path = Path(path).with_suffix(".npy")

    summary = {
        "recon_png": None,
        "recon_npy": None,
        "recon_shape": [int(v) for v in np.abs(image).shape],
        "recon_abs_max": float(np.max(np.abs(image))),
        "recon_abs_mean": float(np.mean(np.abs(image))),
    }

    if npy_path is not None:
        npy_path = Path(npy_path)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, image)
        summary["recon_npy"] = str(npy_path)
        _last_save_paths["npy"] = summary["recon_npy"]

    if png_path is not None:
        import matplotlib.pyplot as plt

        png_path = Path(png_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        magnitude = np.abs(image)
        vmax = float(np.percentile(magnitude, 95.0))
        if vmax <= 0.0:
            vmax = None
        fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
        im = ax.imshow(magnitude, cmap="gray", origin="lower", vmin=0.0, vmax=vmax)
        ax.set_title("abs(recon), p95 display")
        ax.set_xlabel("read")
        ax.set_ylabel("phase")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)
        summary["recon_png"] = str(png_path)
        _last_save_paths["png"] = summary["recon_png"]

    return summary


def show(image: np.ndarray, *, title: str = "abs(recon)") -> None:
    """Display reconstruction magnitude with matplotlib."""
    import matplotlib.pyplot as plt

    magnitude = np.abs(np.asarray(image))
    vmax = float(np.percentile(magnitude, 95.0))
    if vmax <= 0.0:
        vmax = None
    fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
    im = ax.imshow(magnitude, cmap="gray", origin="lower", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()
