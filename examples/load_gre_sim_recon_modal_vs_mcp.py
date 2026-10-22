"""Compare modal, MCP, and local MRzeroCore on one GRE.

All three simulate one axial slice of BrainWeb subject 04 (3T, 1 mm),
covering the GRE field of view (256 mm, 2 mm in-plane, 3 mm slice at
isocenter). MCP uses ``endres-brainweb-001/subj04-3T-1mm-tra.json``. The
modal cache has that subject as ``brainweb-20-v2/subj04-3T-1mm-tra``.
Local MRzeroCore uses the same MCP file, resliced with the ``bifti`` package
onto that grid, and keeps the tissues separate.
Images are ``reco_pynufft`` at 256x256, matching the 256 readouts.

MCP needs an MRX API key. The script prompts for one when ``MRX_API_KEY`` is
unset, then asks whether to save it in the user environment. Create a key at
https://mrx-org--mcp-gateway-func.modal.run/keys

``accuracy`` (modal and local MRzeroCore) and ``min_mag`` (mcp) are ``1e-3``.
They are the precision argument of each backend.
"""

import getpass
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import MRzeroCloud as mr0

EXAMPLE_DIR = Path(__file__).resolve().parent
SEQ_PATH = EXAMPLE_DIR / "gre.seq"
PHANTOM_MCP = "endres-brainweb-001/subj04-3T-1mm-tra.json"
PHANTOM_MODAL = "brainweb-20-v2/subj04-3T-1mm-tra"
# Axial slice through the head. The bifti itself is the full 1 mm volume.
PHANTOM_RES = [128, 128, 1]
PHANTOM_AFFINE = [
    [2.0, 0.0, 0.0, -128.0],
    [0.0, 2.0, 0.0, -128.0],
    [0.0, 0.0, 3.0, -1.5],
]
RESOLUTION = (256, 256, 1)
FOV = (0.256, 0.256, 0.003)
PRECISION = 1e-3


def _configs() -> tuple[dict, dict]:
    modal = mr0.api.default_modal_config()
    modal["phantom_bifti"] = PHANTOM_MODAL
    modal["res"] = list(PHANTOM_RES)
    modal["affine"] = [row[:] for row in PHANTOM_AFFINE]

    mcp = mr0.api.default_mcp_config()
    mcp["phantom_bifti"] = PHANTOM_MCP
    mcp["res"] = list(PHANTOM_RES)
    mcp["affine"] = [row[:] for row in PHANTOM_AFFINE]
    return modal, mcp


def _saved_mrx_api_key() -> str | None:
    """Key from this process, or the Windows user environment.

    A terminal opened by Cursor keeps the environment from when Cursor started,
    so a key saved earlier is in the registry but not in ``os.environ``.
    """
    token = os.environ.get("MRX_API_KEY")
    if token:
        return token.strip() or None
    if sys.platform != "win32":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _kind = winreg.QueryValueEx(key, "MRX_API_KEY")
    except FileNotFoundError:
        return None
    token = str(value).strip()
    return token or None


def _save_mcp_token(token: str) -> None:
    """Write ``MRX_API_KEY`` for this process and, on Windows, the user account."""
    os.environ["MRX_API_KEY"] = token
    if sys.platform != "win32":
        print("MRX_API_KEY is set for this process only.")
        return
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE
    ) as key:
        winreg.SetValueEx(key, "MRX_API_KEY", 0, winreg.REG_SZ, token)
    print("Saved MRX_API_KEY for your Windows user.")


def _ensure_mcp_token() -> None:
    """Use a saved ``MRX_API_KEY``, or ask for a key in an interactive terminal."""
    token = _saved_mrx_api_key()
    if token:
        os.environ["MRX_API_KEY"] = token
        mr0.api.configure(mcp_token=token)
        return
    keys_url = f"{mr0.api.get_mcp_url().rstrip('/')}/keys"
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"no MRX API key (create one at {keys_url}, then set MRX_API_KEY)"
        )
    print(f"MCP needs an MRX API key. Create one at {keys_url}")
    token = getpass.getpass("MRX API key: ").strip()
    if not token:
        raise RuntimeError("no MRX API key entered")
    mr0.api.configure(mcp_token=token)
    save = input("Store this key in MRX_API_KEY? [y/N]: ").strip().lower()
    if save in ("y", "yes"):
        _save_mcp_token(token)


def _local_phantom():
    """BrainWeb slice resliced by ``bifti``, as one MRzeroCore phantom per tissue."""
    import MRzeroCore as mr0_local
    import bifti
    from bifti import BiftiPhantom, NumpyPhantom, ResliceTo

    print(f"local: bifti reslice {PHANTOM_MCP} -> {PHANTOM_RES}")
    json_path = bifti.load_registry_phantom(
        PHANTOM_MCP.partition("/")[0],
        PHANTOM_MCP.partition("/")[2],
    )
    config = BiftiPhantom.load(json_path)
    config.reslice_to = ResliceTo(
        affine=[row[:] for row in PHANTOM_AFFINE],
        resolution=list(PHANTOM_RES),
    )
    loaded = NumpyPhantom.load_from_config(config, json_path.parent)

    tissues = {}
    for name, tissue in loaded.tissues.items():
        tissues[name] = mr0_local.VoxelGridPhantom(
            PD=tissue.density,
            T1=tissue.T1,
            T2=tissue.T2,
            T2dash=tissue.T2dash,
            D=tissue.ADC,
            B0=tissue.dB0,
            B1=tissue.B1_tx,
            coil_sens=tissue.B1_rx,
            affine=loaded.scanner_affine(name),
        )
    return mr0_local.TissueDict(tissues)


def _simulate_local():
    import MRzeroCore as mr0_local

    started = time.perf_counter()
    phantom = _local_phantom()
    signal, ktraj = mr0_local.util.simulate(
        str(SEQ_PATH),
        phantom,
        accuracy=PRECISION,
    )
    signal = np.asarray(signal.detach().cpu()).reshape(-1)
    ktraj = np.asarray(ktraj.detach().cpu())
    elapsed = time.perf_counter() - started
    print(
        f"local: {elapsed:.1f}s  signal {tuple(signal.shape)}  "
        f"ktraj {tuple(ktraj.shape)}"
    )
    image = mr0.reco_pynufft(
        signal,
        ktraj,
        resolution=RESOLUTION,
        FOV=FOV,
    )
    return signal, image


def _simulate(label: str, **kwargs):
    started = time.perf_counter()
    signal, ktraj = mr0.simulate(str(SEQ_PATH), **kwargs)
    elapsed = time.perf_counter() - started
    print(
        f"{label}: {elapsed:.1f}s  signal {tuple(signal.shape)}  "
        f"ktraj {tuple(ktraj.shape)}"
    )
    image = mr0.reco_pynufft(
        signal,
        ktraj,
        resolution=RESOLUTION,
        FOV=FOV,
    )
    return signal, image


def main() -> None:
    if not SEQ_PATH.is_file():
        raise FileNotFoundError(f"Missing {SEQ_PATH}")

    _ensure_mcp_token()
    modal_config, mcp_config = _configs()
    print(
        f"modal phantom {PHANTOM_MODAL}\n"
        f"mcp phantom   {PHANTOM_MCP}\n"
        f"res {modal_config['res']}  precision {PRECISION:g}"
    )

    signal_modal, image_modal = _simulate(
        "modal",
        config=modal_config,
        accuracy=PRECISION,
    )
    signal_mcp, image_mcp = _simulate(
        "mcp",
        config=mcp_config,
        min_mag=PRECISION,
    )
    signal_local, image_local = _simulate_local()

    mags = {
        "modal": np.abs(np.squeeze(image_modal)),
        "mcp": np.abs(np.squeeze(image_mcp)),
        "local": np.abs(np.squeeze(image_local)),
    }
    shape = mags["modal"].shape
    for name, mag in mags.items():
        if mag.shape != shape:
            raise RuntimeError(f"recon shapes differ: modal {shape}, {name} {mag.shape}")

    peak = max(float(mag.max()) for mag in mags.values())
    peak = max(peak, 1e-12)
    scaled = {name: mag / peak for name, mag in mags.items()}

    def _pair(a: str, b: str) -> tuple[np.ndarray, float, float]:
        diff = scaled[a] - scaled[b]
        denom = max(float(np.linalg.norm(scaled[a])), 1e-12)
        nrmse = float(np.linalg.norm(diff) / denom)
        corr = float(np.corrcoef(scaled[a].ravel(), scaled[b].ravel())[0, 1])
        print(f"{a} vs {b}: magnitude NRMSE {nrmse:.4f}  correlation {corr:.4f}")
        return diff, nrmse, corr

    diff_modal_local, nrmse_modal_local, _ = _pair("modal", "local")
    diff_mcp_local, _, _ = _pair("mcp", "local")
    diff_modal_mcp, _, _ = _pair("modal", "mcp")
    n_shared = min(signal_modal.size, signal_mcp.size, signal_local.size)
    print(
        "signal samples "
        f"modal {signal_modal.size}  mcp {signal_mcp.size}  local {signal_local.size}"
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = (
        (axes[0, 0], scaled["modal"], "gray", None, "modal"),
        (axes[0, 1], scaled["mcp"], "gray", None, "mcp"),
        (axes[0, 2], scaled["local"], "gray", None, "local MRzero"),
        (axes[1, 0], diff_modal_local, "coolwarm", np.max(np.abs(diff_modal_local)), "modal − local"),
        (axes[1, 1], diff_mcp_local, "coolwarm", np.max(np.abs(diff_mcp_local)), "mcp − local"),
        (axes[1, 2], diff_modal_mcp, "coolwarm", np.max(np.abs(diff_modal_mcp)), "modal − mcp"),
    )
    for ax, data, cmap, clim, title in panels:
        plt.sca(ax)
        im = mr0.imshow(data, cmap=cmap)
        if clim:
            im.set_clim(-clim, clim)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"BrainWeb subj04   modal vs local NRMSE {nrmse_modal_local:.3f}")
    fig.tight_layout()

    fig_sig, ax_sig = plt.subplots(figsize=(8, 3))
    ax_sig.plot(np.abs(signal_modal[:n_shared]), label="modal", alpha=0.8)
    ax_sig.plot(np.abs(signal_mcp[:n_shared]), label="mcp", alpha=0.8)
    ax_sig.plot(np.abs(signal_local[:n_shared]), label="local", alpha=0.8)
    ax_sig.set_title("|signal|")
    ax_sig.legend()
    fig_sig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
