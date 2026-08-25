"""Simulate bundled gre.seq on the default modal backend and reconstruct with iFFT."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import MRzeroCloud as mr0

EXAMPLE_DIR = Path(__file__).resolve().parent
SEQ_PATH = EXAMPLE_DIR / "gre.seq"


def main() -> None:
    if not SEQ_PATH.is_file():
        raise FileNotFoundError(f"Missing {SEQ_PATH}")

    signal, ktraj = mr0.simulate(str(SEQ_PATH))

    print(f"signal: {signal.size} samples, ktraj: {ktraj.shape[0]}x{ktraj.shape[1]}")

    fig = plt.figure()
    ax_sig = fig.add_subplot(2, 1, 1)
    ax_sig.plot(np.abs(signal))
    ax_sig.set_title("|signal|")
    fig.add_subplot(2, 2, 3).plot(ktraj[:, 0], ktraj[:, 1], ".")
    ax_kyz = fig.add_subplot(2, 2, 4)
    ax_kyz.plot(ktraj[:, 1], ktraj[:, 2], ".")
    ax_kyz.set_title("k-space trajectory")

    # MR image recon of signal (MRzero FLASH notebook port)
    n = int(round(np.sqrt(signal.size)))
    kspace = np.reshape(signal, (n, n), order="F")

    spectrum = np.fft.fftshift(kspace)
    space = np.fft.ifft2(spectrum)
    space = np.fft.ifftshift(space)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    panels = (
        (axes[0, 0], np.abs(kspace), "k-space", None),
        (axes[0, 1], np.log(np.abs(kspace)), "log. k-space", None),
        (axes[1, 0], np.abs(space), "recon-magnitude", None),
        (axes[1, 1], np.angle(space), "recon-phase", (-np.pi, np.pi)),
    )
    for ax, data, title, clim in panels:
        shown = np.flipud(data.T)
        im = ax.imshow(shown, cmap="gray", aspect="equal")
        if clim is not None:
            im.set_clim(*clim)
        fig.colorbar(im, ax=ax)
        ax.set_title(title)

    plt.show()


if __name__ == "__main__":
    main()
