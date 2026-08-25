"""Simulate bundled gre.seq on the default modal backend and reconstruct with NUFFT."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import MRzeroCloud as mr0

EXAMPLE_DIR = Path(__file__).resolve().parent
SEQ_PATH = EXAMPLE_DIR / "gre.seq"


def main() -> None:
    if not SEQ_PATH.is_file():
        raise FileNotFoundError(f"Missing {SEQ_PATH}")

    signal, ktraj_adc = mr0.simulate(str(SEQ_PATH))

    image = mr0.reco_pynufft(
        signal,
        ktraj_adc,
        resolution=(64, 64, 1),
        FOV=(0.256, 0.256, 0.003),
    )
    mr0.imshow(np.abs(image))
    plt.title("reco_pynufft magnitude")
    plt.show()


if __name__ == "__main__":
    main()
