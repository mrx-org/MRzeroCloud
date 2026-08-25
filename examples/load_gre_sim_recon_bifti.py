"""Simulate gre.seq on a bifti phantom selected by registry id.

Default modal jobs use ``user/numerical_brain_cropped_bifti``. This example
switches to ``user/numerical_brain_cropped_bifti_2``. ``res`` and ``affine``
stay on the default 2D grid so the gateway reslices that phantom to the same FOV.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import MRzeroCloud as mr0

EXAMPLE_DIR = Path(__file__).resolve().parent
SEQ_PATH = EXAMPLE_DIR / "gre.seq"
PHANTOM_BIFTI = "user/numerical_brain_cropped_bifti_2"

def main() -> None:
    if not SEQ_PATH.is_file():
        raise FileNotFoundError(f"Missing {SEQ_PATH}")

    config = mr0.api.default_modal_config()
    config["phantom_bifti"] = PHANTOM_BIFTI

    signal, ktraj_adc = mr0.simulate(str(SEQ_PATH), config=config)

    image = mr0.reco_pynufft(
        signal,
        ktraj_adc,
        resolution=(64, 64, 1),
        FOV=(0.256, 0.256, 0.003),
    )
    mr0.imshow(np.abs(image))
    plt.title(f"reco_pynufft  ({PHANTOM_BIFTI})")
    plt.show()


if __name__ == "__main__":
    main()
