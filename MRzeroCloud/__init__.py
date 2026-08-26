"""MR-zero Cloud: MRzeroCore-compatible simulation over modal HTTP."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from . import api
from . import util
from .exceptions import SimulationAborted
from .reconstruction import reco_adjoint, reco_pynufft
from .simulation import SimulationJob
from .util import (
    imshow,
    simulate,
    simulate_async,
    stop_simulation,
)

try:
    __version__ = _pkg_version("MRzeroCloud")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0.dev0"


def get_version() -> str:
    """Package version string (mirrors MATLAB ``mr0.version()``)."""
    return __version__


__all__ = [
    "SimulationAborted",
    "SimulationJob",
    "__version__",
    "api",
    "get_version",
    "imshow",
    "reco_adjoint",
    "reco_pynufft",
    "simulate",
    "simulate_async",
    "stop_simulation",
    "util",
]
