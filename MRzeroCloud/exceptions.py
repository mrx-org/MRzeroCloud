"""MRzeroCloud-specific exceptions."""


class SimulationAborted(RuntimeError):
    """Raised when a cloud simulation is cancelled via :func:`util.stop_simulation`."""
