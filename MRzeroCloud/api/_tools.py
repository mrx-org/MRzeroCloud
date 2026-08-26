"""Modal HTTP endpoint and progress / abort helpers."""

from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from typing import Callable, Iterator

DEFAULT_URLS = {
    "modal": "https://mzaiss--tool-mr0sim-modal-http-gateway.modal.run",
}

_urls = dict(DEFAULT_URLS)
_on_message: Callable[[str], bool] | None = None
_verbose = True
_progress_line_width = 0
_abort_ctx: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "mr0cloud_abort", default=None
)


def configure(
    *,
    urls: dict[str, str] | None = None,
    on_message: Callable[[str], bool] | None = None,
    verbose: bool | None = None,
) -> None:
    """Override the modal URL and progress reporting.

    Only explicitly supplied arguments are applied; url overrides accumulate
    across calls. Use ``urls={"modal": "https://…"}`` to point at a different
    tool-mr0sim-modal_http endpoint.
    """
    global _urls, _on_message, _verbose
    if urls:
        _urls = {**_urls, **urls}
    if on_message is not None:
        _on_message = on_message
    if verbose is not None:
        _verbose = bool(verbose)


def reset_configuration() -> None:
    """Restore the built-in URL, progress callback, and verbosity."""
    global _urls, _on_message, _verbose
    _urls = dict(DEFAULT_URLS)
    _on_message = None
    _verbose = True


@contextmanager
def abort_context(abort: threading.Event | None) -> Iterator[None]:
    """Bind a per-job abort event checked by progress callbacks."""
    token = _abort_ctx.set(abort)
    try:
        yield
    finally:
        _abort_ctx.reset(token)


def stop_simulation(job) -> None:
    """Request abort of a background simulation job."""
    job.stop()


def _default_on_message(msg: str) -> bool:
    global _progress_line_width
    if not _verbose:
        return True
    line = f" > {msg}"
    if len(line) < _progress_line_width:
        line = line + (" " * (_progress_line_width - len(line)))
    else:
        _progress_line_width = len(line)
    print(f"\r{line}", end="", flush=True)
    return True


def _resolve_on_message() -> Callable[[str], bool]:
    base = _on_message or _default_on_message
    abort = _abort_ctx.get()

    def on_message(msg: str) -> bool:
        if abort is not None and abort.is_set():
            return False
        return base(msg)

    return on_message


def get_modal_url() -> str:
    """Base URL for the modal HTTP simulation backend."""
    return _urls["modal"]
