"""ToolAPI endpoints and low-level call helpers."""

from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from time import perf_counter
from typing import Callable, Iterator

import toolapi

from ..exceptions import SimulationAborted

DEFAULT_URLS = {
    "conseq": "wss://tool-conseq.fly.dev/tool",
    "trajex": "wss://tool-trajex.fly.dev/tool",
    "phantomlib": "wss://tool-phantomlib-flyio.fly.dev/tool",
    "rapisim": "wss://tool-rapisim.fly.dev/tool",
    "mr0sim": "wss://tool-mr0sim.fly.dev/tool",
    "modal": "https://mzaiss--tool-mr0sim-modal-http-gateway.modal.run",
}

SIM_BACKENDS = {
    "rapisim": "rapisim",
    "mr0sim": "mr0sim",
    "mr0": "mr0sim",
    "modal": "modal",
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
    """Override default tool URLs and progress reporting.

    Only explicitly supplied arguments are applied; url overrides accumulate
    across calls. Use ``urls={"modal": "https://…"}`` to point the ``modal``
    backend at a different tool-mr0sim-modal_http endpoint.
    """
    global _urls, _on_message, _verbose
    if urls:
        _urls = {**_urls, **urls}
    if on_message is not None:
        _on_message = on_message
    if verbose is not None:
        _verbose = bool(verbose)


def reset_configuration() -> None:
    """Restore the built-in URLs, progress callback, and verbosity."""
    global _urls, _on_message, _verbose
    _urls = dict(DEFAULT_URLS)
    _on_message = None
    _verbose = True


@contextmanager
def abort_context(abort: threading.Event | None) -> Iterator[None]:
    """Bind a per-job abort event for nested :func:`call_tool` invocations."""
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


def _is_abort_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "onmessageabort" in msg or "client requested abort" in msg


def call_tool(name: str, **kwargs):
    global _progress_line_width
    _progress_line_width = 0
    url = _urls[name]
    start = perf_counter()
    try:
        result = toolapi.call(url, kwargs, _resolve_on_message())
    except Exception as exc:
        if _is_abort_error(exc):
            raise SimulationAborted("cloud simulation aborted by client") from exc
        raise
    if _verbose:
        print(f"\n --- {url} done ({perf_counter() - start:.2f} s) ---", flush=True)
    _progress_line_width = 0
    return result


def get_modal_url() -> str:
    """Base URL for the ``modal`` HTTP simulation backend."""
    return _urls["modal"]


def assert_tool_ok(label: str, result):
    from ._convert import tool_error_message

    msg = tool_error_message(result)
    if msg:
        raise RuntimeError(f"{label} failed: {msg}")
    return result
