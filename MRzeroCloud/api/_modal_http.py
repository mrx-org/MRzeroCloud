"""Full-pipeline simulation via tool-mr0sim-modal_http (Modal / local FastAPI)."""

from __future__ import annotations

import io
import json
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import requests

from ..exceptions import SimulationAborted
from ._config import (
    DEFAULT_MODAL_WORKER,
    modal_worker_from_config,
    phantom_id_from_config,
    phantom_grid_from_config,
)
from ._pulseq import check_pulseq_version
from ._tools import _resolve_on_message, get_modal_url

POLL_INTERVAL_S = 5.0
_POLL_TIMEOUT_S = 3600.0

# BrainWeb subject id → bifti registry id (matches bench_http default).
_BIFTI_BY_SUBJECT: dict[int, str] = {
    4: "brainweb-20-v2/subj04-3T-1mm-tra",
}


def _phantom_bifti_id(config: dict[str, Any]) -> str:
    phantom = config.get("phantom_bifti")
    if phantom:
        return str(phantom)
    subject = phantom_id_from_config(config)
    if isinstance(subject, int) and subject in _BIFTI_BY_SUBJECT:
        return _BIFTI_BY_SUBJECT[subject]
    if isinstance(subject, str) and "/" in subject:
        return subject
    raise ValueError(
        f"No bifti phantom mapping for subject {subject!r}; "
        "set config['phantom_bifti'] or use a known subject id"
    )


def _job_options(
    config: dict[str, Any],
    *,
    accuracy: float,
    use_gpu: bool,
    exact_trajectories: bool,
    worker: str | None = None,
) -> dict[str, Any]:
    params = phantom_grid_from_config(config)
    opts: dict[str, Any] = {
        "exact_trajectories": exact_trajectories,
        "accuracy": accuracy,
        "phantom": {
            "type": "bifti",
            "id": _phantom_bifti_id(config),
            "res": [params["res_x"], params["res_y"], params["res_z"]],
            "affine": params["affine"],
        },
    }
    if worker is not None:
        opts["worker"] = worker
    else:
        opts["use_gpu"] = use_gpu
    return opts


@contextmanager
def _seq_file(seq) -> Iterator[Path]:
    """Yield a ``.seq`` path for upload (temp file when needed)."""
    if isinstance(seq, (str, Path)):
        path = Path(seq)
        if path.is_file():
            yield path
            return
        if path.suffix == ".seq" or "/" in str(seq) or "\\" in str(seq):
            raise FileNotFoundError(f"Pulseq sequence file not found: {seq}")
    if hasattr(seq, "write"):
        with tempfile.NamedTemporaryFile(suffix=".seq", delete=False) as handle:
            tmp = Path(handle.name)
        try:
            seq.write(str(tmp))
            yield tmp
        finally:
            tmp.unlink(missing_ok=True)
        return
    if isinstance(seq, str):
        with tempfile.NamedTemporaryFile(
            suffix=".seq", delete=False, mode="w", encoding="utf-8"
        ) as handle:
            handle.write(seq)
            tmp = Path(handle.name)
        try:
            yield tmp
        finally:
            tmp.unlink(missing_ok=True)
        return
    raise TypeError(f"Unsupported sequence type for modal backend: {type(seq).__name__}")


def _submit_job(base_url: str, seq_path: Path, options: dict[str, Any]) -> str:
    with seq_path.open("rb") as handle:
        resp = requests.post(
            f"{base_url.rstrip('/')}/v1/jobs",
            files={"seq": (seq_path.name, handle, "application/octet-stream")},
            data={"options": json.dumps(options)},
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()["job_id"]


def _poll_job(
    base_url: str,
    job_id: str,
    *,
    on_message,
) -> None:
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        resp = requests.get(f"{base_url.rstrip('/')}/v1/jobs/{job_id}", timeout=30)
        resp.raise_for_status()
        status = resp.json()
        msg = status.get("message") or status.get("status", "")
        rep = status.get("repetition")
        total = status.get("total")
        if rep and total:
            msg = f"{msg} {rep}/{total}"
        if msg and not on_message(msg):
            requests.post(
                f"{base_url.rstrip('/')}/v1/jobs/{job_id}/abort",
                timeout=30,
            )
            raise SimulationAborted("modal simulation aborted by client")
        st = status.get("status")
        if st == "done":
            return
        if st == "failed":
            raise RuntimeError(status.get("error") or status.get("message") or "job failed")
        if st == "aborted":
            raise SimulationAborted("modal simulation aborted")
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"modal job {job_id} did not finish within {_POLL_TIMEOUT_S}s")


def _fetch_result(base_url: str, job_id: str) -> tuple[np.ndarray, np.ndarray]:
    resp = requests.get(
        f"{base_url.rstrip('/')}/v1/jobs/{job_id}/result",
        timeout=120,
    )
    resp.raise_for_status()
    data = np.load(io.BytesIO(resp.content))
    signal = np.asarray(data["signal"], dtype=np.complex64).ravel()
    ktraj = np.asarray(data["ktraj"], dtype=np.float32)
    return signal, ktraj


def run_modal_http(
    seq,
    *,
    config: dict[str, Any] | None = None,
    accuracy: float = 1e-5,
    use_gpu: bool = True,
    exact_trajectories: bool = True,
    base_url: str | None = None,
    worker: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate via HTTP job API; returns ``(signal, ktraj)``."""
    if config is None:
        from ._config import default_modal_config

        config = default_modal_config(seq if hasattr(seq, "get_definition") else None)

    url = (base_url or get_modal_url()).rstrip("/")
    if not worker:
        worker = modal_worker_from_config(config)
    if not worker:
        worker = DEFAULT_MODAL_WORKER if use_gpu else "cpu"
    options = _job_options(
        config,
        accuracy=accuracy,
        use_gpu=use_gpu,
        exact_trajectories=exact_trajectories,
        worker=worker,
    )
    on_message = _resolve_on_message()

    with _seq_file(seq) as seq_path:
        # Local validation only — no network I/O until this passes.
        check_pulseq_version(seq_path)
        if not on_message(f"modal: submitting {seq_path.name} → {url}"):
            raise SimulationAborted("modal simulation aborted by client")
        job_id = _submit_job(url, seq_path, options)
        if not on_message(f"modal: job {job_id}"):
            requests.post(f"{url}/v1/jobs/{job_id}/abort", timeout=30)
            raise SimulationAborted("modal simulation aborted by client")
        _poll_job(url, job_id, on_message=on_message)
        signal, ktraj = _fetch_result(url, job_id)
    if not on_message("modal: complete"):
        raise SimulationAborted("modal simulation aborted by client")
    if signal.size == 0:
        raise RuntimeError("modal backend returned empty signal")
    return signal, ktraj
