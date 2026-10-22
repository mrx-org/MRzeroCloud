"""Simulation via the MRX MCP gateway (rapisim signal + trajex k-space)."""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import requests

from ..exceptions import SimulationAborted
from ._config import default_mcp_config
from ._modal_http import _http_get_bytes, _seq_file
from ._pulseq import check_pulseq_version
from ._tools import _resolve_on_message, get_mcp_token, get_mcp_url

_TOOL_TIMEOUT_S = 3600.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# (phantom id, res, affine) → sha256 of the uploaded bifti, per process.
_phantom_cache: dict[tuple, str] = {}


def _rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """One JSON-RPC call to the (stateless) streamable-HTTP ``/mcp`` endpoint."""
    resp = requests.post(
        f"{get_mcp_url().rstrip('/')}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        headers={
            "Authorization": f"Bearer {get_mcp_token()}",
            "Accept": "application/json, text/event-stream",
        },
        timeout=_TOOL_TIMEOUT_S,
    )
    resp.raise_for_status()
    if resp.headers.get("content-type", "").startswith("text/event-stream"):
        # Notifications may precede the reply; the reply is the last message.
        data = [
            line[len("data:"):].strip()
            for line in resp.text.splitlines()
            if line.startswith("data:")
        ]
        message = json.loads(data[-1])
    else:
        message = resp.json()
    if "error" in message:
        raise RuntimeError(f"mcp {method} failed: {message['error'].get('message')}")
    return message["result"]


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = _rpc("tools/call", {"name": name, "arguments": arguments})
    if result.get("isError"):
        text = " ".join(c.get("text", "") for c in result.get("content", []))
        raise RuntimeError(f"mcp tool {name} failed: {text}")
    return result


def _resource_link(result: dict[str, Any], suffix: str) -> str:
    """URI of the resource link whose name ends with ``suffix``."""
    for item in result.get("content", []):
        if item.get("type") == "resource_link" and item.get("name", "").endswith(suffix):
            return item["uri"]
    names = [c.get("name") for c in result.get("content", []) if c.get("name")]
    raise RuntimeError(f"no {suffix} in mcp result (got {names})")


def _upload(data: bytes, file_name: str) -> str:
    """Upload to the user's files; returns the sha256 (the CAS hash)."""
    result = _call_tool("create_upload_url", {"file_name": file_name})
    reply = result.get("structuredContent") or json.loads(result["content"][0]["text"])
    url = reply["url"]
    resp = requests.put(url, data=data, timeout=600)
    resp.raise_for_status()
    return resp.text.strip()


def _load_npy(uri: str) -> np.ndarray:
    return np.load(io.BytesIO(_http_get_bytes(uri, timeout=600)))


def _with_reslice_to(bifti_zip: bytes, res, affine) -> bytes:
    """Rewrite the bifti JSON's ``reslice_to``; rapisim simulates on that grid."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(bifti_zip)) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            data = src.read(info)
            if info.filename.endswith(".json"):
                meta = json.loads(data)
                if "tissues" in meta:
                    meta["reslice_to"] = {
                        "affine": [[float(v) for v in row] for row in affine],
                        "resolution": [int(n) for n in res],
                    }
                    data = json.dumps(meta, indent=2).encode()
            dst.writestr(info, data)
    return out.getvalue()


def _phantom_hash(config: dict[str, Any]) -> str:
    phantom = str(config["phantom_bifti"])
    res, affine = config.get("res"), config.get("affine")
    if (res is None) != (affine is None):
        raise ValueError("config must set both 'res' and 'affine' (or neither)")
    if _SHA256.match(phantom) and res is None:
        return phantom

    key = (phantom, json.dumps(res), json.dumps(affine))
    if key in _phantom_cache:
        return _phantom_cache[key]

    if _SHA256.match(phantom):
        url = f"{get_mcp_url().rstrip('/')}/files/{phantom}"
        name = phantom[:12]
    else:
        collection, sep, file_name = phantom.partition("/")
        if not sep:
            raise ValueError(
                f"mcp phantom {phantom!r} must be 'collection/file_name' or a sha256"
            )
        result = _call_tool(
            "bifti_get_phantom",
            {"collection": collection, "phantom": {"file_name": file_name}},
        )
        url = _resource_link(result, ".zip")
        name = Path(file_name).stem
        if res is None:
            _phantom_cache[key] = url.rstrip("/").rsplit("/", 1)[-1]
            return _phantom_cache[key]

    resliced = _with_reslice_to(_http_get_bytes(url, timeout=600), res, affine)
    _phantom_cache[key] = _upload(resliced, f"{name}_resliced.zip")
    return _phantom_cache[key]


def run_mcp(
    seq,
    *,
    config: dict[str, Any] | None = None,
    min_mag: float = 1e-3,
    t1: float = 1.0,
    t2: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate via the MRX MCP gateway; returns ``(signal, ktraj)``.

    ``signal`` comes from rapisim on ``config["phantom_bifti"]``, ``ktraj``
    from trajex (a single voxel with ``t1``/``t2``). ``min_mag`` is passed to
    both.
    """
    defaults = default_mcp_config(seq if hasattr(seq, "get_definition") else None)
    config = {**defaults, **(config or {})}
    on_message = _resolve_on_message()

    def step(msg: str) -> None:
        if not on_message(f"mcp: {msg}"):
            raise SimulationAborted("mcp simulation aborted by client")

    with _seq_file(seq) as seq_path:
        # Local validation only — no network I/O until this passes.
        check_pulseq_version(seq_path)
        step(f"uploading {seq_path.name} → {get_mcp_url()}")
        seq_hash = _upload(seq_path.read_bytes(), seq_path.name)

    step(f"phantom {config['phantom_bifti']}")
    phantom_hash = _phantom_hash(config)

    step("trajex")
    trajex = _call_tool(
        "trajex_extract_trajectory",
        {"sequence": seq_hash, "t1": t1, "t2": t2, "min_mag": min_mag},
    )
    step("rapisim")
    rapisim = _call_tool(
        "rapisim_simulate",
        {
            "sequence": seq_hash,
            "phantom": phantom_hash,
            "min_mag": min_mag,
            "sampler": config.get("sampler", "direct"),
            "kernel_size": int(config.get("kernel_size", 4)),
        },
    )
    signal = _load_npy(_resource_link(rapisim, "signal.npy")).astype(np.complex64).ravel()
    ktraj = _load_npy(_resource_link(trajex, "kspace.npy")).astype(np.float32)
    step("complete")
    if signal.size == 0:
        raise RuntimeError("mcp backend returned empty signal")
    return signal, ktraj
