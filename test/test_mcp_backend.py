"""Offline tests for the MRX MCP backend (HTTP mocked)."""

from __future__ import annotations

import io
import json
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import MRzeroCloud as mr0
from MRzeroCloud import api
from MRzeroCloud.api import _mcp_http

SEQ = Path(__file__).resolve().parents[1] / "examples" / "gre.seq"
BASE = "https://gateway.test"


def _npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _bifti_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("p.json", json.dumps({"tissues": {"gm": {"density": "p.nii.gz[0]"}}}))
        zf.writestr("p.nii.gz", b"nifti")
    return buf.getvalue()


def _link(name: str, sha: str) -> dict:
    return {"type": "resource_link", "name": name, "uri": f"{BASE}/files/{sha}"}


class FakeGateway:
    """Answers ``/mcp`` tool calls and uploads; records what it was sent."""

    def __init__(self, n: int = 16) -> None:
        self.files = {
            "a" * 64: _bifti_zip(),
            "b" * 64: _npy(np.arange(n) * (1 + 1j)),
            "c" * 64: _npy(np.ones((n, 4), dtype=np.float32)),
        }
        self.calls: list[tuple[str, dict]] = []
        self.uploads: dict[str, bytes] = {}

    def post(self, url, json=None, headers=None, timeout=None):
        assert url == f"{BASE}/mcp"
        assert headers["Authorization"] == "Bearer mrx_test"
        name, args = json["params"]["name"], json["params"]["arguments"]
        self.calls.append((name, args))
        if name == "create_upload_url":
            result = {"structuredContent": {"url": f"{BASE}/upload/{args['file_name']}"}}
        elif name == "bifti_get_phantom":
            result = {"content": [_link("p.zip", "a" * 64)]}
        elif name == "rapisim_simulate":
            result = {"content": [{"type": "text", "text": "ok"}, _link("signal.npy", "b" * 64)]}
        elif name == "trajex_extract_trajectory":
            result = {"content": [_link("gre-signal-est.npy", "d" * 64),
                                  _link("gre-kspace.npy", "c" * 64)]}
        else:
            raise AssertionError(name)
        resp = MagicMock(headers={"content-type": "application/json"})
        resp.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": result}
        return resp

    def put(self, url, data=None, timeout=None):
        name = url.rsplit("/", 1)[-1]
        self.uploads[name] = data
        return MagicMock(text=f"sha-{name}")

    def get(self, url, timeout=None):
        return self.files[url.rsplit("/", 1)[-1]]


class McpBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        api.configure(urls={"mcp": BASE}, mcp_token="mrx_test", verbose=False)
        _mcp_http._phantom_cache.clear()
        self.gw = FakeGateway()
        self.patches = [
            patch.object(_mcp_http.requests, "post", self.gw.post),
            patch.object(_mcp_http.requests, "put", self.gw.put),
            patch.object(_mcp_http, "_http_get_bytes", self.gw.get),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        api.reset_configuration()

    def test_default_phantom(self):
        signal, ktraj = mr0.simulate(str(SEQ), backend="mcp", min_mag=1e-4)
        self.assertEqual(signal.dtype, np.complex64)
        self.assertEqual(ktraj.shape, (16, 4))
        np.testing.assert_allclose(signal, np.arange(16) * (1 + 1j))

        calls = dict(self.gw.calls)
        self.assertEqual(calls["bifti_get_phantom"]["collection"], "endres-bifti_demo-001")
        rapisim = calls["rapisim_simulate"]
        self.assertEqual(rapisim["sequence"], "sha-gre.seq")
        # Native grid: the registry phantom's hash is used as is, no re-upload.
        self.assertEqual(rapisim["phantom"], "a" * 64)
        self.assertEqual(rapisim["min_mag"], 1e-4)
        self.assertEqual(calls["trajex_extract_trajectory"]["min_mag"], 1e-4)
        self.assertEqual(list(self.gw.uploads), ["gre.seq"])

    def test_config_backend_and_reslice(self):
        config = api.default_mcp_config()
        config["res"] = [32, 32, 1]
        config["affine"] = [[2, 0, 0, -32], [0, 2, 0, -32], [0, 0, 8, -4]]
        mr0.simulate(str(SEQ), config=config)
        mr0.simulate(str(SEQ), config=config)

        rapisim = dict(self.gw.calls)["rapisim_simulate"]
        self.assertEqual(rapisim["phantom"], "sha-subj42-3T_resliced.zip")
        with zipfile.ZipFile(io.BytesIO(self.gw.uploads["subj42-3T_resliced.zip"])) as zf:
            meta = json.loads(zf.read("p.json"))
            self.assertEqual(zf.read("p.nii.gz"), b"nifti")
        self.assertEqual(meta["reslice_to"]["resolution"], [32, 32, 1])
        self.assertEqual(meta["reslice_to"]["affine"][0], [2.0, 0.0, 0.0, -32.0])
        # Second run reuses the uploaded phantom.
        names = [name for name, _ in self.gw.calls]
        self.assertEqual(names.count("bifti_get_phantom"), 1)

    def test_missing_token(self):
        api.reset_configuration()
        api.configure(urls={"mcp": BASE}, verbose=False)
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "/keys"):
                mr0.simulate(str(SEQ), backend="mcp")

    def test_tool_error(self):
        def failing_post(*args, **kwargs):
            resp = MagicMock(headers={"content-type": "text/event-stream"})
            resp.text = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":' \
                        '{"isError":true,"content":[{"type":"text","text":"bad seq"}]}}\n'
            return resp

        with patch.object(_mcp_http.requests, "post", failing_post):
            with self.assertRaisesRegex(RuntimeError, "bad seq"):
                mr0.simulate(str(SEQ), backend="mcp")


if __name__ == "__main__":
    unittest.main()
