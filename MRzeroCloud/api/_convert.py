"""Convert ToolAPI values to Python / NumPy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def to_py(obj: Any) -> Any:
    if hasattr(obj, "to_py"):
        return obj.to_py()
    return obj


def to_arr(obj: Any) -> list:
    obj = to_py(obj)
    if obj is None:
        return []
    if isinstance(obj, np.ndarray):
        return obj.ravel().tolist()
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if hasattr(obj, "__len__") and not isinstance(obj, (str, bytes, dict)):
        try:
            return list(obj)
        except TypeError:
            return []
    return []


def tool_error_message(result) -> str | None:
    result = to_py(result)
    if not isinstance(result, dict):
        return None
    if result.get("Error") is not None:
        return str(result["Error"])
    if result.get("err") is not None:
        return str(result["err"])
    ok = to_py(result.get("Ok"))
    if isinstance(ok, dict):
        if ok.get("Error") is not None:
            return str(ok["Error"])
        if ok.get("err") is not None:
            return str(ok["err"])
    return None


def sequence_for_trajex(conseq_result):
    seq = to_py(conseq_result)
    if isinstance(seq, dict) and seq.get("Ok") is not None:
        seq = to_py(seq["Ok"])
    if isinstance(seq, dict):
        typed_list = seq.get("TypedList")
        if isinstance(typed_list, dict) and typed_list.get("InstantSeqEvent") is not None:
            return {"TypedList": {"InstantSeqEvent": typed_list["InstantSeqEvent"]}}
    return seq


def trajectory_from_trajex_result(result) -> np.ndarray | None:
    result = to_py(result)
    if not result:
        return None
    if isinstance(result, dict):
        if result.get("Ok") is not None:
            result = to_py(result["Ok"])
        if isinstance(result, dict) and result.get("Error"):
            return None
        if isinstance(result, dict) and result.get("Trajectory") is not None:
            result = to_py(result["Trajectory"])
        if isinstance(result, dict):
            typed_list = result.get("TypedList")
            if isinstance(typed_list, dict):
                vec4 = typed_list.get("Vec4")
                if vec4 is not None:
                    out = []
                    for v in to_arr(vec4):
                        v = to_py(v)
                        if isinstance(v, dict):
                            out.append([
                                float(v.get("k_x", v.get("kx", v.get("x", 0.0)))),
                                float(v.get("k_y", v.get("ky", v.get("y", 0.0)))),
                                float(v.get("k_z", v.get("kz", v.get("z", 0.0)))),
                            ])
                        elif isinstance(v, (list, tuple)) and len(v) >= 2:
                            out.append([
                                float(v[0]),
                                float(v[1]),
                                float(v[2]) if len(v) > 2 else 0.0,
                            ])
                    if out:
                        return np.asarray(out, dtype=np.float64)
                kx = to_arr(typed_list.get("k_x", typed_list.get("kx", typed_list.get(0))))
                ky = to_arr(typed_list.get("k_y", typed_list.get("ky", typed_list.get(1))))
                kz = to_arr(typed_list.get("k_z", typed_list.get("kz", typed_list.get(2))))
                if kx or ky:
                    n = max(len(kx), len(ky), len(kz) if kz else 0)
                    kx = (kx + [0.0] * n)[:n]
                    ky = (ky + [0.0] * n)[:n]
                    kz = (kz + [0.0] * n)[:n] if kz else [0.0] * n
                    return np.stack([np.asarray(kx), np.asarray(ky), np.asarray(kz)], axis=1)
    if isinstance(result, (list, tuple)) and result:
        first = to_py(result[0])
        if hasattr(first, "data"):
            rows = []
            for kt in result:
                data = getattr(to_py(kt), "data", None)
                if data is not None:
                    rows.append(np.asarray(to_py(data), dtype=np.float64).ravel())
            if rows:
                return np.stack(rows, axis=0)
        if isinstance(first, (list, tuple, np.ndarray)):
            return np.asarray(result, dtype=np.float64)
    return None


def signal_to_complex_np(result) -> np.ndarray:
    result = to_py(result)
    if isinstance(result, dict):
        if result.get("Ok") is not None:
            result = to_py(result["Ok"])
        if isinstance(result, dict) and result.get("Error"):
            raise ValueError(f"simulation returned error: {result['Error']}")
        if isinstance(result, dict):
            typed_list = result.get("TypedList")
            if isinstance(typed_list, dict):
                complex_part = typed_list.get("Complex")
                if complex_part is not None:
                    complex_part = to_py(complex_part)
                    if isinstance(complex_part, dict):
                        real = to_arr(complex_part.get("real", complex_part.get("Real")))
                        imag = to_arr(complex_part.get("imag", complex_part.get("Imag")))
                        n = max(len(real), len(imag))
                        if n:
                            real = (real + [0.0] * n)[:n]
                            imag = (imag + [0.0] * n)[:n]
                            return (
                                np.asarray(real, dtype=np.float32)
                                + 1j * np.asarray(imag, dtype=np.float32)
                            )
                    result = complex_part
            if isinstance(result, dict) and result.get("List") is not None:
                result = result["List"]
    arr = np.asarray(to_py(result))
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64).ravel()
    if arr.ndim >= 2 and arr.shape[-1] >= 2:
        return (arr[..., 0].astype(np.float32) + 1j * arr[..., 1].astype(np.float32)).ravel()
    if arr.ndim == 1:
        out = []
        for item in arr:
            item = to_py(item)
            if isinstance(item, complex):
                out.append(item)
            elif isinstance(item, dict) and item.get("Complex") is not None:
                c = to_py(item["Complex"])
                if isinstance(c, dict):
                    out.append(
                        complex(
                            float(c.get("real", c.get("Real", 0.0))),
                            float(c.get("imag", c.get("Imag", 0.0))),
                        )
                    )
                elif isinstance(c, (list, tuple)) and len(c) >= 2:
                    out.append(complex(float(c[0]), float(c[1])))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append(complex(float(item[0]), float(item[1])))
            elif isinstance(item, (int, float, np.number)):
                out.append(complex(float(item), 0.0))
        if out:
            return np.asarray(out, dtype=np.complex64).ravel()
    raise ValueError(f"Unsupported signal format: shape={arr.shape}, dtype={arr.dtype}")


def parse_phantom_id(phantom_id) -> int:
    if isinstance(phantom_id, int):
        return phantom_id
    if isinstance(phantom_id, str):
        import re

        match = re.search(r"subject(\d+)", phantom_id, re.I)
        if match:
            return int(match.group(1))
        if phantom_id.isdigit():
            return int(phantom_id)
    raise ValueError(f"Cannot parse phantom id: {phantom_id!r}")


def seq_to_text(seq) -> str:
    if hasattr(seq, "write"):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".seq", delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            seq.write(str(tmp_path))
            return tmp_path.read_text(encoding="utf-8")
        finally:
            tmp_path.unlink(missing_ok=True)
    if isinstance(seq, str):
        if seq.startswith(("http://", "https://")):
            from urllib.request import urlopen

            return urlopen(seq, timeout=30).read().decode("utf-8")
        path = Path(seq)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        looks_like_path = path.suffix == ".seq" or "/" in seq or "\\" in seq
        if looks_like_path:
            raise FileNotFoundError(f"Pulseq sequence file not found: {seq}")
        return seq
    raise TypeError(f"Unsupported sequence type: {type(seq).__name__}")
