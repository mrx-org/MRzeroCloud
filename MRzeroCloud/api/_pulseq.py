"""PyPulseq helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

MAX_PULSEQ_VERSION = (1, 5, 0)
MAX_SEQ_LINES = 20000

_VERSION_FIELD = re.compile(r"^\s*(\w+)\s+(\S+)")


def _parse_pulseq_version(text: str) -> tuple[int, int, int]:
    fields: dict[str, int] = {}
    in_version = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[VERSION]":
            in_version = True
            continue
        if line.startswith("[") and in_version:
            break
        if not in_version:
            continue
        match = _VERSION_FIELD.match(line)
        if match is None:
            continue
        key = match.group(1).lower()
        if key in ("major", "minor", "revision"):
            try:
                fields[key] = int(float(match.group(2)))
            except ValueError:
                continue

    missing = {"major", "minor", "revision"} - fields.keys()
    if missing:
        raise ValueError(
            "Could not parse Pulseq [VERSION] (major/minor/revision) from sequence file"
        )
    return (fields["major"], fields["minor"], fields["revision"])


def check_pulseq_version(seq: str | Path) -> tuple[int, int, int]:
    """Reject unsupported ``.seq`` files before submission (no network I/O).

    The server runs ``pulseq_rs``, so the Pulseq version must be
    ``<= 1.5.0`` and the file must have at most 20000 lines.
    """
    path = Path(seq)
    if not path.is_file():
        raise FileNotFoundError(f"Sequence file not found: {seq}")

    text = path.read_text(encoding="utf-8", errors="replace")
    n_lines = len(text.splitlines()) or (1 if text else 0)
    if n_lines > MAX_SEQ_LINES:
        raise ValueError(
            f"Sequence file has {n_lines} lines "
            f"({MAX_SEQ_LINES} maximum currently supported)"
        )

    version = _parse_pulseq_version(text)
    if version > MAX_PULSEQ_VERSION:
        supported = ".".join(str(v) for v in MAX_PULSEQ_VERSION)
        found = ".".join(str(v) for v in version)
        raise ValueError(
            f"Pulseq version {found} is not supported (maximum {supported})."
        )
    return version


def coerce_sequence(result):
    if result is not None and hasattr(result, "write") and hasattr(result, "check_timing"):
        return result
    if isinstance(result, (list, tuple)):
        for item in result:
            if item is not None and hasattr(item, "write") and hasattr(item, "check_timing"):
                return item
    raise TypeError(f"Protocol returned {type(result).__name__}, not a PyPulseq sequence")


def seq_definition(seq, key: str):
    try:
        return seq.get_definition(key)
    except Exception:
        return None


def seq_definitions(seq) -> dict[str, Any]:
    return {
        "fov": seq_definition(seq, "fov") or seq_definition(seq, "FOV"),
        "matrix": seq_definition(seq, "matrix"),
    }


def recon_matrix_from_summary(summary: dict[str, Any], *, config: dict[str, Any] | None = None):
    """Prefer Pulseq definitions, fall back to AnyField recon metadata."""
    from ._config import optional_config, recon_matrix_from_config

    matrix = (summary.get("definitions") or {}).get("matrix")
    if matrix is not None:
        return (
            int(matrix[0]),
            int(matrix[1]),
            int(matrix[2]) if len(matrix) > 2 else 1,
        )
    metadata = summary.get("metadata") or config or optional_config()
    if metadata is not None:
        return recon_matrix_from_config(metadata)
    raise ValueError("No recon matrix in sequence definitions or metadata")
