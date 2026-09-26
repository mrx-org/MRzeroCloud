"""Local CUDA vs cloud T4, with A10 optional.

With no arguments, every ``*.seq`` in ``test/test_seqs`` is compared to the
T4 mrzero and T4 pdgv2 workers. Pass ``a10`` to also run A10 mrzero and A10
pdgv2. Other arguments are file names or stems and select a subset of
sequences. Writes ``test/results/core_vs_cloud.md`` with one runtime table
and one figure per sequence.

    python test/test_core_vs_clouds.py
    python test/test_core_vs_clouds.py a10
    python test/test_core_vs_clouds.py pp_tse_1.4.seq mrseq_spiral_1.5
    python test/test_core_vs_clouds.py a10 pp_tse_1.4.seq
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TESTS = Path(__file__).resolve().parent
SEQ_DIR = TESTS / "test_seqs"
OUT = TESTS / "results"
FIG_DIR = OUT / "figures"
PHANTOM = "user/numerical_brain_cropped_bifti"
MRZERO_URL = "https://mzaiss--tool-mr0sim-modal-http-gateway.modal.run"
PDGV2_URL = "https://mzaiss--pdgv2-gateway.modal.run"
T4_CASES = (
    ("T4 mrzero", "t4", "mrzero", MRZERO_URL),
    ("T4 pdgv2", "t4", "pdgv2", PDGV2_URL),
)
A10_CASES = (
    ("A10 mrzero", "a10g", "mrzero", MRZERO_URL),
    ("A10 pdgv2", "a10g", "pdgv2", PDGV2_URL),
)
A10_TOKENS = {"a10", "a10g"}
ROWS = (
    ("seq_load", "seq load"),
    ("resample", "resample"),
    ("prepass", "prepass"),
    ("main_pass", "main pass"),
    ("recon", "recon"),
)
CLOUD_KEYS = {
    "seq_load": "seq_load",
    "resample": "obj_resample",
    "prepass": "pre_pass",
    "main_pass": "main_pass",
    "recon": "recon",
}


def _defs(seq_path: Path) -> dict[str, str]:
    text = seq_path.read_text(encoding="utf-8", errors="replace")
    block = text.split("[DEFINITIONS]", 1)[1].split("[", 1)[0]
    out = {}
    for line in block.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = " ".join(parts[1:])
    return out


def _grid(seq_path: Path) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
    defs = _defs(seq_path)
    fov_vals = [float(v) for v in defs["FOV"].split()]
    fov = (fov_vals[0], fov_vals[1], fov_vals[2] if len(fov_vals) > 2 else 0.003)
    if "ReconMatrix" in defs:
        m = [int(float(v)) for v in defs["ReconMatrix"].split()]
        res = (m[0], m[1], m[2] if len(m) > 2 else 1)
    else:
        adc_block = seq_path.read_text(encoding="utf-8", errors="replace").split("[ADC]", 1)[1].split("[", 1)[0]
        adc = None
        for line in adc_block.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "1" and parts[1].isdigit():
                adc = int(parts[1])
                break
        over = int(float(defs.get("ReadoutOversamplingFactor", "1")))
        n = max((adc or 64) // over, 1)
        res = (n, n, 1)
    return fov, res


def _fmt(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _reco(signal, kspace, fov, res):
    import MRzeroCloud as mr0

    t0 = time.perf_counter()
    image = np.squeeze(
        mr0.reco_pynufft(np.asarray(signal).reshape(-1), np.asarray(kspace), resolution=res, FOV=fov)
    )
    return time.perf_counter() - t0, np.abs(image)


def time_core(seq_path: Path, fov, res) -> tuple[dict[str, str], np.ndarray]:
    import torch
    import MRzeroCore as mr0

    os.chdir(TESTS)
    steps: dict[str, float] = {}
    t0 = time.perf_counter()
    seq = mr0.Sequence.import_file(str(seq_path))
    steps["seq_load"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    data = mr0.util.load_phantom().build()
    steps["resample"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    graph = mr0.compute_graph(seq, data, 2000, 1e-5)
    steps["prepass"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        signal = mr0.execute_graph(graph, seq.cuda(), data.cuda(), 1e-3, 1e-3, print_progress=False).cpu()
    else:
        signal = mr0.execute_graph(graph, seq, data, 1e-3, 1e-3, print_progress=False)
    steps["main_pass"] = time.perf_counter() - t0
    recon_s, mag = _reco(signal, seq.get_kspace(), fov, res)
    steps["recon"] = recon_s
    shown = {key: _fmt(value) for key, value in steps.items()}
    shown["total"] = _fmt(sum(steps.values()))
    return shown, mag


def _parse_complete(message: str) -> dict[str, str]:
    body = message.split("durations:", 1)[-1]
    out = {}
    for part in body.split(","):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _cloud(seq_path: Path, fov, res, worker: str, sim_backend: str, base_url: str):
    import MRzeroCloud as mr0

    captured: list[str] = []

    def _log(msg: str) -> bool:
        text = msg.encode("ascii", "replace").decode("ascii")
        print(f"[{worker}/{sim_backend}] {text}", flush=True)
        captured.append(text)
        return True

    mr0.api.configure(urls={"modal": base_url}, on_message=_log)
    config = mr0.api.default_modal_config()
    config["phantom_bifti"] = PHANTOM
    config["sim_backend"] = sim_backend
    t0 = time.perf_counter()
    signal, ktraj = mr0.simulate(str(seq_path), config=config, worker=worker)
    wall = time.perf_counter() - t0
    recon_s, mag = _reco(signal, ktraj, fov, res)
    message = next((line for line in captured if line.startswith("Complete, durations:")), "(none)")
    parsed = _parse_complete(message)
    shown = {row: parsed.get(cloud, "—") for row, cloud in CLOUD_KEYS.items()}
    shown["total"] = _fmt(wall)
    shown["recon_client"] = _fmt(recon_s)
    return shown, mag, message


def _corr(a, b) -> float:
    x = np.abs(a).ravel().astype(np.float64)
    y = np.abs(b).ravel().astype(np.float64)
    x = x / (np.percentile(x, 99.5) + 1e-12)
    y = y / (np.percentile(y, 99.5) + 1e-12)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _save_figure(name: str, panels: list[tuple[str, np.ndarray]]) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, mag) in zip(axes, panels):
        scale = np.percentile(mag, 99.5) or 1.0
        im = ax.imshow((mag / scale).T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(name)
    fig.tight_layout()
    path = FIG_DIR / f"{Path(name).stem}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _split_args(names: list[str]) -> tuple[list[str], tuple]:
    seq_names = []
    include_a10 = False
    for name in names:
        if name.lower() in A10_TOKENS:
            include_a10 = True
        else:
            seq_names.append(name)
    cases = T4_CASES + (A10_CASES if include_a10 else ())
    return seq_names, cases


def _sequences(names: list[str]) -> list[Path]:
    files = sorted(SEQ_DIR.glob("*.seq"))
    if not files:
        raise SystemExit(f"no sequences in {SEQ_DIR}")
    if not names:
        return files
    by_key = {}
    for path in files:
        by_key[path.name] = path
        by_key[path.stem] = path
    chosen = []
    missing = []
    for name in names:
        path = by_key.get(name)
        if path is None:
            missing.append(name)
        elif path not in chosen:
            chosen.append(path)
    if missing:
        raise SystemExit(f"not in {SEQ_DIR.name}: {', '.join(missing)}")
    return chosen


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "|---|" + "|".join("---:" for _ in headers[1:]) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _write_report(records: list[dict], cases: tuple) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Core vs cloud",
        "",
        f"Run on {date.today().isoformat()}. Phantom `{PHANTOM}`.",
        "Local times are MRzeroCore `util.simulate` thresholds (`min_state_mag=1e-5`, main pass `1e-3`).",
        "Cloud step times are the worker complete line. Wall time includes queue and the client NUFFT.",
        "Worker `recon` is 0 s because reconstruction runs on the client; that client time is the `recon (client)` row.",
        "",
        "## Runtime summary",
        "",
    ]
    headers = ["Sequence", "Local total", *[label for label, *_ in cases]]
    summary_rows = []
    for rec in records:
        summary_rows.append([
            f"`{rec['name']}`",
            rec["local"]["total"],
            *[rec["clouds"][label]["total"] for label, *_ in cases],
        ])
    lines.append(_md_table(headers, summary_rows))
    lines.append("")
    corr_headers = ["Sequence", *[label for label, *_ in cases]]
    corr_rows = []
    for rec in records:
        corr_rows.append([
            f"`{rec['name']}`",
            *[f"{rec['corrs'][label]:.4f}" for label, *_ in cases],
        ])
    lines.extend(["Magnitude correlation vs local:", "", _md_table(corr_headers, corr_rows), ""])

    for i, rec in enumerate(records, start=1):
        fov = rec["fov"]
        res = rec["res"]
        lines.extend([
            f"## {rec['name']}",
            "",
            f"FOV {fov[0]:.3f} × {fov[1]:.3f} × {fov[2]:.3f} m, reconstruction {res[0]}×{res[1]}×{res[2]}.",
            "",
        ])
        step_headers = ["Step", "Local RTX 4070", *[label for label, *_ in cases]]
        step_rows = []
        for key, title in ROWS:
            step_rows.append([
                title,
                rec["local"][key],
                *[rec["clouds"][label][key] for label, *_ in cases],
            ])
        step_rows.append([
            "recon (client)",
            rec["local"]["recon"],
            *[rec["clouds"][label]["recon_client"] for label, *_ in cases],
        ])
        step_rows.append([
            "total / wall",
            rec["local"]["total"],
            *[rec["clouds"][label]["total"] for label, *_ in cases],
        ])
        lines.extend([_md_table(step_headers, step_rows), ""])
        rel = Path("figures") / rec["figure"].name
        corr_txt = ", ".join(f"{label} {rec['corrs'][label]:.4f}" for label, *_ in cases)
        lines.extend([
            f'<figure>',
            f'<img src="{rel.as_posix()}" alt="{rec["name"]} magnitude images">',
            "<figcaption>",
            f"Figure {i}. <code>{rec['name']}</code>. "
            f"Each panel is scaled to its own 99.5th percentile. "
            f"Magnitude correlation vs local: {corr_txt}.",
            "</figcaption>",
            "</figure>",
            "",
        ])
    path = OUT / "core_vs_cloud.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare local MRzeroCore with cloud GPU workers.")
    parser.add_argument(
        "sequences",
        nargs="*",
        help="Sequence names or stems in test/test_seqs. Pass a10 to also run A10. Default: all sequences, T4 only.",
    )
    args = parser.parse_args(argv)
    seq_names, cases = _split_args(args.sequences)
    seqs = _sequences(seq_names)
    records = []
    for seq_path in seqs:
        fov, res = _grid(seq_path)
        print(f"\n=== {seq_path.name} FOV {fov} res {res} ===", flush=True)
        local, mag_l = time_core(seq_path, fov, res)
        print("local", local, flush=True)
        panels = [("local RTX 4070", mag_l)]
        clouds = {}
        corrs = {}
        for label, worker, backend, base_url in cases:
            shown, mag, http = _cloud(seq_path, fov, res, worker, backend, base_url)
            corr = _corr(mag_l, mag)
            panels.append((label, mag))
            clouds[label] = shown
            corrs[label] = corr
            print(label, http, "wall", shown["total"], "corr", f"{corr:.4f}", flush=True)
        figure = _save_figure(seq_path.name, panels)
        records.append({
            "name": seq_path.name,
            "fov": fov,
            "res": res,
            "local": local,
            "clouds": clouds,
            "corrs": corrs,
            "figure": figure,
        })
    report = _write_report(records, cases)
    print("report", report, flush=True)


if __name__ == "__main__":
    main()
