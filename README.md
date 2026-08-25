# MRzeroCloud

**Version 1.0.2**

Cloud MRI simulation with an [MRzeroCore](https://mrsources.github.io/MRzero-Core/)-compatible Python API. Swap the import and run the same workflow against Fly ToolAPI backends or the Modal HTTP pipeline. `mr0.simulate()` defaults to the **modal** HTTP backend (same as MATLAB).

## Install

```bash
# from GitHub: https://github.com/mrx-org/MRzeroCloud
pip install git+https://github.com/mrx-org/MRzeroCloud.git
# from PyPI
pip install MRzeroCloud
# from a local clone
pip install -e .
```

Requires Python 3.9+, `pypulseq` for sequence building, and network access to Fly tools or a Modal HTTP endpoint.

The `modal` backend runs `pulseq_rs` server-side, so `.seq` files are validated locally
before upload: Pulseq version **≤ 1.5.0**, **≤ 20 000 lines**. Signal follows the
MRzeroCore 1.0 sign convention, so Cartesian recon uses **ifft**, not fft.

## Quick start (Modal API)

```python
import MRzeroCloud as mr0

signal, ktraj_adc = mr0.simulate("gre.seq")  # default modal backend

image = mr0.reco_pynufft(
    signal,
    ktraj_adc,
    resolution=(64, 64, 1),
    FOV=(0.256, 0.256, 0.003),
)
mr0.imshow(image)
```

`simulate`, `simulate_async`, `load_phantom`, `imshow`, and `stop_simulation` are also
available under `mr0.util` for backwards compatibility.

Examples:

- `examples/load_gre_sim_recon_nufft.py` — this modal quick start (`examples/gre.seq`, `reco_pynufft`)
- `examples/load_gre_sim_recon_fft.py` — same `gre.seq` and modal backend, iFFT recon
- `examples/load_gre_sim_recon_bifti.py` — same seq/backend, bifti phantom by id (`user/numerical_brain_cropped_bifti_2`)

## Modal HTTP backend

The `modal` backend talks to [tool-mr0sim-modal_http](../tool-mr0sim-modal_http): upload a `.seq` file, poll a job, download `(signal, ktraj)` as NPZ. No Fly conseq/phantomlib/trajex chain is required.

**Local dev** (from `tool-mr0sim-modal_http`):

```bash
python local_app.py   # http://127.0.0.1:8080
```

`simulate()` defaults to this backend. The default gateway is already configured, so no setup is needed:

```python
import MRzeroCloud as mr0

signal, ktraj = mr0.simulate("gre.seq")

# optional GPU tier (cpu, t4, a10g, a100); default t4
signal, ktraj = mr0.simulate("gre.seq", worker="a10g")
```

**Local dev or your own deployment** — point the backend somewhere else:

```bash
modal deploy modal_app.py
# use the printed gateway URL, e.g.
# https://YOUR-WORKSPACE--tool-mr0sim-modal-http-gateway.modal.run
```

```python
mr0.api.configure(urls={"modal": "http://127.0.0.1:8080"})
```

With no `config`, the `modal` backend uses the cached bifti phantom
`user/numerical_brain_cropped_bifti` on the **t4** worker pool, with `res`/`affine`
matching that phantom's native grid — identical to `mr0.defaultConfig()` in the
MATLAB package. The gateway reslices bifti phantoms server-side and requires both
`res` and `affine` on every job, so override them together via
`config["phantom_bifti"]`, `config["res"]`, and `config["affine"]`.

## Protocol mode

Generated protocol artifacts can register AnyField metadata:

```python
import MRzeroCloud as mr0

config = mr0.api.load_config(_anyfield_json)
obj = mr0.load_phantom(affine=config["affine"], res=config["res"])
signal, ktraj = mr0.simulate(seq, obj, backend=config["backend"])
image = mr0.reco_pynufft(signal, ktraj, resolution=config["recon_matrix"], FOV=fov)
```

Set `"backend": "mr0sim"` in protocol metadata to use the Fly ToolAPI chain. The default is ``modal``.

## Configuration

Override tool endpoints:

```python
mr0.api.configure(urls={
    "mr0sim": "wss://...",
    "modal": "http://127.0.0.1:8080",
})
```

Progress reporting:

```python
mr0.api.configure(on_message=lambda msg: print(msg) or True)
mr0.api.configure(verbose=False)   # silence built-in progress output
```

`configure` only applies the arguments you pass, and url overrides accumulate across
calls. Use `mr0.api.reset_configuration()` to go back to the built-in defaults.

## MATLAB (HTTP / modal only)

See [MRzerocloud_m](../MRzerocloud_m) for a MATLAB package with the same modal HTTP workflow (`mr0.simulate`, `mr0.configure`, `mr0.loadConfig`).

## vs MRzeroCore

| | MRzeroCore | MRzeroCloud |
|---|------------|-------------|
| Import | `import MRzeroCore as mr0` | `import MRzeroCloud as mr0` |
| Simulate | Local Bloch graph | Fly ToolAPI or Modal HTTP |
| Phantom | `VoxelGridPhantom` / `.mat` | phantomlib (Fly) or bifti registry (modal) |
| Recon | `mr0.reco_adjoint` | `mr0.reco_adjoint` (same API); `mr0.reco_pynufft` for NUFFT |

## Release notes

### 1.0.2

- `simulate()` defaults to the **modal** HTTP backend (Fly ToolAPI: `backend="mr0sim"`)
- Install from GitHub: `pip install git+https://github.com/mrx-org/MRzeroCloud.git`
- Examples: `examples/load_gre_sim_recon_nufft.py` (`reco_pynufft`), `examples/load_gre_sim_recon_fft.py` (iFFT), and `examples/load_gre_sim_recon_bifti.py` (bifti id), bundled `examples/gre.seq`
- `mr0.reco_adjoint` matches the MRzeroCore API (dense adjoint DFT)

### 1.0.1

Package version tracks the MRzeroCore version on mr0-cloud.

- Fly ToolAPI (`mr0sim`) and Modal HTTP simulation
- `simulate`, `simulate.start` / `simulate_async`, `load_phantom`, `reco_pynufft`, `imshow`
- Modal defaults: cached bifti `user/numerical_brain_cropped_bifti` on the **t4** worker
- Pre-flight `.seq` checks: Pulseq version **≤ 1.5.0**, file **≤ 20 000 lines**
- Cartesian recon uses **ifft** for the MRzeroCore 1.0 signal sign convention
