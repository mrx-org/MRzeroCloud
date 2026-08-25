# MRzeroCloud

Cloud MRI simulation with an [MRzeroCore](https://mrsources.github.io/MRzero-Core/)-compatible Python API. Swap the import and run the same workflow against Fly ToolAPI backends or the Modal HTTP pipeline.

## Install

```bash
pip install MRzeroCloud
# or from this repo:
pip install -e .
```

Requires Python 3.9+, `pypulseq` for sequence building, and network access to Fly tools or a Modal HTTP endpoint.

The `modal` backend runs `pulseq_rs` server-side, so `.seq` files are validated locally
before upload: Pulseq version **≤ 1.5.0**, **≤ 20 000 lines**. Signal follows the
MRzeroCore 1.0 sign convention, so Cartesian recon uses **ifft**, not fft.

## Quick start (Fly ToolAPI)

```python
import MRzeroCloud as mr0
import pypulseq as pp

seq = build_your_sequence()   # PyPulseq Sequence
seq.write("out.seq")          # standard PyPulseq I/O

obj = mr0.load_phantom(4, res=(64, 64, 1))
signal, ktraj_adc = mr0.simulate(seq, obj, accuracy=1e-5)

image = mr0.reco_pynufft(
    signal,
    ktraj_adc,
    resolution=(64, 64, 1),
    FOV=(0.22, 0.22, 0.003),
)
mr0.imshow(image)
```

`simulate`, `simulate_async`, `load_phantom`, `imshow`, and `stop_simulation` are also
available under `mr0.util` for backwards compatibility.

## Modal HTTP backend

The `modal` backend talks to [tool-mr0sim-modal_http](../tool-mr0sim-modal_http): upload a `.seq` file, poll a job, download `(signal, ktraj)` as NPZ. No Fly conseq/phantomlib/trajex chain is required.

**Local dev** (from `tool-mr0sim-modal_http`):

```bash
python local_app.py   # http://127.0.0.1:8080
```

The default gateway is already configured, so no setup is needed:

```python
import MRzeroCloud as mr0

signal, ktraj = mr0.simulate("gre.seq", backend="modal")

# optional GPU tier (cpu, t4, a10g, a100); default t4
signal, ktraj = mr0.simulate("gre.seq", backend="modal", worker="a10g")
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

Set `"backend": "modal"` in protocol metadata to use the HTTP pipeline.

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
| Recon | `mr0.reco_adjoint` | `mr0.reco_pynufft` (portable to Core later) |
