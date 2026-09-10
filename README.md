# Quant Convert GUI

A simple, browser-based GUI for quantizing `.safetensors` models to **FP8**,
**INT8**, **INT8 ConvRot**, or **NVFP4** — always `.safetensors` out, never
GGUF. Same formats used by the
[Kroma-Quant](https://huggingface.co/silveroxides/Kroma-Quant) and
[PotatoForge/Kroma-INT8-Quants](https://huggingface.co/PotatoForge/Kroma-INT8-Quants)
files.

It does not reimplement quantization math itself. It's a front end over
[`silveroxides/convert_to_quant`](https://github.com/silveroxides/convert_to_quant)
(the `ctq` CLI), the real tool that builds those files, so the output is
identical to what you'd get running `ctq` by hand — just without memorizing
its flags.

## Which format for your GPU

FP8 and NVFP4 need specific tensor-core hardware; picking one your GPU
doesn't have usually still *writes* a file, it just won't load or run fast.
The app's **Target GPU** picker on the Convert tab handles this for you, but
the short version:

| GPU generation | Example cards | Use |
|---|---|---|
| Turing/Ampere | RTX 20/30-series, A100 — **incl. RTX 3080 Ti** | **INT8 / INT8 ConvRot** (no FP8 or NVFP4 hardware path exists on these) |
| Ada Lovelace | RTX 40-series, L40 | FP8 or INT8 ConvRot |
| Hopper | H100, H200 | FP8 or INT8 ConvRot |
| Blackwell | RTX 50-series, B100/B200 | NVFP4, MXFP8, or INT8 ConvRot |

An RTX 3080 Ti (12GB, Ampere) has full native INT8 tensor cores, so INT8
ConvRot is both the fastest and the highest-quality option on it — pick
that, keep **Low memory mode** on, and it'll comfortably fit most diffusion
transformers in 12GB.

## Mixed precision (per-layer custom format)

[PotatoForge/Kroma-INT8-Quants](https://huggingface.co/PotatoForge/Kroma-INT8-Quants)'s
"mixed" files aren't one uniform format — their own `_quantization_metadata`
shows `attn.wk`/`attn.wv` left as plain tensorwise INT8 while `attn.wq`,
`attn.wo`, `attn.gate`, and every MLP layer get row-wise INT8 ConvRot. ctq
supports exactly this kind of per-layer split via `--custom-layers`, and the
**Advanced options → Mixed precision** section on the Convert tab exposes it:
pick a base format for most layers (step 3), then give a regex for the
layers that should get a different type/scaling/ConvRot instead. The regex
matches your *source* model's original tensor names, before any
`--comfy_quant` renaming — check your model's actual layer names first (e.g.
open it in the [safetensors metadata viewer on Hugging Face](https://huggingface.co/docs/safetensors)
or list keys locally with `safetensors.safe_open`).

Verified end to end against a real ctq install: a regex splitting
`attn.wq/wo/gate` + `mlp.*` from `attn.wk/wv` reproduces PotatoForge's exact
split (12 custom-ConvRot layers, 4 plain-tensorwise layers on a 2-block test
model).

## What does "txtfusion" in a filename mean?

`kroma-v0.3-txtfusion-edition-turbo-int8-convrot-simple.safetensors` isn't a
different quantization *format* — "txtfusion" is a **layer name**, and its
presence in the filename means that edition was converted with ctq's
`krea2` preset. That preset's actual definition (from
`convert_to_quant.constants.MODEL_FILTERS["krea2"]`) is:

```
highprec: ["firs", "las", "tml", "txtfusion", "last.modulatio", "tpro"]
```

i.e. any tensor whose name contains one of those substrings — including the
model's `txtfusion` layer (a text/image-fusion block, sensitive to
quantization like modulation and final layers usually are) — is kept at
full precision instead of being quantized. That's almost certainly why the
txtfusion-edition file quantizes better than a plain "blanket ConvRot, no
preset" conversion for this model family: it isn't quantizing everything.

**To get this yourself:** open the **Model preset (layer exclusions)**
accordion on the Convert tab and pick **krea2**. The GUI also does this
automatically now: if your input filename or Hugging Face URL contains
`txtfusion`, `krea`, or `kroma`, it auto-selects the `krea2` preset for you
(as long as you haven't already picked a different one), and tells you why
in the note under the file picker.

## Estimating output size

Hit **Estimate output size** (next to the command preview) after picking a
file and format. It reads only the input file's safetensors *header*
(tensor names/shapes/dtypes — a few KB, regardless of the model's actual
size) and estimates the converted size from your chosen format, preset, and
any custom-layer rules, without running ctq or needing torch/CUDA installed
yet. If a GPU was detected, it also flags whether the estimate fits in its
VRAM. It's a planning figure, not exact — it approximates ctq's per-layer
choices rather than replicating them (see `quant_gui/size_estimate.py`).

## About "INT4 ConvRot"

As of `convert_to_quant` v1.3.4, **there is no INT4 (integer 4-bit) output
format upstream** — only INT8 (including ConvRot), FP8, NVFP4, and MXFP8.
NVFP4 is a 4-bit *floating point* format and uses a different scheme than
ConvRot; it also requires a Blackwell GPU. The app is upfront about this: if
you pick "INT4 ConvRot" it explains the gap instead of pretending to convert.
The format list mirrors ctq's real CLI flags, so the moment ctq ships true
INT4, adding it here is a one-line change in `quant_gui/cli_builder.py`.

## Install (automatic)

**Windows:** double-click `install.bat` (or run it from a terminal). It
finds Python, creates a `.venv`, detects your NVIDIA GPU via `nvidia-smi`,
installs the matching PyTorch CUDA build and Triton automatically, installs
everything else, then launches the app.

**Linux/Mac:** `./install.sh` does the same thing.

Only requirement: [Python 3.10+](https://www.python.org/downloads/) already
installed and on PATH — the installer can't bootstrap Python itself (that
needs an elevated installer), but it'll open the download page for you if
it's missing. Safe to re-run any time; it skips whatever's already
installed.

Once installed, use `run.bat` (Windows) / `run.sh` (Linux/Mac) to start the
app without reinstalling anything, and `update.bat` / `update.sh` to pull
the latest code (if you cloned via git) and upgrade the GUI's own
dependencies (gradio, huggingface_hub, convert-to-quant, scipy). PyTorch and
Triton aren't touched by update — they're large GPU-specific downloads;
re-run install if you want a newer CUDA build.

<details>
<summary>What the installer actually detects</summary>

It runs `nvidia-smi` to read your driver's max supported CUDA version, then
picks the newest PyTorch wheel tag that driver can run (cu130 down to
cu118), or falls back to a CPU-only build if no NVIDIA GPU/driver is found.
On Windows it also installs `triton-windows`, version-constrained to match
the installed PyTorch release the way
[convert_to_quant's own README](https://github.com/silveroxides/convert_to_quant)
documents. None of this touches your system Python — everything installs
into the project's own `.venv`.

</details>

## Install (manual)

```bash
git clone <this repo>
cd quant-convert-gui
pip install -r requirements.txt

# PyTorch is deliberately not in requirements.txt — install the build that
# matches your GPU from https://pytorch.org/get-started/locally/, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Optional, speeds up INT8 kernels:
pip install -U triton          # Linux
pip install -U "triton-windows<3.7"   # Windows
```

Minimum: Python 3.10+, PyTorch 2.8+, CUDA 12.8+ for FP8/INT8. NVFP4/MXFP8
additionally need Python 3.12+, PyTorch 2.10+, CUDA 13.0+, and
[`comfy-kitchen`](https://github.com/silveroxides/comfy-kitchen).

## Run

```bash
python app.py
```

(or `run.bat` / `run.sh` if you used the automatic installer, which points
at the project's own `.venv` for you.)

Then open http://127.0.0.1:7860. The **Environment** tab tells you exactly
what's missing (ctq, PyTorch, CUDA, a compatible GPU) before you try to
convert anything.

## Using it

1. Paste a Hugging Face file URL (or a local path) to your model.
2. Pick a format — **INT8 ConvRot** is selected by default and is what the
   reference Kroma-Quant `*-int8-convrot-simple.safetensors` files use.
3. Optionally pick a model preset (keeps sensitive layers like norms/
   modulation at full precision) and a quality mode (Simple is fast;
   Learned/AdaRound is slower but higher quality).
4. Hit **Convert** and watch the live log. The finished file shows up under
   **Converted file** when it's done.

Advanced ctq flags (exclude-layers regex, device override, calibration
settings, etc.) are available under **Advanced options** for power users;
everyone else can ignore them.

## Project layout

- `install.bat` / `install.sh`, `run.bat` / `run.sh`, `update.bat` /
  `update.sh` — automated setup, launch, and updates.
- `scripts/setup_env.py` — the actual installer logic (GPU/CUDA detection,
  PyTorch/Triton install, final environment check); OS-agnostic Python, so
  the `.bat`/`.sh` wrappers are thin.
- `app.py` — the Gradio UI.
- `quant_gui/cli_builder.py` — turns GUI state into `ctq` CLI arguments (pure
  function, unit-testable without a GUI).
- `quant_gui/runner.py` — runs `ctq` as a subprocess and streams its output.
- `quant_gui/env_check.py` — detects ctq/PyTorch/CUDA/GPU availability.
- `quant_gui/gpu_profiles.py` — GPU-generation-to-format recommendations.
- `quant_gui/hf.py` — downloads a single file from a Hugging Face URL.
- `quant_gui/filters.py` — the model-family presets ctq exposes (e.g.
  `--flux2`, `--wan`, `--krea2` for txtfusion/Krea2/Kroma-style models).
- `quant_gui/size_estimate.py` — estimates output file size from the input
  file's safetensors header, without needing torch/ctq installed.
