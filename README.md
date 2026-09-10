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

## About "INT4 ConvRot"

As of `convert_to_quant` v1.3.4, **there is no INT4 (integer 4-bit) output
format upstream** — only INT8 (including ConvRot), FP8, NVFP4, and MXFP8.
NVFP4 is a 4-bit *floating point* format and uses a different scheme than
ConvRot; it also requires a Blackwell GPU. The app is upfront about this: if
you pick "INT4 ConvRot" it explains the gap instead of pretending to convert.
The format list mirrors ctq's real CLI flags, so the moment ctq ships true
INT4, adding it here is a one-line change in `quant_gui/cli_builder.py`.

## Install

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

- `app.py` — the Gradio UI.
- `quant_gui/cli_builder.py` — turns GUI state into `ctq` CLI arguments (pure
  function, unit-testable without a GUI).
- `quant_gui/runner.py` — runs `ctq` as a subprocess and streams its output.
- `quant_gui/env_check.py` — detects ctq/PyTorch/CUDA/GPU availability.
- `quant_gui/hf.py` — downloads a single file from a Hugging Face URL.
- `quant_gui/filters.py` — the model-family presets ctq exposes (e.g.
  `--flux2`, `--wan`, `--krea2` for txtfusion/Krea2/Kroma-style models).
