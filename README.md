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

### Krea2 size profiles (for now, just this preset)

Once krea2 is selected, three quick-preset buttons appear right below it,
covering the space between "keep those layers full BF16" and "quantize
absolutely everything":

- **Balanced (recommended)** — krea2's normal behavior; largest, best quality.
- **Compact** — the krea2-sensitive layers (`txtfusion`, `last.modulatio`, etc.)
  get quantized too, but gently: plain row-wise INT8, no ConvRot rotation,
  via `--custom-layers` (which ctq applies with priority over the preset's
  exclusions). Meaningfully smaller than Balanced.
- **Smallest** — no exclusions at all; every layer gets the same INT8 ConvRot
  as everything else. Same as leaving the preset on `none`.

Verified against a real ctq run: on a test file matching every krea2
keyword, Balanced left 5 of 6 tensors unquantized (ctq logged `krea2 skip`
for each); Compact quantized all 6 (logged `custom INT8` for the 5 that
would've been skipped) and the output file came out ~44% smaller. These
aren't reproductions of PotatoForge's own undisclosed v3/v4/v5 recipes —
they're this GUI's own size/quality points, built from ctq's real flags.
Use **Estimate output size** to preview the difference before converting.

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

Real INT4/W4A4 ConvRot models exist and are in use — e.g.
[LAXMAYDAY/Krea-2-Turbo-int4-tensorwise-mixed](https://huggingface.co/LAXMAYDAY/Krea-2-Turbo-int4-tensorwise-mixed)
and [Lockout/krea2-comfy-int4-mixed](https://huggingface.co/Lockout/krea2-comfy-int4-mixed): packed signed
INT4 weights, group-256 Hadamard ConvRot, and a per-layer sensitivity
ranking mixing INT4 and int8_tensorwise layers, keeping the text-fusion
transformer and a handful of other sensitive layers in BF16 — the same
layers ctq's own `krea2` preset already protects.

`convert_to_quant` (ctq) itself still has **no INT4 CLI flag** on its
current `main` branch — there's an open, unaddressed request for it,
[INT4 ConvRot support? · Issue #50](https://github.com/silveroxides/convert_to_quant/issues/50),
no branch or PR attached — and Lockout's own model card explains they
found the technique by tracing LAXMAYDAY's recipe to the
[Starnodes Model Converter](https://github.com/Starnodes2024/comfyui-starnodes-modelconverter),
a ComfyUI custom node that can't run standalone outside ComfyUI.

Rather than fake this format or require ComfyUI, this app calls the actual
kernel underneath both of those — [`comfy_kitchen`](https://github.com/Comfy-Org/comfy-kitchen),
a standalone, pip-installable package (`comfy_kitchen.tensor.convrot_w4a4`)
that Starnodes wraps and that any future ctq INT4 support would presumably
also use. `quant_gui/int4_backend.py` is a small, independent converter
built directly on that kernel — **not** a wrapper around ctq or around the
Starnodes node:

- Install it with `pip install comfy-kitchen` (a separate, optional install —
  not pulled in by `install.bat`/`install.sh` or `requirements.txt`, since
  most formats in this app don't need it). The Environment tab and the
  in-app notice on this format both tell you if it's missing.
- Pick "INT4 ConvRot" as the format, then type a regex matching the layer
  names you want converted to real packed-signed INT4 (e.g. `attn\.wq|mlp\.gate`).
  Everything else quantizable falls back to plain INT8 tensorwise (via the
  same `comfy_kitchen` package, for metadata compatibility); your model
  preset's excluded/high-precision layers, and anything matched by the
  exclude-layers regex, stay at original precision either way.
- Needs Turing+ (SM 7.5+) for real speed — broader hardware support than
  FP8 (Ada+) — but also runs, slowly, on CPU via `comfy_kitchen`'s `eager`
  backend, so it works even without a GPU present.
- This is **not** a reproduction of LAXMAYDAY's or Lockout's undisclosed
  exact layer list — you choose which layers go INT4. It's verified against
  real `comfy_kitchen` output (correct `convrot_w4a4`/`int8_tensorwise`
  `_quantization_metadata`, correct packed tensor shapes), but it's this
  app's own recipe, not a copy of theirs.

If `comfy_kitchen` isn't installed, picking this format explains exactly
what to install and why, instead of silently failing or pretending to
convert.

## About "GGUF"

`convert_to_quant` only ever writes `.safetensors` — GGUF is llama.cpp's
own, completely different container/quantization format, read by ComfyUI
through the separate [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
custom node. Like INT4 ConvRot above, this bypasses ctq entirely: it calls
the [`gguf`](https://github.com/ggerganov/llama.cpp/tree/master/gguf-py)
package (llama.cpp's own Python bindings) directly, the same one
ComfyUI-GGUF's own `tools/convert.py` uses.

- **Real, pure-Python block quantization** for Q8_0/Q5_1/Q5_0/Q4_1/Q4_0
  (`gguf.quants`, confirmed against the installed package's source — no
  C++ build, no GPU, no ComfyUI install needed just to *produce* the
  file). F16/BF16 skip quantization entirely (just repacks into a GGUF
  container) — useful as input to your own `llama-quantize` run.
- **K-quants (Q4_K_M, Q5_K_S, Q6_K, etc.) are not available.** That family
  only has a *decoder* in the `gguf` package; producing them needs a
  patched `llama-quantize` binary compiled from llama.cpp source (see
  [ComfyUI-GGUF's `tools/README.md`](https://github.com/city96/ComfyUI-GGUF/blob/main/tools/README.md)).
  Convert to F16/BF16 here first if you want to run that yourself.
- **Architecture is auto-detected**, not guessed: `quant_gui/gguf_backend.py`
  ports the exact same detection keys (`double_blocks.0.img_attn.proj.weight`
  for Flux, etc.) that ComfyUI-GGUF's own `tools/convert.py` uses, because
  its loader hard-rejects any `general.architecture` value outside a fixed
  allowlist (flux, sd3, aura, hidream, cosmos, hyvid, wan, ltxv, sdxl, sd1,
  lumina2). An unrecognized model is refused rather than silently
  mislabeled.
- 1D tensors, tensors with ≤ 1024 elements, and each architecture's own
  precision-sensitive layers (e.g. Wan's `.modulation`, matching
  ComfyUI-GGUF's own script) always stay F32; a shape whose last dimension
  isn't divisible by 32 falls back to F16, exactly like `tools/convert.py`
  does. Your model preset / exclude-layers rules apply on top of that.

Verified end-to-end against a real GGUF round-trip (`gguf.GGUFReader`/
`quants.dequantize`): correct `general.architecture` field, correct
per-tensor `GGMLQuantizationType`, and the packed shapes ComfyUI-GGUF's own
loader validation expects.

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

**If you had `ctq` installed globally before using this app:** the GUI
always runs ctq through the *same Python interpreter that's running
`app.py`* (the project's own `.venv`, when launched via run.bat/run.sh) as
long as that interpreter has `convert_to_quant` installed - it deliberately
ignores any other `ctq` sitting elsewhere on PATH, so a stale or
differently-configured global install can't shadow it and silently break
things (e.g. missing `safetensors`/`scipy` from before those were added to
`requirements.txt`).

## Using it

1. Paste a Hugging Face file URL, or switch to **Local file path** — it
   auto-lists any `.safetensors` file already sitting in this app's
   `downloads/` folder (including ones you downloaded earlier), so you
   usually don't need to type/paste a path at all.
2. Pick a format — **INT8 ConvRot** is selected by default and is what the
   reference Kroma-Quant `*-int8-convrot-simple.safetensors` files use.
3. Optionally pick a model preset (keeps sensitive layers like norms/
   modulation at full precision) and a quality mode (Simple is fast;
   Learned/AdaRound is slower but higher quality).
4. Hit **Convert** and watch the live log, plus a progress bar tracking
   which tensor ctq is on (parsed from its own `(N/M) Processing: ...`
   output — no polling, no guessing). The finished file shows up under
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
