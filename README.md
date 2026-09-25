# Quant Convert GUI

A simple, browser-based GUI for quantizing `.safetensors` models to **FP8**,
**INT8**, **INT8 ConvRot**, **NVFP4**, real **INT4 ConvRot**, or **GGUF**.
The `.safetensors` formats are the same ones used by the
[Kroma-Quant](https://huggingface.co/silveroxides/Kroma-Quant) and
[PotatoForge/Kroma-INT8-Quants](https://huggingface.co/PotatoForge/Kroma-INT8-Quants)
files.

For FP8/INT8/INT8 ConvRot/NVFP4/MXFP8, this app does not reimplement
quantization math itself — it's a front end over
[`silveroxides/convert_to_quant`](https://github.com/silveroxides/convert_to_quant)
(the `ctq` CLI), the real tool that builds those files, so the output is
identical to what you'd get running `ctq` by hand — just without memorizing
its flags. INT4 ConvRot and GGUF are formats `ctq` doesn't produce, so those
two go through their own independent backends instead (`comfy_kitchen` and
`gguf` respectively) — see [About "INT4 ConvRot"](#about-int4-convrot) and
[About "GGUF"](#about-gguf) below.

There's also a separate **LLM → GGUF** tab for converting text models
(Gemma, Llama, Qwen, etc.) — a genuinely different pipeline built around a
real [llama.cpp](https://github.com/ggerganov/llama.cpp) checkout rather
than any of the above; see
[About "LLM to GGUF"](#about-llm-to-gguf-gemma-llama-qwen-etc) below.
An **Image → GGUF** tab covers Krea 2 (below too), and the **Inspector**,
**Tools**, and **History** tabs round out the app
([About the other tabs](#about-the-other-tabs)).

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
- **Parallel + optional GPU quantization** — see [Performance](#performance).

Verified end-to-end against a real GGUF round-trip (`gguf.GGUFReader`/
`quants.dequantize`): correct `general.architecture` field, correct
per-tensor `GGMLQuantizationType`, and the packed shapes ComfyUI-GGUF's own
loader validation expects.

## About "LLM to GGUF" (Gemma, Llama, Qwen, etc.)

This is a separate tab and a separate tool from everything above it.

Everything above is for diffusion models. Text LLMs need a real, different
pipeline: proper tokenizer/vocab conversion plus per-architecture
hyperparameter mapping (attention head count, rope settings, etc.), which
`quant_gui/gguf_backend.py` above doesn't attempt - that's exactly what
[llama.cpp](https://github.com/ggerganov/llama.cpp) itself implements, in
a `conversion/` package that's 90+ files and pinned to an exact
`transformers` version as of this writing. There's no equivalent
lightweight pip package, and reimplementing it here would mean re-chasing
every new model release by hand.

So the **LLM → GGUF** tab manages a real llama.cpp checkout instead
(cloned into `llama.cpp/`, in its *own* venv so its pinned deps never touch
this app's), and shells out to its actual tools:

- **`convert_hf_to_gguf.py`** does the HF → GGUF step - real tokenizer
  conversion, real per-architecture metadata, whatever llama.cpp supports
  (Gemma included). Direct output is limited to F32/F16/BF16/Q8_0 (the
  same pure-Python-quantizable types as the diffusion GGUF backend, for
  the same reason).
- **`llama-quantize`**, once built, produces real K-quants (Q4_K_M, Q6_K,
  etc. - what most LLM GGUFs on Hugging Face actually are). Building it
  needs a C/C++ toolchain and `cmake` already on your machine (Visual
  Studio Build Tools on Windows, `build-essential` on Linux, Xcode command
  line tools on Mac) - a bigger ask than any pip install this app has
  needed so far, which is why it's a separate, optional step with its own
  button, not bundled into setup automatically.
- A **Hugging Face repo ID** downloads the *whole* model repo (config,
  tokenizer, every safetensors shard) via `huggingface_hub.snapshot_download`
  - unlike the Convert tab above, which only ever needs one file.
- **`llama-imatrix`**, once built, generates a real importance matrix -
  the same mechanism behind most "imatrix" GGUF quants on Hugging Face,
  and the foundation Unsloth's own "Dynamic" quants are built on (per
  their own docs: an imatrix calibration file that "helps the quantizer
  identify model layers that need more precision"). It runs calibration
  text through the full-precision model and records which weights
  actually matter, then `llama-quantize --imatrix ...` uses that to round
  more carefully where it counts. A small generic calibration text ships
  bundled (`quant_gui/data/default_calibration.txt`) so the feature works
  with no setup; paste your own file for better results on a specific
  domain - quality depends heavily on how representative the calibration
  text is of real usage, which is most of what Unsloth's own curation
  actually does.
- **Per-layer type overrides**, via `llama-quantize`'s own
  `--tensor-type-file` (a plain text file of `tensor.name=GGML_TYPE`
  lines) - the real mechanism behind manually assigning a different type
  per layer, which is what Unsloth's "Dynamic 3.0" per-layer mixing is
  built from. Paste any such file into step 5's override box to quantize
  with it directly.
- **"Find best quant (sweep)"** automates the search: it runs the
  target-size family through `llama-quantize` on your file (generating an
  imatrix first if none is given), measures size, time, and
  reconstruction error vs the input, and pre-selects the winner in the
  Quant type dropdown. Weight-space error, not perplexity - it ranks
  settings, it doesn't judge generation quality.
- **Smart per-tensor tuning (Dynamic-quant style)** generates a
  `--tensor-type-file` automatically: it ranks every tensor's
  quantization sensitivity using your imatrix as the
  activation-importance weight (minutes, pure Python, no GPU), then
  greedily assigns per-tensor types under a size budget and quantizes
  with the result. "K-ladder" mode maps sensitivity onto q3_k..q6_k;
  "Legacy" uses only pure-Python-measurable types with exact errors.
  Embeddings, the output tensor, and MoE routers are never dropped below
  the floor. MoE models get a two-zone recipe (Q8_0-level attention/
  shared FFN, IQ-quantized experts). Target-size presets
  (~3/~4/~5 bpw families) compute the budget from parameter count;
  "Custom" takes a GB figure.
- **"Auto (sweep → tune → validate)"** chains the whole tuning pipeline in
  one click: sweep the family, per-tensor-tune the winner, then compare
  perplexity of the tuned file vs a plain baseline quant on a held-out
  text (something the calibration didn't see) with llama-perplexity.
  Stages reuse each other's artifacts, so re-running skips work already
  done.
- **KV-cache hints** for Ollama: the companion `tools/ollama_modelfile.py`
  emits a `Modelfile` with the right `num_ctx`/flash-attention settings
  for your GPU (see [docs/OLLAMA.md](docs/OLLAMA.md) for the thinking-mode
  recipe on Gemma 4).

What this is *not*: a reproduction of Unsloth's exact per-model layer
choices (still unpublished). It's the same real llama.cpp mechanisms -
imatrix + per-layer types + a size budget - driven by *your* calibration
data instead of theirs.

Verified end-to-end against a real synthetic Llama-architecture model
(tiny SentencePiece tokenizer, trained with byte-fallback so it can encode
arbitrary calibration text, + tiny safetensors weights, built with
`transformers`/`sentencepiece` directly, since no real model was
downloaded for this) run through the actual cloned `convert_hf_to_gguf.py`,
a real compiled `llama-quantize` binary, and a real compiled
`llama-imatrix` binary - correct tensor names/shapes, correct
hyperparameters (context length, head counts, rope theta), correct
tokenizer/special-token IDs, a real generated importance matrix
(`llama-quantize`'s own log confirms `have_imatrix`/entries loaded), a
real per-layer type override applied via `--tensor-type-file` (confirmed
by reading the exact GGML type back out of the output file), and a real
Q4_K_M output file that `gguf.GGUFReader` reads back with the right
architecture and per-tensor quant types. The smart-tuning pipeline is
covered by its own tests: imatrix loading (both legacy and GGUF
container formats), sensitivity scoring, budget assignment on synthetic
tensors, and the sweep's candidate ranking.

## About "Image → GGUF" (Krea 2)

A dedicated tab for **Krea 2 (Raw / Turbo)** - the 12.9B text-to-image
diffusion transformer - which the generic GGUF backend above deliberately
refuses (the krea2 arch is outside ComfyUI-GGUF's allowlist, so a
community-patched loader fork is needed anyway). It wraps the repo's own
`tools/convert_krea2_to_gguf.py` (pure numpy + gguf-py, no torch): only
the DiT goes into the GGUF with `arch = krea2` and ComfyUI-native tensor
names; the text encoder (Qwen3-VL fp8) and VAE (Qwen-Image) stay separate
safetensors, with download buttons for all three on the tab. Quant
choices are q8_0 down to q4_0 plus lossless bf16. See
[docs/KREA2.md](docs/KREA2.md) for the full precision rules, the ComfyUI
loading recipe, and the two-stage K-quant flow (a
`tools/krea2_kquant_plan.py` sensitivity planner that emits a
`--tensor-type-file` template for a krea2-patched `llama-quantize`).

## About the other tabs

- **Editor** - edit a GGUF's metadata and save a new file: change values
  (`general.name`, `tokenizer.chat_template`, ...), add keys, delete keys.
  Tensor data is copied **byte-for-byte** (never re-quantized) and streams
  through disk in constant memory - a metadata fix on a 26 GB model takes
  ~1 minute and under 100 MB of RAM. A local, offline equivalent of the
  Hugging Face *GGUF Editor* space. Writes are crash-atomic; the backend
  (`quant_gui/gguf_edit.py`) also supports tensor rename/drop for scripts.
- **Inspector** - read-only look inside any GGUF: architecture, type
  histogram, bits-per-weight, biggest tensors, metadata. Nothing is
  loaded or executed, so inspection takes seconds even for 100 GB files.
- **History** - every finished conversion (from `run_history.json`),
  newest first, with the metrics that make runs comparable: output size,
  wall time, tensor count, and a speed verdict against the previous run
  of the same conversion.
- **Tools** - wrappers around the repo's CLI tools, same code as the
  command line with logs streamed into the UI: **Compare GGUFs**
  (inference-free quality report - reconstruction error per tensor,
  optionally imatrix-weighted, vs a reference; much faster than a
  perplexity run), **Ollama Modelfile** (see
  [docs/OLLAMA.md](docs/OLLAMA.md)), and a pointer to the Image → GGUF
  tab.

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
pip install -r requirements.txt   # pulls ctq from the 8bit-boom main branch
                                  # (native stop/resume checkpoint support)

# PyTorch is deliberately not in requirements.txt — install the build that
# matches your GPU from https://pytorch.org/get-started/locally/, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Optional, speeds up INT8 kernels:
pip install -U triton          # Linux
pip install -U "triton-windows<3.7"   # Windows
```

> ctq note: `requirements.txt` installs `convert-to-quant` from
> [8bit-boom/convert_to_quant](https://github.com/8bit-boom/convert_to_quant)
> `@main` rather than PyPI, because that build carries the `--checkpoint-dir`
> / `--stop-file` stop-and-resume support the Pause/Stop/Resume buttons use.
> Stock PyPI ctq works too — the GUI detects the missing support and Stop
> falls back to the legacy terminate + restart-from-scratch snapshot.

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

**Converting several models with the same settings:** list one path per
line in the **Batch queue** box (or pick a primary local file) and hit
**Convert all (batch)** - auto output naming is forced so items can't
overwrite each other, and Stop/Pause applies to the current item.

Advanced ctq flags (exclude-layers regex, device override, calibration
settings, etc.) are available under **Advanced options** for power users;
everyone else can ignore them.

## Pausing & resuming

Conversions can run for hours, so the Convert tab has progress controls:

- **⏸ Pause / ▶ Resume** freezes the running conversion (the whole ctq
  process is suspended — SIGSTOP/SIGCONT on Linux, thread suspension on
  Windows) and continues it exactly where it was. The app needs to stay
  open for this.
- **⏹ Stop & save progress** (with *Save resumable progress* checked)
  stops the run after the current tensor and writes a **checkpoint** to the
  `checkpoints/` folder: every finished tensor's data plus a manifest.
  INT4 ConvRot and GGUF conversions can then be **resumed from exactly the
  tensor where they stopped**, even after closing and reopening the app —
  finished tensors are replayed from the checkpoint instead of being
  recomputed, and the completed output is identical to an uninterrupted
  run (covered by unit tests). Checkpoints are cleaned up automatically
  once a run finishes; unfinished ones can also be deleted from the
  **Saved checkpoints** list.
- `ctq` itself (FP8/INT8/NVFP4/MXFP8) runs as a subprocess. With a
  checkpoint-aware ctq build (one that supports `--checkpoint-dir`), stopping
  saves **per-tensor progress** and *Resume from checkpoint* continues from
  the last finished tensor — the same true-resume semantics as INT4/GGUF.
  With older ctq builds, stopping saves a **session snapshot** (the exact
  command plus all your settings) and resume relaunches that identical
  conversion with one click; any partially written output file is kept under
  a `*.partial-<timestamp>` name rather than left to be overwritten.

## Performance

Two speedups apply to the GGUF path (image/DiT models); both are on by
default where applicable and neither changes the output:

- **Parallel quantization** (always on): tensors are prefetched and
  quantized across a small thread pool while the previous tensor is still
  being written. Measured ~2.6× on a typical DiT vs strictly sequential
  conversion; output is byte-identical.
- **GPU quantization** (opt-in): the **"Use GPU quantization (Q8_0)"**
  checkbox on the GGUF tab runs the Q8_0 block quantizer on your NVIDIA
  GPU via a Triton kernel instead of numpy. Requirements: a CUDA GPU plus
  `pip install triton-windows` (Windows; on Linux, `pip install triton`)
  and a CUDA build of torch. If any piece is missing it silently falls
  back to the CPU path. The checkbox choice is saved across launches
  (`ui_settings.json`).

  Measured on an RTX 5090 (1 GiB of f32 weights, warm kernel):
  ~0.09 s on GPU vs 2.5 s numpy single-thread (~28×), and ~12× vs the
  4-thread CPU path above — so it stays a real win on top of the
  threading. The kernel is **bit-exact** against the numpy reference
  (verified over 8M+ blocks, including rounding ties and zero blocks);
  CPU and GPU runs produce byte-identical files. Without a GPU it also
  works headless: set `QUANT_GUI_GPU_QUANT=1` in the environment.

## Project layout

- `install.bat` / `install.sh`, `run.bat` / `run.sh`, `update.bat` /
  `update.sh` — automated setup, launch, and updates.
- `scripts/setup_env.py` — the actual installer logic (GPU/CUDA detection,
  PyTorch/Triton install, final environment check); OS-agnostic Python, so
  the `.bat`/`.sh` wrappers are thin.
- `app.py` — the Gradio UI.
- `quant_gui/cli_builder.py` — turns GUI state into `ctq` CLI arguments (pure
  function, unit-testable without a GUI).
- `quant_gui/runner.py` — runs `ctq` as a subprocess and streams its output;
  also implements process suspension for the Pause button.
- `quant_gui/run_control.py` — the pause/cancel switches the Pause/Stop
  buttons flip on whichever conversion is currently running.
- `quant_gui/checkpoints.py` — on-disk checkpoints (manifest + per-tensor
  shards) that make Stop-&-resume possible; ctq runs store a session
  snapshot instead (see [Pausing & resuming](#pausing--resuming)).
- `quant_gui/env_check.py` — detects ctq/PyTorch/CUDA/GPU availability.
- `quant_gui/gpu_profiles.py` — GPU-generation-to-format recommendations.
- `quant_gui/hf.py` — downloads a single file, or a whole repo (for LLMs),
  from Hugging Face.
- `quant_gui/filters.py` — the model-family presets ctq exposes (e.g.
  `--flux2`, `--wan`, `--krea2` for txtfusion/Krea2/Kroma-style models).
- `quant_gui/size_estimate.py` — estimates output file size from the input
  file's safetensors header, without needing torch/ctq installed.
- `quant_gui/int4_backend.py` — real INT4 ConvRot conversion via
  `comfy_kitchen`, independent of ctq (see [About "INT4 ConvRot"](#about-int4-convrot)).
- `quant_gui/gguf_backend.py` — GGUF export via llama.cpp's `gguf` package,
  independent of ctq (see [About "GGUF"](#about-gguf)).
- `quant_gui/llamacpp_backend.py` — manages a real llama.cpp checkout/venv
  and shells out to its `convert_hf_to_gguf.py`/`llama-quantize`/
  `llama-imatrix` for LLMs
  (see [About "LLM to GGUF"](#about-llm-to-gguf-gemma-llama-qwen-etc)).
- `quant_gui/data/default_calibration.txt` — the small bundled generic
  imatrix calibration text (multi-domain: prose, code, lists, Q&A) used
  when no custom calibration file is supplied.
- `quant_gui/gpu_quant.py` — the opt-in Triton Q8_0 GPU quantizer with a
  bit-exact numpy fallback (see [Performance](#performance)).
- `quant_gui/smart_quant.py` — imatrix loading (legacy + GGUF formats),
  per-tensor sensitivity scoring, and size-budget assignment for the
  Dynamic-style tuner on the LLM tab.
- `quant_gui/gguf_inspect.py` — read-only GGUF inspection for the
  Inspector tab (header metadata + type histogram, no tensor loading).
- `quant_gui/gguf_edit.py` — the GGUF metadata/tensor editor behind the
  Editor tab: edits stream through a disk-spooled writer, so big files
  are copied in constant memory; crash-atomic output.
- `quant_gui/gguf_bench.py` — bits-per-weight target families and
  candidate lists for the find-best sweep.
- `quant_gui/loop_timing.py` — per-loop optimizer timing for the History
  tab's speed verdicts.
- `quant_gui/run_history.py` — append-only `run_history.json` log and the
  History tab's comparison rows.
- `quant_gui/ui_settings.py` — the small persisted-settings store behind
  checkboxes that survive relaunches (e.g. GPU quantization).
- `tools/convert_krea2_to_gguf.py` — the Krea 2 image-DiT converter behind
  the Image → GGUF tab (see
  [About "Image → GGUF"](#about-image--gguf-krea-2)).
- `tools/krea2_kquant_plan.py` — sensitivity-proxy planner that emits a
  `--tensor-type-file` for the two-stage Krea 2 K-quant flow
  (see [docs/KREA2.md](docs/KREA2.md)).
- `tools/compare_gguf.py` — inference-free GGUF quality comparison used by
  the Tools tab.
- `tools/ollama_modelfile.py` — Ollama `Modelfile` generator with
  KV-cache hints (see [docs/OLLAMA.md](docs/OLLAMA.md)).
- `docs/KREA2.md`, `docs/OLLAMA.md`, `docs/AUDIT.md`, `docs/ROADMAP.md` —
  deeper write-ups: Krea 2 conversion + K-quants, Ollama thinking-mode
  recipe, the backend audit, and the feature roadmap.

## Sources

Background research and reference implementations this app's INT4 ConvRot,
diffusion-GGUF, and LLM-GGUF backends are built on/verified against, since
none of the three go through `ctq`:

- [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant) —
  the `ctq` CLI this app wraps for every `.safetensors` format
  (FP8/INT8/INT8 ConvRot/NVFP4/MXFP8), including its
  [issue #50](https://github.com/silveroxides/convert_to_quant/issues/50)
  ("INT4 convrot support?"), open and unaddressed as of this writing.
- [silveroxides/Kroma-Quant](https://huggingface.co/silveroxides/Kroma-Quant)
  and [PotatoForge/Kroma-INT8-Quants](https://huggingface.co/PotatoForge/Kroma-INT8-Quants) —
  the reference `.safetensors` files this app's presets and mixed-precision
  templates are designed to reproduce.
- [LAXMAYDAY/Krea-2-Turbo-int4-tensorwise-mixed](https://huggingface.co/LAXMAYDAY/Krea-2-Turbo-int4-tensorwise-mixed)
  and [Lockout/krea2-comfy-int4-mixed](https://huggingface.co/Lockout/krea2-comfy-int4-mixed) —
  the real INT4/W4A4 ConvRot community models that started the INT4
  investigation; Lockout's own model card documents tracing LAXMAYDAY's
  undisclosed recipe back to the Starnodes tool below.
- [Starnodes2024/comfyui-starnodes-modelconverter](https://github.com/Starnodes2024/comfyui-starnodes-modelconverter) —
  a ComfyUI-only custom node whose `int4_convrot` code path is the reference
  for this app's exact `convrot_w4a4`/`int8_tensorwise` conventions
  (metadata shape, layer key suffixes) - it can't run standalone outside
  ComfyUI, which is why this app calls the kernel beneath it directly
  instead of wrapping the node itself.
- [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) —
  the standalone, pip-installable kernel library (`comfy_kitchen.tensor.convrot_w4a4`)
  that both Starnodes and this app's `quant_gui/int4_backend.py` actually
  quantize with.
- [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), specifically
  its `tools/convert.py` and `loader.py` — the reference this app's GGUF
  architecture detection and F32-fallback rules are ported from, and the
  ComfyUI-side loader that GGUF files from this app are meant to load into.
- [gguf-py](https://github.com/ggerganov/llama.cpp/tree/master/gguf-py)
  (the `gguf` PyPI package) — llama.cpp's own Python bindings, and the
  actual pure-Python block-quantization implementation
  (`quant_gui/gguf_backend.py` calls `gguf.quants`/`gguf.GGUFWriter`
  directly).
- [llama.cpp](https://github.com/ggerganov/llama.cpp) itself, specifically
  its `convert_hf_to_gguf.py` and `conversion/` package (tokenizer
  conversion, per-architecture hyperparameter mapping), its
  `tools/quantize` (`llama-quantize`, the real K-quant and
  `--imatrix`/`--tensor-type-file` implementation), and its
  `tools/imatrix` (`llama-imatrix`, the real importance-matrix
  implementation) — `quant_gui/llamacpp_backend.py` clones and shells out
  to all of these directly, the same way the rest of this app shells out
  to `ctq`, rather than reimplementing any of it.
- [Unsloth's Dynamic 2.0/3.0 GGUF docs](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs) -
  read via web search (unsloth.ai itself was network-blocked in the
  environment this was built in) to understand what their technique
  actually is: an imatrix calibration file plus finer per-layer type
  decisions, both real llama.cpp mechanisms and not a new algorithm of
  their own. Their specific calibration dataset and specific per-model
  layer-type choices aren't published, so this app exposes the same real
  mechanisms as general-purpose tools rather than claiming to reproduce
  their exact recipe.

Every claim above about what's real vs. not (e.g. which GGUF quant types
have pure-Python support, whether a package needs a GPU, whether K-quants
need a compiled binary, whether imatrix/per-layer overrides actually work)
was checked directly against these sources' own code - including a real
`cmake` build of `llama-quantize` and `llama-imatrix`, a real
`convert_hf_to_gguf.py` run, a real generated importance matrix, and a
real per-layer type override read back out of the quantized file - not
assumed from documentation alone.
