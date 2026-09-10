"""Quant Convert GUI

A friendly front end over silveroxides/convert_to_quant (`ctq`), the tool
that actually produces the FP8 / INT8 / INT8-ConvRot / NVFP4 safetensors
files used by projects like Kroma-Quant. This app builds the right `ctq`
command for you, runs it, and streams the log — no CLI flags to memorize.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import gradio as gr

from quant_gui.cli_builder import ConvertOptions, OptionsError, build_args, format_command
from quant_gui.env_check import check_environment, report_markdown
from quant_gui.filters import preset_choices, preset_highprec_regex, preset_label, suggest_preset
from quant_gui.gpu_profiles import GPU_PROFILE_BY_KEY, GPU_PROFILES, detect_profile_key
from quant_gui.hf import HFUrlError, download as hf_download, parse_hf_url
from quant_gui.gguf_backend import stream_gguf_conversion, stream_install as stream_gguf_install
from quant_gui.gguf_backend import QUANT_TYPE_CHOICES as GGUF_QUANT_TYPE_CHOICES
from quant_gui.gguf_backend import SUPPORTED_ARCH_NAMES as GGUF_SUPPORTED_ARCH_NAMES
from quant_gui.gguf_backend import is_available as gguf_is_available
from quant_gui.int4_backend import stream_int4_conversion, stream_install as stream_int4_install
from quant_gui.int4_backend import is_available as int4_is_available
from quant_gui.runner import stream_conversion
from quant_gui.size_estimate import estimate_from_file, estimate_gguf_from_file, estimate_int4_mixed_from_file

APP_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = APP_DIR / "downloads"
OUTPUT_DIR = APP_DIR / "converted"

FORMAT_CHOICES = [
    ("INT8 — ConvRot (recommended, matches Kroma-Quant *-int8-convrot* files)", "int8_convrot"),
    ("INT8 — mixed precision (row/block/tensor scaling, no rotation)", "int8_plain"),
    ("FP8 (E4M3) — Ada / Hopper+ GPUs only", "fp8"),
    ("NVFP4 — 4-bit, closest available today (Blackwell GPUs only)", "nvfp4"),
    ("MXFP8 — Blackwell GPUs only", "mxfp8"),
    ("INT4 ConvRot — experimental, real, via comfy_kitchen (not ctq)", "int4_convrot"),
    ("GGUF — for ComfyUI-GGUF's loader (Q4_0..Q8_0, not ctq)", "gguf"),
]

FORMAT_LABEL_BY_KEY = {v: k for k, v in FORMAT_CHOICES}

GGUF_NOTICE_READY = (
    "### GGUF — this one doesn't go through ctq either\n\n"
    "`convert_to_quant` only ever writes `.safetensors` — GGUF is a completely different container/quant "
    "format (llama.cpp's), read by ComfyUI through the separate "
    "[city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) custom node. This format calls the "
    "`gguf` package (llama.cpp's own Python bindings) directly, the same one that project's own "
    "`tools/convert.py` uses.\n\n"
    f"**Real, pure-Python block quantization** — {', '.join(t for t in GGUF_QUANT_TYPE_CHOICES if t.startswith('Q'))} "
    "are genuinely computed here (`gguf.quants`, no C++ build needed), not just relabeled F16. "
    "**K-quants (Q4_K_M, Q5_K_S, Q6_K, etc.) are *not* available** — that family only has a *decoder* in "
    "the Python package; producing them needs a patched `llama-quantize` binary compiled from source (see "
    "ComfyUI-GGUF's `tools/README.md`), which this app doesn't build or shell out to. If you need K-quants, "
    "convert here to F16/BF16 first, then run `llama-quantize` yourself on that file.\n\n"
    "**Architecture is auto-detected** from the tensor names already in your file — GGUF's own loader "
    f"only accepts a fixed set of names, so this app recognizes exactly those: {', '.join(GGUF_SUPPORTED_ARCH_NAMES)}. "
    "Anything else (including diffusers-format checkpoints) is refused rather than silently mislabeled. "
    "1D tensors, tiny tensors (≤ 1024 elements), and a per-architecture handful of precision-sensitive "
    "layers (matching city96's own conversion script) always stay F32; your model preset / exclude-layers "
    "rules apply on top of that."
)

GGUF_NOTICE_MISSING = (
    "### GGUF needs one more package\n\n"
    "This format calls the `gguf` package directly (not ctq — see the About tab for why). It isn't "
    "installed yet — click **Install gguf** below (streams to the log on the right), or run it yourself:\n\n"
    "```bash\npip install gguf\n```\n\n"
    "Pure Python/numpy — no GPU, no compiler, no ComfyUI install needed to *produce* the file."
)

INT4_NOTICE_READY = (
    "### INT4 ConvRot — this one doesn't go through ctq\n\n"
    "Real INT4/W4A4 ConvRot models exist in the wild — e.g. "
    "[LAXMAYDAY/Krea-2-Turbo-int4-tensorwise-mixed](https://huggingface.co/LAXMAYDAY/Krea-2-Turbo-int4-tensorwise-mixed) "
    "and [Lockout/krea2-comfy-int4-mixed](https://huggingface.co/Lockout/krea2-comfy-int4-mixed) — but "
    "`convert_to_quant` (ctq, the tool the rest of this app wraps) still has no `--int4` flag "
    "([tracked, unaddressed: issue #50](https://github.com/silveroxides/convert_to_quant/issues/50)). "
    "This format instead calls **`comfy_kitchen`'s own `convrot_w4a4` kernel directly** — the same "
    "primitive that recipe (and ctq's own future int4 support, whenever it lands) would use — bypassing "
    "ctq entirely for this one format.\n\n"
    "**You pick which layers actually go to INT4** via the regex below; everything else quantizable falls "
    "back to plain INT8 tensorwise, and your model preset / exclude-layers rules still apply on top. This "
    "is *not* a reproduction of LAXMAYDAY's or Lockout's exact undisclosed layer list — verified end to end "
    "against real `comfy_kitchen` output (correct `convrot_w4a4`/`int8_tensorwise` metadata, correct packed "
    "shapes), but it's this app's own recipe, not theirs."
)

INT4_NOTICE_MISSING = (
    "### INT4 ConvRot needs one more package\n\n"
    "This format calls `comfy_kitchen`'s real INT4 ConvRot kernel directly (not ctq — see the About tab for "
    "why). It isn't installed yet — click **Install comfy-kitchen** below (streams to the log on the right), "
    "or run it yourself:\n\n"
    "```bash\npip install comfy-kitchen\n```\n\n"
    "Needs a Turing-or-newer GPU (SM 7.5+ — RTX 20-series onward, so your Ampere/Ada/Blackwell card is fine) "
    "for real speed; the same eager PyTorch path also runs correctly on CPU, just slowly."
)

FORMAT_HELP = {
    "int8_convrot": (
        "INT8 weights plus a group-wise Hadamard rotation that spreads out outlier values before "
        "quantizing, which is what gives ConvRot files their quality edge over plain INT8. "
        "Runs on any GPU (or CPU, slowly)."
    ),
    "int8_plain": (
        "Straight INT8 quantization with row/block/tensor scaling, no rotation step — any GPU. "
        "Pair it with a model preset below to keep sensitive layers (norms, embeddings, modulation) in "
        "BF16 while the rest goes to INT8: this is the 'mixed precision INT8' style used by repos like "
        "PotatoForge/Kroma-INT8-Quants."
    ),
    "fp8": (
        "8-bit float. Needs an Ada/Hopper/Blackwell NVIDIA GPU (compute capability >= 8.9) — "
        "Ampere cards (RTX 30-series, e.g. RTX 3080 Ti) have no FP8 hardware path at all. Use INT8 on those instead."
    ),
    "nvfp4": "NVIDIA's 4-bit float block format. Requires a Blackwell GPU and the comfy-kitchen package.",
    "mxfp8": "Microscaling FP8. Requires a Blackwell GPU.",
    "int4_convrot": (
        "Real packed-signed-INT4 ConvRot, via comfy_kitchen directly (not ctq). Pick which layers go INT4 "
        "with the regex below; the rest fall back to INT8 tensorwise. Needs SM 7.5+ (Turing onward) for real "
        "speed, works (slowly) on CPU too."
    ),
    "gguf": (
        "A completely different container/quant format from everything else here - for ComfyUI's separate "
        "GGUF loader, via the `gguf` package directly (not ctq). Real Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 quantization; "
        "K-quants need a compiled llama-quantize binary this app doesn't provide. Architecture is "
        "auto-detected from your model's tensor names - unsupported architectures are refused outright."
    ),
}

PRESET_CHOICES = preset_choices()
PRESET_LABELS = {name: preset_label(name) for name in PRESET_CHOICES}
LABEL_TO_PRESET = {v: k for k, v in PRESET_LABELS.items()}

GPU_PROFILE_LABELS = {p.key: p.label for p in GPU_PROFILES}
GPU_LABEL_TO_KEY = {v: k for k, v in GPU_PROFILE_LABELS.items()}

_initial_env = check_environment()
_initial_gpu_key = detect_profile_key(_initial_env.gpu_family)


def build_options(
    input_path: str,
    output_name: str,
    auto_output: bool,
    fmt: str,
    quality_mode: str,
    convrot_group_size: int,
    dynamic_convrot: bool,
    scaling_mode: str,
    block_size: float | None,
    preset_label_value: str,
    comfy_quant: bool,
    save_metadata: bool,
    low_memory: bool,
    exclude_layers: str,
    custom_layers: str,
    custom_type: str,
    custom_scaling_mode: str,
    custom_convrot: bool,
    custom_convrot_group_size: int,
    custom_simple: bool,
    fallback: str,
    fallback_simple: bool,
    device: str,
    output_dtype: str,
    verbose: str,
    calib_samples: float,
    optimizer: str,
    num_iter: float,
    manual_seed: float,
) -> ConvertOptions:
    simple = quality_mode == "simple"

    if fmt == "int8_convrot":
        quant_format, convrot, eff_scaling_mode = "int8", True, "row"
    elif fmt == "int8_plain":
        quant_format, convrot, eff_scaling_mode = "int8", False, scaling_mode
    else:
        quant_format, convrot, eff_scaling_mode = fmt, False, scaling_mode

    preset = LABEL_TO_PRESET.get(preset_label_value, "none")

    output_path = None
    if not auto_output:
        name = (output_name or "").strip()
        if name:
            output_path = str(OUTPUT_DIR / name) if not os.path.isabs(name) and os.sep not in name else name

    return ConvertOptions(
        input_path=input_path,
        output_path=output_path,
        quant_format=quant_format,
        scaling_mode=eff_scaling_mode,
        block_size=int(block_size) if block_size else None,
        convrot=convrot,
        convrot_group_size=int(convrot_group_size),
        dynamic_convrot=dynamic_convrot,
        simple=simple,
        comfy_quant=comfy_quant,
        save_quant_metadata=save_metadata,
        low_memory=low_memory,
        preset=preset,
        exclude_layers=(exclude_layers or "").strip() or None,
        custom_layers=(custom_layers or "").strip() or None,
        custom_type=custom_type if custom_type and custom_type != "none" else None,
        custom_scaling_mode=custom_scaling_mode if custom_scaling_mode and custom_scaling_mode != "none" else None,
        custom_convrot=custom_convrot,
        custom_convrot_group_size=int(custom_convrot_group_size),
        custom_simple=custom_simple,
        fallback=fallback if fallback and fallback != "none" else None,
        fallback_simple=fallback_simple,
        device=(device or "").strip() or None,
        output_dtype=output_dtype,
        verbose=verbose,
        calib_samples=int(calib_samples),
        optimizer=optimizer,
        num_iter=int(num_iter),
        manual_seed=int(manual_seed),
    )


def on_format_change(fmt: str):
    is_int8_convrot = fmt == "int8_convrot"
    is_plain_int8_or_fp8 = fmt in ("int8_plain", "fp8")
    is_int4 = fmt == "int4_convrot"
    is_gguf = fmt == "gguf"
    int4_ready = int4_is_available()
    gguf_ready = gguf_is_available()
    int4_notice_text = INT4_NOTICE_READY if int4_ready else INT4_NOTICE_MISSING if is_int4 else ""
    gguf_notice_text = GGUF_NOTICE_READY if gguf_ready else GGUF_NOTICE_MISSING if is_gguf else ""
    return (
        gr.update(visible=is_int8_convrot),  # convrot group
        gr.update(visible=is_plain_int8_or_fp8),  # scaling mode group
        gr.update(value=FORMAT_HELP.get(fmt, "")),
        gr.update(value=int4_notice_text, visible=is_int4),  # int4 notice
        gr.update(visible=is_int4),  # int4 options group
        gr.update(visible=is_int4 and not int4_ready),  # int4 install button
        gr.update(value=gguf_notice_text, visible=is_gguf),  # gguf notice
        gr.update(visible=is_gguf),  # gguf options group
        gr.update(visible=is_gguf and not gguf_ready),  # gguf install button
    )


def on_source_change(source: str):
    return gr.update(visible=source == "Local file path"), gr.update(visible=source == "Hugging Face URL")


def list_downloaded_models() -> list[tuple[str, str]]:
    """(label, absolute path) for every .safetensors file under downloads/,
    newest first, so 'Local file path' can offer them without retyping."""
    if not DOWNLOAD_DIR.is_dir():
        return []
    files = sorted(DOWNLOAD_DIR.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    choices = []
    for f in files:
        size_gb = f.stat().st_size / (1024**3)
        rel = f.relative_to(DOWNLOAD_DIR)
        choices.append((f"{rel} ({size_gb:.2f} GB)", str(f)))
    return choices


def refresh_local_models():
    choices = list_downloaded_models()
    if not choices:
        return gr.update(choices=[], value=None, label="Found in downloads/ (none yet)")
    return gr.update(choices=choices, value=None, label=f"Found in downloads/ ({len(choices)})")


def do_hf_download(url: str, token: str, progress=gr.Progress()):
    url = (url or "").strip()
    if not url:
        return "", "Paste a Hugging Face file URL first."
    try:
        parse_hf_url(url)
    except HFUrlError as exc:
        return "", str(exc)

    progress(0, desc="Downloading...")

    def cb(read, total):
        if total:
            progress(min(read / total, 1.0), desc=f"{read / 1e6:.0f} / {total / 1e6:.0f} MB")

    try:
        local_path = hf_download(url, str(DOWNLOAD_DIR), token=(token or "").strip() or None, progress_cb=cb)
    except Exception as exc:  # noqa: BLE001 - surfaced directly to the user
        return "", f"Download failed: {exc}"

    return local_path, f"Downloaded to `{local_path}`"


def on_input_resolved(local_path: str, hf_url: str, current_preset_label: str):
    hint = local_path or hf_url
    preset = suggest_preset(hint)
    if not preset:
        return "", gr.update()

    name = Path(hint).name if local_path else hf_url
    if LABEL_TO_PRESET.get(current_preset_label, "none") == "none":
        msg = (
            f"Detected `{name}` — auto-selected the **{preset}** preset below "
            "(its layer list literally includes `txtfusion`; this is what gives txtfusion-edition "
            "models better quality than plain ConvRot). Change it below if you don't want that."
        )
        return msg, gr.update(value=PRESET_LABELS[preset])

    msg = f"Detected `{name}` — this model usually does best with the **{preset}** preset, but you've already picked a different one."
    return msg, gr.update()


def on_local_input_resolved(local_path: str, current_preset_label: str):
    """Same as on_input_resolved, but for local-file-only triggers (the path
    textbox, the downloads/ dropdown) - never falls back to the (irrelevant,
    possibly still prefilled) Hugging Face URL field."""
    return on_input_resolved(local_path, "", current_preset_label)


def _progress_bar_html(fraction: float, desc: str) -> str:
    """A small standalone progress bar, rendered as its own component
    instead of relying on Gradio's built-in gr.Progress() overlay - that
    overlay draws on top of *every* output of the event (log_box and
    result_file both), hiding their real content behind a duplicated
    "stuck" progress readout once the run finishes streaming."""
    pct = max(0.0, min(1.0, fraction)) * 100
    import html as _html

    return (
        '<div style="margin:2px 0 10px;">'
        f'<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.82rem;'
        f'opacity:0.85;margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
        f'{_html.escape(desc)}</div>'
        '<div style="background:rgba(128,128,128,0.25);border-radius:6px;height:10px;overflow:hidden;">'
        f'<div style="background:#6366f1;height:100%;width:{pct:.1f}%;transition:width .15s linear;"></div>'
        "</div></div>"
    )


def run_int4_convert(
    input_path: str,
    output_name: str,
    auto_output: bool,
    preset_label_value: str,
    int4_layers_regex: str,
    int4_fallback_int8: bool,
    exclude_layers: str,
    device: str,
):
    input_path = (input_path or "").strip()
    if not input_path:
        yield "Pick an input file (local path or downloaded Hugging Face file) first.", None, ""
        return
    if not Path(input_path).is_file():
        yield f"Input file not found on disk: {input_path}", None, ""
        return
    if not int4_is_available():
        yield f"comfy-kitchen isn't installed. Install it with:\n  pip install comfy-kitchen\n\nThen re-run.", None, ""
        return

    preset = LABEL_TO_PRESET.get(preset_label_value, "none")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = (output_name or "").strip()
    if not auto_output and name:
        output_path = str(OUTPUT_DIR / name) if not os.path.isabs(name) and os.sep not in name else name
    else:
        stem = Path(input_path).stem
        output_path = str(OUTPUT_DIR / f"{stem}-int4-mixed.safetensors")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    log = f"Converting (INT4 ConvRot via comfy_kitchen, device={device or 'cpu'})\n"
    log += f"  input:  {input_path}\n  output: {output_path}\n"
    log += f"  INT4 layers regex: {int4_layers_regex or '(none — no layers go INT4)'}\n"
    log += f"  preset: {preset}\n\n"
    yield log, None, _progress_bar_html(0, "Starting INT4 conversion...")

    result_path = None
    for item in stream_int4_conversion(
        input_path, output_path, (int4_layers_regex or "").strip() or None,
        preset=preset, exclude_regex=(exclude_layers or "").strip() or None,
        fallback_int8=int4_fallback_int8, device=(device or "cpu").strip() or "cpu",
    ):
        kind = item[0]
        if kind == "progress":
            _, current, total, key = item
            bar = _progress_bar_html(current / total if total else 0, f"{current}/{total}: {key}")
            log += f"({current}/{total}) {key}\n"
            yield log, result_path, bar
        elif kind == "ok":
            stats = item[1]
            result_path = output_path if Path(output_path).is_file() else None
            log += (
                f"\n✅ Conversion finished.\n"
                f"  {stats.int4_count} layer(s) → INT4 ConvRot\n"
                f"  {stats.int8_count} layer(s) → INT8 tensorwise (fallback)\n"
                f"  {stats.kept_count} layer(s) kept at original/BF16 precision\n"
            )
            if stats.skipped_shape_count:
                log += (
                    f"  ⚠️ {stats.skipped_shape_count} layer(s) matched the INT4 regex but their shape isn't "
                    f"divisible by 256/64, so they fell back to INT8 instead.\n"
                )
            if result_path:
                log += f"Output: {result_path}\n"
            yield log, result_path, _progress_bar_html(1.0, "Done")
        elif kind == "fail":
            log += f"\n❌ Conversion failed: {item[1]}\n"
            yield log, None, _progress_bar_html(1.0, "Failed")


def run_gguf_convert(
    input_path: str,
    output_name: str,
    auto_output: bool,
    preset_label_value: str,
    gguf_quant_type: str,
    exclude_layers: str,
):
    input_path = (input_path or "").strip()
    if not input_path:
        yield "Pick an input file (local path or downloaded Hugging Face file) first.", None, ""
        return
    if not Path(input_path).is_file():
        yield f"Input file not found on disk: {input_path}", None, ""
        return
    if not gguf_is_available():
        yield "gguf isn't installed. Install it with:\n  pip install gguf\n\nThen re-run.", None, ""
        return

    preset = LABEL_TO_PRESET.get(preset_label_value, "none")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = (output_name or "").strip()
    if not auto_output and name:
        output_path = str(OUTPUT_DIR / name) if not os.path.isabs(name) and os.sep not in name else name
    else:
        stem = Path(input_path).stem
        output_path = str(OUTPUT_DIR / f"{stem}-{gguf_quant_type}.gguf")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    log = f"Converting to GGUF ({gguf_quant_type} via the gguf package)\n"
    log += f"  input:  {input_path}\n  output: {output_path}\n"
    log += f"  preset: {preset}\n\n"
    yield log, None, _progress_bar_html(0, "Starting GGUF conversion...")

    result_path = None
    for item in stream_gguf_conversion(
        input_path, output_path, gguf_quant_type,
        preset=preset, exclude_regex=(exclude_layers or "").strip() or None,
    ):
        kind = item[0]
        if kind == "progress":
            _, current, total, key = item
            bar = _progress_bar_html(current / total if total else 0, f"{current}/{total}: {key}")
            log += f"({current}/{total}) {key}\n"
            yield log, result_path, bar
        elif kind == "ok":
            stats = item[1]
            result_path = output_path if Path(output_path).is_file() else None
            log += (
                f"\n✅ Conversion finished. Detected architecture: {stats.arch}\n"
                f"  {stats.quantized_count} layer(s) → {gguf_quant_type}\n"
                f"  {stats.f32_kept_count} layer(s) kept F32 (1D / tiny / precision-sensitive)\n"
            )
            if stats.fallback_f16_count:
                log += (
                    f"  ⚠️ {stats.fallback_f16_count} layer(s) had a shape not divisible by 32 and fell "
                    f"back to F16 instead of {gguf_quant_type}.\n"
                )
            if stats.skipped_high_dim_count:
                log += (
                    f"  ⚠️ {stats.skipped_high_dim_count} tensor(s) with more than 4 dimensions were skipped "
                    "entirely - GGUF can't represent them (see ComfyUI-GGUF's fix_5d_tensors.py).\n"
                )
            if result_path:
                log += f"Output: {result_path}\n"
            yield log, result_path, _progress_bar_html(1.0, "Done")
        elif kind == "fail":
            log += f"\n❌ Conversion failed: {item[1]}\n"
            yield log, None, _progress_bar_html(1.0, "Failed")


def run_convert(
    input_local: str,
    input_hf: str,
    source: str,
    output_name: str,
    auto_output: bool,
    fmt: str,
    quality_mode: str,
    convrot_group_size: int,
    dynamic_convrot: bool,
    scaling_mode: str,
    block_size: float,
    preset_label_value: str,
    comfy_quant: bool,
    save_metadata: bool,
    low_memory: bool,
    exclude_layers: str,
    custom_layers: str,
    custom_type: str,
    custom_scaling_mode: str,
    custom_convrot: bool,
    custom_convrot_group_size: int,
    custom_simple: bool,
    fallback: str,
    fallback_simple: bool,
    device: str,
    output_dtype: str,
    verbose: str,
    calib_samples: float,
    optimizer: str,
    num_iter: float,
    manual_seed: float,
    python_exe: str,
    int4_layers_regex: str,
    int4_fallback_int8: bool,
    gguf_quant_type: str,
):
    input_path = (input_local or "").strip() if source == "Local file path" else (input_hf or "").strip()

    if fmt == "int4_convrot":
        yield from run_int4_convert(
            input_path, output_name, auto_output, preset_label_value,
            int4_layers_regex, int4_fallback_int8, exclude_layers, device,
        )
        return

    if fmt == "gguf":
        yield from run_gguf_convert(
            input_path, output_name, auto_output, preset_label_value, gguf_quant_type, exclude_layers,
        )
        return

    if not input_path:
        yield "Pick an input file (local path or downloaded Hugging Face file) first.", None, ""
        return

    if not Path(input_path).is_file():
        yield f"Input file not found on disk: {input_path}", None, ""
        return

    try:
        opts = build_options(
            input_path=input_path, output_name=output_name, auto_output=auto_output, fmt=fmt,
            quality_mode=quality_mode, convrot_group_size=convrot_group_size, dynamic_convrot=dynamic_convrot,
            scaling_mode=scaling_mode, block_size=block_size, preset_label_value=preset_label_value,
            comfy_quant=comfy_quant, save_metadata=save_metadata, low_memory=low_memory,
            exclude_layers=exclude_layers, custom_layers=custom_layers, custom_type=custom_type,
            custom_scaling_mode=custom_scaling_mode, custom_convrot=custom_convrot,
            custom_convrot_group_size=custom_convrot_group_size, custom_simple=custom_simple,
            fallback=fallback, fallback_simple=fallback_simple, device=device, output_dtype=output_dtype,
            verbose=verbose, calib_samples=calib_samples, optimizer=optimizer, num_iter=num_iter,
            manual_seed=manual_seed,
        )
        args = build_args(opts)
    except OptionsError as exc:
        yield f"Can't build a valid command: {exc}", None, ""
        return

    if opts.output_path:
        Path(opts.output_path).parent.mkdir(parents=True, exist_ok=True)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    yield "", None, _progress_bar_html(0, "Starting ctq...")
    # Matches both "(12/264) Processing (INT8): blocks.0.mlp.up.weight" and
    # "(2/6) Skipping tensor: blocks.0.firs.weight (Reason: krea2 skip)".
    tensor_progress_re = re.compile(r"\((\d+)/(\d+)\)\s*(Processing|Skipping)")

    log = ""
    result_path = None
    bar = _progress_bar_html(0, "Starting ctq...")
    for chunk in stream_conversion(args, python_executable=((python_exe or "").strip() or None)):
        if chunk == "__CTQ_OK__":
            found = opts.output_path
            if not found:
                m = re.search(r"Saved to[:\s]+(\S+\.safetensors)", log, re.IGNORECASE)
                found = m.group(1) if m else None
            result_path = found if found and Path(found).is_file() else None
            log += "\n✅ Conversion finished.\n"
            if result_path:
                log += f"Output: {result_path}\n"
            elif found:
                log += f"Output (reported, not found on disk yet): {found}\n"
            yield log, result_path, _progress_bar_html(1.0, "Done")
        elif chunk.startswith("__CTQ_FAIL__"):
            code = chunk.split(":", 1)[-1]
            log += f"\n❌ ctq exited with code {code}.\n"
            yield log, None, _progress_bar_html(1.0, "Failed")
        else:
            log += chunk
            m = tensor_progress_re.search(chunk)
            if m:
                current, total, action = int(m.group(1)), int(m.group(2)), m.group(3)
                layer = ""
                if ":" in chunk:
                    tail = chunk.split(":", 1)[-1].strip()
                    layer = tail.split(" (")[0].strip()
                if total:
                    desc = f"{action} {current}/{total}: {layer}".rstrip(": ")
                    bar = _progress_bar_html(current / total, desc)
            yield log, result_path, bar


def refresh_env():
    return report_markdown(check_environment())


CSS = """
#convert-btn { font-size: 1.1rem; font-weight: 600; }
#log-box textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85rem; }
"""

with gr.Blocks(title="Quant Convert GUI") as demo:
    gr.Markdown(
        "# Quant Convert GUI\n"
        "Turn a `.safetensors` model into **FP8**, **INT8**, **INT8 ConvRot**, **NVFP4**, real **INT4 ConvRot**, "
        "or **GGUF** — using "
        "[silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant) for the `.safetensors` "
        "formats (the same tool used to build the [Kroma-Quant](https://huggingface.co/silveroxides/Kroma-Quant) "
        "and PotatoForge/Kroma-INT8-Quants files), plus two independent backends — `comfy_kitchen` for INT4 and "
        "`gguf` for GGUF — for the formats ctq doesn't produce. Pick your GPU below and it'll steer you toward a "
        "format your card can actually accelerate."
    )

    with gr.Tabs():
        with gr.Tab("Convert"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### 1. Choose your model")
                    source = gr.Radio(["Local file path", "Hugging Face URL"], value="Hugging Face URL", label="Source")

                    with gr.Group(visible=False) as local_group:
                        with gr.Row():
                            local_models_dd = gr.Dropdown(
                                choices=[], value=None, label="Found in downloads/",
                                info="Models already sitting in this app's downloads/ folder.",
                                scale=4,
                            )
                            refresh_local_models_btn = gr.Button("🔄", scale=1, min_width=40)
                        input_local = gr.Textbox(
                            label="Local .safetensors path",
                            placeholder="/path/to/model.safetensors",
                        )

                    with gr.Group(visible=True) as hf_group:
                        input_hf_url = gr.Textbox(
                            label="Hugging Face file URL",
                            value="https://huggingface.co/silveroxides/Kroma-Quant/blob/main/kroma-v0.3-txtfusion-edition-turbo-int8-convrot-simple.safetensors",
                            placeholder="https://huggingface.co/owner/repo/blob/main/model.safetensors",
                        )
                        hf_token = gr.Textbox(label="HF access token (only needed for gated/private repos)", type="password")
                        download_btn = gr.Button("Download")
                        download_status = gr.Markdown()
                        input_hf_local = gr.Textbox(visible=False)

                    resolved_hint = gr.Markdown()

                    gr.Markdown("### 2. What GPU will run this model?")
                    gpu_dd = gr.Dropdown(
                        [GPU_PROFILE_LABELS[p.key] for p in GPU_PROFILES],
                        value=GPU_PROFILE_LABELS[_initial_gpu_key],
                        label="Target GPU",
                        info="This never touches your files — it just picks a sane default format and warns you off formats your GPU can't accelerate.",
                    )
                    gpu_note = gr.Markdown(GPU_PROFILE_BY_KEY[_initial_gpu_key].note)
                    apply_gpu_recommendation = gr.Button("Use recommended format for this GPU")

                    gr.Markdown("### 3. Choose an output format")
                    fmt = gr.Radio(
                        [c[0] for c in FORMAT_CHOICES],
                        value=FORMAT_CHOICES[0][0],
                        label="Quantization format",
                    )
                    fmt_value = gr.State(FORMAT_CHOICES[0][1])
                    fmt_help = gr.Markdown(FORMAT_HELP["int8_convrot"])
                    int4_notice = gr.Markdown(visible=False)
                    int4_install_btn = gr.Button("Install comfy-kitchen", visible=False)
                    with gr.Group(visible=False) as int4_group:
                        int4_layers_regex = gr.Textbox(
                            label="INT4 layers (regex)",
                            placeholder=r"e.g. attn\.wq|mlp\.gate — layers matching this get real INT4 ConvRot",
                            info="Matches source tensor names. Leave empty to convert nothing to INT4 "
                            "(falls back to plain INT8 tensorwise everywhere quantizable).",
                        )
                        gr.Markdown(
                            "**Templates** — for Flux/Krea-style blocks (`blocks.N.attn.wq/wk/wv/wo`, "
                            "`blocks.N.mlp.gate/up/down`); check the live log from a previous conversion "
                            "for your model's actual layer names before trusting these blindly.",
                            elem_id="int4-template-note",
                        )
                        with gr.Row():
                            int4_tpl_attn_btn = gr.Button("Attention only", size="sm")
                            int4_tpl_mlp_btn = gr.Button("MLP only", size="sm")
                            int4_tpl_balanced_btn = gr.Button("Attention + MLP", size="sm")
                            int4_tpl_all_btn = gr.Button("Everything quantizable (aggressive)", size="sm")
                            int4_tpl_clear_btn = gr.Button("Clear", size="sm")
                        int4_fallback_int8 = gr.Checkbox(
                            value=True,
                            label="INT8 tensorwise fallback for other quantizable layers (recommended)",
                            info="Unchecked, non-matched layers stay at full precision instead.",
                        )

                    gguf_notice = gr.Markdown(visible=False)
                    gguf_install_btn = gr.Button("Install gguf", visible=False)
                    with gr.Group(visible=False) as gguf_group:
                        gguf_quant_type = gr.Dropdown(
                            GGUF_QUANT_TYPE_CHOICES,
                            value="Q8_0",
                            label="GGUF quant type",
                            info="Q8_0 = best quality/largest of the real quant types here; Q4_0 = smallest/"
                            "roughest. F16/BF16 skip quantization entirely (just repacks into a GGUF "
                            "container) - useful as input to your own llama-quantize run for K-quants.",
                        )
                        gr.Markdown(
                            f"Architecture is auto-detected from your model's own tensor names - supported: "
                            f"{', '.join(GGUF_SUPPORTED_ARCH_NAMES)}. Anything else is refused rather than "
                            "silently mislabeled; the Estimate button below will tell you which one it found."
                        )

                    with gr.Group(visible=True) as convrot_group:
                        gr.Markdown("**ConvRot group size** — must divide the layer width; 256 is ctq's default.")
                        convrot_group_size = gr.Dropdown([4, 16, 64, 256, 1024], value=256, label="ConvRot group size")
                        dynamic_convrot = gr.Checkbox(
                            value=False,
                            label="Dynamic group size (let ctq pick the largest compatible size per layer, min 256)",
                        )

                    with gr.Group(visible=False) as scaling_group:
                        scaling_mode = gr.Radio(["row", "block", "tensor"], value="row", label="Scaling mode")
                        block_size = gr.Number(value=128, label="Block size (block scaling only)", precision=0)

                    gr.Markdown("### 4. Quality vs. speed")
                    quality_mode = gr.Radio(
                        [
                            ("Simple — direct rounding, fast, matches *-simple files", "simple"),
                            ("Learned / AdaRound — slower, best quality, needs GPU + calibration", "learned"),
                        ],
                        value="simple",
                        label="Optimization mode",
                    )

                    with gr.Accordion("Model preset (layer exclusions)", open=False):
                        preset_dd = gr.Dropdown(
                            [PRESET_LABELS[n] for n in PRESET_CHOICES],
                            value=PRESET_LABELS["none"],
                            label="Model-specific preset",
                            info="Keeps sensitive layers (norms, embeddings, modulation) at full precision. "
                            "Pick the family closest to your model.",
                        )
                        with gr.Group(visible=False) as krea2_size_group:
                            gr.Markdown(
                                "**Krea2 size profile** (for now, just this preset) — trades quality for a "
                                "smaller file by deciding how the layers krea2 normally keeps full-precision "
                                "get treated. Check **Estimate output size** after picking one."
                            )
                            with gr.Row():
                                krea2_balanced_btn = gr.Button("Balanced (recommended) — largest")
                                krea2_compact_btn = gr.Button("Compact — sensitive layers → plain INT8")
                                krea2_smallest_btn = gr.Button("Smallest — no exclusions, blanket ConvRot")

                    with gr.Accordion("Advanced options", open=False):
                        with gr.Row():
                            comfy_quant = gr.Checkbox(
                                value=True, label="ComfyUI-compatible layout (--comfy_quant)",
                                info="Writes quantized tensors in the layout/naming ComfyUI expects, so the file "
                                "loads directly as a checkpoint there. Turn off only if another tool needs ctq's "
                                "plain layout instead.",
                            )
                            save_metadata = gr.Checkbox(
                                value=True, label="Save quantization metadata",
                                info="Embeds a _quantization_metadata JSON header (format/scale/group-size per "
                                "layer). Most loaders (ComfyUI, this app's own INT4 path) need this to know a "
                                "layer is quantized at all — off saves a few KB but risks an unreadable file.",
                            )
                            low_memory = gr.Checkbox(
                                value=True, label="Low memory mode",
                                info="Streams tensors through conversion one at a time instead of loading the "
                                "whole model into RAM/VRAM at once. Doesn't change the output file, only avoids "
                                "OOM on large models with limited memory (at a small speed cost).",
                            )
                        exclude_layers = gr.Textbox(
                            label="Exclude layers (regex)",
                            placeholder="e.g. (final_layer|txt_attn.proj)",
                            info="Layers matching this stay at their original precision, overriding the format/"
                            "preset above entirely. Use it to protect specific layers your preset doesn't already "
                            "cover — this changes accuracy and file size, never speed.",
                        )
                        with gr.Row():
                            excl_tpl_norms_btn = gr.Button("Norms & embeddings", size="sm")
                            excl_tpl_mod_btn = gr.Button("Modulation & gating", size="sm")
                            excl_tpl_final_btn = gr.Button("Final output layer", size="sm")
                            excl_tpl_clear_btn = gr.Button("Clear", size="sm")
                        gr.Markdown(
                            "_Generic starting points, not guaranteed to match your model — check the live log "
                            "from a previous conversion (or the Estimate panel) for its actual layer names first._"
                        )

                        gr.Markdown(
                            "**Mixed precision (per-layer custom format)** — pick a base format above for most "
                            "layers, then carve out a regex of layers that should use a *different* format/scaling "
                            "instead. This is how files like [PotatoForge/Kroma-INT8-Quants]"
                            "(https://huggingface.co/PotatoForge/Kroma-INT8-Quants) mix formats per layer — "
                            "their metadata shows attn.wk/wv projections left as plain tensorwise INT8 while "
                            "attn.wq/wo/gate and the MLP layers get row-wise INT8 ConvRot. The regex matches your "
                            "*source* model's original tensor names (before any ComfyUI renaming), so check your "
                            "model's actual layer names first."
                        )
                        custom_layers = gr.Textbox(
                            label="Custom layers (regex)",
                            placeholder=r"e.g. attn\.(wq|wo|gate)|mlp\.(gate|up|down)",
                            info="Leave empty to disable per-layer overrides entirely (every quantizable layer "
                            "just uses the format chosen in step 3).",
                        )
                        with gr.Row():
                            custom_tpl_potatoforge_btn = gr.Button(
                                "PotatoForge-style (wk/wv → plain INT8)", size="sm"
                            )
                            custom_tpl_clear_btn = gr.Button("Clear", size="sm")
                        with gr.Row():
                            custom_type = gr.Dropdown(
                                ["none", "fp8", "int8", "mxfp8", "nvfp4"], value="none", label="Custom layer type",
                                info="Format the matched layers use instead of the main format above.",
                            )
                            custom_scaling_mode = gr.Dropdown(
                                ["none", "tensor", "row", "block"], value="none", label="Custom layer scaling mode",
                                info="Scale granularity for those layers: tensor = one scale for the whole "
                                "tensor (smallest, least accurate), row = one per output row (ctq's usual "
                                "default), block = one per fixed-size block (most accurate, more scale data).",
                            )
                        with gr.Row():
                            custom_convrot = gr.Checkbox(
                                value=False, label="ConvRot on custom layers (INT8 only)",
                                info="Applies the same Hadamard-rotation trick as the main ConvRot option, but "
                                "only to these custom layers. Improves low-bit accuracy; only valid with custom "
                                "type INT8.",
                            )
                            custom_convrot_group_size = gr.Dropdown(
                                [4, 16, 64, 256, 1024], value=256, label="Custom ConvRot group size",
                                info="Rotation block width for these layers - must evenly divide their width. "
                                "Larger groups usually rotate more outliers away but need a wider layer.",
                            )
                            custom_simple = gr.Checkbox(
                                value=True,
                                label="Simple quant for custom layers",
                                info="Recommended: without this, ctq runs slow AdaRound optimization on custom "
                                "layers even when the base format above uses Simple mode.",
                            )
                        with gr.Row():
                            fallback = gr.Dropdown(
                                ["none", "fp8", "int8", "mxfp8", "nvfp4"], value="none", label="Fallback type",
                                info="What layers excluded by your preset or the exclude-layers regex above "
                                "become, instead of staying at full original precision. none (default) keeps "
                                "them full precision - safest for accuracy; picking a format here shrinks the "
                                "file further at that layer's expense.",
                            )
                            fallback_simple = gr.Checkbox(
                                value=False, label="Simple quant for fallback layers",
                                info="Same simple-vs-learned trade-off as above, applied to fallback layers only.",
                            )

                        with gr.Row():
                            device = gr.Textbox(
                                label="Device override", placeholder="cuda / cuda:0 / cpu",
                                info="Where ctq computes the conversion. Never changes the output file's "
                                "contents - only speed. Learned/AdaRound mode in particular is far slower on CPU.",
                            )
                            output_dtype = gr.Radio(
                                ["bfloat16", "float16"], value="bfloat16", label="Output dtype",
                                info="Float type for any layer that stays unquantized, plus scale tensors. "
                                "bfloat16 matches how most modern diffusion models were trained (wide dynamic "
                                "range, safer against outliers); float16 has finer precision at small magnitudes "
                                "but can overflow on extreme values - some older inference stacks expect it.",
                            )
                            verbose = gr.Radio(
                                ["MINIMAL", "NORMAL", "VERBOSE", "DEBUG"], value="NORMAL", label="Log verbosity",
                                info="Console log detail only - has no effect on the resulting model.",
                            )
                        gr.Markdown(
                            "**Learned/AdaRound settings** (used only in Learned mode above) — AdaRound learns "
                            "per-tensor rounding offsets from a handful of calibration passes instead of rounding "
                            "each weight to the nearest representable value, trading conversion time for accuracy."
                        )
                        with gr.Row():
                            calib_samples = gr.Number(
                                value=3072, label="Calibration samples", precision=0,
                                info="How many calibration samples AdaRound uses per tensor. More generally means "
                                "more accurate rounding but a slower conversion.",
                            )
                            optimizer = gr.Dropdown(
                                ["prodigy", "adamw", "radam", "original"], value="prodigy", label="Optimizer",
                                info="Which optimizer learns the per-tensor rounding offsets. prodigy (ctq's "
                                "default) self-tunes its learning rate; adamw/radam are classic alternatives; "
                                "original matches the original AdaRound paper's optimizer.",
                            )
                        with gr.Row():
                            num_iter = gr.Number(
                                value=4000, label="Iterations per tensor", precision=0,
                                info="Optimization steps per tensor. Higher converges closer to the ideal "
                                "rounding but multiplies total conversion time - e.g. halving this roughly "
                                "halves Learned-mode runtime at some accuracy cost.",
                            )
                            manual_seed = gr.Number(
                                value=-1, label="Manual seed (-1 = random)", precision=0,
                                info="Fixes the random seed used to pick calibration samples, for reproducible "
                                "conversions. -1 picks a new seed each run.",
                            )
                        python_exe = gr.Textbox(
                            label="Python executable running ctq (optional)",
                            placeholder="leave blank to use this app's Python / the ctq command on PATH",
                            info="Only matters with multiple Python installs - point this at a specific venv/"
                            "conda interpreter (e.g. one with GPU PyTorch) instead of this app's own or "
                            "whichever ctq is first on PATH. Doesn't affect the resulting model.",
                        )

                    gr.Markdown("### 5. Output")
                    auto_output = gr.Checkbox(value=True, label="Auto-generate output filename (recommended)")
                    output_name = gr.Textbox(label="Output filename", interactive=False, placeholder="auto")

                    command_preview = gr.Textbox(label="Command ctq will run", interactive=False, lines=2)

                    estimate_btn = gr.Button("Estimate output size")
                    estimate_md = gr.Markdown()

                    convert_btn = gr.Button("Convert", elem_id="convert-btn", variant="primary")

                with gr.Column(scale=2):
                    convert_progress = gr.HTML(value="")
                    log_box = gr.Textbox(label="Live conversion log", lines=28, elem_id="log-box", interactive=False, autoscroll=True)
                    result_file = gr.File(label="Converted file", interactive=False)

        with gr.Tab("Environment"):
            gr.Markdown(
                "Conversion runs on **your machine** through `ctq`. This checks whether it (and PyTorch/CUDA) "
                "are actually installed and what hardware is available."
            )
            env_md = gr.Markdown(report_markdown(_initial_env))
            refresh_btn = gr.Button("Re-check environment")
            gr.Markdown(
                "**Install ctq:**\n```bash\npip install convert-to-quant\n"
                "# then install PyTorch separately for your GPU, e.g.\n"
                "pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
                "pip install -U triton   # optional, speeds up INT8 kernels\n```"
            )

        with gr.Tab("About"):
            gr.Markdown(
                "## What these formats mean\n"
                "- **FP8 (E4M3)** — 8-bit float, halves VRAM vs. bf16, needs Ada/Hopper+.\n"
                "- **INT8** — 8-bit integer with a per-row/block/tensor scale. Combined with a model preset "
                "that keeps a handful of sensitive layers in BF16, this is the 'mixed precision INT8' style "
                "used by repos like PotatoForge/Kroma-INT8-Quants.\n"
                "- **INT8 ConvRot** — INT8 plus a group-wise Hadamard rotation applied before quantizing, "
                "which spreads out the outlier values that normally hurt low-bit accuracy in diffusion "
                "transformers. This is the recipe behind the `*-int8-convrot*` Kroma-Quant files.\n"
                "- **NVFP4** — NVIDIA's 4-bit floating point block format; needs a Blackwell GPU.\n"
                "- **INT4 ConvRot** — real packed-signed-INT4 ConvRot (group-256 Hadamard rotation + INT4 "
                "quantization), the same recipe behind LAXMAYDAY's and Lockout's Krea-2 int4-mixed releases. "
                "`convert_to_quant` itself still has no INT4 CLI flag "
                "([issue #50](https://github.com/silveroxides/convert_to_quant/issues/50) remains unaddressed), "
                "so this format doesn't go through ctq at all — it calls "
                "[comfy_kitchen](https://github.com/Comfy-Org/comfy-kitchen)'s own `convrot_w4a4` kernel "
                "directly (the same kernel the Starnodes Model Converter uses). Needs `pip install comfy-kitchen` "
                "and works best on Turing+ (SM 7.5+) GPUs, though it also runs — slowly — on CPU.\n"
                "- **GGUF** — a completely different container/quant format (llama.cpp's), read by ComfyUI "
                "through the separate [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) custom "
                "node. `convert_to_quant` never touches GGUF at all, so this also bypasses ctq — it calls the "
                "`gguf` package (llama.cpp's Python bindings) directly, real Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 block "
                "quantization computed in pure Python. K-quants (Q4_K_M, Q6_K, etc.) aren't available — that "
                "family only has a *decoder* in the Python package; producing them needs a compiled, patched "
                "`llama-quantize` binary that this app doesn't build. Architecture (flux/sdxl/wan/etc.) is "
                "auto-detected from your model's own tensor names, using the exact same detection logic as "
                "ComfyUI-GGUF's own converter, since its loader rejects anything it doesn't recognize.\n\n"
                "## Which format for which GPU\n"
                "| GPU generation | Example cards | Hardware-accelerated formats |\n"
                "|---|---|---|\n"
                "| Turing/Ampere | RTX 20/30-series, A100 — **incl. RTX 3080 Ti** | **INT8 / INT8 ConvRot / INT4 ConvRot** — no FP8 or NVFP4 hardware path |\n"
                "| Ada Lovelace | RTX 40-series, L40 | FP8, INT8 ConvRot, INT4 ConvRot |\n"
                "| Hopper | H100, H200 | FP8, INT8 ConvRot, INT4 ConvRot |\n"
                "| Blackwell | RTX 50-series, B100/B200 | NVFP4, MXFP8, INT8 ConvRot, INT4 ConvRot |\n\n"
                "Picking FP8 or NVFP4 on an unsupported card doesn't reliably fail at conversion time — "
                "the file often still gets written, it just won't load or run fast in ComfyUI. The "
                "**Target GPU** picker on the Convert tab exists to head that off. GGUF isn't in the table "
                "above because its conversion runs in pure Python on any GPU or CPU — what matters instead is "
                "whether ComfyUI-GGUF recognizes your model's architecture (see above).\n\n"
                "## What does \"txtfusion\" in a filename mean?\n"
                "It's not a format — it's a **layer name**. `kroma-v0.3-txtfusion-edition-...` was converted "
                "with ctq's `krea2` preset, whose high-precision keyword list literally includes `txtfusion` "
                "(alongside a few other sensitive layers). That layer stays BF16 instead of getting quantized, "
                "which is almost certainly why that edition looks better than a preset-less conversion. Pick "
                "**krea2** under **Model preset (layer exclusions)** to get the same effect — the app also "
                "auto-selects it when your filename/URL contains `txtfusion`, `krea`, or `kroma`.\n\n"
                "## Estimating output size\n"
                "**Estimate output size** reads only the input file's safetensors header (tensor names/shapes, "
                "a few KB) and estimates the converted size for your chosen format/preset/custom-layer rules — "
                "no torch or ctq install required for this step. It also flags whether the estimate fits your "
                "detected GPU's VRAM. It's a planning figure, not an exact prediction.\n\n"
                "## Credit\n"
                "All the actual quantization math lives in "
                "[silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant) (`ctq`). "
                "This app is just a GUI wrapper around it, and downloads/uploads nothing on its own besides "
                "the model file you point it at. It only ever produces `.safetensors` output — it has no "
                "GGUF code path at all."
            )

    # --- wiring ---

    def fmt_label_to_value(label: str) -> str:
        for lbl, val in FORMAT_CHOICES:
            if lbl == label:
                return val
        return "int8_convrot"

    def on_fmt_select(label: str):
        value = fmt_label_to_value(label)
        vis = on_format_change(value)
        return (value, *vis)

    fmt_outputs = [
        fmt_value, convrot_group, scaling_group, fmt_help,
        int4_notice, int4_group, int4_install_btn,
        gguf_notice, gguf_group, gguf_install_btn,
    ]

    fmt.change(on_fmt_select, inputs=[fmt], outputs=fmt_outputs)

    def run_int4_install(python_exe_v: str):
        log = "Installing comfy-kitchen...\n\n"
        yield log, gr.update(), gr.update(interactive=False)
        for line in stream_int4_install((python_exe_v or "").strip() or None):
            if line == "__INT4_INSTALL_OK__":
                log += "\n✅ comfy-kitchen installed.\n"
                yield log, gr.update(value=INT4_NOTICE_READY), gr.update(visible=False, interactive=True)
                return
            if line.startswith("__INT4_INSTALL_FAIL__"):
                code = line.split(":", 1)[1] if ":" in line else "?"
                log += f"\n❌ pip install failed (exit code {code}). See the log above for details.\n"
                yield log, gr.update(), gr.update(interactive=True)
                return
            log += line
            yield log, gr.update(), gr.update()

    int4_install_btn.click(
        run_int4_install, inputs=[python_exe], outputs=[log_box, int4_notice, int4_install_btn]
    )

    def run_gguf_install(python_exe_v: str):
        log = "Installing gguf...\n\n"
        yield log, gr.update(), gr.update(interactive=False)
        for line in stream_gguf_install((python_exe_v or "").strip() or None):
            if line == "__GGUF_INSTALL_OK__":
                log += "\n✅ gguf installed.\n"
                yield log, gr.update(value=GGUF_NOTICE_READY), gr.update(visible=False, interactive=True)
                return
            if line.startswith("__GGUF_INSTALL_FAIL__"):
                code = line.split(":", 1)[1] if ":" in line else "?"
                log += f"\n❌ pip install failed (exit code {code}). See the log above for details.\n"
                yield log, gr.update(), gr.update(interactive=True)
                return
            log += line
            yield log, gr.update(), gr.update()

    gguf_install_btn.click(
        run_gguf_install, inputs=[python_exe], outputs=[log_box, gguf_notice, gguf_install_btn]
    )

    INT4_TEMPLATE_ATTN = r"attn\.(wq|wk|wv|wo)\.weight"
    INT4_TEMPLATE_MLP = r"mlp\.(gate|up|down)\.weight"
    INT4_TEMPLATE_BALANCED = rf"{INT4_TEMPLATE_ATTN}|{INT4_TEMPLATE_MLP}"
    INT4_TEMPLATE_ALL = r".*"

    int4_tpl_attn_btn.click(lambda: INT4_TEMPLATE_ATTN, outputs=[int4_layers_regex])
    int4_tpl_mlp_btn.click(lambda: INT4_TEMPLATE_MLP, outputs=[int4_layers_regex])
    int4_tpl_balanced_btn.click(lambda: INT4_TEMPLATE_BALANCED, outputs=[int4_layers_regex])
    int4_tpl_all_btn.click(lambda: INT4_TEMPLATE_ALL, outputs=[int4_layers_regex])
    int4_tpl_clear_btn.click(lambda: "", outputs=[int4_layers_regex])

    def on_gpu_select(label: str):
        key = GPU_LABEL_TO_KEY.get(label, "not_sure")
        return GPU_PROFILE_BY_KEY[key].note

    def recommended_fmt_label(gpu_label: str) -> str:
        key = GPU_LABEL_TO_KEY.get(gpu_label, "not_sure")
        rec_key = GPU_PROFILE_BY_KEY[key].recommended_format
        return FORMAT_LABEL_BY_KEY.get(rec_key, FORMAT_CHOICES[0][0])

    gpu_dd.change(on_gpu_select, inputs=[gpu_dd], outputs=[gpu_note])
    apply_gpu_recommendation.click(recommended_fmt_label, inputs=[gpu_dd], outputs=[fmt]).then(
        on_fmt_select, inputs=[fmt], outputs=fmt_outputs
    )

    def krea2_group_visibility(label: str):
        return gr.update(visible=LABEL_TO_PRESET.get(label, "none") == "krea2")

    krea2_profile_outputs = [custom_layers, custom_type, custom_scaling_mode, custom_convrot, preset_dd]

    def krea2_balanced():
        return "", "none", "none", False, gr.update()

    def krea2_compact():
        return preset_highprec_regex("krea2") or "", "int8", "row", False, gr.update()

    def krea2_smallest():
        return "", "none", "none", False, PRESET_LABELS["none"]

    preset_dd.change(krea2_group_visibility, inputs=[preset_dd], outputs=[krea2_size_group])
    krea2_balanced_btn.click(krea2_balanced, outputs=krea2_profile_outputs)
    krea2_compact_btn.click(krea2_compact, outputs=krea2_profile_outputs)
    krea2_smallest_btn.click(krea2_smallest, outputs=krea2_profile_outputs).then(
        krea2_group_visibility, inputs=[preset_dd], outputs=[krea2_size_group]
    )

    EXCLUDE_TEMPLATE_NORMS = r"(norm|embed)"
    EXCLUDE_TEMPLATE_MOD = r"(modulat|\.gate\b)"
    EXCLUDE_TEMPLATE_FINAL = r"(final_layer|proj_out|last)"

    excl_tpl_norms_btn.click(lambda: EXCLUDE_TEMPLATE_NORMS, outputs=[exclude_layers])
    excl_tpl_mod_btn.click(lambda: EXCLUDE_TEMPLATE_MOD, outputs=[exclude_layers])
    excl_tpl_final_btn.click(lambda: EXCLUDE_TEMPLATE_FINAL, outputs=[exclude_layers])
    excl_tpl_clear_btn.click(lambda: "", outputs=[exclude_layers])

    custom_mixed_outputs = [custom_layers, custom_type, custom_scaling_mode, custom_convrot]

    def custom_tpl_potatoforge():
        # Matches PotatoForge/Kroma-INT8-Quants: attn.wk/wv stay plain
        # tensorwise INT8 while everything else uses the main format above
        # (pick "INT8 — ConvRot" in step 3 to reproduce their recipe exactly).
        return r"attn\.(wk|wv)\.weight", "int8", "tensor", False

    custom_tpl_potatoforge_btn.click(custom_tpl_potatoforge, outputs=custom_mixed_outputs)
    custom_tpl_clear_btn.click(lambda: ("", "none", "none", False), outputs=custom_mixed_outputs)

    source.change(on_source_change, inputs=[source], outputs=[local_group, hf_group]).then(
        refresh_local_models, outputs=[local_models_dd]
    )
    refresh_local_models_btn.click(refresh_local_models, outputs=[local_models_dd])
    demo.load(refresh_local_models, outputs=[local_models_dd])

    download_btn.click(do_hf_download, inputs=[input_hf_url, hf_token], outputs=[input_hf_local, download_status]).then(
        on_input_resolved, inputs=[input_hf_local, input_hf_url, preset_dd], outputs=[resolved_hint, preset_dd]
    ).then(krea2_group_visibility, inputs=[preset_dd], outputs=[krea2_size_group]).then(
        refresh_local_models, outputs=[local_models_dd]
    )
    input_local.change(
        on_local_input_resolved, inputs=[input_local, preset_dd], outputs=[resolved_hint, preset_dd]
    ).then(krea2_group_visibility, inputs=[preset_dd], outputs=[krea2_size_group])

    def apply_local_model_choice(path):
        return path if path else gr.update()

    local_models_dd.change(apply_local_model_choice, inputs=[local_models_dd], outputs=[input_local]).then(
        on_local_input_resolved, inputs=[input_local, preset_dd], outputs=[resolved_hint, preset_dd]
    ).then(krea2_group_visibility, inputs=[preset_dd], outputs=[krea2_size_group])

    auto_output.change(lambda auto: gr.update(interactive=not auto), inputs=[auto_output], outputs=[output_name])

    def refresh_preview(*args):
        input_local_v, input_hf_local_v, source_v = args[0], args[1], args[2]
        input_path = input_local_v if source_v == "Local file path" else input_hf_local_v
        rest = args[3:]
        if rest[preview_field_index["fmt_value"]] == "int4_convrot":
            return "(INT4 ConvRot doesn't run through ctq - see the notice above the format picker.)"
        if rest[preview_field_index["fmt_value"]] == "gguf":
            return "(GGUF doesn't run through ctq - see the notice above the format picker.)"
        try:
            opts = build_options(input_path or "placeholder.safetensors", *rest)
            return format_command(opts)
        except OptionsError as exc:
            return f"(invalid options: {exc})"

    preview_inputs = [
        input_local, input_hf_local, source,
        output_name, auto_output, fmt_value, quality_mode, convrot_group_size, dynamic_convrot,
        scaling_mode, block_size, preset_dd, comfy_quant, save_metadata, low_memory,
        exclude_layers, custom_layers, custom_type, custom_scaling_mode, custom_convrot,
        custom_convrot_group_size, custom_simple, fallback, fallback_simple,
        device, output_dtype, verbose,
        calib_samples, optimizer, num_iter, manual_seed,
    ]
    # Names for preview_inputs[3:] (the build_options-ordered "rest" slice), so
    # code that needs one specific field doesn't have to hand-count positions.
    preview_field_index = {
        name: i
        for i, name in enumerate([
            "output_name", "auto_output", "fmt_value", "quality_mode", "convrot_group_size", "dynamic_convrot",
            "scaling_mode", "block_size", "preset_dd", "comfy_quant", "save_metadata", "low_memory",
            "exclude_layers", "custom_layers", "custom_type", "custom_scaling_mode", "custom_convrot",
            "custom_convrot_group_size", "custom_simple", "fallback", "fallback_simple",
            "device", "output_dtype", "verbose",
            "calib_samples", "optimizer", "num_iter", "manual_seed",
        ])
    }
    assert len(preview_field_index) == len(preview_inputs) - 3

    for comp in preview_inputs:
        comp.change(refresh_preview, inputs=preview_inputs, outputs=[command_preview])

    fmt_value.change(refresh_preview, inputs=preview_inputs, outputs=[command_preview])

    def do_estimate(*args):
        input_local_v, input_hf_local_v, source_v = args[0], args[1], args[2]
        input_path = input_local_v if source_v == "Local file path" else input_hf_local_v
        rest = args[3 : len(preview_inputs)]
        int4_regex_v, int4_fallback_v, gguf_quant_type_v = args[-3], args[-2], args[-1]

        input_path = (input_path or "").strip()
        if not input_path or not Path(input_path).is_file():
            return "Pick an input file (local path, or download a Hugging Face file) first."
        vram = check_environment().gpu_vram_gb

        fmt_v = rest[preview_field_index["fmt_value"]]
        preset = LABEL_TO_PRESET.get(rest[preview_field_index["preset_dd"]], "none")
        exclude_layers_v = rest[preview_field_index["exclude_layers"]]
        if fmt_v == "int4_convrot":
            return estimate_int4_mixed_from_file(
                input_path, (int4_regex_v or "").strip() or None, preset=preset,
                exclude_regex=(exclude_layers_v or "").strip() or None,
                fallback_int8=int4_fallback_v, gpu_vram_gb=vram,
            )
        if fmt_v == "gguf":
            return estimate_gguf_from_file(
                input_path, gguf_quant_type_v, preset=preset,
                exclude_regex=(exclude_layers_v or "").strip() or None, gpu_vram_gb=vram,
            )
        try:
            opts = build_options(input_path, *rest)
        except OptionsError as exc:
            return f"Can't estimate: {exc}"
        return estimate_from_file(input_path, opts, gpu_vram_gb=vram)

    estimate_btn.click(
        do_estimate,
        inputs=preview_inputs + [int4_layers_regex, int4_fallback_int8, gguf_quant_type],
        outputs=[estimate_md],
    )

    convert_btn.click(
        run_convert,
        inputs=[
            input_local, input_hf_local, source, output_name, auto_output, fmt_value, quality_mode,
            convrot_group_size, dynamic_convrot, scaling_mode, block_size, preset_dd, comfy_quant,
            save_metadata, low_memory, exclude_layers, custom_layers, custom_type, custom_scaling_mode,
            custom_convrot, custom_convrot_group_size, custom_simple, fallback, fallback_simple,
            device, output_dtype, verbose,
            calib_samples, optimizer, num_iter, manual_seed, python_exe,
            int4_layers_regex, int4_fallback_int8, gguf_quant_type,
        ],
        outputs=[log_box, result_file, convert_progress],
        # We render our own bar into convert_progress; gr.Progress()'s built-in
        # overlay otherwise blankets *every* output of this event (log_box and
        # result_file included) for the whole run and can get stuck once streaming ends.
        show_progress="hidden",
    )

    refresh_btn.click(refresh_env, outputs=[env_md])


if __name__ == "__main__":
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo.queue().launch(
        server_name=os.environ.get("QUANT_GUI_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("QUANT_GUI_PORT", "7860")),
        theme=gr.themes.Soft(),
        css=CSS,
    )
