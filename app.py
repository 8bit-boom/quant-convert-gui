"""Quant Convert GUI

A friendly front end over silveroxides/convert_to_quant (`ctq`), the tool
that actually produces the FP8 / INT8 / INT8-ConvRot / NVFP4 safetensors
files used by projects like Kroma-Quant. This app builds the right `ctq`
command for you, runs it, and streams the log — no CLI flags to memorize.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

import gradio as gr

from quant_gui.cli_builder import ConvertOptions, OptionsError, build_args, format_command
from quant_gui.env_check import check_environment, report_markdown
from quant_gui.filters import preset_choices, preset_highprec_regex, preset_label, suggest_preset
from quant_gui.gpu_profiles import GPU_PROFILE_BY_KEY, GPU_PROFILES, detect_profile_key
from quant_gui.hf import HFUrlError, download as hf_download, download_repo as hf_download_repo, parse_hf_url
from quant_gui.loop_timing import LoopPhaseTimer
from quant_gui import llamacpp_backend as lcpp
from quant_gui import gguf_bench as gb
from quant_gui.gguf_backend import stream_gguf_conversion, stream_install as stream_gguf_install
from quant_gui.gguf_backend import QUANT_TYPE_CHOICES as GGUF_QUANT_TYPE_CHOICES
from quant_gui.gguf_backend import SUPPORTED_ARCH_NAMES as GGUF_SUPPORTED_ARCH_NAMES
from quant_gui.gguf_backend import is_available as gguf_is_available
from quant_gui.gguf_inspect import format_inspection, inspect_gguf
from quant_gui import gguf_edit as ge
from quant_gui.int4_backend import stream_int4_conversion, stream_install as stream_int4_install
from quant_gui.int4_backend import is_available as int4_is_available
from quant_gui import runner
from quant_gui.runner import stream_conversion
from quant_gui.size_estimate import estimate_from_file, estimate_gguf_from_file, estimate_int4_mixed_from_file
from quant_gui import checkpoints as ckpt
from quant_gui import run_control
from quant_gui import run_history as rh
from quant_gui.run_control import RunControl
from quant_gui import gpu_quant as _gpu_quant
from quant_gui import ui_settings

APP_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = APP_DIR / "downloads"
OUTPUT_DIR = APP_DIR / "converted"
CHECKPOINT_ROOT = ckpt.checkpoint_root(APP_DIR)
RUN_HISTORY = APP_DIR / rh.HISTORY_NAME
LLAMACPP_DIR = lcpp.default_llamacpp_dir(APP_DIR)
LLM_MODELS_DIR = APP_DIR / "llm_models"

# Persisted UI settings (ui_settings.json). GPU quantization is the first
# one: the checkbox defaults to the saved value, and a saved "on" is applied
# to the environment here, at import time, so worker threads spawned later
# see it without any plumbing through the convert handlers.
_initial_gpu_quant = bool(ui_settings.load_settings(APP_DIR).get("gguf_gpu_quant", False))
if _initial_gpu_quant:
    os.environ[_gpu_quant.GPU_QUANT_ENV] = "1"

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
    fast_math: bool,
    loss_sync_batch: str,
    snapshot_interval: str,
    compile_loop: bool,
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
        fast_math=fast_math,
        loss_sync_batch=int(loss_sync_batch) if str(loss_sync_batch or "").strip() else 1,
        snapshot_interval=int(snapshot_interval) if str(snapshot_interval or "").strip() else 1,
        compile_loop=compile_loop,
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
    checkpoint=None,
    save_progress: bool = False,
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

    if checkpoint is None and save_progress:
        checkpoint = ckpt.create_checkpoint(
            CHECKPOINT_ROOT, "int4",
            params={
                "input_path": input_path, "output_path": output_path,
                "output_name": output_name, "auto_output": auto_output,
                "int4_layers_regex": int4_layers_regex, "preset": preset_label_value,
                "exclude_layers": exclude_layers, "fallback_int8": int4_fallback_int8,
                "device": device,
            },
            output_path=output_path,
        )

    log = f"Converting (INT4 ConvRot via comfy_kitchen, device={device or 'cpu'})\n"
    log += f"  input:  {input_path}\n  output: {output_path}\n"
    log += f"  INT4 layers regex: {int4_layers_regex or '(none — no layers go INT4)'}\n"
    log += f"  preset: {preset}\n\n"
    if checkpoint is not None and checkpoint.completed_count:
        log += (
            f"⏯ Resuming: {checkpoint.completed_count}/{checkpoint.total} tensors already done "
            f"- replaying them from the checkpoint, recomputing the rest.\n\n"
        )
    elif checkpoint is not None:
        log += "💾 Resumable progress is being saved - you can Stop & resume this run at any point.\n\n"
    yield log, None, _progress_bar_html(0, "Starting INT4 conversion...")

    control = RunControl()
    run_control.register(control)
    try:
        result_path = None
        for item in stream_int4_conversion(
            input_path, output_path, (int4_layers_regex or "").strip() or None,
            preset=preset, exclude_regex=(exclude_layers or "").strip() or None,
            fallback_int8=int4_fallback_int8, device=(device or "cpu").strip() or "cpu",
            control=control, checkpoint=checkpoint,
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
                if checkpoint is not None:
                    # The run completed - the full output file exists, so the
                    # per-tensor shards are dead weight; clean them up.
                    ckpt.delete_checkpoint(checkpoint.path.parent, checkpoint.id)
                yield log, result_path, _progress_bar_html(1.0, "Done")
            elif kind == "cancelled":
                done = checkpoint.completed_count if checkpoint is not None else 0
                total = checkpoint.total if checkpoint is not None else 0
                frac = done / total if total else 0
                log += (
                    f"\n⏹ Stopped - progress saved ({done}/{total} tensors).\n"
                    f"Resume it any time from the 'Resume a saved run' list below (even after closing the app).\n"
                )
                yield log, None, _progress_bar_html(frac, f"Paused at {done}/{total} - resume from checkpoint")
            elif kind == "fail":
                log += f"\n❌ Conversion failed: {item[1]}\n"
                if checkpoint is not None and checkpoint.completed_count:
                    log += (
                        f"💡 {checkpoint.completed_count}/{checkpoint.total} finished tensors are in the "
                        f"checkpoint - you can resume instead of starting over.\n"
                    )
                yield log, None, _progress_bar_html(1.0, "Failed")
    finally:
        run_control.register(None)


def run_gguf_convert(
    input_path: str,
    output_name: str,
    auto_output: bool,
    preset_label_value: str,
    gguf_quant_type: str,
    exclude_layers: str,
    checkpoint=None,
    save_progress: bool = False,
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

    if checkpoint is None and save_progress:
        checkpoint = ckpt.create_checkpoint(
            CHECKPOINT_ROOT, "gguf",
            params={
                "input_path": input_path, "output_path": output_path,
                "output_name": output_name, "auto_output": auto_output,
                "quant_type": gguf_quant_type, "preset": preset_label_value,
                "exclude_layers": exclude_layers,
            },
            output_path=output_path,
        )

    log = f"Converting to GGUF ({gguf_quant_type} via the gguf package)\n"
    log += f"  input:  {input_path}\n  output: {output_path}\n"
    log += f"  preset: {preset}\n\n"
    if checkpoint is not None and checkpoint.completed_count:
        log += (
            f"⏯ Resuming: {checkpoint.completed_count}/{checkpoint.total} tensors already done "
            f"- reusing their packed data from the checkpoint, quantizing the rest.\n\n"
        )
    elif checkpoint is not None:
        log += "💾 Resumable progress is being saved - you can Stop & resume this run at any point.\n\n"
    yield log, None, _progress_bar_html(0, "Starting GGUF conversion...")

    control = RunControl()
    run_control.register(control)
    try:
        result_path = None
        for item in stream_gguf_conversion(
            input_path, output_path, gguf_quant_type,
            preset=preset, exclude_regex=(exclude_layers or "").strip() or None,
            control=control, checkpoint=checkpoint,
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
                if checkpoint is not None:
                    ckpt.delete_checkpoint(checkpoint.path.parent, checkpoint.id)
                yield log, result_path, _progress_bar_html(1.0, "Done")
            elif kind == "cancelled":
                done = checkpoint.completed_count if checkpoint is not None else 0
                total = checkpoint.total if checkpoint is not None else 0
                frac = done / total if total else 0
                log += (
                    f"\n⏹ Stopped - progress saved ({done}/{total} tensors).\n"
                    f"Resume it any time from the 'Resume a saved run' list below (even after closing the app).\n"
                )
                yield log, None, _progress_bar_html(frac, f"Paused at {done}/{total} - resume from checkpoint")
            elif kind == "fail":
                log += f"\n❌ Conversion failed: {item[1]}\n"
                if checkpoint is not None and checkpoint.completed_count:
                    log += (
                        f"💡 {checkpoint.completed_count}/{checkpoint.total} finished tensors are in the "
                        f"checkpoint - you can resume instead of starting over.\n"
                    )
                yield log, None, _progress_bar_html(1.0, "Failed")
    finally:
        run_control.register(None)


def llamacpp_status_markdown() -> str:
    def ok(flag: bool) -> str:
        return "✅" if flag else "❌"

    cloned = lcpp.is_cloned(LLAMACPP_DIR)
    venv_ready = lcpp.is_venv_ready(LLAMACPP_DIR)
    tf_version = lcpp.venv_transformers_version(LLAMACPP_DIR) if venv_ready else None
    tf_stale = venv_ready and lcpp.transformers_too_old(tf_version)
    quantize_built = lcpp.is_quantize_built(LLAMACPP_DIR)
    imatrix_built = lcpp.is_imatrix_built(LLAMACPP_DIR)
    tf_line = (
        f"**Python deps installed** (transformers/sentencepiece/gguf, in their own venv) — "
        f"{ok(venv_ready)}"
        + (f" `transformers {tf_version}`" if tf_version else "")
        + (
            f"\n\n⚠️ transformers {tf_version or 'unknown'} is too old for Gemma 3/4 tokenizers "
            f"(crashes with `'list' object has no attribute 'keys'`) - re-run **2. Python deps** "
            f"to upgrade to {lcpp.TRANSFORMERS_MIN_SPEC}."
            if tf_stale else ""
        )
    )
    return (
        f"**llama.cpp cloned** — {ok(cloned)} `{LLAMACPP_DIR}`\n\n"
        f"{tf_line}\n\n"
        f"**llama-quantize + llama-imatrix ready** (built from source, or official prebuilt binaries "
        f"downloaded - for real K-quants like Q4_K_M, and for imatrix/dynamic-"
        f"style calibrated quants) — {ok(quantize_built and imatrix_built)} "
        + ("_optional - skip this if F16/BF16/Q8_0 is enough for you_" if not (quantize_built and imatrix_built) else "")
    )


def run_llamacpp_setup_step(step: str, jobs: str):
    """step is 'clone', 'venv', or 'quantize' - drives one of the three
    setup buttons, all sharing the same log box/status refresh."""
    log = ""
    if step == "clone":
        stream = lcpp.stream_clone_or_update(LLAMACPP_DIR)
    elif step == "venv":
        stream = lcpp.stream_setup_venv(LLAMACPP_DIR)
    else:
        try:
            jobs_n = int(jobs) if (jobs or "").strip() else None
        except ValueError:
            jobs_n = None
        stream = lcpp.stream_build_quantize(LLAMACPP_DIR, jobs=jobs_n)

    for line in stream:
        if line == "__OK__":
            yield log, llamacpp_status_markdown()
        elif line.startswith("__FAIL__"):
            yield log, llamacpp_status_markdown()
        else:
            log += line
            yield log, gr.update()


def llm_downloaded_models(models_dir=None, hf_cache=None) -> list[str]:
    """Model directories downloaded through this tab plus whatever sits in the
    Hugging Face cache - so a model fetched in any session (by this app, by
    another tool, gated or not) shows up in the picker without re-downloading.
    Only dirs that actually look like HF models (config.json) are listed.
    """
    models_dir = Path(LLM_MODELS_DIR if models_dir is None else models_dir)
    hf_cache = Path(hf_cache) if hf_cache is not None else Path.home() / ".cache" / "huggingface" / "hub"
    out: list[str] = []
    if models_dir.is_dir():
        for d in sorted(models_dir.iterdir()):
            if d.is_dir() and (d / "config.json").is_file():
                out.append(str(d))
    if hf_cache.is_dir():
        for repo in sorted(hf_cache.iterdir()):
            if not repo.is_dir() or not repo.name.startswith("models--"):
                continue
            snaps = repo / "snapshots"
            if not snaps.is_dir():
                continue
            for s in sorted(snaps.iterdir()):
                if s.is_dir() and (s / "config.json").is_file():
                    out.append(str(s))
    seen: set[str] = set()
    dedup: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


def newest_model_gguf(directory=None) -> str | None:
    """Newest real model .gguf in `directory` (imatrix outputs excluded by
    name, since those also carry a .gguf extension)."""
    directory = Path(OUTPUT_DIR if directory is None else directory)
    if not directory.is_dir():
        return None
    candidates = [
        p for p in directory.glob("*.gguf")
        if "imatrix" not in p.name.lower() and p.stat().st_size > 0
    ]
    return str(max(candidates, key=lambda p: p.stat().st_mtime)) if candidates else None


def newest_imatrix(directory=None) -> str | None:
    """Newest imatrix file (`.imatrix` or `.imatrix.gguf`) in `directory`."""
    directory = Path(OUTPUT_DIR if directory is None else directory)
    if not directory.is_dir():
        return None
    candidates = [p for p in directory.glob("*.imatrix*") if p.stat().st_size > 0]
    return str(max(candidates, key=lambda p: p.stat().st_mtime)) if candidates else None


def _resolve_calibration(calib_mode: str, calibration_file: str) -> str:
    """Auto mode -> '' (backend substitutes its bundled default); Custom -> the given path."""
    if (calib_mode or "").strip().lower().startswith("auto"):
        return ""
    return (calibration_file or "").strip()


def run_llm_download(source: str, repo_id: str, local_dir: str, token: str):
    if source == "Local directory":
        path = (local_dir or "").strip()
        if not path or not Path(path).is_dir():
            yield f"That doesn't look like a directory: {path}", gr.update()
            return
        yield f"Using local directory: {path}\n", path
        return

    repo_id = (repo_id or "").strip()
    if not repo_id:
        yield "Paste a Hugging Face repo ID first (e.g. google/gemma-3-4b-it).", gr.update()
        return

    import queue
    import threading

    q: "queue.Queue" = queue.Queue()
    SENTINEL = object()

    def worker():
        try:
            dest = str(LLM_MODELS_DIR / repo_id.replace("/", "__"))
            local_path = hf_download_repo(repo_id, dest, token=(token or "").strip() or None)
            q.put(("ok", local_path))
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            q.put(("fail", str(exc)))
        finally:
            q.put(SENTINEL)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    log = f"Downloading {repo_id} from Hugging Face (this can take a while for multi-GB models)...\n"
    yield log, gr.update()
    elapsed = 0
    while True:
        try:
            item = q.get(timeout=3)
        except queue.Empty:
            elapsed += 3
            yield log + f"\n({elapsed}s elapsed...)", gr.update()
            continue
        if item is SENTINEL:
            break
        kind, payload = item
        if kind == "ok":
            log += f"\n✅ Downloaded to {payload}\n"
            yield log, payload
        else:
            log += f"\n❌ Download failed: {payload}\n"
            yield log, gr.update()


def run_llm_convert(model_dir: str, output_name: str, outtype: str):
    model_dir = (model_dir or "").strip()
    if not model_dir or not Path(model_dir).is_dir():
        yield f"Pick a model directory first (download one above, or point at an existing local HF model folder).", None
        return
    if not lcpp.is_venv_ready(LLAMACPP_DIR):
        yield "llama.cpp's Python environment isn't set up yet - see the Setup section above.", None
        return
    tf_version = lcpp.venv_transformers_version(LLAMACPP_DIR)
    tf_warn = ""
    if lcpp.transformers_too_old(tf_version):
        tf_warn = (
            f"⚠️ transformers {tf_version or 'unknown'} is installed; Gemma 3/4 models crash on the "
            f"tokenizer step with this version ('list' object has no attribute 'keys'). "
            f"Re-run Setup step 2 (Python deps) to upgrade to {lcpp.TRANSFORMERS_MIN_SPEC}.\n\n"
        )

    LLM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    name = (output_name or "").strip()
    if name:
        output_path = str(OUTPUT_DIR / name) if not os.path.isabs(name) and os.sep not in name else name
    else:
        stem = Path(model_dir).name
        output_path = str(OUTPUT_DIR / f"{stem}-{outtype}.gguf")

    log = f"Converting {model_dir} to GGUF (outtype={outtype})\n  output: {output_path}\n\n{tf_warn}"
    yield log, None
    result_path = None
    for line in lcpp.stream_convert_to_gguf(LLAMACPP_DIR, model_dir, output_path, outtype=outtype):
        if line == "__OK__":
            result_path = output_path if Path(output_path).is_file() else None
            log += f"\n✅ Conversion finished.\nOutput: {result_path or output_path}\n"
            yield log, result_path
        elif line.startswith("__FAIL__"):
            log += f"\n❌ Conversion failed (see log above).\n"
            yield log, None
        else:
            log += line
            yield log, result_path


def run_llm_generate_imatrix(model_gguf: str, calib_mode: str, calibration_file: str, output_name: str):
    model_gguf = (model_gguf or "").strip()
    auto_note = ""
    if not model_gguf:
        detected = newest_model_gguf()
        if detected:
            model_gguf = detected
            auto_note = f"(auto-detected newest model GGUF: {detected})\n"
    if not model_gguf or not Path(model_gguf).is_file():
        yield (
            "No model GGUF given and none auto-detected in the output folder - convert in step 3 "
            "first, or paste any .gguf path.",
            None,
        )
        return
    if not lcpp.is_imatrix_built(LLAMACPP_DIR):
        yield "llama-imatrix isn't built yet - see the Setup section above.", None
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = (output_name or "").strip()
    if name:
        output_path = str(OUTPUT_DIR / name) if not os.path.isabs(name) and os.sep not in name else name
    else:
        output_path = str(OUTPUT_DIR / f"{Path(model_gguf).stem}.imatrix.gguf")

    calib = _resolve_calibration(calib_mode, calibration_file)
    log = f"Generating importance matrix for {model_gguf}\n{auto_note}"
    if calib_mode and not (calib_mode or "").strip().lower().startswith("auto") and not calib:
        yield "Pick a calibration file, or switch Calibration back to Auto (bundled default).", None
        return
    log += f"  calibration: {calib or 'bundled default (quant_gui/data/default_calibration.txt)'}\n"
    log += f"  output: {output_path}\n\n"
    yield log, None
    result_path = None
    for line in lcpp.stream_generate_imatrix(LLAMACPP_DIR, model_gguf, output_path, calibration_file=calib or None):
        if line == "__OK__":
            result_path = output_path if Path(output_path).is_file() else None
            log += f"\n✅ Importance matrix generated.\nOutput: {result_path or output_path}\n"
            yield log, result_path
        elif line.startswith("__FAIL__"):
            log += "\n❌ Generation failed (see log above).\n"
            yield log, None
        else:
            log += line
            yield log, result_path


def run_llm_quantize(input_gguf: str, output_name: str, quant_type: str, imatrix_file: str, tensor_type_file: str):
    input_gguf = (input_gguf or "").strip()
    auto_note = ""
    if not input_gguf:
        detected = newest_model_gguf()
        if detected:
            input_gguf = detected
            auto_note = f"(auto-detected newest model GGUF: {detected})\n"
    if not input_gguf or not Path(input_gguf).is_file():
        yield (
            "No input GGUF given and none auto-detected in the output folder - convert in step 3 "
            "first, or paste any .gguf path.",
            None,
        )
        return
    if not lcpp.is_quantize_built(LLAMACPP_DIR):
        yield "llama-quantize isn't built yet - see the Setup section above.", None
        return

    imatrix_file = (imatrix_file or "").strip()
    if not imatrix_file:
        detected_im = newest_imatrix()
        if detected_im:
            imatrix_file = detected_im
            auto_note += f"(auto-detected imatrix: {detected_im})\n"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = (output_name or "").strip()
    if name:
        output_path = str(OUTPUT_DIR / name) if not os.path.isabs(name) and os.sep not in name else name
    else:
        stem = Path(input_gguf).stem
        output_path = str(OUTPUT_DIR / f"{stem}-{quant_type}.gguf")

    log = f"Quantizing {input_gguf} -> {quant_type}\n{auto_note}"
    log += f"  output: {output_path}\n"
    if imatrix_file:
        log += f"  imatrix: {imatrix_file}\n"
    if (tensor_type_file or "").strip():
        log += f"  tensor-type overrides: {tensor_type_file.strip()}\n"
    log += "\n"
    yield log, None
    result_path = None
    for line in lcpp.stream_quantize(
        LLAMACPP_DIR, input_gguf, output_path, quant_type,
        imatrix_file=(imatrix_file or "").strip() or None,
        tensor_type_file=(tensor_type_file or "").strip() or None,
    ):
        if line == "__OK__":
            result_path = output_path if Path(output_path).is_file() else None
            log += f"\n✅ Quantization finished.\nOutput: {result_path or output_path}\n"
            yield log, result_path
        elif line.startswith("__FAIL__"):
            log += f"\n❌ Quantization failed (see log above).\n"
            yield log, None
        else:
            log += line
            yield log, result_path


def run_llm_find_best(input_gguf: str, imatrix_file: str, target_bpw: str = "", quants: list[str] | None = None):
    """Sweep a target-size quant family on `input_gguf` and pick the winner.

    `target_bpw` selects the candidate set (~2/~3/~4/~5 bpw family);
    `quants` overrides it (tests). Streams progress into the LLM log; the
    final yield returns the winning quant type (pre-selects the Quant type
    dropdown) and the imatrix path used (fills the imatrix box). If no
    imatrix is given but llama-imatrix is built, one is generated first
    with the bundled default calibration - imatrix is what makes the
    "Dynamic" recipes work, so the sweep is only half-useful without it.
    """
    input_gguf = (input_gguf or "").strip()
    auto_note = ""
    if not input_gguf:
        detected = newest_model_gguf()
        if detected:
            input_gguf = detected
            auto_note = f"(auto-detected newest model GGUF: {detected})\n"
    if not input_gguf or not Path(input_gguf).is_file():
        yield "Pick an input GGUF file first (the output of the conversion step above, or any existing .gguf file).", None, gr.update(), gr.update()
        return
    if not lcpp.is_quantize_built(LLAMACPP_DIR):
        yield "llama-quantize isn't built yet - see the Setup section above.", None, gr.update(), gr.update()
        return

    import queue as _queue
    import tempfile
    import threading

    if quants is None:
        quants = gb.family_candidates(target_bpw)

    q: "_queue.Queue" = _queue.Queue()
    SENTINEL = object()
    state: dict = {}

    def wlog(msg):
        q.put(str(msg))

    def worker():
        try:
            im = (imatrix_file or "").strip()
            if not im and lcpp.is_imatrix_built(LLAMACPP_DIR):
                cand = str(OUTPUT_DIR / f"{Path(input_gguf).stem}.bench.imatrix")
                if Path(cand).is_file():
                    im = cand
                    wlog(f"Reusing previously generated imatrix: {cand}\n")
                else:
                    wlog("No imatrix given - generating one first with llama-imatrix (bundled default calibration)...\n")
                    for line in lcpp.stream_generate_imatrix(
                        LLAMACPP_DIR, input_gguf, cand, calibration_file=None, chunks=16,
                    ):
                        if line == "__OK__":
                            break
                        if line.startswith("__FAIL__"):
                            wlog(f"imatrix generation failed ({line}) - sweeping without imatrix.\n")
                            break
                    if Path(cand).is_file():
                        im = cand
            if not im:
                wlog("⚠️ No imatrix available - sweeping without it. IQ-quant results will be degraded or fail.\n")
            out_dir = Path(tempfile.mkdtemp(prefix="quant-sweep-"))
            try:
                state["result"] = gb.run_sweep(
                    ref_gguf=input_gguf,
                    imatrix_file=im or None,
                    out_dir=out_dir,
                    quantize_bin=lcpp._quantize_binary(LLAMACPP_DIR),
                    quants=quants,
                    log=wlog,
                )
                state["imatrix"] = im
            finally:
                import shutil as _shutil

                _shutil.rmtree(out_dir, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001 - surfaced in the log, not swallowed
            state["error"] = str(exc)
        finally:
            q.put(SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    log = (
        f"Finding the best {target_bpw or '~3 bpw'} quant setting for {input_gguf}\n"
        f"{auto_note}"
        f"  candidates: {', '.join(quants)}\n"
        "  measuring size, bits-per-weight, time, and reconstruction error vs this file\n\n"
    )
    yield log, None, gr.update(), gr.update()
    while True:
        item = q.get()
        if item is SENTINEL:
            break
        log += item + "\n"
        yield log, None, gr.update(), gr.update()

    if "error" in state:
        log += f"\n❌ Sweep failed: {state['error']}\n"
        yield log, None, gr.update(), gr.update()
        return
    result = state["result"]
    im = state.get("imatrix", "")
    log += "\n" + gb.format_report(result) + "\n"
    try:
        picks = gb.best_settings(result)
    except ValueError as exc:
        log += f"\n❌ No usable results: {exc}\n"
        yield log, None, gr.update(), gr.update()
        return
    winner = picks["best_value"]
    log += (
        f"\n🏆 Winner: {winner.label} — pre-selected in the Quant type dropdown.\n"
        f"   (best quality: {picks['best_quality'].label}, smallest: {picks['smallest'].label})\n"
        "   Now just pick an output filename and hit Quantize.\n"
    )
    global _LAST_SWEEP
    _LAST_SWEEP = {
        "model": input_gguf,
        "imatrix": im or "",
        "winner_quant": winner.quant,
        "winner_bpw": float(getattr(winner, "bpw", 0.0) or 0.0),
        "family": gb.family_for_bpw(float(getattr(winner, "bpw", 0.0) or 0.0)),
    }
    log += "   Or hit 'Tune winner (Dynamic 3.0)' in step 6 to per-tensor-tune at this size.\n"
    yield log, None, winner.quant, gr.update(value=im) if im else gr.update()


_LAST_SWEEP: dict = {}


def _stage_failed(stage_log: str) -> bool:
    return "❌" in stage_log


def run_llm_auto_pipeline(input_gguf: str, target_bpw: str, output_name: str,
                          val_baseline: str, val_text: str):
    """One-click chain: sweep -> tune winner -> perplexity-validate.

    Each stage reuses the previous one's artifacts (the sweep's bench imatrix,
    the tune's output file via newest_smart_gguf auto-detect), so a re-run
    after an interruption skips work that's already on disk. Stops at the
    first stage that fails.
    """
    header = (
        "# Auto pipeline\n\n"
        "Stages: **1. sweep** the target-size family → **2. per-tensor-tune the winner** "
        "(Dynamic 3.0) → **3. validate** vs a plain baseline quant with perplexity.\n\n"
    )
    yield header, None

    # -- stage 1: sweep ----------------------------------------------------
    log = header + "## Stage 1/3: find-best sweep\n\n"
    sweep_log = ""
    for item in run_llm_find_best(input_gguf, "", target_bpw):
        sweep_log = item[0]
        yield log + sweep_log, None
    if _stage_failed(sweep_log) or not _LAST_SWEEP.get("winner_quant"):
        yield log + sweep_log + "\n\n**Pipeline stopped** - sweep did not produce a winner.\n", None
        return

    # -- stage 2: tune winner ----------------------------------------------
    log += sweep_log + "\n\n## Stage 2/3: per-tensor tuning (Dynamic 3.0)\n\n"
    tune_log, tune_file = "", None
    for tune_log, tune_file in run_llm_tune_winner(output_name):
        yield log + tune_log, tune_file
    if _stage_failed(tune_log) or not tune_file:
        yield log + tune_log + "\n\n**Pipeline stopped** - tuning did not produce a file.\n", tune_file
        return

    # -- stage 3: validate ---------------------------------------------------
    log += tune_log + "\n\n## Stage 3/3: perplexity validation\n\n"
    val_log = ""
    for val_log, val_file in run_llm_validate("", "", val_baseline or "Q4_K_M", val_text, ""):
        yield log + val_log, val_file or tune_file
    if _stage_failed(val_log):
        yield log + val_log + "\n\n**Pipeline finished with a validation failure** - the tuned file from stage 2 is still on disk.\n", tune_file
        return
    yield log + val_log + "\n\n✅ **Auto pipeline complete.**\n", tune_file



def run_llm_tune_winner(output_name: str):
    """Hand the last sweep winner to the Dynamic tuner: same model + imatrix,
    budget from the winner's bits-per-weight family, K-ladder assignment."""
    sweep = dict(_LAST_SWEEP)
    if not sweep.get("model"):
        yield "Run 'Find best quant (sweep)' first - there is no winner to tune yet.", None
        return
    name = (output_name or "").strip()
    if not name:
        stem = Path(sweep["model"]).stem
        name = f"{stem}-dynamic3-{sweep['winner_quant'].lower()}.gguf"
    yield from run_llm_smart_tune(
        sweep["model"], sweep.get("imatrix", ""),
        "K-ladder (rank-mapped)", sweep["family"], "", name,
    )


def run_llm_smart_tune(model_gguf: str, imatrix_file: str, mode: str, target_size: str, budget_gb: str, output_name: str):
    """Stage 1-3 of the smart tuner: score sensitivity -> assign under budget
    -> write tensor-type file -> quantize with it. Stage 1 runs in a worker
    thread (pure-Python quantization is CPU-heavy); progress streams here."""
    import queue as _queue
    import threading

    from quant_gui import smart_quant as sq

    model_gguf = (model_gguf or "").strip()
    auto_note = ""
    if not model_gguf:
        detected = newest_model_gguf()
        if detected:
            model_gguf = detected
            auto_note = f"(auto-detected newest model GGUF: {detected})\n"
    if not model_gguf or not Path(model_gguf).is_file():
        yield "No model GGUF given and none auto-detected in the output folder - convert in step 3 first, or paste any .gguf path.", None
        return
    imatrix_file = (imatrix_file or "").strip()
    if not imatrix_file:
        detected_im = newest_imatrix()
        if detected_im:
            imatrix_file = detected_im
            auto_note += f"(auto-detected imatrix: {detected_im})\n"
    if not imatrix_file or not Path(imatrix_file).is_file():
        yield "No imatrix given and none auto-detected - generate one in step 4 first (Auto calibration is fine).", None
        return
    if not lcpp.is_quantize_built(LLAMACPP_DIR):
        yield "llama-quantize isn't built yet - see the Setup section above.", None
        return
    try:
        budget = float((budget_gb or "").strip()) * 1e9
    except ValueError:
        budget = 0.0
    family_note = ""
    if (target_size or "").strip().lower() != "custom":
        # target-size selector drives the budget: family file-bpw x params / 8
        try:
            params = sq.count_params(model_gguf)
            budget = float(sq.family_budget_bytes(target_size, params))
            family_note = (
                f"  target: {target_size} ({sq.FAMILY_FILE_BPW.get(target_size):.2f} file bpw "
                f"x {params / 1e9:.1f}B params)\n"
            )
        except Exception as exc:  # noqa: BLE001 - fall back to the textbox budget
            family_note = f"  (could not size {target_size!r} from the model: {exc}; using the GB box)\n"
    if budget <= 0:
        yield f"Size budget must be a positive number of GB, got {budget_gb!r}.", None
        return

    use_k_ladder = (mode or "").lower().startswith("k-ladder")
    log = (
        f"Smart per-tensor tuning: {model_gguf}\n{auto_note}"
        f"  imatrix: {imatrix_file}\n"
        f"{family_note}"
        f"  mode: {mode}   budget: {budget / 1e9:.2f} GB\n\n"
        "Stage 1 - imatrix-weighted sensitivity ranking (pure Python, CPU-bound)...\n"
    )
    yield log, None

    q: "_queue.Queue" = _queue.Queue()
    SENTINEL = object()
    state: dict = {}

    def worker():
        try:
            scores = sq.score_model(
                model_gguf, imatrix_file,
                progress=lambda m: q.put(str(m) + "\n"),
            )
            state["scores"] = scores
        except Exception as exc:  # noqa: BLE001 - surfaced in the log
            state["error"] = str(exc)
        finally:
            q.put(SENTINEL)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is SENTINEL:
            break
        log += item
        yield log, None
    if "error" in state:
        yield log + f"\n❌ Scoring failed: {state['error']}\n", None
        return

    scores = state["scores"]
    scored = [s for s in scores if not s.skipped and s.err]
    n_moe = len([s for s in scored if s.is_moe])
    log += f"\nscored {len(scored)} tensors ({n_moe} MoE expert stacks).\nStage 2 - assignment under budget...\n"
    yield log, None

    try:
        if use_k_ladder:
            assignment, report = sq.assign_k_ladder(scores, int(budget))
            base_ftype = "Q4_K_M"
        else:
            assignment, report = sq.assign_legacy(scores, int(budget))
            base_ftype = "Q4_0"
    except sq.SmartQuantError as exc:
        yield log + f"\n❌ Assignment failed: {exc}\n", None
        return
    for line in report:
        log += f"  {line}\n"
    for line in sq.moe_expert_report(scores):
        log += f"  MoE: {line}\n"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(model_gguf).stem
    tt_path = OUTPUT_DIR / f"{stem}.tensor-types.txt"
    tt_path.write_text(sq.emit_tensor_type_file(assignment), encoding="utf-8")
    report_path = OUTPUT_DIR / f"{stem}.smart-report.md"
    report_path.write_text(
        sq.smart_report_markdown(scores, assignment, report), encoding="utf-8",
    )
    log += (f"\nStage 3 - tensor-type file: {tt_path}\n"
            f"  tuning report: {report_path}\n"
            f"quantizing with llama-quantize (base {base_ftype})...\n")
    yield log, None

    name = (output_name or "").strip()
    if name:
        output_path = str(OUTPUT_DIR / name) if not os.path.isabs(name) and os.sep not in name else name
    else:
        suffix = "smart" if use_k_ladder else "smart-legacy"
        output_path = str(OUTPUT_DIR / f"{stem}-{suffix}.gguf")
    result_path = None
    for line in lcpp.stream_quantize(
        LLAMACPP_DIR, model_gguf, output_path, base_ftype,
        imatrix_file=imatrix_file, tensor_type_file=str(tt_path),
    ):
        if line == "__OK__":
            result_path = output_path if Path(output_path).is_file() else None
            size_gb = Path(output_path).stat().st_size / 1e9 if result_path else 0.0
            log += f"\n✅ Done. Output: {result_path or output_path} ({size_gb:.2f} GB)\n"
            yield log, result_path
        elif line.startswith("__FAIL__"):
            log += "\n❌ Quantization failed (see log above).\n"
            yield log, None
        else:
            log += line
            yield log, result_path


def newest_smart_gguf(directory=None) -> str | None:
    """Newest *.gguf with 'smart' in the name (tuner output), else None."""
    directory = Path(OUTPUT_DIR if directory is None else directory)
    if not directory.is_dir():
        return None
    candidates = [
        p for p in directory.glob("*smart*.gguf")
        if p.stat().st_size > 0
    ]
    return str(max(candidates, key=lambda p: p.stat().st_mtime)) if candidates else None


def run_llm_validate(tuned_gguf: str, ref_gguf: str, baseline_type: str, text_file: str, ngl: str):
    """Stage-3 validation: measure perplexity of the tuned file and a plain
    baseline quant of the same reference on a held-out text, and compare."""
    tuned_gguf = (tuned_gguf or "").strip()
    if not tuned_gguf:
        detected = newest_smart_gguf()
        if detected:
            tuned_gguf = detected
    if not tuned_gguf or not Path(tuned_gguf).is_file():
        yield ("No tuned GGUF given and none auto-detected - run step 6 (Tune + quantize) "
               "first, or paste a .gguf path."), None
        return
    text_file = (text_file or "").strip()
    if not text_file or not Path(text_file).is_file():
        yield f"Held-out validation text not found: {text_file!r} - point at a .txt the calibration didn't use.", None
        return
    if not lcpp.is_perplexity_built(LLAMACPP_DIR):
        yield ("llama-perplexity isn't available - re-download the prebuilt binaries "
               "(Setup step 3), the release zip ships it."), None
        return
    if baseline_type not in lcpp.QUANT_TYPE_CHOICES:
        yield f"Unknown baseline quant type: {baseline_type!r}.", None
        return
    try:
        ngl_n = int((ngl or "").strip())
    except ValueError:
        ngl_n = 99

    ref_gguf = (ref_gguf or "").strip()
    auto_note = ""
    if not ref_gguf:
        detected = newest_model_gguf()
        if detected and Path(detected).resolve() != Path(tuned_gguf).resolve():
            ref_gguf = detected
            auto_note = f"(auto-detected reference: {detected})\n"
    if not ref_gguf or not Path(ref_gguf).is_file():
        yield ("Reference GGUF not found - needed to build the baseline. Paste the F16/BF16 "
               "file step 6 converted from."), None
        return
    if not lcpp.is_quantize_built(LLAMACPP_DIR):
        yield "llama-quantize isn't built yet - see the Setup section above.", None
        return

    imatrix = newest_imatrix() or None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(ref_gguf).stem
    baseline_path = str(OUTPUT_DIR / f"{stem}-{baseline_type}.gguf")
    log = (
        f"Validation: {tuned_gguf}\n  vs baseline {baseline_type} of {ref_gguf}\n{auto_note}"
        f"  text: {text_file}   gpu layers: {ngl_n}\n\n"
    )
    if not Path(baseline_path).is_file():
        log += f"Baseline not built yet - quantizing {baseline_type}"
        log += f" (with imatrix: {imatrix})\n" if imatrix else " (no imatrix found)\n"
        yield log, None
        ok = False
        for line in lcpp.stream_quantize(
            LLAMACPP_DIR, ref_gguf, baseline_path, baseline_type, imatrix_file=imatrix,
        ):
            if line == "__OK__":
                ok = True
            elif line.startswith("__FAIL__"):
                yield log + "\n❌ Baseline quantization failed.\n", None
                return
            else:
                log += line
                yield log, tuned_gguf
        if not ok:
            yield log + "\n❌ Baseline quantization failed.\n", None
            return
    else:
        log += f"Using existing baseline: {baseline_path}\n"

    results: dict[str, float] = {}
    for label, path in (("baseline", baseline_path), ("tuned", tuned_gguf)):
        log += f"\nMeasuring perplexity: {label} ({Path(path).name})...\n"
        yield log, tuned_gguf
        chunk = ""
        for line in lcpp.stream_perplexity(LLAMACPP_DIR, path, text_file, ngl=ngl_n):
            if line == "__OK__":
                break
            if line.startswith("__FAIL__"):
                yield log + f"\n❌ Perplexity run failed for {label}.\n", tuned_gguf
                return
            chunk += line
            log += line
            yield log, tuned_gguf
        ppl = lcpp.parse_final_ppl(chunk)
        if ppl is None:
            yield log + f"\n❌ Couldn't parse a final PPL for {label}.\n", tuned_gguf
            return
        results[label] = ppl

    b, t = results["baseline"], results["tuned"]
    delta = (t - b) / b * 100
    verdict = "better" if t < b else "worse" if t > b else "identical"
    log += (
        f"\n{'=' * 56}\n"
        f"baseline ({baseline_type}): PPL {b:.4f}\n"
        f"tuned:                      PPL {t:.4f}\n"
        f"delta: {delta:+.2f}% ({verdict} - lower is better)\n"
        f"{'=' * 56}\n"
    )
    yield log, tuned_gguf


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
    fast_math: bool,
    loss_sync_batch: str,
    snapshot_interval: str,
    compile_loop: bool,
    python_exe: str,
    int4_layers_regex: str,
    int4_fallback_int8: bool,
    gguf_quant_type: str,
    save_progress: bool,
):
    input_path = (input_local or "").strip() if source == "Local file path" else (input_hf or "").strip()

    if fmt == "int4_convrot":
        yield from run_int4_convert(
            input_path, output_name, auto_output, preset_label_value,
            int4_layers_regex, int4_fallback_int8, exclude_layers, device,
            save_progress=save_progress,
        )
        return

    if fmt == "gguf":
        yield from run_gguf_convert(
            input_path, output_name, auto_output, preset_label_value, gguf_quant_type, exclude_layers,
            save_progress=save_progress,
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
            manual_seed=manual_seed, fast_math=fast_math, loss_sync_batch=loss_sync_batch,
            snapshot_interval=snapshot_interval, compile_loop=compile_loop,
        )
        args = build_args(opts)
    except OptionsError as exc:
        yield f"Can't build a valid command: {exc}", None, ""
        return

    if opts.uses_perf_flags and not runner.ctq_supports_perf_flags((python_exe or "").strip() or None):
        yield (
            "The installed ctq doesn't support the GPU speed flags — update it with:\n"
            "  pip install --upgrade git+https://github.com/8bit-boom/convert_to_quant@main\n"
            "(or uncheck the GPU speed options and try again.)",
            None,
            "",
        )
        return

    if opts.output_path:
        Path(opts.output_path).parent.mkdir(parents=True, exist_ok=True)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    yield "", None, _progress_bar_html(0, "Starting ctq...")
    # Matches both "(12/264) Processing (INT8): blocks.0.mlp.up.weight" and
    # "(2/6) Skipping tensor: blocks.0.firs.weight (Reason: krea2 skip)".
    tensor_progress_re = re.compile(r"\((\d+)/(\d+)\)\s*(Processing|Skipping)")

    # Native stop/resume: when the installed ctq supports checkpoints, a
    # stopped run saves per-tensor progress and resume continues from the
    # last finished tensor. Older ctq builds fall back to the legacy stop
    # (terminate + session snapshot that restarts from the beginning).
    native_cp_dir = None
    stop_file = None
    if save_progress and runner.ctq_supports_checkpoints((python_exe or "").strip() or None):
        native_cp_dir = CHECKPOINT_ROOT / f"ctq-native-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        stop_file = native_cp_dir / "stop.request"
        native_cp_dir.mkdir(parents=True, exist_ok=True)
        args = args + ["--checkpoint-dir", str(native_cp_dir), "--stop-file", str(stop_file)]

    control = RunControl()
    run_control.register(control)
    loop_timer = LoopPhaseTimer()
    t0 = time.time()
    try:
        log = ""
        result_path = None
        bar = _progress_bar_html(0, "Starting ctq...")
        for chunk in stream_conversion(
            args,
            python_executable=((python_exe or "").strip() or None),
            control=control,
            stop_file=str(stop_file) if stop_file else None,
        ):
            if chunk == "__CTQ_OK__":
                found = opts.output_path
                if not found:
                    m = re.search(r"Saved to[:\s]+(\S+\.safetensors)", log, re.IGNORECASE)
                    found = m.group(1) if m else None
                result_path = found if found and Path(found).is_file() else None
                if native_cp_dir and native_cp_dir.exists():
                    shutil.rmtree(native_cp_dir, ignore_errors=True)
                log += "\n" + loop_timer.finish() + "\n"
                try:
                    hist_key = str(Path(result_path or opts.output_path or input_path).resolve())
                except OSError:
                    hist_key = result_path or opts.output_path or input_path
                log += _record_run_history(
                    key=hist_key,
                    input_path=input_path,
                    output_path=result_path or opts.output_path or "",
                    command=format_command(opts),
                    loop_timer=loop_timer,
                    duration_s=time.time() - t0,
                )
                log += "\n✅ Conversion finished.\n"
                if result_path:
                    log += f"Output: {result_path}\n"
                elif found:
                    log += f"Output (reported, not found on disk yet): {found}\n"
                yield log, result_path, _progress_bar_html(1.0, "Done")
            elif chunk == "__CTQ_CANCELLED__":
                if save_progress:
                    ckpt.create_checkpoint(
                        CHECKPOINT_ROOT, "ctq",
                        params={
                            "input_path": input_path,
                            "args": args,
                            "command": format_command(opts),
                            "python_exe": (python_exe or "").strip(),
                        },
                        output_path=opts.output_path or "",
                    )
                    log += (
                        "\n⏹ Stopped. A session snapshot was saved - 'Resume from checkpoint' below "
                        "relaunches this exact conversion with one click.\n"
                        "(ctq itself keeps no partial state, so resuming it restarts from the beginning.)\n"
                    )
                else:
                    log += (
                        "\n⏹ Stopped by user. ctq keeps no partial state, so nothing is resumable - "
                        "re-run the conversion when ready.\n"
                    )
                yield log, None, bar
            elif chunk == "__CTQ_STOPPED__":
                # Checkpoint-aware ctq exited cleanly (code 3) with per-tensor
                # progress saved. Keep a session snapshot pointing at the
                # native checkpoint dir so resume continues mid-run.
                if save_progress:
                    done, total = 0, 0
                    manifest = (native_cp_dir / "manifest.json") if native_cp_dir else None
                    if manifest is not None and manifest.is_file():
                        try:
                            import json

                            m = json.loads(manifest.read_text(encoding="utf-8"))
                            done, total = len(m.get("entries", [])), int(m.get("total", 0))
                        except (ValueError, OSError):
                            pass
                    ckpt.create_checkpoint(
                        CHECKPOINT_ROOT, "ctq",
                        params={
                            "input_path": input_path,
                            "args": args,
                            "command": format_command(opts),
                            "python_exe": (python_exe or "").strip(),
                            "checkpoint_dir": str(native_cp_dir) if native_cp_dir else "",
                        },
                        output_path=opts.output_path or "",
                    )
                    if native_cp_dir:
                        where = f"tensor {done}/{total}" if total else f"{done} tensors in"
                        log += (
                            f"\n⏹ Stopped. Per-tensor progress saved ({where}) - "
                            "'Resume from checkpoint' below continues from the last finished tensor.\n"
                        )
                    else:
                        log += (
                            "\n⏹ Stopped. A session snapshot was saved - 'Resume from checkpoint' below "
                            "relaunches this exact conversion with one click.\n"
                        )
                else:
                    log += "\n⏹ Stopped by user.\n"
                yield log, None, bar
            elif chunk.startswith("__CTQ_FAIL__"):
                code = chunk.split(":", 1)[-1]
                log += f"\n❌ ctq exited with code {code}.\n"
                yield log, None, _progress_bar_html(1.0, "Failed")
            else:
                log += chunk
                for line in chunk.splitlines():
                    note = loop_timer.feed(line)
                    if note:
                        log += note + "\n"
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
    finally:
        run_control.register(None)


def refresh_env():
    return report_markdown(check_environment())


def _record_run_history(
    *, key: str, input_path: str, output_path: str, command: str,
    loop_timer: LoopPhaseTimer, resumed: bool = False, duration_s: float | None = None,
) -> str:
    """Persist this run's loop stats; return a '[loop]' comparison line.

    Compares against the most recent prior run of the same conversion so
    the GPU speed flags' effect is visible across conversions, not just
    inside one log. History is best-effort: any I/O error yields "".
    """
    try:
        loop = loop_timer.stats()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        output_bytes = None
        if output_path:
            try:
                output_bytes = Path(output_path).stat().st_size
            except OSError:
                output_bytes = None
        rh.record(
            RUN_HISTORY,
            {
                "timestamp": stamp,
                "key": key,
                "input": input_path,
                "output": output_path,
                "command": command,
                "resumed": resumed,
                "loop": loop,
                "duration_s": round(duration_s, 1) if duration_s is not None else None,
                "output_bytes": output_bytes,
            },
        )
        prev = rh.previous(rh.load(RUN_HISTORY), key, before=stamp)
        if prev:
            line = rh.comparison_line(prev, loop)
            if line:
                return line + "\n"
    except OSError:
        pass
    return ""


def _without_checkpoint_flags(args: list[str]) -> list[str]:
    """Strip --checkpoint-dir/--stop-file pairs (re-added fresh on resume)."""
    out: list[str] = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a in ("--checkpoint-dir", "--stop-file"):
            skip_next = True
            continue
        if a.startswith("--checkpoint-dir=") or a.startswith("--stop-file="):
            continue
        out.append(a)
    return out


def resume_ctq(cp):
    """Relaunch a stopped ctq conversion from its saved session snapshot.

    With a checkpoint-aware ctq build this is a true resume: the snapshot's
    ``checkpoint_dir`` holds per-tensor progress, so ctq continues from the
    last finished tensor. With older ctq builds it relaunches the exact
    argument list the snapshot saved (restart from the beginning, but no
    re-entering every setting by hand).
    """
    args = _without_checkpoint_flags(list(cp.params.get("args") or []))
    if not args:
        yield "That ctq snapshot has no stored command - can't resume.", None, ""
        return

    checkpoint_dir = (cp.params.get("checkpoint_dir") or "").strip()
    stop_file = None
    if checkpoint_dir and runner.ctq_supports_checkpoints(cp.params.get("python_exe") or None):
        stop_file = str(Path(checkpoint_dir) / "stop.request")
        # A stale stop file from the previous stop would kill the resume
        # instantly; the runner also cleans it up when ctq exits.
        try:
            if os.path.exists(stop_file):
                os.unlink(stop_file)
        except OSError:
            pass
        args = args + ["--checkpoint-dir", checkpoint_dir, "--stop-file", stop_file]
        resume_msg = "ctq resumes from its saved per-tensor checkpoint."
    else:
        checkpoint_dir = ""
        resume_msg = "ctq can't continue mid-conversion - this restarts the run with the exact saved settings."

    log = "Resuming ctq run from a session snapshot.\n"
    log += f"  command: {cp.params.get('command') or ' '.join(args)}\n"
    log += f"  ({resume_msg})\n\n"
    yield log, None, _progress_bar_html(0, "Resuming ctq...")
    # Matches the same "(12/264) Processing (INT8): ..." lines as run_convert.
    tensor_progress_re = re.compile(r"\((\d+)/(\d+)\)\s*(Processing|Skipping)")

    control = RunControl()
    run_control.register(control)
    loop_timer = LoopPhaseTimer()
    t0 = time.time()
    try:
        result_path = None
        bar = _progress_bar_html(0, "Starting ctq...")
        for chunk in stream_conversion(
            args,
            python_executable=(cp.params.get("python_exe") or None),
            control=control,
            stop_file=stop_file,
        ):
            if chunk == "__CTQ_OK__":
                found = cp.output_path
                if not found:
                    m = re.search(r"Saved to[:\s]+(\S+\.safetensors)", log, re.IGNORECASE)
                    found = m.group(1) if m else None
                result_path = found if found and Path(found).is_file() else None
                log += "\n" + loop_timer.finish() + "\n"
                try:
                    hist_key = str(Path(
                        result_path or cp.output_path or cp.params.get("input_path") or ""
                    ).resolve())
                except OSError:
                    hist_key = result_path or cp.output_path or str(cp.params.get("input_path") or "")
                log += _record_run_history(
                    key=hist_key,
                    input_path=str(cp.params.get("input_path") or ""),
                    output_path=result_path or cp.output_path or "",
                    command=str(cp.params.get("command") or " ".join(args)),
                    loop_timer=loop_timer,
                    resumed=True,
                    duration_s=time.time() - t0,
                )
                log += "\n✅ Conversion finished.\n"
                if result_path:
                    log += f"Output: {result_path}\n"
                ckpt.delete_checkpoint(CHECKPOINT_ROOT, cp.id)
                if checkpoint_dir:
                    shutil.rmtree(checkpoint_dir, ignore_errors=True)
                yield log, result_path, _progress_bar_html(1.0, "Done")
            elif chunk == "__CTQ_STOPPED__":
                log += "\n⏹ Stopped again - progress is kept; resume it whenever.\n"
                yield log, None, bar
            elif chunk == "__CTQ_CANCELLED__":
                log += "\n⏹ Stopped again - the session snapshot is kept; resume it whenever.\n"
                yield log, None, bar
            elif chunk.startswith("__CTQ_FAIL__"):
                code = chunk.split(":", 1)[-1]
                log += f"\n❌ ctq exited with code {code}.\n"
                yield log, None, _progress_bar_html(1.0, "Failed")
            else:
                log += chunk
                for line in chunk.splitlines():
                    note = loop_timer.feed(line)
                    if note:
                        log += note + "\n"
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
    finally:
        run_control.register(None)


def run_resume(checkpoint_id: str):
    checkpoint_id = (checkpoint_id or "").strip()
    if not checkpoint_id:
        yield "Pick a saved checkpoint from the list first.", None, ""
        return
    try:
        cp = ckpt.load_checkpoint(CHECKPOINT_ROOT, checkpoint_id)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        yield f"Couldn't read that checkpoint: {exc}", None, ""
        return

    if cp.finished:
        yield "That checkpoint is already finished - its output file was written. Nothing to resume.", None, ""
        return

    if cp.backend == "int4":
        p = cp.params
        yield from run_int4_convert(
            p.get("input_path", ""), p.get("output_name", ""), bool(p.get("auto_output", True)),
            p.get("preset", ""), p.get("int4_layers_regex", ""), bool(p.get("fallback_int8", True)),
            p.get("exclude_layers", ""), p.get("device", "cpu"), checkpoint=cp,
        )
        return
    if cp.backend == "gguf":
        p = cp.params
        yield from run_gguf_convert(
            p.get("input_path", ""), p.get("output_name", ""), bool(p.get("auto_output", True)),
            p.get("preset", ""), p.get("quant_type", "Q8_0"), p.get("exclude_layers", ""),
            checkpoint=cp,
        )
        return
    if cp.backend == "ctq":
        yield from resume_ctq(cp)
        return
    yield f"Unknown checkpoint type {cp.backend!r} - can't resume.", None, ""


def on_pause_click():
    ctl = run_control.current()
    if ctl is None:
        return gr.update(value="⏸ Pause / ▶ Resume"), "No conversion is running."
    if ctl.toggle_pause():
        return gr.update(value="▶ Resume"), "⏸ Paused (the current tensor finishes first). Click Resume to continue."
    return gr.update(value="⏸ Pause / ▶ Resume"), "▶ Running."


def on_stop_click():
    ctl = run_control.current()
    if ctl is None:
        return gr.update(value="⏸ Pause / ▶ Resume"), "No conversion is running."
    ctl.cancel()
    return (
        gr.update(value="⏸ Pause / ▶ Resume"),
        "⏹ Stopping after the current tensor - progress is being saved...",
    )


def checkpoint_choices() -> list[tuple[str, str]]:
    return [(ckpt.summary(cp), cp.id) for cp in ckpt.list_checkpoints(CHECKPOINT_ROOT)]


def refresh_checkpoint_dd():
    return gr.update(choices=checkpoint_choices(), value=None)


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
        "format your card can actually accelerate. There's also a separate **LLM → GGUF** tab for text models "
        "(Gemma, Llama, Qwen, etc.) — a different pipeline built around a real llama.cpp checkout."
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
                    batch_paths = gr.Textbox(
                        label="Batch queue (optional) — extra model paths, one per line",
                        placeholder="/path/to/model-a.safetensors\n/path/to/model-b.safetensors",
                        info="With the 'Convert all (batch)' button: every listed model is converted "
                        "with the settings above, one after another. Auto output naming is forced "
                        "so files don't overwrite each other.",
                        lines=3,
                    )

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
                        gguf_gpu_quant = gr.Checkbox(
                            value=_initial_gpu_quant,
                            label="Use GPU quantization (Q8_0)",
                            info="Runs the Q8_0 block quantizer on your NVIDIA GPU via Triton - ~10x faster "
                            "than the CPU path on large tensors, bit-identical output. Needs triton + CUDA; "
                            "silently falls back to CPU if either is missing. The choice is saved for next "
                            "launch.",
                        )

                        def _persist_gpu_quant(enabled: bool) -> None:
                            # Worker threads read the env var at quantize
                            # time, so flipping it here is enough - no need
                            # to thread the value through convert handlers.
                            if enabled:
                                os.environ[_gpu_quant.GPU_QUANT_ENV] = "1"
                            else:
                                os.environ.pop(_gpu_quant.GPU_QUANT_ENV, None)
                            ui_settings.set_setting(APP_DIR, "gguf_gpu_quant", bool(enabled))

                        gguf_gpu_quant.change(_persist_gpu_quant, inputs=[gguf_gpu_quant], outputs=[])

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
                        gr.Markdown(
                            "**GPU speed (opt-in, Learned mode only)** — measured on an RTX 5090. "
                            "Everything here is off by default; outputs stay byte-identical with default settings."
                        )
                        with gr.Row():
                            fast_math = gr.Checkbox(
                                value=False, label="Fast math (TF32 + bf16)",
                                info="~4.7x faster optimizer iterations on NVIDIA GPUs. Internal math at reduced "
                                "precision, but measured output error ratio 1.000 — visually identical quants.",
                            )
                            loss_sync_batch = gr.Dropdown(
                                ["1", "4", "8", "16"], value="1", label="Loss sync every K iters",
                                info="CUDA only: read the optimizer loss back once per K iterations instead of "
                                "every one. K=8 measured ~6x faster wall-clock. LR/early-stop decisions lag up to "
                                "K-1 iterations — tiny drift possible (~4e-4 relative at K=8).",
                            )
                        with gr.Row():
                            snapshot_interval = gr.Dropdown(
                                ["1", "4", "8", "16"], value="1", label="Snapshot every N improvements",
                                info="How often the best-so-far tensor is cloned during optimization (1 = every "
                                "improvement, the original behavior). Higher values cut per-iteration copy overhead.",
                            )
                            compile_loop = gr.Checkbox(
                                value=False, label="Compile optimizer loop (triton)",
                                info="JIT-compile the optimizer's forward pass with torch.compile: ~5% steady-state "
                                "gain but a 1-3 s compile warmup per tensor shape. Only worth enabling for large "
                                "models with many same-shaped layers. Requires triton installed.",
                            )
                        gpu_perf_notice = gr.Markdown(visible=False)
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
                    batch_convert_btn = gr.Button("Convert all (batch)")

                    save_progress = gr.Checkbox(
                        value=True,
                        label="Save resumable progress (Stop & resume later)",
                        info="Writes per-tensor checkpoints while converting (some extra disk I/O). "
                             "Pause/Resume works either way; without this, a stopped run can't be continued.",
                    )
                    with gr.Row():
                        pause_btn = gr.Button("⏸ Pause / ▶ Resume")
                        stop_btn = gr.Button("⏹ Stop & save progress")
                    run_status_md = gr.Markdown()

                    gr.Markdown(
                        "#### Resume a saved run\n"
                        "Conversions stopped mid-way (or interrupted) with saved progress show up here - "
                        "even after closing and reopening the app. Pick one and it continues from exactly "
                        "the tensor where it stopped."
                    )
                    checkpoint_dd = gr.Dropdown(label="Saved checkpoints", choices=[], interactive=True)
                    with gr.Row():
                        resume_btn = gr.Button("Resume from checkpoint", variant="primary")
                        delete_checkpoint_btn = gr.Button("Delete selected")

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
                "**Install ctq** (the 8bit-boom main build — it has the native stop/resume "
                "checkpoint support the Pause/Stop/Resume buttons use; plain PyPI "
                "`convert-to-quant` works too but resumes restart from scratch):\n"
                "```bash\n"
                "pip install git+https://github.com/8bit-boom/convert_to_quant@main\n"
                "# then install PyTorch separately for your GPU, e.g.\n"
                "pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
                "pip install -U triton   # optional, speeds up INT8 kernels\n```"
            )

        with gr.Tab("LLM → GGUF"):
            gr.Markdown(
                "Converts **text LLMs** (Gemma, Llama, Qwen, etc.) to GGUF - a completely separate pipeline "
                "from the rest of this app. Diffusion models above go through ctq/comfy_kitchen/gguf directly; "
                "LLMs need real tokenizer conversion and per-architecture hyperparameter mapping, which only "
                "[llama.cpp](https://github.com/ggerganov/llama.cpp) itself implements well - so this tab "
                "manages its own clone of llama.cpp (in its own Python environment, kept separate from this "
                "app's) and runs its real `convert_hf_to_gguf.py` / `llama-quantize`, the same tools you'd run "
                "by hand."
            )

            gr.Markdown("### 1. Setup (one-time)")
            llamacpp_status = gr.Markdown(llamacpp_status_markdown())
            with gr.Row():
                llamacpp_clone_btn = gr.Button("Clone / update llama.cpp")
                llamacpp_venv_btn = gr.Button("Install Python deps")
                llamacpp_build_btn = gr.Button("Get llama-quantize + llama-imatrix (build or download)")
            llamacpp_build_jobs = gr.Textbox(
                label="Build parallelism (optional)", placeholder="defaults to CPU core count",
            )
            llamacpp_refresh_btn = gr.Button("Re-check status")
            llamacpp_log = gr.Textbox(label="Setup log", lines=12, interactive=False, autoscroll=True)

            gr.Markdown(
                "### 2. Get a model\n"
                "Paste a Hugging Face repo ID (e.g. `google/gemma-3-4b-it`) to download the whole repo "
                "(config, tokenizer, safetensors - not just one file, unlike the Convert tab above), or point "
                "at a model folder already on disk."
            )
            llm_source = gr.Radio(["Hugging Face repo", "Local directory"], value="Hugging Face repo", label="Source")
            with gr.Row():
                llm_repo_id = gr.Textbox(label="Hugging Face repo ID", placeholder="google/gemma-3-4b-it")
                llm_hf_token = gr.Textbox(label="HF access token (gated repos only)", type="password")
            llm_local_dir = gr.Textbox(
                label="Local model directory", placeholder="/path/to/model/folder", visible=False,
            )
            llm_download_btn = gr.Button("Download model")
            llm_model_dir = gr.Textbox(label="Resolved model directory", interactive=False)
            llm_model_dd = gr.Dropdown(
                choices=llm_downloaded_models(), value=None, interactive=True,
                label="Downloaded models",
                info="Everything downloaded through this tab or found in the Hugging Face cache - "
                "pick one to fill the model directory above.",
            )

            gr.Markdown("### 3. Convert to GGUF")
            with gr.Row():
                llm_outtype = gr.Dropdown(
                    lcpp.DIRECT_OUTTYPE_CHOICES, value="auto", label="Output type",
                    info="auto keeps the model's own dtype; q8_0 quantizes directly (pure Python, real). For "
                    "K-quants (Q4_K_M etc.), convert to f16/bf16 here first, then quantize below.",
                )
                llm_convert_output_name = gr.Textbox(label="Output filename (optional)", placeholder="auto")
            llm_convert_btn = gr.Button("Convert to GGUF", variant="primary")

            gr.Markdown(
                "### 4. Generate an importance matrix (optional, for calibrated/\"dynamic\"-style quants)\n"
                "Needs **llama-imatrix built** (step 1). Runs calibration text through the full-precision "
                "model and records which weights actually matter, so quantizing below can round more "
                "carefully on the layers that need it - the same real mechanism behind most \"imatrix\" GGUF "
                "quants on Hugging Face, and the foundation Unsloth's own \"Dynamic\" quants are built on "
                "(per their own docs). Uses a small bundled generic calibration text by default; paste your "
                "own file below for better results on a specific domain."
            )
            llm_calib_mode = gr.Radio(
                ["Auto (bundled generic calibration)", "Custom calibration file"],
                value="Auto (bundled generic calibration)", label="Calibration",
                info="Auto runs llama-imatrix on a small bundled generic text - the same default the "
                "find-best sweep uses. Custom is better for a specific domain.",
            )
            with gr.Row():
                llm_imatrix_model = gr.Textbox(
                    label="Model GGUF (F16/BF16)",
                    placeholder="auto-filled from step 3; if empty, the newest GGUF in the output folder is used",
                )
                llm_imatrix_calibration = gr.Textbox(
                    label="Custom calibration text file", placeholder="/path/to/calibration.txt", visible=False,
                )
            llm_imatrix_output_name = gr.Textbox(label="Output filename (optional)", placeholder="auto")
            llm_imatrix_btn = gr.Button("Generate importance matrix")

            gr.Markdown(
                "### 5. Quantize to a K-quant (optional)\n"
                "Needs **llama-quantize built** (step 1). Takes any GGUF file (typically this tab's own F16/"
                "BF16 output above) and produces a real, smaller K-quant."
            )
            with gr.Row():
                llm_quantize_input = gr.Textbox(
                    label="Input GGUF",
                    placeholder="auto-filled from step 3; if empty, the newest GGUF in the output folder is used",
                )
                llm_quant_type = gr.Dropdown(lcpp.QUANT_TYPE_CHOICES, value="Q4_K_M", label="Quant type")
            with gr.Row():
                llm_quantize_imatrix = gr.Textbox(
                    label="Importance matrix (optional)",
                    placeholder="auto-filled from step 4, or paste any imatrix.gguf path",
                )
                llm_quantize_tensor_types = gr.Textbox(
                    label="Per-layer type overrides (optional, advanced)",
                    placeholder="path to a tensor-type-file, e.g. lines like 'blk.0.attn_k.weight=Q8_0'",
                    info="The real mechanism behind manual \"dynamic\" per-layer mixing. Step 6 below "
                    "generates one automatically from imatrix-weighted sensitivity; paste any such file "
                    "here to quantize with it directly.",
                )
            llm_quantize_output_name = gr.Textbox(label="Output filename (optional)", placeholder="auto")
            llm_target_bpw = gr.Radio(
                list(gb.BPP_FAMILY_CANDIDATES.keys()), value=gb.DEFAULT_BPP_TARGET,
                label="Target size",
                info="Weight-space bits-per-weight class to sweep; model-level bpw runs higher "
                "when a big vocabulary keeps embeddings heavy.",
            )
            with gr.Row():
                llm_quantize_btn = gr.Button("Quantize")
                llm_find_best_btn = gr.Button("Find best quant (sweep)")
                llm_auto_btn = gr.Button("Auto (sweep → tune → validate)", variant="primary")
            gr.Markdown(
                "**Auto** chains the whole tuning pipeline in one click: sweep the target-size family, "
                "per-tensor-tune the winner (Dynamic 3.0 style), then validate the tuned file against a "
                "plain baseline quant with perplexity on a held-out text. Stages reuse each other's "
                "artifacts (the sweep's imatrix, the tune's output file), so re-running skips work "
                "that's already done. Needs llama-quantize + llama-imatrix built (Setup section)."
            )
            gr.Markdown(
                "**Find best quant** sweeps the target-size family through llama-quantize on the input "
                "file — generating an imatrix first if none is given — and measures size, time, and "
                "reconstruction error vs the input. The winner is pre-selected in the Quant type "
                "dropdown; intermediate outputs are discarded. Big-vocab models (Gemma 4, Qwen 3.5-3.8) "
                "automatically also get Q8_0 token-embedding variants. Weight-space error, not "
                "perplexity — it ranks settings, it doesn't judge generation quality."
            )

            gr.Markdown(
                "### 6. Smart per-tensor tuning (Dynamic-quant style)\n"
                "Three stages: **(1)** rank every tensor's quantization sensitivity using your imatrix "
                "as the activation-importance weight (minutes, pure Python — no GPU needed); "
                "**(2)** greedily assign per-tensor types under a size budget — the K-ladder mode maps "
                "sensitivity onto q3_k/q4_k/q5_k/q6_k, the legacy mode uses exactly-measured q4_0/q5_0/"
                "q8_0 errors; **(3)** write a llama-quantize `--tensor-type-file` and quantize with it. "
                "Embeddings, the output tensor and MoE routers are never dropped below the floor."
            )
            with gr.Row():
                llm_smart_model = gr.Textbox(
                    label="Model GGUF (F16/BF16 reference)",
                    placeholder="auto-detects the newest model GGUF in the output folder",
                )
                llm_smart_imatrix = gr.Textbox(
                    label="Importance matrix",
                    placeholder="auto-detects the newest imatrix in the output folder",
                )
            with gr.Row():
                llm_smart_mode = gr.Radio(
                    ["K-ladder (rank-mapped)", "Legacy (exact-scored)"], value="K-ladder (rank-mapped)",
                    label="Assignment mode",
                    info="K-ladder produces llama.cpp-friendly q3_k..q6_k mixes sized to your budget; "
                    "legacy uses only pure-Python-measurable types with exact errors.",
                )
                llm_smart_target = gr.Radio(
                    list(gb.BPP_FAMILY_CANDIDATES.keys()) + ["Custom"],
                    value="~4 bpw",
                    label="Target size (Dynamic 3.0 preset)",
                    info="One-click preset: the budget is computed from the model's parameter count at "
                    "the family's file bits-per-weight. MoE models get the two-zone recipe - Q8_0-level "
                    "attention/shared FFN, IQ-quantized experts (beats Unsloth UD-IQ4_XS at equal size "
                    "on gemma-4-26B). 'Custom' uses the GB box.",
                )
            with gr.Row():
                llm_smart_budget = gr.Textbox(
                    label="Size budget (GB, used with Custom)", value="13.5",
                    info="Target total file size. The tuner downgrades insensitive tensors and "
                    "upgrades sensitive ones until the estimate fits.",
                )
                llm_smart_output = gr.Textbox(label="Output filename (optional)", placeholder="auto")
            with gr.Row():
                llm_smart_btn = gr.Button("Tune + quantize (Dynamic 3.0)", variant="primary")
                llm_tune_winner_btn = gr.Button("Tune winner (Dynamic 3.0)")
            gr.Markdown(
                "**Tune winner** takes the last 'Find best quant (sweep)' result and re-runs it "
                "through the per-tensor tuner: same model and imatrix, budget set to the winner's "
                "bits-per-weight family, K-ladder assignment. On MoE models this upgrades the "
                "sweep's single uniform type to a two-zone mix (Q8_0-level attention/shared FFN, "
                "IQ-quantized experts) at the same file size."
            )
            gr.Markdown(
                "#### Stage 3 validation — perplexity comparison\n"
                "Measures perplexity of the tuned file and a plain baseline quant of the same "
                "reference on a **held-out text** (something the calibration didn't see), with "
                "llama-perplexity. Lower PPL wins; a small regression vs Q4_K_M is normal when "
                "the budget forced aggressive downgrades - the point is to see exactly how much "
                "you paid for the size."
            )
            with gr.Row():
                llm_val_tuned = gr.Textbox(
                    label="Tuned GGUF",
                    placeholder="auto-detects the newest *smart*.gguf in the output folder",
                )
                llm_val_ref = gr.Textbox(
                    label="Reference GGUF (for the baseline)",
                    placeholder="auto-detects the newest model GGUF (F16/BF16)",
                )
            with gr.Row():
                llm_val_baseline = gr.Dropdown(lcpp.QUANT_TYPE_CHOICES, value="Q4_K_M", label="Baseline quant")
                llm_val_ngl = gr.Textbox(label="GPU layers", value="99",
                                         info="99 = full offload on your RTX 5090; 0 = CPU")
                llm_val_text = gr.Textbox(
                    label="Held-out text file",
                    placeholder="path to a .txt the calibration didn't use",
                )
            llm_val_btn = gr.Button("Compare perplexity")

            with gr.Row():
                with gr.Column():
                    llm_log = gr.Textbox(
                        label="Conversion log", lines=20, interactive=False, autoscroll=True, elem_id="llm-log-box",
                    )
                with gr.Column():
                    llm_result_file = gr.File(label="Output file", interactive=False)

        with gr.Tab("Image → GGUF"):
            gr.Markdown(
                "Converts **Krea 2 (Raw / Turbo)** - the 12.9B text-to-image diffusion "
                "transformer - into GGUF for ComfyUI. This is *not* an LLM conversion: "
                "the GGUF holds only the DiT, with `arch = krea2` and ComfyUI-native "
                "tensor names. The text encoder and VAE stay separate safetensors "
                "(download buttons below). See [docs/KREA2.md](https://github.com/8bit-boom/quant-convert-gui/blob/main/docs/KREA2.md) for details."
            )
            with gr.Row():
                img_src = gr.File(label="Source safetensors (Comfy-Org BF16 single file, or HF diffusers shards merged)",
                                  file_types=[".safetensors"], type="filepath")
            with gr.Row():
                img_quant = gr.Dropdown(
                    choices=["q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "bf16"],
                    value="q8_0", label="Quant for big BF16 linears",
                )
                img_dst_name = gr.Textbox(label="Output name", value="krea2_turbo_q8_0.gguf")
                img_convert_btn = gr.Button("Convert", variant="primary")
            img_convert_log = gr.Textbox(label="Conversion log", lines=12, interactive=False, autoscroll=True)

            gr.Markdown("### Downloads (Comfy-Org/Krea-2 on Hugging Face)")
            with gr.Row():
                img_dl_turbo_btn = gr.Button("Turbo BF16 source (26 GB)")
                img_dl_raw_btn = gr.Button("Raw BF16 source (26 GB)")
                img_dl_te_btn = gr.Button("Text encoder (Qwen3-VL fp8)")
                img_dl_vae_btn = gr.Button("VAE (Qwen-Image)")
            img_dl_log = gr.Textbox(label="Download log", lines=4, interactive=False)
            gr.Markdown(
                "Files land in `image_models/` next to the app. Note the two 26 GB "
                "sources - you only need **one** (Turbo = 8-step generation; Raw = higher quality, more steps)."
            )

            gr.Markdown(
                "### ComfyUI recipe\n"
                "Requires ComfyUI ≥ v0.25 plus a krea2-patched ComfyUI-GGUF fork "
                "(RealRebelAI/ComfyUI-GGUF_KREA-2 or molbal's fork; stock city96 rejects the arch).\n"
                "1. Copy the GGUF into `models/diffusion_models/`\n"
                "2. **Unet Loader (GGUF)** ← the converted GGUF\n"
                "3. **CLIPLoader** ← `qwen3vl_4b_fp8_scaled.safetensors`, type `krea2`\n"
                "4. **VAELoader** ← `qwen_image_vae.safetensors`\n"
                "5. Turbo: 8 steps, CFG 1.0, euler/simple, shift 1.15 · Raw: 20–52 steps, CFG 3–7"
            )

        with gr.Tab("History"):
            gr.Markdown(
                "Every finished conversion (from `run_history.json`), newest first, with the metrics "
                "that make runs comparable: output size, wall time, optimizer-loop time, tensor count, "
                "and a speed verdict against the previous run of the same conversion."
            )
            history_tbl = gr.Dataframe(
                headers=rh.HEADERS, value=rh.rows(rh.load(RUN_HISTORY)),
                interactive=False, wrap=True,
            )
            history_refresh_btn = gr.Button("Refresh")

        with gr.Tab("Inspector"):
            gr.Markdown(
                "Read-only look inside any GGUF: architecture, type histogram, "
                "bits-per-weight, biggest tensors, metadata. Nothing is loaded or "
                "executed - inspection takes seconds even for 100 GB files."
            )
            with gr.Row():
                inspect_file = gr.File(
                    label="GGUF file", file_types=[".gguf"], type="filepath",
                )
                inspect_btn = gr.Button("Inspect", variant="primary")
            inspect_out = gr.Markdown("Pick a GGUF and press **Inspect**.")

        with gr.Tab("Editor"):
            gr.Markdown(
                "Edit a GGUF's **metadata** and save a new file - tensor data is copied "
                "byte-for-byte, never re-quantized, and streams through disk in constant "
                "memory (a 26 GB model is fine). Local, offline equivalent of the "
                "Hugging Face *GGUF Editor* space: fix a broken `general.name`, update "
                "`general.quantized_by`, patch a chat template, delete stray keys."
            )
            with gr.Row():
                edit_file = gr.File(
                    label="GGUF file", file_types=[".gguf"], type="filepath",
                )
                edit_load_btn = gr.Button("Load", variant="primary")
            edit_summary = gr.Markdown("Pick a GGUF and press **Load**.")
            edit_meta_tbl = gr.Dataframe(
                headers=["Key", "Type", "Value", "Editable"],
                interactive=True, wrap=True,
                label="Metadata - edit the Value column, then Save",
            )
            with gr.Row():
                edit_del_keys = gr.Dropdown(
                    multiselect=True, choices=[], label="Delete keys",
                    info="Selected keys are omitted from the output file.",
                )
                with gr.Column():
                    edit_new_key = gr.Textbox(
                        label="New key (optional)",
                        placeholder="e.g. general.quantized_by",
                    )
                    edit_new_type = gr.Dropdown(
                        choices=list(ge.SCALAR_TYPES), value="STRING",
                        label="New key type",
                        info="Arrays can be edited on existing keys only.",
                    )
                    edit_new_val = gr.Textbox(label="New key value")
            edit_out_name = gr.Textbox(
                label="Output filename",
                placeholder="model-edited.gguf (lands in converted/)",
            )
            edit_save_btn = gr.Button("Save edited GGUF", variant="primary")
            edit_log = gr.Textbox(
                label="Log", lines=10, interactive=False, autoscroll=True,
            )
            edit_result = gr.File(label="Output file", interactive=False)
            edit_state = gr.State()

        with gr.Tab("Tools"):
            gr.Markdown(
                "Wrappers around the repo's CLI tools (`tools/`) - same code as "
                "the command line, with logs streamed here."
            )
            with gr.Accordion("Compare GGUFs (inference-free quality report)", open=True):
                with gr.Row():
                    cmp_ref = gr.File(label="Reference GGUF (F16/BF16/F32)",
                                      file_types=[".gguf"], type="filepath")
                    cmp_cands = gr.File(label="Candidate GGUF(s) (pick several with Ctrl/Shift)",
                                        file_types=[".gguf"], type="filepath", file_count="multiple")
                with gr.Row():
                    cmp_imatrix = gr.File(label="Imatrix (optional, for activation weighting)",
                                          file_types=[".imatrix", ".gguf"], type="filepath")
                    cmp_rows = gr.Number(label="Sample rows per tensor", value=48, precision=0)
                    cmp_btn = gr.Button("Compare", variant="primary")
                cmp_log = gr.Textbox(label="Report", lines=18, interactive=False, autoscroll=True)
            with gr.Accordion("Ollama Modelfile", open=False):
                with gr.Row():
                    oll_gguf = gr.File(label="GGUF", file_types=[".gguf"], type="filepath")
                    oll_name = gr.Textbox(label="Model name (optional)", placeholder="my-model")
                    oll_ctx = gr.Number(label="num_ctx", value=32768, precision=0)
                    oll_btn = gr.Button("Generate Modelfile", variant="primary")
                oll_log = gr.Textbox(label="Result", lines=6, interactive=False)
            with gr.Accordion("Krea 2 (image DiT) → GGUF", open=False):
                gr.Markdown("Moved to its own **Image → GGUF** tab (with companion-file downloads and the ComfyUI recipe).")

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
                "quantization computed in pure Python (with an optional Triton GPU path for Q8_0 - the "
                "\"Use GPU quantization\" checkbox, bit-identical output). K-quants (Q4_K_M, Q6_K, etc.) "
                "aren't available *here* — that family only has a *decoder* in the Python package; producing "
                "them needs a compiled llama.cpp toolchain, which is exactly what the LLM → GGUF tab sets up "
                "for text models. Architecture (flux/sdxl/wan/etc.) is "
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
                "## What about LLMs (Gemma, Llama, Qwen, etc.)?\n"
                "That's the separate **LLM → GGUF** tab, not this one — everything above is for diffusion "
                "models. Text LLMs need real tokenizer conversion and per-architecture hyperparameter mapping "
                "that only [llama.cpp](https://github.com/ggerganov/llama.cpp) itself implements well, so that "
                "tab manages its own llama.cpp checkout (in its own Python environment) and runs its real "
                "`convert_hf_to_gguf.py` / `llama-quantize` directly, instead of reimplementing any of it here. "
                "Real K-quants (Q4_K_M, Q6_K, etc.) need `llama-quantize` compiled from source - that's an "
                "optional, separate build step in that tab since it needs a C/C++ toolchain, not just a pip "
                "install. That tab also has a real **imatrix / \"dynamic\"-style calibrated quant** pipeline "
                "(`llama-imatrix` + `llama-quantize --imatrix`) - the same mechanism behind most \"imatrix\" "
                "GGUF quants on Hugging Face and the one Unsloth's own Dynamic quants are built on, per their "
                "docs. A small bundled calibration text makes it work out of the box. On top of that: a "
                "**find-best sweep** ranks the target-size family by reconstruction error, and **smart "
                "per-tensor tuning** generates a `--tensor-type-file` automatically - it scores every "
                "tensor's sensitivity from your imatrix and assigns per-layer types under a size budget "
                "(Dynamic 3.0 style, driven by *your* calibration data rather than Unsloth's unpublished "
                "per-model picks). **Auto** chains sweep → tune → perplexity validation in one click.\n\n"
                "## What about image models (Krea 2)?\n"
                "The **Image → GGUF** tab converts Krea 2 (Raw / Turbo), the 12.9B text-to-image DiT, into "
                "the GGUF format the ComfyUI-GGUF community uses for image models - `arch = krea2`, "
                "ComfyUI-native tensor names, text encoder and VAE kept as separate safetensors (download "
                "buttons on the tab). It's deliberately separate from both the Convert tab above and the "
                "LLM tab: krea2 isn't in ComfyUI-GGUF's architecture allowlist, so it needs a "
                "community-patched loader fork anyway.\n\n"
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
                "All the ctq-path quantization math lives in "
                "[silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant) (`ctq`), "
                "and the GGUF paths call llama.cpp's own Python bindings and tools directly — this app is a "
                "GUI wrapper around those real implementations, not a reimplementation of them. It downloads "
                "nothing on its own besides the model files you point it at.\n\n"
                "## The other tabs\n"
                "- **Editor** — edit a GGUF's metadata and save a new file (tensor data copied "
                "byte-for-byte, streaming in constant memory). The local equivalent of the Hugging Face "
                "*GGUF Editor* space.\n"
                "- **Inspector** — read-only look inside any GGUF (architecture, type histogram, "
                "bits-per-weight, biggest tensors) without loading or executing it.\n"
                "- **Tools** — the repo's CLI utilities with logs streamed here: GGUF comparison "
                "(inference-free quality report), Ollama Modelfile generation (thinking-mode recipe "
                "included).\n"
                "- **History** — every finished conversion with size/time metrics and a speed verdict "
                "against the previous run of the same conversion."
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
    demo.load(refresh_checkpoint_dd, outputs=[checkpoint_dd])
    demo.load(lambda: gr.update(choices=llm_downloaded_models()), outputs=[llm_model_dd])

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
        fast_math, loss_sync_batch, snapshot_interval, compile_loop,
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
            "fast_math", "loss_sync_batch", "snapshot_interval", "compile_loop",
        ])
    }
    assert len(preview_field_index) == len(preview_inputs) - 3

    for comp in preview_inputs:
        comp.change(refresh_preview, inputs=preview_inputs, outputs=[command_preview])

    fmt_value.change(refresh_preview, inputs=preview_inputs, outputs=[command_preview])

    def refresh_gpu_perf_notice(fast_math_v, loss_sync_batch_v, snapshot_interval_v, compile_loop_v, python_exe_v):
        flags_on = (
            bool(fast_math_v)
            or bool(compile_loop_v)
            or str(loss_sync_batch_v or "1") != "1"
            or str(snapshot_interval_v or "1") != "1"
        )
        if not flags_on:
            return gr.update(visible=False)
        if runner.ctq_supports_perf_flags((python_exe_v or "").strip() or None):
            return gr.update(visible=False)
        return gr.update(
            visible=True,
            value="⚠️ The ctq this app would launch doesn't support the GPU speed flags. Update it with: "
            "`pip install --upgrade git+https://github.com/8bit-boom/convert_to_quant@main`",
        )

    for comp in (fast_math, loss_sync_batch, snapshot_interval, compile_loop, python_exe):
        comp.change(
            refresh_gpu_perf_notice,
            inputs=[fast_math, loss_sync_batch, snapshot_interval, compile_loop, python_exe],
            outputs=[gpu_perf_notice],
        )

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
            calib_samples, optimizer, num_iter, manual_seed, fast_math, loss_sync_batch,
            snapshot_interval, compile_loop, python_exe,
            int4_layers_regex, int4_fallback_int8, gguf_quant_type, save_progress,
        ],
        outputs=[log_box, result_file, convert_progress],
        # We render our own bar into convert_progress; gr.Progress()'s built-in
        # overlay otherwise blankets *every* output of this event (log_box and
        # result_file included) for the whole run and can get stuck once streaming ends.
        show_progress="hidden",
    )

    def run_convert_batch(batch_paths, input_local, input_hf, source, *rest):
        """Convert several models with the same settings, one after another.

        Every path comes from the batch box (plus the primary local input, if
        set); each item runs the full run_convert flow. Auto output naming is
        forced so items can't overwrite each other. Stop/Pause from the UI
        applies to the current item; a Stop ends the whole queue.
        """
        paths = [p.strip() for p in (batch_paths or "").splitlines() if p.strip()]
        primary = (input_local or "").strip()
        if source == "Local file path" and primary and primary not in paths:
            paths.insert(0, primary)
        if not paths:
            yield "Add at least one path to the batch box (or pick a primary local file) first.", None, ""
            return
        # rest[0] is output_name, rest[1] is auto_output - force auto naming.
        rest = (rest[0], True) + rest[2:]
        total_log = (
            f"# Batch queue: {len(paths)} model(s)\n"
            "Auto output naming is forced in batch mode.\n\n"
        )
        done = failed = 0
        for i, path in enumerate(paths, 1):
            ctl = run_control.current()
            if ctl is not None and ctl.is_cancelled:
                total_log += "\n**Batch stopped by user.**\n"
                yield total_log, None, ""
                return
            total_log += f"\n## [{i}/{len(paths)}] {path}\n\n"
            if not Path(path).is_file():
                total_log += "❌ not found on disk - skipped.\n"
                failed += 1
                yield total_log, None, ""
                continue
            last = ("", None, "")
            for last in run_convert(path, input_hf, "Local file path", *rest):
                yield total_log + (last[0] or ""), last[1], last[2]
            total_log += (last[0] or "") + "\n"
            if last[1]:
                done += 1
            else:
                failed += 1
        total_log += f"\n---\n**Batch finished: {done} ok, {failed} failed, {len(paths)} total.**\n"
        yield total_log, None, ""

    batch_convert_btn.click(
        run_convert_batch,
        inputs=[
            batch_paths,
            input_local, input_hf_local, source, output_name, auto_output, fmt_value, quality_mode,
            convrot_group_size, dynamic_convrot, scaling_mode, block_size, preset_dd, comfy_quant,
            save_metadata, low_memory, exclude_layers, custom_layers, custom_type, custom_scaling_mode,
            custom_convrot, custom_convrot_group_size, custom_simple, fallback, fallback_simple,
            device, output_dtype, verbose,
            calib_samples, optimizer, num_iter, manual_seed, fast_math, loss_sync_batch,
            snapshot_interval, compile_loop, python_exe,
            int4_layers_regex, int4_fallback_int8, gguf_quant_type, save_progress,
        ],
        outputs=[log_box, result_file, convert_progress],
        show_progress="hidden",
    )

    pause_btn.click(on_pause_click, outputs=[pause_btn, run_status_md])
    stop_btn.click(on_stop_click, outputs=[pause_btn, run_status_md])

    def on_delete_checkpoint(checkpoint_id):
        if (checkpoint_id or "").strip():
            ckpt.delete_checkpoint(CHECKPOINT_ROOT, checkpoint_id.strip())
        return refresh_checkpoint_dd(), "Deleted."

    resume_btn.click(
        run_resume, inputs=[checkpoint_dd], outputs=[log_box, result_file, convert_progress],
    ).then(refresh_checkpoint_dd, outputs=[checkpoint_dd])
    delete_checkpoint_btn.click(
        on_delete_checkpoint, inputs=[checkpoint_dd], outputs=[checkpoint_dd, run_status_md]
    )

    refresh_btn.click(refresh_env, outputs=[env_md])

    def run_inspect(path):
        if not path:
            yield "Pick a GGUF file first."
            return
        try:
            yield format_inspection(inspect_gguf(path))
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            yield f"**Inspection failed:** `{exc}`"

    inspect_btn.click(run_inspect, inputs=[inspect_file], outputs=[inspect_out])

    def run_editor_load(path):
        empty_dd = gr.update(choices=[], value=[])
        if not path:
            yield "Pick a GGUF file first.", [], empty_dd, None
            return
        try:
            plan = ge.load_for_edit(path)
        except ge.GGUFEditError as exc:
            yield f"**Load failed:** `{exc}`", [], empty_dd, None
            return
        size_gb = (plan.total_bytes / 1e9) if plan.total_bytes else 0
        summary = (
            f"**{Path(path).name}** — arch `{plan.architecture}`, "
            f"{plan.n_kv} metadata keys, {plan.n_tensors} tensors "
            f"({size_gb:.1f} GB of tensor data will be copied verbatim)."
        )
        # Update the dropdown's CHOICES with an empty selection - passing the
        # key list bare would set its VALUE instead, which Gradio rejects
        # ("not in the list of choices") and would pre-select every key for
        # deletion.
        del_dd = gr.update(choices=[kv.key for kv in plan.metadata], value=[])
        yield summary, plan.rows(), del_dd, plan

    edit_load_btn.click(
        run_editor_load,
        inputs=[edit_file],
        outputs=[edit_summary, edit_meta_tbl, edit_del_keys, edit_state],
    )

    def run_editor_save(plan, rows, del_keys, new_key, new_type, new_val, out_name):
        if plan is None:
            yield "Load a GGUF first.", None
            return
        try:
            rows = rows.values.tolist() if hasattr(rows, "values") else (rows or [])
        except AttributeError:
            rows = rows or []

        original = {kv.key: kv for kv in plan.metadata}
        set_meta: dict[str, ge.EditableKV] = {}
        notes = []
        errors = []
        for row in rows:
            if not row or len(row) < 3:
                continue
            key, vtype_txt, value_txt = str(row[0]), str(row[1]), "" if row[2] is None else str(row[2])
            kv = original.get(key)
            if kv is None:
                notes.append(f"ignored unknown row {key!r}")
                continue
            declared = kv.vtype if kv.sub_type is None else f"ARRAY[{kv.sub_type}]"
            if str(vtype_txt).strip() != declared:
                notes.append(f"{key}: type column changed, ignored - the original {declared} is kept")
                continue
            if not kv.editable:
                if value_txt != kv.display_value():
                    notes.append(f"{key}: read-only (over {ge.ARRAY_EDIT_LIMIT} items), edits ignored")
                continue
            if kv.vtype == "ARRAY":
                import json as _json
                try:
                    items = _json.loads(value_txt)
                    new_value = ge.parse_array(kv.sub_type, items)
                except (ValueError, ge.GGUFEditError) as exc:
                    errors.append(f"{key}: {exc}")
                    continue
                if new_value != list(kv.value):
                    set_meta[key] = ge.EditableKV(key, "ARRAY", kv.sub_type, new_value, True)
            else:
                try:
                    new_value = ge.parse_scalar(kv.vtype, value_txt)
                except ge.GGUFEditError as exc:
                    errors.append(str(exc))
                    continue
                if new_value != kv.value:
                    set_meta[key] = ge.EditableKV(key, kv.vtype, None, new_value, True)

        new_key = (new_key or "").strip()
        if new_key:
            if new_key in original or new_key in set_meta:
                errors.append(f"{new_key}: key already exists - edit its row instead")
            elif new_key in set(del_keys or ()):
                errors.append(f"{new_key}: also marked for deletion - pick one")
            else:
                try:
                    value = ge.parse_scalar(new_type, new_val or "")
                    set_meta[new_key] = ge.EditableKV(new_key, new_type, None, value, True)
                except ge.GGUFEditError as exc:
                    errors.append(f"{new_key}: {exc}")

        if errors:
            log = "Fix these before saving:\n  - " + "\n  - ".join(errors)
            if notes:
                log += "\n\nNotes:\n  - " + "\n  - ".join(notes)
            yield log, None
            return

        name = (out_name or "").strip()
        if not name:
            stem = Path(plan.source).stem
            name = f"{stem}-edited.gguf"
        out_path = name if os.path.isabs(name) else str(OUTPUT_DIR / name)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        log = f"Saving edited GGUF:\n  source: {plan.source}\n  output: {out_path}\n"
        if set_meta:
            log += f"  set/added: {', '.join(sorted(set_meta))}\n"
        if del_keys:
            log += f"  deleted:   {', '.join(sorted(del_keys))}\n"
        log += f"  copying {plan.n_tensors} tensors ({plan.total_bytes / 1e9:.1f} GB) byte-for-byte...\n"
        yield log, None

        progress = {"done": 0, "total": plan.total_bytes or 1}
        done_flags = {"ok": False, "err": None}

        def _work():
            try:
                ge.save_edited(
                    plan.source, out_path,
                    set_meta=set_meta, del_keys=list(del_keys or ()),
                    progress_cb=lambda d, t: progress.update(done=d, total=t),
                )
                done_flags["ok"] = True
            except Exception as exc:  # noqa: BLE001 - surfaced below
                done_flags["err"] = exc

        import threading as _threading
        worker = _threading.Thread(target=_work, daemon=True)
        worker.start()
        last_shown = -1
        while worker.is_alive():
            worker.join(timeout=0.25)
            pct = int(100 * progress["done"] / progress["total"])
            if pct != last_shown:
                last_shown = pct
                yield log + f"  {pct}% copied...", None
        if done_flags["err"] is not None:
            yield log + f"\n❌ Save failed: {done_flags['err']}", None
            return
        if notes:
            log += "\nNotes:\n  - " + "\n  - ".join(notes) + "\n"
        log += f"\n✅ Saved. {out_path}\n"
        yield log, out_path

    edit_save_btn.click(
        run_editor_save,
        inputs=[edit_state, edit_meta_tbl, edit_del_keys,
                edit_new_key, edit_new_type, edit_new_val, edit_out_name],
        outputs=[edit_log, edit_result],
    )
    history_refresh_btn.click(
        lambda: rh.rows(rh.load(RUN_HISTORY)), outputs=[history_tbl],
    )

    def _stream_tool_cmd(cmd: list[str]):
        """Run a tools/ CLI in a subprocess, streaming merged output into a
        Gradio textbox. Yields the accumulated log on every line."""
        import subprocess
        import sys
        proc = subprocess.Popen(
            [sys.executable, *cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(APP_DIR),
        )
        lines: list[str] = []
        for line in proc.stdout:
            lines.append(line)
            yield "".join(lines)
        proc.wait()
        lines.append(f"\n[exit code {proc.returncode}]")
        yield "".join(lines)

    def run_compare(ref, cands, imatrix, rows):
        if not ref or not cands:
            yield "Pick a reference GGUF and at least one candidate."
            return
        cmd = [str(APP_DIR / "tools" / "compare_gguf.py"), str(ref), *map(str, cands),
               "--sample-rows", str(int(rows or 48))]
        if imatrix:
            cmd += ["--imatrix", str(imatrix)]
        yield from _stream_tool_cmd(cmd)

    def run_ollama_modelfile(gguf, name, ctx):
        if not gguf:
            yield "Pick a GGUF first."
            return
        yield from _stream_tool_cmd([
            str(APP_DIR / "tools" / "ollama_modelfile.py"), str(gguf),
            "--name", name or "", "--num-ctx", str(int(ctx or 32768)),
        ])

    def run_krea_convert(src, quant, dst_name):
        if not src:
            yield "Pick a source safetensors first."
            return
        dst = OUTPUT_DIR / (dst_name or "krea2_q8_0.gguf")
        dst.parent.mkdir(parents=True, exist_ok=True)
        yield from _stream_tool_cmd([
            str(APP_DIR / "tools" / "convert_krea2_to_gguf.py"),
            "--src", str(src), "--dst", str(dst), "--quant", quant or "q8_0",
        ])

    cmp_btn.click(run_compare, inputs=[cmp_ref, cmp_cands, cmp_imatrix, cmp_rows],
                  outputs=[cmp_log])
    oll_btn.click(run_ollama_modelfile, inputs=[oll_gguf, oll_name, oll_ctx],
                  outputs=[oll_log])

    IMG_MODELS_DIR = APP_DIR / "image_models"
    KREA_BASE = "https://huggingface.co/Comfy-Org/Krea-2/resolve/main"

    def run_img_download(label: str, url: str):
        import threading
        IMG_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        state: dict = {}
        yield f"Downloading {label}...\n(this can take a while for multi-GB files; progress prints to the server console)"

        def work():
            try:
                state["path"] = hf_download(url, str(IMG_MODELS_DIR))
            except Exception as exc:  # noqa: BLE001 - surfaced below
                state["error"] = str(exc)

        t = threading.Thread(target=work, daemon=True)
        t.start()
        while t.is_alive():
            yield f"Downloading {label}... {time.strftime('%H:%M:%S')}"
            time.sleep(2)
        if "error" in state:
            yield f"Download failed: {state['error']}"
        else:
            yield f"Done.\n`{state['path']}`"

    def run_img_convert(src, quant, dst_name):
        if not src:
            yield "Pick a source safetensors first (or download one below)."
            return
        dst = OUTPUT_DIR / (dst_name or "krea2_q8_0.gguf")
        dst.parent.mkdir(parents=True, exist_ok=True)
        yield from _stream_tool_cmd([
            str(APP_DIR / "tools" / "convert_krea2_to_gguf.py"),
            "--src", str(src), "--dst", str(dst), "--quant", quant or "q8_0",
        ])

    def dl_turbo():
        yield from run_img_download(
            "krea2_turbo_bf16.safetensors (26 GB)",
            f"{KREA_BASE}/diffusion_models/krea2_turbo_bf16.safetensors")

    def dl_raw():
        yield from run_img_download(
            "krea2_raw_bf16.safetensors (26 GB)",
            f"{KREA_BASE}/diffusion_models/krea2_raw_bf16.safetensors")

    def dl_te():
        yield from run_img_download(
            "qwen3vl_4b_fp8_scaled.safetensors (text encoder)",
            f"{KREA_BASE}/text_encoders/qwen3vl_4b_fp8_scaled.safetensors")

    def dl_vae():
        yield from run_img_download(
            "qwen_image_vae.safetensors (VAE)",
            f"{KREA_BASE}/vae/qwen_image_vae.safetensors")

    img_dl_turbo_btn.click(dl_turbo, outputs=[img_dl_log])
    img_dl_raw_btn.click(dl_raw, outputs=[img_dl_log])
    img_dl_te_btn.click(dl_te, outputs=[img_dl_log])
    img_dl_vae_btn.click(dl_vae, outputs=[img_dl_log])
    img_convert_btn.click(run_img_convert,
                          inputs=[img_src, img_quant, img_dst_name],
                          outputs=[img_convert_log])

    def run_llamacpp_clone():
        yield from run_llamacpp_setup_step("clone", "")

    def run_llamacpp_venv():
        yield from run_llamacpp_setup_step("venv", "")

    def run_llamacpp_build(jobs):
        yield from run_llamacpp_setup_step("quantize", jobs)

    llamacpp_clone_btn.click(run_llamacpp_clone, outputs=[llamacpp_log, llamacpp_status])
    llamacpp_venv_btn.click(run_llamacpp_venv, outputs=[llamacpp_log, llamacpp_status])
    llamacpp_build_btn.click(
        run_llamacpp_build, inputs=[llamacpp_build_jobs], outputs=[llamacpp_log, llamacpp_status],
    )
    llamacpp_refresh_btn.click(llamacpp_status_markdown, outputs=[llamacpp_status])

    def on_llm_source_change(s: str):
        is_hf = s == "Hugging Face repo"
        return gr.update(visible=is_hf), gr.update(visible=is_hf), gr.update(visible=not is_hf)

    llm_source.change(
        on_llm_source_change, inputs=[llm_source], outputs=[llm_repo_id, llm_hf_token, llm_local_dir],
    )
    llm_download_btn.click(
        run_llm_download, inputs=[llm_source, llm_repo_id, llm_local_dir, llm_hf_token],
        outputs=[llm_log, llm_model_dir],
    ).then(
        lambda: gr.update(choices=llm_downloaded_models()), outputs=[llm_model_dd],
    )
    llm_model_dd.change(
        lambda v: v if v else gr.update(), inputs=[llm_model_dd], outputs=[llm_model_dir],
    )
    llm_calib_mode.change(
        lambda m: gr.update(visible=(m or "").startswith("Custom")),
        inputs=[llm_calib_mode], outputs=[llm_imatrix_calibration],
    )

    def carry_over_result(result_path):
        return (result_path if result_path else gr.update())

    def carry_over_result_x2(result_path):
        val = result_path if result_path else gr.update()
        return val, val

    llm_convert_btn.click(
        run_llm_convert, inputs=[llm_model_dir, llm_convert_output_name, llm_outtype],
        outputs=[llm_log, llm_result_file],
    ).then(
        carry_over_result_x2, inputs=[llm_result_file], outputs=[llm_imatrix_model, llm_quantize_input],
    )
    llm_imatrix_btn.click(
        run_llm_generate_imatrix,
        inputs=[llm_imatrix_model, llm_calib_mode, llm_imatrix_calibration, llm_imatrix_output_name],
        outputs=[llm_log, llm_result_file],
    ).then(carry_over_result, inputs=[llm_result_file], outputs=[llm_quantize_imatrix])
    llm_quantize_btn.click(
        run_llm_quantize,
        inputs=[
            llm_quantize_input, llm_quantize_output_name, llm_quant_type,
            llm_quantize_imatrix, llm_quantize_tensor_types,
        ],
        outputs=[llm_log, llm_result_file],
    )
    llm_find_best_btn.click(
        run_llm_find_best,
        inputs=[llm_quantize_input, llm_quantize_imatrix, llm_target_bpw],
        outputs=[llm_log, llm_result_file, llm_quant_type, llm_quantize_imatrix],
    )
    llm_smart_btn.click(
        run_llm_smart_tune,
        inputs=[llm_smart_model, llm_smart_imatrix, llm_smart_mode, llm_smart_target, llm_smart_budget, llm_smart_output],
        outputs=[llm_log, llm_result_file],
    )
    llm_tune_winner_btn.click(
        run_llm_tune_winner,
        inputs=[llm_smart_output],
        outputs=[llm_log, llm_result_file],
    )
    llm_auto_btn.click(
        run_llm_auto_pipeline,
        inputs=[llm_quantize_input, llm_target_bpw, llm_smart_output, llm_val_baseline, llm_val_text],
        outputs=[llm_log, llm_result_file],
    )

    llm_val_btn.click(
        run_llm_validate,
        inputs=[llm_val_tuned, llm_val_ref, llm_val_baseline, llm_val_text, llm_val_ngl],
        outputs=[llm_log, llm_result_file],
    )


if __name__ == "__main__":
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    demo.queue().launch(
        server_name=os.environ.get("QUANT_GUI_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("QUANT_GUI_PORT", "7860")),
        theme=gr.themes.Soft(),
        css=CSS,
    )
