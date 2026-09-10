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
from quant_gui.int4_backend import stream_int4_conversion
from quant_gui.int4_backend import is_available as int4_is_available
from quant_gui.runner import stream_conversion
from quant_gui.size_estimate import estimate_from_file, estimate_int4_mixed_from_file

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
]

FORMAT_LABEL_BY_KEY = {v: k for k, v in FORMAT_CHOICES}

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
    "why). It isn't installed yet:\n\n"
    "```bash\npip install comfy-kitchen\n```\n\n"
    "Needs a Turing-or-newer GPU (SM 7.5+ — RTX 20-series onward, so your Ampere/Ada/Blackwell card is fine) "
    "for real speed; the same eager PyTorch path also runs correctly on CPU, just slowly. Re-check the "
    "**Environment** tab after installing."
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
    notice_text = ""
    if is_int4:
        notice_text = INT4_NOTICE_READY if int4_is_available() else INT4_NOTICE_MISSING
    return (
        gr.update(visible=is_int8_convrot),  # convrot group
        gr.update(visible=is_plain_int8_or_fp8),  # scaling mode group
        gr.update(value=FORMAT_HELP.get(fmt, "")),
        gr.update(value=notice_text, visible=is_int4),  # int4 notice
        gr.update(visible=is_int4),  # int4 options group
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


def run_int4_convert(
    input_path: str,
    output_name: str,
    auto_output: bool,
    preset_label_value: str,
    int4_layers_regex: str,
    int4_fallback_int8: bool,
    exclude_layers: str,
    device: str,
    progress: gr.Progress,
):
    input_path = (input_path or "").strip()
    if not input_path:
        yield "Pick an input file (local path or downloaded Hugging Face file) first.", None
        return
    if not Path(input_path).is_file():
        yield f"Input file not found on disk: {input_path}", None
        return
    if not int4_is_available():
        yield f"comfy-kitchen isn't installed. Install it with:\n  pip install comfy-kitchen\n\nThen re-run.", None
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
    yield log, None
    progress(0, desc="Starting INT4 conversion...")

    result_path = None
    for item in stream_int4_conversion(
        input_path, output_path, (int4_layers_regex or "").strip() or None,
        preset=preset, exclude_regex=(exclude_layers or "").strip() or None,
        fallback_int8=int4_fallback_int8, device=(device or "cpu").strip() or "cpu",
    ):
        kind = item[0]
        if kind == "progress":
            _, current, total, key = item
            if total:
                progress(current / total, desc=f"{current}/{total}: {key}")
            log += f"({current}/{total}) {key}\n"
            yield log, result_path
        elif kind == "ok":
            stats = item[1]
            progress(1.0, desc="Done")
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
            yield log, result_path
        elif kind == "fail":
            log += f"\n❌ Conversion failed: {item[1]}\n"
            yield log, None


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
    progress: gr.Progress = gr.Progress(),
):
    input_path = (input_local or "").strip() if source == "Local file path" else (input_hf or "").strip()

    if fmt == "int4_convrot":
        yield from run_int4_convert(
            input_path, output_name, auto_output, preset_label_value,
            int4_layers_regex, int4_fallback_int8, exclude_layers, device, progress,
        )
        return

    if not input_path:
        yield "Pick an input file (local path or downloaded Hugging Face file) first.", None
        return

    if not Path(input_path).is_file():
        yield f"Input file not found on disk: {input_path}", None
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
        yield f"Can't build a valid command: {exc}", None
        return

    if opts.output_path:
        Path(opts.output_path).parent.mkdir(parents=True, exist_ok=True)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    progress(0, desc="Starting ctq...")
    # Matches both "(12/264) Processing (INT8): blocks.0.mlp.up.weight" and
    # "(2/6) Skipping tensor: blocks.0.firs.weight (Reason: krea2 skip)".
    tensor_progress_re = re.compile(r"\((\d+)/(\d+)\)\s*(Processing|Skipping)")

    log = ""
    result_path = None
    for chunk in stream_conversion(args, python_executable=((python_exe or "").strip() or None)):
        if chunk == "__CTQ_OK__":
            found = opts.output_path
            if not found:
                m = re.search(r"Saved to[:\s]+(\S+\.safetensors)", log, re.IGNORECASE)
                found = m.group(1) if m else None
            result_path = found if found and Path(found).is_file() else None
            progress(1.0, desc="Done")
            log += "\n✅ Conversion finished.\n"
            if result_path:
                log += f"Output: {result_path}\n"
            elif found:
                log += f"Output (reported, not found on disk yet): {found}\n"
            yield log, result_path
        elif chunk.startswith("__CTQ_FAIL__"):
            code = chunk.split(":", 1)[-1]
            log += f"\n❌ ctq exited with code {code}.\n"
            yield log, None
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
                    progress(current / total, desc=desc)
            yield log, result_path


def refresh_env():
    return report_markdown(check_environment())


CSS = """
#convert-btn { font-size: 1.1rem; font-weight: 600; }
#log-box textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85rem; }
"""

with gr.Blocks(title="Quant Convert GUI") as demo:
    gr.Markdown(
        "# Quant Convert GUI\n"
        "Turn a `.safetensors` model into **FP8**, **INT8**, **INT8 ConvRot**, or **NVFP4** — always "
        "`.safetensors` out, never GGUF — using "
        "[silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant) under the hood, "
        "the same tool used to build the [Kroma-Quant](https://huggingface.co/silveroxides/Kroma-Quant) and "
        "PotatoForge/Kroma-INT8-Quants files. Pick your GPU below and it'll steer you toward a format your "
        "card can actually accelerate."
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
                    with gr.Group(visible=False) as int4_group:
                        int4_layers_regex = gr.Textbox(
                            label="INT4 layers (regex)",
                            placeholder=r"e.g. attn\.wq|mlp\.gate — layers matching this get real INT4 ConvRot",
                            info="Matches source tensor names. Leave empty to convert nothing to INT4 "
                            "(falls back to plain INT8 tensorwise everywhere quantizable).",
                        )
                        int4_fallback_int8 = gr.Checkbox(
                            value=True,
                            label="INT8 tensorwise fallback for other quantizable layers (recommended)",
                            info="Unchecked, non-matched layers stay at full precision instead.",
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
                            comfy_quant = gr.Checkbox(value=True, label="ComfyUI-compatible layout (--comfy_quant)")
                            save_metadata = gr.Checkbox(value=True, label="Save quantization metadata")
                            low_memory = gr.Checkbox(value=True, label="Low memory mode")
                        exclude_layers = gr.Textbox(label="Exclude layers (regex)", placeholder="e.g. (final_layer|txt_attn.proj)")

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
                        )
                        with gr.Row():
                            custom_type = gr.Dropdown(
                                ["none", "fp8", "int8", "mxfp8", "nvfp4"], value="none", label="Custom layer type"
                            )
                            custom_scaling_mode = gr.Dropdown(
                                ["none", "tensor", "row", "block"], value="none", label="Custom layer scaling mode"
                            )
                        with gr.Row():
                            custom_convrot = gr.Checkbox(value=False, label="ConvRot on custom layers (INT8 only)")
                            custom_convrot_group_size = gr.Dropdown(
                                [4, 16, 64, 256, 1024], value=256, label="Custom ConvRot group size"
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
                                info="What excluded/unmatched layers become, instead of staying full precision.",
                            )
                            fallback_simple = gr.Checkbox(value=False, label="Simple quant for fallback layers")

                        with gr.Row():
                            device = gr.Textbox(label="Device override", placeholder="cuda / cuda:0 / cpu")
                            output_dtype = gr.Radio(["bfloat16", "float16"], value="bfloat16", label="Output dtype")
                            verbose = gr.Radio(["MINIMAL", "NORMAL", "VERBOSE", "DEBUG"], value="NORMAL", label="Log verbosity")
                        gr.Markdown("**Learned/AdaRound settings** (used only in Learned mode)")
                        with gr.Row():
                            calib_samples = gr.Number(value=3072, label="Calibration samples", precision=0)
                            optimizer = gr.Dropdown(["prodigy", "adamw", "radam", "original"], value="prodigy", label="Optimizer")
                        with gr.Row():
                            num_iter = gr.Number(value=4000, label="Iterations per tensor", precision=0)
                            manual_seed = gr.Number(value=-1, label="Manual seed (-1 = random)", precision=0)
                        python_exe = gr.Textbox(
                            label="Python executable running ctq (optional)",
                            placeholder="leave blank to use this app's Python / the ctq command on PATH",
                        )

                    gr.Markdown("### 5. Output")
                    auto_output = gr.Checkbox(value=True, label="Auto-generate output filename (recommended)")
                    output_name = gr.Textbox(label="Output filename", interactive=False, placeholder="auto")

                    command_preview = gr.Textbox(label="Command ctq will run", interactive=False, lines=2)

                    estimate_btn = gr.Button("Estimate output size")
                    estimate_md = gr.Markdown()

                    convert_btn = gr.Button("Convert", elem_id="convert-btn", variant="primary")

                with gr.Column(scale=2):
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
                "and works best on Turing+ (SM 7.5+) GPUs, though it also runs — slowly — on CPU.\n\n"
                "## Which format for which GPU\n"
                "| GPU generation | Example cards | Hardware-accelerated formats |\n"
                "|---|---|---|\n"
                "| Turing/Ampere | RTX 20/30-series, A100 — **incl. RTX 3080 Ti** | **INT8 / INT8 ConvRot / INT4 ConvRot** — no FP8 or NVFP4 hardware path |\n"
                "| Ada Lovelace | RTX 40-series, L40 | FP8, INT8 ConvRot, INT4 ConvRot |\n"
                "| Hopper | H100, H200 | FP8, INT8 ConvRot, INT4 ConvRot |\n"
                "| Blackwell | RTX 50-series, B100/B200 | NVFP4, MXFP8, INT8 ConvRot, INT4 ConvRot |\n\n"
                "Picking FP8 or NVFP4 on an unsupported card doesn't reliably fail at conversion time — "
                "the file often still gets written, it just won't load or run fast in ComfyUI. The "
                "**Target GPU** picker on the Convert tab exists to head that off.\n\n"
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

    fmt_outputs = [fmt_value, convrot_group, scaling_group, fmt_help, int4_notice, int4_group]

    fmt.change(on_fmt_select, inputs=[fmt], outputs=fmt_outputs)

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
        int4_regex_v, int4_fallback_v = args[-2], args[-1]

        input_path = (input_path or "").strip()
        if not input_path or not Path(input_path).is_file():
            return "Pick an input file (local path, or download a Hugging Face file) first."
        vram = check_environment().gpu_vram_gb

        fmt_v = rest[preview_field_index["fmt_value"]]
        if fmt_v == "int4_convrot":
            preset = LABEL_TO_PRESET.get(rest[preview_field_index["preset_dd"]], "none")
            exclude_layers_v = rest[preview_field_index["exclude_layers"]]
            return estimate_int4_mixed_from_file(
                input_path, (int4_regex_v or "").strip() or None, preset=preset,
                exclude_regex=(exclude_layers_v or "").strip() or None,
                fallback_int8=int4_fallback_v, gpu_vram_gb=vram,
            )
        try:
            opts = build_options(input_path, *rest)
        except OptionsError as exc:
            return f"Can't estimate: {exc}"
        return estimate_from_file(input_path, opts, gpu_vram_gb=vram)

    estimate_btn.click(
        do_estimate,
        inputs=preview_inputs + [int4_layers_regex, int4_fallback_int8],
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
            int4_layers_regex, int4_fallback_int8,
        ],
        outputs=[log_box, result_file],
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
