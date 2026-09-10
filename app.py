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
from quant_gui.filters import preset_choices, preset_label, suggest_preset
from quant_gui.hf import HFUrlError, download as hf_download, parse_hf_url
from quant_gui.runner import stream_conversion

APP_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = APP_DIR / "downloads"
OUTPUT_DIR = APP_DIR / "converted"

FORMAT_CHOICES = [
    ("INT8 — ConvRot (recommended, matches Kroma-Quant *-int8-convrot* files)", "int8_convrot"),
    ("INT8 — plain (row / block / tensor scaling, no rotation)", "int8_plain"),
    ("FP8 (E4M3) — Ada / Hopper+ GPUs", "fp8"),
    ("NVFP4 — 4-bit, closest available today (Blackwell GPUs only)", "nvfp4"),
    ("MXFP8 — Blackwell GPUs only", "mxfp8"),
    ("INT4 ConvRot — not released by upstream ctq yet", "int4_convrot"),
]

INT4_NOTICE = (
    "### INT4 ConvRot isn't available yet\n\n"
    "You asked for **int4 convrot**, but as of `convert_to_quant` v1.3.4 the upstream tool "
    "that actually performs ConvRot quantization only ships **INT8 ConvRot**. There is no "
    "integer 4-bit output format in the converter (the `pack_uint4` code in ctq belongs to "
    "**NVFP4**, a 4-bit *floating point* format, not INT4).\n\n"
    "The Kroma-Quant file you linked is itself an `int8-convrot-simple` file — the reference "
    "model, and today's ceiling for this exact technique.\n\n"
    "Your closest real options:\n"
    "- **NVFP4** — an actual 4-bit format, but it requires a Blackwell GPU (RTX 50-series / B-series) "
    "plus the `comfy-kitchen` package, and is a different rotation/scaling scheme than ConvRot.\n"
    "- **INT8 ConvRot** — the same recipe used to build the file you linked; works on any GPU.\n\n"
    "Pick one of the buttons below to switch, or keep watching the "
    "[silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant) repo — "
    "if they add a real INT4 path this GUI will pick it up (the format list mirrors ctq's own flags)."
)

FORMAT_HELP = {
    "int8_convrot": (
        "INT8 weights plus a group-wise Hadamard rotation that spreads out outlier values before "
        "quantizing, which is what gives ConvRot files their quality edge over plain INT8. "
        "Runs on any GPU (or CPU, slowly)."
    ),
    "int8_plain": "Straight INT8 quantization with row/block/tensor scaling, no rotation step. Any GPU.",
    "fp8": "8-bit float. Needs an Ada/Hopper-or-newer NVIDIA GPU to actually run faster than bf16.",
    "nvfp4": "NVIDIA's 4-bit float block format. Requires a Blackwell GPU and the comfy-kitchen package.",
    "mxfp8": "Microscaling FP8. Requires a Blackwell GPU.",
    "int4_convrot": "Not implemented upstream yet — see the notice above.",
}

PRESET_CHOICES = preset_choices()
PRESET_LABELS = {name: preset_label(name) for name in PRESET_CHOICES}
LABEL_TO_PRESET = {v: k for k, v in PRESET_LABELS.items()}


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
    can_convert = not is_int4
    return (
        gr.update(visible=is_int8_convrot),  # convrot group
        gr.update(visible=is_plain_int8_or_fp8),  # scaling mode group
        gr.update(value=FORMAT_HELP.get(fmt, "")),
        gr.update(visible=is_int4),  # int4 notice
        gr.update(interactive=can_convert),  # convert button
        gr.update(visible=is_int4),  # int4 switch-to row
    )


def on_source_change(source: str):
    return gr.update(visible=source == "Local file path"), gr.update(visible=source == "Hugging Face URL")


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


def on_input_resolved(local_path: str, hf_url: str):
    hint = local_path or hf_url
    preset = suggest_preset(hint)
    if preset:
        return f"Detected `{Path(hint).name if local_path else hf_url}` — suggesting the **{preset}** preset below (used for txtfusion/Krea2/Kroma-style models)."
    return ""


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
    device: str,
    output_dtype: str,
    verbose: str,
    calib_samples: float,
    optimizer: str,
    num_iter: float,
    manual_seed: float,
    python_exe: str,
):
    input_path = (input_local or "").strip() if source == "Local file path" else (input_hf or "").strip()

    if fmt == "int4_convrot":
        yield "INT4 ConvRot isn't supported by ctq yet — see the notice above. Nothing was run.", None
        return

    if not input_path:
        yield "Pick an input file (local path or downloaded Hugging Face file) first.", None
        return

    if not Path(input_path).is_file():
        yield f"Input file not found on disk: {input_path}", None
        return

    try:
        opts = build_options(
            input_path, output_name, auto_output, fmt, quality_mode, convrot_group_size,
            dynamic_convrot, scaling_mode, block_size, preset_label_value, comfy_quant,
            save_metadata, low_memory, exclude_layers, custom_layers, device, output_dtype,
            verbose, calib_samples, optimizer, num_iter, manual_seed,
        )
        args = build_args(opts)
    except OptionsError as exc:
        yield f"Can't build a valid command: {exc}", None
        return

    if opts.output_path:
        Path(opts.output_path).parent.mkdir(parents=True, exist_ok=True)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log = ""
    result_path = None
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
            yield log, result_path
        elif chunk.startswith("__CTQ_FAIL__"):
            code = chunk.split(":", 1)[-1]
            log += f"\n❌ ctq exited with code {code}.\n"
            yield log, None
        else:
            log += chunk
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
        "Turn a `.safetensors` model into **FP8**, **INT8**, **INT8 ConvRot**, or **NVFP4**, "
        "using [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant) under the hood — "
        "the same tool used to build the [Kroma-Quant](https://huggingface.co/silveroxides/Kroma-Quant) files."
    )

    with gr.Tabs():
        with gr.Tab("Convert"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### 1. Choose your model")
                    source = gr.Radio(["Local file path", "Hugging Face URL"], value="Hugging Face URL", label="Source")

                    with gr.Group(visible=False) as local_group:
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

                    gr.Markdown("### 2. Choose an output format")
                    fmt = gr.Radio(
                        [c[0] for c in FORMAT_CHOICES],
                        value=FORMAT_CHOICES[0][0],
                        label="Quantization format",
                    )
                    fmt_value = gr.State(FORMAT_CHOICES[0][1])
                    fmt_help = gr.Markdown(FORMAT_HELP["int8_convrot"])
                    int4_notice = gr.Markdown(INT4_NOTICE, visible=False)
                    with gr.Row(visible=False) as int4_switch_row:
                        switch_to_int8 = gr.Button("Use INT8 ConvRot instead")
                        switch_to_nvfp4 = gr.Button("Use NVFP4 instead")

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

                    gr.Markdown("### 3. Quality vs. speed")
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

                    with gr.Accordion("Advanced options", open=False):
                        with gr.Row():
                            comfy_quant = gr.Checkbox(value=True, label="ComfyUI-compatible layout (--comfy_quant)")
                            save_metadata = gr.Checkbox(value=True, label="Save quantization metadata")
                            low_memory = gr.Checkbox(value=True, label="Low memory mode")
                        exclude_layers = gr.Textbox(label="Exclude layers (regex)", placeholder="e.g. (final_layer|txt_attn.proj)")
                        custom_layers = gr.Textbox(label="Custom layers (regex, advanced)")
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

                    gr.Markdown("### 4. Output")
                    auto_output = gr.Checkbox(value=True, label="Auto-generate output filename (recommended)")
                    output_name = gr.Textbox(label="Output filename", interactive=False, placeholder="auto")

                    command_preview = gr.Textbox(label="Command ctq will run", interactive=False, lines=2)

                    convert_btn = gr.Button("Convert", elem_id="convert-btn", variant="primary")

                with gr.Column(scale=2):
                    log_box = gr.Textbox(label="Live conversion log", lines=28, elem_id="log-box", interactive=False, autoscroll=True)
                    result_file = gr.File(label="Converted file", interactive=False)

        with gr.Tab("Environment"):
            gr.Markdown(
                "Conversion runs on **your machine** through `ctq`. This checks whether it (and PyTorch/CUDA) "
                "are actually installed and what hardware is available."
            )
            env_md = gr.Markdown(report_markdown(check_environment()))
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
                "- **INT8** — 8-bit integer with a per-row/block/tensor scale.\n"
                "- **INT8 ConvRot** — INT8 plus a group-wise Hadamard rotation applied before quantizing, "
                "which spreads out the outlier values that normally hurt low-bit accuracy in diffusion "
                "transformers. This is the recipe behind the `*-int8-convrot*` Kroma-Quant files.\n"
                "- **NVFP4** — NVIDIA's 4-bit floating point block format; needs a Blackwell GPU.\n"
                "- **INT4 ConvRot** — not released by upstream `convert_to_quant` as of v1.3.4. "
                "This GUI mirrors ctq's real flags, so it will pick this up the moment ctq ships it.\n\n"
                "## Credit\n"
                "All the actual quantization math lives in "
                "[silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant) (`ctq`). "
                "This app is just a GUI wrapper around it, and downloads/uploads nothing on its own besides "
                "the model file you point it at."
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

    fmt_outputs = [fmt_value, convrot_group, scaling_group, fmt_help, int4_notice, convert_btn, int4_switch_row]

    fmt.change(on_fmt_select, inputs=[fmt], outputs=fmt_outputs)

    switch_to_int8.click(lambda: FORMAT_CHOICES[0][0], outputs=[fmt]).then(
        on_fmt_select, inputs=[fmt], outputs=fmt_outputs
    )
    switch_to_nvfp4.click(lambda: FORMAT_CHOICES[3][0], outputs=[fmt]).then(
        on_fmt_select, inputs=[fmt], outputs=fmt_outputs
    )

    source.change(on_source_change, inputs=[source], outputs=[local_group, hf_group])

    download_btn.click(do_hf_download, inputs=[input_hf_url, hf_token], outputs=[input_hf_local, download_status]).then(
        on_input_resolved, inputs=[input_hf_local, input_hf_url], outputs=[resolved_hint]
    )
    input_local.change(on_input_resolved, inputs=[input_local, input_hf_url], outputs=[resolved_hint])

    auto_output.change(lambda auto: gr.update(interactive=not auto), inputs=[auto_output], outputs=[output_name])

    def refresh_preview(*args):
        input_local_v, input_hf_local_v, source_v = args[0], args[1], args[2]
        input_path = input_local_v if source_v == "Local file path" else input_hf_local_v
        rest = args[3:]
        try:
            opts = build_options(input_path or "placeholder.safetensors", *rest)
            return format_command(opts)
        except OptionsError as exc:
            return f"(invalid options: {exc})"

    preview_inputs = [
        input_local, input_hf_local, source,
        output_name, auto_output, fmt_value, quality_mode, convrot_group_size, dynamic_convrot,
        scaling_mode, block_size, preset_dd, comfy_quant, save_metadata, low_memory,
        exclude_layers, custom_layers, device, output_dtype, verbose,
        calib_samples, optimizer, num_iter, manual_seed,
    ]
    for comp in preview_inputs:
        comp.change(refresh_preview, inputs=preview_inputs, outputs=[command_preview])

    fmt_value.change(refresh_preview, inputs=preview_inputs, outputs=[command_preview])

    convert_btn.click(
        run_convert,
        inputs=[
            input_local, input_hf_local, source, output_name, auto_output, fmt_value, quality_mode,
            convrot_group_size, dynamic_convrot, scaling_mode, block_size, preset_dd, comfy_quant,
            save_metadata, low_memory, exclude_layers, custom_layers, device, output_dtype, verbose,
            calib_samples, optimizer, num_iter, manual_seed, python_exe,
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
