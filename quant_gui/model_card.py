"""Hugging Face model card (README.md) for a finished conversion.

Every statement in the card comes from facts the app actually has: the input
and output names, the format and options used, the conversion summary the
backend printed, and the output's real size. Nothing about quality is claimed
(no benchmark is run), the licence is never guessed (a quantized file inherits
its base model's licence), and usage hints are limited to what the format is
documented to work with.

The card is written NEXT TO the output as ``<output stem>.README.md`` so it
never overwrites another model's card; copy or rename it to ``README.md`` in
the Hugging Face repo you upload to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("ctq", "int4", "gguf_image", "gguf_llm")

_USAGE = {
    "ctq": (
        "This is a `.safetensors` checkpoint quantized with "
        "[convert_to_quant](https://github.com/silveroxides/convert_to_quant). Per-layer quantization "
        "metadata (`comfy_quant`) is stored in the file when that option was enabled, which lets "
        "loaders that understand it pick the right kernels. Use a ComfyUI build that supports quantized "
        "checkpoints of this format and load it like any other diffusion model checkpoint."
    ),
    "int4": (
        "This is a `.safetensors` checkpoint with real INT4 ConvRot layers made through "
        "[comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen); layers whose shape does not suit "
        "INT4 fall back to INT8 or stay at full precision (see the summary above). It needs a loader that "
        "understands comfy-kitchen's quantized layouts - plain loaders will not read it."
    ),
    "gguf_image": (
        "Put the file in `ComfyUI/models/unet/` and load it with the **Unet Loader (GGUF)** node from "
        "[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), or use a runtime that supports this "
        "architecture, such as [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp). "
        "Text encoder and VAE are separate files and are not part of this GGUF."
    ),
    "gguf_llm": (
        "Run it with [llama.cpp](https://github.com/ggml-org/llama.cpp), e.g. "
        "`llama-cli -m {file} -p \"Hello\"`, or load it in any llama.cpp-based app (LM Studio, Ollama "
        "via a Modelfile, etc.)."
    ),
}

_KIND_TAGS = {
    "ctq": ["quantized", "safetensors", "convert-to-quant"],
    "int4": ["quantized", "safetensors", "int4"],
    "gguf_image": ["quantized", "gguf"],
    "gguf_llm": ["quantized", "gguf", "llama.cpp"],
}


@dataclass
class CardInfo:
    kind: str
    output_path: str
    source: str = ""                  # the input file / folder / repo as the user gave it
    base_model: str = ""              # Hugging Face id, only if the user supplied one
    quant_label: str = ""             # e.g. "INT8 ConvRot", "Q8_0", "Mixed 4-bit"
    pipeline_tag: str = ""            # text-generation / text-to-image / ... only when known
    settings: list[tuple[str, str]] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)   # what the backend reported (counts, fallbacks)
    command: str = ""                 # reproducible command, when there is one
    notes: list[str] = field(default_factory=list)     # extra honest caveats


def _human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def _yaml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _cell(v) -> str:
    return str(v).replace("|", "\\|").replace("\n", " ")


def _valid_repo_id(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][\w.\-]*/[\w.\-]+", s or ""))


def card_path(output_path: str | Path) -> Path:
    p = Path(output_path)
    return p.with_name(p.stem + ".README.md")


def render(info: CardInfo, size_bytes: int | None = None) -> str:
    if info.kind not in KINDS:
        raise ValueError(f"unknown card kind {info.kind!r}")
    out = Path(info.output_path)
    base = info.base_model.strip()
    has_base = _valid_repo_id(base)

    fm = ["---"]
    if has_base:
        fm += [f"base_model: {base}", "base_model_relation: quantized"]
    if info.kind.startswith("gguf"):
        fm.append("library_name: gguf")
    if info.pipeline_tag:
        fm.append(f"pipeline_tag: {info.pipeline_tag}")
    fm.append("tags:")
    fm += [f"- {t}" for t in _KIND_TAGS[info.kind]]
    fm.append("---")

    title = out.stem
    body = [f"# {title}", ""]
    if has_base:
        body.append(f"Quantized version of [{base}](https://huggingface.co/{base}).")
    elif info.source:
        src_name = Path(info.source.rstrip("/\\")).name or info.source
        body.append(f"Quantized from `{src_name}`.")
    else:
        body.append("Quantized model.")
    body += ["", f"Made with [quant-convert-gui](https://github.com/8bit-boom/quant-convert-gui).", ""]

    body += ["## Files", "", "| File | Size |", "|---|---|",
             f"| `{out.name}` | {_human(size_bytes) if size_bytes else 'n/a'} |", ""]

    body += ["## Quantization", ""]
    rows = []
    if info.quant_label:
        rows.append(("Format", info.quant_label))
    rows += [(k, v) for k, v in info.settings if str(v).strip()]
    if rows:
        body += ["| Setting | Value |", "|---|---|"] + [f"| {k} | {_cell(v)} |" for k, v in rows] + [""]
    if info.summary:
        body += ["What the converter reported:", ""] + [f"- {s}" for s in info.summary] + [""]
    if info.command:
        body += ["Command used:", "", "```", info.command, "```", ""]

    body += ["## Usage", "", _USAGE[info.kind].replace("{file}", out.name), ""]

    body += ["## Quality", "",
             "No quality evaluation (perplexity, KL divergence, image comparisons) has been run on this file. "
             "Quantization is lossy; compare against the original for your own use case.", ""]
    for n in info.notes:
        body.append(f"- {n}")
    if info.notes:
        body.append("")

    body += ["## License", "",
             "A quantized file inherits the license of the model it was made from. "
             + (f"Check the [original model page](https://huggingface.co/{base}) and set the `license:` field above "
                "to match before publishing." if has_base else
                "Check the original model's page and add the matching `license:` field above before publishing."), ""]
    return "\n".join(fm) + "\n\n" + "\n".join(body)


def write_card(info: CardInfo) -> Path:
    """Writes ``<output stem>.README.md`` next to the output; returns its path. Never raises on a missing
    output (size is then 'n/a'); raises OSError only if the folder is not writable."""
    out = Path(info.output_path)
    try:
        size = out.stat().st_size if out.is_file() else None
    except OSError:
        size = None
    path = card_path(out)
    path.write_text(render(info, size), encoding="utf-8")
    return path
