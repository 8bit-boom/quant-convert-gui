"""Estimate a conversion's output file size by reading the safetensors
*header* only (no torch, no ctq, no loading multi-GB tensor data).

safetensors layout: 8 bytes little-endian header length N, then N bytes of
JSON describing every tensor's dtype/shape/byte offsets, then the raw
tensor bytes. Reading just the header is enough to know every tensor's
exact original size - fast even on a huge file, and needs nothing beyond
the standard library.

The output-size math is a heuristic, not a simulation of ctq's actual
per-layer decisions (it doesn't know ctq's exact size/aspect-ratio skip
rules) - it's meant as a planning figure ("will this fit in my VRAM"), not
an exact prediction.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from .cli_builder import ConvertOptions
from .filters import get_model_filters

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}
FLOAT_DTYPES = {"F64", "F32", "F16", "BF16"}

# Bytes per element for each quantized format, plus a generous fudge factor
# for per-row/per-block scale tensors so the estimate doesn't undershoot.
FORMAT_BYTES_PER_ELEM = {"fp8": 1.0, "int8": 1.0, "mxfp8": 1.0, "nvfp4": 0.5}
FORMAT_SCALE_OVERHEAD = {"fp8": 0.01, "int8": 0.01, "mxfp8": 0.03, "nvfp4": 0.08}

MIN_QUANTIZABLE_DIM = 8


class SafetensorsHeaderError(ValueError):
    pass


@dataclass
class TensorHeader:
    name: str
    dtype: str
    shape: list[int]
    nbytes: int


@dataclass
class SizeEstimate:
    original_bytes: int
    estimated_bytes: int
    quantized_count: int
    kept_count: int
    removed_count: int
    total_count: int


def read_header(path: str) -> list[TensorHeader]:
    """Parse a safetensors file's header without loading any tensor data."""
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise SafetensorsHeaderError("File is too small to be a safetensors file.")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_json = f.read(header_len)
        if len(header_json) < header_len:
            raise SafetensorsHeaderError("Truncated safetensors header - is the file fully downloaded?")

    try:
        header = json.loads(header_json)
    except json.JSONDecodeError as exc:
        raise SafetensorsHeaderError(f"Couldn't parse the safetensors header: {exc}") from exc

    tensors: list[TensorHeader] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        offsets = info.get("data_offsets")
        if not offsets or len(offsets) != 2:
            continue
        tensors.append(
            TensorHeader(name=name, dtype=info.get("dtype", "F32"), shape=info.get("shape", []), nbytes=offsets[1] - offsets[0])
        )
    return tensors


def _preset_lists(preset: str) -> tuple[list[str], list[str], list[str]]:
    if not preset or preset == "none":
        return [], [], []
    info = get_model_filters().get(preset, {})
    return (
        list(info.get("exclude", []) or []),
        list(info.get("highprec", []) or []),
        list(info.get("remove", []) or []),
    )


def _matches_any(name: str, keywords: list[str]) -> bool:
    return any(kw in name for kw in keywords)


def estimate_output_size(tensors: list[TensorHeader], opts: ConvertOptions) -> SizeEstimate:
    exclude_kw, highprec_kw, remove_kw = _preset_lists(opts.preset)
    custom_re = re.compile(opts.custom_layers) if (opts.custom_layers and opts.custom_type) else None
    exclude_re = re.compile(opts.exclude_layers) if opts.exclude_layers else None

    original_total = 0
    estimated_total = 0
    quantized = kept = removed = 0

    for t in tensors:
        original_total += t.nbytes

        if _matches_any(t.name, remove_kw):
            removed += 1
            continue  # dropped entirely by ctq (e.g. --t5xxl's decoder removal)

        is_quantizable_shape = len(t.shape) == 2 and min(t.shape, default=0) >= MIN_QUANTIZABLE_DIM and t.dtype in FLOAT_DTYPES

        chosen_format: str | None
        if custom_re is not None and custom_re.search(t.name) and is_quantizable_shape:
            chosen_format = opts.custom_type
        elif _matches_any(t.name, exclude_kw) or _matches_any(t.name, highprec_kw):
            chosen_format = None
        elif exclude_re is not None and exclude_re.search(t.name):
            chosen_format = opts.fallback
        elif is_quantizable_shape:
            chosen_format = opts.quant_format
        else:
            chosen_format = None

        if chosen_format and chosen_format in FORMAT_BYTES_PER_ELEM:
            dtype_size = DTYPE_BYTES.get(t.dtype, 4)
            elem_count = t.nbytes / dtype_size if dtype_size else 0
            bytes_per_elem = FORMAT_BYTES_PER_ELEM[chosen_format] * (1 + FORMAT_SCALE_OVERHEAD[chosen_format])
            estimated_total += int(elem_count * bytes_per_elem)
            quantized += 1
        else:
            estimated_total += t.nbytes
            kept += 1

    return SizeEstimate(
        original_bytes=original_total,
        estimated_bytes=estimated_total,
        quantized_count=quantized,
        kept_count=kept,
        removed_count=removed,
        total_count=len(tensors),
    )


def human_size(nbytes: int) -> str:
    gb = nbytes / (1024**3)
    if gb >= 0.5:
        return f"{gb:.2f} GB"
    return f"{nbytes / (1024**2):.0f} MB"


def format_estimate_markdown(est: SizeEstimate, gpu_vram_gb: float | None = None) -> str:
    if est.total_count == 0:
        return "Couldn't find any tensors in that file's header."

    reduction = 100 * (1 - est.estimated_bytes / est.original_bytes) if est.original_bytes else 0
    change_str = f"{reduction:.0f}% smaller" if reduction >= 0 else f"{-reduction:.0f}% larger"

    lines = [
        f"**Original:** {human_size(est.original_bytes)}  →  **Estimated output:** ~{human_size(est.estimated_bytes)} ({change_str})",
        f"{est.quantized_count} tensor(s) quantized, {est.kept_count} kept at full precision"
        + (f", {est.removed_count} removed" if est.removed_count else "")
        + f" (of {est.total_count} total).",
    ]

    if gpu_vram_gb:
        vram_bytes = gpu_vram_gb * (1024**3)
        if est.estimated_bytes < vram_bytes * 0.8:
            fit = f"✅ should fit comfortably in your {gpu_vram_gb:.0f} GB VRAM."
        elif est.estimated_bytes < vram_bytes:
            fit = f"⚠️ close to your {gpu_vram_gb:.0f} GB VRAM limit — leaves little room for activations/other models."
        else:
            fit = f"❌ likely won't fit in {gpu_vram_gb:.0f} GB VRAM on its own."
        lines.append(fit)

    lines.append(
        "_Estimate only, from the file's tensor shapes — it approximates ctq's per-layer choices "
        "rather than replicating them exactly. Treat it as a planning figure._"
    )
    return "\n\n".join(lines)


def estimate_from_file(path: str, opts: ConvertOptions, gpu_vram_gb: float | None = None) -> str:
    if not path or not Path(path).is_file():
        return "Pick an input file first."
    try:
        tensors = read_header(path)
    except SafetensorsHeaderError as exc:
        return f"Couldn't read that file: {exc}"
    est = estimate_output_size(tensors, opts)
    return format_estimate_markdown(est, gpu_vram_gb)


def estimate_int4_mixed(
    tensors: list[TensorHeader],
    int4_regex: str | None,
    preset: str = "none",
    exclude_regex: str | None = None,
    fallback_int8: bool = True,
) -> SizeEstimate:
    """Same idea as estimate_output_size, for the separate INT4 ConvRot
    (comfy_kitchen) path - int4_backend.py owns the real divisibility rules,
    imported here rather than re-guessed so the two never drift apart."""
    from .int4_backend import CONVROT_GROUPSIZE, INT4_QUANT_GROUPSIZE

    exclude_kw, highprec_kw, _remove_kw = _preset_lists(preset)
    int4_re = re.compile(int4_regex) if int4_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None

    original_total = 0
    estimated_total = 0
    int4_count = int8_count = kept_count = 0

    for t in tensors:
        original_total += t.nbytes
        excluded = _matches_any(t.name, exclude_kw) or _matches_any(t.name, highprec_kw)
        if exclude_re is not None and exclude_re.search(t.name):
            excluded = True

        is_quantizable_shape = len(t.shape) == 2 and min(t.shape, default=0) >= MIN_QUANTIZABLE_DIM and t.dtype in FLOAT_DTYPES
        dtype_size = DTYPE_BYTES.get(t.dtype, 4)
        elem_count = t.nbytes / dtype_size if dtype_size else 0

        wants_int4 = (
            not excluded and is_quantizable_shape and int4_re is not None and int4_re.search(t.name) is not None
        )
        int4_shape_ok = (
            wants_int4
            and len(t.shape) == 2
            and t.shape[1] % CONVROT_GROUPSIZE == 0
            and t.shape[1] % INT4_QUANT_GROUPSIZE == 0
        )

        if int4_shape_ok:
            estimated_total += int(elem_count * 0.5 * 1.02)  # 0.5 byte/elem + small scale overhead
            int4_count += 1
        elif not excluded and is_quantizable_shape and fallback_int8:
            estimated_total += int(elem_count * 1.0 * 1.01)
            int8_count += 1
        else:
            estimated_total += t.nbytes
            kept_count += 1

    return SizeEstimate(
        original_bytes=original_total,
        estimated_bytes=estimated_total,
        quantized_count=int4_count + int8_count,
        kept_count=kept_count,
        removed_count=0,
        total_count=len(tensors),
    )


def estimate_gguf(tensors: list[TensorHeader], quant_type: str, preset: str = "none", exclude_regex: str | None = None) -> SizeEstimate:
    """Same idea as estimate_output_size, for the GGUF export path - shares
    gguf_backend's own arch detection and F32-fallback rules (1D / small /
    hiprec / shape-not-divisible-by-32) so the estimate can't drift from the
    real converter."""
    from .gguf_backend import MAX_TENSOR_DIMS, QUANT_BYTES_PER_ELEM, QUANTIZATION_THRESHOLD, detect_arch

    exclude_kw, highprec_kw, _remove_kw = _preset_lists(preset)
    exclude_re = re.compile(exclude_regex) if exclude_regex else None
    bytes_per_elem = QUANT_BYTES_PER_ELEM.get(quant_type, 4.0)

    arch = detect_arch({t.name for t in tensors})
    hiprec_kw = list(arch.keys_hiprec) if arch else []

    original_total = 0
    estimated_total = 0
    quantized = kept = 0

    for t in tensors:
        original_total += t.nbytes
        dtype_size = DTYPE_BYTES.get(t.dtype, 4)
        elem_count = t.nbytes / dtype_size if dtype_size else 0

        if len(t.shape) > MAX_TENSOR_DIMS:
            estimated_total += t.nbytes  # skipped, kept as-is in the estimate
            kept += 1
            continue

        force_f32 = (
            len(t.shape) == 1
            or elem_count <= QUANTIZATION_THRESHOLD
            or _matches_any(t.name, hiprec_kw)
            or _matches_any(t.name, exclude_kw)
            or _matches_any(t.name, highprec_kw)
            or (exclude_re is not None and exclude_re.search(t.name))
        )
        shape_ok = len(t.shape) >= 1 and t.shape[-1] % 32 == 0

        if force_f32 or not shape_ok:
            estimated_total += int(elem_count * (4.0 if force_f32 else 2.0))  # F32 kept, else F16 fallback
            kept += 1
        else:
            estimated_total += int(elem_count * bytes_per_elem)
            quantized += 1

    return SizeEstimate(
        original_bytes=original_total,
        estimated_bytes=estimated_total,
        quantized_count=quantized,
        kept_count=kept,
        removed_count=0,
        total_count=len(tensors),
    )


def estimate_gguf_from_file(
    path: str, quant_type: str, preset: str = "none", exclude_regex: str | None = None, gpu_vram_gb: float | None = None,
) -> str:
    if not path or not Path(path).is_file():
        return "Pick an input file first."
    try:
        tensors = read_header(path)
    except SafetensorsHeaderError as exc:
        return f"Couldn't read that file: {exc}"
    from .gguf_backend import detect_arch

    arch = detect_arch({t.name for t in tensors})
    if arch is None:
        from .gguf_backend import SUPPORTED_ARCH_NAMES

        return (
            "Unknown model architecture - GGUF export only recognizes: "
            f"{', '.join(SUPPORTED_ARCH_NAMES)}. This file doesn't match any of them."
        )
    est = estimate_gguf(tensors, quant_type, preset, exclude_regex)
    header = f"Detected architecture: **{arch.arch}**\n\n"
    return header + format_estimate_markdown(est, gpu_vram_gb)


def estimate_int4_mixed_from_file(
    path: str,
    int4_regex: str | None,
    preset: str = "none",
    exclude_regex: str | None = None,
    fallback_int8: bool = True,
    gpu_vram_gb: float | None = None,
) -> str:
    if not path or not Path(path).is_file():
        return "Pick an input file first."
    try:
        tensors = read_header(path)
    except SafetensorsHeaderError as exc:
        return f"Couldn't read that file: {exc}"
    est = estimate_int4_mixed(tensors, int4_regex, preset, exclude_regex, fallback_int8)
    return format_estimate_markdown(est, gpu_vram_gb)
