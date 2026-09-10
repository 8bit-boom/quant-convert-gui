"""Real INT4 ConvRot ("W4A4") conversion, built directly on comfy_kitchen.

`convert_to_quant` (ctq) - the tool the rest of this app wraps - has no INT4
CLI flag as of this writing (tracked, unaddressed:
https://github.com/silveroxides/convert_to_quant/issues/50). But the actual
kernel it would need already exists, shipped by the same ComfyUI ecosystem:
`comfy_kitchen.tensor.convrot_w4a4` implements the identical group-256
Hadamard-rotate + signed-INT4-pack recipe real int4-mixed community models
use (confirmed against silveroxides/comfy-kitchen source and cross-checked
with the Starnodes Model Converter's own working int4_convrot code path,
which produces files ComfyUI actually loads).

This module is a small, from-scratch converter around that same primitive -
not a wrapper around ctq or around the Starnodes ComfyUI node (which can't
run outside ComfyUI). It intentionally keeps the recipe simple for a first
cut: a user-supplied regex picks which layers go to real INT4 ConvRot,
everything else quantizable goes to plain INT8 tensorwise (also via
comfy_kitchen, for metadata compatibility with the rest of the ecosystem),
and preset/exclude-matched layers stay BF16.
"""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .filters import get_model_filters

CONVROT_GROUPSIZE = 256
INT4_QUANT_GROUPSIZE = 64  # fixed by comfy_kitchen's int4 tensor-core kernel
MIN_QUANTIZABLE_DIM = 8
MIN_SM_VERSION = (7, 5)  # Turing+ - comfy_kitchen's own TensorCoreConvRotW4A4Layout.MIN_SM_VERSION


class Int4BackendError(RuntimeError):
    pass


def is_available() -> bool:
    return importlib.util.find_spec("comfy_kitchen") is not None


def install_hint() -> str:
    return "pip install comfy-kitchen  # https://github.com/Comfy-Org/comfy-kitchen"


@dataclass
class Int4ConvertStats:
    total: int = 0
    int4_count: int = 0
    int8_count: int = 0
    kept_count: int = 0
    skipped_shape_count: int = 0
    int4_layer_names: list[str] = field(default_factory=list)


def _preset_lists(preset: str) -> tuple[list[str], list[str]]:
    if not preset or preset == "none":
        return [], []
    info = get_model_filters().get(preset, {})
    return list(info.get("exclude", []) or []), list(info.get("highprec", []) or [])


def _matches_any(name: str, keywords: list[str]) -> bool:
    return any(kw in name for kw in keywords)


def convert_int4_mixed(
    input_path: str,
    output_path: str,
    int4_layers_regex: str | None,
    preset: str = "none",
    exclude_regex: str | None = None,
    fallback_int8: bool = True,
    device: str = "cpu",
    progress_cb=None,
) -> Int4ConvertStats:
    """Convert a safetensors model to a mixed INT4 ConvRot / INT8 tensorwise
    / BF16 file, streaming one tensor at a time (never holds the whole model
    in RAM at once).

    `int4_layers_regex` matches against the *source* tensor names (e.g.
    `attn.wq|mlp.gate`). Layers that match get real packed-signed-INT4
    ConvRot weights; everything else 2D/float gets INT8 tensorwise (unless
    `fallback_int8` is False, in which case it stays BF16); anything matched
    by `preset`'s exclusion list or `exclude_regex`, or that isn't a
    quantizable 2D float tensor, stays at its original precision (cast to
    BF16 if it was a wider float type).
    """
    if not is_available():
        raise Int4BackendError(
            f"comfy-kitchen isn't installed. Install it with:\n  {install_hint()}"
        )

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from comfy_kitchen.tensor import TensorWiseINT8Layout
    from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout

    exclude_kw, highprec_kw = _preset_lists(preset)
    int4_re = re.compile(int4_layers_regex) if int4_layers_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None

    stats = Int4ConvertStats()
    out_tensors: dict[str, "torch.Tensor"] = {}
    quant_map: dict[str, dict] = {"layers": {}}

    with safe_open(input_path, framework="pt", device=device) as f:
        keys = list(f.keys())
        stats.total = len(keys)

        for i, key in enumerate(keys):
            tensor = f.get_tensor(key)
            if progress_cb:
                progress_cb(i + 1, stats.total, key)

            base_key = key[: -len(".weight")] if key.endswith(".weight") else key

            excluded = _matches_any(key, exclude_kw) or _matches_any(key, highprec_kw)
            if exclude_re is not None and exclude_re.search(key):
                excluded = True

            is_quantizable_shape = (
                tensor.dim() == 2
                and min(tensor.shape) >= MIN_QUANTIZABLE_DIM
                and tensor.dtype in (torch.float32, torch.float16, torch.bfloat16)
            )
            wants_int4 = (
                not excluded
                and is_quantizable_shape
                and int4_re is not None
                and int4_re.search(key) is not None
            )
            int4_shape_ok = (
                wants_int4
                and tensor.shape[1] % CONVROT_GROUPSIZE == 0
                and tensor.shape[1] % INT4_QUANT_GROUPSIZE == 0
            )

            if wants_int4 and not int4_shape_ok:
                stats.skipped_shape_count += 1

            if int4_shape_ok:
                qdata, params = TensorCoreConvRotW4A4Layout.quantize(
                    tensor.float(), convrot_groupsize=CONVROT_GROUPSIZE, quant_group_size=INT4_QUANT_GROUPSIZE,
                )
                for suffix, t in TensorCoreConvRotW4A4Layout.state_dict_tensors(qdata, params).items():
                    out_tensors[key + suffix] = t.cpu()
                quant_map["layers"][base_key] = {
                    "format": "convrot_w4a4",
                    "convrot_groupsize": CONVROT_GROUPSIZE,
                    "quant_group_size": INT4_QUANT_GROUPSIZE,
                }
                stats.int4_count += 1
                stats.int4_layer_names.append(base_key)
            elif not excluded and is_quantizable_shape and fallback_int8:
                qdata, params = TensorWiseINT8Layout.quantize(tensor.float(), per_channel=True)
                for suffix, t in TensorWiseINT8Layout.state_dict_tensors(qdata, params).items():
                    out_tensors[key + suffix] = t.cpu()
                quant_map["layers"][base_key] = {"format": "int8_tensorwise"}
                stats.int8_count += 1
            else:
                out_tensors[key] = tensor.to(torch.bfloat16) if tensor.dtype.is_floating_point else tensor.cpu()
                stats.kept_count += 1

    metadata = {"converted_by": "quant-convert-gui (INT4 ConvRot via comfy_kitchen)"}
    if quant_map["layers"]:
        metadata["_quantization_metadata"] = json.dumps(quant_map)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, output_path, metadata=metadata)
    return stats


def stream_int4_conversion(
    input_path: str,
    output_path: str,
    int4_layers_regex: str | None,
    preset: str = "none",
    exclude_regex: str | None = None,
    fallback_int8: bool = True,
    device: str = "cpu",
):
    """Generator wrapper around convert_int4_mixed for UIs: runs the (blocking)
    conversion in a background thread and yields text/progress events as it
    goes, mirroring quant_gui.runner.stream_conversion's interface.

    Yields either ("log", text) or ("progress", current, total, key), and
    finally either ("ok", stats) or ("fail", error_message).
    """
    import queue
    import threading

    q: "queue.Queue" = queue.Queue()
    SENTINEL = object()

    def progress_cb(current, total, key):
        q.put(("progress", current, total, key))

    def worker():
        try:
            stats = convert_int4_mixed(
                input_path, output_path, int4_layers_regex,
                preset=preset, exclude_regex=exclude_regex, fallback_int8=fallback_int8,
                device=device, progress_cb=progress_cb,
            )
            q.put(("ok", stats))
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
            q.put(("fail", str(exc)))
        finally:
            q.put(SENTINEL)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is SENTINEL:
            break
        yield item
