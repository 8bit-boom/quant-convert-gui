"""GGUF export, built directly on the `gguf` package (llama.cpp's Python
bindings) - not a wrapper around ctq, which never touches GGUF at all
(everything else in this app always produces `.safetensors`).

Architecture detection and the F32-for-1D/small/sensitive-tensor rules are
ported from city96/ComfyUI-GGUF's own `tools/convert.py` (Apache-2.0) -
that project's loader is what actually reads these files back in ComfyUI,
and its loader.py rejects any `general.architecture` value it doesn't
recognize, so this module intentionally matches its exact detection keys
rather than inventing new ones.

Real block quantization (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) is implemented in pure
Python/numpy by the `gguf` package itself (`gguf.quants`, added upstream
for llama.cpp's `convert_hf_to_gguf.py --outtype` support) - confirmed
against the installed package's source, no C++ build required. The K-quant
family (Q4_K/Q5_K/Q6_K/...) only has a *dequantize* implementation in that
same module; producing them still requires compiling a patched
`llama-quantize` binary (see ComfyUI-GGUF/tools/README.md) - out of scope
for this app, and this module refuses those types rather than silently
mislabeling a legacy quant as a K-quant.
"""

from __future__ import annotations

import importlib.util
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .checkpoints import Checkpoint, record_tensor, set_total
from .filters import get_model_filters
from .run_control import RunCancelled

MAX_TENSOR_DIMS = 4
MAX_TENSOR_NAME_LENGTH = 127
QUANTIZATION_THRESHOLD = 1024  # tensors with fewer elements stay F32


class GGUFBackendError(RuntimeError):
    pass


def is_available() -> bool:
    return importlib.util.find_spec("gguf") is not None


def install_hint() -> str:
    return "pip install gguf  # https://github.com/ggerganov/llama.cpp/tree/master/gguf-py"


def stream_install(python_executable: str | None = None):
    """Run `pip install gguf` via the given (or current) interpreter, yielding
    stdout/stderr lines as they arrive - mirrors int4_backend.stream_install.

    Final yielded line is "__GGUF_INSTALL_OK__" or "__GGUF_INSTALL_FAIL__:<c>".
    """
    import subprocess
    import sys

    py = python_executable or sys.executable
    cmd = [py, "-m", "pip", "install", "-U", "gguf"]
    yield f"$ {' '.join(cmd)}\n"

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    except OSError as exc:
        yield f"Could not launch pip: {exc}\n"
        yield "__GGUF_INSTALL_FAIL__:127"
        return

    assert proc.stdout is not None
    for line in proc.stdout:
        yield line
    code = proc.wait()

    if code == 0:
        importlib.invalidate_caches()
        yield "__GGUF_INSTALL_OK__"
    else:
        yield f"__GGUF_INSTALL_FAIL__:{code}"


@dataclass
class ModelArch:
    arch: str
    keys_detect: list[tuple[str, ...]]
    keys_banned: tuple[str, ...] = ()
    keys_hiprec: tuple[str, ...] = ()


# Ported from city96/ComfyUI-GGUF tools/convert.py (Apache-2.0) - these are
# the exact key signatures its detect_arch()/is_model_arch() use, matched
# against the same allowlist ComfyUI-GGUF's own loader checks on load.
MODEL_ARCHES: list[ModelArch] = [
    ModelArch(
        arch="flux",
        keys_detect=[("double_blocks.0.img_attn.proj.weight",)],
        keys_banned=("transformer_blocks.0.attn.norm_added_k.weight",),
    ),
    ModelArch(
        arch="sd3",
        keys_detect=[("joint_blocks.0.x_block.attn.qkv.weight",)],
        keys_banned=("transformer_blocks.0.attn.add_q_proj.weight",),
    ),
    ModelArch(
        arch="aura",
        keys_detect=[("double_layers.3.modX.1.weight",)],
        keys_banned=("joint_transformer_blocks.3.ff_context.out_projection.weight",),
    ),
    ModelArch(
        arch="hidream",
        keys_detect=[(
            "caption_projection.0.linear.weight",
            "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight",
        )],
        keys_hiprec=(".ff_i.gate.weight", "img_emb.emb_pos"),
    ),
    ModelArch(
        arch="cosmos",
        keys_detect=[(
            "blocks.0.mlp.layer1.weight",
            "blocks.0.adaln_modulation_cross_attn.1.weight",
        )],
        keys_hiprec=("pos_embedder",),
    ),
    ModelArch(
        arch="hyvid",
        keys_detect=[(
            "double_blocks.0.img_attn_proj.weight",
            "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight",
        )],
    ),
    ModelArch(
        arch="wan",
        keys_detect=[(
            "blocks.0.self_attn.norm_q.weight", "text_embedding.2.weight", "head.modulation",
        )],
        keys_hiprec=(".modulation",),
    ),
    ModelArch(
        arch="ltxv",
        keys_detect=[(
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "transformer_blocks.27.scale_shift_table",
            "caption_projection.linear_2.weight",
        )],
        keys_hiprec=("scale_shift_table",),
    ),
    ModelArch(
        arch="sdxl",
        keys_detect=[
            ("down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight"),
            (
                "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight",
                "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight",
            ),
            ("label_emb.0.0.weight",),
        ],
    ),
    ModelArch(
        arch="sd1",
        keys_detect=[
            ("down_blocks.0.downsamplers.0.conv.weight",),
            (
                "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight",
                "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight",
            ),
        ],
    ),
    ModelArch(
        arch="lumina2",
        keys_detect=[("cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight")],
    ),
]

SUPPORTED_ARCH_NAMES = [a.arch for a in MODEL_ARCHES]

# Legacy quant types with real pure-Python quantize() support in the gguf
# package. K-quants (Q4_K/Q5_K/Q6_K/...) are dequantize-only there and are
# deliberately left out - see the module docstring.
QUANT_TYPE_CHOICES = ["Q8_0", "Q5_1", "Q5_0", "Q4_1", "Q4_0", "F16", "BF16"]

# type_size / block_size in bytes-per-element, straight from gguf.GGML_QUANT_SIZES.
QUANT_BYTES_PER_ELEM = {
    "Q8_0": 34 / 32, "Q5_1": 24 / 32, "Q5_0": 22 / 32, "Q4_1": 20 / 32, "Q4_0": 18 / 32,
    "F16": 2.0, "BF16": 2.0, "F32": 4.0,
}


def detect_arch(keys: set[str]) -> ModelArch | None:
    for arch in MODEL_ARCHES:
        matched = any(all(key in keys for key in group) for group in arch.keys_detect)
        if matched:
            if any(key in keys for key in arch.keys_banned):
                continue  # e.g. diffusers-format Flux - looks similar but unsupported
            return arch
    return None


def _preset_lists(preset: str) -> tuple[list[str], list[str]]:
    if not preset or preset == "none":
        return [], []
    info = get_model_filters().get(preset, {})
    return list(info.get("exclude", []) or []), list(info.get("highprec", []) or [])


def _matches_any(name: str, keywords: tuple[str, ...] | list[str]) -> bool:
    return any(kw in name for kw in keywords)


@dataclass
class GGUFConvertStats:
    total: int = 0
    arch: str = ""
    quantized_count: int = 0
    f32_kept_count: int = 0
    skipped_high_dim_count: int = 0
    fallback_f16_count: int = 0


def _quant_worker_count() -> int:
    """Threads for parallel quantization. gguf-py's quantize is numpy, which
    releases the GIL on large array ops, so threads give real speedup without
    the process-spawn risks of a Gradio server. Bounded so a handful of
    in-flight f32 tensors can't exhaust RAM."""
    import os
    try:
        return max(1, min(4, int(os.environ.get("QUANT_GUI_QUANT_THREADS", "0")) or
                          (os.cpu_count() or 4) // 4))
    except ValueError:
        return 2


def _prepare_tensor(f, key: str, arch, qtype, exclude_kw, highprec_kw, exclude_re):
    """Load + quantize one tensor. Runs in a worker thread; returns
    (key, packed_array, qtype, kind) or None for the >4-dim skip."""
    import gguf
    import numpy as np
    import torch
    from gguf import quants

    tensor = f.get_tensor(key)
    if tensor.dim() > MAX_TENSOR_DIMS:
        return None

    n_dims = tensor.dim()
    n_params = tensor.numel()
    force_f32 = (
        n_dims == 1
        or n_params <= QUANTIZATION_THRESHOLD
        or _matches_any(key, arch.keys_hiprec)
        or _matches_any(key, exclude_kw)
        or _matches_any(key, highprec_kw)
        or (exclude_re is not None and exclude_re.search(key))
    )
    this_qtype = gguf.GGMLQuantizationType.F32 if force_f32 else qtype
    kind = "f32" if force_f32 else "quantized"

    # Match city96/ComfyUI-GGUF's own dtype handling exactly: bf16 and
    # fp8 have no native numpy dtype, so upcast before quantizing.
    if tensor.dtype == torch.bfloat16:
        data = tensor.to(torch.float32).numpy()
    elif tensor.dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
        data = tensor.to(torch.float16).numpy()
    else:
        data = tensor.numpy()
    try:
        if this_qtype == gguf.GGMLQuantizationType.Q8_0:
            # Optional Triton GPU path (opt-in via QUANT_GUI_GPU_QUANT=1);
            # silently falls back to gguf-py's numpy quantize when triton
            # or a CUDA device isn't available. Bit-exact either way.
            from . import gpu_quant
            try:
                packed = gpu_quant.quantize_q8_0(data)
            except ValueError:
                raise gguf.QuantError(f"shape {data.shape} not Q8_0-blockable")
        else:
            packed = quants.quantize(data, this_qtype)
    except (AttributeError, gguf.QuantError):
        # Shape isn't divisible by the quant type's block size (32) -
        # ctq's own presets fall back the same way for shape mismatches.
        this_qtype = gguf.GGMLQuantizationType.F16
        packed = quants.quantize(data, this_qtype)
        if not force_f32:
            kind = "fallback_f16"
    return key, packed, this_qtype, kind


def convert_to_gguf(
    input_path: str,
    output_path: str,
    quant_type: str,
    preset: str = "none",
    exclude_regex: str | None = None,
    progress_cb=None,
    control=None,
    checkpoint: Checkpoint | None = None,
):
    """Quantize `input_path` into a GGUF file at `output_path`.

    Pause/resume: `control` (run_control.RunControl) lets the UI pause
    between tensors or stop the run; `checkpoint` (checkpoints.Checkpoint)
    saves every finished tensor's packed array to disk so a stopped run
    resumes from the next tensor without re-quantizing - already-finished
    tensors are re-added to the writer straight from the checkpoint shards.
    (Tensors skipped for having >4 dims aren't checkpointed - they're
    deterministically re-skipped on resume.)
    """
    if not is_available():
        raise GGUFBackendError(f"gguf isn't installed. Install it with:\n  {install_hint()}")
    if quant_type not in QUANT_TYPE_CHOICES:
        raise GGUFBackendError(
            f"Unsupported GGUF quant type {quant_type!r}. This app can only produce "
            f"{', '.join(QUANT_TYPE_CHOICES)} - K-quants (Q4_K_M etc.) need a compiled "
            "llama-quantize binary, which this app doesn't build or invoke."
        )

    import gguf
    import numpy as np
    from gguf import quants
    from safetensors import safe_open

    with safe_open(input_path, framework="pt") as f:
        keys = list(f.keys())
        key_set = set(keys)
        arch = detect_arch(key_set)
        if arch is None:
            raise GGUFBackendError(
                "Unknown model architecture - GGUF export (via this app or ComfyUI-GGUF's own loader) only "
                f"recognizes: {', '.join(SUPPORTED_ARCH_NAMES)}. If this is a diffusers-format checkpoint, "
                "convert it to the reference/checkpoint key format first (e.g. ComfyUI's 'ModelSave' node)."
            )

        long_names = [k for k in keys if len(k) > MAX_TENSOR_NAME_LENGTH]
        if long_names:
            raise GGUFBackendError(
                f"{len(long_names)} tensor name(s) exceed GGUF's {MAX_TENSOR_NAME_LENGTH}-character limit "
                f"(e.g. {long_names[0]!r}) - this file can't be exported to GGUF."
            )

        exclude_kw, highprec_kw = _preset_lists(preset)
        exclude_re = re.compile(exclude_regex) if exclude_regex else None
        qtype = getattr(gguf.GGMLQuantizationType, quant_type)

        stats = GGUFConvertStats(total=len(keys), arch=arch.arch)
        if checkpoint is not None:
            set_total(checkpoint, stats.total)
        writer = gguf.GGUFWriter(path=None, arch=arch.arch)
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
        file_type = getattr(gguf.LlamaFileType, f"MOSTLY_{quant_type}", None)
        if file_type is not None:
            writer.add_file_type(file_type)

        # Resume: re-add every tensor the checkpoint already finished, from
        # its packed-array shard - no re-quantization. Indices with no shard
        # are the deterministically-skipped >4-dim tensors.
        start = checkpoint.next_index if checkpoint is not None else 0
        if start:
            for i in range(start):
                meta = checkpoint.tensors.get(str(i))
                if meta is None:
                    stats.skipped_high_dim_count += 1
                    continue
                packed = np.load(str(checkpoint.shard_path(i, ".npy")))
                writer.add_tensor(
                    meta["key"], packed, raw_dtype=gguf.GGMLQuantizationType[meta["qtype"]],
                )
                kind = meta.get("kind")
                if kind == "f32":
                    stats.f32_kept_count += 1
                elif kind == "fallback_f16":
                    stats.fallback_f16_count += 1
                else:
                    stats.quantized_count += 1

        # Parallel quantization: worker threads load+quantize up to `window`
        # tensors ahead while the main thread writes shards/checkpoints and
        # feeds the writer in strict key order. numpy releases the GIL on
        # large array ops, so threads scale; the writer/checkpoint state is
        # only touched by the main thread. Pause holds the writer (workers
        # may finish their in-flight window first); cancel abandons pending
        # futures without waiting for the window to drain.
        from concurrent.futures import ThreadPoolExecutor

        n_workers = _quant_worker_count()
        window = max(1, n_workers * 2)
        pending: dict[int, "Future"] = {}
        pool = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="gguf-quant")
        submit_i = start

        def _submit(idx: int) -> None:
            pending[idx] = pool.submit(
                _prepare_tensor, f, keys[idx], arch, qtype,
                exclude_kw, highprec_kw, exclude_re,
            )

        try:
            for i in range(start, len(keys)):
                while submit_i < len(keys) and len(pending) < window:
                    _submit(submit_i)
                    submit_i += 1

                if progress_cb:
                    progress_cb(i + 1, stats.total, keys[i])

                # Tensor-boundary cooperation point: a started tensor always
                # finishes, so checkpoint state stays consistent.
                if control is not None:
                    control.wait_if_paused()
                    control.raise_if_cancelled()

                result = pending.pop(i).result()
                if result is None:  # >4 dims: deterministically re-skipped
                    stats.skipped_high_dim_count += 1
                    continue
                key, packed, this_qtype, kind = result

                if kind == "f32":
                    stats.f32_kept_count += 1
                elif kind == "fallback_f16":
                    stats.fallback_f16_count += 1
                else:
                    stats.quantized_count += 1

                if checkpoint is not None:
                    # Shard first, manifest second: the manifest never
                    # references a shard that isn't fully written.
                    np.save(str(checkpoint.shard_path(i, ".npy")), packed)
                    record_tensor(checkpoint, i, key, qtype=this_qtype.name, kind=kind)

                writer.add_tensor(key, packed, raw_dtype=this_qtype)
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Crash-atomic: build the file at a .tmp path, rename into place.
        # GGUFWriter opens the path it's given on write_header_to_file, so
        # pointing it at the temp path is enough.
        tmp_path = str(output_path) + ".tmp"
        writer.write_header_to_file(path=tmp_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        os.replace(tmp_path, str(output_path))

    return stats


def stream_gguf_conversion(
    input_path: str,
    output_path: str,
    quant_type: str,
    preset: str = "none",
    exclude_regex: str | None = None,
    control=None,
    checkpoint: Checkpoint | None = None,
):
    """Generator wrapper mirroring int4_backend.stream_int4_conversion's
    interface: runs the (blocking) conversion in a background thread,
    yielding ("progress", current, total, key) while running, then exactly
    one of ("ok", stats) / ("cancelled", message) / ("fail", error_message).
    "cancelled" means the user hit Stop & save - checkpoint shards for every
    finished tensor are on disk and the run can be resumed."""
    import queue
    import threading

    q: "queue.Queue" = queue.Queue()
    SENTINEL = object()

    def progress_cb(current, total, key):
        q.put(("progress", current, total, key))

    def worker():
        try:
            stats = convert_to_gguf(
                input_path, output_path, quant_type,
                preset=preset, exclude_regex=exclude_regex, progress_cb=progress_cb,
                control=control, checkpoint=checkpoint,
            )
            q.put(("ok", stats))
        except RunCancelled:
            q.put(("cancelled", "Stopped by user - progress saved to the checkpoint."))
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
