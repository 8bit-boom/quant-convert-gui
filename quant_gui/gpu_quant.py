"""Optional GPU (Triton) Q8_0 quantization path, with a numpy fallback.

WHY THIS EXISTS
--------------
gguf-py's ``quants.quantize`` is pure numpy. It releases the GIL and we
already parallelize it across threads (gguf_backend._quant_worker_count),
but each block still runs on the CPU. A Triton kernel moves the per-block
scale + round onto the GPU and can be several times faster on large
tensors.

STATUS - READ BEFORE TRUSTING THE GPU PATH
------------------------------------------
The Triton kernel below is written to be bit-exact against gguf-py's
reference numpy path (same rounding via round-half-away-from-zero, same
zero-block handling), but it is **UNVERIFIED on real hardware in this
repo's CI** - the managed test environment has no triton and a CPU-only
torch. It only activates when ALL of the following hold:

  1. environment variable ``QUANT_GUI_GPU_QUANT=1`` is set (opt-in),
  2. ``triton`` imports successfully,
  3. a CUDA device is actually present.

Any failure at setup time falls back to the numpy path silently. The
numpy path is the default, is what CI exercises, and is byte-identical
to ``gguf.quants.quantize(..., GGMLQuantizationType.Q8_0)``.

Only Q8_0 is implemented: it's the highest-value target (default output
type) and the simplest block format. Other types keep using gguf-py.
"""
from __future__ import annotations

import os

GPU_QUANT_ENV = "QUANT_GUI_GPU_QUANT"
BLOCK = 32  # Q8_0 block width (elements)


def gpu_quant_enabled() -> bool:
    """Opt-in gate: env var must be set to a truthy value."""
    return os.environ.get(GPU_QUANT_ENV, "").strip().lower() in (
        "1", "true", "yes", "on")


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - any import failure -> fall back
        return False


def gpu_quant_ready() -> bool:
    """True only when the Triton path can actually run."""
    if not (gpu_quant_enabled() and triton_available()):
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def quantize_q8_0_numpy(data) -> bytes:
    """Bit-exact reimplementation of gguf-py's Q8_0 row quantization.

    ``data``: float32 numpy array whose last dim is a multiple of 32.
    Returns the packed uint8 array: per 32-element block, a little-endian
    f16 scale followed by 32 int8 quants.
    """
    import numpy as np
    from gguf.quants import np_roundf

    rows = np.ascontiguousarray(data, dtype=np.float32)
    shape = rows.shape
    blocks = rows.reshape(-1, BLOCK)
    d = np.abs(blocks).max(axis=1, keepdims=True) / np.float32(127.0)
    with np.errstate(divide="ignore"):
        id_ = np.where(d == 0, 0, 1.0 / d)
    qs = np_roundf(blocks * id_)
    d_u8 = d.astype(np.float16).view(np.uint8)
    qs_u8 = qs.astype(np.int8).view(np.uint8)
    return np.concatenate([d_u8, qs_u8], axis=1).reshape(shape[:-1] + (-1,))


def _quantize_q8_0_triton(data):
    """Triton kernel path. UNVERIFIED on hardware - see module docstring.

    Mirrors quantize_q8_0_numpy: one program per 32-wide block, computes
    amax -> f16 scale, round-half-away-from-zero for the quants.
    """
    import numpy as np
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _q8_0_kernel(x_ptr, out_ptr, n_blocks, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        amax = tl.max(tl.abs(x), axis=0)
        d = amax / 127.0
        d = tl.where(d == 0.0, 0.0, d)
        id_ = tl.where(d == 0.0, 0.0, 1.0 / d)
        # round half away from zero: sign(x) * floor(|x| + 0.5)
        q = x * id_
        # round half away from zero: sign(x) * floor(|x| + 0.5)
        q = tl.where(q >= 0, tl.floor(q + 0.5), -tl.floor(-q + 0.5))
        q = q.to(tl.int8)
        d_f16 = d.to(tl.float16)
        # out block layout: 2-byte f16 scale, then 32 int8 = 34 bytes.
        # scale at byte offset pid*34 (little-endian f16 bit pattern)
        scale_bits = d_f16.to(tl.uint16, bitcast=True)
        out_base = pid * 34
        tl.store(out_ptr + out_base + 2 + tl.arange(0, BLOCK),
                 q.to(tl.uint8, bitcast=True))
        # store the 2 scale bytes individually (little-endian)
        tl.store(out_ptr + out_base + 0, (scale_bits & 0xFF).to(tl.uint8))
        tl.store(out_ptr + out_base + 1, (scale_bits >> 8).to(tl.uint8))

    rows = np.ascontiguousarray(data, dtype=np.float32)
    shape = rows.shape
    n_blocks = rows.size // BLOCK
    x = torch.from_numpy(rows.reshape(-1)).cuda()
    out = torch.empty(n_blocks * 34, dtype=torch.uint8, device="cuda")
    _q8_0_kernel[(n_blocks,)](x, out, n_blocks, BLOCK=BLOCK)
    packed = out.cpu().numpy()
    return packed.reshape(shape[:-1] + (-1,))


def quantize_q8_0(data):
    """Dispatch: Triton kernel if ready, else bit-exact numpy fallback.

    Raises the same error gguf-py would on a shape that isn't a multiple
    of the 32-element block.
    """
    shape = getattr(data, "shape", None)
    if shape is None or shape[-1] % BLOCK != 0:
        raise ValueError(f"Q8_0 needs last dim % {BLOCK} == 0, got {shape}")
    if gpu_quant_ready():
        try:
            return _quantize_q8_0_triton(data)
        except Exception:  # noqa: BLE001 - never break a run on GPU trouble
            pass
    return quantize_q8_0_numpy(data)
