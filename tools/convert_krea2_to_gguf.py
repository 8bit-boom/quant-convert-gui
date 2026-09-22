"""Convert Krea 2 (Raw / Turbo) diffusion transformers to GGUF for ComfyUI.

Krea 2 is a 12.9B text-to-image diffusion transformer (NOT an LLM - the
llama.cpp convert_hf_to_gguf path does not apply). The GGUF produced here
follows the community convention used by molbal/krea2-gguf and
RealRebelAI/KREA-2_GGUFs:

* The GGUF contains ONLY the diffusion transformer. The Qwen3-VL-4B text
  encoder and the Qwen-Image VAE stay as separate safetensors loaded by
  ComfyUI (CLIPLoader type "krea2" / VAELoader).
* Tensor names are stored ComfyUI-native (`blocks.0.attn.wq.weight`, ...)
  and `general.architecture = "krea2"`, which is what gates the GGUF unet
  loader (RealRebelAI/ComfyUI-GGUF_KREA-2 or molbal/ComfyUI-GGUF fork).
* Precision rules (community consensus, see ModelKrea2 in molbal's fork):
    1. 1-D tensors -> F32 (norm scales, modulations, biases)
    2. <= 1024 elements -> F32
    3. key starts with first./last./tproj./tmlp./txtmlp./txtfusion.projector.
       -> F32 (conditioning / output paths)
    4. big BF16 2-D linears -> the requested quant (or BF16 passthrough)
    5. remaining F32 2-D -> F16
* gguf-py can emit F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 directly. K-quants
  (Q4_K_M ...) need the two-stage patched-llama-quantize flow instead.

Sources:
  * Best: Comfy-Org/Krea-2 `diffusion_models/krea2_{raw,turbo}_bf16.safetensors`
    (single file, ComfyUI-native key names, no renaming needed).
  * Also accepted: HF diffusers shards (krea/Krea-2-*/transformer/...), merged
    and renamed via the diffusers->ComfyUI table below.

Loading (ComfyUI >= v0.25 + a krea2-patched ComfyUI-GGUF fork):
  Unet Loader (GGUF) <- this file
  CLIPLoader <- qwen3vl_4b_fp8_scaled.safetensors, type "krea2"
  VAELoader  <- qwen_image_vae.safetensors
  Turbo: 8 steps, CFG 1.0, euler/simple, shift 1.15.

Usage:
    python tools/convert_krea2_to_gguf.py --src krea2_turbo_bf16.safetensors \
        --dst krea2_turbo_q8_0.gguf --quant q8_0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# minimal safetensors reader (numpy only - no torch dependency)
# ---------------------------------------------------------------------------

_ST_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


class SafetensorsFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            (header_len,) = np.frombuffer(f.read(8), dtype=np.uint64)
            self.header = json.loads(f.read(int(header_len)))
        self._data = np.memmap(self.path, dtype=np.uint8, mode="r")
        self.metadata = self.header.get("__metadata__", {})

    def keys(self) -> list[str]:
        return [k for k in self.header if k != "__metadata__"]

    def tensor_info(self, key: str) -> dict:
        return self.header[key]

    def load(self, key: str) -> tuple[np.ndarray, str]:
        """Return (numpy array, source dtype string). BF16 comes back as
        float32 (lossless widening) with source tag 'BF16'."""
        info = self.header[key]
        dt = info["dtype"]
        if dt == "BF16":
            raw = self._slice(info).view(np.uint16)
            return bf16_to_f32(raw).reshape(info["shape"]), "BF16"
        if dt == "F8_E4M3":
            raise ValueError(
                f"{key}: FP8 source is not supported by this converter - use the "
                "BF16 repackaging (Comfy-Org/Krea-2 diffusion_models/*_bf16.safetensors)"
            )
        np_dt = _ST_DTYPES[dt]
        return self._slice(info).view(np_dt).reshape(info["shape"]), dt

    def _slice(self, info: dict) -> np.ndarray:
        b0, b1 = info["data_offsets"]
        base = 8 + self._header_len
        return self._data[base + b0: base + b1]

    @property
    def _header_len(self) -> int:
        if not hasattr(self, "_hl"):
            with open(self.path, "rb") as f:
                (self._hl,) = np.frombuffer(f.read(8), dtype=np.uint64)
        return int(self._hl)


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------------------
# diffusers (HF) -> ComfyUI key renaming
# ---------------------------------------------------------------------------

_RENAME_EXACT = {
    "img_in.weight": "first.weight",
    "img_in.bias": "first.bias",
    "time_embed.linear_1.weight": "tmlp.0.weight",
    "time_embed.linear_1.bias": "tmlp.0.bias",
    "time_embed.linear_2.weight": "tmlp.2.weight",
    "time_embed.linear_2.bias": "tmlp.2.bias",
    "time_mod_proj.weight": "tproj.1.weight",
    "time_mod_proj.bias": "tproj.1.bias",
    "txt_in.norm.weight": "txtmlp.0.scale",
    "txt_in.linear_1.weight": "txtmlp.1.weight",
    "txt_in.linear_1.bias": "txtmlp.1.bias",
    "txt_in.linear_2.weight": "txtmlp.3.weight",
    "txt_in.linear_2.bias": "txtmlp.3.bias",
    "final_layer.norm.weight": "last.norm.scale",
    "final_layer.linear.weight": "last.linear.weight",
    "final_layer.linear.bias": "last.linear.bias",
}

_RENAME_PATTERNS = [
    (re.compile(r"^text_fusion\."), "txtfusion."),
    (re.compile(r"^transformer_blocks\.(\d+)\."), r"blocks.\1."),
    (re.compile(r"\.norm1\.weight$"), ".prenorm.scale"),
    (re.compile(r"\.norm2\.weight$"), ".postnorm.scale"),
    (re.compile(r"\.attn\.to_q\.weight$"), ".attn.wq.weight"),
    (re.compile(r"\.attn\.to_k\.weight$"), ".attn.wk.weight"),
    (re.compile(r"\.attn\.to_v\.weight$"), ".attn.wv.weight"),
    (re.compile(r"\.attn\.to_gate\.weight$"), ".attn.gate.weight"),
    (re.compile(r"\.attn\.to_out\.0\.weight$"), ".attn.wo.weight"),
    (re.compile(r"\.attn\.norm_q\.weight$"), ".attn.qknorm.qnorm.scale"),
    (re.compile(r"\.attn\.norm_k\.weight$"), ".attn.qknorm.knorm.scale"),
    (re.compile(r"\.ff\."), ".mlp."),
]


def rename_diffusers_key(key: str) -> str:
    if key in _RENAME_EXACT:
        return _RENAME_EXACT[key]
    if key == "final_layer.scale_shift_table":
        return "last.modulation.lin"  # (2, hidden) kept as-is
    m = re.match(r"^transformer_blocks\.(\d+)\.scale_shift_table$", key)
    if m:
        return f"blocks.{m.group(1)}.mod.lin"  # flattened to 1-D downstream
    new = key
    for pat, rep in _RENAME_PATTERNS:
        new = pat.sub(rep, new)
    return new


def looks_like_diffusers(keys: list[str]) -> bool:
    return any(k.startswith(("transformer_blocks.", "img_in.", "time_embed.")) for k in keys)


# ---------------------------------------------------------------------------
# per-tensor GGML type selection (community ModelKrea2 rules)
# ---------------------------------------------------------------------------

HIPREC_PREFIXES = ("first.", "last.", "tproj.", "tmlp.", "txtmlp.", "txtfusion.projector.")
QUANT_THRESHOLD_ELEMS = 1024


def pick_qtype(key: str, shape: tuple[int, ...], src_dtype: str,
               quant: str | None) -> "object":
    """Return the GGMLQuantizationType for a tensor. `quant` (e.g. 'q8_0')
    is applied to big 2-D BF16 linears only; everything else follows the
    precision rules."""
    from gguf import GGMLQuantizationType as QT

    n_elems = int(np.prod(shape)) if shape else 0
    if len(shape) <= 1:
        return QT.F32
    if n_elems <= QUANT_THRESHOLD_ELEMS:
        return QT.F32
    if key.startswith(HIPREC_PREFIXES):
        return QT.F32
    if (quant and src_dtype == "BF16" and len(shape) == 2
            and shape[-1] % 32 == 0 and n_elems > QUANT_THRESHOLD_ELEMS):
        return QT[quant.upper()]
    if src_dtype == "BF16":
        return QT.BF16
    return QT.F16


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------

def convert(src: Path, dst: Path, quant: str | None, arch: str = "krea2",
            verbose: bool = False) -> dict:
    import gguf
    from gguf import GGUFWriter

    LlamaFileType = getattr(gguf, "GGMLFileType", None) or gguf.LlamaFileType

    st = SafetensorsFile(src)
    keys = st.keys()
    diffusers = looks_like_diffusers(keys)
    if diffusers:
        print("source looks like HF diffusers naming - applying rename map")
    if not any(k.endswith("attn.wq.weight") or "transformer_blocks.0.attn.to_q" in k
               for k in keys):
        raise SystemExit("not a Krea 2 transformer checkpoint - no attn.wq/to_q keys found")

    file_type = {
        None: LlamaFileType.MOSTLY_BF16,
        "bf16": LlamaFileType.MOSTLY_BF16,
        "f16": LlamaFileType.MOSTLY_F16,
        "q8_0": LlamaFileType.MOSTLY_Q8_0,
        "q5_1": LlamaFileType.MOSTLY_Q5_1,
        "q5_0": LlamaFileType.MOSTLY_Q5_0,
        "q4_1": LlamaFileType.MOSTLY_Q4_1,
        "q4_0": LlamaFileType.MOSTLY_Q4_0,
    }[quant]

    writer = GGUFWriter(path=None, arch=arch)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(file_type)
    cfg = st.metadata.get("config")
    if cfg:
        writer.add_string("config", cfg)

    stats: dict[str, int] = {}
    t0 = time.time()
    for i, key in enumerate(sorted(keys)):
        data, src_dt = st.load(key)
        name = rename_diffusers_key(key) if diffusers else key
        if name.endswith(".mod.lin") and data.ndim == 2:
            data = data.reshape(-1)  # scale_shift_table (6, h) -> [6*h]
        qtype = pick_qtype(name, tuple(data.shape), src_dt, quant)
        tag = qtype.name
        packed = gguf.quants.quantize(
            data.astype(np.float32, copy=False)
            if qtype.name not in ("F32", "F16") else data,
            qtype,
        )
        # For quantized targets the writer expects the PACKED byte shape
        # (it converts byte shape -> element shape via raw_dtype). For
        # F32/F16 targets `packed` is the float array itself and the shape
        # passes through unchanged.
        writer.add_tensor(name, packed, raw_dtype=qtype)
        stats[tag] = stats.get(tag, 0) + 1
        if verbose or (i + 1) % 50 == 0 or i == len(keys) - 1:
            print(f"[{time.time() - t0:6.1f}s] {i + 1}/{len(keys)} {name} {src_dt}->{tag}", flush=True)

    writer.write_header_to_file(path=str(dst))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    print(f"done: {dst} ({dst.stat().st_size / 1e9:.2f} GB, {len(keys)} tensors)")
    print("type histogram:", dict(sorted(stats.items())))
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="source safetensors")
    ap.add_argument("--dst", type=Path, required=True, help="output .gguf")
    ap.add_argument("--quant", default="q8_0", choices=["bf16", "f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"],
                    help="quant for big BF16 2-D linears (default q8_0; 'bf16' = lossless passthrough)")
    ap.add_argument("--arch", default="krea2",
                    help="architecture tag; use 'qwen_image' only if targeting stock city96 nodes")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    if not args.src.is_file():
        print(f"not found: {args.src}", file=sys.stderr)
        return 1
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    convert(args.src, args.dst, None if args.quant == "bf16" else args.quant,
            arch=args.arch, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
