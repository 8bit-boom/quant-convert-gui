# Krea 2 → GGUF conversion

Krea 2 (Raw / Turbo) is a 12.9B text-to-image **diffusion transformer**, not an
LLM — the llama.cpp `convert_hf_to_gguf` path does not apply. This repo ships a
dedicated converter that emits the GGUF format the ComfyUI-GGUF community uses
for image models:

- `tools/convert_krea2_to_gguf.py` — the converter (pure numpy + gguf-py, no torch)
- `tests/test_krea2_convert.py` — precision-rule, renaming, and round-trip tests

## What goes into the GGUF

Only the **DiT transformer**. The text encoder and VAE stay separate safetensors
loaded by ComfyUI:

| Component | File | Loaded via |
|---|---|---|
| DiT transformer | this GGUF | **Unet Loader (GGUF)** |
| Text encoder (Qwen3-VL-4B) | `qwen3vl_4b_fp8_scaled.safetensors` | CLIPLoader, type `krea2` |
| VAE (Qwen-Image) | `qwen_image_vae.safetensors` | VAELoader |

Tensor names are stored ComfyUI-native (`blocks.0.attn.wq.weight`, `txtfusion.*`,
`first.weight`, `last.*`) and `general.architecture = "krea2"`. That arch tag is
what gates the GGUF unet loader — stock city96/ComfyUI-GGUF will NOT load it;
you need a krea2-patched fork:

- RealRebelAI/ComfyUI-GGUF_KREA-2 (city96 + one-line `"krea2"` in `IMG_ARCH_LIST`), or
- molbal/ComfyUI-GGUF fork (has the public `ModelKrea2` converter this tool's
  precision rules follow).

## Converting

```bash
python tools/convert_krea2_to_gguf.py \
    --src krea2_turbo_bf16.safetensors \
    --dst krea2_turbo_q8_0.gguf \
    --quant q8_0
```

Sources:

1. **Best** — `Comfy-Org/Krea-2` on Hugging Face:
   `diffusion_models/krea2_turbo_bf16.safetensors` (Turbo) or
   `krea2_raw_bf16.safetensors` (Raw). Single file, ComfyUI-native keys, no
   renaming. The BF16 repackaging is required — raw `krea/Krea-2-*` repos are
   FP8, which this converter rejects.
2. HF diffusers shards (`krea/Krea-2-*/transformer/*.safetensors`) also work:
   they are merged and renamed through the built-in diffusers→ComfyUI table.

Memory-mapped streaming: RAM usage stays modest even for the 26 GB source.

### Quant options

| `--quant` | Result | Size (approx, 12.9B DiT) |
|---|---|---|
| `bf16` | lossless passthrough | ~26 GB |
| `f16` | float16 | ~26 GB (F32/F16 rules still apply per-tensor) |
| `q8_0` | per-block 8-bit, default | ~13 GB |
| `q5_1` / `q5_0` | 5-bit legacy | ~8–9 GB |
| `q4_1` / `q4_0` | 4-bit legacy | ~7 GB |

K-quants (Q4_K_M, IQ4_XS, …) are **not** single-stage: gguf-py cannot emit them
directly. Use the two-stage flow (this converter → BF16 GGUF → patched
`llama-quantize`) if you need K-quants.

### Per-tensor precision rules (community ModelKrea2 consensus)

1. 1-D tensors → F32 (norm scales, modulations, biases)
2. ≤ 1024 elements → F32
3. `first.` / `last.` / `tproj.` / `tmlp.` / `txtmlp.` / `txtfusion.projector.`
   prefixes → F32 (conditioning and output paths)
4. big BF16 2-D linears → the requested `--quant`
5. remaining F32 2-D → F16

## ComfyUI loading

Requires ComfyUI ≥ v0.25 plus a krea2-patched ComfyUI-GGUF fork.

1. Copy the GGUF into `models/diffusion_models/`
2. **Unet Loader (GGUF)** ← the GGUF
3. **CLIPLoader** ← `qwen3vl_4b_fp8_scaled.safetensors`, type `krea2`
4. **VAELoader** ← `qwen_image_vae.safetensors`

Sampling:

- **Turbo**: 8 steps, CFG 1.0, euler/simple, shift 1.15
- **Raw**: 20–52 steps, CFG 3–7

## Verification status

The converter is verified structurally: synthetic safetensors → GGUF → GGUFReader
round-trips pass (tensor names, shapes, GGML types, bit-exact BF16 passthrough).
ComfyUI inference itself is not exercised in this repo's tests — the first real
load happens on your ComfyUI install.
