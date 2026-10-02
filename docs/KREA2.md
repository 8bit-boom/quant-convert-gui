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

### Two-stage K-quant flow with a sensitivity plan

Stock `llama-quantize` refuses the krea2 arch; community forks
(RealRebelAI/molbal) ship a krea2-patched `llama-quantize` that accepts
`--tensor-type-file` for per-tensor K-quant types. The missing input is
activation stats — DiTs have no `llama-imatrix` — so this repo provides a
sensitivity-proxy planner:

```bash
python tools/krea2_kquant_plan.py model_q8_0.gguf --target-bpw 4.0 \
    -o model.tensor-types.txt
llama-quantize --tensor-type-file model.tensor-types.txt \
    model_bf16.gguf out-Q4_K.gguf Q4_K
```

What the planner does, per plannable 2-D tensor (skipping 1-D, tiny, and
hi-precision conditioning/output prefixes):

1. decodes the tensor to F32 (BF16/F16/F32 pass through; legacy-quant
   sources are dequantized — Q8_0 sources work and are near-lossless),
2. scores reconstruction error under Q4_0 vs Q5_0 proxy quantization on a
   row subsample (≤ 4096 rows, so a 14 GB DiT scores in ~3 minutes),
3. ranks tensors by the error gap (how much the tensor suffers when
   squeezed) and tiers them onto a Q6_K → Q5_K → Q4_K → Q3_K ladder,
   shifting the cut points with `--target-bpw` (3.0–6.0).

Caveat: this is an *uncalibrated* proxy — no activation weighting — so the
plan is a starting template to hill-climb from, not a proven optimum. Treat
the output as "better than one-size-fits-all Q4_K", not "optimal".

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
