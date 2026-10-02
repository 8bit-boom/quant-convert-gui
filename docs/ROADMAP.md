# Feature Roadmap — quant-convert-gui

Prioritized, grounded in the 2026-09-24 audit (`docs/AUDIT.md`). User context:
RTX 5090 (32 GB), Windows, Ollama for serving, ComfyUI for image models.
Effort: S < 1 day, M 1–3 days, L > 3 days. Impact: what it buys.

---

## Tier 1 — quick wins (do first)

### 1.1 Repo hygiene sprint (S, credibility)
- Add `LICENSE` (repo is public, no license = all rights reserved by default).
- Rename default branch to `main` on GitHub, delete stale
  `feature/pause-save-resume`.
- **GitHub Actions CI**: run `pytest -m "not bench"` on push/PR (Windows +
  Ubuntu). Catches breakage on the exact platforms users run. The suite is
  already fast (~7 s) and dependency-light.

### 1.2 Crash-atomic outputs (S, robustness)
Final GGUF is written straight to `output_path`; a hard kill mid-write leaves a
truncated file. Write to `<output>.tmp` then `os.replace`. From audit finding;
touches `gguf_backend.py` + `int4_backend.py` finalize paths.

### 1.3 GGUF inspector tab (S, high utility)
New small tab: pick any GGUF → arch, file type, tensor count, per-type size
histogram, bpw, biggest tensors, metadata KV. Pure `GGUFReader` — most of the
code already exists across `tools/compare_gguf.py` and the Krea-2 verifier.
Answers "what is this file / will it fit my GPU" in seconds without loading
the model anywhere.

### 1.4 Wire existing CLI tools into the GUI (S)
`tools/compare_gguf.py`, `tools/ollama_modelfile.py`,
`tools/convert_krea2_to_gguf.py` are built, tested, documented — but invisible
to anyone who only opens the app. Add a **Tools** tab with three buttons that
shell out with captured logs + file pickers. Near-zero new logic, big
discoverability win.

---

## Tier 2 — workflow & quality

### 2.1 Image-model → GGUF tab (M)
Promote the Krea-2 converter from CLI tool to a first-class tab: source picker
(BF16 safetensors or diffusers), quant dropdown, companion-file downloader
(Qwen3-VL text encoder + Qwen-Image VAE), ComfyUI recipe hint. The converter
already handles diffusers renaming, so generalizing beyond Krea-2 (Qwen-Image,
SD3.5-class DiTs) is mostly a rename-table expansion.

### 2.2 One-click smart pipeline (M, quality flagship)
Today: user runs sweep → reads output → clicks "Tune winner" → optionally runs
perplexity validation. Pipeline idea: **"Auto (recommended)"** button that
chains imatrix → stage-1 scoring → stage-2 knapsack → llama-quantize →
perplexity check vs baseline, resumable at each stage (checkpoint machinery
already exists). The pieces are all built; this is orchestration + UI.

### 2.3 Batch queue (M)
Queue N conversions with shared or per-item settings; runs sequentially;
pause/resume applies to the queue; overnight batch is the target use case.
Checkpoint dirs already store full params per run — a queue is largely a
scheduler over that.

### 2.4 MoE expert report in UI (S–M)
`moe_expert_report()` exists in `smart_quant.py` but only reaches logs.
Surface top-sensitive experts + per-expert assignment table in the LLM tab
after tuning. Gemma 4 / Qwen-MoE users get visible insight into where the
quality budget went.

### 2.5 Structured run history + compare table (M)
`run_history.json` exists; extend it with outcome metrics (output size, bpw,
ppl if measured, duration, throughput) and render a sortable history table
with "compare selected runs" that diffs type histograms. Turns one-off
conversions into an auditable tuning workflow.

---

## Tier 3 — speed & capability

### 3.1 Multiprocess quantization (M, 2–4× on big models)
`gguf_backend` quantizes single-threaded numpy (~450 MB/s measured). Big
tensors are independent → `ProcessPoolExecutor` over tensors (each worker
re-mmaps the source; packed shards are the IPC, which doubles as the
checkpoint format). Watch: memory ceiling when N workers each hold a 400 MB
f32 tensor — cap the pool by tensor size.

### 3.2 GPU-side quant kernels (L, 10×+ potential, higher risk)
Q8_0/Q4_0 etc. are embarrassingly parallel per 32-value block — a small
Triton/CUDA kernel on the RTX 5090 could crush the numpy path. User already
tried triton-windows install; a single Q8_0 kernel is a contained first step.
Keep numpy fallback; make GPU quant an opt-in toggle like ctq's.

### 3.3 K-quants for image DiT GGUF (M)
Documented gap: the Krea-2 converter emits legacy quants only. Two-stage
flow (BF16 GGUF → patched `llama-quantize` with tensor-type file) would give
Q4_K_M/IQ4_XS-class image models. The smart-quant tuner machinery
(tensor-type file + budget knapsack) transfers directly — imatrix for a DiT
is the missing piece (activation stats need a ComfyUI calibration pass).

### 3.4 KV-cache + serving hints (S)
Ollama modelfile generator could also emit `OLLAMA_KV_CACHE_TYPE` and
context-size recommendations from the detected arch + GPU VRAM — cheap,
uses data already in the app.

---

## Explicit non-goals (for now)
- AWQ/GPTQ export — different ecosystem, ctq doesn't produce them either.
- Distributed/multi-GPU conversion — single-workstation tool by design.
- Phone/edge formats (CoreML/ONNX) — outside the GGUF/safetensors focus.

---

## Suggested order
1.1 → 1.2 → 1.3 → 1.4 → 2.1 → 2.2 → 2.4 → 3.1 → (2.3 / 2.5 by appetite) → 3.2/3.3
