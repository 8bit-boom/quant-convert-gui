# Repository Audit — 2026-09-24

Scope: full codebase (`app.py`, `quant_gui/`, `tools/`, `scripts/`, `tests/`,
`docs/`, git hygiene, docs, security, performance). Branch
`claude/clever-archimedes-95cggg` @ `d164b4c` (+ local hygiene fixes in this
commit). ~11,800 lines of Python across 40 tracked files.

## Verdict

**Healthy.** 199 tests pass (1 skip = optional `comfy-kitchen` dependency, by
design; 2 deselected = benchmark-marked). No `shell=True`, no `eval/exec`, no
bare `except:`, no hardcoded credentials, no unclosed files, compile-clean.
The pause/stop-&-save/resume design (tensor-boundary cooperation + atomic
manifest writes) is sound and correctly documented.

## What was fixed in this audit

1. **`.gitignore` gaps** — `checkpoints/` (can hold GB-scale shard payloads)
   and `tmp_compare/` (scratch) were untracked-but-unignored; also removed a
   duplicated `llama.cpp/` line and added `*.log`.
2. **Unused imports** — `field` from `dataclasses` in `gguf_backend.py` and
   `llamacpp_backend.py` (0 uses each).

## Findings (not blocking, by area)

### Security — clean
- All subprocess calls use list-form argv; zero `shell=True` / `os.system` /
  `eval` / `exec`.
- `np.load` on checkpoint shards uses default `allow_pickle=False` → not
  exposed to arbitrary-pickle execution.
- HF tokens are passed through parameters (`hf.py`, `app.py`), never hardcoded;
  nothing secret in tracked files.
- Atomic manifest replace (`tmp` + `os.replace`) in `checkpoints.py` is
  correct on both POSIX and Windows.

### Correctness — sound, two observations
- **Output-not-atomic**: `gguf_backend.py` writes the final GGUF directly to
  `output_path`. A cooperative Stop is safe (checkpoint shards are the source
  of truth), but a hard kill mid-`write_tensors_to_file` leaves a truncated
  GGUF next to valid checkpoint shards. Low risk; a temp-file + rename would
  close it.
- **Resume replay trusts manifest indices** (`gguf_backend.py:293`): correct
  by construction (shard written before `record_tensor`), and a stray shard
  without a manifest entry is harmlessly ignored. No action needed.

### Git hygiene
- **No `main` branch exists** — repo default (`origin/HEAD`) is
  `claude/clever-archimedes-95cggg`; `feature/pause-save-resume` is stale and
  superseded. Consistent with the earlier "only main with all updates" request
  only if you intend the claude branch as the permanent default; otherwise
  rename it to `main` on GitHub and delete the stale feature branch.
- Untracked working data is heavy: `converted/` **114 GB**, `gguf_bench_tmp/`
  **22 GB** (both gitignored). Disk F: is at **94%** (117 GB free) — cleanup
  of `gguf_bench_tmp/` alone frees 22 GB.
- No `LICENSE` file — worth adding if the repo is public.

### Performance — no red flags in this pass
- Quant hot path (`gguf.quants.quantize`) is numpy-vectorized; measured
  ~450 MB/s single-thread on this machine.
- `smart_quant.py` keeps Python loops out of the per-element math (loops over
  tensors/rungs only).
- Checkpoint shards double as resume cache — a resumed run skips
  re-quantization entirely by replaying packed `.npy` shards.

### Structure / maintainability
- `app.py` is 2,862 lines / 64 defs — a monolith. The split between
  `quant_gui/` (backends, reusable) and `app.py` (Gradio wiring) is already
  good; a future split of `app.py` per-tab would help but is not urgent.
- `tools/convert_krea2_to_gguf.py` duplicates a minimal safetensors reader
  intentionally (keeps the tool torch-free) — documented in its docstring.

### Docs
- `README.md` is comprehensive and matches implemented behavior (verified spot
  claims: mixed-precision regex, target-GPU table, Learned/AdaRound notes).
- `docs/KREA2.md`, `docs/OLLAMA.md` exist and are current. No doc references
  the audit itself (this file).

## Test coverage snapshot (count per suite)

| Suite | Tests | Notes |
|---|---|---|
| test_llamacpp_backend | 35 | largest backend |
| test_smart_quant | 27 | tuner rules |
| test_llm_models_listing | 20 | |
| test_gguf_unsloth3_bench | 18 | bench-marked, deselected by default |
| test_cli_builder | 17 | |
| test_runner | 13 | |
| test_run_history / loop_timing | 10 + 10 | |
| test_gguf_backend / int4_backend | 9 + 7 | |
| test_run_control | 7 | pause/cancel state machine |
| test_checkpoints | 7 | manifest round-trip |
| test_size_estimate | 6 | |
| test_krea2_convert | 6 | +1 e2e param → 7 runs |
| test_compare_gguf / filters | 5 + 4 | |
| test_llm_find_best / int4_resume / gguf_resume | 3 / 2 / 1 | thinnest area: resume e2e |

Thinnest coverage: GGUF resume path has 1 test; the end-to-end
cancel-then-resume flow for `gguf_backend` and `int4_backend` is proven by
targeted unit tests rather than a full model run. Acceptable given how slow a
real-model e2e would be, but the first place to add tests if resume bugs
appear.

## Recommended follow-ups (priority order)

1. Clean `gguf_bench_tmp/` (22 GB) — disk is at 94%.
2. Add `LICENSE`.
3. Decide the canonical branch name (`main` vs the claude branch) and delete
   `feature/pause-save-resume`.
4. (Optional) Write final GGUF to temp + rename for crash-atomic outputs.
5. (Optional) Split `app.py` per tab.
