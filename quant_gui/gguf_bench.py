"""Sweep llama.cpp quant settings at a target bits-per-weight and rank them.

This is the test harness behind "which setting converts best into GGUF" for
Unsloth-style "Dynamic 3.0" workflows: the ~3.0 bpw family (Q3_K_*, IQ3_*)
produced by ``llama-quantize --imatrix`` on an F16/BF16 reference GGUF.
Unsloth's published per-model layer mixes aren't consumable here, so the
sweep measures what IS reproducible locally: for every candidate quant type,
with and without the imatrix calibration file, it records

  * output size (bytes and effective bits-per-weight),
  * quantization wall time,
  * reconstruction error vs the F16 reference (parameter-weighted RMS of
    per-tensor relative L2 errors — robust against huge constant tables
    like rope frequencies, which a naive global L2 lets swamp everything)

`best_settings()` then reports the quality winner, the size winner, and a
"best value" pick (lowest error-per-byte) so the choice is explicit instead
of folklore.

The sweep auto-tunes per model (tune_sweep): big-vocab families (Gemma 4's
262k, Qwen 3.5-3.8's 151-256k vocab) get an error budget scaled past their
embedding and Q8_0 token-embedding variants — without those, the sample
degenerates to the unquantized embedding and every variant ties.

Requires the real toolchain (see quant_gui/llamacpp_backend.py):
convert_hf_to_gguf.py + llama-imatrix + llama-quantize. Everything degrades
to a clear error/skip rather than a fake measurement when a piece is missing.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

# The ~3.0 bits-per-weight candidates, quality-descending as usually
# published: K-quants work with or without an imatrix; IQ-quants are
# imatrix-native (llama-quantize falls back to a default importance table
# without one, which is what Unsloth's recipe never does).
DYNAMIC3_CANDIDATES = ["Q3_K_L", "Q3_K_M", "Q3_K_S", "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS"]

# Candidate sets per target bits-per-weight class (weight-space bpw of the
# quant types, not model-level - embeddings shift the latter). Keys are the
# labels the GUI shows; every entry must be a quant type a real
# llama-quantize supports (mirrors llamacpp_backend.QUANT_TYPE_CHOICES).
BPP_FAMILY_CANDIDATES = {
    "~2 bpw": ["Q3_K_S", "Q2_K", "Q2_K_S", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS"],
    "~3 bpw": DYNAMIC3_CANDIDATES,
    "~4 bpw": ["Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "IQ4_NL", "IQ4_XS"],
    "~5 bpw": ["Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0", "Q6_K"],
}
DEFAULT_BPP_TARGET = "~3 bpw"


def family_candidates(target: str) -> list[str]:
    """Candidate list for a target-size label; unknown labels fall back to
    the ~3bpw family so a stale GUI value never empties the sweep."""
    return list(BPP_FAMILY_CANDIDATES.get(target or "", DYNAMIC3_CANDIDATES))

# Families with huge vocabularies (Gemma's 262k, Qwen's 151-256k) carry
# token embeddings worth a large share of total params, which llama-quantize
# otherwise keeps F16. `--token-embedding-type Q8_0` is the standard lever
# for those models (llama.cpp QAT GGUFs do the same) and is swept as an
# extra variant whenever the embedding dominates the file.
BIG_EMBED_FRACTION = 0.15


@dataclass
class BenchRow:
    quant: str
    imatrix: bool
    size_bytes: int = 0
    seconds: float = 0.0
    error: float | None = None  # parameter-weighted RMS of per-tensor rel L2 vs F16 ref
    bpw: float = 0.0  # bits per weight, from tensor metadata
    measured_params: int = 0  # params covered by the error sample (0 = all)
    emb_q8: bool = False  # --token-embedding-type Q8_0 variant
    failed: str | None = None

    @property
    def label(self) -> str:
        s = f"{self.quant}{' + imatrix' if self.imatrix else ''}"
        return f"{s} + embQ8" if self.emb_q8 else s


@dataclass
class SweepResult:
    rows: list[BenchRow] = field(default_factory=list)
    ref_gguf: str = ""
    imatrix_file: str = ""
    n_params: int = 0

    def ok_rows(self) -> list[BenchRow]:
        return [r for r in self.rows if r.failed is None]


# ---------------------------------------------------------------- error metric


def _dequantized_tensors(gguf_path: str | Path, names: set[str] | None = None) -> dict[str, "object"]:
    """name -> float32 numpy tensor, dequantized via the gguf package.

    ``names`` restricts which tensors are converted (error sampling).
    """
    import numpy as np
    from gguf import GGUFReader
    from gguf.quants import dequantize

    out = {}
    reader = GGUFReader(str(gguf_path))
    for t in reader.tensors:
        if names is not None and t.name not in names:
            continue
        arr = np.ascontiguousarray(t.data)
        out[t.name] = dequantize(arr, t.tensor_type).astype(np.float32)
    return out


def sampled_tensor_names(ref_path: str | Path, quant_path: str | Path, budget_params: int) -> tuple[set[str], int]:
    """Deterministic error-sampling set: the largest shared tensors whose
    combined parameter count fits ``budget_params``. Returns (names, total).

    Whole tensors only (per-tensor error can't be stitched from partial
    blocks), largest-first, so the aggregate is dominated by exactly the
    tensors that dominate the model - and the result is reproducible.
    """
    from gguf import GGUFReader

    ref_names = {t.name for t in GGUFReader(str(ref_path)).tensors}
    shared = [t for t in GGUFReader(str(quant_path)).tensors if t.name in ref_names]
    shared.sort(key=lambda t: -t.n_elements)
    names, total = set(), 0
    for t in shared:
        if total >= budget_params:
            break
        names.add(t.name)
        total += t.n_elements
    return names, total


def relative_error(ref_map: dict, quant_map: dict) -> float:
    """Parameter-weighted RMS of per-tensor relative L2 errors.

    Each tensor contributes its own ||q-r||/||r|| weighted by its parameter
    count. A single global ||Q-R||/||R|| would let numerically-huge but
    weight-irrelevant constant tables (Gemma's 1M-context rope_freqs holds
    values up to 1e30) swamp the denominator and report ~0 error for
    everything; per-tensor normalization keeps the metric about weights.
    Accumulated in float64 — float32 squares overflow on such tables.
    """
    import numpy as np

    shared = set(ref_map) & set(quant_map)
    if not shared:
        raise ValueError("no shared tensor names between reference and quantized file")
    num = 0.0
    den = 0.0
    for name in shared:
        r = ref_map[name].astype(np.float64).ravel()
        q = quant_map[name].astype(np.float64).ravel()
        if q.shape != r.shape:
            raise ValueError(f"shape mismatch on {name!r}: {q.shape} vs {r.shape}")
        d = float(np.square(r).sum())
        if d <= 0:
            continue
        num += r.size * (float(np.square(q - r).sum()) / d)
        den += r.size
    return (num / den) ** 0.5 if den > 0 else 0.0


def _bits_per_weight(gguf_path: str | Path) -> tuple[float, int]:
    """(bits per weight, parameter count) from GGUF tensor metadata."""
    from gguf import GGUFReader
    from gguf.constants import GGML_QUANT_SIZES

    reader = GGUFReader(str(gguf_path))
    total_bits = 0
    n_params = 0
    for t in reader.tensors:
        block_size, type_size = GGML_QUANT_SIZES[t.tensor_type]  # (elems/block, bytes/block)
        n_blocks = t.n_elements // block_size
        total_bits += n_blocks * type_size * 8
        n_params += t.n_elements
    return (total_bits / n_params if n_params else 0.0), n_params


def model_info(gguf_path: str | Path) -> dict:
    """Architecture + size profile used to tune the sweep per model family."""
    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path))
    arch = ""
    field = reader.fields.get("general.architecture")
    if field is not None and field.parts:
        try:  # parts[-1] is the value (older gguf-py splits type/key/len too)
            last = field.parts[-1]
            arch = bytes(last).decode("utf-8", "replace").strip("\x00")
        except (TypeError, UnicodeDecodeError):
            arch = ""
    tensors = list(reader.tensors)
    n_params = sum(t.n_elements for t in tensors)
    largest = max((t.n_elements for t in tensors), default=0)
    largest_name = next((t.name for t in tensors if t.n_elements == largest), "")
    return {
        "arch": arch,
        "n_params": n_params,
        "largest_tensor_params": largest,
        "largest_tensor_name": largest_name,
        "embed_fraction": (largest / n_params) if n_params else 0.0,
    }


def tune_sweep(info: dict) -> dict:
    """Per-family sweep tuning, from model_info() output.

    * error_budget: never below 2x the largest tensor — big-vocab models
      (Gemma 262k / Qwen 151-256k vocab) have embeddings of several hundred
      M params; a smaller budget samples only the (F16, unquantized)
      embedding and every variant measures identically.
    * emb_q8_variants: when one tensor (the embedding) holds a large share
      of params, add --token-embedding-type Q8_0 variants — the standard
      size lever for exactly these families.
    """
    budget = max(300_000_000, 2 * int(info.get("largest_tensor_params") or 0))
    big_embed = (info.get("embed_fraction") or 0.0) >= BIG_EMBED_FRACTION
    return {
        "family": info.get("arch") or "unknown",
        "error_budget": budget,
        "emb_q8_variants": big_embed,
    }


# -------------------------------------------------------------------- sweep


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
    return proc.returncode, tail


def bench_quantize(
    quantize_bin: str | Path,
    ref_gguf: str | Path,
    out_path: str | Path,
    quant: str,
    imatrix_file: str | Path | None = None,
    error_budget_params: int | None = None,
    emb_q8: bool = False,
) -> BenchRow:
    """Quantize once and measure; returns a BenchRow (``failed`` set on error).

    ``error_budget_params`` caps error measurement at the largest shared
    tensors up to that many parameters (deterministic) — full-model
    dequantization in pure numpy takes ~20s per 1.7B-param variant, so a
    budget keeps sweeps practical while the biggest tensors (which dominate
    the aggregate) are always measured. None = measure everything.
    ``emb_q8`` adds llama-quantize's ``--token-embedding-type Q8_0`` (the
    standard size lever for big-vocab models like Gemma/Qwen).
    """
    row = BenchRow(quant=quant, imatrix=bool(imatrix_file), emb_q8=emb_q8)
    cmd = [str(quantize_bin)]
    if imatrix_file:
        cmd += ["--imatrix", str(imatrix_file)]
    if emb_q8:
        cmd += ["--token-embedding-type", "Q8_0"]
    cmd += [str(ref_gguf), str(out_path), quant]
    t0 = time.perf_counter()
    code, tail = _run(cmd)
    row.seconds = time.perf_counter() - t0
    if code != 0:
        row.failed = f"llama-quantize exited {code}: {tail}"
        return row
    size = Path(out_path).stat().st_size
    if size <= 0:
        row.failed = "quantized output is empty"
        return row
    row.size_bytes = size
    try:
        row.bpw, _ = _bits_per_weight(out_path)
    except Exception:  # noqa: BLE001 - size/time still valid without bpw
        row.bpw = 0.0
    try:
        if error_budget_params:
            names, row.measured_params = sampled_tensor_names(
                ref_gguf, out_path, error_budget_params
            )
        else:
            names = None
        row.error = relative_error(
            _dequantized_tensors(ref_gguf, names), _dequantized_tensors(out_path, names)
        )
    except Exception as exc:  # noqa: BLE001
        row.failed = f"error measurement failed: {exc}"
    return row


def run_sweep(
    *,
    ref_gguf: str | Path,
    imatrix_file: str | Path | None,
    out_dir: str | Path,
    quantize_bin: str | Path,
    quants: list[str] | None = None,
    with_imatrix: bool = True,
    without_imatrix: bool = True,
    keep_outputs: bool = False,
    error_budget_params: int | None = 300_000_000,
    emb_q8_variants: bool | None = None,
    log=print,
) -> SweepResult:
    """Quantize `ref_gguf` with every (quant × imatrix [× embQ8]) combination.

    Auto-tunes from the model itself (see tune_sweep): the error budget
    never drops below 2x the largest tensor — big-vocab models (Gemma 262k,
    Qwen 151-256k vocab) have embeddings of several hundred M params, and a
    smaller budget samples only the (F16, unquantized) embedding, making
    every variant measure identically. When one tensor (the embedding)
    holds >= BIG_EMBED_FRACTION of params, embQ8 variants are added and the
    sweep restricts to imatrix runs (imatrix-free IQ quants are degraded or
    fail anyway, and the variant matrix stays tractable).

    `log` receives human-readable progress lines. Output files are deleted
    after measurement unless `keep_outputs`.
    """
    ref_gguf = Path(ref_gguf)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = SweepResult(
        rows=[], ref_gguf=str(ref_gguf), imatrix_file=str(imatrix_file or "")
    )
    _, result.n_params = _bits_per_weight(ref_gguf)

    info = model_info(ref_gguf)
    tune = tune_sweep(info)
    if error_budget_params is not None:
        error_budget_params = max(error_budget_params, tune["error_budget"])
    if emb_q8_variants is None:
        emb_q8_variants = tune["emb_q8_variants"]
    log(
        f"[bench] model: arch={tune['family']}, {info['n_params']/1e9:.2f}B params, "
        f"largest tensor {info['largest_tensor_name'] or '?'} "
        f"({info['largest_tensor_params']/1e6:.0f}M, "
        f"{info['embed_fraction']*100:.0f}% of params)"
        + (", adding Q8_0 token-embedding variants" if emb_q8_variants else "")
    )

    # Variant matrix: big-embedding models get (imatrix × embQ8 on/off);
    # others get (imatrix on/off), embQ8 off.
    if emb_q8_variants:
        imatrix_variants = [True] if imatrix_file else [False]
    else:
        imatrix_variants = []
        if with_imatrix and imatrix_file:
            imatrix_variants.append(True)
        if without_imatrix:
            imatrix_variants.append(False)
    emb_variants = [False, True] if emb_q8_variants else [False]

    for quant in (quants or DYNAMIC3_CANDIDATES):
        for use_imatrix in imatrix_variants:
            for use_emb_q8 in emb_variants:
                out_path = out_dir / (
                    f"bench-{quant}{'-imatrix' if use_imatrix else ''}"
                    f"{'-embq8' if use_emb_q8 else ''}.gguf"
                )
                log(f"[bench] {quant}{' + imatrix' if use_imatrix else ''}"
                    f"{' + embQ8' if use_emb_q8 else ''} ...")
                row = bench_quantize(
                    quantize_bin, ref_gguf, out_path, quant,
                    imatrix_file if use_imatrix else None,
                    error_budget_params=error_budget_params,
                    emb_q8=use_emb_q8,
                )
                if row.failed:
                    log(f"[bench] {row.label}: FAILED — {row.failed.splitlines()[-1] if row.failed else ''}")
                else:
                    sample = (
                        f", ~{row.measured_params/1e6:.0f}M params sampled"
                        if row.measured_params
                        else ""
                    )
                    log(
                        f"[bench] {row.label}: {row.size_bytes/1e6:.1f} MB, "
                        f"{row.bpw:.2f} bpw, {row.seconds:.1f}s, error {row.error:.3e}{sample}"
                    )
                result.rows.append(row)
                if not keep_outputs and out_path.exists():
                    out_path.unlink()
    return result


def best_settings(result: SweepResult) -> dict[str, BenchRow]:
    """Pick winners across the quality/size/value views.

    * best_quality — lowest reconstruction error (regardless of size).
    * smallest     — smallest file (regardless of error).
    * best_value   — lowest error-per-byte: the usual "Dynamic" sweet spot,
                     near-max quality at the target size class.

    Raises ValueError when no row succeeded.
    """
    rows = result.ok_rows()
    if not rows:
        raise ValueError("no successful quantize runs in the sweep")
    best_quality = min(rows, key=lambda r: (r.error if r.error is not None else float("inf")))
    smallest = min(rows, key=lambda r: r.size_bytes)
    best_value = min(rows, key=lambda r: (r.error or float("inf")) / max(1, r.size_bytes))
    return {"best_quality": best_quality, "smallest": smallest, "best_value": best_value}


def format_report(result: SweepResult) -> str:
    """Fixed-width table of the sweep, for logs and tests."""
    lines = [
        f"{'setting':<24} {'size MB':>9} {'bpw':>6} {'time s':>8} {'rel err':>11}",
        "-" * 62,
    ]
    for r in result.rows:
        if r.failed:
            lines.append(f"{r.label:<24} FAILED: {r.failed.splitlines()[-1] if r.failed else ''}")
            continue
        lines.append(
            f"{r.label:<24} {r.size_bytes/1e6:>9.1f} {r.bpw:>6.2f} "
            f"{r.seconds:>8.1f} {r.error:>11.3e}"
        )
    measured = [r.measured_params for r in result.rows if r.measured_params]
    if measured:
        lines.append(
            f"(rel err measured on the largest tensors, "
            f"~{max(measured)/1e6:.0f}M of {result.n_params/1e9:.2f}B params)"
        )
    try:
        picks = best_settings(result)
        lines.append("-" * 62)
        for name, row in picks.items():
            lines.append(f"{name:<24} {row.label}")
    except ValueError:
        pass
    return "\n".join(lines)


# ----------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None):
    """Standalone sweep runner: python -m quant_gui.gguf_bench REF.gguf --imatrix FILE."""
    import argparse

    from quant_gui import llamacpp_backend as lcpp

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ref_gguf", help="F16/BF16 reference GGUF")
    parser.add_argument("--imatrix", default=None, help="imatrix file (default: skip imatrix variants)")
    parser.add_argument("--quants", default=None, help="comma-separated quant list (default: the 3.0bpw family)")
    parser.add_argument("--out-dir", default=None, help="scratch dir (default: <ref>.bench-tmp)")
    parser.add_argument(
        "--error-budget", type=int, default=300_000_000,
        help="max params covered by the error sample, largest tensors first "
        "(0 = measure the whole model; default 300M)",
    )
    parser.add_argument("--keep-outputs", action="store_true")
    args = parser.parse_args(argv)

    llamacpp_dir = lcpp.default_llamacpp_dir(Path(__file__).resolve().parent.parent)
    quantize_bin = lcpp._quantize_binary(llamacpp_dir)
    if quantize_bin is None:
        raise SystemExit(
            f"llama-quantize not found under {llamacpp_dir} - build or unpack it first."
        )
    out_dir = args.out_dir or (Path(args.ref_gguf).parent / (Path(args.ref_gguf).stem + ".bench-tmp"))
    result = run_sweep(
        ref_gguf=args.ref_gguf,
        imatrix_file=args.imatrix,
        out_dir=out_dir,
        quantize_bin=quantize_bin,
        quants=args.quants.split(",") if args.quants else None,
        keep_outputs=args.keep_outputs,
        error_budget_params=args.error_budget or None,
    )
    print(format_report(result))


if __name__ == "__main__":
    main()
