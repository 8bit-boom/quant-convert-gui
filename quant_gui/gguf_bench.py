"""Sweep llama.cpp quant settings at a target bits-per-weight and rank them.

This is the test harness behind "which setting converts best into GGUF" for
Unsloth-style "Dynamic 3.0" workflows: the ~3.0 bpw family (Q3_K_*, IQ3_*)
produced by ``llama-quantize --imatrix`` on an F16/BF16 reference GGUF.
Unsloth's published per-model layer mixes aren't consumable here, so the
sweep measures what IS reproducible locally: for every candidate quant type,
with and without the imatrix calibration file, it records

  * output size (bytes and effective bits-per-weight),
  * quantization wall time,
  * reconstruction error vs the F16 reference (aggregate relative L2 of the
    dequantized tensors — lower means the weights survived rounding better,
    which is exactly what the imatrix is supposed to improve).

`best_settings()` then reports the quality winner, the size winner, and a
"best value" pick (lowest error-per-byte) so the choice is explicit instead
of folklore.

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


@dataclass
class BenchRow:
    quant: str
    imatrix: bool
    size_bytes: int = 0
    seconds: float = 0.0
    error: float | None = None  # aggregate relative L2 vs F16 reference
    bpw: float = 0.0  # bits per weight, from tensor metadata
    measured_params: int = 0  # params covered by the error sample (0 = all)
    failed: str | None = None

    @property
    def label(self) -> str:
        return f"{self.quant}{' + imatrix' if self.imatrix else ''}"


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
    """Aggregate relative L2 across the shared tensors: ||q-r|| / ||r||.

    Per-tensor relative errors would let big well-preserved tensors hide
    small destroyed ones; the aggregate keeps total signal dominant.
    """
    import numpy as np

    num = 0.0
    den = 0.0
    shared = set(ref_map) & set(quant_map)
    if not shared:
        raise ValueError("no shared tensor names between reference and quantized file")
    for name in shared:
        r = ref_map[name].ravel()
        q = quant_map[name].ravel()
        if q.shape != r.shape:
            raise ValueError(f"shape mismatch on {name!r}: {q.shape} vs {r.shape}")
        num += float(np.square(q - r).sum())
        den += float(np.square(r).sum())
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
) -> BenchRow:
    """Quantize once and measure; returns a BenchRow (``failed`` set on error).

    ``error_budget_params`` caps error measurement at the largest shared
    tensors up to that many parameters (deterministic) — full-model
    dequantization in pure numpy takes ~20s per 1.7B-param variant, so a
    budget keeps sweeps practical while the biggest tensors (which dominate
    the aggregate) are always measured. None = measure everything.
    """
    row = BenchRow(quant=quant, imatrix=bool(imatrix_file))
    cmd = [str(quantize_bin)]
    if imatrix_file:
        cmd += ["--imatrix", str(imatrix_file)]
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
    log=print,
) -> SweepResult:
    """Quantize `ref_gguf` with every (quant × imatrix) combination.

    `log` receives human-readable progress lines. Output files are deleted
    after measurement unless `keep_outputs`. `error_budget_params` caps the
    error sample at the largest tensors up to that many params (None = all).
    Keep it well above the largest single tensor — token embeddings are
    usually kept F16, so a sample of just the embedding measures ~zero error
    for every variant and can't discriminate (bit-identical results).
    """
    ref_gguf = Path(ref_gguf)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = SweepResult(
        rows=[], ref_gguf=str(ref_gguf), imatrix_file=str(imatrix_file or "")
    )
    _, result.n_params = _bits_per_weight(ref_gguf)

    for quant in (quants or DYNAMIC3_CANDIDATES):
        variants = []
        if with_imatrix and imatrix_file:
            variants.append(True)
        if without_imatrix:
            variants.append(False)
        for use_imatrix in variants:
            out_path = out_dir / f"bench-{quant}{'-imatrix' if use_imatrix else ''}.gguf"
            log(f"[bench] {quant}{' + imatrix' if use_imatrix else ''} ...")
            row = bench_quantize(
                quantize_bin, ref_gguf, out_path, quant,
                imatrix_file if use_imatrix else None,
                error_budget_params=error_budget_params,
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
