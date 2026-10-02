"""Fast GGUF quant-quality comparison without running inference.

Compares one reference GGUF (F16/BF16/F32) against any number of quantized
candidates by dequantizing *sampled rows* of every shared tensor and scoring
the error - optionally weighted by a llama-imatrix activation-importance
file (the same importance signal llama-quantize --imatrix optimises for).

This answers "which of these GGUFs preserves the weights best, and at what
bits-per-weight" in seconds-to-minutes, with no GPU and no llama.cpp runtime
- useful both as a stage-1 quant tuner and as a final A/B check when running
full perplexity on every candidate is impractical.

Usage:
    python tools/compare_gguf.py REF.gguf CAND1.gguf [CAND2.gguf ...]
        [--imatrix FILE] [--sample-rows N] [--top N] [--json OUT]

Exit status is 0 even when candidates differ; the verdict is in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_gui.smart_quant import load_imatrix  # noqa: E402

FLOAT_TYPES = {"F32", "F16", "BF16"}


# ---------------------------------------------------------------------------
# tensor reading helpers
# ---------------------------------------------------------------------------

def _float_rows_channels(data: np.ndarray) -> tuple[int, int]:
    """(rows_total, channels) from a float tensor's data array (channels-last)."""
    shape = data.shape
    if len(shape) < 2:
        return 1, int(shape[-1]) if shape else 1
    rows = 1
    for d in shape[:-1]:
        rows *= int(d)
    return rows, int(shape[-1])


def _as_f32(tensor) -> np.ndarray:
    """GGUFReader tensor -> float32 (channels-last) array or memmap view."""
    data = np.asarray(tensor.data)
    tname = tensor.tensor_type.name
    if tname == "F32":
        return data
    if tname == "F16":
        return data.view(np.float16) if data.dtype == np.uint8 else data
    if tname == "BF16":
        u16 = data.view(np.uint16) if data.dtype == np.uint8 else data.view(np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32)
    raise ValueError(f"not a float tensor: {tname}")


def _packed_rows(tensor, row_idx: np.ndarray) -> np.ndarray:
    """Slice packed (quantized) rows out of a tensor. GGUFReader exposes
    quantized data as (..., row_bytes) with rows independent along the
    leading axes, so any row subset dequantizes cleanly."""
    raw = np.asarray(tensor.data)
    if raw.dtype != np.uint8:
        raw = raw.view(np.uint8)
    raw2d = raw.reshape(-1, raw.shape[-1])
    return raw2d[row_idx]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

@dataclass
class TensorCmp:
    name: str
    qtype: str
    rows_sampled: int
    rel_l2: float          # ||dq-ref|| / ||ref||   (unweighted)
    w_rel_l2: float        # imatrix-weighted relative error
    cos: float
    snr_db: float
    ref_elems: int
    cand_bytes: int


@dataclass
class FileReport:
    path: str
    file_bytes: int
    total_params: int = 0
    tensors: list[TensorCmp] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def bpw(self) -> float:
        return self.file_bytes * 8.0 / self.total_params if self.total_params else 0.0

    def global_w_rel_l2(self) -> float:
        num = den = 0.0
        for t in self.tensors:
            w = t.ref_elems
            num += (t.w_rel_l2 ** 2) * w
            den += w
        return float(np.sqrt(num / den)) if den else float("nan")

    def global_rel_l2(self) -> float:
        num = den = 0.0
        for t in self.tensors:
            w = t.ref_elems
            num += (t.rel_l2 ** 2) * w
            den += w
        return float(np.sqrt(num / den)) if den else float("nan")


def _sample_indices(n_rows: int, max_rows: int, seed: int = 1234) -> np.ndarray:
    if n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    # evenly strided + jitter keeps coverage across experts/groups
    base = np.linspace(0, n_rows - 1, max_rows).astype(np.int64)
    return np.unique(base)


def compare_file(ref, cand, cand_path: Path, profiles, max_rows: int, progress=None) -> FileReport:
    from gguf import quants

    rep = FileReport(path=str(cand_path), file_bytes=cand_path.stat().st_size)
    ref_by_name = {t.name: t for t in ref.tensors}
    n = len(cand.tensors)
    for i, ct in enumerate(cand.tensors):
        if progress and (i % 25 == 0 or i == n - 1):
            progress(f"  tensor {i + 1}/{n}")
        rt = ref_by_name.get(ct.name)
        if rt is None:
            rep.skipped["missing-in-reference"] = rep.skipped.get("missing-in-reference", 0) + 1
            continue
        if tuple(rt.shape) != tuple(ct.shape):
            rep.skipped["shape-mismatch"] = rep.skipped.get("shape-mismatch", 0) + 1
            continue
        n_rows, n_ch = _float_rows_channels(np.asarray(rt.data))
        rep.total_params += int(np.prod(ct.shape))
        qtype = ct.tensor_type.name
        if qtype in FLOAT_TYPES:
            # candidate still float: error vs ref is pure dtype rounding
            if rt.tensor_type.name not in FLOAT_TYPES:
                rep.skipped["ref-quantized-cand-float"] = rep.skipped.get("ref-quantized-cand-float", 0) + 1
                continue
        if n_rows < 2 or n_ch < 2:
            rep.skipped["1-D-or-scalar"] = rep.skipped.get("1-D-or-scalar", 0) + 1
            continue

        row_idx = _sample_indices(n_rows, max_rows)
        try:
            ref_rows = _as_f32(rt).reshape(n_rows, n_ch)[row_idx].astype(np.float32)
        except ValueError as e:
            rep.skipped[f"ref-read:{e}"] = rep.skipped.get(f"ref-read:{e}", 0) + 1
            continue
        if qtype in FLOAT_TYPES:
            cand_rows = _as_f32(ct).reshape(n_rows, n_ch)[row_idx].astype(np.float32)
        else:
            try:
                cand_rows = quants.dequantize(_packed_rows(ct, row_idx),
                                              ct.tensor_type).astype(np.float32)
            except Exception as e:  # noqa: BLE001 - report and continue
                key = f"dequant-fail:{qtype}:{type(e).__name__}"
                rep.skipped[key] = rep.skipped.get(key, 0) + 1
                continue
        if cand_rows.shape != ref_rows.shape:
            cand_rows = cand_rows.reshape(ref_rows.shape)

        diff = cand_rows - ref_rows
        ref2 = float((ref_rows ** 2).sum())
        rel_l2 = float(np.sqrt(float((diff ** 2).sum()) / ref2)) if ref2 > 0 else float("nan")
        cos = float(
            np.dot(ref_rows.ravel(), cand_rows.ravel())
            / max(1e-12, np.linalg.norm(ref_rows) * np.linalg.norm(cand_rows))
        )
        snr = float(10.0 * np.log10(ref2 / max(1e-30, float((diff ** 2).sum()))))

        # imatrix-weighted relative error (uniform weights when no profile)
        prof = profiles.get(ct.name) if profiles else None
        if prof is None or prof.shape[-1] != n_ch:
            im = np.ones((1, n_ch), dtype=np.float32)
            groups = 1
        else:
            im = np.atleast_2d(np.asarray(prof, dtype=np.float32))
            groups = im.shape[0]
        rows_per_group = int(np.asarray(rt.data).shape[-2]) if np.asarray(rt.data).ndim >= 2 else 1
        # align sampled flat rows to their group
        g_idx = np.clip(row_idx // rows_per_group, 0, groups - 1)
        im_rows = im[g_idx]                              # (rows_sampled, n_ch)
        w_num = float((im_rows * diff ** 2).sum())
        w_den = float((im_rows * ref_rows ** 2).sum())
        w_rel = float(np.sqrt(w_num / w_den)) if w_den > 0 else float("nan")

        rep.tensors.append(TensorCmp(
            name=ct.name, qtype=qtype, rows_sampled=len(row_idx),
            rel_l2=rel_l2, w_rel_l2=w_rel, cos=cos, snr_db=snr,
            ref_elems=int(ref_rows.size), cand_bytes=int(ct.n_bytes),
        ))
    return rep


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def print_report(reps: list[FileReport], ref_path: str, top: int, dt: float) -> None:
    print(f"\nreference: {ref_path}")
    print(f"{'candidate':<44} {'bpw':>6} {'relL2%':>8} {'wRelL2%':>8} {'SNR dB':>7}")
    for r in reps:
        print(f"{Path(r.path).name:<44} {r.bpw:6.2f} "
              f"{100 * r.global_rel_l2():8.3f} {100 * r.global_w_rel_l2():8.3f} "
              f"{-20 * np.log10(max(1e-12, r.global_rel_l2())):7.1f}")
    if len(reps) > 1:
        best = min(reps, key=lambda r: r.global_w_rel_l2())
        print(f"\nbest weighted error: {Path(best.path).name}")

    for r in reps:
        by_type: dict[str, list[float]] = {}
        for t in r.tensors:
            by_type.setdefault(t.qtype, []).append(t.w_rel_l2)
        print(f"\n{Path(r.path).name} — per-type weighted rel error:")
        for qt, vals in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
            v = np.asarray(vals)
            print(f"  {qt:<10} n={len(v):<4} wRelL2% mean={100 * v.mean():7.3f} max={100 * v.max():7.3f}")
        worst = sorted(r.tensors, key=lambda t: -t.w_rel_l2)[:top]
        print(f"  worst {top} tensors (weighted):")
        for t in worst:
            print(f"    {t.w_rel_l2 * 100:7.3f}%  {t.qtype:<8} {t.name}")
        if r.skipped:
            print(f"  skipped: {r.skipped}")
    print(f"\ncompared in {dt:.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="reference GGUF (F16/BF16/F32)")
    ap.add_argument("candidates", nargs="+", help="quantized GGUF(s) to score")
    ap.add_argument("--imatrix", help="llama-imatrix file for activation weighting")
    ap.add_argument("--sample-rows", type=int, default=48,
                    help="max rows sampled per tensor (default 48; 0 = all rows)")
    ap.add_argument("--top", type=int, default=8, help="worst tensors to list (default 8)")
    ap.add_argument("--json", dest="json_out", help="write machine-readable report here")
    args = ap.parse_args()

    from gguf import GGUFReader

    t0 = time.time()
    ref_path = Path(args.reference)
    if not ref_path.is_file():
        print(f"reference not found: {ref_path}", file=sys.stderr)
        return 2
    profiles = load_imatrix(args.imatrix) if args.imatrix else None
    if profiles:
        print(f"imatrix: {args.imatrix} ({len(profiles)} tensors)")
    ref = GGUFReader(str(ref_path))
    if any(t.tensor_type.name not in FLOAT_TYPES for t in ref.tensors if t.n_bytes > 1 << 20):
        print("warning: reference has quantized tensors; metrics will be vs those", file=sys.stderr)

    max_rows = args.sample_rows if args.sample_rows > 0 else 1 << 30
    reps = []
    for c in args.candidates:
        p = Path(c)
        if not p.is_file():
            print(f"candidate not found: {p}", file=sys.stderr)
            continue
        print(f"scoring {p.name} ...")
        cand = GGUFReader(str(p))
        reps.append(compare_file(ref, cand, p, profiles, max_rows,
                                 progress=lambda m: print(f"\r{m}    ", end="")))
        print()
    if not reps:
        return 2
    dt = time.time() - t0
    print_report(reps, str(ref_path), args.top, dt)

    if args.json_out:
        payload = {
            "reference": str(ref_path),
            "imatrix": args.imatrix,
            "sample_rows": args.sample_rows,
            "seconds": round(dt, 2),
            "files": [{
                "path": r.path, "bytes": r.file_bytes, "bpw": round(r.bpw, 4),
                "rel_l2": r.global_rel_l2(), "w_rel_l2": r.global_w_rel_l2(),
                "tensors": [t.__dict__ for t in r.tensors],
            } for r in reps],
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"json report: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
