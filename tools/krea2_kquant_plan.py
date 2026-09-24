"""Build a K-quant tensor-type plan for a Krea-2 (image DiT) GGUF.

K-quants (Q3_K/Q4_K/Q5_K/Q6_K, IQ4_XS, ...) can't be encoded by gguf-py
(legacy Q4_0..Q8_0 only), and stock llama-quantize refuses the krea2 arch -
so K-quant DiT files come from the community two-stage flow:

  1. convert_krea2_to_gguf.py  -> BF16/legacy-quant GGUF  (this repo)
  2. a krea2-PATCHED llama.cpp's llama-quantize --tensor-type-file <plan>
     (RealRebelAI/molbal forks) -> the final K-quant GGUF

The missing input for llama.cpp's imatrix-weighted flow is activation stats:
DiTs have no llama-imatrix. This tool substitutes a UNIFORM-importance
sensitivity proxy: per-tensor reconstruction error of the legacy types the
Python stack CAN measure, ranked and tiered onto the K ladder. It's a
starting template to hill-climb from, not a calibrated optimum - say so in
anything you ship.

Usage:
    python tools/krea2_kquant_plan.py model_bf16.gguf --target-bpw 4.0 \\
        -o model.tensor-types.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_krea2_to_gguf import HIPREC_PREFIXES, QUANT_THRESHOLD_ELEMS  # noqa: E402

from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf import GGUFReader, quants  # noqa: E402

# proxy types the Python stack can actually measure
PROXIES = ("Q4_0", "Q5_0", "Q8_0")
# rows sampled per tensor when scoring (keeps big-DiT runs to minutes)
MAX_SCORE_ROWS = 4096
# K-ladder tier per sensitivity rank (best -> worst kept precision)
TIERS = ("Q6_K", "Q5_K", "Q4_K", "Q3_K")


def _is_plannable(name: str, shape, ftype: str) -> bool:
    shape = tuple(int(d) for d in shape)
    n = int(np.prod(shape)) if shape else 0
    if len(shape) != 2 or n <= QUANT_THRESHOLD_ELEMS:
        return False
    if name.startswith(HIPREC_PREFIXES):
        return False
    return ftype in ("BF16", "F32", "F16", "Q8_0")


def score_tensors(gguf_path: str | Path, progress: bool = False) -> list[tuple[str, float]]:
    """(tensor name, Q8_0->Q4_0 error gap) sorted most-sensitive first.

    The gap between a fine and a coarse proxy reconstruction measures how
    much the tensor suffers under aggressive quantization - the same signal
    smart_quant uses, minus the activation weighting.
    """
    reader = GGUFReader(str(gguf_path))
    scored = []
    tensors = list(reader.tensors)
    for i, t in enumerate(tensors):
        if not _is_plannable(t.name, t.shape, t.tensor_type.name):
            continue
        data = np.asarray(t.data)
        rows = t.shape[0]
        arr = data.reshape(rows, -1)  # byte view: (rows, bytes_per_row)
        src_type = t.tensor_type.name
        if src_type == "BF16":
            f32 = (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            arr = f32
        elif src_type == "F16":
            arr = arr.view(np.float16).astype(np.float32)
        elif src_type == "F32":
            arr = arr.view(np.float32)
        else:  # legacy-quant source: dequantize first
            arr = quants.dequantize(arr.reshape(-1), t.tensor_type).reshape(rows, -1)
        if arr.shape[-1] % 32 != 0:
            continue
        # subsample rows so a 14 GB DiT scores in minutes, not hours;
        # sensitivity structure lives in row magnitude, not row count
        if arr.shape[0] > MAX_SCORE_ROWS:
            step = arr.shape[0] // MAX_SCORE_ROWS
            arr = arr[::step][:MAX_SCORE_ROWS]
        errs = {}
        for p in PROXIES:
            if p == src_type:
                continue  # quantizing to the source type is a no-op -> ~0 error, no signal
            try:
                errs[p] = float(np.mean((arr - quants.dequantize(
                    quants.quantize(arr, QT[p]), QT[p]).reshape(arr.shape)) ** 2))
            except Exception:  # noqa: BLE001 - proxy unavailable; skip tier
                continue
        if len(errs) < 2:
            continue
        fine = errs.get("Q5_0", errs.get("Q8_0"))
        gap = errs["Q4_0"] / max(fine, 1e-12)  # >1 = hurts when squeezed
        scored.append((t.name, gap))
        if progress:
            print(f"[{i + 1}/{len(tensors)}] {t.name} gap={gap:.3f}", flush=True)
    scored.sort(key=lambda kv: -kv[1])
    return scored


def plan_assignment(scored: list[tuple[str, float]], target_bpw: float) -> dict[str, str]:
    """Tier tensors onto the K ladder by sensitivity rank. `target_bpw`
    shifts the cut points: lower target pushes more tensors to Q3_K."""
    if not scored:
        return {}
    target = min(max(target_bpw, 3.0), 6.0)
    # fraction of the ladder kept at Q6_K/Q5_K: 1.0 at target 6.0, 0.0 at 3.0
    frac_top = (target - 3.0) / 3.0
    n = len(scored)
    # rank-fraction cuts, scaled by frac_top so low targets actually move
    # the boundaries: t1/t2 sweep the top of the list, t3 sits halfway
    # between frac_top and 1.0 so the bottom always degrades to Q3_K.
    t1, t2, t3 = frac_top * 0.5, frac_top, (1.0 + frac_top) / 2.0
    assignment = {}
    for rank, (name, _gap) in enumerate(scored):
        r = rank / n
        tier = TIERS[0] if r < t1 else \
               TIERS[1] if r < t2 else \
               TIERS[2] if r < t3 else TIERS[3]
        assignment[name] = tier
    return assignment


def render(assignment: dict[str, str]) -> str:
    # llama.cpp matches these names as regexes - escape so '.' in tensor
    # names can't wildcard-match a different tensor's pattern.
    import re as _re
    return "\n".join(f"{_re.escape(n)}={t}" for n, t in sorted(assignment.items())) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf", type=Path, help="Krea-2 GGUF (BF16 or legacy quant) from convert_krea2_to_gguf.py")
    ap.add_argument("--target-bpw", type=float, default=4.0,
                    help="size target in weight-space bits-per-weight (3.0-6.0, default 4.0)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="tensor-type file path (default: <gguf>.tensor-types.txt)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    if not args.gguf.is_file():
        print(f"not found: {args.gguf}", file=sys.stderr)
        return 1
    scored = score_tensors(args.gguf, progress=args.verbose)
    if not scored:
        print("no plannable 2-D tensors found - is this a Krea-2 GGUF?", file=sys.stderr)
        return 1
    assignment = plan_assignment(scored, args.target_bpw)
    out = args.output or args.gguf.with_suffix(".tensor-types.txt")
    out.write_text(render(assignment), encoding="utf-8")
    from collections import Counter
    print(f"scored {len(scored)} tensors; plan: {dict(Counter(assignment.values()))}")
    print(f"wrote {out}")
    print("next: run a krea2-patched llama-quantize with --tensor-type-file on this plan:")
    print(f"  llama-quantize --tensor-type-file {out} {args.gguf} out-Q4_K.gguf Q4_K_KREA2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
