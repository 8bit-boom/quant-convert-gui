"""Per-tensor K-quant analysis and a size-budgeted mixed-precision plan.

What it does: for every quantizable 2-D tensor it quantizes a sample of rows at each level of the
ladder Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 with the real encoders (quant_gui/kquants.py, gguf-py for
Q8_0), decodes them again, and records the squared weight error. `allocate` then chooses a type per
tensor that minimises the total error for a size budget (default: the size of a uniform Q4_K file),
by walking each tensor's lower convex hull of (bytes, error) and upgrading the steepest
error-per-byte segment first - the classic Lagrangian allocation, optimal when the hulls are convex.

What the number means - read this before trusting a plan:
  * The objective is the SUM OF SQUARED WEIGHT ERROR. If a layer's input activations had the same
    energy everywhere, that sum would be proportional to the squared error of the layer outputs; real
    layers differ (inputs of an MLP down-projection are not like attention inputs, and errors in the
    first and last blocks travel through the whole network). No activation or image is measured here.
  * So a plan is "lowest weight-space error at this size", not "best images at this size". Whether it
    beats a uniform Q4_K in generated images is a hypothesis to check with fixed-seed A/B renders.
  * Error is measured on a row sample (default 128 rows per tensor) and scaled up; K-quant blocks run
    along a row, so sampling rows is exact per row but not for the whole tensor.
"""

from __future__ import annotations

import heapq
import json
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import kquants

LADDER = ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0")
BYTES_PER_ELEM = {
    "Q2_K": 84 / 256, "Q3_K": 110 / 256, "Q4_K": 144 / 256, "Q5_K": 176 / 256, "Q6_K": 210 / 256, "Q8_0": 34 / 32,
}
PLAN_FORMAT = "kquant-plan-1"


@dataclass
class TensorEntry:
    """A tensor to analyse; `get_rows(idx)` returns those rows as float32 (len(idx), cols)."""
    name: str
    rows: int
    cols: int
    get_rows: Callable[[np.ndarray], np.ndarray]


@dataclass
class TensorMeasure:
    name: str
    rows: int
    cols: int
    sample_rows: int
    norm2: float                      # estimated sum of squared weights of the whole tensor
    err: dict = field(default_factory=dict)   # type -> estimated sum of squared error of the whole tensor

    @property
    def numel(self) -> int:
        return self.rows * self.cols

    def bytes_for(self, t: str) -> int:
        return int(self.numel * BYTES_PER_ELEM[t])


@dataclass
class Plan:
    assign: dict                      # tensor name -> type
    total_bytes: int                  # including fixed_bytes
    total_err: float
    fits: bool
    min_bytes: int
    max_bytes: int
    budget_bytes: int
    notes: list = field(default_factory=list)


def _roundtrip_sse(x: np.ndarray, qtype: str, threads: int) -> float:
    import gguf
    from gguf import quants

    t = gguf.GGMLQuantizationType[qtype]
    packed = kquants.quantize(x, qtype, threads=threads) if qtype in kquants.SUPPORTED else quants.quantize(x, t)
    deq = quants.dequantize(packed, t).reshape(x.shape)
    return float(((deq.astype(np.float64) - x) ** 2).sum())


def measure(entries, types=LADDER, sample_rows: int = 128, threads: int = 1, progress_cb=None, should_stop=None):
    """Quantize+decode a row sample of every entry at every type; returns [TensorMeasure].
    Tensors whose sample holds non-finite values are skipped (listed in the second return value)."""
    out, skipped = [], []
    for i, e in enumerate(entries):
        if should_stop is not None and should_stop():
            break
        n = max(1, min(int(sample_rows), e.rows))
        idx = np.unique(np.linspace(0, e.rows - 1, n).round().astype(np.int64))
        x = np.ascontiguousarray(e.get_rows(idx), dtype=np.float32)
        if not np.isfinite(x).all():
            skipped.append(e.name)
            continue
        scale = e.rows / len(idx)
        err = {}
        for t in types:
            if (t in kquants.SUPPORTED and e.cols % kquants.QK_K == 0) or (t == "Q8_0" and e.cols % 32 == 0):
                err[t] = _roundtrip_sse(x, t, threads) * scale
        out.append(TensorMeasure(e.name, e.rows, e.cols, len(idx), float((x.astype(np.float64) ** 2).sum()) * scale, err))
        if progress_cb:
            progress_cb(i + 1, len(entries), e.name)
    return out, skipped


# --------------------------------------------------------------------------- allocation

def _hull(points):
    """points: [(bytes, err, type)] -> lower convex hull with strictly falling error, bytes ascending."""
    pts, best = [], float("inf")
    for b, e, t in sorted(points):
        if e < best:                                   # more bytes must buy less error
            pts.append((b, e, t))
            best = e
    hull = []
    for p in pts:
        # drop b if it lies above the segment a..p: the rate (err drop per byte) must fall along the hull
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (a[1] - b[1]) * (p[0] - b[0]) < (b[1] - p[1]) * (b[0] - a[0]):
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def allocate(measures, budget_bytes: int, fixed_bytes: int = 0, floor: str = "Q2_K", ceiling: str = "Q8_0") -> Plan:
    """Choose a type per tensor minimising total error with `fixed_bytes + sum(tensor bytes) <= budget_bytes`.
    Deterministic (ties broken by tensor name). If even the floor does not fit, the floor is returned
    with fits=False."""
    lo, hi = LADDER.index(floor), LADDER.index(ceiling)
    allowed = LADDER[lo:hi + 1]
    hulls = {}
    for m in measures:
        pts = [(m.bytes_for(t), m.err[t], t) for t in allowed if t in m.err]
        if pts:
            hulls[m.name] = _hull(pts)
    level = {n: 0 for n in hulls}
    total = fixed_bytes + sum(h[0][0] for h in hulls.values())
    min_bytes = total
    max_bytes = fixed_bytes + sum(h[-1][0] for h in hulls.values())
    fits = total <= budget_bytes
    notes = []
    if fits:
        heap = []
        for n, h in hulls.items():
            if len(h) > 1:
                rate = (h[0][1] - h[1][1]) / (h[1][0] - h[0][0])
                heap.append((-rate, n, 0))
        heapq.heapify(heap)
        while heap:
            _, n, lv = heapq.heappop(heap)
            h = hulls[n]
            extra = h[lv + 1][0] - h[lv][0]
            if total + extra > budget_bytes:
                continue                                # too big for what is left; a smaller upgrade may still fit
            total += extra
            level[n] = lv + 1
            if lv + 2 < len(h):
                rate = (h[lv + 1][1] - h[lv + 2][1]) / (h[lv + 2][0] - h[lv + 1][0])
                heapq.heappush(heap, (-rate, n, lv + 1))
    else:
        notes.append(f"Even the lowest type ({floor}) needs {min_bytes / 1e9:.2f} GB, more than the "
                     f"{budget_bytes / 1e9:.2f} GB asked for; the lowest types are returned.")
    if fits and total >= max_bytes - 1:
        notes.append(f"The budget is more than the highest type ({ceiling}) can use: every tensor got it.")
    assign = {n: hulls[n][level[n]][2] for n in hulls}
    err = sum(hulls[n][level[n]][1] for n in hulls)
    return Plan(assign, int(total), float(err), fits, int(min_bytes), int(max_bytes), int(budget_bytes), notes)


def uniform(measures, qtype: str, fixed_bytes: int = 0):
    """(total bytes, total error) if every tensor that supports `qtype` used it."""
    b = fixed_bytes + sum(m.bytes_for(qtype) for m in measures if qtype in m.err)
    e = sum(m.err[qtype] for m in measures if qtype in m.err)
    return int(b), float(e)


# --------------------------------------------------------------------------- reporting and plan files

def rel_rmse(err: float, norm2: float) -> float:
    return float(np.sqrt(err / norm2)) if norm2 > 0 else 0.0


def _group(name: str) -> str:
    return re.sub(r"\.\d+\.", ".N.", name)


def report(measures, plan: Plan, fixed_bytes: int = 0, uniform_types=("Q3_K", "Q4_K", "Q5_K", "Q6_K")) -> str:
    norm2 = sum(m.norm2 for m in measures if m.name in plan.assign)
    lines = ["Mixed K-quant plan (weight-space analysis)", ""]
    lines.append(f"{'':28s}{'size':>10s}{'weight error':>15s}")
    for t in uniform_types:
        if any(t in m.err for m in measures):
            b, e = uniform(measures, t, fixed_bytes)
            lines.append(f"  uniform {t:<18s}{b / 1e9:>7.2f} GB{rel_rmse(e, norm2):>14.4f}")
    lines.append(f"  mixed plan{'':<15s}{plan.total_bytes / 1e9:>7.2f} GB{rel_rmse(plan.total_err, norm2):>14.4f}"
                 f"   (budget {plan.budget_bytes / 1e9:.2f} GB)")
    lines.append("  'weight error' = relative RMS error of the planned tensors' weights; lower is better.")
    lines.append("")
    by_type: dict = {}
    for m in measures:
        t = plan.assign.get(m.name)
        if t:
            c, p = by_type.get(t, (0, 0))
            by_type[t] = (c + 1, p + m.numel)
    tot_p = sum(p for _, p in by_type.values()) or 1
    lines.append("types chosen:")
    for t in LADDER:
        if t in by_type:
            c, p = by_type[t]
            lines.append(f"  {t:<5s} {c:>4d} tensors  {100 * p / tot_p:5.1f}% of planned weights")
    groups: dict = {}
    for m in measures:
        t = plan.assign.get(m.name)
        if t:
            groups.setdefault(_group(m.name), []).append((m, t))
    lines += ["", "by tensor role (average bits per weight, lowest first):"]
    rows = []
    for g, items in groups.items():
        num = sum(m.numel for m, _ in items)
        bpw = sum(m.numel * BYTES_PER_ELEM[t] * 8 for m, t in items) / num
        rows.append((bpw, g, len(items)))
    for bpw, g, n in sorted(rows)[:12]:
        lines.append(f"  {bpw:5.2f}  {g}  ({n})")
    blocks: dict = {}
    for m in measures:
        t = plan.assign.get(m.name)
        mm = re.match(r"blocks\.(\d+)\.", m.name)
        if t and mm:
            blocks.setdefault(int(mm.group(1)), []).append((m, t))
    if blocks:
        lines += ["", "by block index (average bits per weight):"]
        row = []
        for i in sorted(blocks):
            num = sum(m.numel for m, _ in blocks[i])
            row.append(f"{i}:{sum(m.numel * BYTES_PER_ELEM[t] * 8 for m, t in blocks[i]) / num:.1f}")
        for k in range(0, len(row), 10):
            lines.append("  " + "  ".join(row[k:k + 10]))
    lines += [""] + [f"note: {n}" for n in plan.notes]
    lines += ["note: error is measured on the weights only, on a row sample. It is not an image-quality measurement.",
              "      Compare a uniform file and this plan with fixed-seed renders before trusting it."]
    return "\n".join(lines)


def save_plan(path, plan: Plan, meta: dict) -> None:
    doc = {"format": PLAN_FORMAT, "objective": "sum of squared weight error (uniform activation energy assumed)",
           "meta": meta, "types": dict(sorted(plan.assign.items()))}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)


def load_plan(path) -> dict:
    """tensor name -> type, validated. Raises ValueError with a plain message."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read plan file {path}: {exc}")
    if not isinstance(doc, dict) or doc.get("format") != PLAN_FORMAT or not isinstance(doc.get("types"), dict):
        raise ValueError(f"{path} is not a {PLAN_FORMAT} plan file")
    bad = sorted({t for t in doc["types"].values() if t not in LADDER})
    if bad:
        raise ValueError(f"plan file uses unsupported types: {', '.join(map(str, bad))}")
    return dict(doc["types"])
