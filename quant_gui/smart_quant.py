"""Three-stage smart per-tensor quantization tuner ("Dynamic-quant style").

Implements the plan of scoring per-tensor quantization sensitivity with the
imatrix as the activation-importance weight, then assigning per-tensor
quant types under a size budget, then emitting a llama-quantize
``--tensor-type-file``:

Stage 1 - cheap per-tensor sensitivity ranking.
    For each weight tensor of the F16/BF16 reference GGUF, for each
    candidate type the pure-Python ``gguf.quants`` path can quantize
    (legacy Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 + F16/BF16), quantize + dequantize
    and measure the importance-weighted RMSE

        err = sqrt( sum_c im[c] * sum_r (w - w_hat)[r, c]^2
                    / (sum_c im[c] * n_rows) )

    where ``im`` is the tensor's imatrix profile (mean squared input
    activations per input channel). That is exactly the error llama-quantize
    tries to minimise with ``--imatrix``, so the ranking predicts which
    tensors tolerate downgrading and which deserve upgrading.

Stage 2 - greedy assignment under a size budget.
    Two modes:
    * "legacy" - only the stage-1-scored types are used (exact measured
      errors, exact measured sizes); upgrade-only ladder.
    * "k-ladder" - tensors are ranked by measured sensitivity and mapped
      onto the q3_k/q4_k/q5_k/q6_k K-quant ladder (sizes exact from ggml
      block constants, per-type errors not pure-Python measurable - the
      measured legacy errors provide the ranking, the ladder mapping is by
      rank). This is the mode that produces llama.cpp-serving-friendly
      files comparable to Unsloth's Dynamic quants.
    Gotchas enforced: token embeddings, the output/lm_head tensor and MoE
    router (ffn_gate_inp) never fall below the floor type regardless of
    ranking; 1-D norm tensors are never touched.

Stage 3 - validation is done by the caller (app.py): the emitted
tensor-type file is fed to llama-quantize together with the imatrix, and
perplexity can be compared against the plain baseline with llama-perplexity.

Layout facts this module relies on (all verified against llama.cpp b11070+
sources and real files, not assumed):
* GGUF tensor arrays as exposed by gguf-py are C-contiguous with the LAST
  axis = input channels (ggml ne0, the file-contiguous dim); the imatrix
  profile indexes that last axis; earlier axes are (groups..., rows).
* gguf.quants.quantize blocks along the LAST axis of the array it is given
  (reshape(-1, shape[-1])), so tensors are passed as (rows, channels) with
  channels last.
* llama.cpp imatrix (GGUF format): per source tensor ``<name>.in_sum2``
  (f32) and ``<name>.counts`` (f32); flat float data is group-major
  ``[group][channel]`` where group = counts index (1 group for dense
  tensors, one per expert for MoE - llama.cpp quantizes experts in chunks
  that never cross an expert boundary and picks the imatrix slice per
  expert). Legacy .dat format parses to the same layout.
* llama-quantize --tensor-type-file: whitespace-separated ``name=type``
  tokens; name is matched as a REGEX (lowercased), type must be an atomic
  ggml type name ("q4_k", "q6_k", "q8_0", ... - NOT mixtures like Q4_K_M).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Candidate types the pure-Python gguf.quants path can quantize AND
# dequantize (verified by probing every GGMLQuantizationType). K-quants and
# IQ-quants are dequantize-only there, so k-ladder mode ranks by these and
# maps by rank instead of measuring K-quant errors directly.
SCORABLE_TYPES = ("q4_0", "q5_0", "q8_0")

# K-quant ladder for rank-mapped assignment (atomic types only).
K_LADDER = ("q3_k", "q4_k", "q5_k", "q6_k")

# Never let these fall below the floor, whatever the ranking says
# (standard practice - Unsloth / ik_llama.cpp "Chess" quants do the same).
# NOTE: the final lm_head is exactly "output.weight"; attention output
# projections ("attn_output.weight") must NOT match - they are regular
# 2-D weights that the ranking may move freely.
def is_sensitive_name(name: str) -> bool:
    n = name.lower()
    if "token_embd" in n:
        return True
    if n.startswith("output.") and "attn_output" not in n:
        return True
    if "ffn_gate_inp" in n:
        return True
    return False


DEFAULT_FLOOR = "q6_k"


class SmartQuantError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Stage 0: imatrix loading
# ---------------------------------------------------------------------------

def load_imatrix(path: str | Path) -> dict[str, np.ndarray]:
    """Load a llama-imatrix file -> {tensor_name: float32 array [groups, channels]}.

    Supports both the current GGUF format (default output of recent
    llama-imatrix) and the legacy .dat binary format. Missing/empty
    profiles are simply absent from the dict.
    """
    path = Path(path)
    if not path.is_file():
        raise SmartQuantError(f"imatrix not found: {path}")
    with open(path, "rb") as f:
        head = f.read(4)
    if head == b"GGUF":
        return _load_imatrix_gguf(path)
    return _load_imatrix_legacy(path)


def _load_imatrix_gguf(path: Path) -> dict[str, np.ndarray]:
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    sums = {}
    counts = {}
    for t in reader.tensors:
        if t.name.endswith(".in_sum2"):
            sums[t.name[: -len(".in_sum2")]] = np.asarray(t.data, dtype=np.float32).ravel()
        elif t.name.endswith(".counts"):
            counts[t.name[: -len(".counts")]] = np.asarray(t.data, dtype=np.float32).ravel()
    out = {}
    for name, flat in sums.items():
        n_groups = int(counts.get(name, np.array([1.0])).shape[0]) or 1
        n_channels = flat.shape[0] // n_groups
        if n_channels <= 0:
            continue
        out[name] = flat[: n_groups * n_channels].reshape(n_groups, n_channels)
    return out


def _load_imatrix_legacy(path: Path) -> dict[str, np.ndarray]:
    out = {}
    with open(path, "rb") as f:
        (n_entries,) = struct.unpack("<i", f.read(4))
        for _ in range(n_entries):
            (nlen,) = struct.unpack("<i", f.read(4))
            name = f.read(nlen).decode("utf-8", errors="replace")
            f.read(4)  # ncall
            (nval,) = struct.unpack("<i", f.read(4))
            (nmat,) = struct.unpack("<i", f.read(4))
            nval = max(nval, 0)
            nmat = max(nmat, 1)
            flat = np.frombuffer(f.read(nval * 4), dtype=np.float32)
            n_channels = flat.shape[0] // nmat
            if n_channels > 0:
                out[name] = flat[: nmat * n_channels].reshape(nmat, n_channels).copy()
    return out


# ---------------------------------------------------------------------------
# Stage 1: per-tensor sensitivity scoring
# ---------------------------------------------------------------------------

@dataclass
class TensorScore:
    name: str
    shape: tuple[int, ...]  # reader order: channels LAST, (groups..., rows, channels)
    n_rows: int  # prod(shape[1:])
    n_channels: int  # shape[0]
    n_groups: int  # imatrix groups (1 dense, n_experts for MoE)
    rows_per_group: int
    # measured importance-weighted rmse per scored type
    err: dict[str, float] = field(default_factory=dict)
    # per-group rmse for the primary scorable type (MoE expert report)
    group_err: dict[str, list[float]] = field(default_factory=dict)
    # measured packed bytes per scored type
    packed_bytes: dict[str, int] = field(default_factory=dict)
    # bytes the tensor occupies in the input file (skipped/non-float tensors)
    existing_bytes: int = 0
    skipped: str = ""  # reason when not scored

    @property
    def is_moe(self) -> bool:
        return len(self.shape) >= 3

    @property
    def is_sensitive(self) -> bool:
        return is_sensitive_name(self.name)


def is_scorable_name(name: str) -> bool:
    return not is_sensitive_name(name)


def _channels_last(data: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    """Return (A, n_rows, n_channels, rows_per_group, n_groups) with channels last.

    A[r, c] = weight element with channel c (ggml ne0, the file-contiguous
    dim) at flat row r; earlier axes are (groups..., rows) with rows directly
    before channels, so flattening to (rows_total, channels) gives
    group-major row order (row = g * rows_per_group + r), matching ggml
    storage and llama.cpp's per-expert imatrix slicing, with channels
    contiguous per row - exactly what gguf.quants.quantize blocks along.
    """
    shape = data.shape
    n_channels = int(shape[-1])
    rows_per_group = int(shape[-2]) if len(shape) >= 2 else 1
    n_groups = 1
    for d in shape[:-2]:
        n_groups *= int(d)
    a = np.ascontiguousarray(data.reshape(-1, n_channels))
    return a, a.shape[0], n_channels, rows_per_group, n_groups


def weighted_rmse(
    w: np.ndarray,          # (rows, channels) f32
    dq: np.ndarray,         # (rows, channels) f32
    profile: np.ndarray,    # (channels,) or (groups, channels) f32 importance
    rows_per_group: int,
) -> np.ndarray:
    """Importance-weighted rmse per group.

    Returns a float32 array of shape (n_groups,): for group g,
        sqrt( sum_c im[g, c] * sum_r (w - dq)[r, c]^2 / (sum_c im[g, c] * rows_g) )
    Tensors with (near-)zero total importance get the plain unweighted rmse
    for every group, so ranking still works.
    """
    diff2_col = ((w - dq) ** 2).reshape(-1, rows_per_group, w.shape[1]).sum(axis=1)
    # diff2_col: (groups, channels)
    prof = np.asarray(profile, dtype=np.float32)
    if prof.ndim == 1:
        prof = prof.reshape(1, -1)
    if prof.shape[-1] != diff2_col.shape[1] or prof.shape[0] not in (1, diff2_col.shape[0]):
        prof = np.ones_like(diff2_col)
    elif prof.shape[0] == 1 and diff2_col.shape[0] > 1:
        prof = np.broadcast_to(prof, diff2_col.shape)
    im = np.maximum(prof, 0.0)
    denom = (im * rows_per_group).sum(axis=1)
    total = (diff2_col * im).sum(axis=1)
    use_weighted = denom > 1e-12
    weighted = np.sqrt(total[use_weighted] / denom[use_weighted])
    plain = np.sqrt(diff2_col.sum(axis=1) / (diff2_col.shape[1] * rows_per_group))
    out = plain.copy()
    out[use_weighted] = weighted
    return out


def score_tensor(
    name: str,
    data: np.ndarray,
    profiles: dict[str, np.ndarray] | None,
    types: tuple[str, ...] = SCORABLE_TYPES,
) -> TensorScore:
    """Quantize one tensor with each candidate type and measure weighted rmse.

    `data` is the array as GGUFReader exposes it (channels LAST axis), any float dtype.
    Pure numpy + gguf.quants - unit-testable without a GGUF file.
    """
    from gguf import GGMLQuantizationType, quants

    if data.ndim < 2:
        return TensorScore(name, tuple(int(x) for x in data.shape), 1, 1, 1, 1, skipped="1-D tensor (norm/scale)")
    w, n_rows, n_channels, rows_per_group, n_groups = _channels_last(np.asarray(data))
    score = TensorScore(
        name=name,
        shape=tuple(int(x) for x in data.shape),
        n_rows=n_rows,
        n_channels=n_channels,
        n_groups=n_groups,
        rows_per_group=rows_per_group,
    )
    if n_rows == 0 or n_channels == 0:
        score.skipped = "empty tensor"
        return score

    wf = w.astype(np.float32, copy=False)
    profile = profiles.get(name) if profiles else None
    if profile is None or profile.shape[-1] != n_channels:
        profile = np.ones(n_channels, dtype=np.float32)
    # dense tensors: single profile row; moe: one row per group
    if profile.ndim == 1:
        profile = profile.reshape(1, -1)
    if profile.shape[0] not in (1, n_groups):
        profile = np.ones((n_groups, n_channels), dtype=np.float32)
    elif profile.shape[0] == 1 and n_groups > 1:
        profile = np.broadcast_to(profile, (n_groups, n_channels)).copy()

    for type_name in types:
        qtype = GGMLQuantizationType[type_name.upper()]
        packed = quants.quantize(wf, qtype)
        dq = quants.dequantize(packed, qtype)
        per_group = weighted_rmse(wf, dq, profile, rows_per_group)
        score.err[type_name] = float(
            np.sqrt((per_group ** 2 * rows_per_group).sum() / n_rows)
        )
        score.packed_bytes[type_name] = int(packed.nbytes)
        if type_name == types[0]:
            score.group_err[type_name] = [float(x) for x in per_group]
    return score


def score_model(
    ref_gguf: str | Path,
    imatrix: str | Path,
    types: tuple[str, ...] = SCORABLE_TYPES,
    progress=None,
) -> list[TensorScore]:
    """Score every scorable tensor of a reference GGUF. `progress(msg)` is
    called per tensor (already-scored count and total are the caller's job)."""
    from gguf import GGUFReader

    ref_gguf = Path(ref_gguf)
    if not ref_gguf.is_file():
        raise SmartQuantError(f"reference GGUF not found: {ref_gguf}")
    profiles = load_imatrix(imatrix)
    reader = GGUFReader(str(ref_gguf))
    scores: list[TensorScore] = []
    total = len(reader.tensors)

    def meta(name: str, shape) -> TensorScore:
        shape = tuple(int(x) for x in shape)
        ch = shape[-1] if len(shape) >= 2 else 1
        rpg = shape[-2] if len(shape) >= 2 else 1
        ng = 1
        for d in shape[:-2]:
            ng *= int(d)
        return TensorScore(name=name, shape=shape, n_rows=rpg * ng,
                           n_channels=ch, n_groups=ng, rows_per_group=rpg)

    for i, t in enumerate(reader.tensors):
        if progress and (i % 10 == 0 or i == total - 1):
            progress(f"scoring tensor {i + 1}/{total}...")
        if t.tensor_type.name not in ("F32", "F16", "BF16"):
            s = meta(t.name, t.shape)
            s.skipped = f"already {t.tensor_type.name}"
            s.existing_bytes = int(t.n_bytes)
            scores.append(s)
            continue
        if not is_scorable_name(t.name):
            # pinned at the floor by the knapsack - scoring would only waste
            # time (embeddings are the biggest tensors in PLE models)
            s = meta(t.name, t.shape)
            scores.append(s)
            continue
        data = np.asarray(t.data)
        s = score_tensor(t.name, data, profiles, types)
        if s.skipped:
            s.existing_bytes = int(t.n_bytes)
        if not s.skipped and progress:
            errs = ", ".join(f"{k}={v:.4g}" for k, v in s.err.items())
            progress(f"  {t.name}: {errs}")
        scores.append(s)
    return scores


# ---------------------------------------------------------------------------
# Stage 2: size model + greedy assignment
# ---------------------------------------------------------------------------

def type_bytes(shape: tuple[int, ...], type_name: str) -> int:
    """Exact ggml packed size for a tensor of `shape` as `type_name`.

    `shape` is in GGUFReader order (channels LAST, matching ggml ne reversed):
    rows = prod(shape[:-1]), channels = shape[-1]; ggml blocks channels.
    """
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

    qtype = GGMLQuantizationType[type_name.upper()]
    n_elems = int(np.prod(shape)) if shape else 0
    if qtype in (GGMLQuantizationType.F32,):
        return n_elems * 4
    if qtype in (GGMLQuantizationType.F16, GGMLQuantizationType.BF16):
        return n_elems * 2
    block, tsize = GGML_QUANT_SIZES[qtype]
    rows = 1
    for d in shape[:-1]:
        rows *= int(d)
    cols = int(shape[-1]) if shape else 0
    return rows * ((cols + block - 1) // block) * tsize


def estimate_total_bytes(scores: list[TensorScore], base_type: str, floor_type: str) -> int:
    """Whole-file size estimate: scored tensors at base_type, sensitive ones at
    floor_type, skipped (already-quantized) tensors at their existing size,
    1-D norms approximated at base_type (their true size is tiny)."""
    total = 0
    for s in scores:
        if s.skipped:
            total += s.existing_bytes or type_bytes(s.shape, base_type)
            continue
        t = floor_type if s.is_sensitive else base_type
        total += type_bytes(s.shape, t)
    return total


def assign_k_ladder(
    scores: list[TensorScore],
    budget_bytes: int,
    base_type: str = "q4_k",
    floor_type: str = DEFAULT_FLOOR,
) -> tuple[dict[str, str], list[str]]:
    """Rank tensors by measured q4_0 sensitivity, then water-fill the
    q3_k/q4_k/q5_k/q6_k ladder around `base_type` until the estimated file
    size meets `budget_bytes`.

    Over budget: least-sensitive movable tensors step down the ladder
    (q4_k -> q3_k) until it fits. Under budget: most-sensitive tensors step
    up (q4_k -> q5_k -> q6_k) while headroom allows. Sensitive tensors
    (embeddings/output/router) are pinned at floor_type and never move.
    K-quant per-type errors are not pure-Python measurable, so the ladder
    mapping is rank-based, not error-scored; sizes are exact.
    """
    report: list[str] = []
    scored = [s for s in scores if not s.skipped and s.err]
    if not scored:
        raise SmartQuantError("no scorable tensors - is this a float reference GGUF?")
    primary = next(iter(scored[0].err), "q4_0")
    sens = {s.name: s.err.get(primary, 0.0) for s in scored}

    base_idx = K_LADDER.index(base_type) if base_type in K_LADDER else 1
    cur: dict[str, int] = {}
    for s in scores:
        if s.skipped:
            continue
        cur[s.name] = base_idx
    pinned = {s.name for s in scores if not s.skipped and s.is_sensitive}
    for name in pinned:
        cur[name] = K_LADDER.index(floor_type) if floor_type in K_LADDER else len(K_LADDER) - 1

    def total_size() -> int:
        return sum(
            (s.existing_bytes if s.skipped else type_bytes(s.shape, K_LADDER[cur[s.name]]))
            for s in scores
        )

    movable = sorted(
        (s for s in scored if not s.is_sensitive),
        key=lambda s: sens[s.name],
    )

    # Downgrade pass: walk least-sensitive first, one ladder step per touch.
    size = total_size()
    down_rounds = 0
    while size > budget_bytes and down_rounds < base_idx:
        moved = False
        for s in movable:
            if size <= budget_bytes:
                break
            if cur[s.name] > 0:
                cur[s.name] -= 1
                size = total_size()
                moved = True
        if not moved:
            break
        down_rounds += 1

    # Upgrade pass: walk most-sensitive first, one ladder step per touch.
    max_idx = len(K_LADDER) - 1
    up_rounds = 0
    while size < budget_bytes and up_rounds < max_idx - base_idx:
        moved = False
        for s in reversed(movable):
            if cur[s.name] < max_idx:
                nxt = cur[s.name] + 1
                new_size = total_size() - type_bytes(s.shape, K_LADDER[cur[s.name]]) + type_bytes(s.shape, K_LADDER[nxt])
                if new_size > budget_bytes:
                    continue
                cur[s.name] = nxt
                size = new_size
                moved = True
        if not moved:
            break
        up_rounds += 1

    assignment = {name: K_LADDER[idx] for name, idx in cur.items()}
    n_up = sum(1 for s in movable if cur[s.name] > base_idx)
    n_down = sum(1 for s in movable if cur[s.name] < base_idx)
    report.append(
        f"assignment: {n_up} tensors upgraded, {n_down} downgraded, "
        f"estimated size {size / 1e9:.2f} GB (budget {budget_bytes / 1e9:.2f} GB)"
    )
    if size > budget_bytes:
        report.append(
            "⚠️ could not reach the budget even with every movable tensor at "
            f"{K_LADDER[0]} - raise the budget or accept a larger file."
        )
    movable_names = {s.name for s in movable}
    top_up = sorted((n for n in cur if n in movable_names and cur[n] > base_idx),
                    key=lambda n: -sens[n])[:5]
    if top_up:
        report.append("most upgraded: " + ", ".join(f"{n}->{assignment[n]}" for n in top_up))
    top_down = sorted((n for n in cur if n in movable_names and cur[n] < base_idx),
                      key=lambda n: sens[n])[:5]
    if top_down:
        report.append("most downgraded: " + ", ".join(f"{n}->{assignment[n]}" for n in top_down))
    return assignment, report


def assign_legacy(
    scores: list[TensorScore],
    budget_bytes: int,
    base_type: str = "q4_0",
    upgrade_types: tuple[str, ...] = ("q5_0", "q8_0"),
) -> tuple[dict[str, str], list[str]]:
    """Exact-measured greedy: start everything scorable at `base_type`, then
    spend remaining budget on upgrades with the best measured
    error-drop-per-byte. Downgrades below q4_0 are not offered because no
    lower pure-Python-scorable type is sane to use."""
    report: list[str] = []
    scored = [s for s in scores if not s.skipped and s.err and base_type in s.packed_bytes]
    if not scored:
        raise SmartQuantError("no scorable tensors with measured sizes")
    assignment: dict[str, str] = {}
    for s in scores:
        if s.skipped:
            continue
        assignment[s.name] = base_type if not s.is_sensitive else DEFAULT_FLOOR
    size = sum(
        s.packed_bytes.get(assignment[s.name]) or type_bytes(s.shape, assignment[s.name])
        for s in scores if not s.skipped
    )
    # upgrade value: (err_base - err_up) / (bytes_up - bytes_base), descending
    options = []
    for s in scored:
        if s.is_sensitive:
            continue
        e0 = s.err[base_type]
        b0 = s.packed_bytes[base_type]
        for up in upgrade_types:
            if up not in s.err:
                continue
            de = e0 - s.err[up]
            db = s.packed_bytes[up] - b0
            if de > 0 and db > 0:
                options.append((de / db, s, up, db))
    options.sort(key=lambda x: -x[0])
    used: set[str] = set()
    for _, s, up, db in options:
        if s.name in used:
            continue
        if size + db > budget_bytes:
            continue
        assignment[s.name] = up
        size += db
        used.add(s.name)
    report.append(
        f"assignment: {len(used)} tensors upgraded ({', '.join(upgrade_types)}), "
        f"estimated size {size / 1e9:.2f} GB (budget {budget_bytes / 1e9:.2f} GB)"
    )
    if size > budget_bytes:
        report.append("⚠️ base type alone already exceeds the budget; legacy mode cannot "
                      "downgrade - use k-ladder mode or a bigger budget.")
    return assignment, report


# ---------------------------------------------------------------------------
# Stage 2 output: llama-quantize --tensor-type-file
# ---------------------------------------------------------------------------

def emit_tensor_type_file(assignment: dict[str, str]) -> str:
    """Render the assignment as --tensor-type-file content.

    One ``name=type`` token per line (llama.cpp lowercases the name and
    compiles it as a regex - tensor names are already lowercase and contain
    no regex metacharacters in practice, but dots are escaped anyway).
    """
    import re as _re

    lines = []
    for name in sorted(assignment):
        t = assignment[name]
        safe = _re.escape(name)
        lines.append(f"{safe}={t}")
    return "\n".join(lines) + "\n"


def moe_expert_report(scores: list[TensorScore], top: int = 10) -> list[str]:
    """Per-expert sensitivity report for MoE models (informational - a
    tensor-type-file pattern covers a whole tensor, so experts can't be
    assigned different types; this shows where the headroom would be)."""
    lines = []
    for s in scores:
        if not s.is_moe or not s.group_err:
            continue
        primary = next(iter(s.group_err))
        errs = s.group_err[primary]
        if len(errs) < 2:
            continue
        lo_i = int(np.argmin(errs))
        hi_i = int(np.argmax(errs))
        lines.append(
            f"{s.name}: {len(errs)} experts, rmse min={errs[lo_i]:.4g} (expert {lo_i}) "
            f"max={errs[hi_i]:.4g} (expert {hi_i}), spread x{errs[hi_i] / max(errs[lo_i], 1e-12):.2f}"
        )
    return lines[:top]
