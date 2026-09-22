"""Tests for the smart per-tensor quant tuner (quant_gui.smart_quant)."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

gguf = pytest.importorskip("gguf")

from quant_gui import smart_quant as sq


# ---------------------------------------------------------------------------
# imatrix loading (legacy .dat format - easy to synthesize byte-exact)
# ---------------------------------------------------------------------------

def _write_legacy_imatrix(path, entries):
    """entries: {name: (n_groups, channels)} with value = channel index + 1."""
    buf = struct.pack("<i", len(entries))
    for name in sorted(entries):
        n_groups, channels = entries[name]
        nval = n_groups * channels
        flat = np.arange(1, nval + 1, dtype=np.float32)
        nb = name.encode()
        buf += struct.pack("<i", len(nb)) + nb
        buf += struct.pack("<i", 1)              # ncall
        buf += struct.pack("<i", nval) + struct.pack("<i", n_groups)
        buf += flat.tobytes()
    path.write_bytes(buf)
    return {n: np.arange(1, g * c + 1, dtype=np.float32).reshape(g, c)
            for n, (g, c) in entries.items()}


def test_load_legacy_imatrix(tmp_path):
    p = tmp_path / "m.imatrix"
    expect = _write_legacy_imatrix(p, {"blk.0.attn_q.weight": (1, 64), "blk.1.ffn_down_exps.weight": (4, 32)})
    got = sq.load_imatrix(p)
    assert set(got) == set(expect)
    for k in expect:
        assert got[k].shape == expect[k].shape
        np.testing.assert_allclose(got[k], expect[k])


def test_load_imatrix_missing(tmp_path):
    with pytest.raises(sq.SmartQuantError):
        sq.load_imatrix(tmp_path / "nope")


def test_load_gguf_imatrix_roundtrip(tmp_path):
    """GGUF-format imatrix: synthesize via gguf writer, read back."""
    from gguf import GGUFWriter

    w = GGUFWriter(tmp_path / "m.imatrix", "imatrix")
    vals = np.arange(1, 65, dtype=np.float32)
    counts = np.array([5.0], dtype=np.float32)
    w.add_tensor("blk.0.attn_q.weight.in_sum2", vals, raw_dtype=gguf.GGMLQuantizationType.F32)
    w.add_tensor("blk.0.attn_q.weight.counts", counts, raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    got = sq.load_imatrix(tmp_path / "m.imatrix")
    assert "blk.0.attn_q.weight" in got
    assert got["blk.0.attn_q.weight"].shape == (1, 64)
    np.testing.assert_allclose(got["blk.0.attn_q.weight"][0], vals)


# ---------------------------------------------------------------------------
# scoring math
# ---------------------------------------------------------------------------

def test_weighted_rmse_perfect_zero():
    w = np.random.randn(128, 256).astype(np.float32)
    im = np.ones(256, dtype=np.float32)
    per = sq.weighted_rmse(w, w.copy(), im, 128)
    np.testing.assert_allclose(per, 0.0, atol=1e-7)


def test_weighted_rmse_weights_channels():
    # error only in channels with zero importance -> weighted rmse ~0
    w = np.zeros((64, 32), dtype=np.float32)
    dq = w.copy()
    dq[:, :16] = 10.0  # error in first 16 channels
    im = np.zeros(32, dtype=np.float32)
    im[16:] = 1.0      # importance only on the perfect channels
    per = sq.weighted_rmse(w, dq, im, 64)
    assert per[0] < 1e-6
    # uniform importance -> nonzero
    per_u = sq.weighted_rmse(w, dq, np.ones(32, dtype=np.float32), 64)
    assert per_u[0] > 1.0


def test_weighted_rmse_groups():
    w = np.zeros((4, 8, 32), dtype=np.float32)  # (groups, rows, channels)
    dq = w.copy()
    dq[2] = 5.0           # group 2 has error in all its rows/channels
    im = np.ones(32, dtype=np.float32)
    aw, *_ = sq._channels_last(w)
    ad, *_ = sq._channels_last(dq)
    per = sq.weighted_rmse(aw, ad, im, 8)
    assert per.shape == (4,)
    np.testing.assert_allclose(per[[0, 1, 3]], 0.0, atol=1e-6)
    np.testing.assert_allclose(per[2], 5.0, rtol=1e-5)


def test_channels_last_layout():
    # reader order (groups..., rows, channels), C-contiguous like GGUFReader
    data = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    a, n_rows, n_channels, rpg, ng = sq._channels_last(data)
    assert (n_rows, n_channels, rpg, ng) == (12, 2, 3, 4)
    # group-major row order: A[g*rpg + r, c] = data[g, r, c]
    assert a[3 * 3 + 2, 1] == data[3, 2, 1]


def test_score_tensor_orders_types():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(64, 128)).astype(np.float32)
    prof = {"t.weight": np.ones(64, dtype=np.float32)}
    s = sq.score_tensor("t.weight", data, prof, ("q4_0", "q5_0", "q8_0"))
    assert not s.skipped
    assert s.err["q4_0"] > s.err["q5_0"] > s.err["q8_0"] > 0
    # measured packed sizes match ggml block constants (channels = last axis)
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType as T
    rows, channels = 64, 128
    for tname in ("q4_0", "q5_0", "q8_0"):
        block, tsize = GGML_QUANT_SIZES[T[tname.upper()]]
        assert s.packed_bytes[tname] == rows * (channels // block) * tsize


def test_score_tensor_weighting_changes_ranking():
    """Quant error scales with value magnitude, so a tensor whose large values
    sit on zero-importance channels must score better (weighted) than the same
    energy placed on important channels - while scoring equal unweighted.
    Importance split is aligned to Q8_0's 32-channel blocks so block-coupled
    scale effects don't leak error across the boundary."""
    rng = np.random.default_rng(1)
    clean = rng.normal(size=(64, 128)).astype(np.float32) * 0.01
    im = np.zeros(128, dtype=np.float32)
    im[32:] = 1.0
    # A: big values on the UNimportant channels (0-31, one full Q8_0 block)
    a = clean.copy()
    a[:, :32] += rng.normal(size=(64, 32)).astype(np.float32) * 0.25
    # B: same total energy on the important channels (32-127)
    b = clean.copy()
    b[:, 32:] += rng.normal(size=(64, 96)).astype(np.float32) * 0.25 * np.sqrt(32 / 96)
    sa = sq.score_tensor("a.weight", a, {"a.weight": im.reshape(1, -1)}, ("q8_0",))
    sb = sq.score_tensor("b.weight", b, {"b.weight": im.reshape(1, -1)}, ("q8_0",))
    sau = sq.score_tensor("a.weight", a, None, ("q8_0",))
    sbu = sq.score_tensor("b.weight", b, None, ("q8_0",))
    assert sa.err["q8_0"] < sb.err["q8_0"]          # weighting separates them
    assert abs(sau.err["q8_0"] - sbu.err["q8_0"]) < 0.2 * sbu.err["q8_0"]  # unweighted ~equal


def test_score_tensor_skips_1d():
    s = sq.score_tensor("n.weight", np.ones(64, dtype=np.float32), None)
    assert s.skipped


def test_score_tensor_missing_profile_uses_uniform():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(64, 128)).astype(np.float32)
    s1 = sq.score_tensor("t.weight", data, None)
    s2 = sq.score_tensor("t.weight", data, {"other": np.ones(64, dtype=np.float32).reshape(1, -1)})
    assert s1.err == s2.err


# ---------------------------------------------------------------------------
# size model + assignment
# ---------------------------------------------------------------------------

def _mk_score(name, shape, err=None, sensitive=False, skipped=""):
    rpg = int(shape[-2]) if len(shape) >= 2 else 1
    ng = 1
    for d in shape[:-2]:
        ng *= int(d)
    s = sq.TensorScore(name=name, shape=shape, n_rows=rpg * ng,
                       n_channels=int(shape[-1]), n_groups=ng, rows_per_group=rpg,
                       skipped=skipped)
    if err:
        s.err = err
        s.packed_bytes = {k: sq.type_bytes(shape, k) for k in err}
    if sensitive:
        # make is_sensitive true by including a pattern in the real name instead
        pass
    return s


def test_type_bytes_matches_ggml():
    # q4_k: 144 bytes per 256-channel block, per row
    assert sq.type_bytes((10, 256), "q4_k") == 10 * 144
    # MoE exps reader-order (experts, rows, channels)
    assert sq.type_bytes((128, 704, 2816), "q6_k") == (128 * 704) * (2816 // 256) * 210
    assert sq.type_bytes((100,), "f32") == 400


def test_sensitive_detection():
    assert _mk_score("token_embd.weight", (64, 32)).is_sensitive
    assert _mk_score("output.weight", (64, 32)).is_sensitive
    assert _mk_score("blk.0.ffn_gate_inp.weight", (32, 128)).is_sensitive
    assert not _mk_score("blk.0.attn_q.weight", (64, 32)).is_sensitive


def test_k_ladder_upgrades_most_sensitive():
    # 5 identical-shape tensors, distinct sensitivities; generous budget
    shapes = (8, 256)
    scores = [
        _mk_score(f"blk.{i}.attn_q.weight", shapes, err={"q4_0": 0.001 * (i + 1)})
        for i in range(5)
    ]
    base = sq.type_bytes(shapes, "q4_k")
    budget = 5 * base + 2 * (sq.type_bytes(shapes, "q5_k") - base)  # room for 2 upgrades
    assignment, report = sq.assign_k_ladder(scores, budget)
    upgraded = {n for n, t in assignment.items() if t == "q5_k"}
    assert "blk.4.attn_q.weight" in upgraded  # most sensitive upgraded
    assert "blk.0.attn_q.weight" not in upgraded
    assert all(t in sq.K_LADDER for t in assignment.values())
    assert any("upgraded" in r for r in report)


def test_k_ladder_downgrades_least_sensitive_when_over_budget():
    shapes = (8, 256)
    scores = [
        _mk_score(f"blk.{i}.attn_q.weight", shapes, err={"q4_0": 0.001 * (i + 1)})
        for i in range(5)
    ]
    base = sq.type_bytes(shapes, "q4_k")
    budget = 5 * base - (base - sq.type_bytes(shapes, "q3_k"))  # force one downgrade
    assignment, _ = sq.assign_k_ladder(scores, budget)
    base_i = sq.K_LADDER.index("q4_k")
    downgraded = {n for n, t in assignment.items() if sq.K_LADDER.index(t) < base_i}
    assert "blk.0.attn_q.weight" in downgraded
    # water-fill spreads downgrades round-robin: nobody sits above the most
    # sensitive tensor, and the least sensitive move at least as far down
    idx = {n: sq.K_LADDER.index(t) for n, t in assignment.items()}
    assert idx["blk.0.attn_q.weight"] <= idx["blk.4.attn_q.weight"]


def test_k_ladder_pins_sensitive():
    shapes = (8, 256)
    scores = [
        _mk_score("token_embd.weight", shapes, err={"q4_0": 999.0}),
        _mk_score("blk.0.attn_q.weight", shapes, err={"q4_0": 0.001}),
    ]
    assignment, _ = sq.assign_k_ladder(scores, 10**9)
    assert assignment["token_embd.weight"] == "q6_k"


def test_k_ladder_counts_skipped_existing_bytes():
    shapes = (8, 256)
    s_skip = _mk_score("already.quant", shapes, skipped="already Q4_K", )
    s_skip.existing_bytes = 12345
    s = _mk_score("blk.0.attn_q.weight", shapes, err={"q4_0": 0.01})
    assignment, report = sq.assign_k_ladder([s_skip, s], 10**9)
    assert "already.quant" not in assignment


def test_legacy_assignment_uses_best_ratio():
    shapes = (8, 256)
    # unambiguous ratios: big_drop's q8_0 drop-per-byte beats everything else
    big_drop = _mk_score("blk.0.attn_q.weight", shapes,
                         err={"q4_0": 0.10, "q5_0": 0.099, "q8_0": 0.001})
    small_drop = _mk_score("blk.1.attn_q.weight", shapes,
                           err={"q4_0": 0.10, "q5_0": 0.095, "q8_0": 0.09})
    budget = 2 * sq.type_bytes(shapes, "q4_0") + (sq.type_bytes(shapes, "q8_0") - sq.type_bytes(shapes, "q4_0"))
    assignment, report = sq.assign_legacy([big_drop, small_drop], budget)
    # budget fits only ONE q8_0 upgrade -> it must go to the bigger error drop
    assert assignment["blk.0.attn_q.weight"] == "q8_0"
    assert assignment["blk.1.attn_q.weight"] == "q4_0"


# ---------------------------------------------------------------------------
# tensor-type file emission
# ---------------------------------------------------------------------------

def test_emit_tensor_type_file_format():
    content = sq.emit_tensor_type_file({"blk.0.attn_q.weight": "q5_k", "token_embd.weight": "q6_k"})
    lines = content.strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        name, t = line.split("=")
        assert t in sq.K_LADDER or t in ("q8_0", "q6_k")
        assert name == name.lower()


def test_emit_escapes_regex_dots():
    content = sq.emit_tensor_type_file({"blk.0.attn_q.weight": "q4_k"})
    assert content.strip() == r"blk\.0\.attn_q\.weight=q4_k"


def test_moe_expert_report():
    s = _mk_score("blk.0.ffn_down_exps.weight", (64, 8, 4))
    s.group_err = {"q4_0": [0.1, 0.1, 0.5, 0.1]}
    lines = sq.moe_expert_report([s])
    assert lines and "expert 2" in lines[0] and "spread" in lines[0]


def test_sensitive_name_does_not_match_attn_output():
    assert sq.is_sensitive_name("output.weight")
    assert sq.is_sensitive_name("token_embd.weight")
    assert sq.is_sensitive_name("per_layer_token_embd.weight")
    assert sq.is_sensitive_name("blk.0.ffn_gate_inp.weight")
    assert not sq.is_sensitive_name("blk.0.attn_output.weight")
    assert not sq.is_sensitive_name("blk.0.attn_q.weight")
    assert not _mk_score("blk.0.attn_output.weight", (64, 32)).is_sensitive


def test_k_ladder_never_assigns_256_block_rung_to_704_channels():
    # Gemma-4 ffn_down_exps has 704 channels: q*_k / iq4_xs would silently
    # fall back to Q4_0 inside llama-quantize, wrecking the size model.
    scores = [
        _mk_score("blk.0.ffn_down_exps.weight", (8, 64, 704), err={"q4_0": 0.01}),
        _mk_score("blk.0.ffn_down.weight", (8, 2112), err={"q4_0": 0.02}),
        _mk_score("blk.0.attn_q.weight", (8, 2816), err={"q4_0": 0.5}),
    ]
    assignment, _ = sq.assign_k_ladder(scores, 10**9)
    blocked_256 = {t for t, b in sq.TYPE_BLOCK.items() if b == 256}
    for name in ("blk.0.ffn_down_exps.weight", "blk.0.ffn_down.weight"):
        assert assignment[name] not in blocked_256, assignment[name]
    # 2816-channel tensors keep full ladder access (unlimited budget -> top rung)
    assert assignment["blk.0.attn_q.weight"] == "q8_0"
    # and allowed_rungs itself agrees
    assert not (set(sq.allowed_rungs((8, 64, 704))) & blocked_256)


def test_family_budget_bytes_math():
    # budget = file_bpw * params / 8; unknown labels fall back to ~4 bpw
    assert sq.family_budget_bytes("~4 bpw", 8) == int(4.3 * 8 / 8)
    assert sq.family_budget_bytes("~3 bpw", 8_000_000_000) == int(3.3 * 8e9 / 8)
    assert sq.family_budget_bytes("bogus", 8) == sq.family_budget_bytes("~4 bpw", 8)
    for fam, bpw in sq.FAMILY_FILE_BPW.items():
        got = sq.family_budget_bytes(fam, 25_233_000_000)
        assert abs(got / 25.233e9 * 8 - bpw) < 1e-9


def _has_two_zone(report: list[str]) -> bool:
    return any("two-zone" in line for line in report)


def test_dense_model_uses_uniform_layout():
    # no 3-D expert stacks -> uniform water-fill from base, nobody at q8_0
    shapes = (8, 256)
    scores = [
        _mk_score(f"blk.{i}.attn_q.weight", shapes, err={"q4_0": 0.001 * (i + 1)})
        for i in range(5)
    ]
    base = sq.type_bytes(shapes, "q4_k")
    assignment, report = sq.assign_k_ladder(scores, 5 * base)
    assert not _has_two_zone(report)
    assert set(assignment.values()) == {"q4_k"}


def test_moe_model_uses_two_zone_when_experts_dominate():
    # small attention weights + big expert stacks -> non-experts start at q8_0
    attn = _mk_score("blk.0.attn_q.weight", (8, 256), err={"q4_0": 0.5})
    exps = _mk_score("blk.0.ffn_gate_up_exps.weight", (4, 64, 256), err={"q4_0": 0.01})
    # exact-fit budget: attn @ q8_0 + expert @ q4_k -> no upgrade headroom
    budget = sq.type_bytes((8, 256), "q8_0") + sq.type_bytes((4, 64, 256), "q4_k")
    assignment, report = sq.assign_k_ladder([attn, exps], budget)
    assert _has_two_zone(report)
    assert assignment["blk.0.attn_q.weight"] == "q8_0"
    # expert stays at base: no headroom to upgrade
    assert assignment["blk.0.ffn_gate_up_exps.weight"] == "q4_k"


def test_moe_falls_back_to_uniform_when_nonexperts_are_fat():
    # fat dense weights: q8_0 on all non-experts would exceed 55% of budget
    # -> uniform layout, non-experts start at base like everyone else
    big = (64, 256)
    attn = _mk_score("blk.0.attn_q.weight", big, err={"q4_0": 0.5})
    exps = _mk_score("blk.0.ffn_gate_up_exps.weight", (4, 64, 256), err={"q4_0": 0.01})
    base = sq.type_bytes(big, "q4_k")
    budget = int(2.5 * base)  # q8_0 on `attn` alone is ~1.9x base -> >55%
    assignment, report = sq.assign_k_ladder([attn, exps], budget)
    assert not _has_two_zone(report)
    assert assignment["blk.0.attn_q.weight"] != "q8_0"
