import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

pytest.importorskip("gguf", reason="the analysis decodes with gguf-py")

from quant_gui import kquant_plan as kp


def _entry(name, rows, cols, scale, seed, outliers=False):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((rows, cols)) * scale).astype(np.float32)
    if outliers:
        w[:, ::37] *= 25
    return kp.TensorEntry(name, rows, cols, lambda idx, w=w: w[idx])


@pytest.fixture(scope="module")
def measures():
    entries = [
        _entry("blocks.0.attn.wq.weight", 128, 512, 0.02, 1),
        _entry("blocks.0.mlp.down.weight", 128, 1024, 0.02, 2, outliers=True),
        _entry("blocks.1.attn.wq.weight", 128, 512, 0.2, 3),            # 10x larger weights
        _entry("blocks.1.mlp.down.weight", 128, 1024, 0.02, 4),
        _entry("blocks.2.attn.wq.weight", 128, 512, 0.02, 5),
        _entry("blocks.2.mlp.down.weight", 128, 1024, 0.01, 6),
    ]
    m, skipped = kp.measure(entries, sample_rows=64)
    assert not skipped
    return m


def test_measure_error_falls_with_bits_and_scales_to_the_whole_tensor(measures):
    for m in measures:
        errs = [m.err[t] for t in kp.LADDER]
        assert errs == sorted(errs, reverse=True) and len(errs) == 6, m.name
        assert m.sample_rows == 64 and m.rows == 128
    # sample of 64 of 128 rows is scaled x2: norm2 is about the full tensor's sum of squares
    e = _entry("x", 128, 512, 0.02, 1)
    full = float((e.get_rows(np.arange(128)).astype(np.float64) ** 2).sum())
    assert abs(measures[0].norm2 / full - 1) < 0.15


def test_non_finite_tensors_are_skipped_not_poisoning_the_plan():
    bad = np.ones((8, 256), np.float32)
    bad[2, 3] = np.nan
    m, skipped = kp.measure([kp.TensorEntry("bad", 8, 256, lambda i: bad[i]),
                             _entry("ok", 8, 256, 0.02, 1)], sample_rows=8)
    assert skipped == ["bad"] and [x.name for x in m] == ["ok"]


def test_columns_not_a_multiple_of_256_only_offer_q8_0():
    m, _ = kp.measure([_entry("odd", 8, 288, 0.02, 1)], sample_rows=8)
    assert set(m[0].err) == {"Q8_0"}


def test_hull_drops_dominated_and_non_convex_points():
    h = kp._hull([(10, 100.0, "a"), (20, 90.0, "b"), (30, 20.0, "c"), (40, 25.0, "dominated"), (50, 10.0, "d")])
    names = [t for _, _, t in h]
    assert "dominated" not in names                      # more bytes, more error
    assert "b" not in names                              # above the segment a..c: not convex
    assert names[0] == "a" and names[-1] == "d"
    rates = [(h[i][1] - h[i + 1][1]) / (h[i + 1][0] - h[i][0]) for i in range(len(h) - 1)]
    assert rates == sorted(rates, reverse=True)


def test_allocation_respects_budget_is_monotone_and_beats_uniform(measures):
    ub, ue = kp.uniform(measures, "Q4_K")
    plan = kp.allocate(measures, ub)
    assert plan.fits and plan.total_bytes <= ub
    assert plan.total_err <= ue + 1e-9                    # same size, no more error than uniform Q4_K
    assert plan.total_err < ue * 0.999                    # and strictly better: the tensors differ
    last_b, last_e = 0, float("inf")
    for gb in np.linspace(plan.min_bytes, plan.max_bytes, 12):
        p = kp.allocate(measures, int(gb))
        assert p.total_bytes <= int(gb) and p.total_bytes >= last_b and p.total_err <= last_e + 1e-9
        last_b, last_e = p.total_bytes, p.total_err


def test_big_weights_get_more_bits_than_small_ones(measures):
    ub, _ = kp.uniform(measures, "Q4_K")
    a = kp.allocate(measures, ub).assign
    order = {t: i for i, t in enumerate(kp.LADDER)}
    assert order[a["blocks.1.attn.wq.weight"]] >= order[a["blocks.0.attn.wq.weight"]]   # 10x larger weights


def test_close_to_the_brute_force_optimum_on_a_small_case(measures):
    sub = measures[:4]
    types = ("Q3_K", "Q4_K", "Q6_K")
    budget = int(sum(m.bytes_for("Q4_K") for m in sub))
    best = min(
        sum(m.err[t] for m, t in zip(sub, combo))
        for combo in itertools.product(types, repeat=len(sub))
        if sum(m.bytes_for(t) for m, t in zip(sub, combo)) <= budget
    )
    plan = kp.allocate(sub, budget, floor="Q3_K", ceiling="Q6_K")
    assert plan.total_err <= best * 1.10                 # greedy-on-hulls is near-optimal, not exact


def test_infeasible_and_generous_budgets_are_reported(measures):
    tiny = kp.allocate(measures, 1000)
    assert not tiny.fits and set(tiny.assign.values()) == {"Q2_K"} and any("lowest type" in n for n in tiny.notes)
    big = kp.allocate(measures, 10 ** 12)
    assert set(big.assign.values()) == {"Q8_0"} and any("every tensor got it" in n for n in big.notes)


def test_floor_and_ceiling_are_honoured(measures):
    ub, _ = kp.uniform(measures, "Q4_K")
    p = kp.allocate(measures, ub, floor="Q3_K", ceiling="Q5_K")
    assert set(p.assign.values()) <= {"Q3_K", "Q4_K", "Q5_K"}


def test_fixed_bytes_count_against_the_budget(measures):
    ub, _ = kp.uniform(measures, "Q4_K")
    p = kp.allocate(measures, ub + 5_000_000, fixed_bytes=5_000_000)
    assert p.total_bytes <= ub + 5_000_000 and p.total_bytes >= 5_000_000


def test_report_plan_file_roundtrip_and_validation(measures, tmp_path):
    ub, _ = kp.uniform(measures, "Q4_K")
    plan = kp.allocate(measures, ub)
    text = kp.report(measures, plan)
    assert "uniform Q4_K" in text and "mixed plan" in text and "not an image-quality measurement" in text
    assert "by block index" in text and "types chosen" in text
    f = tmp_path / "p.json"
    kp.save_plan(f, plan, {"source": "x"})
    assert kp.load_plan(f) == plan.assign
    f.write_text(json.dumps({"format": "other", "types": {}}))
    with pytest.raises(ValueError, match="not a kquant-plan-1"):
        kp.load_plan(f)
    f.write_text(json.dumps({"format": kp.PLAN_FORMAT, "types": {"a": "Q4_K_M"}}))
    with pytest.raises(ValueError, match="unsupported types"):
        kp.load_plan(f)
    with pytest.raises(ValueError, match="cannot read"):
        kp.load_plan(tmp_path / "missing.json")
