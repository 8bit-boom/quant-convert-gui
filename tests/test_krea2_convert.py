"""Tests for tools/convert_krea2_to_gguf.py - precision rules, renaming,
and an end-to-end synthetic safetensors -> GGUF round-trip."""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import convert_krea2_to_gguf as k2g  # noqa: E402

from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf import GGUFReader  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_safetensors(path: Path, tensors: dict[str, tuple[np.ndarray, str]]):
    """tensors: name -> (numpy array, dtype string like 'F32'/'BF16'/'F16')"""
    header = {}
    blobs = []
    off = 0
    for name, (arr, dt) in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


# ---------------------------------------------------------------------------
# precision rules
# ---------------------------------------------------------------------------

def test_pick_qtype_priority_rules():
    # 1-D -> F32 even for BF16 source
    assert k2g.pick_qtype("blocks.0.prenorm.scale", (6144,), "BF16", "q8_0") == QT.F32
    # <=1024 elems -> F32
    assert k2g.pick_qtype("txtfusion.projector.weight", (1, 12), "F32", "q8_0") == QT.F32
    # hiprec prefixes -> F32 (and tmlp. must not be confused with txtmlp.)
    for p in ("first.weight", "last.linear.weight", "tproj.1.weight", "tmlp.0.weight",
              "txtmlp.1.weight", "txtfusion.projector.weight"):
        assert k2g.pick_qtype(p, (6144, 256), "F32", "q8_0") == QT.F32, p
    # big BF16 2-D linear -> requested quant
    assert k2g.pick_qtype("blocks.0.attn.wq.weight", (6144, 6144), "BF16", "q8_0") == QT.Q8_0
    assert k2g.pick_qtype("blocks.0.mlp.down.weight", (6144, 16384), "BF16", "q4_0") == QT.Q4_0
    assert k2g.pick_qtype("txtfusion.refiner_blocks.3.mlp.up.weight", (6912, 2560), "BF16", "q5_1") == QT.Q5_1
    # BF16 passthrough when quant disabled
    assert k2g.pick_qtype("blocks.0.attn.wq.weight", (6144, 6144), "BF16", None) == QT.BF16
    # big F32 2-D non-hiprec -> F16
    assert k2g.pick_qtype("blocks.0.something.weight", (256, 256), "F32", "q8_0") == QT.F16
    # channels not divisible by the 32-block -> no legacy quant, BF16 instead
    assert k2g.pick_qtype("blocks.0.attn.wq.weight", (6144, 100), "BF16", "q8_0") == QT.BF16


# ---------------------------------------------------------------------------
# diffusers renaming
# ---------------------------------------------------------------------------

def test_rename_diffusers_keys():
    r = k2g.rename_diffusers_key
    assert r("img_in.weight") == "first.weight"
    assert r("time_embed.linear_1.bias") == "tmlp.0.bias"
    assert r("time_mod_proj.weight") == "tproj.1.weight"
    assert r("txt_in.norm.weight") == "txtmlp.0.scale"
    assert r("text_fusion.layerwise_blocks.2.attn.to_q.weight") == "txtfusion.layerwise_blocks.2.attn.wq.weight"
    assert r("transformer_blocks.5.norm1.weight") == "blocks.5.prenorm.scale"
    assert r("transformer_blocks.5.norm2.weight") == "blocks.5.postnorm.scale"
    assert r("transformer_blocks.5.attn.to_out.0.weight") == "blocks.5.attn.wo.weight"
    assert r("transformer_blocks.5.attn.norm_q.weight") == "blocks.5.attn.qknorm.qnorm.scale"
    assert r("transformer_blocks.5.ff.gate.weight") == "blocks.5.mlp.gate.weight"
    assert r("transformer_blocks.7.scale_shift_table") == "blocks.7.mod.lin"
    assert r("final_layer.scale_shift_table") == "last.modulation.lin"
    assert r("final_layer.norm.weight") == "last.norm.scale"
    # ComfyUI-native keys pass through untouched
    assert r("blocks.0.attn.wq.weight") == "blocks.0.attn.wq.weight"
    assert r("txtfusion.projector.weight") == "txtfusion.projector.weight"


def test_looks_like_diffusers():
    assert k2g.looks_like_diffusers(["transformer_blocks.0.attn.to_q.weight"])
    assert k2g.looks_like_diffusers(["img_in.weight"])
    assert not k2g.looks_like_diffusers(["blocks.0.attn.wq.weight", "first.weight"])


# ---------------------------------------------------------------------------
# end-to-end: synthetic safetensors -> GGUF -> readback
# ---------------------------------------------------------------------------

def _synth_tensors():
    rng = np.random.default_rng(7)
    big = rng.standard_normal((256, 256), dtype=np.float32)
    return {
        "blocks.0.attn.wq.weight": (big, "BF16"),
        "blocks.0.prenorm.scale": (np.ones(256, dtype=np.float32), "F32"),
        "first.weight": (rng.standard_normal((256, 64)).astype(np.float32), "F32"),
        "txtfusion.projector.weight": (np.zeros((1, 12), dtype=np.float32), "F32"),
        "blocks.0.extra.weight": (rng.standard_normal((256, 256)).astype(np.float32), "F32"),
    }


def _to_bf16_storage(f32: np.ndarray) -> np.ndarray:
    """Return the uint16 raw storage bytes content for a float32 array."""
    u32 = f32.astype(np.float32).view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


@pytest.mark.parametrize("quant", ["q8_0", "bf16"])
def test_convert_end_to_end(tmp_path, quant):
    src = tmp_path / "krea2_tiny.safetensors"
    dst = tmp_path / "krea2_tiny.gguf"
    tensors = _synth_tensors()
    # store BF16 tensor as raw uint16 bits under dtype BF16
    stored = {}
    for name, (arr, dt) in tensors.items():
        if dt == "BF16":
            stored[name] = (_to_bf16_storage(arr), "BF16")
        else:
            stored[name] = (arr, dt)
    _write_safetensors(src, stored)

    k2g.convert(src, dst, None if quant == "bf16" else quant)

    r = GGUFReader(str(dst))
    assert bytes(r.fields["general.architecture"].parts[-1]).decode() == "krea2"
    by_name = {t.name: t for t in r.tensors}
    assert set(by_name) == set(tensors)

    from gguf import quants
    if quant == "q8_0":
        assert by_name["blocks.0.attn.wq.weight"].tensor_type == QT.Q8_0
    else:
        assert by_name["blocks.0.attn.wq.weight"].tensor_type == QT.BF16
    assert by_name["blocks.0.prenorm.scale"].tensor_type == QT.F32
    assert by_name["first.weight"].tensor_type == QT.F32          # hiprec
    assert by_name["txtfusion.projector.weight"].tensor_type == QT.F32  # tiny
    assert by_name["blocks.0.extra.weight"].tensor_type == QT.F16  # big F32 2-D

    if quant == "bf16":
        want = tensors["blocks.0.attn.wq.weight"][0]
        # The source storage itself is truncated-to-bf16 (see
        # _to_bf16_storage), and gguf-py's BF16 quantizer truncates too,
        # so the round trip must be BIT-EXACT against the truncated source.
        want_trunc = (want.view(np.uint32) & 0xFFFF0000).view(np.float32)
        got = quants.dequantize(
            np.asarray(by_name["blocks.0.attn.wq.weight"].data).reshape(-1), QT.BF16
        ).reshape(256, 256)
        np.testing.assert_array_equal(got, want_trunc)


def test_fp8_source_rejected(tmp_path):
    src = tmp_path / "fp8.safetensors"
    _write_safetensors(src, {"blocks.0.attn.wq.weight": (np.zeros((32, 32), np.uint8), "F8_E4M3")})
    st = k2g.SafetensorsFile(src)
    with pytest.raises(ValueError, match="FP8"):
        st.load("blocks.0.attn.wq.weight")


def test_bf16_round_trip_lossless():
    rng = np.random.default_rng(3)
    f32 = rng.standard_normal((64, 64)).astype(np.float32)
    back = k2g.bf16_to_f32(_to_bf16_storage(f32))
    # truncation drops the low 16 bits exactly
    np.testing.assert_array_equal(back.view(np.uint32) & 0xFFFF0000,
                                  f32.view(np.uint32) & 0xFFFF0000)


# ---------------------------------------------------------------------------
# K-quants (written by quant_gui/kquants.py, not gguf-py)
# ---------------------------------------------------------------------------

def test_pick_qtype_kquants_and_fallbacks():
    pq = k2g.pick_qtype
    assert pq("blocks.0.attn.wq.weight", (6144, 6144), "BF16", "q4_k_m") == QT.Q4_K
    assert pq("blocks.0.attn.wq.weight", (6144, 6144), "BF16", "q4_k_s") == QT.Q4_K      # same type, label only
    assert pq("blocks.0.attn.wq.weight", (6144, 6144), "BF16", "q3_k_l") == QT.Q3_K
    assert pq("blocks.0.attn.wq.weight", (6144, 6144), "BF16", "q2_k") == QT.Q2_K
    assert pq("blocks.0.attn.wq.weight", (6144, 6144), "BF16", "q6_k") == QT.Q6_K
    # row length fits the 32-blocks only -> llama.cpp's legacy fallback
    assert pq("blocks.0.attn.wq.weight", (6144, 800), "BF16", "q4_k_m") == QT.Q5_0
    assert pq("blocks.0.attn.wq.weight", (6144, 800), "BF16", "q5_k_m") == QT.Q5_1
    assert pq("blocks.0.attn.wq.weight", (6144, 800), "BF16", "q6_k") == QT.Q8_0
    assert pq("blocks.0.attn.wq.weight", (6144, 800), "BF16", "q2_k") == QT.Q4_0
    # not even a multiple of 32 -> BF16, never a K type
    assert pq("blocks.0.attn.wq.weight", (6144, 100), "BF16", "q4_k_m") == QT.BF16
    # the precision rules still win over the quant
    assert pq("first.weight", (6144, 6144), "BF16", "q4_k_m") == QT.F32
    assert pq("blocks.0.prenorm.scale", (6144,), "BF16", "q4_k_m") == QT.F32


@pytest.mark.parametrize("quant,want", [("q4_k_m", QT.Q4_K), ("q4_k_s", QT.Q4_K), ("q4_k_l", QT.Q4_K),
                                        ("q5_k_m", QT.Q5_K), ("q5_k_l", QT.Q5_K), ("q6_k", QT.Q6_K),
                                        ("q6_k_l", QT.Q6_K), ("q3_k_s", QT.Q3_K), ("q2_k", QT.Q2_K),
                                        ("q2_k_l", QT.Q2_K)])
def test_convert_end_to_end_kquant(tmp_path, quant, want):
    rng = np.random.default_rng(11)
    w = (rng.standard_normal((256, 512)) * 0.02).astype(np.float32)
    odd = (rng.standard_normal((64, 800)) * 0.02).astype(np.float32)
    src = tmp_path / "krea2_tiny.safetensors"
    _write_safetensors(src, {
        "blocks.0.attn.wq.weight": (_to_bf16_storage(w), "BF16"),
        "blocks.0.mlp.odd.weight": (_to_bf16_storage(odd), "BF16"),
        "blocks.0.prenorm.scale": (np.ones(256, dtype=np.float32), "F32"),
        "first.weight": (rng.standard_normal((256, 64)).astype(np.float32), "F32"),
    })
    dst = tmp_path / f"{quant}.gguf"
    k2g.convert(src, dst, quant)

    r = GGUFReader(str(dst))
    by_name = {t.name: t for t in r.tensors}
    assert by_name["blocks.0.attn.wq.weight"].tensor_type == want
    assert by_name["blocks.0.mlp.odd.weight"].tensor_type == {
        QT.Q4_K: QT.Q5_0, QT.Q5_K: QT.Q5_1, QT.Q6_K: QT.Q8_0, QT.Q3_K: QT.Q4_0, QT.Q2_K: QT.Q4_0}[want]
    assert by_name["blocks.0.prenorm.scale"].tensor_type == QT.F32
    assert by_name["first.weight"].tensor_type == QT.F32
    # the shape the loader sees is the logical (rows, cols), not the packed byte shape
    assert list(by_name["blocks.0.attn.wq.weight"].shape) == [512, 256]

    from gguf import quants
    ref = (_to_bf16_storage(w).astype(np.uint32) << 16).view(np.float32)       # what the converter saw
    got = quants.dequantize(np.asarray(by_name["blocks.0.attn.wq.weight"].data), want).reshape(ref.shape)
    rel = float(np.sqrt(((got - ref) ** 2).mean()) / np.sqrt((ref ** 2).mean()))
    assert rel < {QT.Q2_K: 0.4, QT.Q3_K: 0.2, QT.Q4_K: 0.1, QT.Q5_K: 0.05, QT.Q6_K: 0.03}[want]


def test_kquant_labels_of_one_family_are_the_same_bytes(tmp_path):
    rng = np.random.default_rng(2)
    w = (rng.standard_normal((256, 512)) * 0.02).astype(np.float32)
    src = tmp_path / "t.safetensors"
    _write_safetensors(src, {"blocks.0.attn.wq.weight": (_to_bf16_storage(w), "BF16")})
    outs = []
    for q in ("q4_k_s", "q4_k_m", "q4_k_l"):
        dst = tmp_path / f"{q}.gguf"
        k2g.convert(src, dst, q)
        outs.append(np.asarray(next(t for t in GGUFReader(str(dst)).tensors if t.name.endswith("wq.weight")).data).tobytes())
    assert outs[0] == outs[1] == outs[2]


def test_truncated_source_is_reported_plainly_not_as_a_reshape_error(tmp_path):
    src = tmp_path / "full.safetensors"
    _write_safetensors(src, {"blocks.0.attn.wq.weight": (_to_bf16_storage(np.ones((64, 64), np.float32)), "BF16")})
    cut = tmp_path / "cut.safetensors"
    cut.write_bytes(src.read_bytes()[:-1000])            # an upload/download that stopped early

    with pytest.raises(ValueError, match="incomplete.*cut.safetensors") as ei:
        k2g.SafetensorsFile(cut)
    assert "bytes" in str(ei.value) and "missing" in str(ei.value)
    # the CLI prints the message and exits 1 instead of a traceback
    assert k2g.main(["--src", str(cut), "--dst", str(tmp_path / "o.gguf")]) == 1
    assert k2g.SafetensorsFile(src).keys() == ["blocks.0.attn.wq.weight"]      # a complete file is fine


# ---------------------------------------------------------------------------
# mixed K-quant plan: --analyze then --plan
# ---------------------------------------------------------------------------

def _planning_model(path: Path):
    rng = np.random.default_rng(21)
    t = {}
    for b in range(3):
        for n, shp, sc in (("attn.wq", (256, 512), 0.02), ("mlp.down", (512, 1024), 0.01), ("mlp.up", (1024, 512), 0.02)):
            w = rng.standard_normal(shp).astype(np.float32) * sc * (8 if (b, n) == (0, "attn.wq") else 1)
            t[f"blocks.{b}.{n}.weight"] = (_to_bf16_storage(w), "BF16", w)
    t["blocks.0.prenorm.scale"] = (np.ones(512, np.float32), "F32", None)
    _write_safetensors(path, {k: (v[0], v[1]) for k, v in t.items()})
    return t


def test_analyze_writes_a_plan_that_conversion_applies_and_that_beats_uniform(tmp_path, capsys):
    from quant_gui import kquant_plan as kp
    src = tmp_path / "k2.safetensors"
    _planning_model(src)
    plan_f = tmp_path / "plan.json"

    assert k2g.main(["--src", str(src), "--analyze", str(plan_f), "--sample-rows", "32"]) == 0
    text = capsys.readouterr().out
    assert "mixed plan" in text and "not an image-quality measurement" in text
    plan = kp.load_plan(plan_f)
    assert set(plan) == {k for k in _planning_model(tmp_path / "again.safetensors") if k.endswith("weight")}
    assert len(set(plan.values())) > 1                       # tensors differ, so the plan is genuinely mixed
    assert plan["blocks.0.attn.wq.weight"] != "Q2_K"          # the 8x larger weights are not starved

    mixed, uni = tmp_path / "mixed.gguf", tmp_path / "uni.gguf"
    assert k2g.main(["--src", str(src), "--dst", str(mixed), "--quant", "q4_k_m", "--plan", str(plan_f)]) == 0
    assert k2g.main(["--src", str(src), "--dst", str(uni), "--quant", "q4_k_m"]) == 0
    assert mixed.stat().st_size <= uni.stat().st_size * 1.01          # budget = size of uniform Q4_K

    from gguf import quants
    ref = {k: v[2] for k, v in _planning_model(tmp_path / "ref.safetensors").items() if v[2] is not None}
    bf = {k: (_to_bf16_storage(w).astype(np.uint32) << 16).view(np.float32) for k, w in ref.items()}

    def total_err(path):
        sse = n2 = 0.0
        for t in GGUFReader(str(path)).tensors:
            if t.name in bf:
                got = quants.dequantize(np.asarray(t.data), t.tensor_type).reshape(bf[t.name].shape)
                sse += float(((got.astype(np.float64) - bf[t.name]) ** 2).sum())
                n2 += float((bf[t.name].astype(np.float64) ** 2).sum())
        return (sse / n2) ** 0.5

    got_types = {t.name: t.tensor_type.name for t in GGUFReader(str(mixed)).tensors}
    assert all(got_types[k] == v for k, v in plan.items())            # every planned tensor got its planned type
    assert total_err(mixed) < total_err(uni)                          # lower weight error at (about) the same size


def test_plan_for_another_model_is_refused_and_bad_plan_files_are_messages(tmp_path, capsys):
    from quant_gui import kquant_plan as kp
    src = tmp_path / "k2.safetensors"
    _planning_model(src)
    other = tmp_path / "other.json"
    other.write_text('{"format": "kquant-plan-1", "types": {"not.a.tensor.weight": "Q4_K"}}')
    with pytest.raises(SystemExit, match="matches no tensor"):
        k2g.main(["--src", str(src), "--dst", str(tmp_path / "o.gguf"), "--plan", str(other)])
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert k2g.main(["--src", str(src), "--dst", str(tmp_path / "o.gguf"), "--plan", str(bad)]) == 1
    assert "not a kquant-plan-1" in capsys.readouterr().err
    assert k2g.main(["--src", str(src)]) == 1                          # neither --dst nor --analyze


def test_pick_qtype_follows_a_plan_only_where_the_shape_allows():
    plan = {"blocks.0.attn.wq.weight": "Q6_K", "blocks.0.odd.weight": "Q6_K", "first.weight": "Q6_K"}
    assert k2g.pick_qtype("blocks.0.attn.wq.weight", (512, 1024), "BF16", "q4_k_m", plan) == QT.Q6_K
    assert k2g.pick_qtype("blocks.0.odd.weight", (512, 800), "BF16", "q4_k_m", plan) == QT.Q5_0     # plan can't apply: fallback
    assert k2g.pick_qtype("first.weight", (512, 1024), "BF16", "q4_k_m", plan) == QT.F32           # precision rules win
    assert k2g.pick_qtype("blocks.1.attn.wq.weight", (512, 1024), "BF16", "q4_k_m", plan) == QT.Q4_K  # not in plan
