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
