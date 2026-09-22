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
