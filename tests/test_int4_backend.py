import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

comfy_kitchen = pytest.importorskip(
    "comfy_kitchen", reason="comfy-kitchen not installed - INT4 ConvRot path is optional"
)

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from quant_gui.int4_backend import Int4ConvertStats, convert_int4_mixed, is_available, stream_int4_conversion


def _write_test_model(path: Path) -> None:
    sd = {}
    for i in range(2):
        p = f"blocks.{i}."
        sd[p + "attn.wq.weight"] = torch.randn(512, 512, dtype=torch.bfloat16)
        sd[p + "attn.wk.weight"] = torch.randn(512, 512, dtype=torch.bfloat16)
        sd[p + "attn.norm.weight"] = torch.randn(512, dtype=torch.bfloat16)
        sd[p + "mlp.gate.weight"] = torch.randn(512, 512, dtype=torch.bfloat16)
    sd["txtfusion.projector.weight"] = torch.randn(512, 512, dtype=torch.bfloat16)
    save_file(sd, str(path))


def test_is_available():
    assert is_available() is True


def test_convert_int4_mixed_splits_layers_correctly(tmp_path):
    src = tmp_path / "model.safetensors"
    out = tmp_path / "model-int4.safetensors"
    _write_test_model(src)

    stats = convert_int4_mixed(
        str(src), str(out),
        int4_layers_regex=r"attn\.wq|mlp\.gate",
        preset="krea2",
    )

    assert isinstance(stats, Int4ConvertStats)
    # attn.wq + mlp.gate for 2 blocks = 4 INT4 layers.
    assert stats.int4_count == 4
    assert sorted(stats.int4_layer_names) == [
        "blocks.0.attn.wq", "blocks.0.mlp.gate", "blocks.1.attn.wq", "blocks.1.mlp.gate",
    ]
    # attn.wk for 2 blocks -> INT8 tensorwise fallback.
    assert stats.int8_count == 2
    # attn.norm (1D) x2 + txtfusion.projector (krea2-excluded) = 3 kept.
    assert stats.kept_count == 3
    assert out.is_file()


def test_output_file_has_correct_shapes_and_metadata(tmp_path):
    import json

    src = tmp_path / "model.safetensors"
    out = tmp_path / "model-int4.safetensors"
    _write_test_model(src)

    convert_int4_mixed(str(src), str(out), int4_layers_regex=r"attn\.wq", preset="none")

    with safe_open(str(out), framework="pt") as f:
        meta = f.metadata()
        quant_map = json.loads(meta["_quantization_metadata"])

        assert quant_map["layers"]["blocks.0.attn.wq"] == {
            "format": "convrot_w4a4", "convrot_groupsize": 256, "quant_group_size": 64,
        }
        assert quant_map["layers"]["blocks.0.attn.wk"] == {"format": "int8_tensorwise"}

        # Packed INT4: same rows, half the columns (2 values per byte).
        wq = f.get_tensor("blocks.0.attn.wq.weight")
        assert wq.dtype == torch.int8
        assert tuple(wq.shape) == (512, 256)
        wq_scale = f.get_tensor("blocks.0.attn.wq.weight_scale")
        assert tuple(wq_scale.shape) == (512,)

        # Plain INT8 tensorwise: full shape, per-row scale.
        wk = f.get_tensor("blocks.0.attn.wk.weight")
        assert wk.dtype == torch.int8
        assert tuple(wk.shape) == (512, 512)


def test_no_int4_regex_means_no_int4_layers(tmp_path):
    src = tmp_path / "model.safetensors"
    out = tmp_path / "model-int4.safetensors"
    _write_test_model(src)

    stats = convert_int4_mixed(str(src), str(out), int4_layers_regex=None, preset="none")

    assert stats.int4_count == 0
    # wq, wk, mlp.gate (x2 blocks) + txtfusion.projector, all fall back to int8.
    assert stats.int8_count == 7
    assert stats.kept_count == 2  # just the two 1D attn.norm tensors


def test_fallback_int8_false_keeps_non_int4_layers_at_original_precision(tmp_path):
    src = tmp_path / "model.safetensors"
    out = tmp_path / "model-int4.safetensors"
    _write_test_model(src)

    stats = convert_int4_mixed(
        str(src), str(out), int4_layers_regex=r"attn\.wq", preset="none", fallback_int8=False,
    )

    assert stats.int4_count == 2
    assert stats.int8_count == 0
    assert stats.kept_count == 7  # everything else, including attn.wk, stays put


def test_stream_int4_conversion_yields_progress_then_ok(tmp_path):
    src = tmp_path / "model.safetensors"
    out = tmp_path / "model-int4.safetensors"
    _write_test_model(src)

    events = list(stream_int4_conversion(str(src), str(out), r"attn\.wq|mlp\.gate", preset="krea2"))

    kinds = [e[0] for e in events]
    assert kinds[-1] == "ok"
    assert kinds.count("progress") == 9  # 9 tensors total
    final_stats = events[-1][1]
    assert final_stats.int4_count == 4


def test_stream_int4_conversion_reports_failure_for_bad_input():
    events = list(stream_int4_conversion("/no/such/file.safetensors", "/tmp/out.safetensors", None))
    assert events[-1][0] == "fail"


def test_sharded_checkpoint_converts_identically_to_the_single_file(tmp_path):
    import json

    src = tmp_path / "model.safetensors"
    _write_test_model(src)
    shards = tmp_path / "shards"
    shards.mkdir()
    with safe_open(str(src), framework="pt") as f:
        keys = sorted(f.keys())
        weight_map = {}
        for i in range(3):
            part = {k: f.get_tensor(k).clone() for k in keys[i::3]}
            name = f"m-{i + 1:05d}-of-00003.safetensors"
            save_file(part, str(shards / name))
            weight_map.update({k: name for k in part})
    index = shards / "m.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}))

    kwargs = dict(int4_layers_regex=r"attn\.wq|mlp\.gate", preset="krea2")
    a = convert_int4_mixed(str(src), str(tmp_path / "a.safetensors"), **kwargs)
    b = convert_int4_mixed(str(index), str(tmp_path / "b.safetensors"), **kwargs)

    assert (a.int4_count, a.int8_count, a.kept_count) == (b.int4_count, b.int8_count, b.kept_count)
    with safe_open(str(tmp_path / "a.safetensors"), framework="pt") as fa, \
         safe_open(str(tmp_path / "b.safetensors"), framework="pt") as fb:
        assert sorted(fa.keys()) == sorted(fb.keys())
        for k in fa.keys():
            assert torch.equal(fa.get_tensor(k), fb.get_tensor(k)), k
        # key order inside the JSON follows shard enumeration order; the content must be identical
        assert json.loads(fa.metadata()["_quantization_metadata"]) == json.loads(fb.metadata()["_quantization_metadata"])
