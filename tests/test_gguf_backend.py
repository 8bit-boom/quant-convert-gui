import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

gguf = pytest.importorskip("gguf", reason="gguf not installed - GGUF export path is optional")

import torch
from safetensors.torch import save_file

from quant_gui.gguf_backend import (
    GGUFBackendError,
    GGUFConvertStats,
    SUPPORTED_ARCH_NAMES,
    convert_to_gguf,
    detect_arch,
    is_available,
    stream_gguf_conversion,
)


def _write_lumina2_model(path: Path, oddshape=(64, 50)) -> None:
    sd = {
        "cap_embedder.1.weight": torch.randn(1024, 2048, dtype=torch.bfloat16),
        "context_refiner.0.attention.qkv.weight": torch.randn(3072, 1024, dtype=torch.bfloat16),
        "blocks.0.attn.wq.weight": torch.randn(1024, 1024, dtype=torch.bfloat16),
        "blocks.0.norm.weight": torch.randn(1024, dtype=torch.bfloat16),
        "blocks.0.tiny.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "blocks.0.oddshape.weight": torch.randn(*oddshape, dtype=torch.bfloat16),
    }
    save_file(sd, str(path))


def test_is_available():
    assert is_available() is True


def test_detect_arch_recognizes_lumina2():
    keys = {"cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight", "other.key"}
    arch = detect_arch(keys)
    assert arch is not None
    assert arch.arch == "lumina2"


def test_detect_arch_returns_none_for_unknown_model():
    assert detect_arch({"blocks.0.attn.wq.weight", "blocks.0.mlp.gate.weight"}) is None


def test_convert_to_gguf_splits_tensors_correctly(tmp_path):
    src = tmp_path / "model.safetensors"
    out = tmp_path / "model-Q8_0.gguf"
    _write_lumina2_model(src)

    stats = convert_to_gguf(str(src), str(out), "Q8_0")

    assert isinstance(stats, GGUFConvertStats)
    assert stats.arch == "lumina2"
    assert stats.total == 6
    # cap_embedder, context_refiner, attn.wq -> quantized; norm (1D) + tiny (small) -> F32;
    # oddshape (last dim 50, not divisible by 32) -> QuantError fallback to F16.
    assert stats.quantized_count == 3
    assert stats.f32_kept_count == 2
    assert stats.fallback_f16_count == 1
    assert stats.skipped_high_dim_count == 0
    assert out.is_file()


def test_output_file_has_correct_arch_and_tensor_types(tmp_path):
    src = tmp_path / "model.safetensors"
    out = tmp_path / "model-Q4_0.gguf"
    _write_lumina2_model(src)

    convert_to_gguf(str(src), str(out), "Q4_0")

    reader = gguf.GGUFReader(str(out))
    field = reader.get_field("general.architecture")
    arch_str = str(field.parts[field.data[-1]], encoding="utf-8")
    assert arch_str == "lumina2"
    assert arch_str in SUPPORTED_ARCH_NAMES

    by_name = {t.name: t for t in reader.tensors}
    assert by_name["cap_embedder.1.weight"].tensor_type == gguf.GGMLQuantizationType.Q4_0
    assert by_name["blocks.0.norm.weight"].tensor_type == gguf.GGMLQuantizationType.F32
    assert by_name["blocks.0.tiny.weight"].tensor_type == gguf.GGMLQuantizationType.F32
    assert by_name["blocks.0.oddshape.weight"].tensor_type == gguf.GGMLQuantizationType.F16


def test_unknown_architecture_raises(tmp_path):
    src = tmp_path / "model.safetensors"
    save_file({"blocks.0.attn.wq.weight": torch.randn(64, 64, dtype=torch.bfloat16)}, str(src))

    with pytest.raises(GGUFBackendError, match="Unknown model architecture"):
        convert_to_gguf(str(src), str(tmp_path / "out.gguf"), "Q8_0")


def test_kquant_type_is_rejected(tmp_path):
    src = tmp_path / "model.safetensors"
    _write_lumina2_model(src)

    with pytest.raises(GGUFBackendError, match="llama-quantize"):
        convert_to_gguf(str(src), str(tmp_path / "out.gguf"), "Q4_K_M")


def test_stream_gguf_conversion_yields_progress_then_ok(tmp_path):
    src = tmp_path / "model.safetensors"
    out = tmp_path / "model-Q5_1.gguf"
    _write_lumina2_model(src)

    events = list(stream_gguf_conversion(str(src), str(out), "Q5_1"))

    kinds = [e[0] for e in events]
    assert kinds[-1] == "ok"
    assert kinds.count("progress") == 6
    assert events[-1][1].arch == "lumina2"


def test_stream_gguf_conversion_reports_failure_for_bad_input():
    events = list(stream_gguf_conversion("/no/such/file.safetensors", "/tmp/out.gguf", "Q8_0"))
    assert events[-1][0] == "fail"
