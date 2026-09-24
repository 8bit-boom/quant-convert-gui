"""Tests for quant_gui.gguf_inspect."""
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as QT
from gguf import GGUFWriter

from quant_gui.gguf_inspect import format_inspection, inspect_gguf


def _make_gguf(path: Path) -> None:
    writer = GGUFWriter(path=None, arch="testarch")
    writer.add_quantization_version(3)
    writer.add_file_type(15)
    writer.add_string("custom.meta", "hello")
    big = np.random.default_rng(1).standard_normal((256, 256)).astype(np.float32)
    writer.add_tensor("blk.0.weight", __import__("gguf").quants.quantize(big, QT.Q8_0),
                      raw_dtype=QT.Q8_0)
    writer.add_tensor("blk.0.norm", np.ones(64, dtype=np.float32), raw_dtype=QT.F32)
    writer.add_tensor("blk.1.weight", np.random.default_rng(2).standard_normal(
        (64, 64)).astype(np.float16), raw_dtype=QT.F16)
    writer.write_header_to_file(path=str(path))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


def test_inspect_structure(tmp_path):
    p = tmp_path / "m.gguf"
    _make_gguf(p)
    info = inspect_gguf(p)
    assert info["arch"] == "testarch"
    assert info["file_type"] == "MOSTLY_Q4_K_M"  # writer.add_file_type(15)
    assert info["tensor_count"] == 3
    assert info["per_type"]["Q8_0"]["count"] == 1
    assert info["per_type"]["F32"]["count"] == 1
    assert info["per_type"]["F16"]["count"] == 1
    assert info["total_params"] == 256 * 256 + 64 + 64 * 64
    # Q8_0: 34 bytes per 32 weights -> 8.5 bits/weight; F32 norm pulls the
    # overall number up slightly
    assert 8.4 < info["bits_per_weight"] < 9.0
    assert info["biggest_tensors"][0]["name"] == "blk.0.weight"
    assert info["metadata"]["custom.meta"] == "hello"


def test_format_inspection_renders(tmp_path):
    p = tmp_path / "m.gguf"
    _make_gguf(p)
    md = format_inspection(inspect_gguf(p))
    assert "## m.gguf" in md
    assert "testarch" in md
    assert "| Q8_0 |" in md
    assert "blk.0.weight" in md
    assert "custom.meta" in md
