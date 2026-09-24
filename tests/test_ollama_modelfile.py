"""Tests for tools/ollama_modelfile.py - KV cache estimation and hints."""
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as QT
from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import ollama_modelfile as om  # noqa: E402


def _mk_llm_like(path: Path, with_kv_meta: bool = True) -> Path:
    writer = GGUFWriter(path=None, arch="testarch")
    writer.add_quantization_version(3)
    if with_kv_meta:
        writer.add_uint32("testarch.block_count", 32)
        writer.add_uint32("testarch.attention.head_count_kv", 8)
        writer.add_uint32("testarch.attention.key_length", 128)
    writer.add_tensor("blk.0.weight", np.zeros((64, 64), dtype=np.float32), raw_dtype=QT.F32)
    writer.write_header_to_file(path=str(path))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    return path


def test_kv_bytes_per_token_computed_from_metadata(tmp_path):
    p = _mk_llm_like(tmp_path / "m.gguf")
    # 32 layers x 8 kv heads x 128 head_k x 2 bytes x (K+V) = 131072
    assert om.kv_bytes_per_token(p) == 32 * 8 * 128 * 2 * 2


def test_kv_bytes_none_without_metadata(tmp_path):
    p = _mk_llm_like(tmp_path / "m.gguf", with_kv_meta=False)
    assert om.kv_bytes_per_token(p) is None


def test_modelfile_carries_kv_hint_at_num_ctx(tmp_path):
    p = _mk_llm_like(tmp_path / "m.gguf")
    text = om.build_modelfile(p, num_ctx=32768)
    per_tok = 32 * 8 * 128 * 2 * 2
    assert f"{per_tok / 1e6:.2f} MB/token" in text
    assert f"{per_tok * 32768 / 1e9:.2f} GB at num_ctx 32768" in text
    assert "OLLAMA_KV_CACHE_TYPE" in text


def test_modelfile_omits_kv_hint_when_unknown(tmp_path):
    p = _mk_llm_like(tmp_path / "m.gguf", with_kv_meta=False)
    assert "OLLAMA_KV_CACHE_TYPE" not in om.build_modelfile(p)
