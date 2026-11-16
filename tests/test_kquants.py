"""K-quant encoders (quant_gui/kquants.py).

The golden fixture was produced by the REAL ggml encoder (ggml_quantize_chunk without an imatrix, the path
llama-quantize takes) from llama.cpp commit ef6bc6f3 on 27 deliberately awkward blocks; these tests require
our blocks to be byte-identical. Set GGML_LIB to a built libggml-base.so to additionally compare on random data.
"""
import ctypes
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from quant_gui import kquants as kq

GOLDEN = np.load(Path(__file__).parent / "fixtures" / "kquants_golden.npz")
TYPES = list(kq.SUPPORTED)


@pytest.mark.parametrize("qtype", TYPES)
def test_matches_real_ggml_byte_for_byte(qtype):
    mine = kq.quantize_blocks(GOLDEN["x"], qtype)
    ref = GOLDEN[qtype]
    assert mine.shape == ref.shape == (len(GOLDEN["x"]), kq.BLOCK_BYTES[qtype])
    bad = np.where(~(mine == ref).all(axis=1))[0]
    assert not len(bad), f"{qtype}: blocks {bad.tolist()} differ from ggml"


@pytest.mark.skipif(not os.environ.get("GGML_LIB"), reason="set GGML_LIB=/path/to/libggml-base.so for the live comparison")
@pytest.mark.parametrize("qtype", TYPES)
def test_matches_live_ggml_on_random_data(qtype):
    lib = ctypes.CDLL(os.environ["GGML_LIB"])
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    ids = {"Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14}
    rng = np.random.default_rng(99)
    x = (rng.laplace(size=(256, 1024)) * 0.02).astype(np.float32)
    ref = np.zeros(x.size // 256 * kq.BLOCK_BYTES[qtype], dtype=np.uint8)
    lib.ggml_quantize_chunk(ids[qtype], x.ctypes.data, ref.ctypes.data, 0, 256, 1024, None)
    assert (kq.quantize_blocks(x, qtype) == ref.reshape(-1, kq.BLOCK_BYTES[qtype])).all()


@pytest.mark.parametrize("qtype", TYPES)
def test_shape_contract_and_thread_independence(qtype):
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((2, 6, 512)) * 0.05).astype(np.float32)         # >2D like a conv kernel
    out = kq.quantize(x, qtype)
    assert out.shape == (2, 6, 512 // 256 * kq.BLOCK_BYTES[qtype]) and out.dtype == np.uint8
    big = (rng.standard_normal((5000, 256)) * 0.05).astype(np.float32)        # > CHUNK_BLOCKS: several chunks
    assert (kq.quantize_blocks(big, qtype, threads=1) == kq.quantize_blocks(big, qtype, threads=3)).all()


@pytest.mark.parametrize("qtype", TYPES)
def test_rejects_rows_that_are_not_a_multiple_of_256(qtype):
    with pytest.raises(ValueError, match="256"):
        kq.quantize(np.zeros((4, 300), np.float32), qtype)
    with pytest.raises(ValueError):
        kq.quantize_blocks(np.zeros(100, np.float32), qtype)


def test_unknown_type_is_refused():
    with pytest.raises(ValueError, match="not a supported K-quant"):
        kq.quantize_blocks(np.zeros(256, np.float32), "Q4_0")


@pytest.mark.parametrize("qtype", TYPES)
def test_all_zero_input_decodes_to_zero(qtype):
    out = kq.quantize_blocks(np.zeros((3, 256), np.float32), qtype)
    assert not out.any()


def test_gguf_py_decodes_our_blocks_with_the_expected_error_ladder():
    """gguf-py's decoders are an independent implementation of the block formats: decoding our blocks with them
    proves the layout, and the error must fall as bits go up."""
    quants = pytest.importorskip("gguf").quants
    gguf = pytest.importorskip("gguf")
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((64, 1024)) * 0.02).astype(np.float32)
    err = {}
    for t in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"):
        packed = kq.quantize(x, t)
        deq = quants.dequantize(packed, gguf.GGMLQuantizationType[t])
        err[t] = float(np.sqrt(((deq - x) ** 2).mean()) / np.sqrt((x ** 2).mean()))
    assert err["Q2_K"] > err["Q3_K"] > err["Q4_K"] > err["Q5_K"] > err["Q6_K"]
    # ballpark for Gaussian weights (measured: 0.30 / 0.15 / 0.071 / 0.036 / 0.018)
    assert 0.2 < err["Q2_K"] < 0.4 and err["Q4_K"] < 0.1 and err["Q6_K"] < 0.03
