"""Tests for quant_gui/gpu_quant.py - opt-in GPU quant with numpy fallback."""
import os

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from quant_gui import gpu_quant  # noqa: E402


def test_numpy_path_bit_exact_vs_gguf():
    rng = np.random.default_rng(7)
    for shape in [(32,), (64,), (4, 96), (2, 3, 64)]:
        data = rng.standard_normal(shape).astype(np.float32) * 3
        expected = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
        got = gpu_quant.quantize_q8_0(data)
        assert got.dtype == np.uint8
        assert got.shape == expected.shape
        np.testing.assert_array_equal(got, expected)


def test_numpy_path_zero_blocks():
    data = np.zeros((2, 64), dtype=np.float32)
    expected = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
    np.testing.assert_array_equal(gpu_quant.quantize_q8_0(data), expected)


def test_numpy_path_outliers_and_ties():
    # round-half-away-from-zero ties + a huge outlier dominating the scale
    data = np.array([[0.5, -0.5, 1.5, -1.5] + [0.0] * 28] * 2, dtype=np.float32)
    data[1, 0] = 1000.0
    expected = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
    np.testing.assert_array_equal(gpu_quant.quantize_q8_0(data), expected)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        gpu_quant.quantize_q8_0_numpy(np.zeros((33,), dtype=np.float32))


def test_gate_requires_env(monkeypatch):
    monkeypatch.delenv(gpu_quant.GPU_QUANT_ENV, raising=False)
    assert not gpu_quant.gpu_quant_enabled()
    monkeypatch.setenv(gpu_quant.GPU_QUANT_ENV, "1")
    assert gpu_quant.gpu_quant_enabled()
    # without triton installed the full gate stays closed, so dispatch
    # falls back to numpy and stays bit-exact
    if not gpu_quant.triton_available():
        assert not gpu_quant.gpu_quant_ready()
        data = np.random.default_rng(1).standard_normal((64,)).astype(np.float32)
        expected = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
        np.testing.assert_array_equal(gpu_quant.quantize_q8_0(data), expected)


@pytest.mark.skipif(not gpu_quant.gpu_quant_ready(),
                    reason="needs QUANT_GUI_GPU_QUANT=1 + triton + CUDA")
def test_triton_path_bit_exact_when_available():
    """Hardware check: the Triton kernel must match gguf-py byte for byte.

    Verified on RTX 5090; CI (no GPU) skips this and covers the fallback."""
    rng = np.random.default_rng(11)
    for shape in [(32,), (64,), (4, 96), (2, 3, 64), (2048, 4096)]:
        data = rng.standard_normal(shape).astype(np.float32) * 3
        expected = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
        got = gpu_quant.quantize_q8_0(data)
        assert got.shape == expected.shape
        np.testing.assert_array_equal(got, expected)
    # outliers + exact ties (the case that caught non-IEEE division)
    data = np.array([[0.5, -0.5, 1.5, -1.5] + [0.0] * 28] * 64, dtype=np.float32)
    data[1, 0] = 1000.0
    expected = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
    np.testing.assert_array_equal(gpu_quant.quantize_q8_0(data), expected)
