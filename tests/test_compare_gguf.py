"""End-to-end tests for tools/compare_gguf.py on synthetic GGUFs.

The synthetic weights use per-row marker values so that any row-orientation
bug in the sampler shows up as a massive error / collapsed cosine, and real
quantization shows up as the expected small-error ladder (Q8_0 < Q4_0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter, quants

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from compare_gguf import compare_file  # noqa: E402

CH, ROWS = 256, 48
NE = (CH, ROWS)  # ggml ne order (channels first)
N_EXPERTS = 4
M_ROWS = 32


def _weights(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((ROWS, CH)).astype(np.float32) * 0.05


def _moe_weights(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_EXPERTS, M_ROWS, CH)).astype(np.float32) * 0.05


def _write_ref(path: Path, w2d: np.ndarray, w3d: np.ndarray | None = None) -> None:
    writer = GGUFWriter(path, "test-arch")
    writer.add_tensor("test.weight", w2d.astype(np.float16))
    if w3d is not None:
        writer.add_tensor("test.exps.weight", w3d.astype(np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_quant(path: Path, ref_path: Path, quant_map: dict[str, np.ndarray]) -> None:
    """Clone ref metadata, replacing listed tensors with quantized payloads."""
    reader = GGUFReader(str(ref_path))
    writer = GGUFWriter(path, "test-arch")
    for t in reader.tensors:
        data = quant_map.get(t.name)
        if data is None:
            writer.add_tensor(t.name, np.asarray(t.data))
        else:
            packed = quants.quantize(data[0], data[1])
            writer.add_tensor(t.name, np.ascontiguousarray(packed, dtype=np.uint8),
                              raw_dtype=data[1])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture()
def ref_gguf(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    w2d = _weights()
    w3d = _moe_weights()
    p = tmp_path / "ref.gguf"
    _write_ref(p, w2d, w3d)
    return p, w2d, w3d


def _run(ref_path: Path, cand_path: Path, profiles=None):
    ref = GGUFReader(str(ref_path))
    cand = GGUFReader(str(cand_path))
    return compare_file(ref, cand, cand_path, profiles, max_rows=ROWS + 1)


def test_identical_float_candidate_has_zero_error(ref_gguf):
    ref_path, w2d, _ = ref_gguf
    cand = ref_path.parent / "same.gguf"
    _write_ref(cand, w2d)
    rep = _run(ref_path, cand)
    by = {t.name: t for t in rep.tensors}
    assert by["test.weight"].rel_l2 < 1e-3
    assert by["test.weight"].cos > 0.9999


def test_quant_ladder_q8_beats_q4(ref_gguf):
    ref_path, w2d, w3d = ref_gguf
    q8 = ref_path.parent / "q8.gguf"
    q4 = ref_path.parent / "q4.gguf"
    m = {"test.weight": (w2d, GGMLQuantizationType.Q8_0),
         "test.exps.weight": (w3d, GGMLQuantizationType.Q8_0)}
    _write_quant(q8, ref_path, m)
    m["test.weight"] = (w2d, GGMLQuantizationType.Q4_0)
    _write_quant(q4, ref_path, m)

    r8 = _run(ref_path, q8)
    r4 = _run(ref_path, q4)
    e8 = {t.name: t for t in r8.tensors}["test.weight"]
    e4 = {t.name: t for t in r4.tensors}["test.weight"]
    # real quantization of aligned data: small error, high cosine
    assert 0.001 < e8.rel_l2 < e4.rel_l2 < 0.20
    assert e8.cos > 0.999 and e4.cos > 0.99
    assert e8.snr_db > e4.snr_db
    # no tensor may fail to dequantize
    assert not any(k.startswith("dequant-fail") for k in r8.skipped)
    assert not any(k.startswith("dequant-fail") for k in r4.skipped)


def test_row_permutation_is_detected(ref_gguf):
    ref_path, w2d, _ = ref_gguf
    perm = ref_path.parent / "perm.gguf"
    w = w2d.copy()
    w[:] = w[::-1]  # reversed rows: same stats, wrong positions
    _write_quant(perm, ref_path, {"test.weight": (w, GGMLQuantizationType.Q8_0)})
    rep = _run(ref_path, perm)
    t = {x.name: x for x in rep.tensors}["test.weight"]
    assert t.cos < 0.9  # a sampler/orientation bug would land here too


def test_3d_expert_tensor_scores(ref_gguf):
    ref_path, _, w3d = ref_gguf
    cand = ref_path.parent / "moe.gguf"
    _write_quant(cand, ref_path,
                 {"test.exps.weight": (w3d, GGMLQuantizationType.Q8_0)})
    rep = _run(ref_path, cand)
    t = {x.name: x for x in rep.tensors}["test.exps.weight"]
    assert not any(k.startswith("dequant-fail") for k in rep.skipped)
    assert 0.001 < t.rel_l2 < 0.10
    assert t.cos > 0.999


def test_imatrix_weighting_runs_and_uniform_equals_plain(ref_gguf):
    ref_path, w2d, _ = ref_gguf
    cand = ref_path.parent / "q8.gguf"
    _write_quant(cand, ref_path,
                 {"test.weight": (w2d, GGMLQuantizationType.Q8_0)})
    ones = {"test.weight": np.ones(CH, dtype=np.float32)}
    rep = _run(ref_path, cand, profiles=ones)
    t = {x.name: x for x in rep.tensors}["test.weight"]
    assert t.w_rel_l2 == pytest.approx(t.rel_l2, rel=0.05)

    # heavy weight on the channel with the biggest quantization error must
    # raise the weighted error above the plain one
    reader = GGUFReader(str(cand))
    ct = [x for x in reader.tensors if x.name == "test.weight"][0]
    dq = quants.dequantize(np.asarray(ct.data), ct.tensor_type).astype(np.float32)
    err_per_ch = ((dq - w2d) ** 2).sum(axis=0)
    hot = np.ones(CH, dtype=np.float32)
    hot[int(np.argmax(err_per_ch))] = 1000.0
    rep2 = _run(ref_path, cand, profiles={"test.weight": hot})
    t2 = {x.name: x for x in rep2.tensors}["test.weight"]
    assert t2.w_rel_l2 > t.rel_l2
