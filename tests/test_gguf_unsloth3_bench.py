"""Tests for the GGUF quant-setting sweep (quant_gui.gguf_bench).

Two layers:

* Pure unit tests (always run) — error math, sampling determinism, ranking,
  built on synthetic GGUF files written with the gguf package. No toolchain.
* One toolchain-gated benchmark (marked ``bench``) that runs a reduced
  sweep on a real small model and asserts the property the whole thing
  exists for: imatrix calibration improves reconstruction error, and the
  ranking picks winners. Skips cleanly when llama-quantize, the llama.cpp
  conversion venv, or a model dir are unavailable (or when
  GGUF_BENCH_MODEL_DIR isn't set and no smollm2 snapshot is cached).

Run the fast suite with:  pytest -m "not bench"
Run everything with:      pytest -m bench  (minutes, downloads nothing by itself)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_gui import gguf_bench as gb  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

REPO_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------- synthetic GGUFs


def _write_gguf(path: Path, tensors: dict[str, tuple[np.ndarray, str]]) -> Path:
    """tensors: name -> (array, GGMLQuantizationType name)."""
    import gguf

    writer = gguf.GGUFWriter(path=None, arch="llama")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    qtypes = gguf.GGMLQuantizationType
    for name, (arr, qtype) in tensors.items():
        packed = (
            gguf.quants.quantize(np.ascontiguousarray(arr), qtypes[qtype])
            if qtype != "F32"
            else np.ascontiguousarray(arr)
        )
        writer.add_tensor(name, packed, raw_dtype=qtypes[qtype])
    writer.write_header_to_file(path=str(path))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    return path


def _ref_quant_pair(tmp_path):
    rng = np.random.default_rng(7)
    ref = {}
    quant = {}
    for i, n in enumerate((4096, 2048, 512, 64)):
        arr = rng.standard_normal((n, 64), dtype=np.float32)
        ref[f"layer.{i}.weight"] = (arr, "F32")
        quant[f"layer.{i}.weight"] = (arr, "Q8_0")
    return (
        _write_gguf(tmp_path / "ref.gguf", ref),
        _write_gguf(tmp_path / "quant.gguf", quant),
    )


# ---------------------------------------------------------------- unit tests


def test_relative_error_zero_for_identical(tmp_path):
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((256, 64), dtype=np.float32)
    a = _write_gguf(tmp_path / "a.gguf", {"w": (arr, "F32")})
    b = _write_gguf(tmp_path / "b.gguf", {"w": (arr, "F32")})
    assert gb.relative_error(gb._dequantized_tensors(a), gb._dequantized_tensors(b)) == 0.0


def test_relative_error_positive_and_small_for_q8(tmp_path):
    ref_p, quant_p = _ref_quant_pair(tmp_path)
    err = gb.relative_error(
        gb._dequantized_tensors(ref_p), gb._dequantized_tensors(quant_p)
    )
    assert 0.0 < err < 0.05  # Q8_0 rounding of N(0,1) weights


def test_relative_error_no_shared_names_raises(tmp_path):
    rng = np.random.default_rng(2)
    arr = rng.standard_normal((256, 64), dtype=np.float32)
    a = _write_gguf(tmp_path / "a.gguf", {"x": (arr, "F32")})
    b = _write_gguf(tmp_path / "b.gguf", {"y": (arr, "F32")})
    with pytest.raises(ValueError):
        gb.relative_error(gb._dequantized_tensors(a), gb._dequantized_tensors(b))


def test_sampled_tensor_names_deterministic_and_within_budget(tmp_path):
    ref_p, quant_p = _ref_quant_pair(tmp_path)
    names1, total1 = gb.sampled_tensor_names(ref_p, quant_p, budget_params=3000)
    names2, total2 = gb.sampled_tensor_names(ref_p, quant_p, budget_params=3000)
    assert names1 == names2 and total1 == total2
    # Tensors are (n, 64); largest-first prefix: the 4096-row tensor alone
    # overshoots the 3000-param budget.
    assert names1 == {"layer.0.weight"}
    assert total1 == 4096 * 64


def test_sampled_tensor_names_respects_budget(tmp_path):
    ref_p, quant_p = _ref_quant_pair(tmp_path)
    names, total = gb.sampled_tensor_names(ref_p, quant_p, budget_params=300_000)
    # 4096x64=262144 fits, one more 2048x64 would overshoot - prefix stops.
    assert names == {"layer.0.weight", "layer.1.weight"}
    assert total == (4096 + 2048) * 64


def test_best_settings_picks_winners():
    rows = [
        gb.BenchRow("Q3_K_L", True, size_bytes=900, seconds=10, error=0.10),
        gb.BenchRow("Q3_K_S", True, size_bytes=700, seconds=8, error=0.14),
        gb.BenchRow("IQ3_XXS", True, size_bytes=650, seconds=9, error=0.18),
        gb.BenchRow("IQ3_XXS", False, failed="no imatrix support"),
    ]
    result = gb.SweepResult(rows=rows, n_params=1_700_000_000)
    picks = gb.best_settings(result)
    assert picks["best_quality"].label == "Q3_K_L + imatrix"
    assert picks["smallest"].label == "IQ3_XXS + imatrix"
    # value = error/byte: 0.10/900=1.11e-4 vs 0.14/700=2.0e-4 vs 0.18/650=2.77e-4
    assert picks["best_value"].label == "Q3_K_L + imatrix"
    assert len(result.ok_rows()) == 3  # failed rows excluded


def test_best_settings_all_failed_raises():
    with pytest.raises(ValueError):
        gb.best_settings(gb.SweepResult(rows=[gb.BenchRow("X", False, failed="boom")]))


def test_format_report_includes_picks_and_failures():
    rows = [
        gb.BenchRow("Q3_K_M", True, size_bytes=860, seconds=11, error=0.115,
                    bpw=4.0, measured_params=300_000_000),
        gb.BenchRow("IQ3_XXS", False, failed="llama-quantize exited 1"),
    ]
    report = gb.format_report(gb.SweepResult(rows=rows, n_params=1_700_000_000))
    assert "Q3_K_M + imatrix" in report
    assert "FAILED" in report
    assert "best_quality" in report
    assert "300M" in report


def test_bits_per_weight_f32_is_32(tmp_path):
    rng = np.random.default_rng(3)
    p = _write_gguf(tmp_path / "f32.gguf", {"w": (rng.standard_normal((256, 64), dtype=np.float32), "F32")})
    bpw, n = gb._bits_per_weight(p)
    assert bpw == pytest.approx(32.0)
    assert n == 256 * 64


# ------------------------------------------------------- per-family tuning


def test_model_info_reads_arch_and_largest_tensor(tmp_path):
    rng = np.random.default_rng(5)
    p = _write_gguf(
        tmp_path / "m.gguf",
        {
            "big.weight": (rng.standard_normal((1024, 256), dtype=np.float32), "F32"),
            "small.weight": (rng.standard_normal((64, 32), dtype=np.float32), "F32"),
        },
    )
    info = gb.model_info(p)
    assert info["n_params"] == 1024 * 256 + 64 * 32
    assert info["largest_tensor_params"] == 1024 * 256
    assert info["largest_tensor_name"] == "big.weight"
    assert info["embed_fraction"] == pytest.approx(1024 * 256 / info["n_params"])


def test_tune_sweep_scales_budget_past_embedding():
    # Gemma-4-like: a 700M-param embedding in a 3B model.
    tune = gb.tune_sweep(
        {"arch": "gemma4", "largest_tensor_params": 700_000_000, "embed_fraction": 0.23}
    )
    assert tune["error_budget"] == 1_400_000_000  # 2x the embedding
    assert tune["emb_q8_variants"] is True
    assert tune["family"] == "gemma4"


def test_tune_sweep_small_embedding_stays_default():
    tune = gb.tune_sweep(
        {"arch": "llama", "largest_tensor_params": 40_000_000, "embed_fraction": 0.02}
    )
    assert tune["error_budget"] == 300_000_000
    assert tune["emb_q8_variants"] is False


def test_tune_sweep_qwen35_arch_key():
    tune = gb.tune_sweep(
        {"arch": "qwen3_5", "largest_tensor_params": 380_000_000, "embed_fraction": 0.19}
    )
    assert tune["emb_q8_variants"] is True
    assert tune["error_budget"] == 760_000_000


# ---------------------------------------------------- target-size families


def test_family_candidates_all_supported_by_llama_quantize():
    from quant_gui import llamacpp_backend as lcpp

    for target, quants in gb.BPP_FAMILY_CANDIDATES.items():
        assert quants, f"{target} has no candidates"
        unsupported = [q for q in quants if q not in lcpp.QUANT_TYPE_CHOICES]
        assert not unsupported, f"{target}: not in llama-quantize list: {unsupported}"


def test_family_candidates_unknown_target_falls_back():
    assert gb.family_candidates("~99 bpw") == gb.DYNAMIC3_CANDIDATES
    assert gb.family_candidates("") == gb.DYNAMIC3_CANDIDATES
    assert gb.family_candidates(None) == gb.DYNAMIC3_CANDIDATES


def test_family_candidates_returns_copies():
    a = gb.family_candidates("~4 bpw")
    a.append("MUTATED")
    assert len(gb.family_candidates("~4 bpw")) == len(gb.BPP_FAMILY_CANDIDATES["~4 bpw"])


def test_family_for_bpw_maps_sweep_winner_to_size_class():
    assert gb.family_for_bpw(2.1) == "~2 bpw"
    assert gb.family_for_bpw(2.7) == "~3 bpw"
    assert gb.family_for_bpw(3.9) == "~4 bpw"  # top of the ~3 family lands in ~4's class
    assert gb.family_for_bpw(4.31) == "~4 bpw"  # UD-IQ4_XS-class file
    assert gb.family_for_bpw(4.7) == "~5 bpw"
    assert gb.family_for_bpw(6.5) == "~5 bpw"


# ------------------------------------------------- toolchain-gated benchmark


def _llamacpp_dir() -> Path:
    from quant_gui import llamacpp_backend as lcpp

    return lcpp.default_llamacpp_dir(REPO_ROOT)


def _find_model_dir() -> Path | None:
    env = (os.environ.get("GGUF_BENCH_MODEL_DIR") or "").strip()
    if env and Path(env).is_dir():
        return Path(env)
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    for repo in ("models--HuggingFaceTB--smollm2-1.7B", "models--HuggingFaceTB--smollm2-135M"):
        snaps = cache / repo / "snapshots"
        if snaps.is_dir():
            children = [d for d in snaps.iterdir() if d.is_dir()]
            if children:
                return children[0]
    return None


@pytest.mark.bench
def test_unsloth3_sweep_finds_best_settings(tmp_path_factory):
    """Reduced Dynamic-3.0 sweep on a real model: imatrix must help, and the
    ranking must produce winners. This is the 'find best settings' test."""
    from quant_gui import llamacpp_backend as lcpp

    llamacpp_dir = _llamacpp_dir()
    quantize_bin = lcpp._quantize_binary(llamacpp_dir)
    if quantize_bin is None:
        pytest.skip("llama-quantize not built/unpacked under llama.cpp/")
    if not lcpp.is_venv_ready(llamacpp_dir):
        pytest.skip("llama.cpp conversion venv not set up")
    model_dir = _find_model_dir()
    if model_dir is None:
        pytest.skip("no test model - set GGUF_BENCH_MODEL_DIR or cache smollm2")

    work = tmp_path_factory.mktemp("gguf_bench")
    ref_gguf = work / "ref-f16.gguf"
    imatrix = work / "ref.imatrix"
    if not ref_gguf.is_file():
        import subprocess

        r = subprocess.run(
            [
                str(lcpp._venv_python(llamacpp_dir)),
                str(llamacpp_dir / "convert_hf_to_gguf.py"),
                "--outfile", str(ref_gguf), "--outtype", "f16", str(model_dir),
            ],
            capture_output=True, text=True, errors="replace",
        )
        assert r.returncode == 0 and ref_gguf.is_file(), (
            f"convert_hf_to_gguf failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        )
    if not imatrix.is_file():
        import subprocess

        r = subprocess.run(
            [
                str(lcpp._imatrix_binary(llamacpp_dir)), "-m", str(ref_gguf),
                "-f", str(lcpp.DEFAULT_CALIBRATION_FILE), "-o", str(imatrix), "--chunks", "8",
            ],
            capture_output=True, text=True, errors="replace",
        )
        assert r.returncode == 0 and imatrix.is_file(), (
            f"llama-imatrix failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        )

    result = gb.run_sweep(
        ref_gguf=ref_gguf,
        imatrix_file=imatrix,
        out_dir=work / "out",
        quantize_bin=quantize_bin,
        quants=["Q3_K_M", "IQ3_M"],
        # Budget must comfortably exceed the largest tensor (token embeddings
        # ~100M params in small models and always kept F16) - otherwise the
        # sample degenerates to it and every variant measures identically.
        error_budget_params=300_000_000,
        log=lambda _msg: None,
    )
    ok = result.ok_rows()
    assert {r.quant for r in ok} >= {"Q3_K_M", "IQ3_M"}
    assert any(r.imatrix for r in ok) and any(not r.imatrix for r in ok)

    by_label = {r.label: r for r in ok}
    # The property the sweep exists for: imatrix improves reconstruction error.
    assert by_label["Q3_K_M + imatrix"].error < by_label["Q3_K_M"].error

    picks = gb.best_settings(result)
    assert set(picks) == {"best_quality", "smallest", "best_value"}
    for row in picks.values():
        assert row.failed is None and row.error is not None and row.size_bytes > 0
