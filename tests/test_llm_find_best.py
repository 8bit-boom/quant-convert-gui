"""Tests for the LLM tab's "Find best quant" sweep button (run_llm_find_best)."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402


def _last(gen):
    item = None
    for item in gen:
        pass
    return item


def test_find_best_requires_input_file():
    yields = list(app.run_llm_find_best("", ""))
    assert len(yields) == 1
    log, file, quant_update, imatrix_update = yields[0]
    assert "Pick an input GGUF" in log
    assert file is None


def test_find_best_rejects_missing_file():
    log, file, _, _ = list(app.run_llm_find_best("no/such/model.gguf", ""))[0]
    assert "Pick an input GGUF" in log
    assert file is None


@pytest.mark.bench
def test_find_best_end_to_end_selects_winner():
    """Real sweep through the generator: the final yield must name a valid
    quant type for the dropdown. Skips without a reference GGUF - set
    GGUF_BENCH_REF_GGUF (e.g. the F16 output of an earlier convert)."""
    from quant_gui import llamacpp_backend as lcpp

    if not lcpp.is_quantize_built(app.LLAMACPP_DIR):
        pytest.skip("llama-quantize not built/unpacked")
    ref = (os.environ.get("GGUF_BENCH_REF_GGUF") or "").strip()
    if not ref or not Path(ref).is_file():
        pytest.skip("set GGUF_BENCH_REF_GGUF to an existing F16/BF16 GGUF")

    last = _last(app.run_llm_find_best(ref, "", target_bpw="~4 bpw", quants=["Q3_K_M"]))
    log, file, quant_value, imatrix_update = last
    assert "Winner" in log and "Q3_K_M" in log
    assert "~4 bpw" in log  # the requested target is reported
    assert file is None  # sweep discards intermediates
    assert quant_value == "Q3_K_M"  # pre-selects the dropdown
    assert quant_value in lcpp.QUANT_TYPE_CHOICES
