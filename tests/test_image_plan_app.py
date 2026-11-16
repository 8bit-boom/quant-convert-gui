import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

pytest.importorskip("gradio", reason="app import needs gradio")
pytest.importorskip("gguf")

import app
from test_krea2_convert import _planning_model


def _ui_funcs():
    """The handlers are closures inside the Blocks; pull them from the registered event listeners."""
    fns = {}
    for fn in app.demo.fns.values():
        fns[getattr(fn.fn, "__name__", "")] = fn.fn
    return fns


def test_analyze_then_convert_through_the_image_tab_handlers(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path / "out")
    src = tmp_path / "k2.safetensors"
    _planning_model(src)
    fns = _ui_funcs()

    log = list(fns["run_img_analyze"](None, str(src), "q4_k_m", 0, 32))[-1]
    assert "mixed plan" in log and "plan written" in log
    plan = tmp_path / "out" / "k2.kplan.json"
    assert plan.is_file()
    assert fns["fill_img_plan"](None, str(src)) == str(plan)                 # the plan box gets filled in

    out = list(fns["run_img_convert"](None, str(src), "q4_k_m", "mixed.gguf", str(plan)))[-1]
    assert "plan:" in out and "entries match" in out and "done:" in out
    assert (tmp_path / "out" / "mixed.gguf").is_file()


def test_missing_files_are_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path / "out")
    fns = _ui_funcs()
    assert "not found" in list(fns["run_img_analyze"](None, str(tmp_path / "nope.safetensors"), "q4_k_m", 0, 32))[-1]
    src = tmp_path / "k2.safetensors"
    _planning_model(src)
    assert "Plan file not found" in list(fns["run_img_convert"](None, str(src), "q4_k_m", "o.gguf", "missing.json"))[-1]
