"""Tests for downloaded-model listing + auto-detect of newest GGUF/imatrix."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app


def _touch(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# llm_downloaded_models
# ---------------------------------------------------------------------------

def test_downloaded_models_scans_models_dir_with_config(tmp_path):
    models = tmp_path / "models"
    _touch(models / "smollm2" / "config.json")
    _touch(models / "no-config-dir" / "weights.bin")  # no config.json -> excluded
    found = app.llm_downloaded_models(models_dir=models, hf_cache=tmp_path / "no-cache")
    assert str(models / "smollm2") in found
    assert str(models / "no-config-dir") not in found


def test_downloaded_models_scans_hf_cache(tmp_path):
    cache = tmp_path / "hub"
    _touch(cache / "models--org--Repo" / "snapshots" / "abc123" / "config.json")
    _touch(cache / "datasets--org--Data" / "snapshots" / "abc" / "config.json")  # not a model
    found = app.llm_downloaded_models(models_dir=tmp_path / "no-models", hf_cache=cache)
    assert str(cache / "models--org--Repo" / "snapshots" / "abc123") in found
    assert not any("datasets--" in p for p in found)


def test_downloaded_models_missing_dirs_ok(tmp_path):
    assert app.llm_downloaded_models(
        models_dir=tmp_path / "missing", hf_cache=tmp_path / "missing-too"
    ) == []


def test_downloaded_models_dedupes(tmp_path):
    models = tmp_path / "models"
    cache = tmp_path / "hub"
    target = models / "dup"
    _touch(target / "config.json")
    (cache / "models--x--dup" / "snapshots" / "h").mkdir(parents=True)
    try:
        os.symlink(target, cache / "models--x--dup" / "snapshots" / "h" / "link_target",
                   target_is_directory=True)
    except OSError:
        _touch(cache / "models--x--dup" / "snapshots" / "h" / "config.json")
    found = app.llm_downloaded_models(models_dir=models, hf_cache=cache)
    assert len(found) == len(set(found))


# ---------------------------------------------------------------------------
# newest_model_gguf / newest_imatrix
# ---------------------------------------------------------------------------

def test_newest_model_gguf_ignores_imatrix_names(tmp_path):
    out = tmp_path / "out"
    _touch(out / "a.imatrix.gguf")  # imatrix in name -> never picked as model
    _touch(out / "model-f16.gguf")
    os.utime(out / "model-f16.gguf", (100, 100))  # older, but only real model
    newest = app.newest_model_gguf(directory=out)
    assert newest == str(out / "model-f16.gguf")


def test_newest_model_gguf_empty_and_zero_size(tmp_path):
    assert app.newest_model_gguf(directory=tmp_path / "empty") is None
    out = tmp_path / "out"
    _touch(out / "empty.gguf", content="")  # 0 bytes -> ignored
    assert app.newest_model_gguf(directory=out) is None


def test_newest_imatrix(tmp_path):
    out = tmp_path / "out"
    _touch(out / "old.imatrix")
    _touch(out / "new.imatrix")
    os.utime(out / "old.imatrix", (100, 100))
    assert app.newest_imatrix(directory=out) == str(out / "new.imatrix")
    assert app.newest_imatrix(directory=tmp_path / "empty") is None


# ---------------------------------------------------------------------------
# _resolve_calibration
# ---------------------------------------------------------------------------

def test_resolve_calibration_auto_variants():
    assert app._resolve_calibration("Auto (bundled generic calibration)", "ignored.txt") == ""
    assert app._resolve_calibration("Auto", "ignored.txt") == ""


def test_resolve_calibration_empty_mode_passthrough():
    # No mode (legacy callers) keeps passing the file through.
    assert app._resolve_calibration("", "ignored.txt") == "ignored.txt"


def test_resolve_calibration_custom_passthrough():
    assert app._resolve_calibration("Custom calibration file", "/data/calib.txt") == "/data/calib.txt"
    assert app._resolve_calibration("Custom calibration file", "  /data/calib.txt  ") == "/data/calib.txt"


# ---------------------------------------------------------------------------
# Guard paths: empty inputs must guide the user, not crash.
# run_llm_* helpers read the OUTPUT_DIR global, so monkeypatch it.
# ---------------------------------------------------------------------------

def test_generate_imatrix_no_model_gguf_gives_guidance(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)  # empty: nothing detectable
    result = list(app.run_llm_generate_imatrix("", "Auto", "", ""))
    assert result and "gguf" in result[0][0].lower()


def test_generate_imatrix_custom_without_file(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    model = _touch(tmp_path / "m.gguf")
    result = list(app.run_llm_generate_imatrix(str(model), "Custom calibration file", "", ""))
    assert result and "calibration" in result[0][0].lower()


def test_generate_imatrix_autodetects_model(monkeypatch, tmp_path):
    out = tmp_path / "out"
    _touch(out / "latest.gguf")
    monkeypatch.setattr(app, "OUTPUT_DIR", out)
    gen = app.run_llm_generate_imatrix("", "Auto", "", "")
    first = next(gen)[0]  # intro log announces the auto-detected path
    gen.close()
    assert "latest.gguf" in first


def test_quantize_and_find_best_guidance_when_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)  # empty: nothing detectable
    q = next(app.run_llm_quantize("", "Q4_K_M", "32", "", ""))
    assert "gguf" in q[0].lower()
    f = next(app.run_llm_find_best("", "", "4.0", quants=[]))
    assert "gguf" in f[0].lower()


def test_quantize_autodetect_note(monkeypatch, tmp_path):
    out = tmp_path / "out"
    _touch(out / "latest.gguf")
    _touch(out / "latest.imatrix")
    monkeypatch.setattr(app, "OUTPUT_DIR", out)
    gen = app.run_llm_quantize("", "q4.gguf", "Q4_K_M", "", "")
    first = next(gen)[0]
    gen.close()
    assert "latest.gguf" in first


def test_find_best_autodetect_note(monkeypatch, tmp_path):
    out = tmp_path / "out"
    _touch(out / "latest.gguf")
    monkeypatch.setattr(app, "OUTPUT_DIR", out)
    monkeypatch.setattr(app.lcpp, "is_imatrix_built", lambda _d: False)  # keep worker spawn-free
    gen = app.run_llm_find_best("", "", "4.0", quants=[])
    first = next(gen)[0]
    gen.close()
    assert "latest.gguf" in first


def test_validate_guards(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)  # empty: nothing to auto-detect
    log, res = list(app.run_llm_validate("", "", "Q4_K_M", "", "99"))[0]
    assert "step 6" in log
    assert res is None


def test_validate_requires_text_file(monkeypatch, tmp_path):
    out = tmp_path / "out"
    _touch(out / "model-smart.gguf")
    monkeypatch.setattr(app, "OUTPUT_DIR", out)
    log, res = list(app.run_llm_validate("", "", "Q4_K_M", "no/such.txt", "99"))[0]
    assert "text" in log.lower()
    assert res is None


def test_validate_requires_perplexity_binary(monkeypatch, tmp_path):
    out = tmp_path / "out"
    _touch(out / "model-smart.gguf")
    _touch(tmp_path / "held.txt")
    monkeypatch.setattr(app, "OUTPUT_DIR", out)
    monkeypatch.setattr(app.lcpp, "is_perplexity_built", lambda _d: False)
    log, res = list(app.run_llm_validate("", "", "Q4_K_M", str(tmp_path / "held.txt"), "99"))[0]
    assert "llama-perplexity" in log
    assert res is None


def test_newest_smart_gguf_prefers_smart(monkeypatch, tmp_path):
    out = tmp_path / "out"
    _touch(out / "plain.gguf")
    smart = _touch(out / "model-smart.gguf")
    monkeypatch.setattr(app, "OUTPUT_DIR", out)
    assert app.newest_smart_gguf() == str(smart)
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path / "empty")
    assert app.newest_smart_gguf() is None
