"""Tests for quant_gui/ui_settings.py - persistent UI settings store."""
import json

from quant_gui import ui_settings


def test_missing_file_returns_empty(tmp_path):
    assert ui_settings.load_settings(tmp_path) == {}


def test_set_and_reload(tmp_path):
    ui_settings.set_setting(tmp_path, "gguf_gpu_quant", True)
    ui_settings.set_setting(tmp_path, "other", "x")
    loaded = ui_settings.load_settings(tmp_path)
    assert loaded == {"gguf_gpu_quant": True, "other": "x"}
    # update one key, other survives
    ui_settings.set_setting(tmp_path, "gguf_gpu_quant", False)
    loaded = ui_settings.load_settings(tmp_path)
    assert loaded == {"gguf_gpu_quant": False, "other": "x"}


def test_corrupt_file_returns_empty(tmp_path):
    p = ui_settings.settings_path(tmp_path)
    p.write_text("{not json", encoding="utf-8")
    assert ui_settings.load_settings(tmp_path) == {}
    # non-dict JSON also tolerated
    p.write_text("[1, 2]", encoding="utf-8")
    assert ui_settings.load_settings(tmp_path) == {}


def test_write_is_atomic_json(tmp_path):
    ui_settings.set_setting(tmp_path, "k", 1)
    p = ui_settings.settings_path(tmp_path)
    assert json.loads(p.read_text(encoding="utf-8")) == {"k": 1}
    assert not list(tmp_path.glob("*.tmp")), "no tmp files left behind"
