"""Persistent UI settings (simple JSON store in the app directory).

Gradio widget state doesn't survive restarts; a few options (like opting
into GPU quantization) are annoying to re-set every launch, so they're
saved here as a flat key/value dict:

    <app_dir>/ui_settings.json

Writes are crash-atomic (tmp file + os.replace) and cross-process safe
enough for a single-user desktop app: last writer wins, a torn write can
never leave a half-parsed file behind.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

SETTINGS_NAME = "ui_settings.json"


def settings_path(app_dir: str | Path) -> Path:
    return Path(app_dir) / SETTINGS_NAME


def load_settings(app_dir: str | Path) -> dict:
    """All saved settings; {} on missing/corrupt file (never raises)."""
    try:
        raw = settings_path(app_dir).read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def set_setting(app_dir: str | Path, key: str, value) -> dict:
    """Set one key, persist, and return the updated settings dict."""
    settings = load_settings(app_dir)
    settings[key] = value
    path = settings_path(app_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".ui_settings-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return settings
