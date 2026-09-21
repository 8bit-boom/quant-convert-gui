"""Persistent run history for loop-phase (and future per-run) stats.

A single JSON file (default ``<app_dir>/run_history.json``) holds a flat,
newest-last list of finished conversion runs. Each entry carries a ``key``
(the resolved output path, falling back to the input path) so consecutive
runs of the same conversion can be compared — this is what makes the GPU
speed flags' effect trackable across conversions rather than visible only
inside one log.

The file is written atomically and trimmed to ``keep`` entries; a corrupt
file is treated as empty (history is nice-to-have, never load-bearing).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

HISTORY_NAME = "run_history.json"
DEFAULT_KEEP = 200


def load(path: Path | str) -> list[dict]:
    """All recorded runs, oldest first. Missing/corrupt file → []."""
    p = Path(path)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def record(path: Path | str, entry: dict, keep: int = DEFAULT_KEEP) -> list[dict]:
    """Append ``entry`` (adding a ``timestamp`` if absent) and trim to ``keep``.

    Returns the stored list.
    """
    entries = load(path)
    entry.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    entries.append(entry)
    entries = entries[-keep:]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    tmp.replace(p)  # atomic on POSIX and Windows
    return entries


def previous(entries: list[dict], key: str, before: str | None = None) -> dict | None:
    """The most recent recorded run for ``key`` (excluding ``before``'s stamp).

    ``before`` should be the current run's timestamp so a re-recorded run
    doesn't compare against itself.
    """
    for e in reversed(entries):
        if e.get("key") != key:
            continue
        if before is not None and e.get("timestamp") == before:
            continue
        return e
    return None


def comparison_line(prev: dict, current_loop: dict) -> str | None:
    """Human '[loop]' line comparing this run's optimizer phase to ``prev``'s."""
    prev_loop = prev.get("loop") or {}
    prev_s = prev_loop.get("total_seconds")
    if prev_s is None or not current_loop or not current_loop.get("tensor_count"):
        return None
    cur_s = current_loop["total_seconds"]
    stamp = prev.get("timestamp", "unknown time")
    try:
        ratio = float(prev_s) / max(cur_s, 1e-9)
    except (TypeError, ValueError):
        return None
    if ratio >= 1.05:
        verdict = f"~{ratio:.2f}× faster"
    elif ratio <= 0.95:
        verdict = f"~{1.0 / max(ratio, 1e-9):.2f}× slower"
    else:
        verdict = "about the same"
    return (
        f"[loop] vs run {stamp}: optimizer phase {cur_s:.1f}s vs "
        f"{float(prev_s):.1f}s — {verdict}"
    )
