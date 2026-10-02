"""On-disk checkpoints for pause/stop-&-save/resume of conversions.

A checkpoint is a directory:

    <app_dir>/checkpoints/<id>/
        checkpoint.json     manifest: params, output path, per-tensor records
        shards/000123.saf   per-tensor payload written by the backend
                            (safetensors for INT4, .npy for GGUF)

The backends call `record_tensor()` after each fully-processed tensor, so a
checkpoint always describes a consistent prefix of the model: resuming means
replaying the recorded shards for indices < completed count, then continuing
the tensor loop from there.

`ctq` conversions can't be checkpointed at tensor level (the engine is an
opaque subprocess that writes its output in one pass at the end), so for them
a "checkpoint" is a session snapshot: the exact argument list plus every GUI
setting, enough to relaunch the identical conversion with one click.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_NAME = "checkpoint.json"
SHARD_DIR_NAME = "shards"
MANIFEST_VERSION = 1


@dataclass
class Checkpoint:
    path: Path
    backend: str
    params: dict
    output_path: str
    total: int
    created_at: str
    tensors: dict[str, dict] = field(default_factory=dict)
    finished: bool = False

    @property
    def completed_count(self) -> int:
        return len(self.tensors)

    @property
    def next_index(self) -> int:
        return self.completed_count

    @property
    def id(self) -> str:
        return self.path.name

    def shard_path(self, index: int, suffix: str = "") -> Path:
        return self.path / SHARD_DIR_NAME / f"{index:06d}{suffix}"

    def tensor_meta(self, index: int) -> dict:
        return self.tensors[str(index)]


def checkpoint_root(app_dir: Path | str) -> Path:
    return Path(app_dir) / "checkpoints"


def _write_manifest(cp: Checkpoint) -> None:
    data = {
        "version": MANIFEST_VERSION,
        "backend": cp.backend,
        "params": cp.params,
        "output_path": cp.output_path,
        "total": cp.total,
        "created_at": cp.created_at,
        "finished": cp.finished,
        "tensors": cp.tensors,
    }
    manifest = cp.path / MANIFEST_NAME
    tmp = cp.path / f".{MANIFEST_NAME}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(manifest)  # atomic on POSIX and Windows (os.replace)


def _read_manifest(path: Path) -> Checkpoint:
    data = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
    return Checkpoint(
        path=path,
        backend=data.get("backend", "unknown"),
        params=data.get("params", {}),
        output_path=data.get("output_path", ""),
        total=int(data.get("total", 0)),
        created_at=data.get("created_at", ""),
        tensors=data.get("tensors", {}),
        finished=bool(data.get("finished", False)),
    )


def create_checkpoint(
    root: Path | str,
    backend: str,
    params: dict,
    output_path: str,
    total: int = 0,
) -> Checkpoint:
    """Create a fresh checkpoint directory and manifest."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for _ in range(100):
        candidate = root / f"{backend}-{stamp}-{int(time.time() * 1000) % 100000:05d}"
        if not candidate.exists():
            break
        time.sleep(0.01)
    else:  # pragma: no cover - vanishingly unlikely
        raise RuntimeError("Could not allocate a unique checkpoint directory")
    (candidate / SHARD_DIR_NAME).mkdir(parents=True)
    cp = Checkpoint(
        path=candidate,
        backend=backend,
        params=params,
        output_path=output_path,
        total=total,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    _write_manifest(cp)
    return cp


def record_tensor(cp: Checkpoint, index: int, key: str, **extra) -> None:
    """Record one fully-processed tensor (and any backend-specific metadata).

    The backend must have written the tensor's shard file (via
    `cp.shard_path(index)` + its own serializer) *before* calling this, so
    the manifest never points at a shard that isn't fully on disk yet.
    """
    entry = {"key": key}
    entry.update(extra)
    cp.tensors[str(index)] = entry
    _write_manifest(cp)


def mark_finished(cp: Checkpoint) -> None:
    cp.finished = True
    _write_manifest(cp)


def set_total(cp: Checkpoint, total: int) -> None:
    """Record the model's tensor count once the input header has been read
    (the UI creates the checkpoint before the backend opens the file)."""
    if cp.total != total:
        cp.total = total
        _write_manifest(cp)


def load_checkpoint(root: Path | str, checkpoint_id: str) -> Checkpoint:
    return _read_manifest(Path(root) / checkpoint_id)


def list_checkpoints(root: Path | str) -> list[Checkpoint]:
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for child in root.iterdir():
        if child.is_dir() and (child / MANIFEST_NAME).is_file():
            try:
                out.append(_read_manifest(child))
            except (OSError, json.JSONDecodeError):
                continue  # a half-written checkpoint dir - skip, don't crash
    out.sort(key=lambda cp: (cp.created_at, cp.id), reverse=True)
    return out


def delete_checkpoint(root: Path | str, checkpoint_id: str) -> None:
    path = Path(root) / checkpoint_id
    if path.is_dir() and (path / MANIFEST_NAME).is_file():
        shutil.rmtree(path)


def summary(cp: Checkpoint, input_label: str | None = None) -> str:
    """One-line description for dropdowns / info text."""
    name = input_label
    if not name:
        raw = cp.params.get("input_path") or cp.params.get("input") or ""
        name = Path(str(raw)).name if raw else "(unknown input)"
    if cp.backend == "ctq":
        progress = "session snapshot"
    elif cp.total:
        progress = f"{cp.completed_count}/{cp.total} tensors"
    else:
        progress = f"{cp.completed_count} tensors"
    suffix = " · ✅ done" if cp.finished else ""
    return f"{cp.backend} · {name} · {progress} · {cp.created_at}{suffix}"
