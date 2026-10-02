"""Sharded safetensors checkpoints (`model-00001-of-00005.safetensors` plus an
optional `model.safetensors.index.json`), the format large video/image/LLM
checkpoints ship in.

Two ways the rest of the app consumes a shard set:

- **Natively** (INT4 and GGUF backends, size estimates): `open_checkpoint`
  returns a drop-in replacement for `safetensors.safe_open` that presents all
  shards as one tensor namespace. Nothing is copied or re-saved.
- **Merged** (the ctq formats): convert_to_quant only reads a single file and
  has no shard/index handling at all (checked against its source), so
  `iter_merge` streams the shards into one `.safetensors` by copying raw byte
  ranges. It never decodes a tensor, so any dtype works (including bf16/fp8)
  and memory stays at one copy buffer, but it needs free disk equal to the
  model size - that is checked up front rather than failing halfway.

Anything a user can hand over resolves to the same ShardSet: a shard file,
the `.index.json`, or a folder containing exactly one set.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

INDEX_SUFFIX = ".safetensors.index.json"
SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d{3,6})-of-(?P<total>\d{3,6})(?P<suffix>\.safetensors)$")
COPY_CHUNK = 64 * 1024 * 1024
HEADER_ALIGN = 8


class ShardError(ValueError):
    """The shard set is incomplete, ambiguous, or inconsistent."""


@dataclass
class ShardSet:
    files: list[Path]  # every shard, in order, all verified to exist
    logical_name: str  # "model" for model-00001-of-00005.safetensors
    index_path: Path | None = None
    weight_map: dict[str, str] | None = None

    @property
    def entry_path(self) -> Path:
        """The path to hand around as 'the model': the index if there is one,
        else the first shard. Both resolve back to this same ShardSet."""
        return self.index_path or self.files[0]

    @property
    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.files)


def logical_stem(path: str | Path) -> str:
    """Filename stem with any shard/index suffix removed, for output naming:
    model-00001-of-00005.safetensors -> model, x.safetensors.index.json -> x."""
    if Path(path).is_dir():  # a folder of shards: name the output after the model, not the folder
        try:
            found = _from_directory(Path(path))
        except ShardError:
            found = None
        return found.logical_name if found else Path(path).name
    name = Path(path).name
    if name.endswith(INDEX_SUFFIX):
        return name[: -len(INDEX_SUFFIX)]
    m = SHARD_RE.match(name)
    if m:
        return m.group("prefix")
    return Path(name).stem


def _shard_group(directory: Path, prefix: str, suffix: str, total: int) -> tuple[list[Path], list[int]]:
    """Existing shards of one set, and the indices (1-based) that are missing."""
    found: dict[int, Path] = {}
    for p in directory.iterdir():
        m = SHARD_RE.match(p.name)
        if m and m.group("prefix") == prefix and m.group("suffix") == suffix and int(m.group("total")) == total:
            found[int(m.group("idx"))] = p
    missing = [i for i in range(1, total + 1) if i not in found]
    return [found[i] for i in sorted(found)], missing


def _missing_message(prefix: str, total: int, missing: list[int], width: int) -> str:
    names = ", ".join(f"{prefix}-{i:0{width}d}-of-{total:0{width}d}.safetensors" for i in missing[:5])
    more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
    return (
        f"Sharded checkpoint '{prefix}' is incomplete: {len(missing)} of {total} shard files are missing - "
        f"{names}{more}. Download them into the same folder first."
    )


def _from_index(index_path: Path) -> ShardSet:
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data["weight_map"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ShardError(f"Couldn't read the shard index {index_path.name}: {exc}") from exc
    if not isinstance(weight_map, dict) or not weight_map:
        raise ShardError(f"{index_path.name} has an empty or invalid weight_map.")

    names = sorted(set(weight_map.values()))
    missing = [n for n in names if not (index_path.parent / n).is_file()]
    if missing:
        listed = ", ".join(missing[:5]) + (f" (and {len(missing) - 5} more)" if len(missing) > 5 else "")
        raise ShardError(
            f"{index_path.name} lists {len(names)} shard files but {len(missing)} are missing: {listed}. "
            "Download them into the same folder first."
        )

    def order(n: str) -> tuple[int, str]:
        m = SHARD_RE.match(n)
        return (int(m.group("idx")) if m else 0, n)

    files = [index_path.parent / n for n in sorted(names, key=order)]
    return ShardSet(files=files, logical_name=logical_stem(index_path), index_path=index_path, weight_map=weight_map)


def _from_shard_file(path: Path) -> ShardSet:
    m = SHARD_RE.match(path.name)
    assert m is not None
    prefix, suffix, total = m.group("prefix"), m.group("suffix"), int(m.group("total"))
    files, missing = _shard_group(path.parent, prefix, suffix, total)
    if missing:
        raise ShardError(_missing_message(prefix, total, missing, len(m.group("idx"))))
    index = path.parent / f"{prefix}{INDEX_SUFFIX}"
    if index.is_file():
        return _from_index(index)
    return ShardSet(files=files, logical_name=prefix)


def _from_directory(directory: Path) -> ShardSet | None:
    indexes = sorted(directory.glob(f"*{INDEX_SUFFIX}"))
    if len(indexes) == 1:
        return _from_index(indexes[0])
    if len(indexes) > 1:
        raise ShardError(
            f"{directory.name}/ contains {len(indexes)} shard indexes ({', '.join(i.name for i in indexes)}). "
            "Point at the one you want directly."
        )
    groups: dict[tuple[str, str, int], Path] = {}
    for p in sorted(directory.glob("*.safetensors")):
        m = SHARD_RE.match(p.name)
        if m:
            groups.setdefault((m.group("prefix"), m.group("suffix"), int(m.group("total"))), p)
    if len(groups) == 1:
        return _from_shard_file(next(iter(groups.values())))
    if len(groups) > 1:
        raise ShardError(
            f"{directory.name}/ contains {len(groups)} different shard sets "
            f"({', '.join(sorted(g[0] for g in groups))}). Point at one shard file directly."
        )
    return None


def find_shards(path: str | Path) -> ShardSet | None:
    """The ShardSet `path` belongs to, or None if it's a plain single file
    (or isn't a safetensors path at all). Raises ShardError for a set that is
    incomplete or ambiguous - never returns a partial set."""
    p = Path(path)
    if p.is_dir():
        return _from_directory(p)
    if p.name.endswith(INDEX_SUFFIX):
        return _from_index(p) if p.is_file() else None
    if SHARD_RE.match(p.name):
        return _from_shard_file(p)
    return None


def input_exists(path: str | Path) -> bool:
    """True if `path` is a real input: a file, or a folder holding one valid
    shard set. (An incomplete set still 'exists' so the caller reaches
    find_shards and gets the specific missing-shard message.)"""
    p = Path(path)
    if p.is_file():
        return True
    if p.is_dir():
        try:
            return _from_directory(p) is not None
        except ShardError:
            return True
    return False


def describe(path: str | Path) -> str:
    """One line for logs, e.g. '5 shards, 26.1 GB (index: model.safetensors.index.json)'."""
    s = find_shards(path)
    if s is None:
        return ""
    idx = f", index: {s.index_path.name}" if s.index_path else ", no index file"
    gb = s.total_bytes / 1024**3
    size = f"{gb:.2f} GB" if gb >= 0.5 else f"{s.total_bytes / 1024**2:.0f} MB"
    return f"{len(s.files)} shards, {size}{idx}"


# --------------------------------------------------------------------------- raw headers


def read_header(path: str | Path) -> tuple[dict, int]:
    """(header json, byte offset where tensor data starts) for one file."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ShardError(f"{Path(path).name} is too small to be a safetensors file.")
        (n,) = struct.unpack("<Q", raw)
        body = f.read(n)
    if len(body) < n:
        raise ShardError(f"{Path(path).name} has a truncated header - is it fully downloaded?")
    try:
        return json.loads(body), 8 + n
    except ValueError as exc:
        raise ShardError(f"Couldn't parse {Path(path).name}'s safetensors header: {exc}") from exc


# --------------------------------------------------------------------------- native reading


class ShardedSafeOpen:
    """Stand-in for `safetensors.safe_open(...)` over a whole shard set: the
    subset the backends use (`keys`, `get_tensor`, `metadata`, context manager)."""

    def __init__(self, shard_set: ShardSet, framework: str = "pt", device: str = "cpu"):
        self._set = shard_set
        self._framework = framework
        self._device = device
        self._stack = contextlib.ExitStack()
        self._handles: list = []
        self._owner: dict[str, int] = {}
        self._keys: list[str] = []

    def __enter__(self) -> "ShardedSafeOpen":
        from safetensors import safe_open

        try:
            for i, shard in enumerate(self._set.files):
                handle = self._stack.enter_context(safe_open(str(shard), framework=self._framework, device=self._device))
                self._handles.append(handle)
                for key in handle.keys():
                    if key in self._owner:
                        raise ShardError(
                            f"Tensor '{key}' appears in both {self._set.files[self._owner[key]].name} and "
                            f"{shard.name} - these shards don't belong to one checkpoint."
                        )
                    self._owner[key] = i
                    self._keys.append(key)
            if self._set.weight_map:
                absent = [k for k in self._set.weight_map if k not in self._owner]
                if absent:
                    raise ShardError(
                        f"The index lists {len(absent)} tensors not found in any shard (e.g. '{absent[0]}') "
                        "- a shard is probably the wrong version or truncated."
                    )
        except BaseException:
            self._stack.close()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self._stack.close()

    def keys(self) -> list[str]:
        return list(self._keys)

    def get_tensor(self, key: str):
        return self._handles[self._owner[key]].get_tensor(key)

    def metadata(self) -> dict | None:
        merged: dict = {}
        for h in self._handles:
            for k, v in (h.metadata() or {}).items():
                merged.setdefault(k, v)
        return merged or None


def open_checkpoint(path: str | Path, framework: str = "pt", device: str = "cpu"):
    """`safe_open` for a single file, `ShardedSafeOpen` for a shard set."""
    shard_set = find_shards(path)
    if shard_set is not None:
        return ShardedSafeOpen(shard_set, framework=framework, device=device)
    from safetensors import safe_open

    return safe_open(str(path), framework=framework, device=device)


# --------------------------------------------------------------------------- merging for ctq


def _merge_plan(shard_set: ShardSet) -> tuple[bytes, list[tuple[Path, int, int]], int]:
    """(new header bytes incl. length prefix, [(shard, abs_start, nbytes)] in
    write order, total data bytes)."""
    entries = []
    metadata: dict = {}
    seen: dict[str, str] = {}
    for shard in shard_set.files:
        header, data_start = read_header(shard)
        for k, v in (header.pop("__metadata__", None) or {}).items():
            metadata.setdefault(k, v)
        for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            if name in seen:
                raise ShardError(f"Tensor '{name}' appears in both {seen[name]} and {shard.name}.")
            seen[name] = shard.name
            lo, hi = info["data_offsets"]
            entries.append((name, info["dtype"], info["shape"], shard, data_start + lo, hi - lo))

    new_header: dict = {}
    if metadata:
        new_header["__metadata__"] = metadata
    cursor = 0
    plan = []
    for name, dtype, shape, shard, abs_start, nbytes in entries:
        new_header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + nbytes]}
        plan.append((shard, abs_start, nbytes))
        cursor += nbytes

    body = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    body += b" " * (-len(body) % HEADER_ALIGN)  # safetensors wants data 8-byte aligned
    return struct.pack("<Q", len(body)) + body, plan, cursor


def merged_path_for(shard_set: ShardSet, merge_dir: str | Path) -> Path:
    """Cache location: same shard set (names, sizes, mtimes) -> same file, so
    re-running a conversion doesn't re-merge."""
    h = hashlib.sha1()
    for p in shard_set.files:
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return Path(merge_dir) / f"{shard_set.logical_name}-{h.hexdigest()[:8]}.safetensors"


def iter_merge(shard_set: ShardSet, out_path: str | Path):
    """Stream the shards into one safetensors file at `out_path`, yielding
    (bytes_done, bytes_total, shard_name) as it goes. Written to a temp name
    and renamed, so an interrupted merge never leaves a truncated file that
    looks valid. If an identical merged file already exists it is reused."""
    out_path = Path(out_path)
    header_bytes, plan, data_total = _merge_plan(shard_set)
    expected = len(header_bytes) + data_total

    if out_path.is_file() and out_path.stat().st_size == expected:
        yield expected, expected, "(already merged - reusing)"
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_path.parent).free
    if free < expected + (64 << 20):
        raise ShardError(
            f"Merging needs about {expected / 1024**3:.1f} GB free in {out_path.parent} but only "
            f"{free / 1024**3:.1f} GB is available. Free some space, or convert to INT4/GGUF, which read "
            "shards in place without merging."
        )

    tmp = out_path.with_name(out_path.name + ".partial")
    done = 0
    try:
        with open(tmp, "wb") as out:
            out.write(header_bytes)
            done = len(header_bytes)
            handles: dict[Path, object] = {}
            try:
                for shard, start, nbytes in plan:
                    src = handles.get(shard)
                    if src is None:
                        src = handles[shard] = open(shard, "rb")
                    src.seek(start)
                    left = nbytes
                    while left:
                        buf = src.read(min(COPY_CHUNK, left))
                        if not buf:
                            raise ShardError(f"{shard.name} ended early - is it fully downloaded?")
                        out.write(buf)
                        left -= len(buf)
                        done += len(buf)
                        yield done, expected, shard.name
            finally:
                for h in handles.values():
                    h.close()
        os.replace(tmp, out_path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


# --------------------------------------------------------------------------- listing


def collapse_listing(files: list[Path]) -> list[tuple[Path, int, int, bool]]:
    """Collapse shard files into one entry per set, for file pickers.
    Returns (entry_path, total_bytes, n_shards, complete); a plain file is
    (path, size, 1, True). Entry is the index file when one sits beside the shards."""
    groups: dict[tuple[Path, str, str, int], list[Path]] = {}
    out: list[tuple[Path, int, int, bool]] = []
    for f in files:
        m = SHARD_RE.match(f.name)
        if m:
            groups.setdefault((f.parent, m.group("prefix"), m.group("suffix"), int(m.group("total"))), []).append(f)
        else:
            out.append((f, f.stat().st_size, 1, True))
    for (parent, prefix, _suffix, total), members in groups.items():
        entry = parent / f"{prefix}{INDEX_SUFFIX}"
        out.append((
            entry if entry.is_file() else sorted(members)[0],
            sum(p.stat().st_size for p in members),
            total,
            len(members) == total,
        ))
    return out
