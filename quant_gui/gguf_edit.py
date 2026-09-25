"""Edit a GGUF file's metadata (and tensor names) into a new file.

Modelled on the Hugging Face "GGUF Editor" space, but local: load any GGUF,
change/add/delete metadata keys, optionally rename or drop tensors, and save
a new file. Tensor payloads are copied byte-for-byte - never re-quantized -
and stream through a disk-spooled temp file, so a 26 GB model edits in
constant memory.

Implementation notes (verified against the installed gguf package):
- GGUFReader mmaps the file, so per-tensor .data views cost no RAM.
- GGUFWriter(use_temp_file=True) spills tensor payloads to a
  SpooledTemporaryFile (256 MB RAM cap) and copies them into the output at
  the end, so add_tensor() never holds the whole model in memory.
- ReaderField.contents is a *method*; for scalar fields it returns the raw
  value, for arrays a list or np.ndarray.
- Only little-endian files are supported (all mainstream GGUFs); big-endian
  is refused rather than silently byte-swapped.
- Writes are crash-atomic: output goes to <name>.tmp and is os.replace()d
  into place only after the writer closes cleanly (same pattern as the
  conversion backends).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCALAR_TYPES = (
    "UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32",
    "UINT64", "INT64", "FLOAT32", "FLOAT64", "BOOL", "STRING",
)
ALL_TYPES = SCALAR_TYPES + ("ARRAY",)

#: arrays longer than this are shown read-only in the UI (they're things
#: like the 256k-entry tokenizer token list - not something you hand-edit).
ARRAY_EDIT_LIMIT = 64

_INT_TYPES = {"UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32",
              "UINT64", "INT64"}
_FLOAT_TYPES = {"FLOAT32", "FLOAT64"}
_ARRAY_INT_SUBTYPES = _INT_TYPES | {"BOOL"}
_ARRAY_FLOAT_SUBTYPES = _FLOAT_TYPES


class GGUFEditError(Exception):
    """Raised for any user-facing edit failure (bad value, missing key...)."""


@dataclass
class EditableKV:
    key: str
    vtype: str            # SCALAR_TYPES or "ARRAY"
    sub_type: str | None  # element type for ARRAY, else None
    value: object         # python value (scalar) or list/ndarray (array)
    editable: bool        # False: display-only (huge array)

    def display_value(self) -> str:
        """Value as shown in the UI table."""
        if self.vtype == "ARRAY":
            items = list(self.value)
            if self.editable:
                return "[" + ", ".join(
                    repr(x) if isinstance(x, str) else str(x) for x in items) + "]"
            n = len(items)
            head = items[:ARRAY_EDIT_LIMIT]
            shown = ", ".join(repr(x) if isinstance(x, str) else str(x)
                              for x in head)
            suffix = ", ..." if n > len(head) else ""
            return f"[{self.sub_type} x {n}] [{shown}{suffix}]"
        if self.vtype == "STRING":
            return str(self.value)
        return repr(self.value) if self.vtype == "BOOL" else str(self.value)


@dataclass
class EditableTensor:
    name: str
    ggml_type: str
    shape: tuple[int, ...]
    n_bytes: int


@dataclass
class EditPlan:
    """Everything the UI needs to render one loaded GGUF."""
    source: str
    architecture: str
    metadata: list[EditableKV] = field(default_factory=list)
    tensors: list[EditableTensor] = field(default_factory=list)
    n_kv: int = 0
    n_tensors: int = 0
    total_bytes: int = 0

    def rows(self) -> list[list[str]]:
        return [[kv.key, kv.vtype if kv.sub_type is None else
                 f"ARRAY[{kv.sub_type}]", kv.display_value(),
                 "yes" if kv.editable else "read-only"]
                for kv in self.metadata]


def parse_scalar(vtype: str, text: str):
    """Parse a UI text field into a typed value. Raises GGUFEditError."""
    text = (text or "").strip() if vtype != "STRING" else (text or "")
    try:
        if vtype in _INT_TYPES:
            v = int(text, 10)
            bits = int(vtype[4 if vtype.startswith("UINT") else 3:])
            lo = 0 if vtype.startswith("UINT") else -(2 ** (bits - 1))
            hi = 2 ** bits - 1 if vtype.startswith("UINT") else 2 ** (bits - 1) - 1
            if not lo <= v <= hi:
                raise ValueError(f"out of range for {vtype} [{lo}, {hi}]")
            return v
        if vtype in _FLOAT_TYPES:
            return float(text)
        if vtype == "BOOL":
            low = text.strip().lower()
            if low in ("true", "1", "yes", "on"):
                return True
            if low in ("false", "0", "no", "off"):
                return False
            raise ValueError("expected true/false")
        if vtype == "STRING":
            return text
    except ValueError as exc:
        raise GGUFEditError(f"{vtype}: can't parse {text!r} ({exc})") from exc
    raise GGUFEditError(f"Unknown type {vtype!r}")


def parse_array(sub_type: str, items: list):
    """Coerce a parsed JSON list into the right element types."""
    if not isinstance(items, list):
        raise GGUFEditError("Array value must be a JSON list, e.g. [1, 2, 3]")
    if sub_type == "STRING":
        return [str(x) for x in items]
    if sub_type == "BOOL":
        return [bool(x) for x in items]
    if sub_type in _ARRAY_INT_SUBTYPES:
        return [int(x) for x in items]
    if sub_type in _ARRAY_FLOAT_SUBTYPES:
        return [float(x) for x in items]
    raise GGUFEditError(f"Unsupported array element type {sub_type!r}")


def load_for_edit(path: str | Path) -> EditPlan:
    """Read a GGUF's metadata + tensor index without loading tensor data."""
    from gguf import GGUFEndian, GGUFReader

    path = Path(path)
    if not path.is_file():
        raise GGUFEditError(f"File not found: {path}")
    try:
        reader = GGUFReader(str(path))
    except Exception as exc:  # noqa: BLE001
        raise GGUFEditError(f"Not a readable GGUF: {exc}") from exc
    if reader.endianess != GGUFEndian.LITTLE:
        raise GGUFEditError("Big-endian GGUF files aren't supported by the editor.")

    arch_field = reader.fields.get("general.architecture")
    arch = str(arch_field.contents()) if arch_field is not None else "unknown"

    plan = EditPlan(source=str(path), architecture=arch,
                    n_tensors=len(reader.tensors))
    for key, f in reader.fields.items():
        if key in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
            continue  # structural, rewritten by the writer
        types = [t.name for t in f.types]
        vtype = types[0]
        value = f.contents()
        if vtype == "ARRAY":
            sub = types[1] if len(types) > 1 else "UINT8"
            items = list(value) if isinstance(value, np.ndarray) else list(value)
            plan.metadata.append(EditableKV(
                key=key, vtype="ARRAY", sub_type=sub, value=items,
                editable=len(items) <= ARRAY_EDIT_LIMIT,
            ))
        else:
            plan.metadata.append(EditableKV(
                key=key, vtype=vtype, sub_type=None, value=value, editable=True,
            ))
    plan.n_kv = len(plan.metadata)
    for t in reader.tensors:
        shape = tuple(int(d) for d in t.shape)
        plan.tensors.append(EditableTensor(
            name=t.name, ggml_type=t.tensor_type.name, shape=shape,
            n_bytes=int(t.n_bytes),
        ))
        plan.total_bytes += int(t.n_bytes)
    return plan


def _writer_add_kv(writer, kv: EditableKV) -> None:
    from gguf import GGUFValueType
    if kv.vtype == "ARRAY":
        writer.add_array(kv.key, list(kv.value))
        return
    vtype = GGUFValueType[kv.vtype]
    if kv.vtype == "STRING":
        writer.add_string(kv.key, str(kv.value))
    else:
        writer.add_key_value(kv.key, kv.value, vtype)


def save_edited(
    src: str | Path,
    dst: str | Path,
    set_meta: dict[str, EditableKV] | None = None,
    del_keys: list[str] | tuple[str, ...] | None = None,
    renames: dict[str, str] | None = None,
    drop_tensors: list[str] | tuple[str, ...] | None = None,
    progress_cb=None,
) -> dict:
    """Write a copy of `src` to `dst` with edits applied.

    set_meta: key -> EditableKV with the new value/type (adds new keys too).
    del_keys: metadata keys to omit.
    renames: old tensor name -> new tensor name (data copied verbatim).
    drop_tensors: tensor names to omit from the output.
    progress_cb(done_bytes, total_bytes) is called per tensor.
    Returns a stats dict. The output replaces any existing file atomically.
    """
    from gguf import GGUFReader, GGUFWriter

    set_meta = dict(set_meta or {})
    del_keys = set(del_keys or ())
    renames = dict(renames or {})
    drop_tensors = set(drop_tensors or ())

    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise GGUFEditError(f"Source not found: {src}")
    if str(dst.resolve()) == str(src.resolve()):
        raise GGUFEditError("Output must differ from the input file.")

    reader = GGUFReader(str(src))
    arch_field = reader.fields.get("general.architecture")
    arch = str(arch_field.contents()) if arch_field is not None else "unknown"

    tensor_names = [t.name for t in reader.tensors]
    for old in renames:
        if old not in tensor_names:
            raise GGUFEditError(f"Can't rename {old!r} - no such tensor.")
    for name in drop_tensors:
        if name not in tensor_names:
            raise GGUFEditError(f"Can't drop {name!r} - no such tensor.")
    if len(set_meta) + len(del_keys) == 0 and not renames and not drop_tensors:
        raise GGUFEditError("Nothing to change - no edits requested.")

    tmp = dst.with_name(dst.name + ".tmp")
    writer = GGUFWriter(str(tmp), arch=arch, use_temp_file=True)

    # Preserve the source's data alignment (default 32).
    align = reader.fields.get("general.alignment")
    if align is not None:
        try:
            writer.data_alignment = int(align.contents())
        except (TypeError, ValueError):
            pass

    try:
        n_kv = 0
        original_keys = set()
        for key, f in reader.fields.items():
            if key in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                       "general.architecture"):
                continue
            original_keys.add(key)
            if key in del_keys:
                continue
            if key in set_meta:
                kv = set_meta[key]
                kv.key = key
                _writer_add_kv(writer, kv)
            else:
                types = [t.name for t in f.types]
                value = f.contents()
                if types[0] == "ARRAY":
                    items = (list(value) if isinstance(value, np.ndarray)
                             else list(value))
                    writer.add_array(key, items)
                elif types[0] == "STRING":
                    writer.add_string(key, str(value))
                else:
                    from gguf import GGUFValueType
                    writer.add_key_value(key, value, GGUFValueType[types[0]])
            n_kv += 1
        for key, kv in set_meta.items():
            if key not in original_keys and key not in del_keys:
                kv.key = key
                _writer_add_kv(writer, kv)
                n_kv += 1

        total = sum(int(t.n_bytes) for t in reader.tensors
                    if t.name not in drop_tensors)
        done = 0
        n_tensors = 0
        for t in reader.tensors:
            if t.name in drop_tensors:
                continue
            name = renames.get(t.name, t.name)
            writer.add_tensor(name, np.asarray(t.data), raw_dtype=t.tensor_type)
            done += int(t.n_bytes)
            n_tensors += 1
            if progress_cb is not None:
                progress_cb(done, total)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        os.replace(tmp, dst)
    except BaseException:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass
        tmp.unlink(missing_ok=True)
        raise

    return {"output": str(dst), "architecture": arch, "n_kv": n_kv,
            "n_tensors": n_tensors,
            "n_renamed": len(renames), "n_dropped": len(drop_tensors),
            "bytes_copied": done}
