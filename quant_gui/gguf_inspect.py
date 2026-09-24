"""Read-only GGUF inspector: structure, type histogram, bpw, biggest tensors.

Backed purely by gguf-py's GGUFReader - no model code runs, so inspecting a
file is seconds regardless of size. Used by the GUI's Inspector tab and
available as a library function for tests/tools.
"""
from __future__ import annotations

from pathlib import Path

import gguf
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFReader


def _field_text(field) -> str:
    try:
        return bytes(field.parts[-1]).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - metadata is best-effort display only
        return repr(field.parts[-1])[:80]


def inspect_gguf(path: str | Path) -> dict:
    """Return a structured description of a GGUF file."""
    path = Path(path)
    reader = GGUFReader(str(path))

    arch = ""
    file_type = ""
    metadata: dict[str, str] = {}
    for key, field in reader.fields.items():
        if key == "general.architecture":
            arch = _field_text(field)
        elif key == "general.file_type":
            try:
                file_type = gguf.LlamaFileType(int(field.parts[-1][0])).name
            except Exception:  # noqa: BLE001
                file_type = str(field.parts[-1][0])
        elif not key.startswith("general."):  # skip duplicated header noise
            metadata[key] = _field_text(field)[:200]

    per_type: dict[str, dict] = {}
    total_bytes = 0
    total_params = 0
    biggest: list[dict] = []
    for t in reader.tensors:
        tname = t.tensor_type.name
        entry = per_type.setdefault(tname, {"count": 0, "bytes": 0, "params": 0})
        entry["count"] += 1
        entry["bytes"] += int(t.n_bytes)
        block, _ = GGML_QUANT_SIZES[t.tensor_type]
        entry["params"] += int(t.n_elements)
        total_bytes += int(t.n_bytes)
        total_params += int(t.n_elements)
        biggest.append({"name": t.name, "type": tname, "bytes": int(t.n_bytes),
                        "shape": [int(d) for d in t.shape]})
    biggest.sort(key=lambda d: -d["bytes"])

    return {
        "path": str(path),
        "file_size": path.stat().st_size,
        "arch": arch,
        "file_type": file_type,
        "tensor_count": len(reader.tensors),
        "total_bytes": total_bytes,
        "total_params": total_params,
        "bits_per_weight": (total_bytes * 8 / total_params) if total_params else 0.0,
        "per_type": per_type,
        "biggest_tensors": biggest[:10],
        "metadata": metadata,
    }


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def format_inspection(info: dict) -> str:
    """Render inspect_gguf() output as Markdown for the GUI."""
    lines = [
        f"## {Path(info['path']).name}",
        "",
        f"- **Architecture:** `{info['arch'] or '(none)'}`",
        f"- **File type:** `{info['file_type'] or '(none)'}`",
        f"- **File size:** {_gb(info['file_size'])}",
        f"- **Tensors:** {info['tensor_count']}",
        f"- **Tensor payload:** {_gb(info['total_bytes'])}",
        f"- **Params:** {info['total_params'] / 1e9:.2f} B",
        f"- **Bits per weight:** {info['bits_per_weight']:.3f}",
        "",
        "### Type histogram",
        "",
        "| Type | Tensors | Payload | Params | bpw |",
        "|---|---|---|---|---|",
    ]
    for tname, e in sorted(info["per_type"].items(),
                           key=lambda kv: -kv[1]["bytes"]):
        block, _ = GGML_QUANT_SIZES[GGMLQuantizationType[tname]]
        bpw = e["bytes"] * 8 / e["params"] if e["params"] else 0.0
        lines.append(f"| {tname} | {e['count']} | {_gb(e['bytes'])} "
                     f"| {e['params'] / 1e6:.0f} M | {bpw:.2f} |")
    lines += ["", "### Biggest tensors", "", "| Tensor | Type | Shape | Size |",
              "|---|---|---|---|"]
    for t in info["biggest_tensors"]:
        lines.append(f"| {t['name']} | {t['type']} | {t['shape']} | {_gb(t['bytes'])} |")
    if info["metadata"]:
        lines += ["", "### Metadata", ""]
        for k, v in sorted(info["metadata"].items()):
            lines.append(f"- `{k}`: {v}")
    return "\n".join(lines)
