"""A GGUF header reader for the quantisation tools (ported from 8bit-boom/Llm-training worker/ndforge_worker/ggufinfo.py): everything the existing minimal reader (imatrix.read_gguf_header)
leaves out, because those tools need it: the length of every array (the vocabulary size is the length of `tokenizer.ggml.tokens`), and for every tensor its shape, its
type and where its data starts. Only the header is read, never the weights; nothing here raises anything but ValueError (not a readable GGUF) and OSError (cannot open).

Also the bits-per-weight table of the ggml types: bytes per block x 8 / values per block, read from the `static_assert(sizeof(block_*))` lines of llama.cpp's
`ggml/src/ggml-common.h` (QK_K = 256, K_SCALE_SIZE = 12, IQ3S_N_SCALE = 4) and `ggml/include/ggml.h` for the numbers of the types."""
import struct
from pathlib import Path

# ggml type number -> (name, bits per weight)
TYPES = {
    0: ("F32", 32.0), 1: ("F16", 16.0), 30: ("BF16", 16.0), 28: ("F64", 64.0),
    2: ("Q4_0", 4.5), 3: ("Q4_1", 5.0), 6: ("Q5_0", 5.5), 7: ("Q5_1", 6.0), 8: ("Q8_0", 8.5), 9: ("Q8_1", 9.0),
    10: ("Q2_K", 2.625), 11: ("Q3_K", 3.4375), 12: ("Q4_K", 4.5), 13: ("Q5_K", 5.5), 14: ("Q6_K", 6.5625), 15: ("Q8_K", 9.125),
    16: ("IQ2_XXS", 2.0625), 17: ("IQ2_XS", 2.3125), 22: ("IQ2_S", 2.5625), 18: ("IQ3_XXS", 3.0625), 21: ("IQ3_S", 3.4375),
    19: ("IQ1_S", 1.5625), 29: ("IQ1_M", 1.75), 20: ("IQ4_NL", 4.5), 23: ("IQ4_XS", 4.25),
    24: ("I8", 8.0), 25: ("I16", 16.0), 26: ("I32", 32.0), 27: ("I64", 64.0), 34: ("TQ1_0", 1.6875), 35: ("TQ2_0", 2.0625),
}
BPW = {name: bpw for name, bpw in TYPES.values()}
# `general.file_type` of the header (llama_ftype): the 16-bit and 32-bit files
FTYPE_FULL = {0: "F32", 1: "F16", 32: "BF16"}

_FIXED = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_FIXED_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def type_name(t: int) -> str:
    return TYPES[t][0] if t in TYPES else f"type{t}"


def read(path) -> dict:
    """{"version", "meta": {key: value}, "tensors": [{"name", "dims", "type", "offset"}], "data_start": int}. An array value is None and its length is under
    `<key>#len`; an array of strings with at most 64 entries (an importance matrix's `imatrix.datasets`) also has them under `<key>#items`. `offset` is relative to `data_start`. Raises ValueError when the file is not a readable GGUF (not one, version below 2, truncated, absurd sizes)."""
    try:
        with open(path, "rb") as fh:
            def take(fmt):
                size = struct.calcsize(fmt)
                raw = fh.read(size)
                if len(raw) != size:
                    raise ValueError("short")
                return struct.unpack("<" + fmt, raw)[0]

            def string(keep=True):
                n = take("Q")
                if n > 1 << 24:
                    raise ValueError("long string")
                if keep:
                    raw = fh.read(n)
                    if len(raw) != n:
                        raise ValueError("short")
                    return raw.decode("utf-8", "replace")
                fh.seek(n, 1)

            def value(kind, keep=True):
                if kind in _FIXED:
                    return take(_FIXED[kind])
                if kind == 8:
                    return string(keep)
                if kind == 9:
                    inner, n = take("I"), take("Q")
                    if inner in _FIXED_SIZE:
                        fh.seek(_FIXED_SIZE[inner] * n, 1)
                        return ("array", n, None)
                    items = []
                    for _ in range(n):
                        items.append(value(inner, keep and inner == 8 and n <= 64))
                    return ("array", n, items if keep and inner == 8 and n <= 64 else None)
                raise ValueError("unknown value type")

            if fh.read(4) != b"GGUF":
                raise ValueError("not a GGUF file")
            version = take("I")
            if version < 2:
                raise ValueError("an old GGUF version")
            n_tensors, n_kv = take("Q"), take("Q")
            if n_tensors > 1 << 22 or n_kv > 1 << 22:
                raise ValueError("absurd counts")
            meta = {}
            for _ in range(n_kv):
                key = string()
                v = value(take("I"))
                if isinstance(v, tuple):
                    meta[key + "#len"] = v[1]
                    if v[2] is not None:
                        meta[key + "#items"] = v[2]
                    v = None
                meta[key] = v
            tensors = []
            for _ in range(n_tensors):
                name = string()
                nd = take("I")
                if nd > 8:
                    raise ValueError("absurd dimensions")
                dims = [take("Q") for _ in range(nd)]
                typ, off = take("I"), take("Q")
                tensors.append({"name": name, "dims": dims, "type": typ, "offset": off})
            align = meta.get("general.alignment")
            align = align if isinstance(align, int) and align > 0 else 32
            pos = fh.tell()
            return {"version": version, "meta": meta, "tensors": tensors, "data_start": (pos + align - 1) // align * align}
    except struct.error as exc:
        raise ValueError(str(exc))


def numel(t: dict) -> int:
    n = 1
    for d in t["dims"]:
        n *= d
    return n


def n_parameters(info: dict) -> int:
    return sum(numel(t) for t in info["tensors"])


def vocab_size(info: dict):
    """The number of tokens of the vocabulary (the length of the token list, else `<arch>.vocab_size`), None when the file does not say."""
    meta = info["meta"]
    n = meta.get("tokenizer.ggml.tokens#len")
    if isinstance(n, int) and n > 0:
        return n
    arch = meta.get("general.architecture")
    v = meta.get(f"{arch}.vocab_size") if isinstance(arch, str) else None
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None


def architecture(info: dict):
    a = info["meta"].get("general.architecture")
    return a if isinstance(a, str) else None


def is_full_precision(info: dict):
    """True when the file is F32 / F16 / BF16 (by `general.file_type`, else by the types of its big tensors), False when quantised, None when it cannot be told."""
    ft = info["meta"].get("general.file_type")
    if isinstance(ft, int) and not isinstance(ft, bool):
        return ft in FTYPE_FULL
    big = [t for t in info["tensors"] if len(t["dims"]) >= 2]
    if not big:
        return None
    return all(t["type"] in (0, 1, 30) for t in big)


def describe(path) -> dict:
    """What the comparison needs to know about a GGUF file: {"size", "params", "bpw" (file size x 8 / parameters), "vocab", "arch", "full_precision", "tensors"}.
    Raises ValueError / OSError like `read`."""
    info = read(path)
    size = Path(path).stat().st_size
    params = n_parameters(info)
    return {"size": size, "params": params, "bpw": round(size * 8 / params, 4) if params else None, "vocab": vocab_size(info), "arch": architecture(info),
            "full_precision": is_full_precision(info), "tensors": len(info["tensors"])}
