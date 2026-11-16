"""Mixed-precision GGUF sizes ("Mixed precision (our own recipe)")

Ported from the owner's own 8bit-boom/Llm-training repo (worker/ndforge_worker/mixedquant.py + ggufinfo.py, imported as quant_gui/gguf_header.py), stdlib only.
Never run against a real llama-quantize from here; see README "Mixed recipes".: llama-quantize with a base type, an ordered list of `--tensor-type REGEX=TYPE` rules, an
`--output-tensor-type` and a `--token-embedding-type`. The recipes spend more bits where common practice says a hybrid model needs them (the embedding and output tensors, the
attention k / v / output and the linear-attention projections, ffn_down, the first and last layers) and fewer on the ffn_up / ffn_gate of the middle layers.

These are OUR OWN recipes. They are not Unsloth's "Dynamic" quants (Unsloth's pipeline is not published and cannot be reproduced); no name, file or
text here says otherwise.

    PRESETS             the ONE table of recipes (data, not code)
    build_rules         the ordered rule list of a preset for an architecture and a layer count (pure; tested against real tensor-name lists)
    estimate            the size of the file a rule list makes, from the tensor shapes in the header of a 16-bit GGUF (pure, no weights are read)
    fit                 "Fit to a size" (experimental): upgrade tensors from the lowest type in order of importance per extra byte until a size budget is used (pure, deterministic)
    plan_job            everything the page shows before an export (every preset's rules and size, the fit)

llama-quantize facts this relies on (read in llama.cpp's `tools/quantize/quantize.cpp` and `src/llama-quant.cpp`, master, 2026-11): `--tensor-type NAME=TYPE` is a regular
expression searched in the tensor name (`std::regex_search`, ECMAScript), the FIRST rule that matches wins, the pattern text is lower-cased (so no `\\D`, `\\S`, `\\W`), the type
is a ggml type name (case-insensitive, `q4_k`, not `q4_k_m`); tensors no rule names follow the base type's own mixture; 1-D tensors, norms, `ssm_conv1d`, the expert router are
never quantised; a row length that does not fit the type's block size falls back to a smaller block type; `--output-tensor-type` and `--token-embedding-type` set those two."""
import heapq
import math
import re
import sys
from array import array
from pathlib import Path

from . import gguf_header as ggufinfo

LABEL = "Mixed precision (our own recipe)"
HYBRID_ARCHS = ("qwen35", "qwen35moe")          # tensor names read in llama.cpp's src/models/qwen35.cpp and src/llama-arch.cpp
LEVELS = ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K")           # what the recipes and the fit may use, lowest first
ROLES = ("attn_kvo", "attn_q", "lin", "ffn_down", "ffn_updown")

# THE table of recipes. `types[role] = (type in the first and last `edge` layers, type in the layers between)`; `base` is the ftype given to llama-quantize (the tensors no rule
# names follow it); `embd` / `output` go to --token-embedding-type / --output-tensor-type.
PRESETS = [
    {"id": "mixed_4", "label": "Mixed 4-bit (balanced)", "base": "Q4_K_M", "embd": "Q5_K", "output": "Q6_K", "edge": 3,
     "summary": "about 5 bits per weight on average: the attention k / v / output at 6 bits, the other attention and ffn_down at 5, ffn_up / gate of the middle layers at 4",
     "types": {"attn_kvo": ("Q6_K", "Q6_K"), "attn_q": ("Q6_K", "Q5_K"), "lin": ("Q6_K", "Q5_K"), "ffn_down": ("Q6_K", "Q5_K"), "ffn_updown": ("Q5_K", "Q4_K")}},
    {"id": "mixed_3", "label": "Mixed 3-bit (small)", "base": "Q3_K_M", "embd": "Q4_K", "output": "Q6_K", "edge": 2,
     "summary": "about 4 bits per weight on average: ffn_up / gate of the middle layers at 3 bits, the rest at 4 to 6",
     "types": {"attn_kvo": ("Q6_K", "Q5_K"), "attn_q": ("Q5_K", "Q4_K"), "lin": ("Q5_K", "Q4_K"), "ffn_down": ("Q5_K", "Q4_K"), "ffn_updown": ("Q4_K", "Q3_K")}},
    {"id": "mixed_2", "label": "Mixed 2-bit (smallest: short answers only)", "base": "Q2_K", "embd": "Q4_K", "output": "Q6_K", "edge": 2,
     "summary": "about 3 bits per weight on average: ffn_up / gate of the middle layers at 2 bits; quality drops clearly, long answers suffer most",
     "types": {"attn_kvo": ("Q6_K", "Q4_K"), "attn_q": ("Q5_K", "Q3_K"), "lin": ("Q5_K", "Q4_K"), "ffn_down": ("Q5_K", "Q3_K"), "ffn_updown": ("Q3_K", "Q2_K")}},
]
PRESET_IDS = tuple(p["id"] for p in PRESETS)
FIT_ID = "mixed_fit"
FIT_LABEL = "Mixed: fit to a size (experimental)"
FIT_BASE, FIT_EMBD, FIT_OUTPUT = "Q2_K", "Q4_K", "Q6_K"
MIXED_IDS = PRESET_IDS + (FIT_ID,)
# the plain sizes, as llama.cpp's own table gives them (tools/quantize/README.md, Llama-3.1-8B, bits per weight); other models differ a little
PLAIN_BPW = {"Q2_K": 3.1593, "Q3_K_M": 3.9960, "Q4_K_M": 4.8944, "Q5_K_M": 5.7036, "Q6_K": 6.5633, "Q8_0": 8.5008}
BASE_TYPE = {"Q2_K": "Q2_K", "Q3_K_M": "Q3_K", "Q4_K_M": "Q4_K"}      # the type a base ftype gives to a tensor no rule names (its own mixture adds some higher types)
BLOCK = {"Q2_K": 256, "Q3_K": 256, "Q4_K": 256, "Q5_K": 256, "Q6_K": 256, "Q8_0": 32, "Q4_0": 32, "Q5_0": 32, "Q5_1": 32, "IQ4_NL": 32}
FALLBACK = {"Q2_K": "Q4_0", "Q3_K": "Q4_0", "Q4_K": "Q5_0", "Q5_K": "Q5_1", "Q6_K": "Q8_0"}        # llama-quant.cpp tensor_type_fallback
NOT_QUANTISED = ("_norm.weight", "ffn_gate_inp.weight", "ffn_gate_tid2eid.weight", "engram_gate.weight", "altup", "laurel", "per_layer_model_proj", "ssm_conv1d", "shortconv.conv.weight")


class FitError(Exception):
    """"Fit to a size" cannot be done (no readable statistics, a budget nobody can meet): a plain-language text; the caller falls back to the presets."""


def preset(pid: str) -> dict:
    for p in PRESETS:
        if p["id"] == pid:
            return p
    raise KeyError(pid)


# ------------------------------------------------------------------------------------------------------------------ the rules

def role_names(arch) -> dict:
    """The tensor names (without `blk.N.` and `.weight`) of each role. A hybrid model (linear attention in most layers: qwen35, qwen35moe) has the `lin` role; for any other
    architecture only the standard `blk.N.` names and the generic roles are used (a fused attn_qkv counts as attention)."""
    hybrid = arch in HYBRID_ARCHS
    roles = {"attn_kvo": ["attn_k", "attn_v", "attn_output", r"nextn\.eh_proj"] + ([] if hybrid else ["attn_qkv"]), "attn_q": ["attn_q"], "ffn_down": ["ffn_down", "ffn_down_exps", "ffn_down_shexp"],
             "ffn_updown": ["ffn_up", "ffn_gate", "ffn_up_exps", "ffn_gate_exps", "ffn_up_shexp", "ffn_gate_shexp"]}
    if hybrid:
        roles["lin"] = ["attn_qkv_gate", "attn_qkv", "attn_gate", "ssm_out", "ssm_beta_alpha", "ssm_beta", "ssm_alpha"]
    return {r: roles[r] for r in ROLES if r in roles}


def edge_layers(n_layer: int, n: int) -> list:
    """The first and the last `n` layers (all layers when the model is small)."""
    return sorted(set(range(min(n, n_layer))) | set(range(max(0, n_layer - n), n_layer)))


def _alt(items) -> str:
    return "(?:" + "|".join(str(i) for i in items) + ")"


def _rx(names) -> str:
    return r"\.(?:" + "|".join(names) + r")\.weight$"


def build_rules(pid: str, arch, n_layer, names=None):
    """(rules, notes): the ordered `--tensor-type` rules of preset `pid` as [(regex, type)], first match wins. The first and last `edge` layers (and the layer of the speculative-decoding head,
    when the model has one) come first (higher types), then every layer. `n_layer` is the number of blocks (None = taken from `names`, the tensor names of the model). For an architecture this module does not know, only the standard
    names and the generic rules are used and a note says so. With `names`, a rule that matches no tensor of the model is reported in the notes (the names may differ)."""
    p = preset(pid)
    notes = []
    if arch not in HYBRID_ARCHS:
        notes.append(f"The architecture {arch!r} is not one this program knows the tensor names of; only the standard blk.N. names and the generic rules were used (attention k / v / output, ffn_down, ffn_up / gate, the first and last layers; "
                     "no linear-attention rule). Read the list of rules before you rely on it." if arch else "The architecture of the model could not be read; only the standard blk.N. names and the generic rules were used. Read the list of rules before you rely on it.")
    mtp = sorted({int(m.group(1)) for n in names or [] for m in [re.match(r"blk\.(\d+)\..*nextn", n)] if m})          # the layer(s) of the speculative-decoding head
    if n_layer is None and names:
        idx = [int(m.group(1)) for n in names for m in [re.match(r"blk\.(\d+)\.", n)] if m and int(m.group(1)) not in mtp]
        n_layer = max(idx) + 1 if idx else None
    if n_layer is None:
        notes.append("The number of layers could not be read, so the first and last layers get no extra bits.")
    roles = role_names(arch)
    rules = []
    edge = (edge_layers(n_layer, p["edge"]) + mtp) if n_layer else []         # the head is one small layer: it gets the edge layers' types
    if edge:
        for role in roles:
            hi, mid = p["types"][role]
            if hi != mid:
                rules.append((r"blk\." + _alt(edge) + _rx(roles[role]), hi))
    for role in roles:
        rules.append((_rx(roles[role]), p["types"][role][1]))
    if names:
        quant_names = [n for n in names if not any(x in n for x in NOT_QUANTISED)]
        for rx, _ in rules:
            if not any(re.search(rx, n) for n in quant_names):
                notes.append(f"A rule matched no tensor of this model ({rx}): the names may differ here. The tensors it was meant for follow the base type.")
    return rules, notes


def flags(pid: str, rules) -> list:
    """The command-line options of llama-quantize for a preset (before the file names): output and embedding types, then every rule as `--tensor-type REGEX=TYPE`."""
    p = preset(pid) if pid in PRESET_IDS else {"output": FIT_OUTPUT, "embd": FIT_EMBD}
    out = ["--output-tensor-type", p["output"], "--token-embedding-type", p["embd"]]
    for rx, t in rules:
        out += ["--tensor-type", f"{rx}={t}"]
    return out


# ------------------------------------------------------------------------------------------------------------------ sizes

def quantizable(name: str, dims) -> bool:
    """Does llama-quantize quantise this tensor at all (llama-quant.cpp `tensor_allows_quantization`, the generic part)?"""
    return len(dims) >= 2 and name.endswith("weight") and not any(x in name for x in NOT_QUANTISED)


def effective_type(t: str, ncols: int) -> str:
    """The type a tensor really gets when its row length does not fit the block size of `t` (the fallback table of llama-quant.cpp; F16 when even that does not fit)."""
    if ncols % BLOCK.get(t, 256) == 0:
        return t
    f = FALLBACK.get(t, t)
    return f if ncols % BLOCK.get(f, 32) == 0 else "F16"


def bpw(t: str) -> float:
    return ggufinfo.BPW[t]


def tensor_type(name: str, dims, src: str, rules, p: dict, tied: bool):
    """(type this tensor is written in, which rule decided: an index, "embd", "output", "base" or "kept")."""
    if not quantizable(name, dims):
        return src, "kept"
    if name == "token_embd.weight":
        return effective_type(p["output"] if tied else p["embd"], dims[0]), "embd"
    if name == "output.weight":
        return effective_type(p["output"], dims[0]), "output"
    for i, (rx, t) in enumerate(rules):
        if re.search(rx, name):
            return effective_type(t, dims[0]), i
    return effective_type(BASE_TYPE[p["base"]], dims[0]), "base"


def estimate(info: dict, rules, p: dict) -> dict:
    """The size of the file llama-quantize would make for `rules` and the preset-like dict `p` ({"base", "embd", "output"}), from the header `info` (ggufinfo.read) of a GGUF of the
    same model (any size: only the tensor names and shapes are read). {"bytes", "gb", "bpw", "params", "unmatched" (quantised tensors no rule named: they follow the base type, and
    llama.cpp's own mixture of that type adds some higher types, so this part is approximate), "dead_rules" (rules that matched nothing), "by_type": {type: params}}."""
    names = {t["name"] for t in info["tensors"]}
    tied = "output.weight" not in names
    total, params, unmatched, hits, by_type = info["data_start"], 0, 0, {}, {}
    for t in info["tensors"]:
        n = ggufinfo.numel(t)
        src = ggufinfo.type_name(t["type"])
        ty, why = tensor_type(t["name"], t["dims"], src, rules, p, tied)
        total += n * bpw(ty) / 8 if ty in ggufinfo.BPW else n * 16 / 8
        params += n
        by_type[ty] = by_type.get(ty, 0) + n
        if why == "base":
            unmatched += 1
        elif isinstance(why, int):
            hits[why] = hits.get(why, 0) + 1
    return {"bytes": int(total), "gb": round(total / 1e9, 2), "bpw": round(total * 8 / params, 3) if params else None, "params": params, "unmatched": unmatched,
            "dead_rules": [rules[i][0] for i in range(len(rules)) if i not in hits], "by_type": by_type}


# ------------------------------------------------------------------------------------------------------------------ the importance of each tensor, from a matrix

def read_importance(path) -> dict:
    """{weight tensor name: mean squared input activation} from a GGUF-format importance matrix: for every `<name>.in_sum2` (the sum of the squared inputs) and `<name>.counts`
    (how many rows were summed) the mean of in_sum2 / counts over all inputs and experts (float32). Two layouts are read: the one of a dense model, as in the real published
    matrix (verified: `in_sum2` has ONE dimension, [n_in], and `counts` one value, [1]), and the one of a mixture-of-experts model (`in_sum2` [n_in, n_experts] with one count
    per expert). Raises FitError (plain words) when the file cannot be read this way (a legacy .dat, a matrix without those tensors)."""
    try:
        info = ggufinfo.read(path)
    except (OSError, ValueError) as exc:
        raise FitError(f"The matrix {Path(path).name} could not be read as a GGUF matrix here ({exc}); a legacy .dat file cannot be used for fitting.")
    by = {t["name"]: t for t in info["tensors"]}
    out = {}
    try:
        with open(path, "rb") as fh:
            def floats(t):
                if t["type"] != 0:
                    raise FitError("The matrix holds its statistics in a form this program does not read (not float32).")
                fh.seek(info["data_start"] + t["offset"])
                n = ggufinfo.numel(t)
                a = array("f")
                a.frombytes(fh.read(4 * n))
                if len(a) != n:
                    raise FitError("The matrix file is cut short.")
                if sys.byteorder == "big":
                    a.byteswap()
                return a
            for name, t in by.items():
                if not name.endswith(".in_sum2"):
                    continue
                base = name[: -len(".in_sum2")]
                c = by.get(base + ".counts")
                if c is None:
                    continue
                sums, counts = floats(t), floats(c)
                nexp = max(len(counts), 1)
                per = len(sums) // nexp
                if per == 0:
                    continue
                total = 0.0
                for e in range(nexp):
                    if counts[e] > 0:
                        total += sum(sums[e * per:(e + 1) * per]) / counts[e]
                out[base] = total / (per * nexp)
    except OSError as exc:
        raise FitError(f"The matrix could not be read ({exc.strerror or type(exc).__name__}).")
    if not out:
        raise FitError("The matrix has no per-tensor statistics this program can read (tensors named like blk.N.attn_k.weight.in_sum2 with .counts), so a size cannot be fitted from it.")
    return out


# ------------------------------------------------------------------------------------------------------------------ fit to a size (experimental)

def _compress(assign: dict) -> list:
    """Per-tensor assignment {name: type} -> the shortest honest rule list: one rule per (tensor role, type) naming the layers, `blk\\.(?:3|7|12)\\.attn_k\\.weight$=Q5_K`; a tensor
    outside the blocks gets a rule of its own. Deterministic: sorted by role, then the highest type first."""
    groups: dict = {}
    for name, t in assign.items():
        m = re.match(r"blk\.(\d+)\.(.+)$", name)
        key = (m.group(2) if m else None, name if not m else None, t)
        groups.setdefault(key, []).append(int(m.group(1)) if m else name)
    rules = []
    for (suffix, whole, t), items in sorted(groups.items(), key=lambda kv: (kv[0][0] or "~" + (kv[0][1] or ""), -LEVELS.index(kv[0][2]))):
        if suffix is not None:
            rules.append((r"blk\." + _alt(sorted(items)) + r"\." + suffix.replace(".", r"\.") + "$", t))
        else:
            rules.append(("^" + whole.replace(".", r"\.") + "$", t))
    return rules


def fit(info: dict, target_bytes: int, importance: dict) -> dict:
    """"Fit to a size": every large tensor starts at the lowest type (`LEVELS[0]`); then the upgrade with the most importance per extra byte is made, one step at a time, until the
    next one would not fit the budget (`target_bytes`, the whole file). The embedding and output tensors are fixed (FIT_EMBD, FIT_OUTPUT: the matrix does not measure them) and
    tensors that are not quantised stay as they are.

    Importance: rounding a weight to `b` bits adds noise that falls by a factor of 4 for each bit (4^-b); the noise reaches the output weighted by the mean squared input of the
    tensor (the matrix: in_sum2 / counts) times its number of weights. Upgrading a tensor from `b0` to `b1` bits therefore removes `numel x E x (4^-b0 - 4^-b1)` of error for
    `numel x (b1 - b0) / 8` extra bytes: the number of weights cancels, and the ratio `E x (4^-b0 - 4^-b1) x 8 / (b1 - b0)` decides. This ignores how large the weights of each
    tensor are (not in the matrix) and is a **guess, not a measurement**: Compare quants is the test. A tensor the matrix has no figure for gets the median one.
    Pure and deterministic (ties broken by the tensor name). Returns {"rules", "flags_base", "assignment", "bytes", "gb", "bpw", "target_bytes", "fits", "min_bytes", "max_bytes",
    "upgrades", "missing", "notes"}; `fits` is false when even the lowest types need more than the budget (the lowest are returned), the budget is not used further than the first
    upgrade that does not fit (a bigger budget therefore never ends up with a lower type for any tensor)."""
    names = {t["name"] for t in info["tensors"]}
    tied = "output.weight" not in names
    fixed, free = info["data_start"], []
    params = 0
    for t in info["tensors"]:
        n = ggufinfo.numel(t)
        params += n
        name = t["name"]
        if not quantizable(name, t["dims"]):
            fixed += n * (bpw(ggufinfo.type_name(t["type"])) if ggufinfo.type_name(t["type"]) in ggufinfo.BPW else 16) / 8
        elif name in ("token_embd.weight", "output.weight"):
            ty = effective_type(FIT_OUTPUT if (name == "output.weight" or tied) else FIT_EMBD, t["dims"][0])
            fixed += n * bpw(ty) / 8
        else:
            free.append(t)
    if not free:
        raise FitError("This model has no tensors to fit.")
    known = sorted(importance[t["name"]] for t in free if t["name"] in importance)
    if not known:
        raise FitError("The matrix has no statistics for any tensor of this model, so a size cannot be fitted from it (is it the matrix of this model?).")
    median = known[len(known) // 2]
    missing = sum(1 for t in free if t["name"] not in importance)
    imp = {t["name"]: importance.get(t["name"], median) for t in free}
    level = {t["name"]: 0 for t in free}
    by_name = {t["name"]: t for t in free}

    def size(name, lv):
        t = by_name[name]
        return ggufinfo.numel(t) * bpw(effective_type(LEVELS[lv], t["dims"][0])) / 8

    total = fixed + sum(size(n, 0) for n in level)
    min_bytes = total
    max_bytes = fixed + sum(size(n, len(LEVELS) - 1) for n in level)

    def prio(name, lv):
        t = by_name[name]
        b0, b1 = bpw(effective_type(LEVELS[lv], t["dims"][0])), bpw(effective_type(LEVELS[lv + 1], t["dims"][0]))
        if b1 <= b0:
            return math.inf
        return imp[name] * (4.0 ** -b0 - 4.0 ** -b1) * 8 / (b1 - b0)

    notes, upgrades = [], 0
    fits = min_bytes <= target_bytes
    if fits:
        heap = [(-prio(n, 0), n, 0) for n in sorted(level)]
        heapq.heapify(heap)
        while heap:
            _, name, lv = heapq.heappop(heap)
            extra = size(name, lv + 1) - size(name, lv)
            if total + extra > target_bytes:
                break
            total += extra
            level[name] = lv + 1
            upgrades += 1
            if lv + 2 < len(LEVELS):
                heapq.heappush(heap, (-prio(name, lv + 1), name, lv + 1))
    else:
        notes.append(f"Even the lowest types ({LEVELS[0]}) need {min_bytes / 1e9:.2f} GB, more than the {target_bytes / 1e9:.2f} GB asked for; the lowest ones are given.")
    if fits and total >= max_bytes - 1:
        notes.append(f"The size asked for is more than the highest types ({LEVELS[-1]}) can use ({max_bytes / 1e9:.2f} GB): every tensor got the highest type.")
    if missing:
        notes.append(f"{missing} tensors had no figure in the matrix; they were given the median one.")
    assign = {n: LEVELS[lv] for n, lv in level.items()}
    return {"rules": _compress(assign), "assignment": assign, "bytes": int(total), "gb": round(total / 1e9, 2), "bpw": round(total * 8 / params, 3) if params else None,
            "target_bytes": int(target_bytes), "fits": fits, "min_bytes": int(min_bytes), "max_bytes": int(max_bytes), "upgrades": upgrades, "missing": missing, "notes": notes, "base": FIT_BASE}


# ------------------------------------------------------------------------------------------------------------------ what the export and the page use

def describe_model(path) -> dict:
    """Header facts of a GGUF of the model: {"info", "arch", "n_layer", "names"}. Raises ValueError / OSError (not a readable GGUF)."""
    info = ggufinfo.read(path)
    arch = ggufinfo.architecture(info)
    n = info["meta"].get(f"{arch}.block_count") if arch else None
    n = n if isinstance(n, int) and not isinstance(n, bool) else None
    return {"info": info, "arch": arch, "n_layer": n, "names": [t["name"] for t in info["tensors"]]}


def preset_plan(pid: str, model: dict) -> dict:
    p = preset(pid)
    rules, notes = build_rules(pid, model["arch"], model["n_layer"], model["names"])
    est = estimate(model["info"], rules, p)
    if est["unmatched"]:
        notes.append(f"{est['unmatched']} tensors are named by no rule and follow the base type {p['base']} (the size of those is approximate).")
    return {"id": pid, "label": p["label"], "recipe": LABEL, "base": p["base"], "embd": p["embd"], "output": p["output"], "summary": p["summary"], "rules": [f"{rx}={t}" for rx, t in rules],
            "flags": flags(pid, rules), "estimate": est, "notes": notes}


def for_export(quant: str, f16, fit_cfg=None, matrix=None) -> dict:
    """What llama-quantize needs for the mixed size `quant` made from the 16-bit file `f16`: {"id", "label", "base", "flags", "rules": ["regex=TYPE"], "estimate", "notes", "fit"}.
    `fit_cfg` = {"target_gb": ...} and `matrix` (a GGUF matrix file) for "mixed_fit": when no matrix can be read, or the budget cannot be met, the largest preset that fits
    (else the smallest) is used instead and a note says so. Raises ValueError / OSError when `f16` cannot be read."""
    model = describe_model(f16)
    if quant in PRESET_IDS:
        pl = preset_plan(quant, model)
        return {**pl, "fit": None}
    if quant != FIT_ID:
        raise KeyError(quant)
    target = float((fit_cfg or {}).get("target_gb") or 0) * 1e9
    notes = []
    try:
        if not matrix:
            raise FitError("No importance matrix is available for this export (the matrix step was not asked for, or it failed), so the size could not be fitted.")
        if target <= 0:
            raise FitError("No target size was given.")
        res = fit(model["info"], int(target), read_importance(matrix))
        rules = res["rules"]
        est = estimate(model["info"], rules, {"base": FIT_BASE, "embd": FIT_EMBD, "output": FIT_OUTPUT})
        return {"id": FIT_ID, "label": FIT_LABEL, "recipe": LABEL, "base": FIT_BASE, "embd": FIT_EMBD, "output": FIT_OUTPUT, "summary": "experimental: made to fit a size",
                "rules": [f"{rx}={t}" for rx, t in rules], "flags": flags(FIT_ID, rules), "estimate": est, "notes": res["notes"], "fit": {k: v for k, v in res.items() if k not in ("assignment", "rules")}}
    except FitError as exc:
        plans = [preset_plan(p["id"], model) for p in PRESETS]
        ok = [pl for pl in plans if target > 0 and pl["estimate"]["bytes"] <= target]
        pick = max(ok, key=lambda pl: pl["estimate"]["bytes"]) if ok else min(plans, key=lambda pl: pl["estimate"]["bytes"])
        notes.append(f"Fit to a size could not be done ({exc}) The preset \"{pick['label']}\" was used instead.")
        return {**pick, "notes": pick["notes"] + notes, "fit": None, "fell_back_to": pick["id"]}


def plan_job(job: dict) -> dict:
    """Everything the page shows before an export. Job {"job": "mixed_plan", "gguf": a GGUF of the model (any size), "target_gb": optional, "matrix": optional matrix file}."""
    try:
        model = describe_model(job.get("gguf") or "")
    except (OSError, ValueError) as exc:
        raise ValueError(f"The GGUF file could not be read ({exc}). Choose a complete GGUF file of this model (any size: only its tensor shapes are read).")
    presets_ = [preset_plan(p["id"], model) for p in PRESETS]
    out = {"arch": model["arch"], "n_layer": model["n_layer"], "params": sum(ggufinfo.numel(t) for t in model["info"]["tensors"]), "recipe": LABEL, "presets": presets_, "plain_bpw": PLAIN_BPW, "fit": None, "notes": []}
    if job.get("target_gb"):
        try:
            if not job.get("matrix"):
                raise FitError("Choose a saved importance matrix: the fit needs its per-tensor statistics.")
            res = fit(model["info"], int(float(job["target_gb"]) * 1e9), read_importance(job["matrix"]))
            est = estimate(model["info"], res["rules"], {"base": FIT_BASE, "embd": FIT_EMBD, "output": FIT_OUTPUT})
            out["fit"] = {**{k: v for k, v in res.items() if k not in ("assignment", "rules")}, "rules": [f"{rx}={t}" for rx, t in res["rules"]], "flags": flags(FIT_ID, res["rules"]), "estimate": est, "label": FIT_LABEL, "ok": True}
        except FitError as exc:
            out["fit"] = {"ok": False, "error": str(exc), "label": FIT_LABEL}
    return out
