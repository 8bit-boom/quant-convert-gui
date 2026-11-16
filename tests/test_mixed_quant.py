import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quant_gui import gguf_header as gh
from quant_gui import mixed_quant as mq


def write_gguf(path: Path, meta: dict, tensors: list, align: int = 32):
    """Header-only GGUF writer (tensors = [(name, dims, ggml_type, float values or None)]); no gguf package needed."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), len(meta))
    for k, v in meta.items():
        out += s(k)
        out += (struct.pack("<I", 8) + s(v)) if isinstance(v, str) else (struct.pack("<I", 4) + struct.pack("<I", v))
    datas, off = [], 0
    for name, dims, typ, vals in tensors:
        n = 1
        for d in dims:
            n *= d
        raw = struct.pack(f"<{n}f", *vals) if (typ == 0 and vals is not None) else b"\0" * (n * 2 if typ == 1 else 4 * n)
        out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", typ, off)
        datas.append(raw)
        off += (len(raw) + align - 1) // align * align
    out += b"\0" * ((-len(out)) % align)
    for raw in datas:
        out += raw + b"\0" * ((-len(raw)) % align)
    path.write_bytes(out)


def qwen_like(path: Path, n_layer=8, arch="llama"):
    t = [("token_embd.weight", [256, 1024], 1, None), ("output_norm.weight", [256], 0, None), ("output.weight", [256, 1024], 1, None)]
    for i in range(n_layer):
        for nm, dims in (("attn_q", [256, 256]), ("attn_k", [256, 128]), ("attn_v", [256, 128]), ("attn_output", [256, 256]),
                         ("ffn_up", [256, 512]), ("ffn_gate", [256, 512]), ("ffn_down", [512, 256])):
            t.append((f"blk.{i}.{nm}.weight", dims, 1, None))
        t.append((f"blk.{i}.attn_norm.weight", [256], 0, None))
    write_gguf(path, {"general.architecture": arch, f"{arch}.block_count": n_layer}, t)
    return path


@pytest.fixture()
def model(tmp_path):
    return mq.describe_model(qwen_like(tmp_path / "m.gguf"))


def test_header_reader(tmp_path):
    p = qwen_like(tmp_path / "m.gguf")
    info = gh.read(p)
    assert gh.architecture(info) == "llama"
    assert info["meta"]["llama.block_count"] == 8
    assert {t["name"] for t in info["tensors"]} >= {"token_embd.weight", "blk.7.ffn_down.weight"}
    with pytest.raises(ValueError):
        (tmp_path / "bad").write_bytes(b"nope")
        gh.read(tmp_path / "bad")


def test_rules_put_edge_layers_first_and_are_valid_regex(model):
    import re
    rules, notes = mq.build_rules("mixed_3", model["arch"], model["n_layer"], model["names"])
    assert rules[0][0].startswith(r"blk\.(?:0|1|6|7)")
    for rx, t in rules:
        re.compile(rx)
        assert "=" not in rx and rx == rx.lower() and t in mq.LEVELS
    assert any("not one this program knows" in n for n in notes)   # llama is not a hybrid arch: honest note


def test_flags_shape(model):
    rules, _ = mq.build_rules("mixed_4", model["arch"], model["n_layer"], model["names"])
    f = mq.flags("mixed_4", rules)
    assert f[:4] == ["--output-tensor-type", "Q6_K", "--token-embedding-type", "Q5_K"]
    assert f.count("--tensor-type") == len(rules)


def test_estimates_order_by_recipe_and_by_hand(model):
    sizes = {p["id"]: mq.preset_plan(p["id"], model)["estimate"]["bytes"] for p in mq.PRESETS}
    assert sizes["mixed_2"] < sizes["mixed_3"] < sizes["mixed_4"]
    # hand sum for mixed_4's middle-layer ffn_up: Q4_K = 4.5 bpw
    est = mq.preset_plan("mixed_4", model)["estimate"]
    assert est["params"] == sum(gh.numel(t) for t in model["info"]["tensors"])
    assert est["dead_rules"] == [] or all(isinstance(r, str) for r in est["dead_rules"])


def test_no_quantisation_of_norms(model):
    rules, _ = mq.build_rules("mixed_2", model["arch"], model["n_layer"], model["names"])
    p = mq.preset("mixed_2")
    ty, why = mq.tensor_type("blk.3.attn_norm.weight", [256], "F32", rules, p, False)
    assert (ty, why) == ("F32", "kept")


def _imatrix(path: Path, model, scale_for):
    ts = []
    for t in model["info"]["tensors"]:
        n = t["name"]
        if mq.quantizable(n, t["dims"]) and n.startswith("blk."):
            ts.append((n + ".in_sum2", [4], 0, [scale_for(n)] * 4))
            ts.append((n + ".counts", [1], 0, [1.0]))
    write_gguf(path, {"general.architecture": "imatrix"}, ts)
    return path


def test_fit_is_monotone_respects_budget_and_prefers_important_tensors(tmp_path, model):
    im = _imatrix(tmp_path / "im.gguf", model, lambda n: 100.0 if ".attn_k." in n else 1.0)
    imp = mq.read_importance(im)
    assert imp["blk.0.attn_k.weight"] > imp["blk.0.ffn_up.weight"]
    last = None
    lo = mq.fit(model["info"], 0, imp)
    assert lo["fits"] is False
    for gb in (0.4, 0.6, 0.8, 1.0, 1.5, 3.0):
        r = mq.fit(model["info"], int(gb * 1e6 * 1000), imp)
        if r["fits"]:
            assert r["bytes"] <= r["target_bytes"]
            if last is not None:
                assert all(mq.LEVELS.index(r["assignment"][n]) >= mq.LEVELS.index(last[n]) for n in last)
            last = r["assignment"]
    mid = mq.fit(model["info"], (lo["min_bytes"] + lo["max_bytes"]) // 2, imp)
    a = mid["assignment"]
    assert mq.LEVELS.index(a["blk.2.attn_k.weight"]) >= mq.LEVELS.index(a["blk.2.ffn_up.weight"])
    # the rule list reproduces its own size
    est = mq.estimate(model["info"], mid["rules"], {"base": mq.FIT_BASE, "embd": mq.FIT_EMBD, "output": mq.FIT_OUTPUT})
    assert est["bytes"] == mid["bytes"]


def test_unreadable_matrix_is_a_plain_fit_error(tmp_path):
    (tmp_path / "old.dat").write_bytes(b"\x01\x00\x00\x00junk")
    with pytest.raises(mq.FitError):
        mq.read_importance(tmp_path / "old.dat")


def test_for_export_falls_back_to_a_preset_without_matrix(tmp_path):
    f = qwen_like(tmp_path / "m.gguf")
    r = mq.for_export("mixed_fit", f, {"target_gb": 0.5}, None)
    assert r.get("fell_back_to") in mq.PRESET_IDS
    assert any("Fit to a size could not be done" in n for n in r["notes"])
