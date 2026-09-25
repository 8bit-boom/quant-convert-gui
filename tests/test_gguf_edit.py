"""Tests for quant_gui/gguf_edit.py - GGUF metadata/tensor editor."""
import json

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")
from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf import GGUFReader, GGUFWriter, quants  # noqa: E402

from quant_gui import gguf_edit as ge  # noqa: E402


def _mk_gguf(path) -> tuple:
    """Small GGUF with varied KV types + an F32 and a Q8_0 tensor."""
    rng = np.random.default_rng(1)
    w = GGUFWriter(None, arch="testarch")
    w.add_name("orig-name")
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    w.add_uint32("custom.uint", 7)
    w.add_int64("custom.neg", -5)
    w.add_float32("custom.float", 1.5)
    w.add_bool("custom.flag", True)
    w.add_string("custom.str", "hello")
    w.add_array("custom.arr.int", [1, 2, 3])
    w.add_array("custom.arr.str", ["a", "b"])
    w.add_array("custom.arr.big", list(range(100)))  # over ARRAY_EDIT_LIMIT
    f32 = rng.standard_normal((8, 64)).astype(np.float32)
    q8 = quants.quantize(rng.standard_normal((2, 32)).astype(np.float32), QT.Q8_0)
    w.add_tensor("w.f32", f32, raw_dtype=QT.F32)
    w.add_tensor("w.q8", q8, raw_dtype=QT.Q8_0)
    w.write_header_to_file(str(path))
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return f32, q8


@pytest.fixture()
def src(tmp_path):
    _mk_gguf(tmp_path / "src.gguf")
    return tmp_path / "src.gguf"


# --- load_for_edit ---

def test_load_for_edit(tmp_path, src):
    plan = ge.load_for_edit(src)
    assert plan.architecture == "testarch"
    assert plan.n_tensors == 2
    keys = {kv.key: kv for kv in plan.metadata}
    assert keys["general.name"].value == "orig-name"
    assert keys["custom.uint"].vtype == "UINT32"
    assert keys["custom.neg"].value == -5
    assert keys["custom.flag"].value is True
    assert keys["custom.arr.int"].sub_type == "INT32"
    assert list(keys["custom.arr.int"].value) == [1, 2, 3]
    # big arrays are display-only
    assert keys["custom.arr.big"].editable is False
    assert keys["custom.arr.int"].editable is True
    t = {t.name: t for t in plan.tensors}
    assert t["w.q8"].ggml_type == "Q8_0"
    assert t["w.f32"].n_bytes == 8 * 64 * 4
    assert plan.total_bytes == t["w.q8"].n_bytes + t["w.f32"].n_bytes


def test_load_for_edit_missing(tmp_path):
    with pytest.raises(ge.GGUFEditError):
        ge.load_for_edit(tmp_path / "nope.gguf")


# --- parse helpers ---

def test_parse_scalar_roundtrip():
    assert ge.parse_scalar("UINT32", "42") == 42
    assert ge.parse_scalar("INT64", "-9") == -9
    assert ge.parse_scalar("FLOAT32", "1.25") == 1.25
    assert ge.parse_scalar("BOOL", "true") is True
    assert ge.parse_scalar("BOOL", "0") is False
    assert ge.parse_scalar("STRING", "  keep spaces ") == "  keep spaces "


def test_parse_scalar_errors():
    for bad in [("UINT32", "abc"), ("UINT32", "3.5"), ("UINT8", "300"),
                ("INT8", "-999"), ("BOOL", "maybe"), ("FLOAT32", "x")]:
        with pytest.raises(ge.GGUFEditError):
            ge.parse_scalar(*bad)


def test_parse_array():
    assert ge.parse_array("INT32", [1, "2", 3.0]) == [1, 2, 3]
    assert ge.parse_array("STRING", [1, "b"]) == ["1", "b"]
    assert ge.parse_array("BOOL", [1, 0, True]) == [True, False, True]
    with pytest.raises(ge.GGUFEditError):
        ge.parse_array("INT32", "notalist")
    with pytest.raises(ge.GGUFEditError):
        ge.parse_array("NOPE", [1])


# --- save_edited ---

def _kv(key, vtype, value, sub=None):
    return ge.EditableKV(key, vtype, sub, value, True)


def test_save_meta_edits(tmp_path, src):
    out = tmp_path / "out.gguf"
    stats = ge.save_edited(
        src, out,
        set_meta={
            "general.name": _kv("general.name", "STRING", "new-name"),
            "custom.uint": _kv("custom.uint", "UINT32", 42),
            "custom.flag": _kv("custom.flag", "BOOL", False),
            "custom.arr.int": _kv("custom.arr.int", "ARRAY", [9, 8], sub="INT32"),
            "custom.added": _kv("custom.added", "STRING", "fresh"),
        },
        del_keys=["custom.float", "custom.neg"],
    )
    assert stats["n_kv"] == 9  # 10 original - 2 deleted + 1 added
    r = GGUFReader(str(out))
    assert r.fields["general.name"].contents() == "new-name"
    assert r.fields["custom.uint"].contents() == 42
    assert r.fields["custom.flag"].contents() == False  # noqa: E712
    assert list(r.fields["custom.arr.int"].contents()) == [9, 8]
    assert r.fields["custom.added"].contents() == "fresh"
    assert "custom.float" not in r.fields
    assert "custom.neg" not in r.fields
    # untouched keys survive
    assert r.fields["custom.str"].contents() == "hello"
    assert list(r.fields["custom.arr.str"].contents()) == ["a", "b"]


def test_save_tensors_byte_identical(tmp_path, src):
    f32, q8 = None, None
    out = tmp_path / "out.gguf"
    progress = []
    stats = ge.save_edited(src, out, renames={"w.q8": "w.q8b"},
                           drop_tensors=["w.f32"],
                           progress_cb=lambda d, t: progress.append((d, t)))
    assert stats["n_tensors"] == 1
    assert progress and progress[-1][0] == progress[-1][1]

    r = GGUFReader(str(out))
    names = [t.name for t in r.tensors]
    assert names == ["w.q8b"]
    t = list(r.tensors)[0]
    assert t.tensor_type == QT.Q8_0
    np.testing.assert_array_equal(np.asarray(t.data), q8_src(src))


def q8_src(path):
    src_r = GGUFReader(str(path))
    return np.asarray(next(t for t in src_r.tensors if t.name == "w.q8").data)


def test_save_untouched_tensors_identical(tmp_path, src):
    out = tmp_path / "out.gguf"
    ge.save_edited(src, out, set_meta={"general.name": _kv("general.name", "STRING", "x")})
    a, b = GGUFReader(str(src)), GGUFReader(str(out))
    for ta, tb in zip(a.tensors, b.tensors):
        assert ta.name == tb.name
        assert ta.tensor_type == tb.tensor_type
        np.testing.assert_array_equal(np.asarray(ta.data), np.asarray(tb.data))


def test_save_refuses_noop_and_same_path(tmp_path, src):
    with pytest.raises(ge.GGUFEditError):
        ge.save_edited(src, tmp_path / "o.gguf")
    with pytest.raises(ge.GGUFEditError):
        ge.save_edited(src, src, del_keys=["custom.str"])
    with pytest.raises(ge.GGUFEditError):
        ge.save_edited(src, tmp_path / "o.gguf", renames={"nope": "x"})
    with pytest.raises(ge.GGUFEditError):
        ge.save_edited(src, tmp_path / "o.gguf", drop_tensors=["nope"])
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_write_leaves_no_output(tmp_path, src):
    out = tmp_path / "out.gguf"
    with pytest.raises(Exception):
        ge.save_edited(src, out, set_meta={"k": _kv("k", "UINT32", 1)},
                       renames={"w.q8": "w.f32"})  # duplicate tensor name -> writer error
    assert not out.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_metadata_edit_only_copies_everything(tmp_path, src):
    """The common case: metadata-only edit must keep all tensors."""
    out = tmp_path / "out.gguf"
    ge.save_edited(src, out, set_meta={"general.name": _kv("general.name", "STRING", "n")})
    a, b = GGUFReader(str(src)), GGUFReader(str(out))
    assert len(list(a.tensors)) == len(list(b.tensors)) == 2


# --- display ---

def test_display_values():
    big = ge.EditableKV("k", "ARRAY", "UINT32", list(range(100)), editable=False)
    shown = big.display_value()
    assert "x 100" in shown and "..." in shown
    small = ge.EditableKV("k", "ARRAY", "UINT32", [1, 2], editable=True)
    assert json.loads(small.display_value()) == [1, 2]
    assert ge.EditableKV("k", "BOOL", None, True, True).display_value() == "True"
    s = ge.EditableKV("k", "STRING", None, "a,b", True)
    assert s.display_value() == "a,b"
