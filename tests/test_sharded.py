import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from quant_gui import sharded
from quant_gui.sharded import (
    ShardError, collapse_listing, describe, find_shards, input_exists, iter_merge, logical_stem,
    merged_path_for, open_checkpoint,
)


def _tensors():
    rng = np.random.default_rng(0)
    return {
        "a.weight": rng.standard_normal((8, 16)).astype(np.float32),
        "b.weight": rng.standard_normal((4, 4)).astype(np.float16),
        "c.bias": np.arange(7, dtype=np.int8),
        "d.weight": rng.standard_normal((3, 5)).astype(np.float32),
        "e.empty": np.zeros((0, 4), dtype=np.float32),
    }


def make_shards(directory: Path, prefix="model", n=3, index=True, metadata=None):
    """Split _tensors() across n shard files (+ optional index); returns the dict."""
    tensors = _tensors()
    names = list(tensors)
    weight_map = {}
    for i in range(n):
        part = {k: tensors[k] for k in names[i::n]}
        fname = f"{prefix}-{i + 1:05d}-of-{n:05d}.safetensors"
        save_file(part, str(directory / fname), metadata=metadata)
        weight_map.update({k: fname for k in part})
    if index:
        (directory / f"{prefix}.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map})
        )
    return tensors


def test_logical_stem():
    assert logical_stem("/x/model-00001-of-00005.safetensors") == "model"
    assert logical_stem("diffusion_pytorch_model.safetensors.index.json") == "diffusion_pytorch_model"
    assert logical_stem("wan2.2_14B-00002-of-00006.safetensors") == "wan2.2_14B"
    assert logical_stem("plain.safetensors") == "plain"


def test_logical_stem_of_a_folder_is_the_model_not_the_folder(tmp_path):
    folder = tmp_path / "whatever_download_dir"
    folder.mkdir()
    make_shards(folder, prefix="wan_14B")
    assert logical_stem(folder) == "wan_14B"
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    assert logical_stem(empty) == "empty_dir"


@pytest.mark.parametrize("entry", ["shard", "index", "dir"])
def test_find_shards_resolves_every_way_of_pointing_at_a_set(tmp_path, entry):
    make_shards(tmp_path)
    path = {"shard": tmp_path / "model-00002-of-00003.safetensors",
            "index": tmp_path / "model.safetensors.index.json", "dir": tmp_path}[entry]
    s = find_shards(path)
    assert [p.name for p in s.files] == [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]
    assert s.logical_name == "model"
    assert s.index_path.name == "model.safetensors.index.json"
    assert s.entry_path == s.index_path
    assert "3 shards" in describe(path)


def test_set_without_an_index_file_still_resolves(tmp_path):
    make_shards(tmp_path, index=False)
    s = find_shards(tmp_path / "model-00001-of-00003.safetensors")
    assert s.index_path is None and len(s.files) == 3
    assert s.entry_path == s.files[0]


def test_plain_file_is_not_a_shard_set(tmp_path):
    save_file(_tensors(), str(tmp_path / "plain.safetensors"))
    assert find_shards(tmp_path / "plain.safetensors") is None
    assert find_shards(tmp_path) is None  # a folder with no shard set


def test_missing_shard_is_reported_by_name_not_silently_partial(tmp_path):
    make_shards(tmp_path, index=False)
    (tmp_path / "model-00002-of-00003.safetensors").unlink()
    with pytest.raises(ShardError, match=r"1 of 3 shard files are missing.*model-00002-of-00003"):
        find_shards(tmp_path / "model-00001-of-00003.safetensors")


def test_index_pointing_at_a_missing_file_is_reported(tmp_path):
    make_shards(tmp_path)
    (tmp_path / "model-00003-of-00003.safetensors").unlink()
    with pytest.raises(ShardError, match="model-00003-of-00003"):
        find_shards(tmp_path / "model.safetensors.index.json")


def test_folder_with_two_shard_sets_is_ambiguous(tmp_path):
    make_shards(tmp_path, prefix="unet", index=False)
    make_shards(tmp_path, prefix="vae", index=False)
    with pytest.raises(ShardError, match="2 different shard sets"):
        find_shards(tmp_path)
    # ...but pointing at one shard file is unambiguous
    assert find_shards(tmp_path / "vae-00001-of-00003.safetensors").logical_name == "vae"


def test_input_exists(tmp_path):
    assert not input_exists(tmp_path / "nope.safetensors")
    make_shards(tmp_path)
    assert input_exists(tmp_path)
    assert input_exists(tmp_path / "model.safetensors.index.json")


def test_sharded_open_presents_one_namespace(tmp_path):
    expected = make_shards(tmp_path, metadata={"format": "pt"})
    with open_checkpoint(tmp_path / "model.safetensors.index.json") as f:
        assert sorted(f.keys()) == sorted(expected)
        for k, v in expected.items():
            got = f.get_tensor(k)
            assert np.array_equal(got.numpy(), v) and tuple(got.shape) == v.shape
        assert f.metadata() == {"format": "pt"}
        assert f.keys() == f.keys()  # stable order - GGUF resume depends on it


def test_plain_file_still_uses_safe_open(tmp_path):
    save_file(_tensors(), str(tmp_path / "plain.safetensors"))
    with open_checkpoint(tmp_path / "plain.safetensors") as f:
        assert "a.weight" in f.keys()


def test_tensor_present_in_two_shards_is_rejected(tmp_path):
    t = {"x": np.ones((2, 2), dtype=np.float32)}
    save_file(t, str(tmp_path / "m-00001-of-00002.safetensors"))
    save_file(t, str(tmp_path / "m-00002-of-00002.safetensors"))
    with pytest.raises(ShardError, match="appears in both"):
        with open_checkpoint(tmp_path / "m-00001-of-00002.safetensors"):
            pass


def test_index_naming_a_tensor_no_shard_has_is_rejected(tmp_path):
    make_shards(tmp_path)
    idx = tmp_path / "model.safetensors.index.json"
    data = json.loads(idx.read_text())
    data["weight_map"]["ghost.weight"] = "model-00001-of-00003.safetensors"
    idx.write_text(json.dumps(data))
    with pytest.raises(ShardError, match="ghost.weight"):
        with open_checkpoint(idx):
            pass


def _merge(shards_dir: Path, out: Path):
    s = find_shards(shards_dir)
    return list(iter_merge(s, out))


def test_merge_is_byte_faithful_for_every_dtype(tmp_path):
    expected = make_shards(tmp_path, metadata={"format": "pt"})
    out = tmp_path / "merged" / "m.safetensors"
    progress = _merge(tmp_path, out)

    assert progress[-1][0] == progress[-1][1] == out.stat().st_size
    n = struct.unpack("<Q", out.read_bytes()[:8])[0]
    assert (8 + n) % 8 == 0, "tensor data must start 8-byte aligned"
    with safe_open(str(out), framework="np") as f:
        assert sorted(f.keys()) == sorted(expected)
        for k, v in expected.items():
            got = f.get_tensor(k)
            assert got.dtype == v.dtype and np.array_equal(got, v), k
        assert f.metadata() == {"format": "pt"}
    assert not list(out.parent.glob("*.partial"))


def test_merge_preserves_bf16(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file as save_pt

    t = {"w": torch.randn(16, 16, dtype=torch.bfloat16), "v": torch.randn(4, dtype=torch.bfloat16)}
    save_pt({"w": t["w"]}, str(tmp_path / "m-00001-of-00002.safetensors"))
    save_pt({"v": t["v"]}, str(tmp_path / "m-00002-of-00002.safetensors"))
    out = tmp_path / "out.safetensors"
    list(iter_merge(find_shards(tmp_path / "m-00001-of-00002.safetensors"), out))
    with safe_open(str(out), framework="pt") as f:
        for k, v in t.items():
            g = f.get_tensor(k)
            assert g.dtype == torch.bfloat16 and torch.equal(g, v)


def test_merge_is_reused_when_nothing_changed(tmp_path):
    make_shards(tmp_path)
    s = find_shards(tmp_path)
    out = merged_path_for(s, tmp_path / "merged")
    list(iter_merge(s, out))
    mtime = out.stat().st_mtime_ns
    events = list(iter_merge(s, out))
    assert "reusing" in events[-1][2] and out.stat().st_mtime_ns == mtime


def test_merged_cache_name_changes_when_a_shard_changes(tmp_path):
    make_shards(tmp_path)
    before = merged_path_for(find_shards(tmp_path), tmp_path / "m")
    f = tmp_path / "model-00001-of-00003.safetensors"
    f.write_bytes(f.read_bytes() + b"\0")
    assert merged_path_for(find_shards(tmp_path), tmp_path / "m") != before


def test_merge_checks_disk_space_up_front(tmp_path, monkeypatch):
    make_shards(tmp_path)
    monkeypatch.setattr(sharded.shutil, "disk_usage", lambda p: type("U", (), {"free": 1024})())
    with pytest.raises(ShardError, match="free in"):
        list(iter_merge(find_shards(tmp_path), tmp_path / "out" / "m.safetensors"))
    assert not (tmp_path / "out" / "m.safetensors").exists()


def test_truncated_shard_leaves_no_half_written_output(tmp_path):
    make_shards(tmp_path)
    f = tmp_path / "model-00002-of-00003.safetensors"
    f.write_bytes(f.read_bytes()[:-20])  # simulate an interrupted download
    out = tmp_path / "merged" / "m.safetensors"
    with pytest.raises(ShardError, match="ended early"):
        list(iter_merge(find_shards(tmp_path), out))
    assert not out.exists() and not list(out.parent.glob("*.partial"))


def test_collapse_listing_one_entry_per_set(tmp_path):
    make_shards(tmp_path, prefix="big", n=3)
    make_shards(tmp_path, prefix="half", n=3, index=False)
    (tmp_path / "half-00002-of-00003.safetensors").unlink()
    save_file(_tensors(), str(tmp_path / "plain.safetensors"))
    files = sorted(tmp_path.glob("*.safetensors"))

    got = {p.name: (n, complete) for p, _, n, complete in collapse_listing(files)}
    assert got["big.safetensors.index.json"] == (3, True)       # index is the entry
    assert got["half-00001-of-00003.safetensors"] == (3, False)  # incomplete is flagged
    assert got["plain.safetensors"] == (1, True)
    assert len(got) == 3
