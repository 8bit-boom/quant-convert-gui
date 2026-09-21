import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_gui import checkpoints as ckpt


def _make(root, backend="int4", **kwargs):
    return ckpt.create_checkpoint(
        root, backend,
        params=kwargs.pop("params", {"input_path": "/models/flux.safetensors"}),
        output_path=kwargs.pop("output_path", "/out/flux-int4.safetensors"),
        total=kwargs.pop("total", 0),
    )


def test_create_and_load_roundtrip(tmp_path):
    cp = _make(tmp_path, total=264)
    loaded = ckpt.load_checkpoint(tmp_path, cp.id)

    assert loaded.backend == "int4"
    assert loaded.total == 264
    assert loaded.params["input_path"] == "/models/flux.safetensors"
    assert loaded.output_path == "/out/flux-int4.safetensors"
    assert loaded.completed_count == 0
    assert loaded.next_index == 0
    assert not loaded.finished


def test_record_tensor_and_shard_paths(tmp_path):
    cp = _make(tmp_path, total=3)
    shard = cp.shard_path(0, ".safetensors")
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b"payload")

    ckpt.record_tensor(cp, 0, "blocks.0.attn.wq", kind="int4", quant={"format": "convrot_w4a4"})
    ckpt.record_tensor(cp, 1, "blocks.0.attn.wk", kind="int8", quant={"format": "int8_tensorwise"})

    assert cp.completed_count == 2
    assert cp.next_index == 2
    loaded = ckpt.load_checkpoint(tmp_path, cp.id)
    assert loaded.tensor_meta(0)["kind"] == "int4"
    assert loaded.tensor_meta(0)["quant"] == {"format": "convrot_w4a4"}
    assert loaded.tensor_meta(1)["key"] == "blocks.0.attn.wk"
    assert loaded.shard_path(0, ".safetensors").is_file()


def test_set_total_and_mark_finished_persist(tmp_path):
    cp = _make(tmp_path)
    ckpt.set_total(cp, 9)
    ckpt.mark_finished(cp)

    loaded = ckpt.load_checkpoint(tmp_path, cp.id)
    assert loaded.total == 9
    assert loaded.finished


def test_list_checkpoints_newest_first_and_skips_corrupt(tmp_path):
    older = _make(tmp_path)
    newer = _make(tmp_path)
    corrupt = tmp_path / "int4-20990101-00000-broken"
    corrupt.mkdir()
    (corrupt / "checkpoint.json").write_text("{not json")

    listed = ckpt.list_checkpoints(tmp_path)
    ids = [cp.id for cp in listed]
    assert len(ids) == 2
    assert ids[0] == newer.id  # same-second timestamps: order still deterministic enough
    assert set(ids) == {older.id, newer.id}


def test_delete_checkpoint_removes_dir(tmp_path):
    cp = _make(tmp_path)
    assert cp.path.is_dir()
    ckpt.delete_checkpoint(tmp_path, cp.id)
    assert not cp.path.exists()
    assert ckpt.list_checkpoints(tmp_path) == []


def test_summary_formats(tmp_path):
    cp = _make(tmp_path, total=10)
    ckpt.record_tensor(cp, 0, "a")
    ckpt.record_tensor(cp, 1, "b")
    text = ckpt.summary(cp)
    assert "int4" in text
    assert "flux.safetensors" in text
    assert "2/10 tensors" in text

    snap = _make(tmp_path, backend="ctq")
    assert "session snapshot" in ckpt.summary(snap)


def test_list_missing_root_returns_empty(tmp_path):
    assert ckpt.list_checkpoints(tmp_path / "nope") == []
