"""Stop/resume round-trip test for the GGUF backend."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
pytest.importorskip("safetensors", reason="safetensors not installed")
pytest.importorskip("gguf", reason="gguf package not installed")

from safetensors.torch import save_file

from quant_gui import checkpoints as ckpt
from quant_gui.gguf_backend import convert_to_gguf
from quant_gui.run_control import RunCancelled, RunControl


def _write_flux_like_model(path: Path) -> None:
    sd = {
        # Architecture-detection key for "flux" (see gguf_backend.MODEL_ARCHES).
        "double_blocks.0.img_attn.proj.weight": torch.randn(256, 256),
        "double_blocks.0.img_attn.qkv.weight": torch.randn(256, 256),
        "double_blocks.0.img_norm.weight": torch.randn(256),  # 1D -> kept F32
        "double_blocks.1.img_attn.proj.weight": torch.randn(256, 256),
        "double_blocks.1.img_attn.qkv.weight": torch.randn(256, 256),
        "double_blocks.1.img_norm.weight": torch.randn(256),
    }
    save_file(sd, str(path))


def _convert(src, out, control=None, checkpoint=None, progress_cb=None):
    return convert_to_gguf(
        str(src), str(out), "Q8_0", control=control, checkpoint=checkpoint, progress_cb=progress_cb,
    )


def test_stop_saves_checkpoint_and_resume_completes_identically(tmp_path):
    src = tmp_path / "model.safetensors"
    _write_flux_like_model(src)
    out_resumed = tmp_path / "resumed.gguf"

    control = RunControl()
    checkpoint = ckpt.create_checkpoint(
        tmp_path / "checkpoints", "gguf",
        params={"input_path": str(src)}, output_path=str(out_resumed), total=0,
    )

    def cancel_after_first_tensor(current, total, key):
        if current == 2:  # cancel takes effect at the *start* of the next tensor
            control.cancel()

    with pytest.raises(RunCancelled):
        _convert(src, out_resumed, control=control, checkpoint=checkpoint, progress_cb=cancel_after_first_tensor)

    assert checkpoint.completed_count == 1
    assert checkpoint.total == 6
    assert checkpoint.shard_path(0, ".npy").is_file()
    assert not out_resumed.exists()

    loaded = ckpt.load_checkpoint(tmp_path / "checkpoints", checkpoint.id)
    resumed_stats = _convert(src, out_resumed, checkpoint=loaded)
    assert out_resumed.is_file()

    out_fresh = tmp_path / "fresh.gguf"
    fresh_stats = _convert(src, out_fresh)

    assert resumed_stats.quantized_count == fresh_stats.quantized_count
    assert resumed_stats.f32_kept_count == fresh_stats.f32_kept_count
    assert resumed_stats.total == fresh_stats.total == 6

    # Same tensors in, same writer settings -> byte-identical GGUF file.
    assert out_resumed.read_bytes() == out_fresh.read_bytes()
