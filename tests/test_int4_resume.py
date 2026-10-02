"""Pause/stop/resume round-trip tests for the INT4 backend.

The real comfy_kitchen CUDA kernels are optional and heavy, so when the
package is absent we inject a minimal fake that implements the same
quantize()/state_dict_tensors() interface with deterministic math. That
keeps these tests exercising *this app's* checkpoint/resume logic (which is
what could regress), not comfy_kitchen itself.
"""

import importlib.machinery
import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
pytest.importorskip("safetensors", reason="safetensors not installed")


def _install_fake_comfy_kitchen() -> bool:
    """Inject a deterministic stand-in. Returns True if a fake was installed."""
    if importlib.util.find_spec("comfy_kitchen") is not None:
        return False

    class FakeInt4Layout:
        @staticmethod
        def quantize(tensor, convrot_groupsize=None, quant_group_size=None):
            q = (tensor.float() * 16).round().clamp(-127, 127).to(torch.int8)
            return q, {}

        @staticmethod
        def state_dict_tensors(qdata, params):
            return {"": qdata, ".scale": torch.ones(qdata.shape[0], dtype=torch.float32)}

    class FakeInt8Layout:
        @staticmethod
        def quantize(tensor, per_channel=False):
            q = (tensor.float() * 8).round().clamp(-127, 127).to(torch.int8)
            return q, {}

        @staticmethod
        def state_dict_tensors(qdata, params):
            return {"": qdata, ".scale": torch.ones(qdata.shape[0], dtype=torch.float32)}

    pkg = types.ModuleType("comfy_kitchen")
    tensor_mod = types.ModuleType("comfy_kitchen.tensor")
    convrot_mod = types.ModuleType("comfy_kitchen.tensor.convrot_w4a4")
    tensor_mod.TensorWiseINT8Layout = FakeInt8Layout
    convrot_mod.TensorCoreConvRotW4A4Layout = FakeInt4Layout
    pkg.tensor = tensor_mod
    for mod in (pkg, tensor_mod, convrot_mod):
        mod.__spec__ = importlib.machinery.ModuleSpec(mod.__name__, loader=None)
        sys.modules[mod.__name__] = mod
    return True


_install_fake_comfy_kitchen()

from safetensors.torch import load_file, save_file

from quant_gui import checkpoints as ckpt
from quant_gui.int4_backend import convert_int4_mixed
from quant_gui.run_control import RunCancelled, RunControl


def _write_model(path: Path) -> None:
    sd = {}
    for i in range(2):
        p = f"blocks.{i}."
        sd[p + "attn.wq.weight"] = torch.randn(512, 512, dtype=torch.bfloat16)
        sd[p + "attn.wk.weight"] = torch.randn(512, 512, dtype=torch.bfloat16)
        sd[p + "attn.norm.weight"] = torch.randn(512, dtype=torch.bfloat16)
    save_file(sd, str(path))


def _convert(src, out, control=None, checkpoint=None, progress_cb=None):
    return convert_int4_mixed(
        str(src), str(out), r"attn\.wq", preset="none",
        control=control, checkpoint=checkpoint, progress_cb=progress_cb,
    )


def test_stop_saves_checkpoint_and_resume_completes_identically(tmp_path):
    src = tmp_path / "model.safetensors"
    _write_model(src)
    out_resumed = tmp_path / "resumed.safetensors"

    control = RunControl()
    checkpoint = ckpt.create_checkpoint(
        tmp_path / "checkpoints", "int4",
        params={"input_path": str(src)}, output_path=str(out_resumed), total=0,
    )

    def cancel_after_two_tensors(current, total, key):
        if current == 3:  # cancel takes effect at the *start* of the next tensor
            control.cancel()

    with pytest.raises(RunCancelled):
        _convert(src, out_resumed, control=control, checkpoint=checkpoint, progress_cb=cancel_after_two_tensors)

    # Two tensors finished (blocks.0 attn.norm -> kept, blocks.0 attn.wk ->
    # INT8; safetensors stores keys alphabetically); the loop stopped before
    # blocks.0 attn.wq and no output file was written.
    assert checkpoint.completed_count == 2
    assert checkpoint.total == 6
    assert checkpoint.shard_path(0, ".safetensors").is_file()
    assert checkpoint.shard_path(1, ".safetensors").is_file()
    assert not out_resumed.exists()
    assert checkpoint.tensor_meta(0)["kind"] == "kept"
    assert checkpoint.tensor_meta(1)["kind"] == "int8"

    # Resume from the checkpoint (as the app's "Resume from checkpoint" does).
    loaded = ckpt.load_checkpoint(tmp_path / "checkpoints", checkpoint.id)
    resumed_stats = _convert(src, out_resumed, checkpoint=loaded)
    assert out_resumed.is_file()

    # A from-scratch conversion must produce byte-identical tensor content.
    out_fresh = tmp_path / "fresh.safetensors"
    fresh_stats = _convert(src, out_fresh)

    assert resumed_stats.int4_count == fresh_stats.int4_count == 2
    assert resumed_stats.int8_count == fresh_stats.int8_count == 2
    assert resumed_stats.kept_count == fresh_stats.kept_count == 2
    assert sorted(resumed_stats.int4_layer_names) == sorted(fresh_stats.int4_layer_names)

    fresh = load_file(str(out_fresh))
    resumed = load_file(str(out_resumed))
    assert fresh.keys() == resumed.keys()
    for key in fresh:
        assert torch.equal(fresh[key], resumed[key]), f"tensor mismatch after resume: {key}"


def test_pause_blocks_conversion_until_resumed(tmp_path):
    src = tmp_path / "model.safetensors"
    out = tmp_path / "out.safetensors"
    _write_model(src)

    control = RunControl()
    control.pause()
    finished = []

    def worker():
        _convert(src, out, control=control)
        finished.append(True)

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.5)
    assert not finished, "conversion ran while paused"
    control.resume()
    t.join(timeout=30)
    assert finished, "conversion did not finish after resume"
    assert out.is_file()
