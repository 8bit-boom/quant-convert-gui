"""Tests for tools/krea2_kquant_plan.py - sensitivity proxy + K-ladder plan."""
import sys
from pathlib import Path

import numpy as np
import pytest
from gguf import GGMLQuantizationType as QT
from gguf import GGUFWriter, quants

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import krea2_kquant_plan as plan  # noqa: E402


def _mk_gguf(path: Path) -> Path:
    """Tiny krea2-ish GGUF: one sensitive BF16 linear, one flat, hiprec F32."""
    rng = np.random.default_rng(5)
    writer = GGUFWriter(path=None, arch="krea2")
    writer.add_quantization_version(3)
    big = rng.standard_normal((256, 256)).astype(np.float32)
    big[0, :] *= 50.0  # outliers -> high proxy gap -> most sensitive
    writer.add_tensor("blocks.0.attn.wq.weight", quants.quantize(big, QT.BF16), raw_dtype=QT.BF16)
    flat = np.full((256, 256), 0.01, dtype=np.float32)
    writer.add_tensor("blocks.0.mlp.down.weight", quants.quantize(flat, QT.BF16), raw_dtype=QT.BF16)
    writer.add_tensor("first.weight", np.zeros((64, 6144), dtype=np.float32), raw_dtype=QT.F32)
    writer.add_tensor("blocks.0.prenorm.scale", np.ones(256, dtype=np.float32), raw_dtype=QT.F32)
    writer.write_header_to_file(path=str(path))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    return path


def test_score_tensors_ranks_outlier_tensor_first(tmp_path):
    p = _mk_gguf(tmp_path / "m.gguf")
    scored = plan.score_tensors(p)
    names = [n for n, _ in scored]
    assert names[0] == "blocks.0.attn.wq.weight"
    assert "blocks.0.mlp.down.weight" in names
    # hiprec + 1-D + small tensors excluded
    assert "first.weight" not in names
    assert "blocks.0.prenorm.scale" not in names
    gaps = dict(scored)
    assert gaps["blocks.0.attn.wq.weight"] > gaps["blocks.0.mlp.down.weight"]


def test_plan_assignment_respects_target_bpw(tmp_path):
    p = _mk_gguf(tmp_path / "m.gguf")
    scored = plan.score_tensors(p)
    hi = plan.plan_assignment(scored, 6.0)
    lo = plan.plan_assignment(scored, 3.0)
    # high target keeps the sensitive tensor at a high tier; low target drops it
    assert hi["blocks.0.attn.wq.weight"] in ("Q6_K", "Q5_K")
    assert lo["blocks.0.attn.wq.weight"] in ("Q4_K", "Q3_K")
    assert set(hi.values()) | set(lo.values()) <= {"Q6_K", "Q5_K", "Q4_K", "Q3_K"}


def test_render_escapes_names(tmp_path):
    text = plan.render({"blocks.0.attn.wq.weight": "Q4_K"})
    assert text.strip() == r"blocks\.0\.attn\.wq\.weight=Q4_K"


def test_empty_input_fails_main(tmp_path, capsys):
    assert plan.main([str(tmp_path / "nope.gguf")]) == 1


def test_main_end_to_end(tmp_path, capsys):
    p = _mk_gguf(tmp_path / "m.gguf")
    out = tmp_path / "plan.txt"
    assert plan.main([str(p), "--target-bpw", "4.0", "-o", str(out)]) == 0
    content = out.read_text()
    assert r"blocks\.0\.attn\.wq\.weight" in content
    assert "Q4_K" in content or "Q3_K" in content
