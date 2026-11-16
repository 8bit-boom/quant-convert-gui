import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

gr = pytest.importorskip("gradio", reason="app import needs gradio")

import app
from quant_gui import llamacpp_backend as lcpp
from test_mixed_quant import _imatrix, qwen_like
from quant_gui import mixed_quant as mq


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(app, "IMATRIX_DOWNLOAD_DIR", tmp_path / "dl")
    monkeypatch.setattr(app, "newest_model_gguf", lambda *a, **k: None)
    monkeypatch.setattr(app, "newest_imatrix", lambda *a, **k: None)
    monkeypatch.setattr(lcpp, "is_quantize_built", lambda d: True)
    monkeypatch.setattr(lcpp, "is_imatrix_built", lambda d: True)
    cmds = []

    def fake_run(cmd, cwd=None, result=None):
        cmds.append(cmd)
        out = [c for c in cmd if c.endswith(".gguf")]
        # llama-quantize: <in> <out> <type>; llama-imatrix: -o <out>
        target = cmd[cmd.index("-o") + 1] if "-o" in cmd else cmd[-2]
        Path(target).write_bytes(b"fake")
        result.returncode = 0
        yield "fake tool ran\n"
    monkeypatch.setattr(lcpp, "_run_streamed", fake_run)
    monkeypatch.setattr(lcpp, "_quantize_binary", lambda d: Path("/x/llama-quantize"))
    monkeypatch.setattr(lcpp, "_imatrix_binary", lambda d: Path("/x/llama-imatrix"))
    return cmds


def test_preview_lists_every_recipe_and_is_honest(tmp_path, env):
    f = qwen_like(tmp_path / "m.gguf")
    md = app.run_llm_mixed_preview(str(f), "", 0)
    for p in mq.PRESETS:
        assert p["label"] in md
    assert "not Unsloth's UD quants" in md


def test_mixed_quantize_builds_the_expected_command(tmp_path, env):
    f = qwen_like(tmp_path / "m.gguf")
    im = _imatrix(tmp_path / "im.gguf", mq.describe_model(f), lambda n: 1.0)
    res = list(app.run_llm_mixed_quantize(str(f), "mixed_3", 0, str(im), ""))
    assert res[-1][1] and res[-1][1].endswith("m-mixed_3.gguf")
    cmd = env[0]
    assert cmd[1:3] == ["--imatrix", str(im)]
    assert cmd[cmd.index("--output-tensor-type") + 1] == "Q6_K"
    assert cmd[-1] == "Q3_K_M" and "--tensor-type" in cmd


def test_fit_without_matrix_falls_back_and_says_so(tmp_path, env):
    f = qwen_like(tmp_path / "m.gguf")
    res = list(app.run_llm_mixed_quantize(str(f), "mixed_fit", 0.5, "", ""))
    assert "Fit to a size could not be done" in res[0][0]
    assert res[-1][1]


def test_quantized_input_is_refused(tmp_path, env):
    from test_mixed_quant import write_gguf
    f = tmp_path / "q.gguf"
    write_gguf(f, {"general.architecture": "llama", "general.file_type": 15, "llama.block_count": 1},
               [("blk.0.attn_q.weight", [256, 256], 12, None)])
    res = list(app.run_llm_mixed_quantize(str(f), "mixed_4", 0, "", ""))
    assert "already quantized" in res[-1][0] and res[-1][1] is None and not env


def test_merge_two_matrices(tmp_path, env):
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    a.write_bytes(b"1"), b.write_bytes(b"2")
    res = list(app.run_llm_merge_imatrix(f"{a}\n{b}", "", False, ""))
    assert res[-1][1].endswith("imatrix-merged.gguf")
    assert env[0].count("--in-file") == 2
    assert "ADDS the statistics" in res[0][0]


def test_merge_needs_two_inputs(tmp_path, env):
    res = list(app.run_llm_merge_imatrix(str(tmp_path / "a.gguf"), "", False, ""))
    assert "at least two" in res[-1][0] and not env


def test_published_lookup_offline_is_plain(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")
    assert "could not be reached" in app.run_llm_published_lookup("x/y")
