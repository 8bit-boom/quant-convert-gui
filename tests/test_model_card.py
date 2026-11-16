import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quant_gui import model_card as mc


def _info(tmp_path, **kw):
    out = tmp_path / "Kroma-Q8_0.gguf"
    out.write_bytes(b"\0" * 2_000_000)
    return mc.CardInfo(kind=kw.pop("kind", "gguf_image"), output_path=str(out), source="/in/Kroma.safetensors", **kw), out


def test_card_is_written_beside_the_output_without_touching_other_cards(tmp_path):
    info, out = _info(tmp_path, quant_label="Q8_0")
    other = tmp_path / "Other-Q4_0.README.md"
    other.write_text("keep")
    path = mc.write_card(info)
    assert path == tmp_path / "Kroma-Q8_0.README.md" and path.is_file() and other.read_text() == "keep"
    text = path.read_text()
    assert "| `Kroma-Q8_0.gguf` | 2.0 MB |" in text
    assert "Quantized from `Kroma.safetensors`" in text


def test_base_model_only_when_it_looks_like_a_repo_id(tmp_path):
    good, _ = _info(tmp_path, base_model="krea/Krea-2-Turbo")
    t = mc.render(good)
    assert "base_model: krea/Krea-2-Turbo" in t and "base_model_relation: quantized" in t
    for bad in ("", "not a repo", "../x/y z", "a/b/c"):
        info, _ = _info(tmp_path, base_model=bad)
        assert "base_model:" not in mc.render(info)


def test_no_invented_claims(tmp_path):
    info, _ = _info(tmp_path, kind="gguf_llm", quant_label="Q4_K_M")
    t = mc.render(info)
    assert "license:" not in t.split("---")[1]            # front matter never guesses a license
    assert "No quality evaluation" in t
    assert "pipeline_tag" not in t                         # only when the caller knows it
    assert "llama-cli -m Kroma-Q8_0.gguf" in t


def test_settings_table_skips_blanks_and_escapes_pipes(tmp_path):
    info, _ = _info(tmp_path, quant_label="X", settings=[("a", ""), ("b", "x|y"), ("c", 0)])
    t = mc.render(info)
    assert "| a |" not in t and "x\\|y" in t and "| c | 0 |" in t


@pytest.mark.parametrize("kind", mc.KINDS)
def test_every_kind_renders_with_usage(tmp_path, kind):
    info, _ = _info(tmp_path, kind=kind, command="cmd --x", summary=["12 layers"])
    t = mc.render(info)
    assert "## Usage" in t and "```\ncmd --x\n```" in t and "- 12 layers" in t


def test_unknown_kind_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        mc.render(mc.CardInfo(kind="nope", output_path="x.gguf"))
