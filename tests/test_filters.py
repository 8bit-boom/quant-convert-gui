import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_gui.filters import preset_highprec_keywords, preset_highprec_regex


def test_krea2_highprec_keywords_include_txtfusion():
    keywords = preset_highprec_keywords("krea2")
    assert "txtfusion" in keywords
    assert "last.modulatio" in keywords


def test_krea2_highprec_regex_matches_expected_layers():
    pattern = preset_highprec_regex("krea2")
    assert pattern is not None
    regex = re.compile(pattern)

    assert regex.search("txtfusion.projector.weight")
    assert regex.search("last.modulatio.weight")
    assert regex.search("blocks.0.tpro.weight")
    assert not regex.search("blocks.0.attn.wq.weight")


def test_krea2_highprec_regex_escapes_dots():
    # "last.modulatio" contains a literal dot - unescaped, "." in regex
    # matches any character. Confirm the built pattern escapes it instead
    # of leaving it as a wildcard.
    pattern = preset_highprec_regex("krea2")
    assert r"last\.modulatio" in pattern


def test_unknown_preset_has_no_highprec_regex():
    assert preset_highprec_regex("none") is None
    assert preset_highprec_regex("not_a_real_preset") is None
