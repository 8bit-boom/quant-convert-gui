import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quant_gui.filters import FALLBACK_MODEL_FILTERS, get_model_filters, suggest_preset
from quant_gui.gguf_backend import SUPPORTED_ARCH_NAMES
from quant_gui.model_families import (
    FAMILIES, FAMILY_BY_KEY, GENERIC_DIT_SENSITIVE_REGEX, detect_family, family_choices, family_note_markdown,
)


@pytest.mark.parametrize("name,expected", [
    ("wan2.2_t2v_high_noise_14B_fp16.safetensors", "wan"),
    ("Wan2.1-I2V-14B-480P.safetensors", "wan"),
    ("wan_2_1_vace_1.3B.safetensors", "wan"),
    ("hunyuanvideo1.5_8.3B_720p.safetensors", "hunyuan15"),
    ("hunyuan_video_1.5_t2v.safetensors", "hunyuan15"),
    ("hunyuan_video_720_cfgdistill_bf16.safetensors", "hunyuan1"),
    ("ltx-video-2b-v0.9.5.safetensors", "ltx1"),          # the 2B size must not be mistaken for LTX *2*
    ("ltx-2-19b-dev.safetensors", "ltx2"),
    ("ltx2_19B_fp8.safetensors", "ltx2"),
    ("MiniMax-H3-33B.safetensors", "minimaxh3"),
    ("ace_step_v1.5.safetensors", "acestep15"),
    ("minimax_music_3.safetensors", "minimaxmusic3"),
    ("yue2_3b.safetensors", "yue2"),
])
def test_detect_family(name, expected):
    assert detect_family(name).key == expected


@pytest.mark.parametrize("name", [
    "swan_lake_style.safetensors", "juwan_lora.safetensors", "flux1-dev.safetensors",
    "sdxl_base.safetensors", "revue_model.safetensors", "",
])
def test_no_false_positives(name):
    assert detect_family(name) is None


def test_every_ctq_preset_in_catalog_exists_in_ctq_registry():
    # Guards against catalog drift: a typo'd or renamed preset would silently never apply.
    live = get_model_filters()
    for f in FAMILIES:
        if f.ctq_preset:
            assert f.ctq_preset in live, f"{f.key} points at missing ctq preset {f.ctq_preset!r}"
            assert f.ctq_preset in FALLBACK_MODEL_FILTERS


def test_every_gguf_arch_in_catalog_is_one_the_gguf_backend_accepts():
    for f in FAMILIES:
        if f.gguf_arch:
            assert f.gguf_arch in SUPPORTED_ARCH_NAMES


def test_audio_families_have_no_ctq_preset_or_gguf_arch():
    for f in FAMILIES:
        if f.kind == "audio":
            assert f.ctq_preset is None and f.gguf_arch is None


def test_suggest_preset_for_video_families():
    assert suggest_preset("wan2.2_t2v_14B.safetensors") == "wan"
    assert suggest_preset("hunyuanvideo1.5_8.3B.safetensors") == "hunyuan"
    assert suggest_preset("ltx-2-19b.safetensors") == "ltxv2"
    assert suggest_preset("MiniMax-H3.safetensors") == "minimaxh3"
    # families whose ctq preset is for a *different* model get none
    assert suggest_preset("ltx-video-2b-v0.9.safetensors") is None
    assert suggest_preset("hunyuan_video_720.safetensors") is None
    assert suggest_preset("ace_step_1.5.safetensors") is None
    # existing behavior unchanged
    assert suggest_preset("kroma-v0.3-txtfusion.safetensors") == "krea2"


def test_family_choices_cover_all_families_video_first():
    choices = family_choices()
    assert {k for _, k in choices} == set(FAMILY_BY_KEY)
    kinds = [FAMILY_BY_KEY[k].kind for _, k in choices]
    assert kinds == sorted(kinds, key=lambda k: k != "video")


def test_notes_state_gguf_and_preset_status_honestly():
    audio = family_note_markdown(FAMILY_BY_KEY["acestep15"], preset_available=False)
    assert "none exists" in audio and "not available" in audio
    wan = family_note_markdown(FAMILY_BY_KEY["wan"], preset_available=True)
    assert "`wan`" in wan and "supported" in wan


def test_generic_regex_matches_typical_sensitive_layers_but_not_blocks():
    import re
    r = re.compile(GENERIC_DIT_SENSITIVE_REGEX)
    for name in ("timestep_embedder.linear_1.weight", "final_layer.linear.weight",
                 "transformer_blocks.0.scale_shift_table", "adaln_single.linear.weight", "proj_out.weight"):
        assert r.search(name), name
    for name in ("transformer_blocks.0.attn.to_q.weight", "blocks.3.ffn.0.weight"):
        assert not r.search(name), name
