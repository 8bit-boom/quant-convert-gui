import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quant_gui.cli_builder import ConvertOptions, OptionsError, build_args


def test_matches_documented_kroma_int8_convrot_simple_command():
    opts = ConvertOptions(
        input_path="model.safetensors",
        output_path="model-int8-convrot-simple.safetensors",
        quant_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=256,
        simple=True,
        comfy_quant=True,
        save_quant_metadata=True,
        low_memory=False,
    )
    args = build_args(opts)
    for flag in ["--int8", "--scaling_mode", "row", "--convrot", "--convrot-group-size", "256", "--simple", "--comfy_quant", "--save-quant-metadata"]:
        assert flag in args


def test_convrot_forces_row_scaling_validation():
    opts = ConvertOptions(input_path="m.safetensors", quant_format="int8", scaling_mode="block", convrot=True)
    with pytest.raises(OptionsError):
        build_args(opts)


def test_convrot_rejected_for_non_int8():
    opts = ConvertOptions(input_path="m.safetensors", quant_format="fp8", scaling_mode="row", convrot=True)
    with pytest.raises(OptionsError):
        build_args(opts)


def test_invalid_convrot_group_size_rejected():
    opts = ConvertOptions(input_path="m.safetensors", quant_format="int8", scaling_mode="row", convrot=True, convrot_group_size=100)
    with pytest.raises(OptionsError):
        build_args(opts)


def test_missing_input_rejected():
    with pytest.raises(OptionsError):
        build_args(ConvertOptions(input_path=""))


def test_learned_mode_adds_optimizer_flags():
    opts = ConvertOptions(input_path="m.safetensors", quant_format="int8", scaling_mode="row", convrot=True, simple=False)
    args = build_args(opts)
    assert "--simple" not in args
    assert "--calib_samples" in args
    assert "--optimizer" in args


def test_nvfp4_uses_nvfp4_flag_no_scaling_mode():
    opts = ConvertOptions(input_path="m.safetensors", quant_format="nvfp4", convrot=False)
    args = build_args(opts)
    assert "--nvfp4" in args
    assert "--scaling_mode" not in args


def test_preset_flag_appended():
    opts = ConvertOptions(input_path="m.safetensors", quant_format="int8", scaling_mode="row", convrot=True, preset="krea2")
    args = build_args(opts)
    assert "--krea2" in args


def test_output_path_omitted_lets_ctq_auto_name():
    opts = ConvertOptions(input_path="m.safetensors", output_path=None, quant_format="fp8", convrot=False)
    args = build_args(opts)
    assert "-o" not in args
