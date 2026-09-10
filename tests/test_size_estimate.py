import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_gui.cli_builder import ConvertOptions
from quant_gui.size_estimate import estimate_int4_mixed, estimate_output_size, read_header


def write_fake_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    """Write a syntactically valid safetensors file with fabricated (zeroed)
    tensor data, just to exercise the header parser/estimator without
    needing torch or the real safetensors package."""
    dtype_bytes = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1}
    header = {}
    offset = 0
    blobs = []
    for name, (dtype, shape) in tensors.items():
        nelem = 1
        for d in shape:
            nelem *= d
        nbytes = nelem * dtype_bytes[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        blobs.append(b"\x00" * nbytes)
        offset += nbytes

    header_json = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for blob in blobs:
            f.write(blob)


def _make_model(path: Path):
    write_fake_safetensors(path, {
        "blocks.0.attn.wq.weight": ("BF16", [256, 256]),
        "blocks.0.attn.wk.weight": ("BF16", [256, 256]),
        "blocks.0.attn.norm.weight": ("BF16", [256]),  # 1D, never quantized
        "blocks.0.txtfusion.projector.weight": ("BF16", [256, 256]),  # krea2 highprec
        "blocks.0.mlp.gate.weight": ("BF16", [256, 1024]),
    })


def test_read_header_matches_written_tensors(tmp_path):
    model_path = tmp_path / "model.safetensors"
    _make_model(model_path)
    tensors = read_header(str(model_path))
    names = {t.name for t in tensors}
    assert names == {
        "blocks.0.attn.wq.weight", "blocks.0.attn.wk.weight", "blocks.0.attn.norm.weight",
        "blocks.0.txtfusion.projector.weight", "blocks.0.mlp.gate.weight",
    }
    wq = next(t for t in tensors if t.name == "blocks.0.attn.wq.weight")
    assert wq.dtype == "BF16"
    assert wq.shape == [256, 256]
    assert wq.nbytes == 256 * 256 * 2


def test_estimate_int4_mixed_matches_int4_backend_splits(tmp_path):
    model_path = tmp_path / "model.safetensors"
    _make_model(model_path)
    tensors = read_header(str(model_path))

    # attn.wq -> int4 (in_features 256 divides both 256 and 64); attn.wk and
    # mlp.gate -> int8 fallback (quantizable but not int4-matched); txtfusion
    # (krea2-excluded) and norm (1D) stay kept.
    est = estimate_int4_mixed(tensors, int4_regex=r"attn\.wq", preset="krea2", fallback_int8=True)

    assert est.quantized_count == 3  # 1 int4 (wq) + 2 int8 (wk, mlp.gate)
    assert est.kept_count == 2  # norm (1D) + txtfusion (krea2-excluded)
    assert est.estimated_bytes < est.original_bytes


def test_int8_roughly_halves_quantizable_tensors(tmp_path):
    model_path = tmp_path / "model.safetensors"
    _make_model(model_path)
    tensors = read_header(str(model_path))

    opts = ConvertOptions(input_path=str(model_path), quant_format="int8", convrot=False, preset="none")
    est = estimate_output_size(tensors, opts)

    # 3 quantizable 2D float weights (wq, wk, mlp.gate); norm (1D) and
    # txtfusion (no preset applied here) both quantizable too since no preset -> 4 quantized, 1 kept (norm).
    assert est.quantized_count == 4
    assert est.kept_count == 1
    assert est.estimated_bytes < est.original_bytes
    # int8 is ~1 byte/elem vs bf16's 2 bytes/elem -> output should be well under 60% of input.
    assert est.estimated_bytes < est.original_bytes * 0.6


def test_krea2_preset_keeps_txtfusion_layer_full_precision(tmp_path):
    model_path = tmp_path / "model.safetensors"
    _make_model(model_path)
    tensors = read_header(str(model_path))

    opts = ConvertOptions(input_path=str(model_path), quant_format="int8", convrot=False, preset="krea2")
    est = estimate_output_size(tensors, opts)

    # txtfusion.projector + norm (1D) both stay full precision; wq/wk/mlp.gate get quantized.
    assert est.quantized_count == 3
    assert est.kept_count == 2


def test_custom_layers_use_custom_type_for_size():
    tensors = [
        __import__("quant_gui.size_estimate", fromlist=["TensorHeader"]).TensorHeader(
            name="blocks.0.mlp.gate.weight", dtype="BF16", shape=[256, 1024], nbytes=256 * 1024 * 2
        )
    ]
    opts = ConvertOptions(
        input_path="m.safetensors", quant_format="fp8", convrot=False,
        custom_layers=r"mlp\.gate", custom_type="int8", custom_scaling_mode="row", custom_convrot=True,
    )
    est = estimate_output_size(tensors, opts)
    assert est.quantized_count == 1
    # int8 and fp8 have the same bytes/elem, so this mostly checks the custom path doesn't crash
    # and still counts as quantized rather than kept.
    assert est.estimated_bytes < est.original_bytes
