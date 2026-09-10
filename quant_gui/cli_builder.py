"""Translate GUI form state into a `ctq` (convert_to_quant) argument list.

Kept as a pure function of a plain dict -> list[str] so it's easy to unit
test without touching Gradio or spawning a process.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class OptionsError(ValueError):
    """Raised when the chosen combination of options can't be sent to ctq."""


@dataclass
class ConvertOptions:
    input_path: str
    output_path: str | None = None

    # "fp8" | "int8" | "nvfp4" | "mxfp8"
    quant_format: str = "int8"

    # INT8 / FP8 shared scaling knobs
    scaling_mode: str = "row"  # tensor | row | block
    block_size: int | None = None
    convrot: bool = True
    convrot_group_size: int = 256
    dynamic_convrot: bool = False

    simple: bool = True  # False => learned/AdaRound optimization

    comfy_quant: bool = True
    save_quant_metadata: bool = True
    low_memory: bool = True
    full_precision_matrix_mult: bool = False

    preset: str = "none"
    exclude_layers: str | None = None
    custom_layers: str | None = None
    custom_type: str | None = None
    device: str | None = None
    output_dtype: str = "bfloat16"
    verbose: str = "NORMAL"

    # Learned-rounding knobs (only used when simple=False)
    calib_samples: int = 3072
    optimizer: str = "prodigy"
    num_iter: int = 4000
    manual_seed: int = -1

    extra_flags: list[str] = field(default_factory=list)


CONVROT_VALID_GROUP_SIZES = (4, 16, 64, 256, 1024)


def validate(opts: ConvertOptions) -> None:
    if not opts.input_path or not opts.input_path.strip():
        raise OptionsError("Choose an input .safetensors file first.")

    if opts.quant_format not in ("fp8", "int8", "nvfp4", "mxfp8"):
        raise OptionsError(f"Unknown quantization format: {opts.quant_format!r}")

    if opts.convrot:
        if opts.quant_format != "int8":
            raise OptionsError("ConvRot is only available for INT8 quantization.")
        if opts.scaling_mode != "row":
            raise OptionsError("ConvRot requires row-wise scaling (--scaling_mode row).")
        if opts.convrot_group_size not in CONVROT_VALID_GROUP_SIZES:
            raise OptionsError(
                f"ConvRot group size must be a power of 4 ({CONVROT_VALID_GROUP_SIZES}); "
                f"got {opts.convrot_group_size}."
            )

    if opts.scaling_mode not in ("tensor", "row", "block", "block3d", "block2d"):
        raise OptionsError(f"Unknown scaling mode: {opts.scaling_mode!r}")


def build_args(opts: ConvertOptions) -> list[str]:
    """Build the argument list ctq expects (excluding the program name)."""
    validate(opts)

    args: list[str] = ["-i", opts.input_path]
    if opts.output_path:
        args += ["-o", opts.output_path]

    if opts.quant_format == "int8":
        args.append("--int8")
    elif opts.quant_format == "nvfp4":
        args.append("--nvfp4")
    elif opts.quant_format == "mxfp8":
        args.append("--mxfp8")
    # fp8 is ctq's default: no flag needed.

    if opts.quant_format in ("fp8", "int8"):
        args += ["--scaling_mode", opts.scaling_mode]
        if opts.scaling_mode in ("block", "block3d", "block2d") and opts.block_size:
            args += ["--block_size", str(opts.block_size)]

    if opts.convrot:
        args.append("--convrot")
        args += ["--convrot-group-size", str(opts.convrot_group_size)]
        if opts.dynamic_convrot:
            args.append("--dynamic-convrot")

    if opts.simple:
        args.append("--simple")

    if opts.comfy_quant:
        args.append("--comfy_quant")
    if opts.save_quant_metadata:
        args.append("--save-quant-metadata")
    if opts.low_memory:
        args.append("--low-memory")
    if opts.full_precision_matrix_mult:
        args.append("--full_precision_matrix_mult")

    if opts.preset and opts.preset != "none":
        args.append(f"--{opts.preset}")

    if opts.exclude_layers:
        args += ["--exclude-layers", opts.exclude_layers]
    if opts.custom_layers:
        args += ["--custom-layers", opts.custom_layers]
    if opts.custom_type:
        args += ["--custom-type", opts.custom_type]
    if opts.device:
        args += ["--device", opts.device]
    if opts.output_dtype and opts.output_dtype != "bfloat16":
        args += ["--output-dtype", opts.output_dtype]
    if opts.verbose and opts.verbose != "NORMAL":
        args += ["--verbose", opts.verbose]

    if not opts.simple:
        args += ["--calib_samples", str(opts.calib_samples)]
        args += ["--optimizer", opts.optimizer]
        args += ["--num_iter", str(opts.num_iter)]
        if opts.manual_seed is not None and opts.manual_seed != -1:
            args += ["--manual_seed", str(opts.manual_seed)]

    args += opts.extra_flags
    return args


def format_command(opts: ConvertOptions, program: str = "ctq") -> str:
    """Human-readable command line, for display/copy in the UI."""
    import shlex

    return " ".join([program, *[shlex.quote(a) for a in build_args(opts)]])
