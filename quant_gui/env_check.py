"""Detect whether the machine running the GUI can actually run ctq conversions."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass


@dataclass
class EnvReport:
    python_version: str
    ctq_executable: str | None
    ctq_importable: bool
    ctq_version: str | None
    torch_installed: bool
    torch_version: str | None
    cuda_available: bool
    gpu_name: str | None
    gpu_vram_gb: float | None
    cuda_capability: tuple[int, int] | None
    triton_installed: bool
    comfy_kitchen_installed: bool
    gguf_installed: bool
    huggingface_hub_installed: bool

    @property
    def is_blackwell(self) -> bool:
        return self.cuda_capability is not None and self.cuda_capability[0] >= 10

    @property
    def is_ada_or_newer(self) -> bool:
        """FP8 tensor cores exist on Ada (SM 8.9), Hopper (9.0), and Blackwell (10.0+),
        but not on Ampere (8.0-8.6, e.g. RTX 30-series) or older."""
        return self.cuda_capability is not None and self.cuda_capability >= (8, 9)

    @property
    def is_turing_or_newer(self) -> bool:
        """comfy_kitchen's TensorCoreConvRotW4A4Layout.MIN_SM_VERSION - Turing (7.5)
        and up, which is broader than FP8's Ada+ requirement (e.g. Ampere/RTX 30-series
        qualifies, even though it has no FP8 tensor cores)."""
        return self.cuda_capability is not None and self.cuda_capability >= (7, 5)

    @property
    def gpu_family(self) -> str:
        cc = self.cuda_capability
        if cc is None:
            return "unknown"
        if cc[0] >= 10:
            return "blackwell"
        if cc == (9, 0):
            return "hopper"
        if cc >= (8, 9):
            return "ada"
        if cc[0] == 8:
            return "ampere"
        return "pre-ampere"

    def ready_for(self, quant_format: str) -> tuple[bool, str]:
        if quant_format == "gguf":
            # Doesn't go through ctq at all - pure Python/numpy, no GPU needed.
            if not self.gguf_installed:
                return False, "gguf isn't installed. Install it with: pip install gguf"
            return True, "Ready."
        if quant_format == "int4_convrot":
            # Doesn't go through ctq at all - runs on comfy_kitchen directly.
            if not self.torch_installed:
                return False, "PyTorch isn't installed. The INT4 ConvRot backend (comfy_kitchen) needs it."
            if not self.comfy_kitchen_installed:
                return False, "comfy-kitchen isn't installed. Install it with: pip install comfy-kitchen"
            if self.cuda_available and not self.is_turing_or_newer:
                return False, (
                    "Real INT4 ConvRot tensor cores need Turing or newer (compute capability >= 7.5). "
                    f"Your GPU reports SM {self.cuda_capability[0]}.{self.cuda_capability[1]} — "
                    "conversion will still run on the CPU/eager path, just slowly."
                )
            return True, "Ready."
        if not (self.ctq_executable or self.ctq_importable):
            return False, "convert_to_quant (ctq) isn't installed. See the Environment tab for the install command."
        if not self.torch_installed:
            return False, "PyTorch isn't installed. ctq needs it even for CPU-only conversions."
        if quant_format in ("nvfp4", "mxfp8") and not self.is_blackwell:
            return False, (
                f"{quant_format.upper()} needs an NVIDIA Blackwell GPU (compute capability >= 10.0). "
                "Conversion will likely fail or produce a file your hardware can't run."
            )
        if quant_format == "fp8" and self.cuda_available and not self.is_ada_or_newer:
            return False, (
                "FP8 tensor cores need an Ada/Hopper/Blackwell GPU (compute capability >= 8.9). "
                f"Your GPU reports SM {self.cuda_capability[0]}.{self.cuda_capability[1]} (Ampere or older) — "
                "it has no FP8 hardware path. Use INT8 instead (Ampere has full native INT8 tensor-core support)."
            )
        return True, "Ready."


def check_environment() -> EnvReport:
    ctq_executable = shutil.which("ctq")
    ctq_importable = importlib.util.find_spec("convert_to_quant") is not None
    ctq_version = None
    if ctq_importable:
        try:
            from importlib import metadata as importlib_metadata

            ctq_version = importlib_metadata.version("convert-to-quant")
        except Exception:
            ctq_version = None

    torch_installed = importlib.util.find_spec("torch") is not None
    torch_version = None
    cuda_available = False
    gpu_name = None
    gpu_vram_gb = None
    cuda_capability = None

    if torch_installed:
        try:
            import torch

            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                gpu_vram_gb = round(props.total_memory / (1024**3), 1)
                cuda_capability = (props.major, props.minor)
        except Exception:
            pass

    triton_installed = importlib.util.find_spec("triton") is not None
    comfy_kitchen_installed = importlib.util.find_spec("comfy_kitchen") is not None
    gguf_installed = importlib.util.find_spec("gguf") is not None
    huggingface_hub_installed = importlib.util.find_spec("huggingface_hub") is not None

    return EnvReport(
        python_version=sys.version.split()[0],
        ctq_executable=ctq_executable,
        ctq_importable=ctq_importable,
        ctq_version=ctq_version,
        torch_installed=torch_installed,
        torch_version=torch_version,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        cuda_capability=cuda_capability,
        triton_installed=triton_installed,
        comfy_kitchen_installed=comfy_kitchen_installed,
        gguf_installed=gguf_installed,
        huggingface_hub_installed=huggingface_hub_installed,
    )


def report_markdown(report: EnvReport) -> str:
    def ok(flag: bool) -> str:
        return "✅" if flag else "❌"

    lines = [
        f"**Python** — {report.python_version}",
        f"**ctq (convert_to_quant)** — {ok(bool(report.ctq_executable or report.ctq_importable))} "
        + (
            f"`{report.ctq_executable}`"
            if report.ctq_executable
            else ("importable" if report.ctq_importable else "not found")
        )
        + (f" (v{report.ctq_version})" if report.ctq_version else ""),
        f"**PyTorch** — {ok(report.torch_installed)} " + (report.torch_version or "not installed"),
        f"**CUDA GPU** — {ok(report.cuda_available)} "
        + (
            f"{report.gpu_name} · {report.gpu_vram_gb} GB · SM {report.cuda_capability[0]}.{report.cuda_capability[1]}"
            if report.cuda_available and report.gpu_name
            else "none detected (CPU-only conversion will be very slow, or use --device cpu)"
        ),
        f"**GPU family** — {report.gpu_family}"
        + (" (no FP8/NVFP4/MXFP8 hardware — use INT8)" if report.gpu_family == "ampere" else ""),
        f"**Ada/Hopper/Blackwell (for FP8)** — {ok(report.is_ada_or_newer)}",
        f"**Blackwell (for NVFP4/MXFP8)** — {ok(report.is_blackwell)}",
        f"**Triton** (for INT8 kernels) — {ok(report.triton_installed)}",
        f"**comfy-kitchen** (for NVFP4/MXFP8, and for real INT4 ConvRot) — {ok(report.comfy_kitchen_installed)}",
        f"**Turing+ (for INT4 ConvRot tensor cores)** — {ok(report.is_turing_or_newer)}"
        + (" (Ampere and up all qualify, incl. RTX 30-series)" if not report.cuda_available else ""),
        f"**gguf** (for GGUF export) — {ok(report.gguf_installed)}",
        f"**huggingface_hub** (for downloading from a HF URL) — {ok(report.huggingface_hub_installed)}",
    ]
    return "\n\n".join(lines)
