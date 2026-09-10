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
    huggingface_hub_installed: bool

    @property
    def is_blackwell(self) -> bool:
        return self.cuda_capability is not None and self.cuda_capability[0] >= 10

    def ready_for(self, quant_format: str) -> tuple[bool, str]:
        if not (self.ctq_executable or self.ctq_importable):
            return False, "convert_to_quant (ctq) isn't installed. See the Environment tab for the install command."
        if not self.torch_installed:
            return False, "PyTorch isn't installed. ctq needs it even for CPU-only conversions."
        if quant_format in ("nvfp4", "mxfp8") and not self.is_blackwell:
            return False, (
                f"{quant_format.upper()} needs an NVIDIA Blackwell GPU (compute capability >= 10.0). "
                "Conversion will likely fail or produce a file your hardware can't run."
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
        f"**Blackwell (for NVFP4/MXFP8)** — {ok(report.is_blackwell)}",
        f"**Triton** (for INT8 kernels) — {ok(report.triton_installed)}",
        f"**comfy-kitchen** (for NVFP4/MXFP8) — {ok(report.comfy_kitchen_installed)}",
        f"**huggingface_hub** (for downloading from a HF URL) — {ok(report.huggingface_hub_installed)}",
    ]
    return "\n\n".join(lines)
