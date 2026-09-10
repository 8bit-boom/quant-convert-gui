"""Which quantization format actually runs fast on which NVIDIA GPU.

ctq's formats map to hardware generations, not personal taste:
- INT8 tensor cores: Turing (SM 7.5) onward - broadest support.
- FP8 tensor cores: Ada Lovelace (SM 8.9) / Hopper (SM 9.0) onward. Ampere
  (SM 8.0-8.6, e.g. every RTX 30-series card) has none.
- NVFP4 / MXFP8: Blackwell (SM >= 10.0) onward only.

ConvRot's Hadamard rotation is a software step on top of INT8 - it doesn't
need special hardware, so it's the one format that's fast everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPUProfile:
    key: str
    label: str
    recommended_format: str  # key into app.py's FORMAT_CHOICES
    note: str


GPU_PROFILES: list[GPUProfile] = [
    GPUProfile(
        key="not_sure",
        label="Not sure / other / CPU",
        recommended_format="int8_convrot",
        note="INT8 ConvRot runs on any CUDA GPU (and CPU, slowly) - the safest default when you don't know the hardware.",
    ),
    GPUProfile(
        key="ampere",
        label="RTX 30-series / A100 (Ampere) - e.g. RTX 3080 Ti",
        recommended_format="int8_convrot",
        note=(
            "Ampere has full native INT8 tensor-core support but **no FP8 or NVFP4 hardware path** "
            "(those need Ada/Hopper/Blackwell). INT8 ConvRot is both the fastest and the best-quality "
            "low-bit option on this card. A 12GB card comfortably fits most diffusion transformers "
            "quantized to INT8 - keep **Low memory mode** on during conversion."
        ),
    ),
    GPUProfile(
        key="ada",
        label="RTX 40-series / L40 / L4 (Ada Lovelace)",
        recommended_format="fp8",
        note="Ada adds hardware FP8 tensor cores. FP8 and INT8 ConvRot both run fast here; FP8 is usually a touch faster, INT8 ConvRot usually a touch more accurate.",
    ),
    GPUProfile(
        key="hopper",
        label="H100 / H200 (Hopper)",
        recommended_format="fp8",
        note="Same story as Ada: hardware FP8 tensor cores. FP8 and INT8 ConvRot both run fast here.",
    ),
    GPUProfile(
        key="blackwell",
        label="RTX 50-series / B100 / B200 (Blackwell)",
        recommended_format="nvfp4",
        note="Blackwell adds hardware NVFP4/MXFP8. NVFP4 is the newest 4-bit path; INT8 ConvRot remains a safe, broadly-compatible fallback.",
    ),
]

GPU_PROFILE_BY_KEY = {p.key: p for p in GPU_PROFILES}


def detect_profile_key(gpu_family: str) -> str:
    """Map an EnvReport.gpu_family value to a GPU_PROFILES key."""
    return gpu_family if gpu_family in GPU_PROFILE_BY_KEY else "not_sure"
