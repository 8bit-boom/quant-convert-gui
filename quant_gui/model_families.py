"""Catalog of video and audio model families, so the Convert tab can detect
them from a filename/URL and say what is (and isn't) supported for each.

The family list, years, authors and scales come from SwarmUI's published
video/audio model support tables. Everything about *this app's* support is
checked against real code, not assumed:

- `ctq_preset` names a preset in convert_to_quant's own MODEL_FILTERS
  registry (tests assert each one really exists there).
- `gguf_arch` is the architecture string ComfyUI-GGUF's loader accepts for
  that family (tests assert it's in quant_gui.gguf_backend's allowlist).
  Families without one are *not* loadable through ComfyUI-GGUF at all, so
  GGUF export refuses them rather than writing a file nothing can read.
- Audio families have no ctq preset: ctq only knows a handful of audio-ish
  layer names inside its video presets (Wan's audio encoder, LTX-2's audio
  stack). Nothing here invents layer lists for models whose checkpoints
  couldn't be inspected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    key: str
    label: str
    kind: str  # "video" | "audio"
    year: int
    author: str
    params_b: tuple[float, ...]  # published parameter counts, in billions
    ctq_preset: str | None
    gguf_arch: str | None
    patterns: tuple[str, ...]
    notes: str


FAMILIES: list[Family] = [
    Family(
        key="wan", label="Wan 2.1 / 2.2", kind="video", year=2025, author="Alibaba - Wan-AI",
        params_b=(1.3, 5, 14), ctq_preset="wan", gguf_arch="wan",
        patterns=(r"(?<![a-z])wan[-_. ]?(?:v)?2",),
        notes="Text/Image-to-Video. If your checkpoint ships as separate high-noise and low-noise expert "
        "files (Wan 2.2's 14B does), convert each file on its own.",
    ),
    Family(
        key="hunyuan15", label="Hunyuan Video 1.5", kind="video", year=2025, author="Tencent",
        params_b=(8,), ctq_preset="hunyuan", gguf_arch=None,
        patterns=(r"(?:hunyuan|hyvid|hy[-_]?video).*?1[._-]?5",),
        notes="ctq's `hunyuan` preset is documented for exactly this model. GGUF export checks the tensor "
        "names against ComfyUI-GGUF's Hunyuan Video detection (written for the 2024 model); a 1.5 "
        "checkpoint that doesn't match is refused rather than mislabeled.",
    ),
    Family(
        key="hunyuan1", label="Hunyuan Video (2024)", kind="video", year=2024, author="Tencent",
        params_b=(12,), ctq_preset=None, gguf_arch="hyvid",
        patterns=(r"hunyuan[-_ ]?video", r"hyvid", r"hy[-_]?video"),
        notes="Legacy. ctq's `hunyuan` preset is documented for Hunyuan Video 1.5, not this one, so it "
        "isn't auto-selected - pick it yourself if you want its norm exclusions.",
    ),
    Family(
        key="ltx1", label="LTX Video (2024)", kind="video", year=2024, author="Lightricks",
        params_b=(3,), ctq_preset=None, gguf_arch="ltxv",
        patterns=(r"ltx[-_ ]?(?:video|v)",),
        notes="Legacy. ctq's `ltxv2` preset is for LTX Video 2, not this model, so none is auto-selected.",
    ),
    Family(
        key="ltx2", label="LTX Video 2", kind="video", year=2026, author="Lightricks",
        params_b=(19,), ctq_preset="ltxv2", gguf_arch=None,
        patterns=(r"ltx[-_ ]?(?:video|v)?[-_ ]?2(?![b\d])",),
        notes="Video + audio in one model: the `ltxv2` preset keeps its audio stack (audio_vae, audio "
        "adaln/connectors) and the vocoder at full precision.",
    ),
    Family(
        key="minimaxh3", label="MiniMax H3", kind="video", year=2026, author="MiniMax AI",
        params_b=(33, 20), ctq_preset="minimaxh3", gguf_arch=None,
        patterns=(r"minimax.*h3", r"h3.*minimax"),
        notes="Any-to-Video + audio. By far the largest family here, so check the size estimate against "
        "your VRAM before converting.",
    ),
    Family(
        key="acestep15", label="Ace Step 1.5", kind="audio", year=2026, author="StepFun",
        params_b=(2,), ctq_preset=None, gguf_arch=None,
        patterns=(r"ace[-_ ]?step",),
        notes="Music DiT.",
    ),
    Family(
        key="minimaxmusic3", label="MiniMax Music 3", kind="audio", year=2026, author="Hailuo & MiniMax",
        params_b=(2,), ctq_preset=None, gguf_arch=None,
        patterns=(r"minimax.*music", r"music.*minimax"),
        notes="Music DiT.",
    ),
    Family(
        key="yue2", label="YuE2", kind="audio", year=2026, author="Multimodal Art Projection",
        params_b=(3,), ctq_preset=None, gguf_arch=None,
        patterns=(r"(?<![a-z])yue(?![a-z])",),
        notes="Music DiT.",
    ),
]

FAMILY_BY_KEY = {f.key: f for f in FAMILIES}

# First match wins, so the more specific patterns come first.
_DETECTION_ORDER = [
    "minimaxh3", "minimaxmusic3", "acestep15", "yue2", "ltx2", "ltx1", "hunyuan15", "hunyuan1", "wan",
]

# Substrings that already appear in convert_to_quant's own video/image preset
# keep-high-precision lists (scale_shift_table, adaln_single, time_embedding,
# patch_embedding, final_layer, x_embedder, *_modulation, ...), combined into
# one regex. A generic starting point for DiT-style models with no preset of
# their own (e.g. the audio families) - not a verified per-model recipe.
GENERIC_DIT_SENSITIVE_REGEX = (
    r"(time_|timestep|t_embed|modul|adaln|scale_shift|final_layer|patch_embed|proj_out|proj_in|x_embed)"
)


def detect_family(name_hint: str) -> Family | None:
    lowered = (name_hint or "").lower()
    for key in _DETECTION_ORDER:
        fam = FAMILY_BY_KEY[key]
        if any(re.search(p, lowered) for p in fam.patterns):
            return fam
    return None


def family_choices() -> list[tuple[str, str]]:
    """(label, key) pairs for a dropdown, video first then audio."""
    out = []
    for kind in ("video", "audio"):
        for f in FAMILIES:
            if f.kind == kind:
                scale = " / ".join(f"{p:g}B" for p in f.params_b)
                out.append((f"{kind.capitalize()} - {f.label} ({f.year}, {scale})", f.key))
    return out


def size_hint(family: Family) -> str:
    parts = []
    for p in family.params_b:
        bf16 = p * 2
        parts.append(f"{p:g}B ≈ {bf16:.0f} GB in BF16 → ≈ {bf16 / 2:.0f} GB as INT8/FP8")
    return "; ".join(parts)


def family_note_markdown(family: Family, preset_available: bool) -> str:
    lines = [f"**{family.label}** — {family.kind}, {family.author}, {family.year}. {family.notes}"]
    lines.append(f"**Size:** {size_hint(family)} (weights only, rough).")

    if family.ctq_preset:
        if preset_available:
            lines.append(f"**ctq preset:** `{family.ctq_preset}` selected for you (Model preset section).")
        else:
            lines.append(
                f"**ctq preset:** `{family.ctq_preset}` exists upstream but isn't in your installed "
                "convert_to_quant - update it to use the preset."
            )
    else:
        lines.append(
            "**ctq preset:** none exists for this family. ctq still converts it (it only quantizes 2D "
            "linear weights, so conv/VAE/vocoder parts are left alone), but nothing keeps timestep/"
            "modulation/in-out layers at full precision for you - try **Advanced options → Exclude "
            "layers → \"Timestep, modulation & in/out projections\"** as a generic starting point, and "
            "check your checkpoint's actual layer names first."
        )

    if family.gguf_arch:
        lines.append(
            f"**GGUF:** supported (ComfyUI-GGUF architecture `{family.gguf_arch}`); 5D Conv3d tensors are "
            "stored as unquantized F32."
        )
    else:
        lines.append(
            "**GGUF:** not available - ComfyUI-GGUF's loader has no architecture for this family, so a "
            "GGUF file wouldn't load. Use INT8/FP8 `.safetensors` instead."
        )
    return "\n\n".join(lines)
