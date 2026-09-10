"""Model-specific layer-exclusion presets exposed by convert_to_quant (ctq).

We import the live registry from the installed ``convert_to_quant`` package
when it's available so the dropdown always matches whatever version of ctq
the user has installed. If ctq isn't installed yet, we fall back to a
snapshot (from ctq v1.3.4) so the GUI still renders and explains itself.
"""

from __future__ import annotations

import re

# Exact highprec keyword list for krea2, confirmed against
# convert_to_quant.constants.MODEL_FILTERS["krea2"]["highprec"] - kept here
# too so the size-profile buttons still work if ctq isn't installed yet.
_KREA2_HIGHPREC_FALLBACK = ["firs", "las", "tml", "txtfusion", "last.modulatio", "tpro"]

FALLBACK_MODEL_FILTERS: dict[str, dict[str, object]] = {
    "gemma4": {"help": "Gemma4 text/multimodal model: skip audio, per_layer_input_gate, per_layer_projection, vision, multi_modal_projector", "category": "text"},
    "qwen_vlm": {"help": "Qwen VLM family: skip first/last language layers, embeddings, MTP, and the full visual encoder", "category": "text"},
    "qwen35": {"help": "Compatibility alias for --qwen_vlm", "category": "text"},
    "t5xxl": {"help": "T5-XXL text encoder: skip norms/biases, remove decoder layers", "category": "text"},
    "mistral": {"help": "Mistral text encoder exclusions", "category": "text"},
    "visual": {"help": "Visual encoder: skip MLP layers (down/up/gate proj)", "category": "text"},
    "generic_text": {"help": "Generic text encoder: skip MLP layers (down/up/gate proj)", "category": "text"},
    "anima": {"help": "Anima diffusion model: keep first blocks, adaln_modulation, final/embedding layers high-precision", "category": "diffusion"},
    "lens": {"help": "LENS diffusion model: keep time_text_embed, img_in, norm_out, proj_out, some mod layers high-precision", "category": "diffusion"},
    "flux2": {"help": "Flux.2: keep modulation/guidance/time/final layers high-precision", "category": "diffusion"},
    "distillation_large": {"help": "Chroma/distilled (large): keep distilled_guidance, final, img/txt_in high-precision", "category": "diffusion"},
    "distillation_small": {"help": "Chroma/distilled (small): keep only distilled_guidance high-precision", "category": "diffusion"},
    "nerf_large": {"help": "NeRF (large): keep nerf_blocks, distilled_guidance, txt_in high-precision", "category": "diffusion"},
    "nerf_small": {"help": "NeRF (small): keep nerf_blocks, distilled_guidance high-precision", "category": "diffusion"},
    "radiance": {"help": "Radiance model: keep img_in_patch, nerf_final_layer high-precision", "category": "diffusion"},
    "krea2": {
        "help": "Krea2 / txtfusion-style models: keep firs, las, tml, txtfusion, last.modulation, tpro layers high-precision",
        "category": "diffusion",
        "highprec": _KREA2_HIGHPREC_FALLBACK,
    },
    "ideogram4": {"help": "Ideogram4: keep embed_image_indicator, t_embedding, adaln_proj, final_layer, input_proj layers high-precision", "category": "diffusion"},
    "wan": {"help": "WAN video model: skip embeddings, encoders, head", "category": "video"},
    "hunyuan": {"help": "Hunyuan Video 1.5: skip layernorm, attn norms, vision_in", "category": "video"},
    "minimaxh3": {"help": "MiniMax H3: keep patch/condition/final/time and token-refiner layers high-precision", "category": "video"},
    "qwen": {"help": "Qwen Image: skip added norms, keep time_text_embed high-precision", "category": "image"},
    "zimage": {"help": "Z-Image: skip cap_embedder/norms, keep x_embedder/final high-precision", "category": "image"},
    "zimage_refiner": {"help": "Z-Image Refiner: keep context/noise refiner high-precision", "category": "image"},
    "boogu": {"help": "Boogu: keep image_index_embedding, ref_image_patch_embedder, time_caption_embed, x_embedder high-precision", "category": "image"},
    "ltxv2": {"help": "LTXv2: keep some transformer blocks high-precision and exclude vae and vocoder", "category": "video"},
}

# Filenames/repo names containing any of these substrings get "krea2" suggested,
# since that preset is the one whose exclusion list literally mentions
# "txtfusion" (the editions this GUI was built to convert use that filter).
_KREA2_HINTS = ("txtfusion", "krea", "kroma")


def get_model_filters() -> dict[str, dict[str, str]]:
    try:
        from convert_to_quant.constants import MODEL_FILTERS  # type: ignore

        return dict(MODEL_FILTERS)
    except Exception:
        return dict(FALLBACK_MODEL_FILTERS)


def preset_choices() -> list[str]:
    filters = get_model_filters()
    return ["none"] + sorted(filters.keys())


def preset_label(name: str) -> str:
    if name == "none":
        return "none — no model-specific exclusions"
    filters = get_model_filters()
    info = filters.get(name, {})
    help_text = info.get("help", "")
    category = info.get("category", "")
    tag = f"[{category}] " if category else ""
    return f"{name} — {tag}{help_text}" if help_text else name


def suggest_preset(name_hint: str) -> str | None:
    """Best-effort preset suggestion based on a filename or repo id."""
    lowered = (name_hint or "").lower()
    if any(hint in lowered for hint in _KREA2_HINTS):
        return "krea2"
    return None


def preset_highprec_keywords(preset: str) -> list[str]:
    """The substrings a preset keeps at full precision (its --custom-layers
    equivalent target), if ctq's registry exposes one for this preset."""
    info = get_model_filters().get(preset, {})
    return list(info.get("highprec", []) or [])


def preset_highprec_regex(preset: str) -> str | None:
    """A regex matching the same layers `preset` would otherwise keep at
    full precision - lets --custom-layers deliberately re-target them at a
    *lighter* quantization instead of leaving them unquantized."""
    keywords = preset_highprec_keywords(preset)
    if not keywords:
        return None
    return "|".join(re.escape(kw) for kw in keywords)
