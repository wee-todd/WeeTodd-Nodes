"""Lightweight public option contracts for LTX 2.5 video upscaling."""

LTX25_PIXEL_SPATIAL_MODE = "pixel spatial IC-LoRA 2x (recommended)"
LTX25_UPSCALE_MODES = (
    "latent upscale only",
    "latent upscale + stage-two refinement",
    LTX25_PIXEL_SPATIAL_MODE,
)
LTX25_SOURCE_FRAME_ANCHORS = ("none", "first frame", "first + last frames")
LTX25_INPUT_SIZE_POLICIES = (
    "center crop to 32px grid (recommended)",
    "require dimensions divisible by 32",
)

__all__ = [
    "LTX25_INPUT_SIZE_POLICIES",
    "LTX25_PIXEL_SPATIAL_MODE",
    "LTX25_SOURCE_FRAME_ANCHORS",
    "LTX25_UPSCALE_MODES",
]
