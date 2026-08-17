"""In-memory LTX 2.5 IC-LoRA reference-video conditioning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class LTX25ICReferenceReport:
    source_frames: int
    encoded_frames: int
    dropped_tail_frames: int
    source_width: int
    source_height: int
    encoded_width: int
    encoded_height: int
    start_frame: int
    end_frame: int
    reference_downscale_factor: int
    reference_temporal_scale_factor: int
    strength: float
    attention_strength: float
    control_type: str
    conditioning_mode: str
    reference_size_policy: str
    reference_token_count: int
    target_token_count: int
    dense_attention_multiplier_vs_target: float
    attention_mask_layout: str

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _host_video(images: Any):
    import numpy as np

    detach = getattr(images, "detach", None)
    if detach is not None:
        images = detach()
    cpu = getattr(images, "cpu", None)
    if cpu is not None:
        images = cpu()
    video = np.asarray(images, dtype=np.float32)
    if video.ndim != 4 or video.shape[0] < 1 or video.shape[-1] < 3:
        raise ValueError(
            "LTX 2.5 IC-LoRA video reference must be a ComfyUI IMAGE batch "
            "with shape (frames, height, width, channels)."
        )
    video = video[..., :3]
    if not np.isfinite(video).all():
        raise ValueError("LTX 2.5 IC-LoRA reference contains non-finite pixels.")
    return np.ascontiguousarray(np.clip(video, 0.0, 1.0))


def _fit_reference_frames(video, *, maximum_frames: int, temporal_scale_factor: int):
    """Trim to a causal VAE sequence, then apply official temporal subsampling."""
    import numpy as np

    usable = min(int(video.shape[0]), int(maximum_frames))
    if usable < 9:
        raise ValueError(
            "LTX 2.5 video-reference conditioning requires at least nine frames "
            "(the causal VAE 1+8k contract). Use image_keyframe for a still image."
        )
    compatible = 1 + 8 * ((usable - 1) // 8)
    fitted = video[:compatible]
    if temporal_scale_factor > 1:
        indices = [0, *range(1, compatible, temporal_scale_factor)]
        fitted = fitted[np.asarray(indices, dtype=np.int64)]
        compatible = 1 + 8 * ((int(fitted.shape[0]) - 1) // 8)
        if compatible < 9:
            raise ValueError(
                "The IC-LoRA temporal scale leaves fewer than nine VAE-compatible frames."
            )
        fitted = fitted[:compatible]
    return np.ascontiguousarray(fitted), usable - (1 + 8 * ((usable - 1) // 8))


def _resize_center_crop(video, height: int, width: int):
    """Match official reference preprocessing without distorting aspect ratio."""
    import numpy as np
    from PIL import Image

    source_h, source_w = int(video.shape[1]), int(video.shape[2])
    scale = max(width / source_w, height / source_h)
    resized_w = max(width, round(source_w * scale))
    resized_h = max(height, round(source_h * scale))
    left = (resized_w - width) // 2
    top = (resized_h - height) // 2
    output = np.empty((video.shape[0], height, width, 3), dtype=np.float32)
    for index, frame in enumerate(video):
        image = Image.fromarray((frame * 255.0).round().astype(np.uint8), mode="RGB")
        image = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        image = image.crop((left, top, left + width, top + height))
        output[index] = np.asarray(image, dtype=np.float32) / 255.0
    return output


LTX25_INGREDIENTS_REFERENCE_SIZE_POLICIES = {
    "quality": None,
    "balanced": 512 * 288,
    "speed": 384 * 224,
}


class _CompactAttentionStrengthWrapper:
    """Append one reference group with a two-row structured attention mask.

    The first row describes target queries and the second describes reference
    queries. The Sol kernel selects one row per query tile. Dense fallbacks can
    expand the representation only when they execute.
    """

    def __init__(self, conditioning: object, *, attention_strength: float) -> None:
        self.conditioning = conditioning
        self.attention_strength = float(attention_strength)

    def apply(self, state, spatial_dims: tuple[int, int, int]):
        from ltx_core_mlx.conditioning.types.latent_cond import LatentState

        if state.attention_mask is not None:
            raise ValueError(
                "Compact Ingredients attention requires the reference sheet to be the only "
                "attention-strength conditioning group."
            )
        conditioned = self.conditioning.apply(state, spatial_dims)
        target_tokens = int(state.latent.shape[1])
        total_tokens = int(conditioned.latent.shape[1])
        reference_tokens = total_tokens - target_tokens
        if reference_tokens <= 0:
            return conditioned
        batch = int(conditioned.latent.shape[0])
        dtype = conditioned.latent.dtype
        strength = mx.full((batch, reference_tokens), self.attention_strength, dtype=dtype)
        target_ones = mx.ones((batch, target_tokens), dtype=dtype)
        reference_ones = mx.ones((batch, reference_tokens), dtype=dtype)
        target_cross = mx.concatenate([target_ones, strength], axis=1)
        reference_cross = mx.concatenate(
            [
                mx.full((batch, target_tokens), self.attention_strength, dtype=dtype),
                reference_ones,
            ],
            axis=1,
        )
        compact_mask = mx.stack([target_cross, reference_cross], axis=1)
        return LatentState(
            latent=conditioned.latent,
            clean_latent=conditioned.clean_latent,
            denoise_mask=conditioned.denoise_mask,
            positions=conditioned.positions,
            attention_mask=compact_mask,
        )


def plan_ingredients_reference_grid(
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
    policy: str,
) -> tuple[int, int]:
    """Choose the largest bounded 32-pixel grid for one Ingredients sheet."""

    if policy not in LTX25_INGREDIENTS_REFERENCE_SIZE_POLICIES:
        raise ValueError(f"Unsupported Ingredients reference size policy: {policy!r}.")
    max_height = min(target_height, (source_height // 32) * 32)
    max_width = min(target_width, (source_width // 32) * 32)
    if max_height < 32 or max_width < 32:
        raise ValueError("An Ingredients reference sheet must be at least 32 by 32 pixels.")
    budget = LTX25_INGREDIENTS_REFERENCE_SIZE_POLICIES[policy]
    if budget is None or max_height * max_width <= budget:
        return max_height, max_width

    source_ratio = source_width / source_height
    candidates = [
        (height, width)
        for height in range(32, max_height + 1, 32)
        for width in range(32, max_width + 1, 32)
        if height * width <= budget
    ]
    return max(
        candidates,
        key=lambda item: (
            item[0] * item[1],
            -abs(math.log((item[1] / item[0]) / source_ratio)),
        ),
    )


def _reference_mask(mask: Any, *, frames: int, latent_f: int, latent_h: int, latent_w: int):
    if mask is None:
        return None
    import numpy as np
    from ltx_pipelines_mlx.iclora_utils import downsample_mask_video_to_latent

    detach = getattr(mask, "detach", None)
    if detach is not None:
        mask = detach()
    cpu = getattr(mask, "cpu", None)
    if cpu is not None:
        mask = cpu()
    value = np.asarray(mask, dtype=np.float32)
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3 or value.shape[0] not in {1, frames}:
        raise ValueError(
            "An IC-LoRA attention mask must contain one mask or one mask per encoded frame."
        )
    if value.shape[0] == 1:
        value = np.repeat(value, frames, axis=0)
    value = np.clip(value, 0.0, 1.0)[None, None]
    return downsample_mask_video_to_latent(
        mx.array(value), target_f=latent_f, target_h=latent_h, target_w=latent_w
    )


def encode_reference_video_conditioning(
    *,
    images: Any,
    video_encoder,
    target_height: int,
    target_width: int,
    target_num_frames: int,
    frame_rate: float,
    start_frame: int,
    end_frame: int,
    strength: float,
    attention_strength: float,
    attention_mask: Any = None,
    reference_downscale_factor: int = 1,
    reference_temporal_scale_factor: int = 1,
    control_type: str = "custom_preprocessed",
    reference_size_policy: str = "quality",
    compact_attention_mask: bool = False,
):
    """Encode a Comfy IMAGE batch and return an appendable IC-LoRA conditioning."""
    from ltx_core_mlx.conditioning.types.attention_strength_wrapper import (
        ConditioningItemAttentionStrengthWrapper,
    )
    from ltx_core_mlx.conditioning.types.reference_video_cond import (
        VideoConditionByReferenceLatent,
    )
    from ltx_core_mlx.utils.positions import compute_video_positions

    if not 0.0 <= strength <= 1.0 or not 0.0 <= attention_strength <= 1.0:
        raise ValueError("IC-LoRA conditioning and attention strengths must be in [0, 1].")
    if reference_downscale_factor < 1 or reference_temporal_scale_factor < 1:
        raise ValueError("IC-LoRA reference scale factors must be positive.")
    if target_height % reference_downscale_factor or target_width % reference_downscale_factor:
        raise ValueError(
            "The stage-one canvas must be divisible by the IC-LoRA reference downscale factor."
        )
    ref_height = target_height // reference_downscale_factor
    ref_width = target_width // reference_downscale_factor
    video = _host_video(images)
    source_frames, source_h, source_w = map(int, video.shape[:3])
    maximum_frames = min(target_num_frames - start_frame, end_frame - start_frame + 1)
    static_reference_sheet = control_type == "ingredients_reference_sheet"
    if static_reference_sheet:
        # Reference sheets describe identity rather than the output canvas. Avoid
        # inventing source detail when a high-resolution generation would otherwise
        # enlarge a smaller sheet. The VAE still requires a 32-pixel grid.
        ref_height, ref_width = plan_ingredients_reference_grid(
            source_height=source_h,
            source_width=source_w,
            target_height=ref_height,
            target_width=ref_width,
            policy=reference_size_policy,
        )
        if source_frames != 1:
            raise ValueError("An Ingredients reference sheet requires exactly one image.")
        if target_num_frames < 121:
            raise ValueError(
                "An Ingredients reference sheet requires at least 121 generated frames."
            )
        if start_frame != 0 or maximum_frames != target_num_frames:
            raise ValueError(
                "An Ingredients reference sheet must span the complete generated clip."
            )
        repeated_frames = 1 + 8 * ((maximum_frames - 1) // 8)
        if repeated_frames < 9:
            raise ValueError("An Ingredients reference sheet requires at least nine output frames.")
        fitted = video
        dropped = maximum_frames - repeated_frames
    else:
        fitted, dropped = _fit_reference_frames(
            video,
            maximum_frames=maximum_frames,
            temporal_scale_factor=reference_temporal_scale_factor,
        )
    if ref_height < 32 or ref_width < 32 or ref_height % 32 or ref_width % 32:
        raise ValueError(
            "The IC-LoRA reference dimensions must remain positive multiples of 32."
        )
    fitted = _resize_center_crop(fitted, ref_height, ref_width)
    pixels = mx.array(fitted.transpose(3, 0, 1, 2)[None])
    if static_reference_sheet:
        pixels = mx.broadcast_to(
            pixels,
            (1, pixels.shape[1], repeated_frames, pixels.shape[3], pixels.shape[4]),
        )
    pixels = (pixels * 2.0 - 1.0).astype(mx.bfloat16)
    encoded = video_encoder.encode(pixels)
    mx.eval(encoded)
    latent_f, latent_h, latent_w = map(int, encoded.shape[2:])
    tokens = encoded.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
    target_latent_f = (target_num_frames - 1) // 8 + 1
    target_token_count = (
        target_latent_f * (target_height // 32) * (target_width // 32)
    )
    reference_token_count = int(tokens.shape[1])
    positions = compute_video_positions(
        latent_f,
        latent_h,
        latent_w,
        frame_rate=frame_rate / reference_temporal_scale_factor,
    )
    if reference_temporal_scale_factor != 1:
        offset = (reference_temporal_scale_factor - 1) / frame_rate
        temporal = mx.maximum(positions[..., 0] - offset, 0.0)
        positions = mx.concatenate([temporal[..., None], positions[..., 1:]], axis=-1)
    if start_frame:
        positions = mx.concatenate(
            [positions[..., :1] + start_frame / frame_rate, positions[..., 1:]], axis=-1
        )
    conditioning = VideoConditionByReferenceLatent(
        reference_latent=tokens,
        reference_positions=positions,
        downscale_factor=reference_downscale_factor,
        strength=strength,
    )
    latent_mask = _reference_mask(
        attention_mask,
        frames=int(pixels.shape[2]),
        latent_f=latent_f,
        latent_h=latent_h,
        latent_w=latent_w,
    )
    if latent_mask is not None:
        conditioning = ConditioningItemAttentionStrengthWrapper(
            conditioning=conditioning,
            attention_mask=latent_mask * attention_strength,
        )
    elif attention_strength < 1.0:
        if static_reference_sheet and compact_attention_mask:
            conditioning = _CompactAttentionStrengthWrapper(
                conditioning=conditioning,
                attention_strength=attention_strength,
            )
        else:
            conditioning = ConditioningItemAttentionStrengthWrapper(
                conditioning=conditioning,
                attention_mask=attention_strength,
            )
    report = LTX25ICReferenceReport(
        source_frames=source_frames,
        encoded_frames=int(pixels.shape[2]),
        dropped_tail_frames=dropped,
        source_width=source_w,
        source_height=source_h,
        encoded_width=ref_width,
        encoded_height=ref_height,
        start_frame=start_frame,
        end_frame=start_frame + int(pixels.shape[2]) - 1,
        reference_downscale_factor=reference_downscale_factor,
        reference_temporal_scale_factor=reference_temporal_scale_factor,
        strength=strength,
        attention_strength=attention_strength,
        control_type=control_type,
        conditioning_mode=(
            "static_reference_sheet_repeated_to_target"
            if static_reference_sheet
            else "reference_video"
        ),
        reference_size_policy=reference_size_policy,
        reference_token_count=reference_token_count,
        target_token_count=target_token_count,
        dense_attention_multiplier_vs_target=(
            (target_token_count + reference_token_count) ** 2
            / target_token_count**2
        ),
        attention_mask_layout=(
            "compact_two_row_suffix"
            if static_reference_sheet
            and compact_attention_mask
            and latent_mask is None
            and attention_strength < 1.0
            else "full_matrix"
            if latent_mask is not None or attention_strength < 1.0
            else "none"
        ),
    )
    return conditioning, report


__all__ = [
    "LTX25ICReferenceReport",
    "LTX25_INGREDIENTS_REFERENCE_SIZE_POLICIES",
    "encode_reference_video_conditioning",
    "plan_ingredients_reference_grid",
]
