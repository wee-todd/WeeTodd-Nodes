"""Analytical packed-row and operator-cost estimates for H3 continuation research."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import DiTConfig
from .packing import (
    AUDIO_CHANNELS,
    audio_latent_num_frames,
    video_latent_num_frames,
)


@dataclass(frozen=True)
class ContinuationCostEstimate:
    """One geometry's row counts and logical per-layer transformer costs."""

    width: int
    height: int
    num_frames: int
    text_rows: int
    context_frames: int
    rows_per_video_latent: int
    target_video_rows: int
    target_audio_rows: int
    condition_video_rows: int
    condition_audio_rows: int
    target_rows: int
    condition_rows: int
    sequence_rows: int
    attention_score_elements_per_head: int
    attention_score_elements_all_heads: int
    attention_score_bytes_bf16: int
    hidden_state_bytes_bf16: int
    qkv_bytes_bf16: int
    mlp_fused_bytes_bf16: int
    qkv_projection_macs: int
    attention_macs: int
    attention_output_projection_macs: int
    mlp_macs: int

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_continuation_cost(
    *,
    width: int,
    height: int,
    num_frames: int,
    text_rows: int,
    context_frames: int,
    config: DiTConfig | None = None,
) -> ContinuationCostEstimate:
    """Estimate logical row, activation, and multiply-accumulate counts for one H3 layer."""
    config = config or DiTConfig()
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("H3 cost geometry must use positive 32-pixel canvas multiples.")
    if text_rows < 1:
        raise ValueError("H3 cost geometry requires at least one text row.")
    if context_frames not in {0, 5, 22, 39, 56}:
        raise ValueError("H3 cost context must be 0, 5, 22, 39, or 56 frames.")

    latent_height = height // 16
    latent_width = width // 16
    _, patch_height, patch_width = config.patch_size
    if latent_height % patch_height or latent_width % patch_width:
        raise ValueError("H3 latent geometry is not divisible by the transformer patch size.")
    rows_per_video_latent = (latent_height // patch_height) * (
        latent_width // patch_width
    )
    target_video_rows = video_latent_num_frames(num_frames) * rows_per_video_latent
    target_audio_rows = audio_latent_num_frames(num_frames) * AUDIO_CHANNELS
    condition_video_rows = (
        video_latent_num_frames(context_frames) * rows_per_video_latent
        if context_frames
        else 0
    )
    condition_audio_rows = (
        audio_latent_num_frames(context_frames) * AUDIO_CHANNELS
        if context_frames
        else 0
    )
    target_rows = target_video_rows + target_audio_rows
    condition_rows = condition_video_rows + condition_audio_rows
    sequence_rows = text_rows + target_rows + condition_rows

    bytes_per_bf16 = 2
    attention_score_elements_per_head = sequence_rows * sequence_rows
    attention_score_elements_all_heads = (
        config.num_attention_heads * attention_score_elements_per_head
    )
    hidden_state_bytes_bf16 = sequence_rows * config.hidden_size * bytes_per_bf16
    qkv_bytes_bf16 = sequence_rows * 3 * config.inner_dim * bytes_per_bf16
    mlp_fused_bytes_bf16 = (
        sequence_rows * 2 * config.ffn_hidden_size * bytes_per_bf16
    )

    qkv_projection_macs = sequence_rows * config.hidden_size * 3 * config.inner_dim
    attention_macs = (
        2
        * config.num_attention_heads
        * sequence_rows
        * sequence_rows
        * config.attention_head_dim
    )
    attention_output_projection_macs = (
        sequence_rows * config.inner_dim * config.hidden_size
    )
    mlp_macs = sequence_rows * (
        config.hidden_size * 2 * config.ffn_hidden_size
        + config.ffn_hidden_size * config.hidden_size
    )
    return ContinuationCostEstimate(
        width=width,
        height=height,
        num_frames=num_frames,
        text_rows=text_rows,
        context_frames=context_frames,
        rows_per_video_latent=rows_per_video_latent,
        target_video_rows=target_video_rows,
        target_audio_rows=target_audio_rows,
        condition_video_rows=condition_video_rows,
        condition_audio_rows=condition_audio_rows,
        target_rows=target_rows,
        condition_rows=condition_rows,
        sequence_rows=sequence_rows,
        attention_score_elements_per_head=attention_score_elements_per_head,
        attention_score_elements_all_heads=attention_score_elements_all_heads,
        attention_score_bytes_bf16=(
            attention_score_elements_all_heads * bytes_per_bf16
        ),
        hidden_state_bytes_bf16=hidden_state_bytes_bf16,
        qkv_bytes_bf16=qkv_bytes_bf16,
        mlp_fused_bytes_bf16=mlp_fused_bytes_bf16,
        qkv_projection_macs=qkv_projection_macs,
        attention_macs=attention_macs,
        attention_output_projection_macs=attention_output_projection_macs,
        mlp_macs=mlp_macs,
    )
