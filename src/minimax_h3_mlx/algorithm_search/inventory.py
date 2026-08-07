"""Structural H3 operation inventory generated from model and packed-sequence geometry."""

from __future__ import annotations

from dataclasses import dataclass

from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.packing import (
    FPS,
    align_num_frames,
    audio_latent_num_frames,
    video_latent_num_frames,
)

from .schema import OperationRecord


@dataclass(frozen=True)
class InventoryCase:
    name: str
    width: int
    height: int
    duration_seconds: float
    prompt_rows: int = 512

    def geometry(self, config: DiTConfig) -> dict[str, int]:
        frames = align_num_frames(int(round(self.duration_seconds * FPS)))
        latent_frames = video_latent_num_frames(frames)
        latent_height = self.height // 16
        latent_width = self.width // 16
        _, patch_h, patch_w = config.patch_size
        video_rows = latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
        audio_rows = audio_latent_num_frames(frames) * 2
        return {
            "frames": frames,
            "video_rows": video_rows,
            "audio_rows": audio_rows,
            "prompt_rows": self.prompt_rows,
            "sequence_rows": video_rows + audio_rows + self.prompt_rows,
        }


def _linear_flops(rows: int, input_dim: int, output_dim: int) -> int:
    return 2 * rows * input_dim * output_dim


def _linear(
    case: InventoryCase,
    module: str,
    block: int | None,
    rows: int,
    input_dim: int,
    output_dim: int,
    operation_type: str,
    *,
    shared_input_group: str | None = None,
    modalities: tuple[str, ...] = ("video", "text", "audio"),
    frequency: str = "every_diffusion_step",
    notes: str = "",
) -> OperationRecord:
    return OperationRecord(
        case=case.name,
        module=module,
        block=block,
        operation_type=operation_type,
        input_shapes=((1, rows, input_dim),),
        output_shapes=((1, rows, output_dim),),
        weight_shapes=((output_dim, input_dim),),
        approximate_flops=_linear_flops(rows, input_dim, output_dim),
        weight_parameters=input_dim * output_dim,
        shared_input_group=shared_input_group,
        modalities=modalities,
        frequency=frequency,
        temporary_bytes_estimate=rows * output_dim * 2,
        notes=notes,
    )


def build_operation_inventory(
    cases: tuple[InventoryCase, ...], config: DiTConfig | None = None
) -> list[OperationRecord]:
    """Create a deterministic inventory of the actual fused H3 computation graph."""
    cfg = config or DiTConfig()
    records: list[OperationRecord] = []
    for case in cases:
        geometry = case.geometry(cfg)
        rows = geometry["sequence_rows"]
        text_rows = geometry["prompt_rows"]
        hidden = cfg.hidden_size
        inner = cfg.inner_dim
        ffn = cfg.ffn_hidden_size
        for block in range(cfg.token_refiner_num_layers):
            prefix = f"token_refiner.blocks.{block}"
            records.extend(
                [
                    _linear(
                        case,
                        f"{prefix}.attn.qkv_proj",
                        block,
                        text_rows,
                        hidden,
                        3 * inner,
                        "token_refiner_fused_qkv_projection",
                        modalities=("text",),
                    ),
                    OperationRecord(
                        case.name,
                        f"{prefix}.attn.sdpa",
                        block,
                        "token_refiner_dense_sdpa",
                        ((1, cfg.num_attention_heads, text_rows, cfg.attention_head_dim),) * 3,
                        ((1, cfg.num_attention_heads, text_rows, cfg.attention_head_dim),),
                        approximate_flops=(
                            4
                            * cfg.num_attention_heads
                            * text_rows
                            * text_rows
                            * cfg.attention_head_dim
                        ),
                        modalities=("text",),
                        temporary_bytes_estimate=(
                            cfg.num_attention_heads * text_rows * text_rows * 2
                        ),
                    ),
                    _linear(
                        case,
                        f"{prefix}.attn.out_proj",
                        block,
                        text_rows,
                        inner,
                        hidden,
                        "token_refiner_attention_output_projection",
                        modalities=("text",),
                    ),
                    _linear(
                        case,
                        f"{prefix}.mlp.fc1",
                        block,
                        text_rows,
                        hidden,
                        2 * ffn,
                        "token_refiner_fused_swiglu_projection",
                        modalities=("text",),
                    ),
                    _linear(
                        case,
                        f"{prefix}.mlp.fc2",
                        block,
                        text_rows,
                        ffn,
                        hidden,
                        "token_refiner_ffn_output_projection",
                        modalities=("text",),
                    ),
                ]
            )
        for block in range(cfg.num_layers):
            prefix = f"blocks.{block}"
            shared = f"{case.name}:{prefix}:attention_input"
            records.extend(
                [
                    OperationRecord(
                        case.name,
                        f"{prefix}.norm1",
                        block,
                        "rms_norm_and_adaln",
                        ((1, rows, hidden),),
                        ((1, rows, hidden),),
                        weight_shapes=((hidden,),),
                        approximate_flops=8 * rows * hidden,
                        weight_parameters=hidden,
                        shared_input_group=shared,
                        temporary_bytes_estimate=rows * hidden * 2,
                    ),
                    _linear(
                        case,
                        f"{prefix}.attn.qkv_proj",
                        block,
                        rows,
                        hidden,
                        3 * inner,
                        "fused_qkv_projection",
                        shared_input_group=shared,
                        notes=(
                            "Actual checkpoint operation; Q, K, and V share one fused input/GEMM."
                        ),
                    ),
                    OperationRecord(
                        case.name,
                        f"{prefix}.attn.q_norm_k_norm_rope",
                        block,
                        "qk_normalization_and_rotary",
                        ((1, cfg.num_attention_heads, rows, cfg.attention_head_dim),) * 2,
                        ((1, cfg.num_attention_heads, rows, cfg.attention_head_dim),) * 2,
                        approximate_flops=20 * rows * inner,
                        weight_parameters=2 * cfg.attention_head_dim,
                        shared_input_group=shared,
                        temporary_bytes_estimate=4 * rows * inner * 2,
                    ),
                    OperationRecord(
                        case.name,
                        f"{prefix}.attn.sdpa_scores",
                        block,
                        "attention_scores_softmax",
                        ((1, cfg.num_attention_heads, rows, cfg.attention_head_dim),) * 2,
                        ((1, cfg.num_attention_heads, rows, rows),),
                        approximate_flops=(
                            2
                            * cfg.num_attention_heads
                            * rows
                            * rows
                            * cfg.attention_head_dim
                        ),
                        weight_parameters=0,
                        shared_input_group=shared,
                        temporary_bytes_estimate=cfg.num_attention_heads * rows * rows * 2,
                        notes=(
                            "Fused SDPA need not materialize the full score tensor; estimate is "
                            "conceptual."
                        ),
                    ),
                    OperationRecord(
                        case.name,
                        f"{prefix}.attn.sdpa_value",
                        block,
                        "attention_value_product",
                        (
                            (1, cfg.num_attention_heads, rows, rows),
                            (1, cfg.num_attention_heads, rows, cfg.attention_head_dim),
                        ),
                        ((1, cfg.num_attention_heads, rows, cfg.attention_head_dim),),
                        approximate_flops=(
                            2
                            * cfg.num_attention_heads
                            * rows
                            * rows
                            * cfg.attention_head_dim
                        ),
                        weight_parameters=0,
                        shared_input_group=shared,
                        temporary_bytes_estimate=rows * inner * 2,
                        notes="Executed inside fused MLX SDPA with score calculation.",
                    ),
                    _linear(
                        case,
                        f"{prefix}.attn.out_proj",
                        block,
                        rows,
                        inner,
                        hidden,
                        "attention_output_projection",
                    ),
                    OperationRecord(
                        case.name,
                        f"{prefix}.norm2",
                        block,
                        "rms_norm_and_adaln",
                        ((1, rows, hidden),),
                        ((1, rows, hidden),),
                        weight_shapes=((hidden,),),
                        approximate_flops=8 * rows * hidden,
                        weight_parameters=hidden,
                        shared_input_group=f"{case.name}:{prefix}:mlp_input",
                        temporary_bytes_estimate=rows * hidden * 2,
                    ),
                    _linear(
                        case,
                        f"{prefix}.mlp.fc1",
                        block,
                        rows,
                        hidden,
                        2 * ffn,
                        "fused_swiglu_gate_value_projection",
                        shared_input_group=f"{case.name}:{prefix}:mlp_input",
                        notes="Gate and value projections share one fused input/GEMM.",
                    ),
                    OperationRecord(
                        case.name,
                        f"{prefix}.mlp.swiglu",
                        block,
                        "silu_and_elementwise_product",
                        ((1, rows, 2 * ffn),),
                        ((1, rows, ffn),),
                        approximate_flops=8 * rows * ffn,
                        weight_parameters=0,
                        temporary_bytes_estimate=rows * ffn * 2,
                    ),
                    _linear(
                        case,
                        f"{prefix}.mlp.fc2",
                        block,
                        rows,
                        ffn,
                        hidden,
                        "ffn_output_projection",
                    ),
                    OperationRecord(
                        case.name,
                        f"{prefix}.residuals",
                        block,
                        "gated_residual_additions",
                        ((1, rows, hidden),) * 4,
                        ((1, rows, hidden),),
                        approximate_flops=6 * rows * hidden,
                        weight_parameters=0,
                        temporary_bytes_estimate=rows * hidden * 2,
                    ),
                ]
            )

        records.extend(
            [
                _linear(
                    case,
                    "time_embedder.proj_in",
                    None,
                    1,
                    cfg.timestep_input_dim,
                    cfg.time_embed_hidden_size,
                    "timestep_input_projection",
                    modalities=(),
                    frequency="once_per_distinct_schedule_value",
                    notes="Built once per distinct schedule value before cached inference.",
                ),
                _linear(
                    case,
                    "time_embedder.proj_out",
                    None,
                    1,
                    cfg.time_embed_hidden_size,
                    cfg.time_embed_dim,
                    "timestep_output_projection",
                    modalities=(),
                    frequency="once_per_distinct_schedule_value",
                    notes="Built once per distinct schedule value before cached inference.",
                ),
                _linear(
                    case,
                    "video_patch_proj",
                    None,
                    geometry["video_rows"],
                    cfg.video_patch_dim,
                    hidden,
                    "video_patch_projection",
                    modalities=("video",),
                ),
                _linear(
                    case,
                    "audio_patch_proj",
                    None,
                    geometry["audio_rows"],
                    cfg.audio_latents_dim,
                    hidden,
                    "audio_patch_projection",
                    modalities=("audio",),
                ),
                _linear(
                    case,
                    "condition_proj",
                    None,
                    geometry["prompt_rows"],
                    cfg.text_dim,
                    hidden,
                    "text_condition_projection",
                    modalities=("text",),
                ),
                _linear(
                    case,
                    "final_layer.video_out",
                    None,
                    rows,
                    hidden,
                    cfg.video_patch_dim,
                    "video_output_head",
                    modalities=("video",),
                    notes=(
                        "Current implementation runs this head on every packed row before "
                        "selection."
                    ),
                ),
                _linear(
                    case,
                    "final_layer.audio_out",
                    None,
                    rows,
                    hidden,
                    cfg.audio_latents_dim,
                    "audio_output_head",
                    modalities=("audio",),
                    notes=(
                        "Current implementation runs this head on every packed row before "
                        "selection."
                    ),
                ),
            ]
        )
    return records
