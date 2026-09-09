"""LTX 2.5 transformer construction and direct MLX checkpoint loading."""

from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import MethodType
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open

from wee_todd_mlx.adapter_contract import inspect_adapter
from wee_todd_mlx.numpy_import import adopt_numpy_array


class _BiasFreeFeedForward(nn.Module):
    """LTX 2.5 video GELU feed-forward network without projection biases."""

    def __init__(self, dim: int, *, mult: float = 4.0) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.proj_in = nn.Linear(dim, inner_dim, bias=False)
        self.proj_out = nn.Linear(inner_dim, dim, bias=False)

    def __call__(self, value: mx.array) -> mx.array:
        return self.proj_out(nn.gelu_approx(self.proj_in(value)))


@dataclass(frozen=True)
class LTX25TransformerConfig:
    """Checkpoint-driven LTX 2.5 architecture fields used by the MLX port."""

    num_layers: int = 48
    video_dim: int = 4096
    audio_dim: int = 2048
    video_num_heads: int = 32
    audio_num_heads: int = 32
    video_head_dim: int = 128
    audio_head_dim: int = 64
    video_patch_channels: int = 128
    audio_patch_channels: int = 128
    timestep_scale_multiplier: float = 1000.0
    av_ca_timestep_scale_multiplier: float = 1.0
    rope_theta: float = 10000.0
    rope_type: str = "split"
    positional_embedding_max_pos: tuple[int, ...] = (20, 2048, 2048)
    audio_positional_embedding_max_pos: tuple[int, ...] = (20,)
    norm_eps: float = 1e-6
    cross_attention_adaln: bool = True
    use_prompt_adaln_single: bool = True
    ff_bias: bool = False
    audio_ff_bias: bool = True
    caption_proj_before_connector: bool = True
    use_keyframes_abs_pos_embedding: bool = True
    frequencies_precision: str = "float64"

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> LTX25TransformerConfig:
        config = metadata.get("config", metadata)
        transformer = config.get("transformer", config) if isinstance(config, dict) else {}
        if not isinstance(transformer, dict):
            raise ValueError("LTX 2.5 transformer metadata has no transformer configuration.")
        defaults = cls()
        result = cls(
            num_layers=int(transformer.get("num_layers", defaults.num_layers)),
            video_dim=int(transformer.get("cross_attention_dim", defaults.video_dim)),
            audio_dim=int(transformer.get("audio_cross_attention_dim", defaults.audio_dim)),
            video_num_heads=int(transformer.get("num_attention_heads", defaults.video_num_heads)),
            audio_num_heads=int(
                transformer.get("audio_num_attention_heads", defaults.audio_num_heads)
            ),
            video_head_dim=int(transformer.get("attention_head_dim", defaults.video_head_dim)),
            audio_head_dim=int(
                transformer.get("audio_attention_head_dim", defaults.audio_head_dim)
            ),
            video_patch_channels=int(transformer.get("in_channels", defaults.video_patch_channels)),
            audio_patch_channels=int(
                transformer.get("audio_in_channels", defaults.audio_patch_channels)
            ),
            timestep_scale_multiplier=float(
                transformer.get("timestep_scale_multiplier", defaults.timestep_scale_multiplier)
            ),
            av_ca_timestep_scale_multiplier=float(
                transformer.get(
                    "av_ca_timestep_scale_multiplier",
                    defaults.av_ca_timestep_scale_multiplier,
                )
            ),
            rope_theta=float(transformer.get("positional_embedding_theta", defaults.rope_theta)),
            rope_type=str(transformer.get("rope_type", defaults.rope_type)),
            positional_embedding_max_pos=tuple(
                transformer.get(
                    "positional_embedding_max_pos", defaults.positional_embedding_max_pos
                )
            ),
            audio_positional_embedding_max_pos=tuple(
                transformer.get(
                    "audio_positional_embedding_max_pos",
                    defaults.audio_positional_embedding_max_pos,
                )
            ),
            norm_eps=float(transformer.get("norm_eps", defaults.norm_eps)),
            cross_attention_adaln=bool(
                transformer.get("cross_attention_adaln", defaults.cross_attention_adaln)
            ),
            use_prompt_adaln_single=bool(
                transformer.get("use_prompt_adaln_single", defaults.use_prompt_adaln_single)
            ),
            ff_bias=bool(transformer.get("ff_bias", defaults.ff_bias)),
            audio_ff_bias=bool(transformer.get("audio_ff_bias", defaults.audio_ff_bias)),
            caption_proj_before_connector=bool(
                transformer.get(
                    "caption_proj_before_connector",
                    defaults.caption_proj_before_connector,
                )
            ),
            use_keyframes_abs_pos_embedding=bool(
                transformer.get(
                    "use_keyframes_abs_pos_embedding",
                    defaults.use_keyframes_abs_pos_embedding,
                )
            ),
            frequencies_precision=str(
                transformer.get("frequencies_precision", defaults.frequencies_precision)
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        incompatible = []
        if not self.cross_attention_adaln:
            incompatible.append("cross_attention_adaln=true")
        if not self.use_prompt_adaln_single:
            incompatible.append("use_prompt_adaln_single=true")
        if self.ff_bias:
            incompatible.append("ff_bias=false")
        if not self.audio_ff_bias:
            incompatible.append("audio_ff_bias=true")
        if not self.caption_proj_before_connector:
            incompatible.append("caption_proj_before_connector=true")
        if not self.use_keyframes_abs_pos_embedding:
            incompatible.append("use_keyframes_abs_pos_embedding=true")
        if incompatible:
            raise ValueError(
                "The selected checkpoint does not declare the supported LTX 2.5 "
                "transformer architecture: " + ", ".join(incompatible)
            )

    def base_config(self):
        from ltx_core_mlx.model.transformer.model import LTXModelConfig

        return LTXModelConfig(
            num_layers=self.num_layers,
            video_dim=self.video_dim,
            audio_dim=self.audio_dim,
            video_num_heads=self.video_num_heads,
            audio_num_heads=self.audio_num_heads,
            video_head_dim=self.video_head_dim,
            audio_head_dim=self.audio_head_dim,
            av_cross_num_heads=self.audio_num_heads,
            av_cross_head_dim=self.audio_head_dim,
            video_patch_channels=self.video_patch_channels,
            audio_patch_channels=self.audio_patch_channels,
            timestep_scale_multiplier=self.timestep_scale_multiplier,
            av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            rope_theta=self.rope_theta,
            rope_type=self.rope_type,
            positional_embedding_max_pos=self.positional_embedding_max_pos,
            audio_positional_embedding_max_pos=self.audio_positional_embedding_max_pos,
            norm_eps=self.norm_eps,
        )


class LTX25Model:
    """Construct the LTX 2.5 DiT using the established MLX LTX kernels."""

    @staticmethod
    def build(config: LTX25TransformerConfig):
        from ltx_core_mlx.model.transformer.model import LTXModel

        config.validate()
        model = LTXModel(config.base_config())
        # The released 2.5 checkpoint removes video FF biases only. Audio FF and
        # both prompt AdaLN modules retain their trained bias tensors.
        for block in model.transformer_blocks:
            block.ff = _BiasFreeFeedForward(config.video_dim)
        model.keyframes_abs_pos_embedding = mx.zeros((1, config.video_dim))
        model.ltx25_config = config
        model._compute_rope_freqs = MethodType(_compute_rope_freqs_float64, model)
        return model


def precompute_rope_freqs_float64(
    positions: mx.array,
    *,
    inner_dim: int,
    num_heads: int,
    theta: float,
    max_pos: list[int],
    rope_type: str = "split",
):
    """Build the released 2.5 frequency grid in NumPy float64, then use MLX."""
    from ltx_core_mlx.model.transformer.rope import compute_freqs

    num_pos_dims = positions.shape[-1]
    count = inner_dim // (2 * num_pos_dims)
    powers = np.linspace(0.0, 1.0, count, dtype=np.float64)
    indices = np.power(theta, powers) * (math.pi / 2.0)
    freqs = compute_freqs(adopt_numpy_array(indices, dtype=np.float32), positions, max_pos)
    batch, tokens, frequency_count = freqs.shape
    if rope_type == "interleaved":
        cos_f = mx.repeat(mx.cos(freqs), 2, axis=-1)
        sin_f = mx.repeat(mx.sin(freqs), 2, axis=-1)
        padding = inner_dim - cos_f.shape[-1]
        if padding > 0:
            cos_f = mx.concatenate([mx.ones((*cos_f.shape[:-1], padding)), cos_f], axis=-1)
            sin_f = mx.concatenate([mx.zeros((*sin_f.shape[:-1], padding)), sin_f], axis=-1)
        head_dim = inner_dim // num_heads
        return (
            cos_f.reshape(batch, tokens, num_heads, head_dim).transpose(0, 2, 1, 3),
            sin_f.reshape(batch, tokens, num_heads, head_dim).transpose(0, 2, 1, 3),
            rope_type,
        )
    expected = inner_dim // 2
    padding = expected - frequency_count
    if padding > 0:
        freqs = mx.concatenate([mx.zeros((*freqs.shape[:-1], padding)), freqs], axis=-1)
    head_dim_half = inner_dim // (2 * num_heads)
    return (
        mx.cos(freqs).reshape(batch, tokens, num_heads, head_dim_half).transpose(0, 2, 1, 3),
        mx.sin(freqs).reshape(batch, tokens, num_heads, head_dim_half).transpose(0, 2, 1, 3),
        rope_type,
    )


def _compute_rope_freqs_float64(
    self,
    positions: mx.array,
    num_heads: int,
    head_dim: int,
    max_pos_override: list[int] | None = None,
):
    max_pos = (
        max_pos_override
        if max_pos_override is not None
        else list(self.config.positional_embedding_max_pos[: positions.shape[-1]])
    )
    return precompute_rope_freqs_float64(
        positions,
        inner_dim=num_heads * head_dim,
        num_heads=num_heads,
        theta=self.config.rope_theta,
        max_pos=max_pos,
        rope_type=self.config.rope_type,
    )


def transformer_metadata(path: str | Path) -> dict[str, Any]:
    """Return decoded safetensors metadata without materializing tensors."""
    source = Path(path).expanduser()
    if source.is_dir():
        from .paged_checkpoint import LTX25PagedManifest

        manifest = LTX25PagedManifest.load(source)
        if manifest.kind != "transformer":
            raise ValueError(f"Expected a paged transformer, got {manifest.kind!r}: {source}")
        return manifest.metadata
    with safe_open(source, framework="numpy") as handle:
        raw = handle.metadata() or {}
    decoded: dict[str, Any] = {}
    for key, value in raw.items():
        try:
            decoded[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            decoded[key] = value
    return decoded


_LTX25_CANONICAL_IC_TARGETS = {
    f"transformer_blocks.{block}.{target}"
    for block in range(48)
    for target in (
        "attn1.to_k",
        "attn1.to_out",
        "attn1.to_q",
        "attn1.to_v",
        "attn2.to_k",
        "attn2.to_out",
        "attn2.to_q",
        "attn2.to_v",
        "ff.proj_in",
        "ff.proj_out",
    )
}

_LTX25_NON_BLOCK_LORA_TARGETS = {
    *(f"{prefix}.emb.timestep_embedder.linear{index}" for prefix in (
        "adaln_single",
        "audio_adaln_single",
        "audio_prompt_adaln_single",
        "av_ca_a2v_gate_adaln_single",
        "av_ca_audio_scale_shift_adaln_single",
        "av_ca_v2a_gate_adaln_single",
        "av_ca_video_scale_shift_adaln_single",
        "prompt_adaln_single",
    ) for index in (1, 2)),
    *(f"{prefix}.linear" for prefix in (
        "adaln_single",
        "audio_adaln_single",
        "audio_prompt_adaln_single",
        "av_ca_a2v_gate_adaln_single",
        "av_ca_audio_scale_shift_adaln_single",
        "av_ca_v2a_gate_adaln_single",
        "av_ca_video_scale_shift_adaln_single",
        "prompt_adaln_single",
    )),
    "audio_patchify_proj",
    "audio_proj_out",
    "patchify_proj",
    "proj_out",
}


def _normalize_ltx25_adapter_target(target: str) -> str | None:
    mapped = remap_comfy_transformer_key(target + ".weight")
    return mapped.removesuffix(".weight") if mapped is not None else None


def _ltx25_auxiliary_names() -> set[str]:
    return {
        prefix + name
        for prefix in (
            "diffusion_model.reference_slot_embedding.",
            "reference_slot_embedding.",
        )
        for name in _LTX25_MSR_SLOT_SHAPES
    }


def _inspect_ltx25_adapter_contract(path: Path) -> dict[str, Any]:
    return inspect_adapter(
        path,
        target_normalizer=_normalize_ltx25_adapter_target,
        allowed_auxiliary_names=_ltx25_auxiliary_names(),
    )


def _structural_ltx25_adapter_family(
    *,
    is_msr: bool,
    downscale: int,
    temporal_scale: int,
    spatial_scale: int | None,
    adapter_ranks: set[int],
    adapter_targets: set[str],
) -> tuple[str, str]:
    """Classify released task adapters without trusting their filenames."""
    if is_msr:
        return "multi_subject_reference", "learned Fourier reference-slot tensors"
    if spatial_scale is not None:
        return "pixel_spatial_upscaler", "explicit reference_spatial_scale_factor metadata"
    complete_48_block_layout = adapter_targets == _LTX25_CANONICAL_IC_TARGETS
    signature = (downscale, temporal_scale, tuple(sorted(adapter_ranks)))
    if complete_48_block_layout and signature == (1, 1, (32,)):
        return "crossview_warp", "48-block target/rank/reference-scale fingerprint"
    if complete_48_block_layout and signature == (1, 1, (128,)):
        return "ingredients_reference_sheet", "48-block target/rank/reference-scale fingerprint"
    if complete_48_block_layout and signature == (2, 1, (64,)):
        return "union_control", "48-block target/rank/reference-scale fingerprint"
    if complete_48_block_layout and signature == (2, 1, (32,)):
        return "motion_track", "48-block target/rank/reference-scale fingerprint"
    return "unclassified_reference_conditioning", "insufficient structural task evidence"


def _declared_ltx25_adapter_family(metadata: dict[str, str]) -> str | None:
    raw = metadata.get("weetodd_adapter_family") or metadata.get("adapter_family")
    if raw in (None, ""):
        return None
    normalized = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "crossview": "crossview_warp",
        "crossview_warp": "crossview_warp",
        "ingredients": "ingredients_reference_sheet",
        "ingredients_reference_sheet": "ingredients_reference_sheet",
        "motion_track": "motion_track",
        "motion_tracking": "motion_track",
        "union": "union_control",
        "union_control": "union_control",
        "pixel_spatial": "pixel_spatial_upscaler",
        "pixel_spatial_upscaler": "pixel_spatial_upscaler",
        "msr": "multi_subject_reference",
        "multi_subject_reference": "multi_subject_reference",
    }
    if normalized not in aliases:
        raise ValueError(
            "LTX 2.5 LoRA declares an unsupported adapter_family metadata value: "
            f"{raw!r}."
        )
    return aliases[normalized]


def inspect_ltx25_lora(path: str | Path) -> dict[str, Any]:
    """Validate an LTX 2.5-compatible LoRA header without materializing tensors.

    Lightricks ships LTX 2.5 workflows that intentionally pair the 2.5
    distilled transformer with selected 2.3 22B adapters.  The version label
    alone is therefore not a compatibility boundary.  Older adapters receive
    a stricter target-and-shape check before this loader accepts them.
    """
    source = Path(path).expanduser()
    if not source.is_file() or source.suffix != ".safetensors":
        raise FileNotFoundError(f"LTX 2.5 LoRA is not a safetensors file: {source}")
    contract = _inspect_ltx25_adapter_contract(source)
    metadata = contract["metadata"]
    keys = [*contract["auxiliary_tensors"]]
    msr_slot_prefix = next(
        (
            prefix
            for prefix in (
                "diffusion_model.reference_slot_embedding.",
                "reference_slot_embedding.",
            )
            if any(key.startswith(prefix) for key in keys)
        ),
        None,
    )
    is_msr = msr_slot_prefix is not None
    model_version = str(metadata.get("model_version", ""))
    version_match = re.match(r"^(\d+)\.(\d+)", model_version)
    version = (
        (int(version_match.group(1)), int(version_match.group(2)))
        if version_match
        else None
    )
    if version is not None and version < (2, 3):
        raise ValueError(
            "The selected LoRA does not declare a supported LTX 2.3-or-newer model version; "
            f"model_version={model_version!r}."
        )
    try:
        downscale = int(metadata.get("reference_downscale_factor", "1"))
    except (TypeError, ValueError) as exc:
        raise ValueError("LTX 2.5 LoRA has an invalid reference downscale factor.") from exc
    try:
        temporal_scale = int(metadata.get("reference_temporal_scale_factor", "1"))
    except (TypeError, ValueError) as exc:
        raise ValueError("LTX 2.5 LoRA has an invalid reference temporal scale factor.") from exc
    if downscale < 1 or temporal_scale < 1:
        raise ValueError("LTX 2.5 IC-LoRA reference scale factors must be positive.")
    pairs = len(contract["pairs"])
    adapter_ranks = set(contract["ranks"])
    incompatible_targets: list[str] = []
    adapter_targets = {pair["normalized_target"] for pair in contract["pairs"]}
    for pair in contract["pairs"]:
        stem = pair["target"]
        target = pair["normalized_target"] + ".weight"
        expected = _ltx25_bridge_target_shape(target)
        if expected is None and pair["normalized_target"] not in _LTX25_NON_BLOCK_LORA_TARGETS:
            incompatible_targets.append(stem)
        elif version is None or version < (2, 5):
            actual = tuple(pair["logical_shape"])
            if expected is None or actual != expected:
                incompatible_targets.append(stem)
    if incompatible_targets:
        examples = ", ".join(incompatible_targets[:3])
        raise ValueError(
            "The metadata-free or older LTX LoRA does not match the supported LTX 2.5 22B "
            "transformer "
            f"targets; incompatible targets include: {examples}."
        )
    spatial_scale = None
    if metadata.get("reference_spatial_scale_factor") not in (None, ""):
        try:
            spatial_scale = int(metadata["reference_spatial_scale_factor"])
        except (TypeError, ValueError) as exc:
            raise ValueError("LTX 2.5 LoRA has an invalid reference spatial scale factor.") from exc
    adapter_family, classification_basis = _structural_ltx25_adapter_family(
        is_msr=is_msr,
        downscale=downscale,
        temporal_scale=temporal_scale,
        adapter_ranks=adapter_ranks,
        adapter_targets=adapter_targets,
        spatial_scale=spatial_scale,
    )
    declared_family = _declared_ltx25_adapter_family(metadata)
    if declared_family is not None:
        if (
            adapter_family != "unclassified_reference_conditioning"
            and adapter_family != declared_family
        ):
            raise ValueError(
                "LTX 2.5 LoRA adapter_family metadata conflicts with its structural task "
                f"signature: declared {declared_family}, detected {adapter_family}."
            )
        adapter_family = declared_family
        classification_basis = "explicit adapter_family checkpoint metadata"
    family_scales = {
        "crossview_warp": (1, 1),
        "ingredients_reference_sheet": (1, 1),
        "motion_track": (2, 1),
        "union_control": (2, 1),
        "pixel_spatial_upscaler": (2, 1),
        "multi_subject_reference": (1, 1),
    }
    expected_scales = family_scales.get(adapter_family)
    if expected_scales is not None and (downscale, temporal_scale) != expected_scales:
        raise ValueError(
            f"LTX 2.5 {adapter_family} requires reference scale factors "
            f"{expected_scales}; found {(downscale, temporal_scale)}."
        )
    if adapter_family == "pixel_spatial_upscaler" and spatial_scale != 2:
        raise ValueError("LTX 2.5 Pixel-Spatial adapter requires spatial scale factor 2.")
    compatibility = (
        "native_ltx_2_5_msr_slot_contract"
        if is_msr and version is None
        else "ltx_2_3_22b_crossview_bridge"
        if adapter_family == "crossview_warp" and (version is None or version < (2, 5))
        else "native_ltx_2_5"
        if version is not None and version >= (2, 5)
        else "structural_ltx_2_5_22b"
        if version is None
        else "ltx_2_3_22b_bridge"
    )
    if not is_msr and "reference_downscale_factor" not in metadata:
        adapter_family = "standard"
        classification_basis = "no task-conditioning metadata or auxiliary tensors"
    scaling = contract["declared_scaling"]
    global_rank = scaling["rank"]
    global_alpha = scaling["alpha"]
    return {
        "path": source,
        "model_version": model_version,
        "reference_downscale_factor": downscale,
        "reference_temporal_scale_factor": temporal_scale,
        "adapter_pairs": pairs,
        "adapter_ranks": sorted(adapter_ranks),
        "pair_schemas": contract["pair_schemas"],
        "target_fingerprint": contract["target_fingerprint"],
        "normalized_target_count": len(adapter_targets),
        "compatibility": compatibility,
        "compatibility_basis": (
            "declared LTX 2.5 adapter"
            if compatibility == "native_ltx_2_5"
            else "exact 48-block, 480-pair rank-32 LTX 2.5 target contract"
            if compatibility == "ltx_2_3_22b_crossview_bridge"
            else "exact LTX 2.5 MSR learned-slot and 22B transformer target contract"
            if compatibility == "native_ltx_2_5_msr_slot_contract"
            else "exact supported LTX 2.5 22B target names and tensor shapes"
            if compatibility == "structural_ltx_2_5_22b"
            else "LTX 2.3-or-newer block targets and tensor shapes match the LTX 2.5 22B layout"
        ),
        "adapter_family": adapter_family,
        "classification_basis": classification_basis,
        "adapter_role": (
            "ic_lora"
            if is_msr or "reference_downscale_factor" in metadata
            else "transformer_lora"
        ),
        "ic_lora_task": (
            "pixel_spatial_upscaler"
            if adapter_family == "pixel_spatial_upscaler"
            else "multi_subject_reference"
            if is_msr
            else ("reference_conditioning" if "reference_downscale_factor" in metadata else None)
        ),
        "lora_rank": global_rank,
        "lora_alpha": global_alpha,
        "scaling_convention": (
            "per_target_alpha_over_pair_rank"
            if any(pair["alpha_tensor"] for pair in contract["pairs"])
            else "global_alpha_over_global_rank"
            if global_alpha is not None and global_rank is not None
            else "global_alpha_over_pair_rank"
            if global_alpha is not None
            else "unit"
        ),
        "bytes": source.stat().st_size,
    }


_LTX25_MSR_SLOT_SHAPES = {
    "frequencies": (16,),
    "net.0.weight": (256, 33),
    "net.0.bias": (256,),
    "net.2.weight": (128, 256),
    "net.2.bias": (128,),
}


def _validate_ltx25_msr_header(
    metadata: dict[str, str],
    shapes: dict[str, tuple[int, ...]],
    base: dict[str, Any],
) -> dict[str, Any]:
    prefix = next(
        (
            prefix
            for prefix in (
                "diffusion_model.reference_slot_embedding.",
                "reference_slot_embedding.",
            )
            if any(key.startswith(prefix) for key in shapes)
        ),
        None,
    )
    if prefix is None or base["adapter_family"] != "multi_subject_reference":
        raise ValueError(
            "The selected adapter has no reference_slot_embedding tensors and is not an "
            "LTX 2.5 multi-subject reference checkpoint."
        )
    missing = []
    incompatible = []
    for name, expected in _LTX25_MSR_SLOT_SHAPES.items():
        key = prefix + name
        if key not in shapes:
            missing.append(name)
        elif shapes[key] != expected:
            incompatible.append(f"{name}: {shapes[key]} != {expected}")
    if missing or incompatible:
        detail = [*(f"missing {name}" for name in missing), *incompatible]
        raise ValueError("Invalid LTX 2.5 MSR slot embedding: " + "; ".join(detail))
    if metadata.get("reference_slot_embedding_type") != "fourier_mlp":
        raise ValueError("LTX 2.5 MSR requires reference_slot_embedding_type=fourier_mlp.")
    if metadata.get("reference_token_order") != "prepend":
        raise ValueError("LTX 2.5 MSR requires reference_token_order=prepend.")
    if metadata.get("reference_slot_time_offsets") != "pic1_based_negative_time":
        raise ValueError(
            "LTX 2.5 MSR requires reference_slot_time_offsets=pic1_based_negative_time."
        )
    if base["adapter_pairs"] != 480 or base["adapter_ranks"] != [128]:
        raise ValueError(
            "LTX 2.5 MSR requires 480 balanced rank-128 transformer adapter pairs."
        )
    return {
        **base,
        "slot_prefix": prefix,
        "slot_shapes": dict(_LTX25_MSR_SLOT_SHAPES),
        "slot_embedding_type": "fourier_mlp",
        "slot_output_dim": 128,
        "maximum_references": 5,
        "reference_token_order": "prepend",
        "reference_slot_time_offsets": "pic1_based_negative_time",
        "reference_scale_factors_variable": str(
            metadata.get("reference_scale_factors_variable", "False")
        ).lower()
        in {"1", "true", "yes"},
    }


def inspect_ltx25_msr_lora(path: str | Path) -> dict[str, Any]:
    """Validate the learned-slot and LoRA contract of an LTX 2.5 MSR adapter."""

    source = Path(path).expanduser()
    base = inspect_ltx25_lora(source)
    with safe_open(source, framework="numpy") as handle:
        metadata = handle.metadata() or {}
        keys = list(handle.keys())
        shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in keys}
    return _validate_ltx25_msr_header(metadata, shapes, base)


def inspect_ltx25_ic_lora(path: str | Path) -> dict[str, Any]:
    """Backward-compatible alias for generic LTX 2.5 LoRA inspection."""
    return inspect_ltx25_lora(path)


def _ltx25_bridge_target_shape(target: str | None) -> tuple[int, int] | None:
    """Return the LTX 2.5 22B base matrix shape for a compatible block target."""
    if target is None:
        return None
    match = re.fullmatch(r"transformer_blocks\.(\d+)\.(.+)\.weight", target)
    if match is None or not 0 <= int(match.group(1)) < 48:
        return None
    tail = match.group(2)
    # Gate projections produce one logit per attention head, not a full hidden
    # vector. The official distilled adapter includes all six of these targets.
    if tail in {
        f"{name}.to_gate_logits" for name in ("attn1", "attn2", "audio_to_video_attn")
    }:
        return (32, 4096)
    if tail in {
        f"{name}.to_gate_logits"
        for name in ("audio_attn1", "audio_attn2", "video_to_audio_attn")
    }:
        return (32, 2048)
    attention_projections = ("to_q", "to_k", "to_v", "to_out")
    if tail in {
        *(
            f"{name}.{projection}"
            for name in ("attn1", "attn2")
            for projection in attention_projections
        ),
    }:
        return (4096, 4096)
    if tail == "ff.proj_in":
        return (16384, 4096)
    if tail == "ff.proj_out":
        return (4096, 16384)
    if tail in {
        *(
            f"{name}.{projection}"
            for name in ("audio_attn1", "audio_attn2")
            for projection in attention_projections
        ),
    }:
        return (2048, 2048)
    if tail == "audio_ff.proj_in":
        return (8192, 2048)
    if tail == "audio_ff.proj_out":
        return (2048, 8192)
    cross_shapes = {
        "audio_to_video_attn.to_q": (2048, 4096),
        "audio_to_video_attn.to_k": (2048, 2048),
        "audio_to_video_attn.to_v": (2048, 2048),
        "audio_to_video_attn.to_out": (4096, 2048),
        "video_to_audio_attn.to_q": (2048, 2048),
        "video_to_audio_attn.to_k": (2048, 4096),
        "video_to_audio_attn.to_v": (2048, 4096),
        "video_to_audio_attn.to_out": (2048, 2048),
    }
    return cross_shapes.get(tail)


def remap_comfy_transformer_key(key: str) -> str | None:
    """Map an official ComfyUI LTX transformer key to the MLX module tree.

    The official checkpoint stores the text connectors beside the DiT. They
    are excluded here and loaded with the Gemma feature extractor so staged
    unloading can release the language model before sampling.
    """
    for prefix in (
        "base_model.model.model.diffusion_model.",
        "base_model.model.diffusion_model.",
        "base_model.model.transformer.",
        "base_model.model.",
        "model.diffusion_model.",
        "diffusion_model.",
        "transformer.",
    ):
        if key.startswith(prefix):
            key = key.removeprefix(prefix)
            break
    if key.startswith(("video_embeddings_connector.", "audio_embeddings_connector.")):
        return None
    replacements = (
        (".to_out.0.", ".to_out."),
        (".ff.net.0.proj.", ".ff.proj_in."),
        (".ff.net.2.", ".ff.proj_out."),
        (".audio_ff.net.0.proj.", ".audio_ff.proj_in."),
        (".audio_ff.net.2.", ".audio_ff.proj_out."),
        (".linear_1.", ".linear1."),
        (".linear_2.", ".linear2."),
    )
    for source, target in replacements:
        key = key.replace(source, target)
    return key


def remap_comfy_transformer_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap and split official ComfyUI transformer tensors without copying."""
    mapped: dict[str, mx.array] = {}
    for key, value in weights.items():
        mapped_key = remap_comfy_transformer_key(key)
        if mapped_key is not None:
            mapped[mapped_key] = value
    return mapped


def _ltx25_lora_pair_scale(
    pair: dict[str, Any],
    report: dict[str, Any],
    weights: dict[str, mx.array],
) -> float:
    """Resolve one adapter pair's baked scale without guessing exporter semantics."""
    scale = 1.0
    if pair.get("alpha_tensor"):
        scale = float(weights[pair["alpha_tensor"]].item()) / int(pair["rank"])
    elif report.get("lora_alpha") is not None:
        declared_rank = report.get("lora_rank") or pair["rank"]
        scale = float(report["lora_alpha"]) / float(declared_rank)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError(f"Invalid LTX 2.5 LoRA scaling for target: {pair['target']}")
    return scale


def _remap_comfy_lora_weights(
    weights: dict[str, mx.array],
    *,
    contract: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
) -> dict[str, mx.array]:
    """Normalize supported source pair schemas to the MLX loader's A/B layout."""
    if contract is None:
        suffixes = (
            (".lora_A.turbo.weight", "a"),
            (".lora_B.turbo.weight", "b"),
            (".lora_A.default.weight", "a"),
            (".lora_B.default.weight", "b"),
            (".lora_A.weight", "a"),
            (".lora_B.weight", "b"),
            (".lora_down.weight", "a"),
            (".lora_up.weight", "b"),
            (".lora_a.weight", "a"),
            (".lora_b.weight", "b"),
        )
        groups: dict[str, dict[str, str]] = {}
        for key in weights:
            for suffix, side in suffixes:
                if key.endswith(suffix):
                    groups.setdefault(key[: -len(suffix)], {})[side] = key
                    break
        pairs = [
            {
                "target": target,
                "normalized_target": _normalize_ltx25_adapter_target(target),
                "a": group["a"],
                "b": group["b"],
                "rank": int(weights[group["a"]].shape[0]),
                "alpha_tensor": None,
            }
            for target, group in sorted(groups.items())
            if set(group) == {"a", "b"}
        ]
    else:
        pairs = contract["pairs"]
    mapped: dict[str, mx.array] = {}
    for pair in pairs:
        target = pair["normalized_target"]
        if target is None:
            continue
        scale = (
            _ltx25_lora_pair_scale(pair, report, weights)
            if report is not None
            else 1.0
        )
        mapped[target + ".lora_A.weight"] = weights[pair["a"]]
        b = weights[pair["b"]]
        mapped[target + ".lora_B.weight"] = b if scale == 1.0 else b * scale
    return mapped


class _NormalizedLTX25BlockLoraSource:
    """Lazy normalized adapter source for the low-RAM block streamer.

    The safetensors payload remains memory mapped. Only pairs for the block
    currently being bound are exposed to the upstream fusion hook, and source
    alpha conventions are applied to B without writing a converted checkpoint.
    """

    def __init__(
        self,
        path: Path,
        *,
        strength: float,
        report: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.strength = float(strength)
        self.report = report if report is not None else inspect_ltx25_lora(self.path)
        self.contract = _inspect_ltx25_adapter_contract(self.path)
        self._weights = dict(mx.load(str(self.path)))
        self._block_pairs: dict[int, list[tuple[str, dict[str, Any]]]] = {}
        self._non_block_pairs: list[tuple[str, dict[str, Any]]] = []
        for pair in self.contract["pairs"]:
            target = pair["normalized_target"]
            match = re.fullmatch(r"transformer_blocks\.(\d+)\.(.+)", target)
            if match is None:
                self._non_block_pairs.append((target, pair))
            else:
                self._block_pairs.setdefault(int(match.group(1)), []).append(
                    (match.group(2), pair)
                )

    def has_block(self, block_idx: int) -> bool:
        return bool(self._block_pairs.get(block_idx))

    def _normalized_pair(self, target: str, pair: dict[str, Any]) -> dict[str, mx.array]:
        scale = _ltx25_lora_pair_scale(pair, self.report, self._weights)
        a = self._weights[pair["a"]]
        b = self._weights[pair["b"]]
        return {
            target + ".lora_A.weight": a,
            target + ".lora_B.weight": b if scale == 1.0 else b * scale,
        }

    def get_block_lora_dict(self, block_idx: int) -> dict[str, mx.array]:
        result: dict[str, mx.array] = {}
        for target, pair in self._block_pairs.get(block_idx, ()):
            result.update(self._normalized_pair(target, pair))
        return result

    def get_non_block_lora_dict(self) -> dict[str, mx.array]:
        result: dict[str, mx.array] = {}
        for target, pair in self._non_block_pairs:
            result.update(self._normalized_pair(target, pair))
        return result

    def validate_targets(self, streamer, fixed_weights: dict[str, mx.array]) -> None:
        """Fail before sampling if any normalized target is absent from the base."""
        missing = []
        for index, pairs in self._block_pairs.items():
            try:
                available = set(streamer.block_keys(index))
            except KeyError:
                missing.extend(
                    f"transformer_blocks.{index}.{target}.weight" for target, _ in pairs
                )
                continue
            missing.extend(
                f"transformer_blocks.{index}.{target}.weight"
                for target, _ in pairs
                if target + ".weight" not in available
            )
        missing.extend(
            target + ".weight"
            for target, _ in self._non_block_pairs
            if target + ".weight" not in fixed_weights
        )
        if missing:
            raise ValueError(
                "LTX 2.5 LoRA targets are not present in the selected transformer: "
                + ", ".join(sorted(missing)[:3])
            )

    def close(self) -> None:
        self._weights = {}
        self._block_pairs = {}
        self._non_block_pairs = []


def _load_normalized_ltx25_lora(path: Path) -> tuple[dict[str, mx.array], dict[str, Any]]:
    report = inspect_ltx25_lora(path)
    contract = _inspect_ltx25_adapter_contract(path)
    weights = dict(mx.load(str(path)))
    return _remap_comfy_lora_weights(weights, contract=contract, report=report), report


def _fuse_non_block_loras(
    weights: dict[str, mx.array],
    loaded_loras: list[tuple[dict[str, mx.array], float]],
) -> list[tuple[str, mx.array]]:
    """Fuse LoRA targets outside transformer blocks without materializing block deltas."""
    from ltx_core_mlx.loader.fuse_loras import apply_loras
    from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict

    non_block_weights = {
        key: value for key, value in weights.items() if not key.startswith("transformer_blocks.")
    }
    non_block_loras = []
    for remapped, strength in loaded_loras:
        values = {
            key: value
            for key, value in remapped.items()
            if not key.startswith("transformer_blocks.")
        }
        if values:
            non_block_loras.append(
                LoraStateDictWithStrength(
                    StateDict(sd=values, size=0, dtype=set()),
                    strength,
                )
            )
    fused = apply_loras(
        StateDict(sd=non_block_weights, size=0, dtype=set()),
        non_block_loras,
    )
    return list(fused.sd.items())


def _load_resident_transformer_with_loras(
    model,
    weights: dict[str, mx.array],
    loras: tuple[tuple[Path, float], ...],
) -> None:
    """Fuse adapters one block at a time to bound temporary unified memory."""
    from ltx_core_mlx.loader.fuse_loras import apply_loras
    from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict

    loaded_loras = []
    for path, strength in loras:
        remapped, _report = _load_normalized_ltx25_lora(path)
        loaded_loras.append((remapped, float(strength)))
    for remapped, _strength in loaded_loras:
        requested_weights = {
            key.removesuffix(".lora_A.weight") + ".weight"
            for key in remapped
            if key.endswith(".lora_A.weight")
        }
        missing = sorted(requested_weights - set(weights))
        if missing:
            raise ValueError(
                "LTX 2.5 LoRA targets are not present in the selected transformer: "
                + ", ".join(missing[:3])
            )

    model.load_weights(_fuse_non_block_loras(weights, loaded_loras), strict=False)

    for index, block in enumerate(model.transformer_blocks):
        prefix = f"transformer_blocks.{index}."
        block_weights = {
            key.removeprefix(prefix): value
            for key, value in weights.items()
            if key.startswith(prefix)
        }
        if not block_weights:
            raise ValueError(f"LTX 2.5 transformer block {index} has no checkpoint weights.")
        block_loras = []
        for remapped, strength in loaded_loras:
            values = {
                key.removeprefix(prefix): value
                for key, value in remapped.items()
                if key.startswith(prefix)
            }
            if values:
                block_loras.append(
                    LoraStateDictWithStrength(
                        StateDict(sd=values, size=0, dtype=set()),
                        strength,
                    )
                )
        fused = apply_loras(
            StateDict(sd=block_weights, size=0, dtype=set()),
            block_loras,
        )
        block.load_weights(list(fused.sd.items()), strict=True)
        mx.eval(block.parameters())

    loaded_loras.clear()
    mx.eval(model.parameters())


class _OfficialComfyBlockStreamer:
    """Adapt official Comfy block names to the existing MLX streaming API."""

    def __new__(cls, path, *, paged_manifest=None):
        from ltx_core_mlx.loader.block_streaming import BlockStreamer

        prefix = "model.diffusion_model.transformer_blocks."
        streamer = BlockStreamer(path, block_prefix=prefix)
        remapped: dict[int, list[tuple[str, str]]] = {}
        for index, entries in streamer._block_key_map.items():
            converted = []
            for full_key, _relative_key in entries:
                mapped = remap_comfy_transformer_key(full_key)
                block_prefix = f"transformer_blocks.{index}."
                if mapped is None or not mapped.startswith(block_prefix):
                    raise ValueError(f"Could not map streamed LTX 2.5 key: {full_key}")
                converted.append((full_key, mapped.removeprefix(block_prefix)))
            remapped[index] = converted
        streamer._block_key_map = remapped
        if paged_manifest is not None:
            return _PrefetchedBlockStreamer(streamer, paged_manifest)
        return streamer


class _PrefetchedBlockStreamer:
    """Add bounded read-ahead and measurements to the upstream block streamer."""

    def __init__(self, streamer, manifest, *, enabled: bool | None = None):
        from .page_prefetch import LTX25PagePrefetch

        self._streamer = streamer
        self._manifest = manifest
        self._prefetch = LTX25PagePrefetch(
            manifest.root,
            manifest.layers,
            enabled=LTX25PagePrefetch.default_enabled() if enabled is None else enabled,
            thread_name="ltx25-transformer-prefetch",
        )
        self.bind_calls = 0
        self.bind_seconds = 0.0
        self._prefetch.start(0)

    @property
    def block_count(self):
        return self._streamer.block_count

    @property
    def block_prefix(self):
        return self._streamer.block_prefix

    def block_keys(self, index):
        return self._streamer.block_keys(index)

    def bind(self, block, index, evict_previous=None, lora_sources=None):
        import time

        self._prefetch.wait(index)
        started = time.perf_counter()
        self._streamer.bind(
            block,
            index,
            evict_previous=evict_previous,
            lora_sources=lora_sources,
        )
        self.bind_calls += 1
        self.bind_seconds += time.perf_counter() - started
        next_index = (index + 1) % self._manifest.num_layers
        self._prefetch.start(next_index)

    def report(self):
        return {
            "streamed_bind_calls": self.bind_calls,
            "streamed_bind_seconds": self.bind_seconds,
            **self._prefetch.report(),
        }

    def close(self):
        self._prefetch.close()
        self._streamer.close()


_STREAMING_EVAL_LOCK = RLock()


class _StreamingEvalWindow:
    """Delay the upstream streaming barrier until a safe block window ends."""

    def __init__(self, window: int, eval_fn=mx.eval) -> None:
        if window < 1:
            raise ValueError("LTX 2.5 streaming window must be at least one block.")
        self.window = int(window)
        self.eval_fn = eval_fn
        self.calls = 0
        self.flushes = 0
        self._pending = None

    def __call__(self, *arrays):
        self.calls += 1
        self._pending = arrays
        if self.calls % self.window == 0:
            return self.flush()
        return None

    def flush(self):
        if self._pending is None:
            return None
        arrays = self._pending
        self._pending = None
        self.flushes += 1
        return self.eval_fn(*arrays)


def _streaming_window_from_environment(*, paged: bool) -> int:
    raw = os.environ.get("WEETODD_LTX25_STREAMING_WINDOW", "1").strip()
    try:
        window = int(raw)
    except ValueError as exc:
        raise ValueError(
            "WEETODD_LTX25_STREAMING_WINDOW must be the integer 1 or 2."
        ) from exc
    if window not in {1, 2}:
        raise ValueError("WEETODD_LTX25_STREAMING_WINDOW must be the integer 1 or 2.")
    if window > 1 and not paged:
        raise ValueError(
            "LTX 2.5 two-block streaming requires a paged transformer checkpoint."
        )
    return window


class _WindowedStreamingLTXModel(nn.Module):
    """Stream through multiple compiled block slots before one Metal barrier."""

    def __init__(self, model, streamer, *, window: int, lora_sources=None) -> None:
        super().__init__()
        if len(model.transformer_blocks) != window:
            raise ValueError("LTX 2.5 streaming block slots do not match the window size.")
        self.inner = model
        shared_blocks = tuple(model.transformer_blocks)
        compiled_blocks = tuple(mx.compile(block, inputs=block) for block in shared_blocks)
        object.__setattr__(self, "_streamer", streamer)
        object.__setattr__(self, "_shared_blocks", shared_blocks)
        object.__setattr__(self, "_compiled_blocks", compiled_blocks)
        object.__setattr__(self, "_lora_sources", lora_sources or [])
        object.__setattr__(self, "_window", int(window))
        object.__setattr__(self, "_eval_calls", 0)
        object.__setattr__(self, "_eval_flushes", 0)

    def __call__(self, *args, **kwargs):
        if kwargs.get("block_provider") is not None:
            return self.inner(*args, **kwargs)

        from ltx_core_mlx.model.transformer import model as model_module

        streamer = object.__getattribute__(self, "_streamer")
        shared_blocks = object.__getattribute__(self, "_shared_blocks")
        compiled_blocks = object.__getattribute__(self, "_compiled_blocks")
        lora_sources = object.__getattribute__(self, "_lora_sources")
        window = object.__getattribute__(self, "_window")
        previous = [None] * window
        use_compiled = kwargs.get("perturbations") is None

        def provider(index: int):
            slot = index % window
            streamer.bind(
                shared_blocks[slot],
                index,
                evict_previous=previous[slot],
                lora_sources=lora_sources or None,
            )
            previous[slot] = index
            return compiled_blocks[slot] if use_compiled else shared_blocks[slot]

        kwargs["block_provider"] = provider
        with _STREAMING_EVAL_LOCK:
            original_eval = model_module._mx_eval
            gate = _StreamingEvalWindow(window, original_eval)
            model_module._mx_eval = gate
            try:
                result = self.inner(*args, **kwargs)
            finally:
                try:
                    gate.flush()
                finally:
                    model_module._mx_eval = original_eval
                    object.__setattr__(self, "_eval_calls", gate.calls)
                    object.__setattr__(self, "_eval_flushes", gate.flushes)
            return result

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)

    def streaming_window_report(self) -> dict[str, int]:
        return {
            "streaming_window": object.__getattribute__(self, "_window"),
            "streaming_eval_calls": object.__getattribute__(self, "_eval_calls"),
            "streaming_eval_flushes": object.__getattribute__(self, "_eval_flushes"),
        }


def load_ltx25_transformer(
    path: str | Path,
    *,
    low_ram_streaming: bool = False,
    feed_forward_backend: str = "reference_fp32",
    loras: tuple[tuple[str | Path, float], ...] = (),
):
    """Strictly load an LTX 2.5 transformer, including MLX q8/q4 tensors."""
    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors

    source = Path(path).expanduser()
    paged_manifest = None
    if source.is_dir():
        from .paged_checkpoint import LTX25PagedManifest

        paged_manifest = LTX25PagedManifest.load(source)
        if paged_manifest.kind != "transformer":
            raise ValueError(f"Expected a paged transformer, got {paged_manifest.kind!r}.")
        if not low_ram_streaming:
            raise ValueError(
                "Paged LTX 2.5 transformer checkpoints require low_ram_streaming=true."
            )
    resolved_loras = tuple((Path(item).expanduser(), float(strength)) for item, strength in loras)
    lora_reports = [inspect_ltx25_lora(lora_path) for lora_path, _ in resolved_loras]
    config = LTX25TransformerConfig.from_metadata(transformer_metadata(source))
    model = LTX25Model.build(config)
    if paged_manifest is None:
        raw_weights = load_split_safetensors(source)
        block_sources = source
    else:
        raw_weights = dict(mx.load(str(paged_manifest.fixed_path)))
        block_sources = list(paged_manifest.layer_paths)
    weights = remap_comfy_transformer_weights(raw_weights)
    if not weights:
        raise ValueError(f"LTX 2.5 transformer {source} has no recognized weights.")
    if low_ram_streaming:
        from ltx_core_mlx.loader.block_streaming import StreamingLTXModel

        streaming_window = _streaming_window_from_environment(paged=paged_manifest is not None)
        model.transformer_blocks = [model.transformer_blocks[0]]
        quantization_weights = weights
        if paged_manifest is not None:
            first_page = remap_comfy_transformer_weights(
                dict(mx.load(str(paged_manifest.layer_paths[0])))
            )
            quantization_weights = {**weights, **first_page}
        apply_quantization(model, quantization_weights)
        lora_sources = [
            _NormalizedLTX25BlockLoraSource(
                lora_path,
                strength=strength,
                report=report,
            )
            for (lora_path, strength), report in zip(
                resolved_loras, lora_reports, strict=True
            )
        ]
        streamer = _OfficialComfyBlockStreamer(block_sources, paged_manifest=paged_manifest)
        try:
            for lora_source in lora_sources:
                lora_source.validate_targets(streamer, weights)
        except Exception:
            for lora_source in lora_sources:
                lora_source.close()
            streamer.close()
            raise
        loaded_non_block_loras = [
            (lora_source.get_non_block_lora_dict(), lora_source.strength)
            for lora_source in lora_sources
        ]
        model.load_weights(
            _fuse_non_block_loras(weights, loaded_non_block_loras),
            strict=False,
        )
        loaded_non_block_loras.clear()
        if streaming_window > 1:
            first_block = model.transformer_blocks[0]
            model.transformer_blocks = [
                first_block,
                *(copy.deepcopy(first_block) for _ in range(streaming_window - 1)),
            ]
        if streaming_window == 1:
            model = StreamingLTXModel(model, streamer, lora_sources=lora_sources)
        else:
            model = _WindowedStreamingLTXModel(
                model,
                streamer,
                window=streaming_window,
                lora_sources=lora_sources,
            )
        object.__setattr__(model, "_weetodd_streamer", streamer)
        object.__setattr__(model, "_weetodd_lora_sources", tuple(lora_sources))
        if paged_manifest is not None:
            object.__setattr__(
                model,
                "paged_checkpoint_report",
                {
                    "format": paged_manifest.format,
                    "bits": paged_manifest.bits,
                    "group_size": paged_manifest.group_size,
                    "fixed_bytes": paged_manifest.fixed.tensor_bytes,
                    "peak_layer_bytes": max(
                        record.tensor_bytes for record in paged_manifest.layers
                    ),
                    "streaming_window": streaming_window,
                },
            )
            object.__setattr__(
                model,
                "_weetodd_paged_streamer",
                object.__getattribute__(model, "_streamer"),
            )
        mx.eval(model.parameters())
        aggressive_cleanup()
        return model
    apply_quantization(model, weights)
    if resolved_loras:
        _load_resident_transformer_with_loras(model, weights, resolved_loras)
    else:
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
    from .feed_forward import configure_feed_forward_backend

    report = configure_feed_forward_backend(model, feed_forward_backend).to_dict()
    object.__setattr__(model, "feed_forward_backend_report", report)
    aggressive_cleanup()
    return model


__all__ = [
    "LTX25Model",
    "LTX25TransformerConfig",
    "inspect_ltx25_ic_lora",
    "inspect_ltx25_lora",
    "inspect_ltx25_msr_lora",
    "load_ltx25_transformer",
    "remap_comfy_transformer_key",
    "remap_comfy_transformer_weights",
    "precompute_rope_freqs_float64",
    "transformer_metadata",
]
