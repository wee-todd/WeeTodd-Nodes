"""Independent MLX conditioning for LTX 2.5 multi-subject reference adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from .ic_lora import _host_video, _resize_center_crop, plan_ingredients_reference_grid


@dataclass(frozen=True)
class LTX25MSRReferenceReport:
    label: str
    role: str
    slot_id: int
    negative_time_offset: int
    source_width: int
    source_height: int
    encoded_width: int
    encoded_height: int
    encoded_frames: int
    token_count: int
    strength: float
    attention_strength: float
    reference_size_policy: str

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def load_ltx25_msr_slot_state(path: str | Path) -> dict[str, mx.array]:
    """Load only the five small learned-slot tensors from an MSR checkpoint."""

    from .transformer import inspect_ltx25_msr_lora

    report = inspect_ltx25_msr_lora(path)
    prefix = str(report["slot_prefix"])
    # ``safetensors.numpy`` cannot materialize BF16 tensors. MLX's native
    # loader memory-maps the checkpoint lazily, so selecting these five small
    # arrays does not allocate or evaluate the adapter's 960 LoRA tensors.
    checkpoint_state = mx.load(Path(path))
    if not isinstance(checkpoint_state, dict):
        raise ValueError("LTX 2.5 MSR slot checkpoint must contain named tensors.")
    state = {
        name: checkpoint_state[prefix + name]
        for name in report["slot_shapes"]
    }
    mx.eval(*state.values())
    return state


def ltx25_msr_slot_embedding(slot_id: int, state: dict[str, mx.array]) -> mx.array:
    """Evaluate the checkpoint's learned Fourier MLP for one one-based slot."""

    if not 1 <= int(slot_id) <= 5:
        raise ValueError("LTX 2.5 MSR slot IDs must be between one and five.")
    scaled = mx.array([float(slot_id) / 16.0], dtype=mx.float32)
    frequencies = state["frequencies"].astype(mx.float32)
    phases = scaled[0] * frequencies
    features = mx.concatenate([scaled, mx.sin(phases), mx.cos(phases)])
    hidden = features @ state["net.0.weight"].astype(mx.float32).T
    hidden = hidden + state["net.0.bias"].astype(mx.float32)
    hidden = hidden * mx.sigmoid(hidden)
    output = hidden @ state["net.2.weight"].astype(mx.float32).T
    return output + state["net.2.bias"].astype(mx.float32)


def _resize_fit_white(video, height: int, width: int):
    """Fit a complete subject/object reference on a white canvas."""

    import numpy as np
    from PIL import Image

    source_h, source_w = int(video.shape[1]), int(video.shape[2])
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, min(width, round(source_w * scale)))
    resized_h = max(1, min(height, round(source_h * scale)))
    left = (width - resized_w) // 2
    top = (height - resized_h) // 2
    output = np.ones((video.shape[0], height, width, 3), dtype=np.float32)
    for index, frame in enumerate(video):
        image = Image.fromarray((frame * 255.0).round().astype(np.uint8), mode="RGB")
        image = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        output[index, top : top + resized_h, left : left + resized_w] = (
            np.asarray(image, dtype=np.float32) / 255.0
        )
    return output


def plan_ltx25_msr_reference_grid(
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
    policy: str,
) -> tuple[int, int]:
    """Resolve native-size quality or bounded experimental MSR reference grids."""

    if policy == "quality":
        return target_height, target_width
    return plan_ingredients_reference_grid(
        source_height=source_height,
        source_width=source_width,
        target_height=target_height,
        target_width=target_width,
        policy=policy,
    )


class LTX25MSRConditioning:
    """Append independently encoded MSR groups with a compact grouped mask."""

    def __init__(
        self,
        groups: tuple[dict[str, object], ...],
        *,
        compact_attention_mask: bool,
    ) -> None:
        self.groups = groups
        self.group_rows = tuple(int(group["tokens"].shape[1]) for group in groups)
        self.compact_attention_mask = bool(compact_attention_mask)

    def apply(self, state, spatial_dims: tuple[int, int, int]):
        from ltx_core_mlx.conditioning.types.latent_cond import LatentState

        from wee_todd_mlx.sol_attention import materialize_compact_attention_mask

        target_rows = int(spatial_dims[0] * spatial_dims[1] * spatial_dims[2])
        if int(state.latent.shape[1]) != target_rows or state.attention_mask is not None:
            raise ValueError(
                "LTX 2.5 MSR currently requires an unmasked one-shot target before its "
                "reference groups are appended."
            )
        batch = int(state.latent.shape[0])
        dtype = state.latent.dtype
        latents = [state.latent]
        clean = [state.clean_latent]
        masks = [state.denoise_mask]
        positions = [state.positions] if state.positions is not None else []
        for group in self.groups:
            tokens = group["tokens"].astype(dtype)
            latents.append(tokens)
            clean.append(tokens)
            masks.append(
                mx.full(
                    (batch, int(tokens.shape[1]), 1),
                    1.0 - float(group["strength"]),
                    dtype=dtype,
                )
            )
            if positions:
                positions.append(group["positions"])

        segment_rows = (target_rows, *self.group_rows)
        total_rows = sum(segment_rows)
        templates = []
        target_parts = [mx.ones((batch, target_rows), dtype=dtype)]
        target_parts.extend(
            mx.full(
                (batch, rows),
                float(group["attention_strength"]),
                dtype=dtype,
            )
            for rows, group in zip(self.group_rows, self.groups, strict=True)
        )
        templates.append(mx.concatenate(target_parts, axis=1))
        for group in self.groups:
            parts = [
                mx.full(
                    (batch, target_rows),
                    float(group["attention_strength"]),
                    dtype=dtype,
                )
            ]
            for rows in self.group_rows:
                # Learned slots and negative temporal positions separate the references.
                # The published guide-mask contract leaves guide-to-guide attention intact.
                parts.append(mx.ones((batch, rows), dtype=dtype))
            templates.append(mx.concatenate(parts, axis=1))
        compact = mx.stack(templates, axis=1)
        if compact.shape != (batch, len(self.groups) + 1, total_rows):
            raise RuntimeError("LTX 2.5 MSR grouped attention mask has an invalid shape.")
        attention_mask = (
            compact
            if self.compact_attention_mask
            else materialize_compact_attention_mask(compact, self.group_rows)
        )
        return LatentState(
            latent=mx.concatenate(latents, axis=1),
            clean_latent=mx.concatenate(clean, axis=1),
            denoise_mask=mx.concatenate(masks, axis=1),
            positions=mx.concatenate(positions, axis=1) if positions else None,
            attention_mask=attention_mask,
        )


def encode_ltx25_msr_references(
    *,
    references: tuple[dict[str, Any], ...],
    video_encoder,
    slot_checkpoint: str | Path,
    target_height: int,
    target_width: int,
    frame_rate: float,
    compact_attention_mask: bool,
) -> tuple[LTX25MSRConditioning, list[LTX25MSRReferenceReport]]:
    """Encode one to five still references and attach learned slot identities."""

    from ltx_core_mlx.utils.positions import compute_video_positions

    if not 1 <= len(references) <= 5:
        raise ValueError("LTX 2.5 MSR requires one to five reference images.")
    slot_state = load_ltx25_msr_slot_state(slot_checkpoint)
    groups = []
    reports = []
    count = len(references)
    for index, reference in enumerate(references):
        image = _host_video(reference["image"])
        if int(image.shape[0]) != 1:
            raise ValueError("Each LTX 2.5 MSR reference must contain exactly one image.")
        policy = str(reference.get("reference_size_policy", "quality"))
        height, width = plan_ltx25_msr_reference_grid(
            source_height=int(image.shape[1]),
            source_width=int(image.shape[2]),
            target_height=target_height,
            target_width=target_width,
            policy=policy,
        )
        role = str(reference.get("role", "subject"))
        resized = (
            _resize_center_crop(image, height, width)
            if role == "background"
            else _resize_fit_white(image, height, width)
        )
        reference_frames = int(reference.get("reference_frames", 33))
        if reference_frames not in {25, 33}:
            raise ValueError("LTX 2.5 MSR reference_frames must be 25 or 33.")
        pixels = mx.array(resized.transpose(3, 0, 1, 2)[None])
        pixels = mx.broadcast_to(
            pixels,
            (1, pixels.shape[1], reference_frames, pixels.shape[3], pixels.shape[4]),
        )
        encoded = video_encoder.encode((pixels * 2.0 - 1.0).astype(mx.bfloat16))
        slot_id = index + 1
        slot = ltx25_msr_slot_embedding(slot_id, slot_state).astype(encoded.dtype)
        encoded = encoded + slot[None, :, None, None, None]
        mx.eval(encoded)
        latent_f, latent_h, latent_w = map(int, encoded.shape[2:])
        tokens = encoded.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
        positions = compute_video_positions(
            latent_f,
            latent_h,
            latent_w,
            frame_rate=frame_rate,
        )
        negative_offset = -(count - index)
        positions = mx.concatenate(
            [positions[..., :1] + negative_offset / frame_rate, positions[..., 1:]],
            axis=-1,
        )
        groups.append(
            {
                "tokens": tokens,
                "positions": positions,
                "strength": float(reference.get("strength", 1.0)),
                "attention_strength": float(reference.get("attention_strength", 1.0)),
            }
        )
        reports.append(
            LTX25MSRReferenceReport(
                label=f"Image {slot_id}",
                role=role,
                slot_id=slot_id,
                negative_time_offset=negative_offset,
                source_width=int(image.shape[2]),
                source_height=int(image.shape[1]),
                encoded_width=width,
                encoded_height=height,
                encoded_frames=reference_frames,
                token_count=int(tokens.shape[1]),
                strength=float(reference.get("strength", 1.0)),
                attention_strength=float(reference.get("attention_strength", 1.0)),
                reference_size_policy=policy,
            )
        )
    slot_state.clear()
    return (
        LTX25MSRConditioning(tuple(groups), compact_attention_mask=compact_attention_mask),
        reports,
    )


__all__ = [
    "LTX25MSRConditioning",
    "LTX25MSRReferenceReport",
    "encode_ltx25_msr_references",
    "load_ltx25_msr_slot_state",
    "ltx25_msr_slot_embedding",
    "plan_ltx25_msr_reference_grid",
]
