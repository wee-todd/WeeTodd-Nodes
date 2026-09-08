"""MLX generated-keyframe slots for LTX 2.5 keyframe-capable checkpoints."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def evenly_spaced_keyframe_positions(count: int, num_frames: int) -> tuple[int, ...]:
    """Return unique rounded interior pixel-frame positions."""
    if count < 0:
        raise ValueError("Generated keyframe count must not be negative.")
    if count == 0:
        return ()
    if num_frames < count + 2:
        raise ValueError("Generated keyframes require at least count + 2 output frames.")
    values = mx.linspace(0, num_frames - 1, count + 2)[1:-1]
    return tuple(int(value) for value in mx.round(values).tolist())


class GeneratedKeyframeSlots:
    """Append empty, fully denoised slots at exact target pixel-frame positions."""

    def __init__(
        self,
        pixel_frames: tuple[int, ...],
        *,
        spatial_dims: tuple[int, int, int],
        frame_rate: float,
        initial_keyframes: mx.array | None = None,
    ) -> None:
        if not pixel_frames or tuple(sorted(set(pixel_frames))) != pixel_frames:
            raise ValueError("Generated keyframe frames must be non-empty and strictly increasing.")
        self.pixel_frames = pixel_frames
        self.spatial_dims = spatial_dims
        self.frame_rate = frame_rate
        if initial_keyframes is not None:
            if initial_keyframes.ndim != 5:
                raise ValueError("Initial generated keyframes must use (B, C, K, H, W).")
            if initial_keyframes.shape[2] != len(pixel_frames):
                raise ValueError("Initial generated keyframe count does not match pixel frames.")
            if tuple(initial_keyframes.shape[3:]) != tuple(spatial_dims[1:]):
                raise ValueError("Initial generated keyframe geometry does not match the target.")
        self.initial_keyframes = initial_keyframes

    @property
    def token_count(self) -> int:
        return len(self.pixel_frames) * self.spatial_dims[1] * self.spatial_dims[2]

    def apply(self, state, spatial_dims):
        from ltx_core_mlx.conditioning.mask_utils import update_attention_mask
        from ltx_core_mlx.conditioning.types.keyframe_cond import _compute_keyframe_positions
        from ltx_core_mlx.conditioning.types.latent_cond import LatentState

        if tuple(spatial_dims) != tuple(self.spatial_dims):
            raise ValueError("Generated keyframe geometry does not match the target latent grid.")
        batch, _rows, channels = state.latent.shape
        positions = mx.concatenate(
            [
                _compute_keyframe_positions(
                    frame,
                    self.spatial_dims[1],
                    self.spatial_dims[2],
                    self.frame_rate,
                    num_pixel_frames=1,
                )
                for frame in self.pixel_frames
            ],
            axis=1,
        )
        if self.initial_keyframes is None:
            slots = mx.zeros((batch, self.token_count, channels), dtype=state.latent.dtype)
        else:
            from ltx_core_mlx.components.patchifiers import VideoLatentPatchifier

            slots, spatial = VideoLatentPatchifier().patchify(
                self.initial_keyframes.astype(state.latent.dtype)
            )
            if tuple(spatial) != (
                len(self.pixel_frames),
                self.spatial_dims[1],
                self.spatial_dims[2],
            ):
                raise ValueError("Initial generated keyframes produced an invalid token layout.")
        denoise = mx.ones((batch, self.token_count, 1), dtype=state.denoise_mask.dtype)
        attention = update_attention_mask(
            latent_state=state,
            attention_mask=None,
            num_noisy_tokens=spatial_dims[0] * spatial_dims[1] * spatial_dims[2],
            num_new_tokens=self.token_count,
            batch_size=batch,
        )
        return LatentState(
            latent=mx.concatenate([state.latent, slots], axis=1),
            clean_latent=mx.concatenate([state.clean_latent, mx.zeros_like(slots)], axis=1),
            denoise_mask=mx.concatenate([state.denoise_mask, denoise], axis=1),
            positions=(
                mx.concatenate([state.positions, positions], axis=1)
                if state.positions is not None
                else None
            ),
            attention_mask=attention,
        )


class _MarkedProjection(nn.Module):
    def __init__(self, projection, marker) -> None:
        super().__init__()
        self.projection = projection
        self.marker = marker
        self.marked_rows = 0

    def __call__(self, value):
        projected = self.projection(value)
        if self.marked_rows:
            split = projected.shape[1] - self.marked_rows
            projected = mx.concatenate(
                [projected[:, :split], projected[:, split:] + self.marker], axis=1
            )
        return projected


def _projection_owner(model):
    """Resolve the module that owns ``patchify_proj`` through streaming wrappers."""
    current = model
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        inner = getattr(current, "inner", None)
        if inner is not None and inner is not current:
            current = inner
            continue
        if hasattr(current, "patchify_proj"):
            return current
        break
    raise ValueError("The selected LTX 2.5 transformer has no writable video projection.")


def set_generated_keyframe_marker(model, marked_rows: int) -> None:
    """Mark the appended slot rows with the checkpoint's learned absolute embedding."""
    owner = _projection_owner(model)
    projection = owner.patchify_proj
    if not isinstance(projection, _MarkedProjection):
        marker = getattr(model, "keyframes_abs_pos_embedding", None)
        if marker is None:
            raise ValueError("The selected LTX 2.5 checkpoint has no keyframe-slot embedding.")
        projection = _MarkedProjection(projection, marker)
        owner.patchify_proj = projection
    projection.marked_rows = int(marked_rows)


__all__ = [
    "GeneratedKeyframeSlots",
    "evenly_spaced_keyframe_positions",
    "set_generated_keyframe_marker",
]
