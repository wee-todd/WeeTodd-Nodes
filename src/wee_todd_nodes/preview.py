"""Sampler-side MiniMax H3 live-preview configuration and lifecycle."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PREVIEW_GUARD_MODES = (
    "preview only",
    "abort non-finite",
    "conservative collapse guard",
)
PREVIEW_BACKENDS = ("auto", "neural engine", "mlx")


@dataclass(frozen=True)
class H3PreviewConfig:
    tae_path: str
    backend: str = "auto"
    coreml_model_path: str | None = None
    every_n_evaluations: int = 1
    preview_frames: int = 6
    max_edge: int = 384
    guard_mode: str = "conservative collapse guard"
    collapse_start_fraction: float = 0.5
    collapse_patience: int = 2

    def validate(self) -> None:
        path = Path(self.tae_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MiniMax H3 preview TAE not found: {path}")
        if self.every_n_evaluations < 1:
            raise ValueError("Preview interval must be at least one evaluation.")
        if not 1 <= self.preview_frames <= 12:
            raise ValueError("Preview frames must be between 1 and 12.")
        if not 64 <= self.max_edge <= 1024:
            raise ValueError("Preview maximum edge must be between 64 and 1024 pixels.")
        if self.guard_mode not in PREVIEW_GUARD_MODES:
            raise ValueError(f"Unknown H3 preview guard mode: {self.guard_mode!r}.")
        if self.backend not in PREVIEW_BACKENDS:
            raise ValueError(f"Unknown H3 preview backend: {self.backend!r}.")
        if self.backend == "neural engine" and not self.coreml_model_path:
            raise ValueError("The Neural Engine preview backend requires a Core ML model path.")
        if (
            self.backend == "neural engine"
            and self.coreml_model_path
            and not Path(self.coreml_model_path).expanduser().exists()
        ):
            raise FileNotFoundError(
                f"MiniMax H3 Core ML preview model not found: {self.coreml_model_path}"
            )


@dataclass(frozen=True)
class H3PreviewUpdate:
    frames: Any
    statistics: Any
    reject_reason: str | None = None


class H3PreviewSession:
    """Own one tiny decoder for exactly one sampler execution."""

    def __init__(self, config: H3PreviewConfig):
        config.validate()
        self.config = config
        self.backend = "mlx"
        self.fallback_reason: str | None = None
        self.decoder = self._load_decoder()
        self._collapsed_checkpoints = 0

    def _load_decoder(self):
        if self.config.backend in {"auto", "neural engine"} and self.config.coreml_model_path:
            try:
                from minimax_h3_mlx.coreml_preview import H3CoreMLPreviewDecoder

                decoder = H3CoreMLPreviewDecoder(self.config.coreml_model_path)
                self.backend = "neural engine"
                return decoder
            except (ImportError, RuntimeError, ValueError, OSError) as exc:
                if self.config.backend == "neural engine":
                    raise
                self.fallback_reason = str(exc)
        from minimax_h3_mlx.tae_preview import H3TinyPreviewDecoder

        return H3TinyPreviewDecoder.from_safetensors(self.config.tae_path)

    def update(self, latents, completed: int, total: int) -> H3PreviewUpdate | None:
        should_decode = (
            completed == total or completed == 1 or completed % self.config.every_n_evaluations == 0
        )
        if not should_decode:
            return None
        import mlx.core as mx
        import numpy as np

        finite = bool(mx.all(mx.isfinite(latents)).item())
        if not finite:
            reason = "H3 sampling stopped because the predicted video latent became non-finite."
            return H3PreviewUpdate(None, None, reason)

        # Ten consecutive latent frames provide about one second of causal visual context
        # while keeping preview activation memory bounded independently of requested duration.
        count = min(int(latents.shape[2]), 10)
        count -= count % 5
        if count < 5:
            count = min(int(latents.shape[2]), 5)
        preview_latents = latents[:, :, :count]
        if int(preview_latents.shape[2]) < 5:
            padding = 5 - int(preview_latents.shape[2])
            preview_latents = mx.concatenate(
                [preview_latents, mx.repeat(preview_latents[:, :, -1:], padding, axis=2)],
                axis=2,
            )
        if self.backend == "neural engine" and int(preview_latents.shape[2]) < 10:
            padding = 10 - int(preview_latents.shape[2])
            preview_latents = mx.concatenate(
                [preview_latents, mx.repeat(preview_latents[:, :, -1:], padding, axis=2)],
                axis=2,
            )
        decoded = self.decoder.decode(preview_latents, max_edge=self.config.max_edge)
        if decoded.shape[0] > self.config.preview_frames:
            indices = (
                np.linspace(0, decoded.shape[0] - 1, self.config.preview_frames)
                .round()
                .astype(np.int32)
            )
            decoded = decoded[indices]

        from minimax_h3_mlx.tae_preview import preview_statistics

        statistics = preview_statistics(decoded)
        reason = None
        progress = completed / max(total, 1)
        if self.config.guard_mode == "abort non-finite" and not statistics.finite:
            reason = "H3 sampling stopped because the TAE preview contained non-finite pixels."
        elif self.config.guard_mode == "conservative collapse guard":
            if not statistics.finite:
                reason = "H3 sampling stopped because the TAE preview contained non-finite pixels."
            elif progress >= self.config.collapse_start_fraction and statistics.collapsed:
                self._collapsed_checkpoints += 1
            else:
                self._collapsed_checkpoints = 0
            if self._collapsed_checkpoints >= self.config.collapse_patience:
                reason = (
                    "H3 sampling stopped after multiple featureless TAE previews. The predicted "
                    "video remained structurally collapsed after the midpoint of the schedule."
                )
        return H3PreviewUpdate(decoded, statistics, reason)

    def release(self) -> None:
        self.decoder.release()
        self.decoder = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass
