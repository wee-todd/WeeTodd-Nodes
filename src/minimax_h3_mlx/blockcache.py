"""Request-local transformer block reuse for MLX MiniMax H3."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class H3BlockCacheConfig:
    """Controls for reusing blocks 1..N while always evaluating block zero and output heads."""

    mode: str = "automatic_balanced"
    reuse_threshold: float = 0.12
    start_percent: float = 0.15
    end_percent: float = 0.95
    subsample_factor: int = 8
    auto_multiplier: float = 1.4
    max_hit_fraction: float = 0.35
    allow_turbo_experimental: bool = False

    def validate(self) -> None:
        modes = {
            "manual",
            "automatic_conservative",
            "automatic_balanced",
            "automatic_speed",
        }
        if self.mode not in modes:
            raise ValueError(f"BlockCache mode must be one of {sorted(modes)}, got {self.mode!r}.")
        if self.reuse_threshold < 0:
            raise ValueError("BlockCache reuse threshold must be zero or positive.")
        if not 0 <= self.start_percent <= self.end_percent <= 1:
            raise ValueError("BlockCache requires 0 <= start_percent <= end_percent <= 1.")
        if self.subsample_factor < 1:
            raise ValueError("BlockCache subsample factor must be positive.")
        if self.auto_multiplier < 1:
            raise ValueError("BlockCache automatic multiplier must be at least 1.0.")
        if not 0 <= self.max_hit_fraction <= 0.6:
            raise ValueError("BlockCache maximum hit fraction must be between 0 and 0.6.")


class H3BlockCacheState:
    """Cache the post-block-zero to post-stack residual for video and audio target rows."""

    def __init__(self, config: H3BlockCacheConfig) -> None:
        config.validate()
        self.config = config
        self.previous_video_indicator = None
        self.previous_audio_indicator = None
        self.video_tail = None
        self.audio_tail = None
        self.resolved_threshold = config.reuse_threshold if config.mode == "manual" else None
        self.hits = 0
        self.full_evaluations = 0
        self.consecutive_hits = 0
        self.last_video_score: float | None = None
        self.last_audio_score: float | None = None
        self.last_was_hit = False

    @property
    def _automatic(self) -> bool:
        return self.config.mode != "manual"

    @staticmethod
    def _relative_change(current, previous) -> float:
        import mlx.core as mx

        numerator = mx.mean(mx.abs(current.astype(mx.float32) - previous.astype(mx.float32)))
        denominator = mx.maximum(mx.mean(mx.abs(previous.astype(mx.float32))), 1e-12)
        return float((numerator / denominator).item())

    def _sample(self, value):
        stride = self.config.subsample_factor
        return value[:, ::stride, ::stride]

    def _in_window(self, index: int, total_steps: int) -> bool:
        progress = index / max(total_steps - 1, 1)
        return self.config.start_percent <= progress <= self.config.end_percent

    def _hit_limit(self, total_steps: int) -> int:
        fraction = self.config.max_hit_fraction
        if self.config.mode == "automatic_conservative":
            fraction = min(fraction, 0.25)
        elif self.config.mode == "automatic_balanced":
            fraction = max(fraction, 0.35)
        elif self.config.mode == "automatic_speed":
            fraction = max(fraction, 0.5)
        return math.floor(total_steps * fraction)

    def _resolve_threshold(self, observed: float) -> float:
        if self.config.mode == "automatic_speed":
            return min(0.6, max(0.12, observed * max(self.config.auto_multiplier, 1.8)))
        if self.config.mode == "automatic_balanced":
            return min(0.35, max(0.07, observed * max(self.config.auto_multiplier, 1.4)))
        return min(0.2, max(0.035, observed * self.config.auto_multiplier))

    def try_reuse(
        self, before_block_zero, after_block_zero, video_indices, audio_indices, index, total_steps
    ):
        """Return a reconstructed post-stack hidden state, or ``None`` for a full evaluation."""
        self.last_was_hit = False
        if not self._in_window(index, total_steps) or index >= total_steps - 1:
            return None
        if self.video_tail is None or self.audio_tail is None:
            return None

        video_indicator = self._sample(
            after_block_zero[:, video_indices] - before_block_zero[:, video_indices]
        )
        audio_indicator = self._sample(
            after_block_zero[:, audio_indices] - before_block_zero[:, audio_indices]
        )
        if self.previous_video_indicator is None or self.previous_audio_indicator is None:
            return None
        self.last_video_score = self._relative_change(
            video_indicator, self.previous_video_indicator
        )
        self.last_audio_score = self._relative_change(
            audio_indicator, self.previous_audio_indicator
        )
        score = max(self.last_video_score, self.last_audio_score)
        if self._automatic and self.resolved_threshold is None:
            self.resolved_threshold = self._resolve_threshold(score)
        consecutive_limit = 2 if self.config.mode == "automatic_speed" else 1
        if (
            score > (self.resolved_threshold or 0.0)
            or self.consecutive_hits >= consecutive_limit
            or self.hits >= self._hit_limit(total_steps)
        ):
            return None

        reconstructed = after_block_zero + 0
        reconstructed[:, video_indices] = reconstructed[:, video_indices] + self.video_tail
        reconstructed[:, audio_indices] = reconstructed[:, audio_indices] + self.audio_tail
        self.hits += 1
        self.consecutive_hits += 1
        self.last_was_hit = True
        return reconstructed

    def update(
        self, before_block_zero, after_block_zero, after_stack, video_indices, audio_indices
    ) -> None:
        """Refresh modality indicators and cached block-tail residuals after a full evaluation."""
        import mlx.core as mx

        self.previous_video_indicator = self._sample(
            after_block_zero[:, video_indices] - before_block_zero[:, video_indices]
        )
        self.previous_audio_indicator = self._sample(
            after_block_zero[:, audio_indices] - before_block_zero[:, audio_indices]
        )
        self.video_tail = after_stack[:, video_indices] - after_block_zero[:, video_indices]
        self.audio_tail = after_stack[:, audio_indices] - after_block_zero[:, audio_indices]
        mx.eval(
            self.previous_video_indicator,
            self.previous_audio_indicator,
            self.video_tail,
            self.audio_tail,
        )
        self.full_evaluations += 1
        self.consecutive_hits = 0

    @property
    def cache_bytes(self) -> int:
        values = (self.video_tail, self.audio_tail)
        return sum(int(value.nbytes) for value in values if value is not None)
