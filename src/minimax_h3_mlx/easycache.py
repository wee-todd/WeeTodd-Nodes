"""Joint video/audio EasyCache state for the MLX MiniMax H3 sampler."""

from __future__ import annotations

import math
from dataclasses import dataclass

CORE_RESIDUAL_REUSE = object()


@dataclass(frozen=True)
class H3EasyCacheConfig:
    """Residual-reuse controls for manual and calibrated automatic policies."""

    mode: str = "manual"
    reuse_threshold: float = 0.2
    start_percent: float = 0.15
    end_percent: float = 0.95
    subsample_factor: int = 8
    auto_multiplier: float = 1.15
    max_skip_fraction: float = 0.25
    reuse_strategy: str = "output_residual"
    allow_turbo_experimental: bool = False

    def validate(self) -> None:
        if self.mode not in {
            "manual",
            "automatic",
            "automatic_conservative",
            "automatic_balanced",
            "automatic_speed",
        }:
            raise ValueError(
                "EasyCache mode must be 'manual', 'automatic_conservative', "
                "'automatic_balanced', or 'automatic_speed'."
            )
        if self.reuse_threshold < 0:
            raise ValueError("EasyCache reuse threshold must be zero or positive.")
        if not 0 <= self.start_percent <= self.end_percent <= 1:
            raise ValueError("EasyCache requires 0 <= start_percent <= end_percent <= 1.")
        if self.subsample_factor < 1:
            raise ValueError("EasyCache subsample factor must be positive.")
        if self.auto_multiplier < 1:
            raise ValueError("EasyCache automatic multiplier must be at least 1.0.")
        if not 0 <= self.max_skip_fraction <= 0.5:
            raise ValueError("EasyCache maximum skip fraction must be between 0 and 0.5.")
        if self.reuse_strategy not in {"output_residual", "core_residual_fresh_heads"}:
            raise ValueError(
                "EasyCache reuse strategy must be 'output_residual' or "
                "'core_residual_fresh_heads'."
            )


class H3EasyCacheState:
    """Track joint H3 residuals and decide whether to reuse one transformer result."""

    def __init__(self, config: H3EasyCacheConfig) -> None:
        config.validate()
        self.config = config
        self.previous_video_input = None
        self.previous_audio_input = None
        self.previous_video_output = None
        self.previous_audio_output = None
        self.video_residual = None
        self.audio_residual = None
        self.core_residual = None
        self.output_norm: float | None = None
        self.relative_transformation_rate: float | None = None
        self.cumulative_change_rate = 0.0
        self.skipped_steps = 0
        self.consecutive_skips = 0
        self.last_was_core_reuse = False
        self.resolved_threshold: float | None = (
            config.reuse_threshold if config.mode == "manual" else None
        )

    @property
    def _automatic(self) -> bool:
        return self.config.mode != "manual"

    @property
    def _speed_policy(self) -> bool:
        return self.config.mode == "automatic_speed"

    @property
    def _balanced_policy(self) -> bool:
        return self.config.mode == "automatic_balanced"

    @property
    def uses_core_residual(self) -> bool:
        return self.config.reuse_strategy == "core_residual_fresh_heads"

    def _subsample(self, value):
        return value[:, :: self.config.subsample_factor, :]

    @staticmethod
    def _mean_abs(value) -> float:
        import mlx.core as mx

        return float(mx.mean(mx.abs(value)).item())

    def _joint_change(self, video, previous_video, audio, previous_audio) -> float:
        return 0.5 * (
            self._mean_abs(video - previous_video)
            + self._mean_abs(audio - previous_audio)
        )

    def _in_window(self, index: int, total_steps: int) -> bool:
        if self.uses_core_residual and index >= total_steps - 1:
            return False
        progress = index / max(total_steps - 1, 1)
        if not self.config.start_percent <= progress <= self.config.end_percent:
            return False
        if self._automatic:
            return 2 <= index < total_steps - 1
        return True

    def _skip_limit(self, total_steps: int) -> int:
        if self._speed_policy:
            fraction = max(self.config.max_skip_fraction, 0.5)
        elif self._balanced_policy:
            fraction = max(self.config.max_skip_fraction, 0.35)
        else:
            fraction = self.config.max_skip_fraction
        return math.floor(total_steps * fraction)

    def _resolve_automatic_threshold(self, approximate_rate: float) -> float:
        if self._speed_policy:
            multiplier = max(self.config.auto_multiplier, 1.75)
            return min(1.25, max(0.25, approximate_rate * multiplier))
        if self._balanced_policy:
            multiplier = max(self.config.auto_multiplier, 1.4)
            return min(0.8, max(0.1, approximate_rate * multiplier))
        return min(
            0.5,
            max(0.05, approximate_rate * self.config.auto_multiplier),
        )

    def try_reuse(self, video_input, audio_input, index: int, total_steps: int):
        self.last_was_core_reuse = False
        if not self._in_window(index, total_steps):
            return None
        residual_missing = (
            self.core_residual is None
            if self.uses_core_residual
            else self.video_residual is None or self.audio_residual is None
        )
        if (
            residual_missing
            or self.previous_video_input is None
            or self.previous_audio_input is None
            or self.output_norm is None
            or self.relative_transformation_rate is None
        ):
            return None
        video_sample = self._subsample(video_input)
        audio_sample = self._subsample(audio_input)
        input_change = self._joint_change(
            video_sample,
            self.previous_video_input,
            audio_sample,
            self.previous_audio_input,
        )
        approximate_rate = (
            self.relative_transformation_rate * input_change / max(self.output_norm, 1e-12)
        )
        if self._automatic and self.resolved_threshold is None:
            self.resolved_threshold = self._resolve_automatic_threshold(approximate_rate)
        threshold = self.resolved_threshold or 0.0
        self.cumulative_change_rate += approximate_rate
        consecutive_skip_limit = 2 if self._speed_policy else 1
        automatic_limit_reached = self._automatic and (
            self.consecutive_skips >= consecutive_skip_limit
            or self.skipped_steps >= self._skip_limit(total_steps)
        )
        if self.cumulative_change_rate >= threshold or automatic_limit_reached:
            self.cumulative_change_rate = 0.0
            self.consecutive_skips = 0
            return None
        self.skipped_steps += 1
        self.consecutive_skips += 1
        if self.uses_core_residual:
            self.last_was_core_reuse = True
            return CORE_RESIDUAL_REUSE
        return video_input + self.video_residual, audio_input + self.audio_residual

    def update_core(self, before_stack, after_stack) -> None:
        """Cache the full packed block-stack residual after a real evaluation."""
        import mlx.core as mx

        self.core_residual = after_stack - before_stack
        mx.eval(self.core_residual)

    def reuse_core(self, current_input):
        """Apply the most recent block-stack residual to freshly packed inputs."""
        if self.core_residual is None:
            raise RuntimeError("EasyCache core residual reuse was requested before calibration.")
        if tuple(current_input.shape) != tuple(self.core_residual.shape):
            raise ValueError(
                "EasyCache packed sequence shape changed during sampling: "
                f"input {tuple(current_input.shape)}, cache {tuple(self.core_residual.shape)}."
            )
        return current_input + self.core_residual

    def update(self, video_input, audio_input, video_output, audio_output) -> None:
        self.consecutive_skips = 0
        video_sample = self._subsample(video_input)
        audio_sample = self._subsample(audio_input)
        video_output_sample = self._subsample(video_output)
        audio_output_sample = self._subsample(audio_output)
        if self.previous_video_input is not None and self.previous_video_output is not None:
            input_change = self._joint_change(
                video_sample,
                self.previous_video_input,
                audio_sample,
                self.previous_audio_input,
            )
            output_change = self._joint_change(
                video_output_sample,
                self.previous_video_output,
                audio_output_sample,
                self.previous_audio_output,
            )
            if input_change > 1e-12:
                self.relative_transformation_rate = output_change / input_change
        if not self.uses_core_residual:
            self.video_residual = video_output - video_input
            self.audio_residual = audio_output - audio_input
        self.previous_video_input = video_sample
        self.previous_audio_input = audio_sample
        self.previous_video_output = video_output_sample
        self.previous_audio_output = audio_output_sample
        self.output_norm = 0.5 * (
            self._mean_abs(video_output_sample) + self._mean_abs(audio_output_sample)
        )

    @property
    def cache_bytes(self) -> int:
        values = (self.video_residual, self.audio_residual, self.core_residual)
        return sum(int(value.nbytes) for value in values if value is not None)
