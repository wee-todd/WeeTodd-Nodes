"""Bounded post-transformer trajectory forecasting for joint H3 video and audio rows."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class H3TrajectoryForecastConfig:
    """Controls for optional post-stack feature extrapolation between actual evaluations."""

    mode: str = "automatic_balanced"
    forecast_strength: float = 0.75
    warmup_steps: int = 2
    tail_actual_steps: int = 1
    max_history: int = 2
    max_forecast_fraction: float = 0.35
    max_delta_ratio: float = 1.75
    guard_subsample_factor: int = 16
    bootstrap_first_forecast: bool = False

    def validate(self) -> None:
        modes = {
            "manual",
            "automatic_conservative",
            "automatic_balanced",
            "automatic_speed",
        }
        if self.mode not in modes:
            raise ValueError(
                f"Trajectory Forecast mode must be one of {sorted(modes)}, got {self.mode!r}."
            )
        if not 0 <= self.forecast_strength <= 1:
            raise ValueError("Trajectory Forecast strength must be between 0 and 1.")
        if self.warmup_steps < 2:
            raise ValueError("Trajectory Forecast warmup must be at least 2 actual steps.")
        if self.tail_actual_steps < 1:
            raise ValueError("Trajectory Forecast requires at least one actual tail step.")
        if self.max_history != 2:
            raise ValueError("Trajectory Forecast history must contain exactly two actual steps.")
        if not 0 <= self.max_forecast_fraction <= 0.5:
            raise ValueError("Trajectory Forecast maximum fraction must be between 0 and 0.5.")
        if self.max_delta_ratio < 0:
            raise ValueError("Trajectory Forecast delta ratio must be zero or positive.")
        if self.guard_subsample_factor < 1:
            raise ValueError("Trajectory Forecast guard subsample factor must be positive.")
        if not isinstance(self.bootstrap_first_forecast, bool):
            raise TypeError("Trajectory Forecast bootstrap control must be a boolean.")
        if self.bootstrap_first_forecast and self.mode != "automatic_speed":
            raise ValueError(
                "Trajectory Forecast bootstrap requires automatic_speed mode."
            )


class H3TrajectoryForecastState:
    """Extrapolate compact BF16 post-stack features with guarded two-point history."""

    def __init__(self, config: H3TrajectoryForecastConfig) -> None:
        config.validate()
        self.config = config
        self._history: list[tuple[float, object, object]] = []
        self.forecasts = 0
        self.bootstrap_forecasts = 0
        self.fallbacks = 0
        self.consecutive_forecasts = 0
        self.last_was_forecast = False
        self.last_delta_ratio: float | None = None

    def _resolved(self) -> tuple[float, int, int, float, float]:
        if self.config.mode == "automatic_conservative":
            return 0.5, max(3, self.config.warmup_steps), 1, 0.25, 1.25
        if self.config.mode == "automatic_speed":
            warmup = 1 if self.config.bootstrap_first_forecast else 2
            return 1.0, warmup, 1, 0.5, 2.5
        if self.config.mode == "automatic_balanced":
            return 0.75, 2, 1, 0.35, 1.75
        return (
            self.config.forecast_strength,
            self.config.warmup_steps,
            self.config.tail_actual_steps,
            self.config.max_forecast_fraction,
            self.config.max_delta_ratio,
        )

    @staticmethod
    def _mean_guard_delta(current, previous, stride: int) -> float:
        import mlx.core as mx

        sampled_current = current[:, ::stride, ::stride].astype(mx.float32)
        sampled_previous = previous[:, ::stride, ::stride].astype(mx.float32)
        return float(mx.mean(mx.abs(sampled_current - sampled_previous)).item())

    def _forecast_limit(self, total_steps: int, fraction: float) -> int:
        return math.floor(total_steps * fraction)

    def try_predict(self, coordinate: float, index: int, total_steps: int):
        """Return compact forecast video/audio features, or ``None`` for an actual evaluation."""
        import mlx.core as mx

        self.last_was_forecast = False
        strength, warmup, tail, fraction, max_ratio = self._resolved()
        if (
            index < warmup
            or index >= total_steps - tail
            or self.consecutive_forecasts >= 1
            or self.forecasts >= self._forecast_limit(total_steps, fraction)
        ):
            return None

        if (
            self.config.bootstrap_first_forecast
            and index == 1
            and len(self._history) == 1
        ):
            _, video, audio = self._history[-1]
            self.forecasts += 1
            self.bootstrap_forecasts += 1
            self.consecutive_forecasts += 1
            self.last_was_forecast = True
            return video, audio

        if len(self._history) < 2:
            return None

        previous_coordinate, previous_video, previous_audio = self._history[-2]
        latest_coordinate, latest_video, latest_audio = self._history[-1]
        spacing = latest_coordinate - previous_coordinate
        if abs(spacing) <= 1e-12:
            self.fallbacks += 1
            return None
        extrapolation = (float(coordinate) - latest_coordinate) / spacing
        scale = float(strength * extrapolation)
        video = latest_video + (latest_video - previous_video) * scale
        audio = latest_audio + (latest_audio - previous_audio) * scale

        stride = self.config.guard_subsample_factor
        recent = 0.5 * (
            self._mean_guard_delta(latest_video, previous_video, stride)
            + self._mean_guard_delta(latest_audio, previous_audio, stride)
        )
        forecast = 0.5 * (
            self._mean_guard_delta(video, latest_video, stride)
            + self._mean_guard_delta(audio, latest_audio, stride)
        )
        self.last_delta_ratio = forecast / max(recent, 1e-12)
        if self.last_delta_ratio > max_ratio:
            self.fallbacks += 1
            return None

        video = video.astype(latest_video.dtype)
        audio = audio.astype(latest_audio.dtype)
        mx.eval(video, audio)
        self.forecasts += 1
        self.consecutive_forecasts += 1
        self.last_was_forecast = True
        return video, audio

    def update(self, coordinate: float, video_hidden, audio_hidden) -> None:
        """Archive compact owned features after one actual transformer evaluation."""
        import mlx.core as mx

        video = video_hidden + mx.zeros((), dtype=video_hidden.dtype)
        audio = audio_hidden + mx.zeros((), dtype=audio_hidden.dtype)
        mx.eval(video, audio)
        self._history.append((float(coordinate), video, audio))
        if len(self._history) > self.config.max_history:
            self._history.pop(0)
        self.consecutive_forecasts = 0

    @property
    def history_bytes(self) -> int:
        return sum(
            int(video.nbytes + audio.nbytes) for _, video, audio in self._history
        )
