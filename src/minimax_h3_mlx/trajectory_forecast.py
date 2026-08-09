"""Bounded post-transformer trajectory forecasting for joint H3 video and audio rows."""

from __future__ import annotations

import math
from dataclasses import dataclass


class H3OfflineReplayError(RuntimeError):
    """A recoverable replay-contract failure that leaves the capture result valid."""


@dataclass(frozen=True)
class H3TrajectoryForecastConfig:
    """Controls for optional post-stack feature forecasting and offline replay."""

    mode: str = "automatic_balanced"
    forecast_strength: float = 0.75
    warmup_steps: int = 2
    tail_actual_steps: int = 1
    max_history: int = 2
    max_forecast_fraction: float = 0.35
    max_delta_ratio: float = 1.75
    guard_subsample_factor: int = 16
    bootstrap_first_forecast: bool = False
    offline_smoothing_replay: bool = False
    offline_video_blend: float = 0.5
    offline_audio_blend: float = 0.0
    offline_ridge_lambda: float = 1e-6

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
            raise ValueError("Trajectory Forecast bootstrap requires automatic_speed mode.")
        if not isinstance(self.offline_smoothing_replay, bool):
            raise TypeError("Trajectory Forecast offline replay control must be a boolean.")
        for label, value in (
            ("video blend", self.offline_video_blend),
            ("audio blend", self.offline_audio_blend),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"Trajectory Forecast offline {label} must be between 0 and 1.")
        if self.offline_ridge_lambda < 0:
            raise ValueError("Trajectory Forecast offline ridge lambda must be zero or positive.")


@dataclass(frozen=True)
class _OfflineStep:
    index: int
    coordinate: float
    actual: bool


@dataclass(frozen=True)
class _OfflineAnchor:
    index: int
    coordinate: float
    video: object
    audio: object


class H3TrajectoryForecastState:
    """Forecast compact BF16 features and optionally replay from actual anchors."""

    def __init__(self, config: H3TrajectoryForecastConfig) -> None:
        config.validate()
        self.config = config
        self._history: list[tuple[float, object, object]] = []
        self._offline_steps: list[_OfflineStep] = []
        self._offline_anchors: list[_OfflineAnchor] = []
        self._validation_scores: dict[str, dict[int, float]] = {
            "video": {},
            "audio": {},
        }
        self._phase = "online"
        self._offline_total_steps = 0
        self._capture_validated = False
        self._archive_signature: tuple | None = None
        self._capture_invalid_reason: str | None = None
        self.forecasts = 0
        self.bootstrap_forecasts = 0
        self.fallbacks = 0
        self.consecutive_forecasts = 0
        self.last_was_forecast = False
        self.last_delta_ratio: float | None = None
        self.replay_steps = 0
        self.replay_smoothed_steps = 0
        self.replay_anchor_steps = 0
        self.replay_fallback_reason: str | None = None
        self.offline_archive_peak_bytes = 0

    @property
    def replaying(self) -> bool:
        return self._phase == "replay"

    def begin_capture(self, total_steps: int) -> None:
        if not self.config.offline_smoothing_replay:
            raise ValueError("Trajectory Forecast offline replay is not enabled.")
        if total_steps < 2:
            raise ValueError("Trajectory Forecast offline replay requires at least two steps.")
        self.release()
        self._phase = "capture"
        self._offline_total_steps = int(total_steps)
        self._capture_validated = False
        self._archive_signature = None
        self._capture_invalid_reason = None
        self.forecasts = 0
        self.bootstrap_forecasts = 0
        self.fallbacks = 0
        self.consecutive_forecasts = 0
        self.last_was_forecast = False
        self.last_delta_ratio = None
        self.replay_steps = 0
        self.replay_smoothed_steps = 0
        self.replay_anchor_steps = 0
        self.replay_fallback_reason = None
        self.offline_archive_peak_bytes = 0

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

    @staticmethod
    def _rmse(current, reference, stride: int) -> float:
        import mlx.core as mx

        delta = current[:, ::stride, ::stride].astype(mx.float32) - reference[
            :, ::stride, ::stride
        ].astype(mx.float32)
        return float(mx.sqrt(mx.mean(delta * delta)).item())

    @staticmethod
    def _rms(value, stride: int) -> float:
        import mlx.core as mx

        sampled = value[:, ::stride, ::stride].astype(mx.float32)
        return float(mx.sqrt(mx.mean(sampled * sampled)).item())

    def _forecast_limit(self, total_steps: int, fraction: float) -> int:
        return math.floor(total_steps * fraction)

    def _record_capture_step(self, coordinate: float, actual: bool) -> None:
        index = len(self._offline_steps)
        if index >= self._offline_total_steps:
            raise H3OfflineReplayError("Offline capture produced too many logical steps.")
        self._offline_steps.append(_OfflineStep(index, float(coordinate), bool(actual)))

    def try_predict(self, coordinate: float, index: int, total_steps: int):
        """Return compact forecast video/audio features, or ``None`` for an actual evaluation."""
        if self.replaying:
            return self._replay_prediction(coordinate, index, total_steps)

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

        if self.config.bootstrap_first_forecast and index == 1 and len(self._history) == 1:
            _, video, audio = self._history[-1]
            self.forecasts += 1
            self.bootstrap_forecasts += 1
            self.consecutive_forecasts += 1
            self.last_was_forecast = True
            if self._phase == "capture":
                self._record_capture_step(coordinate, False)
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
        causal_strength = 0.0 if self._phase == "capture" else strength
        scale = float(causal_strength * extrapolation)
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
        if self._phase == "capture":
            self._record_capture_step(coordinate, False)
        return video, audio

    def update(self, coordinate: float, video_hidden, audio_hidden) -> None:
        """Archive compact owned features after one actual transformer evaluation."""
        import mlx.core as mx

        video = video_hidden + mx.zeros((), dtype=video_hidden.dtype)
        audio = audio_hidden + mx.zeros((), dtype=audio_hidden.dtype)
        mx.eval(video, audio)
        if self._phase == "capture":
            index = len(self._offline_steps)
            self._record_capture_step(coordinate, True)
            signature = (
                tuple(video.shape),
                str(video.dtype),
                tuple(audio.shape),
                str(audio.dtype),
            )
            if self._archive_signature is None:
                self._archive_signature = signature
            elif signature != self._archive_signature:
                self._capture_invalid_reason = (
                    "Offline capture feature shapes or dtypes changed between actual anchors."
                )
            if self._capture_invalid_reason is None:
                self._offline_anchors.append(_OfflineAnchor(index, float(coordinate), video, audio))
                self.offline_archive_peak_bytes = max(
                    self.offline_archive_peak_bytes,
                    self.archive_bytes,
                )
        self._history.append((float(coordinate), video, audio))
        if len(self._history) > self.config.max_history:
            self._history.pop(0)
        self.consecutive_forecasts = 0

    @staticmethod
    def _affine_weights(coordinates: list[float], target: float, ridge: float) -> list[float]:
        import numpy as np

        if len(coordinates) < 2:
            raise H3OfflineReplayError("Offline affine fitting requires at least two anchors.")
        x = np.asarray(coordinates, dtype=np.float64)
        design = np.stack((np.ones_like(x), x), axis=1)
        regularizer = np.diag(np.asarray((0.0, float(ridge)), dtype=np.float64))
        gram = design.T @ design + regularizer
        try:
            coefficients = np.linalg.solve(gram, design.T)
        except np.linalg.LinAlgError as exc:
            raise H3OfflineReplayError("Offline affine anchor fitting is singular.") from exc
        weights = np.asarray((1.0, float(target)), dtype=np.float64) @ coefficients
        if not np.all(np.isfinite(weights)):
            raise H3OfflineReplayError("Offline affine anchor weights are nonfinite.")
        return [float(value) for value in weights]

    @staticmethod
    def _weighted_features(anchors: list[_OfflineAnchor], weights: list[float], modality: str):
        import mlx.core as mx

        if len(anchors) != len(weights) or not anchors:
            raise H3OfflineReplayError("Offline anchor weights do not match the archive.")
        result = getattr(anchors[0], modality) * weights[0]
        mx.eval(result)
        for anchor, weight in zip(anchors[1:], weights[1:], strict=True):
            result = result + getattr(anchor, modality) * weight
            mx.eval(result)
        return result

    @staticmethod
    def _local_prediction(
        left: _OfflineAnchor, right: _OfflineAnchor, target: float, modality: str
    ):
        spacing = right.coordinate - left.coordinate
        if abs(spacing) <= 1e-12:
            raise H3OfflineReplayError("Offline bracketing anchors have equal coordinates.")
        ratio = (float(target) - left.coordinate) / spacing
        start = getattr(left, modality)
        end = getattr(right, modality)
        return start + (end - start) * ratio

    def _build_validation_scores(self) -> None:
        self._validation_scores = {"video": {}, "audio": {}}
        if len(self._offline_anchors) < 3:
            return
        stride = self.config.guard_subsample_factor
        modalities = []
        if self.config.offline_video_blend > 0:
            modalities.append("video")
        if self.config.offline_audio_blend > 0:
            modalities.append("audio")
        for position in range(1, len(self._offline_anchors) - 1):
            target = self._offline_anchors[position]
            retained = self._offline_anchors[:position] + self._offline_anchors[position + 1 :]
            weights = self._affine_weights(
                [anchor.coordinate for anchor in retained],
                target.coordinate,
                self.config.offline_ridge_lambda,
            )
            for modality in modalities:
                global_prediction = self._weighted_features(retained, weights, modality)
                local_prediction = self._local_prediction(
                    self._offline_anchors[position - 1],
                    self._offline_anchors[position + 1],
                    target.coordinate,
                    modality,
                )
                reference = getattr(target, modality)
                global_error = self._rmse(global_prediction, reference, stride)
                local_error = self._rmse(local_prediction, reference, stride)
                reference_scale = self._rms(reference, stride)
                epsilon = max(reference_scale * 1e-6, 1e-7)
                score = global_error / max(local_error, epsilon)
                self._validation_scores[modality][target.index] = (
                    score if math.isfinite(score) else math.inf
                )

    def complete_capture(self) -> bool:
        if self._phase != "capture":
            raise RuntimeError("Trajectory Forecast offline capture is not active.")
        if self._capture_invalid_reason is not None:
            self.replay_fallback_reason = self._capture_invalid_reason
            return False
        if len(self._offline_steps) != self._offline_total_steps:
            self.replay_fallback_reason = "Offline capture did not record every logical step."
            return False
        actual_indices = [step.index for step in self._offline_steps if step.actual]
        if actual_indices != [anchor.index for anchor in self._offline_anchors]:
            self.replay_fallback_reason = (
                "Offline capture anchor identities do not match the schedule."
            )
            return False
        if len(self._offline_anchors) < 2:
            self.replay_fallback_reason = "Offline capture retained fewer than two actual anchors."
            return False
        for step in self._offline_steps:
            if step.actual:
                continue
            if not any(anchor.index < step.index for anchor in self._offline_anchors) or not any(
                anchor.index > step.index for anchor in self._offline_anchors
            ):
                self.replay_fallback_reason = (
                    "Offline replay requires past and future anchors for every forecast step."
                )
                return False
        try:
            self._build_validation_scores()
        except H3OfflineReplayError as exc:
            self.replay_fallback_reason = str(exc)
            return False
        self._capture_validated = True
        return True

    def begin_replay(self) -> None:
        if self._phase != "capture" or not self._capture_validated:
            raise H3OfflineReplayError(
                self.replay_fallback_reason or "Offline capture is incomplete."
            )
        self._phase = "replay"
        self.last_was_forecast = False

    def _bracketing_anchors(self, index: int) -> tuple[_OfflineAnchor, _OfflineAnchor]:
        left = [anchor for anchor in self._offline_anchors if anchor.index < index]
        right = [anchor for anchor in self._offline_anchors if anchor.index > index]
        if not left or not right:
            raise H3OfflineReplayError("Offline forecast step does not have bracketing anchors.")
        return left[-1], right[0]

    def _effective_blend(
        self,
        modality: str,
        configured: float,
        left: _OfflineAnchor,
        right: _OfflineAnchor,
    ) -> float:
        nearby = [
            self._validation_scores[modality][anchor.index]
            for anchor in (left, right)
            if anchor.index in self._validation_scores[modality]
        ]
        score = max(nearby, default=1.0)
        return float(configured / max(1.0, score))

    def _reconstruct_modality(
        self,
        coordinate: float,
        modality: str,
        configured_blend: float,
        left: _OfflineAnchor,
        right: _OfflineAnchor,
    ):
        import mlx.core as mx

        local = self._local_prediction(left, right, coordinate, modality)
        blend = self._effective_blend(modality, configured_blend, left, right)
        if blend <= 1e-12:
            result = local
        else:
            weights = self._affine_weights(
                [anchor.coordinate for anchor in self._offline_anchors],
                coordinate,
                self.config.offline_ridge_lambda,
            )
            global_prediction = self._weighted_features(self._offline_anchors, weights, modality)
            result = local + (global_prediction - local) * blend
        reference = getattr(left, modality)
        result = result.astype(reference.dtype)
        mx.eval(result)
        return result

    def _replay_prediction(self, coordinate: float, index: int, total_steps: int):
        if total_steps != self._offline_total_steps or index >= len(self._offline_steps):
            raise H3OfflineReplayError("Offline replay schedule length changed.")
        record = self._offline_steps[index]
        if record.index != index or not math.isclose(
            record.coordinate, float(coordinate), rel_tol=1e-6, abs_tol=1e-6
        ):
            raise H3OfflineReplayError("Offline replay timestep identity changed.")
        self.replay_steps += 1
        anchor = next(
            (candidate for candidate in self._offline_anchors if candidate.index == index),
            None,
        )
        if anchor is not None:
            self.last_was_forecast = False
            self.replay_anchor_steps += 1
            return anchor.video, anchor.audio

        left, right = self._bracketing_anchors(index)
        video = self._reconstruct_modality(
            coordinate,
            "video",
            self.config.offline_video_blend,
            left,
            right,
        )
        audio = self._reconstruct_modality(
            coordinate,
            "audio",
            self.config.offline_audio_blend,
            left,
            right,
        )
        self.last_was_forecast = True
        self.replay_smoothed_steps += 1
        return video, audio

    def mark_replay_fallback(self, reason: str) -> None:
        self.fallbacks += 1
        self.replay_fallback_reason = str(reason)

    @property
    def archive_bytes(self) -> int:
        return sum(
            int(anchor.video.nbytes + anchor.audio.nbytes) for anchor in self._offline_anchors
        )

    @property
    def history_bytes(self) -> int:
        if self.config.offline_smoothing_replay:
            return max(self.archive_bytes, self.offline_archive_peak_bytes)
        return sum(int(video.nbytes + audio.nbytes) for _, video, audio in self._history)

    def release(self) -> None:
        self._history.clear()
        self._offline_steps.clear()
        self._offline_anchors.clear()
        self._validation_scores = {"video": {}, "audio": {}}
        self._phase = "online"
        self._offline_total_steps = 0
        self._capture_validated = False
        self._archive_signature = None
        self._capture_invalid_reason = None
        self.consecutive_forecasts = 0
        self.last_was_forecast = False
