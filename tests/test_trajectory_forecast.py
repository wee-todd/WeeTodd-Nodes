import mlx.core as mx
import pytest

from minimax_h3_mlx.trajectory_forecast import (
    H3TrajectoryForecastConfig,
    H3TrajectoryForecastState,
)


def _features(value: float):
    video = mx.full((1, 8, 16), value, dtype=mx.bfloat16)
    audio = mx.full((1, 4, 16), value * 0.5, dtype=mx.bfloat16)
    return video, audio


def test_turbo_length_schedule_forecasts_only_middle_step():
    state = H3TrajectoryForecastState(
        H3TrajectoryForecastConfig(mode="automatic_balanced")
    )
    state.update(1.0, *_features(0.0))
    state.update(2.0 / 3.0, *_features(1.0))

    predicted = state.try_predict(1.0 / 3.0, index=2, total_steps=4)
    assert predicted is not None
    assert state.last_was_forecast is True
    assert state.forecasts == 1
    assert state.try_predict(0.0, index=3, total_steps=4) is None


def test_guard_falls_back_when_forecast_delta_exceeds_ratio():
    state = H3TrajectoryForecastState(
        H3TrajectoryForecastConfig(
            mode="manual",
            forecast_strength=1.0,
            max_delta_ratio=0.1,
        )
    )
    state.update(1.0, *_features(0.0))
    state.update(0.75, *_features(1.0))

    assert state.try_predict(0.5, index=2, total_steps=4) is None
    assert state.fallbacks == 1
    assert state.last_was_forecast is False


def test_history_is_bounded_and_reports_storage():
    state = H3TrajectoryForecastState(
        H3TrajectoryForecastConfig(max_history=2)
    )
    for index in range(4):
        state.update(float(index), *_features(float(index)))
    assert len(state._history) == 2
    assert state.history_bytes > 0


@pytest.mark.parametrize(
    "field,value", [("warmup_steps", 1), ("max_history", 1), ("max_history", 3)]
)
def test_config_rejects_unsafe_history_controls(field, value):
    values = {field: value}
    with pytest.raises(ValueError):
        H3TrajectoryForecastConfig(**values).validate()
