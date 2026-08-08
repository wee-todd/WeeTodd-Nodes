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


def test_speed_bootstrap_holds_first_actual_feature_for_second_step():
    state = H3TrajectoryForecastState(
        H3TrajectoryForecastConfig(
            mode="automatic_speed",
            bootstrap_first_forecast=True,
        )
    )
    video, audio = _features(1.0)
    state.update(1.0, video, audio)

    predicted = state.try_predict(0.75, index=1, total_steps=4)

    assert predicted is not None
    assert mx.array_equal(predicted[0], video)
    assert mx.array_equal(predicted[1], audio)
    assert state.forecasts == 1
    assert state.bootstrap_forecasts == 1
    assert state.try_predict(0.5, index=2, total_steps=4) is None


def test_speed_bootstrap_keeps_twenty_step_forecast_budget():
    state = H3TrajectoryForecastState(
        H3TrajectoryForecastConfig(
            mode="automatic_speed",
            bootstrap_first_forecast=True,
            max_delta_ratio=100.0,
        )
    )
    schedule = []
    for index in range(20):
        predicted = state.try_predict(float(19 - index), index=index, total_steps=20)
        if predicted is None:
            schedule.append("A")
            state.update(float(19 - index), *_features(float(index)))
        else:
            schedule.append("F")

    assert "".join(schedule) == "AFAFAFAFAFAFAFAFAFAA"
    assert state.forecasts == 9
    assert state.bootstrap_forecasts == 1


def test_bootstrap_rejects_non_speed_policy():
    with pytest.raises(ValueError, match="automatic_speed"):
        H3TrajectoryForecastConfig(
            mode="automatic_balanced",
            bootstrap_first_forecast=True,
        ).validate()


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
