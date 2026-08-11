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
    state = H3TrajectoryForecastState(H3TrajectoryForecastConfig(mode="automatic_balanced"))
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


def test_conditioned_row_policy_rejects_unknown_value():
    with pytest.raises(ValueError, match="conditioned-row policy"):
        H3TrajectoryForecastConfig(conditioned_row_policy="unknown").validate()


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
    state = H3TrajectoryForecastState(H3TrajectoryForecastConfig(max_history=2))
    for index in range(4):
        state.update(float(index), *_features(float(index)))
    assert len(state._history) == 2
    assert state.history_bytes > 0


def test_offline_replay_uses_exact_anchors_and_local_audio_interpolation():
    state = H3TrajectoryForecastState(
        H3TrajectoryForecastConfig(
            mode="automatic_balanced",
            offline_smoothing_replay=True,
            offline_video_blend=0.5,
            offline_audio_blend=0.0,
        )
    )
    coordinates = (1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0)
    state.begin_capture(len(coordinates))
    for index, coordinate in enumerate(coordinates):
        predicted = state.try_predict(coordinate, index, len(coordinates))
        if predicted is None:
            state.update(coordinate, *_features(float(index)))

    assert state.forecasts == 1
    assert state.complete_capture() is True
    assert state._validation_scores["video"]
    assert state._validation_scores["audio"] == {}
    archive_bytes = state.archive_bytes
    assert archive_bytes > 0

    state.begin_replay()
    replayed = [
        state.try_predict(coordinate, index, len(coordinates))
        for index, coordinate in enumerate(coordinates)
    ]

    assert mx.array_equal(replayed[0][0], _features(0.0)[0])
    assert mx.array_equal(replayed[1][0], _features(1.0)[0])
    assert mx.array_equal(replayed[3][0], _features(3.0)[0])
    assert mx.array_equal(replayed[2][0], _features(2.0)[0])
    assert mx.array_equal(replayed[2][1], _features(2.0)[1])
    assert state.replay_steps == 4
    assert state.replay_anchor_steps == 3
    assert state.replay_smoothed_steps == 1
    assert state.history_bytes == archive_bytes

    state.release()
    assert state.archive_bytes == 0
    assert state._history == []


def test_offline_capture_requires_future_anchor_for_every_forecast():
    state = H3TrajectoryForecastState(H3TrajectoryForecastConfig(offline_smoothing_replay=True))
    state.begin_capture(3)
    state.update(1.0, *_features(0.0))
    state.update(0.5, *_features(1.0))
    state._record_capture_step(0.0, False)

    assert state.complete_capture() is False
    assert "past and future anchors" in state.replay_fallback_reason


def test_offline_capture_rejects_changed_feature_shape_without_raising():
    state = H3TrajectoryForecastState(H3TrajectoryForecastConfig(offline_smoothing_replay=True))
    state.begin_capture(2)
    state.update(1.0, *_features(0.0))
    state.update(
        0.0,
        mx.zeros((1, 9, 16), dtype=mx.bfloat16),
        mx.zeros((1, 4, 16), dtype=mx.bfloat16),
    )

    assert state.complete_capture() is False
    assert "shapes or dtypes changed" in state.replay_fallback_reason


@pytest.mark.parametrize(
    "field,value",
    [
        ("warmup_steps", 1),
        ("max_history", 1),
        ("max_history", 3),
        ("offline_video_blend", 1.1),
        ("offline_audio_blend", -0.1),
        ("offline_ridge_lambda", -1.0),
    ],
)
def test_config_rejects_unsafe_history_controls(field, value):
    values = {field: value}
    with pytest.raises(ValueError):
        H3TrajectoryForecastConfig(**values).validate()
