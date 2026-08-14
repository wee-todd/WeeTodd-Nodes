from __future__ import annotations

import mlx.core as mx

from minimax_h3_mlx.scheduler import (
    MiniMaxH3ResMultistepScheduler,
    MiniMaxH3Scheduler,
)


def test_h3_video_and_audio_schedules_keep_independent_released_shifts():
    video = MiniMaxH3Scheduler(shift=12.0)
    audio = MiniMaxH3Scheduler(shift=3.0)
    video.set_timesteps(8)
    audio.set_timesteps(8)
    assert len(video.timesteps) == len(audio.timesteps) == 7
    assert video.timesteps.tolist() != audio.timesteps.tolist()
    assert video.shift == 12.0
    assert audio.shift == 3.0


def test_res_multistep_first_update_matches_euler_and_interior_changes_path():
    euler = MiniMaxH3Scheduler(shift=3.0)
    residual = MiniMaxH3ResMultistepScheduler(shift=3.0)
    euler.set_timesteps(5)
    residual.set_timesteps(5)
    euler_sample = mx.array([0.4, -0.2], dtype=mx.float32)
    residual_sample = euler_sample

    first_t = float(euler.timesteps[0].item())
    first_prediction = mx.array([0.3, -0.1], dtype=mx.float32)
    euler_sample = euler.step(first_prediction, first_t, euler_sample)
    residual_sample = residual.step(first_prediction, first_t, residual_sample)
    assert mx.allclose(euler_sample, residual_sample).item()

    second_t = float(euler.timesteps[1].item())
    second_prediction = mx.array([-0.2, 0.5], dtype=mx.float32)
    euler_sample = euler.step(second_prediction, second_t, euler_sample)
    residual_sample = residual.step(second_prediction, second_t, residual_sample)
    assert not mx.allclose(euler_sample, residual_sample).item()


def test_res_multistep_terminal_update_returns_current_denoised_estimate():
    scheduler = MiniMaxH3ResMultistepScheduler(shift=12.0)
    scheduler.set_timesteps(3)
    sample = mx.array([0.25, -0.5], dtype=mx.float32)

    for index, timestep in enumerate(scheduler.timesteps.tolist()):
        prediction = mx.array([0.1 + index, -0.2 - index], dtype=mx.float32)
        before = sample
        sample = scheduler.step(prediction, float(timestep), sample)

    sigma = 1.0 - float(scheduler.timesteps[-1].item())
    expected = before + sigma * prediction
    assert mx.allclose(sample, expected, rtol=1e-6, atol=1e-6).item()


def test_res_multistep_state_resets_with_new_schedule():
    scheduler = MiniMaxH3ResMultistepScheduler()
    scheduler.set_timesteps(4)
    scheduler.step(mx.ones((2,)), float(scheduler.timesteps[0].item()), mx.zeros((2,)))
    scheduler.set_timesteps(4)
    assert scheduler.step_index is None
    assert scheduler._old_denoised is None
