from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from ltx25_mlx.sampling import euler_ancestral_denoise_loop, euler_ancestral_step


@dataclass(frozen=True)
class _State:
    latent: mx.array
    clean_latent: mx.array
    denoise_mask: mx.array
    positions: mx.array | None = None
    attention_mask: mx.array | None = None


def test_euler_ancestral_step_matches_rectified_flow_formula():
    sample = mx.array([[[0.25, -0.5]]], dtype=mx.float32)
    denoised = mx.array([[[0.1, 0.2]]], dtype=mx.float32)
    noise = mx.array([[[0.75, -0.25]]], dtype=mx.float32)
    result = euler_ancestral_step(sample, denoised, 1.0, 0.725, noise=noise)

    sigma_down = 0.725 * (1.0 + (0.725 / 1.0 - 1.0))
    deterministic = sigma_down * np.array([0.25, -0.5]) + (1.0 - sigma_down) * np.array([0.1, 0.2])
    alpha_next = 1.0 - 0.725
    alpha_down = 1.0 - sigma_down
    renoise = max(0.725**2 - sigma_down**2 * alpha_next**2 / alpha_down**2, 0.0) ** 0.5
    expected = alpha_next / alpha_down * deterministic + np.array([0.75, -0.25]) * renoise
    np.testing.assert_allclose(np.asarray(result)[0, 0], expected, rtol=1e-6, atol=1e-6)


def test_euler_ancestral_step_terminal_is_exact_denoised_value():
    sample = mx.array([1.0], dtype=mx.bfloat16)
    denoised = mx.array([0.125], dtype=mx.bfloat16)
    result = euler_ancestral_step(sample, denoised, 0.421875, 0.0, noise=None)
    assert mx.array_equal(result, denoised)


def test_euler_eta_zero_matches_deterministic_rectified_flow_step():
    sample = mx.array([2.0, -1.0], dtype=mx.float32)
    denoised = mx.array([0.5, 3.0], dtype=mx.float32)
    sigma = 0.909375
    sigma_next = 0.725
    result = euler_ancestral_step(
        sample, denoised, sigma, sigma_next, noise=None, eta=0.0
    )
    ratio = sigma_next / sigma
    expected = ratio * sample + (1.0 - ratio) * denoised
    assert mx.allclose(result, expected, rtol=0.0, atol=1e-6)


def test_euler_ancestral_loop_is_seeded_and_preserves_conditioned_rows():
    mask = mx.array([[[1.0], [0.0]]], dtype=mx.float32)
    clean = mx.array([[[0.0], [0.75]]], dtype=mx.float32)
    state = _State(mx.zeros((1, 2, 1)), clean, mask)
    callbacks = []
    timings = []

    def model(**kwargs):
        video = mx.full_like(kwargs["video_latent"], 0.25)
        audio = mx.full_like(kwargs["audio_latent"], -0.5)
        return video, audio

    def run(seed):
        return euler_ancestral_denoise_loop(
            model,
            state,
            state,
            mx.zeros((1, 1, 1)),
            mx.zeros((1, 1, 1)),
            sigmas=(1.0, 0.725, 0.421875),
            noise_seed=seed,
            step_callback=lambda done, total: callbacks.append((done, total)),
            evaluation_timing_callback=lambda done, elapsed: timings.append((done, elapsed)),
        )

    first = run(10000)
    second = run(10000)
    other = run(10001)
    assert mx.array_equal(first.video_latent, second.video_latent)
    assert mx.array_equal(first.audio_latent, second.audio_latent)
    assert not mx.array_equal(first.video_latent[:, :1], other.video_latent[:, :1])
    assert mx.array_equal(first.video_latent[:, 1:], clean[:, 1:])
    assert mx.array_equal(first.audio_latent[:, 1:], clean[:, 1:])
    assert callbacks[:2] == [(1, 2), (2, 2)]
    assert [index for index, _elapsed in timings[:2]] == [1, 2]
    assert all(elapsed >= 0.0 for _index, elapsed in timings)


def test_euler_ancestral_step_requires_noise_before_terminal():
    with pytest.raises(ValueError, match="requires noise"):
        euler_ancestral_step(mx.zeros((1,)), mx.zeros((1,)), 1.0, 0.5, noise=None)
