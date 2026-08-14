from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from ltx25_mlx.chaining import (
    DecodedChainAssembler,
    LatentGuideConditioning,
    assemble_ltx25_latents,
    fit_audio_window,
    mlx_audio_to_numpy,
    motion_matched_overlap,
    plan_ltx25_chain,
    splice_audio_windows,
)


def test_three_window_15_second_plan_is_exact():
    plan = plan_ltx25_chain(
        total_frames=361,
        window_count=3,
        overlap_frames=25,
        frame_rate=24.0,
    )

    assert plan.window_frames == 137
    assert plan.video_overlap_latent_frames == 4
    assert plan.window_audio_tokens == 143
    assert plan.join_audio_tokens == (27, 26)
    assert plan.window_start_frames == (0, 112, 224)
    assert plan.assembled_video_seam_frames == (137, 249)
    assert plan.assembled_audio_seam_tokens == (143, 259)
    assert plan.expected_audio_tokens == 376


@pytest.mark.parametrize(
    ("total", "windows", "overlap"),
    [(360, 3, 25), (361, 5, 25), (361, 3, 17), (361, 3, 24)],
)
def test_invalid_chain_plans_fail_before_inference(total, windows, overlap):
    with pytest.raises(ValueError):
        plan_ltx25_chain(
            total_frames=total,
            window_count=windows,
            overlap_frames=overlap,
            frame_rate=24.0,
        )


def test_guide_conditioning_appends_prior_overlap_with_duplicate_positions():
    state = SimpleNamespace(
        latent=mx.zeros((1, 5, 2), dtype=mx.bfloat16),
        clean_latent=mx.zeros((1, 5, 2), dtype=mx.bfloat16),
        denoise_mask=mx.ones((1, 5, 1), dtype=mx.bfloat16),
        positions=mx.zeros((1, 5, 1)),
        attention_mask=None,
    )
    prefix = mx.full((1, 2, 2), 3.0, dtype=mx.bfloat16)
    result = LatentGuideConditioning(prefix).apply(state, (1, 1, 1))

    assert result.latent.shape == (1, 7, 2)
    assert mx.array_equal(result.latent[:, :5], state.latent)
    assert mx.array_equal(result.latent[:, 5:], prefix)
    assert mx.array_equal(result.clean_latent[:, 5:], prefix)
    assert mx.array_equal(result.denoise_mask[:, :5], mx.ones((1, 5, 1)))
    assert mx.array_equal(result.denoise_mask[:, 5:], mx.zeros((1, 2, 1)))
    assert mx.array_equal(result.positions[:, 5:], state.positions[:, :2])


def test_guide_conditioning_can_lightly_regenerate_overlap():
    state = SimpleNamespace(
        latent=mx.zeros((1, 5, 2), dtype=mx.bfloat16),
        clean_latent=mx.zeros((1, 5, 2), dtype=mx.bfloat16),
        denoise_mask=mx.ones((1, 5, 1), dtype=mx.bfloat16),
        positions=mx.zeros((1, 5, 1)),
        attention_mask=None,
    )
    prefix = mx.full((1, 2, 2), 3.0, dtype=mx.bfloat16)
    result = LatentGuideConditioning(prefix, strength=0.85).apply(state, (1, 1, 1))

    expected = mx.full((1, 2, 1), 0.15, dtype=mx.bfloat16)
    assert mx.allclose(result.denoise_mask[:, -2:], expected)
    assert mx.array_equal(result.denoise_mask[:, :5], mx.ones((1, 5, 1)))


def test_assemble_three_window_latents_uses_causal_overlap_transition():
    plan = plan_ltx25_chain(
        total_frames=361,
        window_count=3,
        overlap_frames=25,
        frame_rate=24.0,
    )
    videos = [mx.full((1, 2, 18, 2, 2), index) for index in range(3)]
    audios = [mx.full((1, 8, 143, 16), index) for index in range(3)]

    video, audio = assemble_ltx25_latents(videos, audios, plan)

    assert video.shape == (1, 2, 46, 2, 2)
    assert audio.shape == (1, 8, 376, 16)
    # Each later window drops its first causal latent and contributes fourteen
    # new latents. The three-frame transition starts with prior history and ends
    # with the new window rather than introducing a hard boundary.
    assert bool(mx.all(video[:, :, :15] == 0).item())
    assert bool(mx.all(video[:, :, 17:29] == 1).item())
    assert bool(mx.all(video[:, :, 31:] == 2).item())
    assert bool(mx.all(audio[:, :, :143] == 0).item())
    assert bool(mx.all(audio[:, :, 143:259] == 1).item())
    assert bool(mx.all(audio[:, :, 259:] == 2).item())


def test_motion_matched_overlap_selects_better_than_fixed_end():
    previous = np.zeros((6, 8, 8, 3), dtype=np.uint8)
    following = np.zeros_like(previous)
    for index in range(6):
        previous[index] = index * 10
        following[index] = index * 10
    following[-1] = 255

    joined, report = motion_matched_overlap(previous, following, blend_frames=2)

    assert joined.shape == previous.shape
    assert report["seam_frame_in_overlap"] < 5
    assert report["seam_score"] < report["fixed_end_score"]


class _FrameWriter:
    def __init__(self) -> None:
        self.parts = []
        self.frames = 0

    def write(self, frames):
        self.parts.append(np.asarray(frames).copy())
        self.frames += int(frames.shape[0])


def test_decoded_chain_assembler_keeps_exact_duration_and_reports_join():
    writer = _FrameWriter()
    assembler = DecodedChainAssembler(writer, target_frames=9, overlap_frames=3)
    first = np.zeros((6, 4, 4, 3), dtype=np.uint8)
    second = np.ones((6, 4, 4, 3), dtype=np.uint8) * 20

    assembler.begin_window(0)
    assembler.write(first)
    assembler.end_window()
    assembler.begin_window(1)
    assembler.write(second)
    assembler.end_window()
    assembler.finish()

    output = np.concatenate(writer.parts)
    assert output.shape[0] == 9
    assert len(assembler.join_reports) == 1
    assert assembler.join_reports[0]["assembled_overlap_start_frame"] == 3


def test_audio_join_uses_selected_visual_seam_and_exact_overlap_length():
    first = np.zeros((2, 12), dtype=np.float32)
    second = np.ones((2, 12), dtype=np.float32)
    waveform, reports = splice_audio_windows(
        [first, second],
        overlap_samples=4,
        video_joins=[{"seam_frame_in_overlap": 2}],
        context_frames=4,
        crossfade_samples=2,
    )

    assert waveform.shape == (2, 20)
    assert reports[0]["seam_sample_in_overlap"] == 2
    assert reports[0]["assembled_seam_sample"] == 10
    assert reports[0]["crossfade_samples"] == 2


def test_bfloat16_audio_is_cast_before_numpy_transfer():
    waveform = mx.full((1, 2, 8), 0.25, dtype=mx.bfloat16)

    array = mlx_audio_to_numpy(waveform)

    assert array.dtype == np.float32
    assert array.shape == (1, 2, 8)
    assert np.all(array == 0.25)


def test_audio_window_is_padded_to_video_clock_without_moving_leading_audio():
    waveform = np.arange(12, dtype=np.float32).reshape(2, 6)

    fitted, adjustment = fit_audio_window(waveform, 8)

    assert fitted.shape == (2, 8)
    assert np.array_equal(fitted[:, :6], waveform)
    assert np.array_equal(fitted[:, 6:], np.zeros((2, 2), dtype=np.float32))
    assert adjustment == "zero_padded_2"
