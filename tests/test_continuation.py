from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from minimax_h3_mlx.packing import build_packed_sequence
from wee_todd_nodes.continuation import (
    continuation_context_from_latents,
    trim_continuation_overlap,
)


def _source_latents():
    return SimpleNamespace(
        video=mx.arange(1 * 24 * 37 * 4 * 6).reshape(1, 24, 37, 4, 6),
        audio=mx.arange(2 * 32 * 207).reshape(2, 32, 207),
        num_frames=124,
        width=192,
        height=128,
        fps=24,
        sample_rate=32000,
        transformer_spec=SimpleNamespace(checkpoint="/models/H3", transformer="/models/H3/dit"),
    )


def test_continuation_context_copies_legal_synchronized_tail():
    source = _source_latents()

    context = continuation_context_from_latents(source, 22)

    assert context.video.shape == (1, 24, 7, 4, 6)
    assert context.audio.shape == (2, 32, 37)
    assert mx.array_equal(context.video, source.video[:, :, -7:]).item()
    assert mx.array_equal(context.audio, source.audio[:, :, -37:]).item()
    assert context.transformer_checkpoint == "/models/H3"


def test_continuation_context_rejects_non_grid_overlap():
    with pytest.raises(ValueError, match="must be one of"):
        continuation_context_from_latents(_source_latents(), 20)


def test_continuation_packing_overlaps_target_timeline():
    layout = build_packed_sequence(
        [1, 1],
        num_latent_frames=12,
        latent_height=4,
        latent_width=6,
        num_audio_latents=68,
        patch_size=(1, 2, 2),
        continuation_video_frames=7,
        continuation_audio_latents=37,
    )
    positions = np.asarray(layout.position_ids)
    video_indices = np.asarray(layout.video_indices)
    audio_indices = np.asarray(layout.audio_indices)
    rows_per_frame = 6

    condition_video_time = positions[video_indices[: 7 * rows_per_frame], 0]
    target_video_time = positions[video_indices[7 * rows_per_frame :], 0]
    assert np.array_equal(
        condition_video_time.reshape(7, rows_per_frame)[:, 0],
        target_video_time.reshape(12, rows_per_frame)[:7, 0],
    )
    condition_audio_time = positions[audio_indices[: 37 * 2], 0]
    target_audio_time = positions[audio_indices[37 * 2 :], 0]
    assert np.array_equal(condition_audio_time[:37], target_audio_time[:37])
    assert np.array_equal(condition_audio_time[37:], target_audio_time[68 : 68 + 37])
    assert layout.num_condition_video_rows == 42
    assert layout.num_condition_audio_rows == 74


def test_continuation_and_timed_keyframes_keep_separate_positions_and_timesteps():
    layout = build_packed_sequence(
        [1, 0],
        num_latent_frames=32,
        latent_height=4,
        latent_width=6,
        num_audio_latents=168,
        patch_size=(1, 2, 2),
        keyframe_anchors=(0, 60),
        continuation_video_frames=7,
        continuation_audio_latents=37,
    )
    positions = np.asarray(layout.position_ids)
    video_indices = np.asarray(layout.video_indices)
    rows_per_frame = 6
    first_timed = 7 * rows_per_frame
    second_timed = first_timed + rows_per_frame

    assert layout.num_continuation_video_rows == 42
    assert layout.num_condition_video_rows == 54
    assert positions[video_indices[first_timed], 0] == pytest.approx(2.0)
    assert positions[video_indices[second_timed], 0] == pytest.approx(102.0)

    distinct, inverse = __import__(
        "minimax_h3_mlx.packing", fromlist=["build_row_timesteps"]
    ).build_row_timesteps(layout, 0.4, 0.6, 0.999, 1.0)
    row_values = np.asarray(distinct)[np.asarray(inverse)]
    assert np.all(row_values[video_indices[:42]] == 1.0)
    assert np.all(row_values[video_indices[42:54]] == 0.999)


def test_trim_continuation_overlap_keeps_exact_audio_duration():
    images = np.zeros((124, 16, 16, 3), dtype=np.float32)
    waveform = np.zeros((1, 2, 166_000), dtype=np.float32)

    trimmed_images, trimmed_audio, info = trim_continuation_overlap(
        images, {"waveform": waveform, "sample_rate": 32000}, 22
    )

    assert trimmed_images.shape[0] == 102
    assert trimmed_audio["waveform"].shape == (1, 2, 136_000)
    assert info["context_samples_removed"] == 29_333
    assert info["audio_adjustment"] == "truncated"


def test_trim_continuation_overlap_zero_pads_short_subframe_tail():
    images = np.zeros((39, 16, 16, 3), dtype=np.float32)
    waveform = np.zeros((1, 2, 51_999), dtype=np.float32)

    _, trimmed_audio, info = trim_continuation_overlap(
        images, {"waveform": waveform, "sample_rate": 32000}, 22
    )

    assert trimmed_audio["waveform"].shape[-1] == round(17 / 24 * 32000)
    assert info["audio_adjustment"] == "zero_padded"
