import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO
from minimax_h3_mlx.packing import build_row_timesteps
from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
from minimax_h3_mlx.ref2va import (
    PreparedReference,
    build_ref2va_packed_sequence,
    encode_reference_video_rows,
    trim_reference_num_frames,
    validate_reference_set,
)

PATCH = (1, 2, 2)


@pytest.mark.parametrize(
    ("frames", "expected"),
    [(5, 5), (6, 5), (21, 5), (22, 22), (38, 22), (39, 39), (124, 124)],
)
def test_reference_video_frame_count_snaps_down(frames, expected):
    assert trim_reference_num_frames(frames) == expected


def test_ref2va_image_encoding_uses_posterior_mean_not_sample():
    class Config:
        latent_channels = 1
        latents_mean = [0.0]
        latents_std = [1.0]

    class FakeVAE:
        config = Config()

        def _encode_clip(self, pixels):
            del pixels
            # Channel-last moments: mean=3, log-variance=20. A sampled posterior would be random
            # and far from three; Ref2VA must ignore the log-variance half.
            mean = mx.full((1, 1, 2, 2, 1), 3.0)
            logvar = mx.full((1, 1, 2, 2, 1), 20.0)
            return mx.concatenate([mean, logvar], axis=-1)

    reference = PreparedReference("image", image=Image.new("RGB", (32, 32)))
    rows = encode_reference_video_rows(FakeVAE(), [reference], PATCH)

    np.testing.assert_array_equal(np.asarray(rows), np.full((1, 4), 3.0, dtype=np.float32))
    assert (reference.num_latent_frames, reference.latent_height, reference.latent_width) == (
        1,
        2,
        2,
    )


def test_reference_set_rejects_audio_only_request():
    with pytest.raises(ValueError, match="at least one image or video"):
        validate_reference_set(
            [PreparedReference("audio", num_audio_latents=4)],
            PATCH,
        )


def test_reference_set_enforces_image_and_audio_limits():
    images = [PreparedReference("image", 1, 4, 4) for _ in range(10)]
    with pytest.raises(ValueError, match="at most 9 image"):
        validate_reference_set(images, PATCH)

    references = [PreparedReference("image", 1, 4, 4)] + [
        PreparedReference("audio", num_audio_latents=2) for _ in range(4)
    ]
    with pytest.raises(ValueError, match="at most 3 audio"):
        validate_reference_set(references, PATCH)


def test_ref2va_packing_preserves_reference_order_and_condition_prefixes():
    references = [
        PreparedReference("image", 1, 4, 4),
        PreparedReference("audio", num_audio_latents=2),
        PreparedReference("video", 2, 4, 4, num_audio_latents=3),
    ]
    layout = build_ref2va_packed_sequence(
        [TAG_TEXT, TAG_TEXT],
        references,
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=3,
        patch_size=PATCH,
    )

    # image 4 + standalone audio 4 + soundtrack 6 + video 8
    assert layout.num_condition_video_rows == 12
    assert layout.num_condition_audio_rows == 10
    assert layout.sequence_length == 38
    np.testing.assert_array_equal(np.asarray(layout.text_indices), [0, 1])
    np.testing.assert_array_equal(
        np.asarray(layout.video_indices),
        [2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35, 36, 37],
    )
    np.testing.assert_array_equal(
        np.asarray(layout.audio_indices),
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29],
    )
    tags = np.asarray(layout.token_tags)
    assert np.all(tags[np.asarray(layout.video_indices)] == TAG_VIDEO)
    assert np.all(tags[np.asarray(layout.audio_indices)] == TAG_AUDIO)

    positions = np.asarray(layout.position_ids)
    np.testing.assert_array_equal(positions[2:6, 0], np.full(4, 2.0))
    np.testing.assert_array_equal(positions[6:10, 0], [3.0, 4.0, 3.0, 4.0])
    np.testing.assert_array_equal(positions[10:16, 0], [5.0, 6.0, 7.0, 5.0, 6.0, 7.0])
    np.testing.assert_array_equal(positions[16:20, 0], np.full(4, 5.0))


def test_ref2va_row_timesteps_keep_all_reference_rows_conditioned():
    layout = build_ref2va_packed_sequence(
        [TAG_TEXT],
        [
            PreparedReference("image", 1, 4, 4),
            PreparedReference("video", 2, 4, 4, num_audio_latents=2),
        ],
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=2,
        patch_size=PATCH,
    )
    distinct, inverse = build_row_timesteps(layout, 0.5, 0.25, 0.999, 1.0)
    rows = mx.take(distinct, inverse)
    video_indices = np.asarray(layout.video_indices)
    audio_indices = np.asarray(layout.audio_indices)
    np.testing.assert_allclose(
        np.asarray(rows)[video_indices[: layout.num_condition_video_rows]],
        0.999,
    )
    np.testing.assert_array_equal(
        np.asarray(rows)[audio_indices[: layout.num_condition_audio_rows]],
        1.0,
    )
    np.testing.assert_array_equal(
        np.asarray(rows)[audio_indices[layout.num_condition_audio_rows :]],
        0.25,
    )


def test_ref2va_continuation_overlaps_target_after_ordered_references():
    layout = build_ref2va_packed_sequence(
        [TAG_TEXT],
        [
            PreparedReference("image", 1, 4, 4),
            PreparedReference("video", 2, 4, 4, num_audio_latents=2),
        ],
        num_latent_frames=3,
        latent_height=4,
        latent_width=4,
        num_audio_latents=4,
        patch_size=PATCH,
        continuation_video_frames=2,
        continuation_audio_latents=2,
    )
    positions = np.asarray(layout.position_ids)
    video_indices = np.asarray(layout.video_indices)
    audio_indices = np.asarray(layout.audio_indices)

    assert layout.num_continuation_video_rows == 8
    assert layout.num_continuation_audio_rows == 4
    assert layout.num_condition_video_rows == 20
    assert layout.num_condition_audio_rows == 8
    np.testing.assert_array_equal(
        positions[video_indices[:8], 0],
        positions[video_indices[20:28], 0],
    )
    np.testing.assert_array_equal(
        positions[audio_indices[:2], 0],
        positions[audio_indices[8:10], 0],
    )
    np.testing.assert_array_equal(
        positions[audio_indices[2:4], 0],
        positions[audio_indices[12:14], 0],
    )

    distinct, inverse = build_row_timesteps(layout, 0.5, 0.25, 0.999, 0.7)
    rows = np.asarray(mx.take(distinct, inverse))
    np.testing.assert_array_equal(rows[video_indices[:8]], 1.0)
    np.testing.assert_allclose(rows[video_indices[8:20]], 0.999)
    np.testing.assert_array_equal(rows[audio_indices[:4]], 1.0)
    np.testing.assert_allclose(rows[audio_indices[4:8]], 0.7)


def test_reference_strength_timestep_plan_tracks_schedule_above_requested_floor():
    layout = build_ref2va_packed_sequence(
        [TAG_TEXT],
        [PreparedReference("video", 1, 4, 4, num_audio_latents=1)],
        num_latent_frames=1,
        latent_height=4,
        latent_width=4,
        num_audio_latents=1,
        patch_size=PATCH,
    )
    table, plans = MiniMaxH3Pipeline._row_timestep_plan(
        None,
        layout,
        mx.array([0.8, 0.2]),
        mx.array([0.6, 0.1]),
        visual_condition_strength=0.7,
        audio_condition_strength=0.4,
    )
    first = mx.take(table, plans[0])
    second = mx.take(table, plans[1])
    video_indices = np.asarray(layout.video_indices)
    audio_indices = np.asarray(layout.audio_indices)

    np.testing.assert_allclose(
        np.asarray(first)[video_indices[: layout.num_condition_video_rows]], 0.8
    )
    np.testing.assert_allclose(
        np.asarray(second)[video_indices[: layout.num_condition_video_rows]], 0.7
    )
    np.testing.assert_allclose(
        np.asarray(first)[audio_indices[: layout.num_condition_audio_rows]], 0.6
    )
    np.testing.assert_allclose(
        np.asarray(second)[audio_indices[: layout.num_condition_audio_rows]], 0.4
    )
