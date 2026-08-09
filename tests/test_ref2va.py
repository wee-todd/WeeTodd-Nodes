import mlx.core as mx
import numpy as np
import pytest

from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO
from minimax_h3_mlx.packing import build_row_timesteps
from minimax_h3_mlx.ref2va import (
    PreparedReference,
    build_ref2va_packed_sequence,
    validate_reference_set,
)

PATCH = (1, 2, 2)


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
