import pytest

from minimax_h3_mlx.continuation_cost import estimate_continuation_cost


def test_544p_22_frame_context_cost_matches_packed_geometry():
    baseline = estimate_continuation_cost(
        width=960,
        height=544,
        num_frames=124,
        text_rows=512,
        context_frames=0,
    )
    continuation = estimate_continuation_cost(
        width=960,
        height=544,
        num_frames=124,
        text_rows=512,
        context_frames=22,
    )

    assert baseline.rows_per_video_latent == 510
    assert baseline.target_video_rows == 18_870
    assert baseline.target_audio_rows == 414
    assert continuation.condition_video_rows == 3_570
    assert continuation.condition_audio_rows == 74
    assert continuation.sequence_rows == baseline.sequence_rows + 3_644
    assert continuation.mlp_macs / baseline.mlp_macs == pytest.approx(
        continuation.sequence_rows / baseline.sequence_rows
    )
    assert continuation.attention_macs / baseline.attention_macs == pytest.approx(
        (continuation.sequence_rows / baseline.sequence_rows) ** 2
    )


@pytest.mark.parametrize("context_frames", [0, 5, 22, 39, 56])
def test_supported_context_costs_are_monotonic(context_frames):
    values = [
        estimate_continuation_cost(
            width=640,
            height=384,
            num_frames=124,
            text_rows=64,
            context_frames=value,
        ).sequence_rows
        for value in [0, 5, 22, 39, 56]
    ]
    assert values == sorted(values)
    assert values[[0, 5, 22, 39, 56].index(context_frames)] > 0


def test_invalid_context_cost_is_rejected():
    with pytest.raises(ValueError, match="context"):
        estimate_continuation_cost(
            width=640,
            height=384,
            num_frames=124,
            text_rows=64,
            context_frames=20,
        )
