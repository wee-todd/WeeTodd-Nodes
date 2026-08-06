from dataclasses import replace

import mlx.core as mx
import pytest

from minimax_h3_mlx.blockcache import H3BlockCacheConfig, H3BlockCacheState


def _hidden(scale=1.0):
    return mx.arange(1 * 8 * 8, dtype=mx.float32).reshape(1, 8, 8) * scale


def test_blockcache_reconstructs_video_and_audio_tail_independently():
    state = H3BlockCacheState(
        H3BlockCacheConfig(mode="manual", reuse_threshold=1.0, subsample_factor=2)
    )
    video = mx.array([0, 1, 2, 3], dtype=mx.int32)
    audio = mx.array([4, 5], dtype=mx.int32)
    before = _hidden()
    after_zero = before + 1
    after_stack = after_zero + 0
    after_stack[:, video] += 3
    after_stack[:, audio] += 5
    state.update(before, after_zero, after_stack, video, audio)

    current_before = before * 1.01
    current_zero = current_before + 1
    reused = state.try_reuse(current_before, current_zero, video, audio, 2, 8)

    assert reused is not None
    assert mx.array_equal(reused[:, video], current_zero[:, video] + 3)
    assert mx.array_equal(reused[:, audio], current_zero[:, audio] + 5)
    assert state.hits == 1
    assert state.cache_bytes > 0


def test_blockcache_uses_worst_modality_score_and_refreshes_after_rejection():
    state = H3BlockCacheState(H3BlockCacheConfig(mode="manual", reuse_threshold=0.01))
    video = mx.array([0, 1], dtype=mx.int32)
    audio = mx.array([2, 3], dtype=mx.int32)
    before = _hidden()
    after_zero = before + 1
    state.update(before, after_zero, after_zero + 2, video, audio)

    current_zero = before + 1
    current_zero[:, audio] += 10
    assert state.try_reuse(before, current_zero, video, audio, 2, 8) is None
    assert state.last_audio_score > state.last_video_score
    assert not state.last_was_hit


@pytest.mark.parametrize(
    ("mode", "lower", "upper"),
    [
        ("automatic_conservative", 0.035, 0.2),
        ("automatic_balanced", 0.07, 0.35),
        ("automatic_speed", 0.12, 0.6),
    ],
)
def test_blockcache_auto_policies_resolve_bounded_thresholds(mode, lower, upper):
    config = H3BlockCacheConfig(mode=mode)
    state = H3BlockCacheState(config)
    threshold = state._resolve_threshold(0.1)
    assert lower <= threshold <= upper


def test_blockcache_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="either EasyCache or BlockCache"):
        from wee_todd_nodes.nodes import WeeToddH3Sample

        WeeToddH3Sample().sample(
            None,
            None,
            None,
            True,
            easycache=object(),
            blockcache=object(),
        )

    with pytest.raises(ValueError, match="maximum hit fraction"):
        replace(H3BlockCacheConfig(), max_hit_fraction=0.7).validate()
