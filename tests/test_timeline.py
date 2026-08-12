from types import SimpleNamespace

import pytest

from wee_todd_nodes.timeline import H3ChainedTimeline, H3LatentChain


def _latents(frames=107, width=896, height=512, spec="spec"):
    return SimpleNamespace(
        num_frames=frames,
        width=width,
        height=height,
        fps=24,
        sample_rate=32000,
        transformer_spec=spec,
    )


def test_four_window_timeline_maps_global_timestamps_and_exact_15s_trim():
    timeline = H3ChainedTimeline(107, 4, 22, target_frames=360)

    assert timeline.stride_frames == 85
    assert timeline.total_frames == 362
    assert timeline.published_frames == 360
    assert timeline.window_start_frame(4) == 255
    assert timeline.local_timestamp(2, 4.0) == pytest.approx(11 / 24)


def test_latent_chain_validates_window_geometry_and_completion():
    timeline = H3ChainedTimeline(107, 2, 22)
    chain = H3LatentChain(timeline).append(_latents()).append(_latents())
    chain.validate_complete()

    with pytest.raises(ValueError, match="canvas"):
        H3LatentChain(timeline).append(_latents()).append(_latents(width=960))
