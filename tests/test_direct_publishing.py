import json
from types import SimpleNamespace

import numpy as np
import pytest

from wee_todd_nodes.decoding import H3AudioWaveform, H3VideoStream
from wee_todd_nodes.direct_publishing import (
    _motion_matched_overlap,
    _splice_audio_windows,
    publish_latent_chain_direct,
    publish_latents_direct,
)
from wee_todd_nodes.timeline import H3ChainedTimeline, H3LatentChain


class FakeVideoCache:
    def __init__(self):
        self.unloaded = False

    def decode_stream(self, spec, latents, write_chunk, **kwargs):
        write_chunk(np.zeros((2, latents.height, latents.width, 3), dtype=np.uint8))
        write_chunk(np.ones((3, latents.height, latents.width, 3), dtype=np.uint8))
        return H3VideoStream(5, latents.width, latents.height, 24, 0.1, 1, 288)

    def unload(self):
        self.unloaded = True


class FakeAudioCache:
    def __init__(self):
        self.unloaded = False

    def decode(self, spec, latents, **kwargs):
        samples = round(latents.num_frames / latents.fps * latents.sample_rate)
        waveform = np.zeros((2, samples), dtype=np.float32)
        return H3AudioWaveform(waveform, 32000, 2, samples, samples / 32000, 5, 24, 0.2)

    def unload(self):
        self.unloaded = True


class FakeEncoder:
    def __init__(self, path, width, height, fps, crf):
        self.path = path
        self.frames = 0
        self.aborted = False
        path.write_bytes(b"silent")

    def write(self, chunk):
        self.frames += chunk.shape[0]

    def close(self):
        pass

    def abort(self):
        self.aborted = True


def _inputs(tmp_path):
    video_vae = tmp_path / "video.safetensors"
    audio_vae = tmp_path / "audio.safetensors"
    video_vae.write_bytes(b"video")
    audio_vae.write_bytes(b"audio")
    components = SimpleNamespace(
        resolved_paths=lambda: {"video_vae": video_vae, "audio_vae": audio_vae}
    )
    transformer_spec = SimpleNamespace(video_vae=str(video_vae), audio_vae=str(audio_vae))
    latents = SimpleNamespace(
        width=12,
        height=8,
        num_frames=5,
        fps=24,
        sample_rate=32000,
        transformer_spec=transformer_spec,
    )
    return components, latents


def test_direct_publication_is_atomic_and_records_stream_metadata(tmp_path):
    components, latents = _inputs(tmp_path)
    video_cache = FakeVideoCache()
    audio_cache = FakeAudioCache()

    def muxer(video, audio, output):
        assert video.read_bytes() == b"silent"
        assert audio.is_file()
        output.write_bytes(b"muxed")

    result = publish_latents_direct(
        tmp_path / "result.mp4",
        components,
        latents,
        generation_metadata='{"seed": 7}',
        metadata_updates=lambda: {"staged_releases": {"video": ["transformer"]}},
        video_cache=video_cache,
        audio_cache=audio_cache,
        encoder_factory=FakeEncoder,
        muxer=muxer,
    )

    sidecar = json.loads(result.metadata_path.read_text())
    assert result.video_path.read_bytes() == b"muxed"
    assert sidecar == result.metadata
    assert sidecar["publication_mode"] == "direct_mlx_stream"
    assert sidecar["seed"] == 7
    assert sidecar["staged_releases"]["video"] == ["transformer"]
    assert video_cache.unloaded is True
    assert audio_cache.unloaded is True
    assert not list(tmp_path.glob(".*partial*"))


def test_direct_publication_failure_cleans_all_partial_files(tmp_path):
    components, latents = _inputs(tmp_path)
    video_cache = FakeVideoCache()
    audio_cache = FakeAudioCache()

    def failing_muxer(video, audio, output):
        output.write_bytes(b"partial")
        raise RuntimeError("synthetic mux failure")

    with pytest.raises(RuntimeError, match="synthetic mux failure"):
        publish_latents_direct(
            tmp_path / "result.mp4",
            components,
            latents,
            video_cache=video_cache,
            audio_cache=audio_cache,
            encoder_factory=FakeEncoder,
            muxer=failing_muxer,
        )

    assert not (tmp_path / "result.mp4").exists()
    assert not (tmp_path / "result.json").exists()
    assert not list(tmp_path.glob(".*partial*"))
    assert video_cache.unloaded is True
    assert audio_cache.unloaded is True


def test_direct_publication_rejects_non_finite_audio(tmp_path):
    components, latents = _inputs(tmp_path)

    class NonFiniteAudioCache(FakeAudioCache):
        def decode(self, spec, latents, **kwargs):
            result = super().decode(spec, latents, **kwargs)
            result.waveform[0, 0] = np.nan
            return result

    with pytest.raises(ValueError, match="non-finite"):
        publish_latents_direct(
            tmp_path / "result.mp4",
            components,
            latents,
            video_cache=FakeVideoCache(),
            audio_cache=NonFiniteAudioCache(),
            encoder_factory=FakeEncoder,
        )

    assert not (tmp_path / "result.mp4").exists()
    assert not list(tmp_path.glob(".*partial*"))


def test_direct_publication_unloads_when_ffmpeg_discovery_fails(monkeypatch, tmp_path):
    components, latents = _inputs(tmp_path)
    video_cache = FakeVideoCache()
    audio_cache = FakeAudioCache()
    monkeypatch.setattr(
        "wee_todd_nodes.direct_publishing.resolve_ffmpeg",
        lambda path=None: (_ for _ in ()).throw(RuntimeError("ffmpeg unavailable")),
    )

    with pytest.raises(RuntimeError, match="ffmpeg unavailable"):
        publish_latents_direct(
            tmp_path / "result.mp4",
            components,
            latents,
            video_cache=video_cache,
            audio_cache=audio_cache,
            encoder_factory=FakeEncoder,
        )

    assert video_cache.unloaded is True
    assert audio_cache.unloaded is True
    assert not list(tmp_path.glob(".*partial*"))


def test_direct_chain_publication_trims_video_audio_joins_and_target(tmp_path):
    components, base = _inputs(tmp_path)
    windows = tuple(SimpleNamespace(**{**base.__dict__, "num_frames": 22}) for _ in range(2))
    timeline = H3ChainedTimeline(22, 2, 5, target_frames=35)
    chain = H3LatentChain(timeline, windows)

    class ChainVideoCache(FakeVideoCache):
        def decode_stream(self, spec, latents, write_chunk, **kwargs):
            write_chunk(np.zeros((9, latents.height, latents.width, 3), dtype=np.uint8))
            write_chunk(np.ones((13, latents.height, latents.width, 3), dtype=np.uint8))
            return H3VideoStream(22, latents.width, latents.height, 24, 0.1, 1, 3744)

    class ChainAudioCache(FakeAudioCache):
        def decode(self, spec, latents, **kwargs):
            samples = round(22 / 24 * 32000)
            waveform = np.zeros((2, samples), dtype=np.float32)
            return H3AudioWaveform(waveform, 32000, 2, samples, samples / 32000, 22, 24, 0.2)

    def muxer(video, audio, output):
        output.write_bytes(b"chained")

    result = publish_latent_chain_direct(
        tmp_path / "chain.mp4",
        components,
        chain,
        video_cache=ChainVideoCache(),
        audio_cache=ChainAudioCache(),
        encoder_factory=FakeEncoder,
        muxer=muxer,
    )

    assert result.video_path.read_bytes() == b"chained"
    assert result.metadata["frames"] == 35
    assert result.metadata["audio_samples"] == round(35 / 24 * 32000)
    assert result.metadata["av_drift_seconds"] == pytest.approx(1 / 96000)
    assert result.metadata["video_windows"][1]["overlap_frames_reconciled"] == 5
    assert result.metadata["audio_windows"][1]["overlap_samples_reconciled"] == 6667
    assert result.metadata["join_policy"] == "motion_matched_video_and_50ms_cosine_audio"
    assert result.metadata["video_joins"][0]["blend_frames"] == 4
    assert result.metadata["audio_joins"][0]["crossfade_samples"] == 1600
    assert result.metadata["audio_adjustment"] == "truncated"
    assert not list(tmp_path.glob(".*partial*"))


def test_motion_matched_overlap_chooses_lower_cost_seam_and_blends():
    previous = np.zeros((6, 4, 4, 3), dtype=np.uint8)
    following = np.full_like(previous, 200)
    previous[2] = 90
    following[3] = 90
    previous_before = previous.copy()

    joined, report = _motion_matched_overlap(previous, following, blend_frames=2)

    assert report["seam_frame_in_overlap"] == 3
    assert report["seam_score"] < report["fixed_end_score"]
    assert np.array_equal(joined[0], previous_before[0])
    assert np.array_equal(joined[-1], following[-1])
    assert np.any(joined[2:4] != previous_before[2:4])


def test_audio_splice_uses_video_seam_and_cosine_crossfade():
    previous = np.zeros((2, 12), dtype=np.float32)
    following = np.ones((2, 12), dtype=np.float32)
    video_joins = [{"seam_frame_in_overlap": 3}]

    joined, reports = _splice_audio_windows(
        [previous, following],
        overlap_samples=6,
        video_joins=video_joins,
        context_frames=6,
        crossfade_samples=4,
    )

    assert joined.shape == (2, 18)
    assert reports[0]["seam_sample_in_overlap"] == 3
    assert reports[0]["crossfade_samples"] == 4
    assert np.max(np.abs(np.diff(joined[0]))) < 1.0
    assert np.allclose(joined[:, :6], 0.0)
    assert np.allclose(joined[:, -6:], 1.0)


def test_cosine_join_reaches_both_aligned_sources():
    previous = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    following = np.full_like(previous, 120)

    joined, _ = _motion_matched_overlap(previous, following, blend_frames=4)

    assert np.array_equal(joined[0], previous[0])
    assert np.array_equal(joined[3], following[3])
