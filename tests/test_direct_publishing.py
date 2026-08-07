import json
from types import SimpleNamespace

import numpy as np
import pytest

from wee_todd_nodes.decoding import H3AudioWaveform, H3VideoStream
from wee_todd_nodes.direct_publishing import publish_latents_direct


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
