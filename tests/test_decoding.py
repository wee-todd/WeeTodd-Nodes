from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wee_todd_nodes.decoding import (
    H3AudioVAECache,
    H3AudioVAESpec,
    H3VideoVAECache,
    H3VideoVAESpec,
)
from wee_todd_nodes.runtime import H3GenerationConfig
from wee_todd_nodes.sampling import H3Latents, H3TransformerSpec


class FakeVideoVAE:
    pass


class FakeVideoVAECache(H3VideoVAECache):
    def _decode_normalized(self, normalized, num_frames):
        return np.zeros((num_frames, 8, 12, 3), dtype=np.float32)


class FailingVideoVAECache(H3VideoVAECache):
    def _decode_normalized(self, normalized, num_frames):
        raise RuntimeError("synthetic decode failure")


class FakeAudioVAE:
    def __init__(self, sample_rate=32000):
        self.config = type("Config", (), {"sampling_rate": sample_rate})()


class FakeAudioVAECache(H3AudioVAECache):
    def _decode_normalized(self, normalized):
        return np.zeros((2, 1600), dtype=np.float32)


class FailingAudioVAECache(H3AudioVAECache):
    def _decode_normalized(self, normalized):
        raise RuntimeError("synthetic audio decode failure")


def _video_vae_spec(tmp_path: Path, name="video_vae") -> H3VideoVAESpec:
    path = tmp_path / name
    (path / "source").mkdir(parents=True)
    (path / "config.json").write_text("{}\n")
    (path / "source" / "config.json").write_text("{}\n")
    (path / "source" / "model.safetensors").write_bytes(b"test")
    return H3VideoVAESpec(str(path))


def _audio_vae_spec(tmp_path: Path, name="audio_vae") -> H3AudioVAESpec:
    path = tmp_path / name
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n")
    (path / "metadata.json").write_text("{}\n")
    (path / "model.safetensors").write_bytes(b"test")
    return H3AudioVAESpec(str(path))


def _latents(video_vae: str = "video_vae", audio_vae: str = "audio_vae") -> H3Latents:
    transformer_spec = H3TransformerSpec(
        checkpoint="checkpoint",
        transformer="transformer",
        text_encoder="text_encoder",
        processor="processor",
        tokenizer="tokenizer",
        video_vae=video_vae,
        audio_vae=audio_vae,
        task="t2va",
    )
    return H3Latents(
        video="normalized-video",
        audio="normalized-audio",
        num_frames=5,
        width=12,
        height=8,
        fps=24,
        sample_rate=32000,
        transformer_evaluations=2,
        seconds_per_evaluation=1.0,
        total_seconds=2.0,
        transformer_spec=transformer_spec,
        generation_config=H3GenerationConfig(),
    )


def test_video_vae_decode_returns_float_frames_and_unloads(tmp_path):
    created = []

    def factory(spec):
        created.append(spec)
        return FakeVideoVAE()

    spec = _video_vae_spec(tmp_path)
    cache = FakeVideoVAECache(factory)
    result = cache.decode(spec, _latents(spec.video_vae), unload_after=True)

    assert result.frames.shape == (5, 8, 12, 3)
    assert result.frames.dtype == np.float32
    assert result.fps == 24
    assert cache.loaded is False
    assert created == [spec]


def test_video_vae_cache_reuses_equal_spec(tmp_path):
    created = []
    spec = _video_vae_spec(tmp_path)
    cache = FakeVideoVAECache(lambda value: created.append(value) or FakeVideoVAE())

    cache.decode(spec, _latents(spec.video_vae), unload_after=False)
    cache.decode(spec, _latents(spec.video_vae), unload_after=False)

    assert len(created) == 1
    assert cache.loaded is True


def test_video_vae_decode_rejects_mismatched_provenance(tmp_path):
    spec = _video_vae_spec(tmp_path)
    cache = FakeVideoVAECache(lambda value: FakeVideoVAE())

    with pytest.raises(ValueError, match="different MiniMax H3 video VAE"):
        cache.decode(spec, _latents("other-video-vae"))

    assert cache.loaded is False


def test_video_vae_failure_unloads(tmp_path):
    spec = _video_vae_spec(tmp_path)
    cache = FailingVideoVAECache(lambda value: FakeVideoVAE())

    with pytest.raises(RuntimeError, match="synthetic decode failure"):
        cache.decode(spec, _latents(spec.video_vae), unload_after=False)

    assert cache.loaded is False


def test_audio_vae_decode_returns_stereo_timing_and_unloads(tmp_path):
    created = []

    def factory(spec):
        created.append(spec)
        return FakeAudioVAE()

    spec = _audio_vae_spec(tmp_path)
    cache = FakeAudioVAECache(factory)
    result = cache.decode(spec, _latents(audio_vae=spec.audio_vae), unload_after=True)

    assert result.waveform.shape == (2, 1600)
    assert result.waveform.dtype == np.float32
    assert result.sample_rate == 32000
    assert result.channels == 2
    assert result.duration_seconds == 0.05
    assert result.video_frames == 5
    assert result.fps == 24
    assert cache.loaded is False
    assert created == [spec]


def test_audio_vae_cache_reuses_equal_spec(tmp_path):
    created = []
    spec = _audio_vae_spec(tmp_path)
    cache = FakeAudioVAECache(lambda value: created.append(value) or FakeAudioVAE())

    cache.decode(spec, _latents(audio_vae=spec.audio_vae), unload_after=False)
    cache.decode(spec, _latents(audio_vae=spec.audio_vae), unload_after=False)

    assert len(created) == 1
    assert cache.loaded is True


def test_audio_vae_decode_rejects_mismatched_provenance(tmp_path):
    spec = _audio_vae_spec(tmp_path)
    cache = FakeAudioVAECache(lambda value: FakeAudioVAE())

    with pytest.raises(ValueError, match="different MiniMax H3 audio VAE"):
        cache.decode(spec, _latents(audio_vae="other-audio-vae"))

    assert cache.loaded is False


def test_audio_vae_decode_rejects_sample_rate_and_unloads(tmp_path):
    spec = _audio_vae_spec(tmp_path)
    cache = FakeAudioVAECache(lambda value: FakeAudioVAE(sample_rate=44100))

    with pytest.raises(ValueError, match="sample rate does not match"):
        cache.decode(spec, _latents(audio_vae=spec.audio_vae), unload_after=False)

    assert cache.loaded is False


def test_audio_vae_failure_unloads(tmp_path):
    spec = _audio_vae_spec(tmp_path)
    cache = FailingAudioVAECache(lambda value: FakeAudioVAE())

    with pytest.raises(RuntimeError, match="synthetic audio decode failure"):
        cache.decode(spec, _latents(audio_vae=spec.audio_vae), unload_after=False)

    assert cache.loaded is False


def test_audio_vae_cancellation_after_decode_unloads(tmp_path):
    spec = _audio_vae_spec(tmp_path)
    cache = FakeAudioVAECache(lambda value: FakeAudioVAE())
    checks = 0

    def cancel_after_decode():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("synthetic cancellation")

    with pytest.raises(RuntimeError, match="synthetic cancellation"):
        cache.decode(
            spec,
            _latents(audio_vae=spec.audio_vae),
            unload_after=False,
            check_interrupted=cancel_after_decode,
        )

    assert cache.loaded is False


def test_audio_vae_mlx_normalization_boundary(tmp_path):
    mx = pytest.importorskip("mlx.core")

    class TinyAudioVAE:
        config = type(
            "Config",
            (),
            {
                "sampling_rate": 32000,
                "latents_mean": (1.0, -1.0),
                "latents_std": (2.0, 0.5),
            },
        )()

        def decode(self, latents):
            return latents[:, :1, :]

    spec = _audio_vae_spec(tmp_path)
    cache = H3AudioVAECache(lambda value: TinyAudioVAE())
    latents = replace(
        _latents(audio_vae=spec.audio_vae), audio=mx.zeros((2, 2, 4))
    )

    result = cache.decode(spec, latents)

    assert result.waveform.shape == (2, 4)
    assert result.waveform.tolist() == [[1.0] * 4, [1.0] * 4]
