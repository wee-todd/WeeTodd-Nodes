"""Direct ComfyUI audio preparation for LTX 2.5 audio-to-video conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class LTX25AudioDrivenReport:
    source_sample_rate: int
    conditioning_sample_rate: int
    source_channels: int
    source_samples: int
    published_samples: int
    audio_tokens: int
    duration_seconds: float
    output_policy: str = "original_audio_trim_or_silence_pad"

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _host_audio(audio: Any):
    import numpy as np

    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("LTX 2.5 audio-driven conditioning requires a ComfyUI AUDIO input.")
    waveform = audio["waveform"]
    detach = getattr(waveform, "detach", None)
    if detach is not None:
        waveform = detach()
    cpu = getattr(waveform, "cpu", None)
    if cpu is not None:
        waveform = cpu()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in {1, 2}:
        raise ValueError(
            "ComfyUI AUDIO waveform must have shape (1, one-or-two channels, samples)."
        )
    sample_rate = int(audio["sample_rate"])
    if sample_rate < 1 or not np.isfinite(waveform).all():
        raise ValueError("LTX 2.5 audio input has an invalid sample rate or non-finite samples.")
    if waveform.shape[1] == 1:
        waveform = np.repeat(waveform, 2, axis=1)
    return np.ascontiguousarray(np.clip(waveform, -1.0, 1.0)), sample_rate


def _fit_samples(waveform, samples: int):
    import numpy as np

    if waveform.shape[-1] >= samples:
        return np.ascontiguousarray(waveform[..., :samples])
    return np.pad(waveform, ((0, 0), (0, 0), (0, samples - waveform.shape[-1])))


def _linear_resample(waveform, source_rate: int, target_rate: int):
    import numpy as np

    if source_rate == target_rate:
        return waveform
    source_samples = int(waveform.shape[-1])
    target_samples = max(1, round(source_samples * target_rate / source_rate))
    old = np.arange(source_samples, dtype=np.float64) / source_rate
    new = np.arange(target_samples, dtype=np.float64) / target_rate
    channels = [np.interp(new, old, waveform[0, channel]) for channel in range(2)]
    return np.asarray(channels, dtype=np.float32)[None]


def prepare_audio_driven_conditioning(
    *,
    audio: Any,
    audio_conditioner,
    audio_patchifier,
    target_tokens: int,
    duration_seconds: float,
):
    """Encode frozen audio tokens and retain a publication-quality source waveform."""
    import numpy as np
    from ltx_core_mlx.model.audio_vae import encode_audio

    waveform, sample_rate = _host_audio(audio)
    source_samples = int(waveform.shape[-1])
    publication_samples = max(1, round(duration_seconds * sample_rate))
    publication = _fit_samples(waveform, publication_samples)
    conditioning_rate = 16000
    conditioning = _linear_resample(publication, sample_rate, conditioning_rate)
    encoder, processor = audio_conditioner.load()
    latent = encode_audio(mx.array(conditioning), conditioning_rate, encoder, processor)
    tokens, _token_count = audio_patchifier.patchify(latent)
    if tokens.shape[1] < target_tokens:
        missing = target_tokens - tokens.shape[1]
        padding = mx.zeros((tokens.shape[0], missing, tokens.shape[2]), dtype=tokens.dtype)
        tokens = mx.concatenate([tokens, padding], axis=1)
    else:
        tokens = tokens[:, :target_tokens, :]
    tokens = mx.contiguous(tokens)
    mx.eval(tokens)
    report = LTX25AudioDrivenReport(
        source_sample_rate=sample_rate,
        conditioning_sample_rate=conditioning_rate,
        source_channels=int(waveform.shape[1]),
        source_samples=source_samples,
        published_samples=publication_samples,
        audio_tokens=int(tokens.shape[1]),
        duration_seconds=duration_seconds,
    )
    return tokens, mx.array(np.ascontiguousarray(publication)), report


__all__ = ["LTX25AudioDrivenReport", "prepare_audio_driven_conditioning"]
