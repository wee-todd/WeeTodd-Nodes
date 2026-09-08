"""DFR frozen-audio joint execution with cross-modality timestep routing."""

from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn


class _RoutedAdaLN(nn.Module):
    def __init__(self, base, embedding):
        super().__init__()
        self.base = base
        self.embedding = embedding

    def __call__(self, incoming):
        # _adaln_per_token flattens B*N before calling the module. Broadcast the
        # selected per-batch timestep to those rows, not the caller's modality.
        batch = self.embedding.shape[0]
        if incoming.shape[0] % batch:
            raise ValueError("DFR AdaLN timestep batch does not match the token batch.")
        selected = mx.repeat(self.embedding, incoming.shape[0] // batch, axis=0)
        return self.base(selected)


@contextmanager
def frozen_audio_timestep_routing(transformer, sigma):
    """Scope routing to this loaded model; restore all modules even on cancellation."""
    owner = transformer
    visited = set()
    while getattr(owner, "inner", None) is not None:
        if id(owner) in visited:
            raise ValueError("DFR transformer wrappers contain a cycle.")
        visited.add(id(owner))
        owner = owner.inner
    if not hasattr(owner, "audio_prompt_adaln_single"):
        raise ValueError("DFR requires joint LTX audio/video timestep modulation.")
    zero_embedding = owner._embed_timestep_scalar(mx.zeros_like(sigma))
    video_embedding = owner._embed_timestep_scalar(sigma.astype(mx.bfloat16))
    # Cross-timestep policy: video AV conditioning follows clean audio (0);
    # audio AV conditioning follows video sigma. Audio prompt modulation is 0.
    routes = {
        "audio_prompt_adaln_single": zero_embedding,
        "av_ca_video_scale_shift_adaln_single": zero_embedding,
        "av_ca_a2v_gate_adaln_single": zero_embedding,
        "av_ca_audio_scale_shift_adaln_single": video_embedding,
    }
    originals = {}
    try:
        for name, embedding in routes.items():
            original = getattr(owner, name)
            originals[name] = original
            setattr(owner, name, _RoutedAdaLN(original, embedding))
        yield
    finally:
        for name, original in originals.items():
            setattr(owner, name, original)


class DFRFrozenAudioX0Model:
    """Run both streams; keep the original model's video path and streaming intact."""

    def __init__(self, transformer):
        from ltx_core_mlx.model.transformer.model import X0Model

        self.transformer = transformer
        self.model = transformer  # Preserve the sampler's Sol Attention context lookup.
        self._x0_model = X0Model(transformer)

    def __call__(self, **kwargs):
        if kwargs.get("audio_latent") is None:
            raise ValueError("DFR temporal refinement requires frozen audio conditioning.")
        with frozen_audio_timestep_routing(self.transformer, kwargs["sigma"]):
            return self._x0_model(**kwargs)
