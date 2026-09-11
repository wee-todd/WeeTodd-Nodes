"""Full-resolution distilled 1.1 T2V orchestration using the shared LTX sampler.

This is an explicit alternative to the two-stage distilled pipeline. It uses a
linear trailing flow schedule, positive-only audio/video guidance, and existing
MLX component loaders and decoders. No process-global scheduler replacement.
"""

from __future__ import annotations

import math

import mlx.core as mx
from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import (
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_positions,
)
from ltx_pipelines_mlx._base import BasePipeline
from ltx_pipelines_mlx.utils.helpers import create_noised_state
from ltx_pipelines_mlx.utils.samplers import guided_denoise_loop


def trailing_sigmas(steps: int, shift: float) -> list[float]:
    """One evaluation per interval, including a final step to zero noise."""
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError("Single-pass distilled steps must be an integer in [1, 100]")
    if not math.isfinite(shift) or not 1 <= shift <= 20:
        raise ValueError("Single-pass distilled Shift must be finite and in [1, 20]")
    times = [(steps - i) / steps for i in range(steps)]
    return [shift * t / (1 + (shift - 1) * t) for t in times] + [0.0]


class LTX23SingleStageDistilledPipeline(BasePipeline):
    """T2V only; staged cleanup also applies when sampling or decoding fails."""

    def _release_generation(self):
        self.dit = None
        self.prompt_encoder.free()
        self._loaded = False
        aggressive_cleanup()

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        height: int,
        width: int,
        num_frames: int,
        *,
        frame_rate: float,
        seed: int = 42,
        num_steps: int = 8,
        shift: float = 5.0,
    ) -> str:
        sigmas = trailing_sigmas(num_steps, shift)
        succeeded = False
        try:
            self._load_text_encoder()
            video_text, audio_text = self._encode_text(prompt)
            mx.eval(video_text, audio_text)
            if self.low_memory:
                self.prompt_encoder.free()
                aggressive_cleanup()
            if self.dit is None:
                self.dit = self._load_transformer_with_optional_streaming(
                    self.model_dir / "transformer-distilled-1.1.safetensors"
                )
            spatial = compute_video_latent_shape(num_frames, height, width)
            frames, rows, columns = spatial
            audio_tokens = compute_audio_token_count(num_frames, frame_rate=frame_rate)
            states = [
                create_noised_state(
                    base_shape=(1, tokens, 128),
                    conditionings=[],
                    spatial_dims=spatial,
                    positions=positions,
                    seed=noise_seed,
                    sigma=1.0,
                    legacy_scalar_blend=True,
                )
                for tokens, positions, noise_seed in (
                    (
                        frames * rows * columns,
                        compute_video_positions(*spatial, frame_rate=frame_rate),
                        seed,
                    ),
                    (audio_tokens, compute_audio_positions(audio_tokens), seed + 1),
                )
            ]
            self._pre_denoise_flush(*states)
            # All guidance is explicitly disabled for both modalities: eight
            # schedule intervals mean eight real transformer evaluations.
            guider = create_multimodal_guider_factory(
                MultiModalGuiderParams(cfg_scale=1, stg_scale=0, modality_scale=1, rescale_scale=0),
                negative_context=None,
            )
            result = guided_denoise_loop(
                model=X0Model(self.dit),
                video_state=states[0],
                audio_state=states[1],
                video_text_embeds=video_text,
                audio_text_embeds=audio_text,
                video_guider_factory=guider,
                audio_guider_factory=guider,
                sigmas=sigmas,
            )
            video = self.video_patchifier.unpatchify(result.video_latent, spatial)
            audio = self.audio_patchifier.unpatchify(result.audio_latent)
            mx.eval(video, audio)
            if self.low_memory:
                self._release_generation()
            # The shared decoder loads each component on demand.
            output = self._decode_and_save_video(video, audio, output_path, frame_rate=frame_rate)
            succeeded = True
            return output
        finally:
            if self.low_memory or not succeeded:
                self._release_generation()
                self.vae_decoder = None
                self.audio_decoder = None
                self.vocoder = None
                aggressive_cleanup()
