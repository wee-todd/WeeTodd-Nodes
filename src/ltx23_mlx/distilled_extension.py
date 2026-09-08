"""Fast latent-native LTX 2.3 extension with the distilled transformer."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, noise_latent_state
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import (
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_positions,
)
from ltx_pipelines_mlx import DistilledPipeline, RetakePipeline
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.utils.samplers import denoise_loop


class LTX23DistilledExtendPipeline(DistilledPipeline):
    """Append/prepend latent frames with eight positive-only distilled evaluations."""

    def _encode_source_video(self, video_path):
        # The upstream helper is component-based and only relies on members also
        # owned by DistilledPipeline. Reuse it to keep source AV preprocessing exact.
        return RetakePipeline._encode_source_video(self, video_path)

    def extend_from_video(
        self,
        prompt: str,
        video_path: str | Path,
        extend_frames: int,
        direction: str = "after",
        seed: int = 42,
        num_steps: int = 8,
        **_unused,
    ) -> tuple[mx.array, mx.array]:
        video_latent, audio_latent, meta = self._encode_source_video(video_path)
        return self.extend(
            prompt=prompt,
            source_video_latent=video_latent,
            source_audio_latent=audio_latent,
            extend_frames=extend_frames,
            direction=direction,
            height=meta.height,
            width=meta.width,
            num_frames=meta.num_frames,
            frame_rate=meta.frame_rate,
            seed=seed,
            num_steps=num_steps,
        )

    def extend(
        self,
        *,
        prompt: str,
        source_video_latent: mx.array,
        source_audio_latent: mx.array,
        extend_frames: int,
        direction: str,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        seed: int,
        num_steps: int,
    ) -> tuple[mx.array, mx.array]:
        if direction not in {"before", "after"}:
            raise ValueError("LTX 2.3 distilled extension direction must be before or after")
        if not 1 <= num_steps <= len(DISTILLED_SIGMAS) - 1:
            raise ValueError("LTX 2.3 distilled extension requires one to eight steps")
        if extend_frames < 1:
            raise ValueError("LTX 2.3 distilled extension requires positive latent frames")

        self._load_text_encoder()
        video_embeds, audio_embeds = self._encode_text(prompt)
        mx.eval(video_embeds, audio_embeds)
        if self.low_memory:
            self.prompt_encoder.free()
            aggressive_cleanup()

        if self.dit is None:
            transformer = self.model_dir / "transformer.safetensors"
            if not transformer.exists():
                transformer = self._resolve_safetensors(
                    self.model_dir, "transformer-distilled"
                )
            self.dit = self._load_transformer_with_optional_streaming(transformer)
        assert self.dit is not None

        batch = int(source_video_latent.shape[0])
        source_f = int(source_video_latent.shape[2])
        _, latent_h, latent_w = compute_video_latent_shape(1, height, width)
        total_f = source_f + extend_frames
        source_video_tokens, _ = self.video_patchifier.patchify(source_video_latent)
        source_audio_tokens, source_audio_t = self.audio_patchifier.patchify(
            source_audio_latent
        )
        total_audio_t = compute_audio_token_count(
            num_frames + extend_frames * 8, frame_rate=frame_rate
        )
        extend_audio_t = total_audio_t - source_audio_t
        if extend_audio_t < 1:
            raise ValueError("LTX 2.3 distilled extension resolved no new audio tokens")

        new_video = mx.zeros(
            (batch, extend_frames * latent_h * latent_w, 128), dtype=mx.bfloat16
        )
        new_audio = mx.zeros((batch, extend_audio_t, 128), dtype=mx.bfloat16)
        frozen_video_mask = mx.zeros(
            (batch, source_video_tokens.shape[1], 1), dtype=mx.bfloat16
        )
        new_video_mask = mx.ones((batch, new_video.shape[1], 1), dtype=mx.bfloat16)
        frozen_audio_mask = mx.zeros((batch, source_audio_t, 1), dtype=mx.bfloat16)
        new_audio_mask = mx.ones((batch, extend_audio_t, 1), dtype=mx.bfloat16)

        if direction == "after":
            clean_video = mx.concatenate((source_video_tokens, new_video), axis=1)
            video_mask = mx.concatenate((frozen_video_mask, new_video_mask), axis=1)
            clean_audio = mx.concatenate((source_audio_tokens, new_audio), axis=1)
            audio_mask = mx.concatenate((frozen_audio_mask, new_audio_mask), axis=1)
        else:
            clean_video = mx.concatenate((new_video, source_video_tokens), axis=1)
            video_mask = mx.concatenate((new_video_mask, frozen_video_mask), axis=1)
            clean_audio = mx.concatenate((new_audio, source_audio_tokens), axis=1)
            audio_mask = mx.concatenate((new_audio_mask, frozen_audio_mask), axis=1)

        video_state = noise_latent_state(
            LatentState(
                latent=clean_video,
                clean_latent=clean_video,
                denoise_mask=video_mask,
                positions=compute_video_positions(
                    total_f, latent_h, latent_w, frame_rate=frame_rate
                ),
            ),
            sigma=1.0,
            seed=seed,
        )
        audio_state = noise_latent_state(
            LatentState(
                latent=clean_audio,
                clean_latent=clean_audio,
                denoise_mask=audio_mask,
                positions=compute_audio_positions(total_audio_t),
            ),
            sigma=1.0,
            seed=seed + 1,
        )
        self._pre_denoise_flush(video_state, audio_state)
        output = denoise_loop(
            model=X0Model(self.dit),
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=DISTILLED_SIGMAS[: num_steps + 1],
        )
        if self.low_memory:
            aggressive_cleanup()
        return (
            self.video_patchifier.unpatchify(
                output.video_latent, (total_f, latent_h, latent_w)
            ),
            self.audio_patchifier.unpatchify(output.audio_latent),
        )


__all__ = ["LTX23DistilledExtendPipeline"]
