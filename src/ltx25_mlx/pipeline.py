"""Project-native MLX pipeline for the official LTX 2.5 distilled workflow."""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx

from .components import (
    LTX25AudioDecoder,
    LTX25ImageConditioner,
    LTX25LatentNormalizer,
    LTX25VideoDecoder,
    load_ltx25_spatial_upsampler,
)
from .gemma_encoder import LTX25Gemma4Conditioner, resolve_prompt_context_length
from .runtime import LTX25_DISTILLED_SIGMAS, LTX25_STAGE2_SIGMAS
from .sampling import euler_ancestral_denoise_loop
from .transformer import load_ltx25_transformer


class _PromptEncoder:
    def __init__(self, text_path: str | Path, transformer_path: str | Path) -> None:
        self.conditioner = LTX25Gemma4Conditioner(text_path, connector_path=transformer_path)
        self._text_encoder = None
        self._feature_extractor = None

    def load(self) -> None:
        self.conditioner.load()
        self._text_encoder = self.conditioner.model
        self._feature_extractor = self.conditioner.feature_extractor

    def encode(self, prompt: str, prompt_context: str):
        resolved = resolve_prompt_context_length(self.conditioner.tokenizer, prompt, prompt_context)
        video, audio, _mask = self.conditioner.encode(prompt, max_length=resolved)
        return video, audio, resolved

    def free(self) -> None:
        self.conditioner.free()
        self._text_encoder = None
        self._feature_extractor = None


class LTX25DistilledPipeline:
    """Official two-stage LTX 2.5 distilled schedule with MLX lifecycle controls."""

    def __init__(
        self,
        *,
        transformer_path: str,
        text_encoder_path: str,
        video_vae_path: str,
        audio_vae_path: str,
        spatial_upscaler_path: str,
        duration_head_path: str = "",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        feed_forward_backend: str = "reference_fp32",
        feed_forward_stage_scope: str = "all",
        verbose: bool = True,
    ) -> None:
        del duration_head_path
        self.transformer_path = Path(transformer_path)
        self.spatial_upscaler_path = Path(spatial_upscaler_path)
        self.low_memory = low_memory
        self.low_ram_streaming = low_ram_streaming
        self.feed_forward_backend = feed_forward_backend
        if feed_forward_stage_scope not in {"all", "stage1", "stage2"}:
            raise ValueError("feed_forward_stage_scope must be all, stage1, or stage2")
        self.feed_forward_stage_scope = feed_forward_stage_scope
        self.verbose = verbose
        self.prompt_encoder = _PromptEncoder(text_encoder_path, transformer_path)
        self.image_conditioner = LTX25ImageConditioner(video_vae_path)
        self.latent_normalizer = LTX25LatentNormalizer(video_vae_path)
        self.video_decoder_block = LTX25VideoDecoder(video_vae_path, verbose=verbose)
        self.audio_decoder_block = LTX25AudioDecoder(audio_vae_path)
        self.dit = None
        self.upsampler = None
        self.last_timings: dict[str, object] = {}
        self.last_prompt_context: int | None = None
        self.feed_forward_report: dict[str, object] | None = None

        from ltx_core_mlx.components.patchifiers import (
            AudioPatchifier,
            VideoLatentPatchifier,
        )

        self.video_patchifier = VideoLatentPatchifier()
        self.audio_patchifier = AudioPatchifier()

    def load(self) -> None:
        if self.dit is None:
            self.dit = load_ltx25_transformer(
                self.transformer_path,
                low_ram_streaming=self.low_ram_streaming,
                feed_forward_backend=self.feed_forward_backend,
            )
            self.feed_forward_report = getattr(self.dit, "feed_forward_backend_report", None)
        if self.upsampler is None:
            self.upsampler = load_ltx25_spatial_upsampler(self.spatial_upscaler_path)

    def _release_sampling(self) -> None:
        self.dit = None
        self.upsampler = None
        self.image_conditioner.free()
        mx.clear_cache()

    def generate_two_stage(
        self,
        prompt: str,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        *,
        frame_rate: float,
        seed: int = 0,
        image: str | None = None,
        images=None,
        stage1_steps: int = 8,
        stage2_steps: int = 3,
        ancestral_noise_seed: int | None = None,
        check_interrupted=None,
        step_callback=None,
        prompt_context: str = "official_1024",
        **_unused,
    ):
        """Generate synchronized latents using official 8+3 stage semantics."""
        if stage1_steps != 8 or stage2_steps != 3:
            raise ValueError("LTX 2.5 distilled generation requires exactly 8+3 evaluations.")
        from ltx_core_mlx.components.patchifiers import (
            compute_video_latent_shape,
            snap_output_dimensions,
        )
        from ltx_core_mlx.model.transformer.model import X0Model
        from ltx_core_mlx.utils.memory import aggressive_cleanup
        from ltx_core_mlx.utils.positions import (
            compute_audio_positions,
            compute_audio_token_count,
            compute_video_positions,
        )
        from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
        from ltx_pipelines_mlx.utils.args import ImageConditioningInput
        from ltx_pipelines_mlx.utils.helpers import create_noised_state

        timings: dict[str, object] = {"stage1_evaluations": [], "stage2_evaluations": []}
        prompt_started = time.perf_counter()
        self.prompt_encoder.load()
        video_embeds, audio_embeds, resolved_context = self.prompt_encoder.encode(
            prompt, prompt_context
        )
        mx.eval(video_embeds, audio_embeds)
        timings["prompt_encode_seconds"] = time.perf_counter() - prompt_started
        self.last_prompt_context = resolved_context
        if self.low_memory:
            release_started = time.perf_counter()
            self.prompt_encoder.free()
            aggressive_cleanup()
            timings["prompt_release_seconds"] = time.perf_counter() - release_started
        sampling_load_started = time.perf_counter()
        self.load()
        timings["sampling_component_load_seconds"] = time.perf_counter() - sampling_load_started
        assert self.dit is not None and self.upsampler is not None
        height, width = snap_output_dimensions(height, width, two_stage=True)
        half_h, half_w = height // 2, width // 2
        latent_f, latent_h, latent_w = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, latent_f * latent_h * latent_w, 128)
        audio_tokens = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_tokens, 128)
        video_positions = compute_video_positions(
            latent_f, latent_h, latent_w, frame_rate=frame_rate
        )
        audio_positions = compute_audio_positions(audio_tokens)

        resolved_images = list(images) if images else []
        if image is not None and not resolved_images:
            resolved_images = [ImageConditioningInput(path=image, frame_idx=0, strength=1.0)]
        conditionings = []
        if resolved_images:
            encoder = self.image_conditioner.load()
            conditionings = combined_image_conditionings(
                resolved_images,
                enc_h=latent_h * 32,
                enc_w=latent_w * 32,
                spatial_dims=(latent_f, latent_h, latent_w),
                video_encoder=encoder,
                frame_rate=frame_rate,
            )
        video_state = create_noised_state(
            base_shape=video_shape,
            conditionings=conditionings,
            spatial_dims=(latent_f, latent_h, latent_w),
            positions=video_positions,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )
        audio_state = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(latent_f, latent_h, latent_w),
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )
        mx.eval(video_state.latent, video_state.clean_latent, audio_state.latent)
        model = X0Model(self.dit)
        from .feed_forward import set_mpp_feed_forward_enabled

        set_mpp_feed_forward_enabled(
            self.dit, self.feed_forward_stage_scope in {"all", "stage1"}
        )
        stage1_started = time.perf_counter()
        stage1 = euler_ancestral_denoise_loop(
            model,
            video_state,
            audio_state,
            video_embeds,
            audio_embeds,
            sigmas=LTX25_DISTILLED_SIGMAS,
            noise_seed=(seed + 10000 if ancestral_noise_seed is None else ancestral_noise_seed),
            eta=1.0,
            s_noise=1.0,
            check_interrupted=check_interrupted,
            step_callback=(
                (lambda completed, _total: step_callback(completed, 11))
                if step_callback is not None
                else None
            ),
            evaluation_timing_callback=(
                lambda index, elapsed: timings["stage1_evaluations"].append(
                    {"evaluation": index, "seconds": elapsed}
                )
            ),
        )
        mx.eval(stage1.video_latent, stage1.audio_latent)
        timings["stage1_seconds"] = time.perf_counter() - stage1_started

        upscale_started = time.perf_counter()
        generated = stage1.video_latent[:, : latent_f * latent_h * latent_w, :]
        video_half = self.video_patchifier.unpatchify(generated, (latent_f, latent_h, latent_w))
        denormalized = self.latent_normalizer.denormalize_latent(
            video_half.transpose(0, 2, 3, 4, 1)
        ).transpose(0, 4, 1, 2, 3)
        upscaled = self.upsampler(denormalized)
        upscaled = self.latent_normalizer.normalize_latent(
            upscaled.transpose(0, 2, 3, 4, 1)
        ).transpose(0, 4, 1, 2, 3)
        mx.eval(upscaled)
        timings["latent_upscale_seconds"] = time.perf_counter() - upscale_started
        full_h, full_w = latent_h * 2, latent_w * 2

        full_conditionings = []
        if resolved_images:
            encoder = self.image_conditioner.load()
            full_conditionings = combined_image_conditionings(
                resolved_images,
                enc_h=full_h * 32,
                enc_w=full_w * 32,
                spatial_dims=(latent_f, full_h, full_w),
                video_encoder=encoder,
                frame_rate=frame_rate,
            )
        if self.low_memory:
            self.image_conditioner.free()
            self.upsampler = None
            aggressive_cleanup()

        video_tokens, _ = self.video_patchifier.patchify(upscaled)
        start_sigma = LTX25_STAGE2_SIGMAS[0]
        video_state2 = create_noised_state(
            base_shape=video_tokens.shape,
            conditionings=full_conditionings,
            spatial_dims=(latent_f, full_h, full_w),
            positions=compute_video_positions(latent_f, full_h, full_w, frame_rate=frame_rate),
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens,
            legacy_scalar_blend=True,
        )
        audio_state2 = create_noised_state(
            base_shape=stage1.audio_latent.shape,
            conditionings=[],
            spatial_dims=(latent_f, full_h, full_w),
            positions=audio_positions,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=stage1.audio_latent,
        )
        mx.eval(video_state2.latent, video_state2.clean_latent, audio_state2.latent)
        set_mpp_feed_forward_enabled(
            self.dit, self.feed_forward_stage_scope in {"all", "stage2"}
        )
        stage2_started = time.perf_counter()
        stage2 = euler_ancestral_denoise_loop(
            model,
            video_state2,
            audio_state2,
            video_embeds,
            audio_embeds,
            sigmas=list(LTX25_STAGE2_SIGMAS),
            noise_seed=seed + 2,
            eta=1.0,
            s_noise=1.0,
            check_interrupted=check_interrupted,
            step_callback=(
                (lambda completed, _total: step_callback(8 + completed, 11))
                if step_callback is not None
                else None
            ),
            evaluation_timing_callback=(
                lambda index, elapsed: timings["stage2_evaluations"].append(
                    {"evaluation": index, "seconds": elapsed}
                )
            ),
        )
        mx.eval(stage2.video_latent, stage2.audio_latent)
        timings["stage2_seconds"] = time.perf_counter() - stage2_started
        video_latent = self.video_patchifier.unpatchify(
            stage2.video_latent[:, : latent_f * full_h * full_w, :],
            (latent_f, full_h, full_w),
        )
        audio_latent = self.audio_patchifier.unpatchify(stage2.audio_latent)
        mx.eval(video_latent, audio_latent)
        timings["sampling_total_seconds"] = sum(
            float(timings.get(name, 0.0))
            for name in (
                "prompt_encode_seconds",
                "prompt_release_seconds",
                "sampling_component_load_seconds",
                "stage1_seconds",
                "latent_upscale_seconds",
                "stage2_seconds",
            )
        )
        self.last_timings = timings
        return video_latent, audio_latent

    def generate_and_save(self, *, output_path: str, frame_rate: float, **kwargs) -> str:
        from ltx_pipelines_mlx.utils._orchestration import decode_and_save_video

        # The node adapter may pass explicit scheduler names and sigma arrays
        # after validating them. The pipeline owns the fixed official recipe.
        for ignored in (
            "stage1_sigmas",
            "stage2_sigmas",
            "stage1_sampler",
            "stage2_sampler",
            "stage1_eta",
            "stage1_s_noise",
        ):
            kwargs.pop(ignored, None)
        generation_started = time.perf_counter()
        video_latent, audio_latent = self.generate_two_stage(frame_rate=frame_rate, **kwargs)
        self.last_timings["generate_latents_seconds"] = time.perf_counter() - generation_started
        if self.low_memory:
            release_started = time.perf_counter()
            self._release_sampling()
            self.last_timings["sampling_release_seconds"] = time.perf_counter() - release_started
        decode_started = time.perf_counter()
        result = decode_and_save_video(
            self.video_decoder_block,
            self.audio_decoder_block,
            video_latent,
            audio_latent,
            output_path,
            frame_rate=frame_rate,
            low_memory=self.low_memory,
        )
        self.last_timings["decode_publish_seconds"] = time.perf_counter() - decode_started
        if self.low_memory:
            self.video_decoder_block.free()
            self.audio_decoder_block.free()
        return result


__all__ = ["LTX25DistilledPipeline"]
