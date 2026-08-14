"""Project-native MLX pipeline for the official LTX 2.5 distilled workflow."""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx

from .chaining import (
    LTX25_CHAIN_CONTINUATION_STRENGTH,
    LatentGuideConditioning,
    LTX25LatentContinuation,
    assemble_ltx25_latents,
    plan_ltx25_chain,
)
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
        self.paged_checkpoint_report = None

    def load(self) -> None:
        self.conditioner.load()
        self._text_encoder = self.conditioner.model
        self._feature_extractor = self.conditioner.feature_extractor

    def encode(self, prompt: str, prompt_context: str):
        resolved = resolve_prompt_context_length(self.conditioner.tokenizer, prompt, prompt_context)
        video, audio, _mask = self.conditioner.encode(prompt, max_length=resolved)
        manifest = getattr(self.conditioner.model, "_weetodd_paged_manifest", None)
        if manifest is not None:
            prefetch = getattr(self.conditioner.model, "_weetodd_page_prefetch", None)
            self.paged_checkpoint_report = {
                "format": manifest.format,
                "bits": manifest.bits,
                "group_size": manifest.group_size,
                "fixed_bytes": manifest.fixed.tensor_bytes,
                "peak_layer_bytes": max(record.tensor_bytes for record in manifest.layers),
                **(prefetch.report() if prefetch is not None else {}),
            }
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
        self.paged_transformer_report: dict[str, object] | None = None

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
            self.paged_transformer_report = getattr(self.dit, "paged_checkpoint_report", None)
        if self.upsampler is None:
            self.upsampler = load_ltx25_spatial_upsampler(self.spatial_upscaler_path)

    def _release_sampling(self) -> None:
        if self.dit is not None:
            streamer = getattr(self.dit, "_weetodd_paged_streamer", None)
            if streamer is not None:
                window_report = getattr(self.dit, "streaming_window_report", None)
                self.paged_transformer_report = {
                    **(self.paged_transformer_report or {}),
                    **streamer.report(),
                    **(window_report() if window_report is not None else {}),
                }
                streamer.close()
        self.dit = None
        self.upsampler = None
        self.image_conditioner.free()
        mx.clear_cache()

    def encode_prompt_batch(
        self,
        prompts: list[str],
        *,
        prompt_context: str,
        check_interrupted=None,
    ) -> tuple[list[tuple[mx.array, mx.array, int]], dict[str, object]]:
        """Encode all chain prompts in one staged Gemma residency window."""
        from ltx_core_mlx.utils.memory import aggressive_cleanup

        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("Every LTX 2.5 chained window requires a non-empty prompt.")
        started = time.perf_counter()
        self.prompt_encoder.load()
        cache: dict[str, tuple[mx.array, mx.array, int]] = {}
        encoded: list[tuple[mx.array, mx.array, int]] = []
        unique_encodes = 0
        for prompt in prompts:
            if check_interrupted is not None:
                check_interrupted()
            result = cache.get(prompt)
            if result is None:
                result = self.prompt_encoder.encode(prompt, prompt_context)
                mx.eval(result[0], result[1])
                cache[prompt] = result
                unique_encodes += 1
            encoded.append(result)
        encode_seconds = time.perf_counter() - started
        release_seconds = 0.0
        if self.low_memory:
            release_started = time.perf_counter()
            self.prompt_encoder.free()
            aggressive_cleanup()
            release_seconds = time.perf_counter() - release_started
        return encoded, {
            "prompt_encode_seconds": encode_seconds,
            "prompt_release_seconds": release_seconds,
            "prompt_count": len(prompts),
            "unique_prompt_encodes": unique_encodes,
        }

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
        encoded_prompt: tuple[mx.array, mx.array, int] | None = None,
        continuation: LTX25LatentContinuation | None = None,
        continuation_strength: float = 1.0,
        output_video_context_frames: int = 0,
        output_audio_context_tokens: int = 0,
        return_continuation: bool = False,
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
        if encoded_prompt is None:
            self.prompt_encoder.load()
            video_embeds, audio_embeds, resolved_context = self.prompt_encoder.encode(
                prompt, prompt_context
            )
            mx.eval(video_embeds, audio_embeds)
        else:
            video_embeds, audio_embeds, resolved_context = encoded_prompt
        timings["prompt_encode_seconds"] = (
            time.perf_counter() - prompt_started if encoded_prompt is None else 0.0
        )
        self.last_prompt_context = resolved_context
        if self.low_memory and encoded_prompt is None:
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
        audio_conditionings = []
        if continuation is not None:
            low_prefix_tokens = continuation.video_latent_frames * latent_h * latent_w
            if continuation.stage1_video_tokens.shape[1] != low_prefix_tokens:
                raise ValueError(
                    "LTX 2.5 stage-one continuation does not match the target latent grid."
                )
            conditionings.insert(
                0,
                LatentGuideConditioning(
                    continuation.stage1_video_tokens,
                    strength=continuation_strength,
                ),
            )
            if continuation.audio_tokens.shape[1] != continuation.audio_token_count:
                raise ValueError("LTX 2.5 audio continuation metadata is inconsistent.")
            audio_conditionings.append(
                LatentGuideConditioning(
                    continuation.audio_tokens,
                    strength=continuation_strength,
                )
            )
        video_state = create_noised_state(
            base_shape=video_shape,
            conditionings=conditionings,
            spatial_dims=(latent_f, latent_h, latent_w),
            positions=video_positions,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=continuation is None,
        )
        audio_state = create_noised_state(
            base_shape=audio_shape,
            conditionings=audio_conditionings,
            spatial_dims=(latent_f, latent_h, latent_w),
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=continuation is None,
        )
        mx.eval(video_state.latent, video_state.clean_latent, audio_state.latent)
        model = X0Model(self.dit)
        from .feed_forward import set_mpp_feed_forward_enabled

        set_mpp_feed_forward_enabled(self.dit, self.feed_forward_stage_scope in {"all", "stage1"})
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
        stage1_audio_generated = stage1.audio_latent[:, :audio_tokens, :]
        stage1_tail = None
        if output_video_context_frames:
            tail_tokens = output_video_context_frames * latent_h * latent_w
            if tail_tokens >= generated.shape[1]:
                raise ValueError("LTX 2.5 continuation must be shorter than its source window.")
            stage1_tail = mx.contiguous(generated[:, -tail_tokens:, :])
            mx.eval(stage1_tail)
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
        audio_conditionings2 = []
        if continuation is not None:
            high_prefix_tokens = continuation.video_latent_frames * full_h * full_w
            if continuation.stage2_video_tokens.shape[1] != high_prefix_tokens:
                raise ValueError(
                    "LTX 2.5 stage-two continuation does not match the target latent grid."
                )
            full_conditionings.insert(
                0,
                LatentGuideConditioning(
                    continuation.stage2_video_tokens,
                    strength=continuation_strength,
                ),
            )
            audio_conditionings2.append(
                LatentGuideConditioning(
                    continuation.audio_tokens,
                    strength=continuation_strength,
                )
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
            legacy_scalar_blend=continuation is None,
        )
        audio_state2 = create_noised_state(
            base_shape=stage1_audio_generated.shape,
            conditionings=audio_conditionings2,
            spatial_dims=(latent_f, full_h, full_w),
            positions=audio_positions,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=stage1_audio_generated,
        )
        mx.eval(video_state2.latent, video_state2.clean_latent, audio_state2.latent)
        cleanup_started = time.perf_counter()
        # Stage two owns materialized copies of every value it needs. Drop all
        # stage-one states and upscaling intermediates before its larger spatial
        # transformer pass begins; retaining these references defeats staged
        # unloading even when the MLX cache itself is cleared.
        del (
            stage1,
            video_state,
            audio_state,
            generated,
            stage1_audio_generated,
            video_half,
            denormalized,
            upscaled,
            video_tokens,
            conditionings,
            full_conditionings,
        )
        aggressive_cleanup()
        timings["stage_boundary_cleanup_seconds"] = time.perf_counter() - cleanup_started
        set_mpp_feed_forward_enabled(self.dit, self.feed_forward_stage_scope in {"all", "stage2"})
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
        stage2_generated = stage2.video_latent[:, : latent_f * full_h * full_w, :]
        video_latent = self.video_patchifier.unpatchify(
            stage2_generated,
            (latent_f, full_h, full_w),
        )
        stage2_audio_generated = stage2.audio_latent[:, :audio_tokens, :]
        audio_latent = self.audio_patchifier.unpatchify(stage2_audio_generated)
        mx.eval(video_latent, audio_latent)
        next_continuation = None
        if return_continuation:
            if not output_video_context_frames or not output_audio_context_tokens:
                raise ValueError(
                    "LTX 2.5 continuation output requires positive video and audio context."
                )
            high_tail_tokens = output_video_context_frames * full_h * full_w
            if stage1_tail is None or high_tail_tokens >= stage2_generated.shape[1]:
                raise ValueError("LTX 2.5 continuation context is invalid for this window.")
            if output_audio_context_tokens >= stage2_audio_generated.shape[1]:
                raise ValueError("LTX 2.5 audio continuation is longer than its source window.")
            stage2_tail = mx.contiguous(stage2_generated[:, -high_tail_tokens:, :])
            audio_tail = mx.contiguous(stage2_audio_generated[:, -output_audio_context_tokens:, :])
            mx.eval(stage2_tail, audio_tail)
            next_continuation = LTX25LatentContinuation(
                stage1_video_tokens=stage1_tail,
                stage2_video_tokens=stage2_tail,
                audio_tokens=audio_tail,
                video_latent_frames=output_video_context_frames,
                audio_token_count=output_audio_context_tokens,
            )
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
        if return_continuation:
            return video_latent, audio_latent, next_continuation
        return video_latent, audio_latent

    def generate_chained_and_save(
        self,
        *,
        prompts: list[str],
        output_path: str,
        height: int,
        width: int,
        total_frames: int,
        window_count: int,
        overlap_frames: int,
        frame_rate: float,
        seed: int,
        check_interrupted=None,
        step_callback=None,
        prompt_context: str = "official_1024",
    ) -> str:
        """Generate, assemble, decode, and publish an exact latent-native chain."""
        if len(prompts) != window_count:
            raise ValueError("LTX 2.5 chained prompt count must match the window count.")
        plan = plan_ltx25_chain(
            total_frames=total_frames,
            window_count=window_count,
            overlap_frames=overlap_frames,
            frame_rate=frame_rate,
        )
        chain_started = time.perf_counter()
        encoded_prompts, prompt_timings = self.encode_prompt_batch(
            prompts,
            prompt_context=prompt_context,
            check_interrupted=check_interrupted,
        )
        video_windows = []
        audio_windows = []
        continuation = None
        window_timings = []
        total_evaluations = window_count * 11
        for index, (prompt, encoded) in enumerate(zip(prompts, encoded_prompts, strict=True)):
            if check_interrupted is not None:
                check_interrupted()
            next_audio_context = plan.join_audio_tokens[index] if index < window_count - 1 else 0
            window_started = time.perf_counter()
            result = self.generate_two_stage(
                prompt,
                height=height,
                width=width,
                num_frames=plan.window_frames,
                frame_rate=frame_rate,
                seed=seed + index,
                ancestral_noise_seed=seed + index + 10000,
                check_interrupted=check_interrupted,
                step_callback=(
                    (
                        lambda completed, _total, offset=index * 11: step_callback(
                            offset + completed, total_evaluations
                        )
                    )
                    if step_callback is not None
                    else None
                ),
                prompt_context=prompt_context,
                encoded_prompt=encoded,
                continuation=continuation,
                continuation_strength=LTX25_CHAIN_CONTINUATION_STRENGTH,
                output_video_context_frames=plan.video_overlap_latent_frames,
                output_audio_context_tokens=next_audio_context,
                return_continuation=index < window_count - 1,
            )
            if index < window_count - 1:
                video_latent, audio_latent, continuation = result
            else:
                video_latent, audio_latent = result
                continuation = None
            video_windows.append(video_latent)
            audio_windows.append(audio_latent)
            window_timings.append(
                {
                    "window": index + 1,
                    "seed": seed + index,
                    "seconds": time.perf_counter() - window_started,
                    "stage_timings": self.last_timings,
                }
            )

        del encoded_prompts, continuation
        if self.low_memory:
            release_started = time.perf_counter()
            self._release_sampling()
            release_seconds = time.perf_counter() - release_started
        else:
            release_seconds = 0.0
        assembly_started = time.perf_counter()
        video_latent, audio_latent = assemble_ltx25_latents(video_windows, audio_windows, plan)
        mx.eval(video_latent, audio_latent)
        assembly_seconds = time.perf_counter() - assembly_started
        del video_windows, audio_windows
        decode_started = time.perf_counter()
        from ltx_pipelines_mlx.utils._orchestration import decode_and_save_video

        decode_and_save_video(
            self.video_decoder_block,
            self.audio_decoder_block,
            video_latent,
            audio_latent,
            output_path,
            frame_rate=frame_rate,
            low_memory=self.low_memory,
        )
        decode_seconds = time.perf_counter() - decode_started
        del video_latent, audio_latent
        if self.low_memory:
            self.video_decoder_block.free()
            self.audio_decoder_block.free()
        self.last_timings = {
            **prompt_timings,
            "windows": window_timings,
            "latent_assembly_seconds": assembly_seconds,
            "sampling_release_seconds": release_seconds,
            "decode_publish_seconds": decode_seconds,
            "chain_total_seconds": time.perf_counter() - chain_started,
            "chain_plan": plan.as_dict(),
            "continuation_strength": LTX25_CHAIN_CONTINUATION_STRENGTH,
            "publication_mode": "single_decode_native_latent_chain",
            "video_join_mode": "causal_drop_plus_linear_latent_overlap",
            "audio_join_mode": "joint_latent_trim_then_single_decode",
        }
        return output_path

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
