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
    LTX25AudioConditioner,
    LTX25AudioDecoder,
    LTX25ImageConditioner,
    LTX25LatentNormalizer,
    LTX25VideoDecoder,
    load_ltx25_spatial_upsampler,
)
from .gemma_encoder import LTX25Gemma4Conditioner, resolve_prompt_context_length
from .runtime import LTX25_CFG_PP_SCHEDULES, LTX25_DISTILLED_SIGMAS, LTX25_STAGE2_SIGMAS
from .sampling import (
    euler_ancestral_cfg_pp_denoise_loop,
    euler_ancestral_denoise_loop,
)
from .transformer import load_ltx25_transformer, transformer_metadata


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
        spatial_upscaler_path: str = "",
        duration_head_path: str = "",
        distilled_lora_path: str = "",
        temporal_upsampler_path: str = "",
        dfr_stage2_transformer_path: str = "",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        feed_forward_backend: str = "reference_fp32",
        feed_forward_stage_scope: str = "all",
        diffvae_optimization: str = "combined",
        diffvae_query_chunk_size: int = 512,
        diffvae_context_width_chunks: int = 4,
        diffvae_stage4_tile_width: int = 0,
        sol_attention_profile: str = "disabled",
        loras: tuple[tuple[str, float], ...] = (),
        ic_loras: tuple[tuple[str, float], ...] = (),
        msr_lora_path: str = "",
        verbose: bool = True,
    ) -> None:
        self.transformer_path = Path(transformer_path)
        self.duration_head_path = Path(duration_head_path) if duration_head_path else None
        self.distilled_lora_path = (
            Path(distilled_lora_path) if distilled_lora_path else None
        )
        self.spatial_upscaler_path = (
            Path(spatial_upscaler_path) if spatial_upscaler_path else None
        )
        self.temporal_upsampler_path = (
            Path(temporal_upsampler_path) if temporal_upsampler_path else None
        )
        self.dfr_stage2_transformer_path = (
            Path(dfr_stage2_transformer_path) if dfr_stage2_transformer_path else None
        )
        self.low_memory = low_memory
        self.low_ram_streaming = low_ram_streaming
        self.feed_forward_backend = feed_forward_backend
        self.sol_attention_profile = sol_attention_profile
        self.sol_attention_report: dict[str, object] = {"enabled": False}
        self.loras = tuple(loras)
        self.ic_loras = tuple(ic_loras)
        self.msr_lora_path = Path(msr_lora_path) if msr_lora_path else None
        self.baked_ic_loras = tuple(
            item
            for item in transformer_metadata(self.transformer_path).get(
                "weetodd_baked_loras", ()
            )
            if item.get("adapter_role") == "ic_lora"
        )
        if feed_forward_stage_scope not in {"all", "stage1", "stage2"}:
            raise ValueError("feed_forward_stage_scope must be all, stage1, or stage2")
        self.feed_forward_stage_scope = feed_forward_stage_scope
        self.verbose = verbose
        self.prompt_encoder = _PromptEncoder(text_encoder_path, transformer_path)
        self.image_conditioner = LTX25ImageConditioner(video_vae_path)
        self.latent_normalizer = LTX25LatentNormalizer(video_vae_path)
        self.video_decoder_block = LTX25VideoDecoder(
            video_vae_path,
            verbose=verbose,
            diffvae_optimization=diffvae_optimization,
            diffvae_query_chunk_size=diffvae_query_chunk_size,
            diffvae_context_width_chunks=diffvae_context_width_chunks,
            diffvae_stage4_tile_width=diffvae_stage4_tile_width,
        )
        self.audio_decoder_block = LTX25AudioDecoder(audio_vae_path)
        self.audio_conditioner = LTX25AudioConditioner(audio_vae_path)
        self.dit = None
        self._loaded_loras = None
        self._loaded_transformer_path: Path | None = None
        self.upsampler = None
        self.temporal_upsampler = None
        self.duration_head = None
        self.last_timings: dict[str, object] = {}
        self.last_prompt_context: int | None = None
        self.feed_forward_report: dict[str, object] | None = None
        self.paged_transformer_report: dict[str, object] | None = None
        self.last_num_frames: int | None = None
        self.last_predicted_duration_seconds: float | None = None
        self.last_output_frame_rate: float | None = None
        self.last_passthrough_audio: tuple[mx.array, int] | None = None

        from ltx_core_mlx.components.patchifiers import (
            AudioPatchifier,
            VideoLatentPatchifier,
        )

        self.video_patchifier = VideoLatentPatchifier()
        self.audio_patchifier = AudioPatchifier()

    def load(
        self,
        *,
        extra_loras: tuple[tuple[str, float], ...] = (),
        include_ic_loras: bool = False,
        load_spatial_upscaler: bool = True,
    ) -> None:
        self._load_transformer(
            extra_loras=extra_loras,
            include_ic_loras=include_ic_loras,
        )
        if load_spatial_upscaler and self.upsampler is None:
            if self.spatial_upscaler_path is None:
                raise ValueError(
                    "The two-stage LTX 2.5 pipeline requires a spatial upscaler checkpoint."
                )
            self.upsampler = load_ltx25_spatial_upsampler(self.spatial_upscaler_path)

    def _load_transformer(
        self,
        *,
        extra_loras: tuple[tuple[str, float], ...] = (),
        transformer_path: Path | None = None,
        include_ic_loras: bool = False,
    ) -> None:
        desired_path = transformer_path or self.transformer_path
        desired_loras = (
            *self.loras,
            *(self.ic_loras if include_ic_loras else ()),
            *extra_loras,
        )
        if self.dit is not None and (
            self._loaded_loras != desired_loras
            or self._loaded_transformer_path != desired_path
        ):
            self._release_transformer()
        if self.dit is None:
            self.dit = load_ltx25_transformer(
                desired_path,
                low_ram_streaming=self.low_ram_streaming,
                feed_forward_backend=self.feed_forward_backend,
                loras=desired_loras,
            )
            from wee_todd_mlx.sol_attention import SolAttentionConfig

            from .sol_attention import configure_ltx25_sol_attention

            profiles = {
                "quality": (0.75, 0.30, 4, False),
                "balanced": (1.0, 0.25, 3, False),
                "speed": (1.25, 0.20, 2, False),
                "paged_speed": (1.25, 0.0, 0, False),
            }
            sol_config = None
            force_dense_bf16 = False
            sol_profile = getattr(self, "sol_attention_profile", "disabled")
            if sol_profile != "disabled":
                tau, start_percent, dense_blocks, force_dense_bf16 = profiles[sol_profile]
                sol_config = SolAttentionConfig(
                    enabled=True,
                    tau=tau,
                    min_tokens=16000,
                    start_percent=start_percent,
                    dense_blocks=dense_blocks,
                )
            self.sol_attention_report = configure_ltx25_sol_attention(
                self.dit,
                sol_config,
                force_dense_bf16=force_dense_bf16,
            )
            self._loaded_loras = desired_loras
            self._loaded_transformer_path = desired_path
            self.feed_forward_report = getattr(self.dit, "feed_forward_backend_report", None)
            self.paged_transformer_report = getattr(self.dit, "paged_checkpoint_report", None)

    def _release_transformer(self) -> None:
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
        self._loaded_loras = None
        self._loaded_transformer_path = None

    def _release_sampling(self) -> None:
        self._release_transformer()
        self.upsampler = None
        self.temporal_upsampler = None
        self.image_conditioner.free()
        self.audio_conditioner.free()
        mx.clear_cache()

    def _load_temporal_upsampler(self):
        if self.temporal_upsampler_path is None:
            raise ValueError("LTX 2.5 DFR temporal rounds require a temporal upsampler checkpoint.")
        if self.temporal_upsampler is None:
            from .components import load_ltx25_latent_upsampler

            model = load_ltx25_latent_upsampler(self.temporal_upsampler_path)
            if model.spatial_upsample or not model.temporal_upsample:
                raise ValueError(
                    "The selected LTX 2.5 checkpoint is not a temporal-only upsampler."
                )
            self.temporal_upsampler = model
        return self.temporal_upsampler

    def _temporal_upsample(self, latent: mx.array) -> mx.array:
        model = self._load_temporal_upsampler()
        denormalized = self.latent_normalizer.denormalize_latent(
            latent.transpose(0, 2, 3, 4, 1)
        ).transpose(0, 4, 1, 2, 3)
        upscaled = model(denormalized)
        normalized = self.latent_normalizer.normalize_latent(
            upscaled.transpose(0, 2, 3, 4, 1)
        ).transpose(0, 4, 1, 2, 3)
        mx.eval(normalized)
        return normalized

    def _run_dfr_temporal_rounds(
        self,
        *,
        video_latent: mx.array,
        carry_frames: tuple[int, ...],
        carry_keyframes: mx.array,
        image_anchors=(),
        num_frames: int,
        requested_num_frames: int,
        frame_rate: float,
        rounds: int,
        latent_h: int,
        latent_w: int,
        video_embeds: mx.array,
        audio_embeds: mx.array,
        seed: int,
        check_interrupted,
        step_callback,
        timings: dict[str, object],
    ) -> tuple[mx.array, int, float]:
        from ltx_core_mlx.components.patchifiers import VideoLatentPatchifier
        from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
        from ltx_core_mlx.conditioning.types.latent_cond import VideoConditionByLatentIndex
        from ltx_core_mlx.utils.positions import compute_video_positions
        from ltx_pipelines_mlx.utils.helpers import create_noised_state

        from .dfr import (
            plan_dfr_temporal_tiles,
            scale_dfr_temporal_image_anchors,
            select_dfr_generated_slot_tokens,
            stitch_dfr_temporal_tiles,
        )
        from .generated_keyframes import GeneratedKeyframeSlots, set_generated_keyframe_marker
        temporal_sigmas = LTX25_DISTILLED_SIGMAS[4:]
        temporal_reports = []
        completed_offset = 11
        current_fps = frame_rate
        patchifier = VideoLatentPatchifier()
        planned_frames = num_frames
        planned_seams = carry_frames
        total_temporal_evaluations = 0
        for level in range(1, rounds + 1):
            planned_frames = 2 * (planned_frames - 1) + 1
            planned_seams = tuple(2 * frame for frame in planned_seams)
            total_temporal_evaluations += 4 * len(
                plan_dfr_temporal_tiles(planned_seams, planned_frames, 2**level)
            )
        total_progress = 11 + total_temporal_evaluations
        for round_index in range(1, rounds + 1):
            round_started = time.perf_counter()
            if self.low_memory:
                self._release_transformer()
            video_latent = self._temporal_upsample(video_latent)
            if self.low_memory:
                from ltx_core_mlx.utils.memory import aggressive_cleanup

                self.temporal_upsampler = None
                aggressive_cleanup()
                reload_started = time.perf_counter()
                self._load_transformer()
                timings.setdefault("temporal_transformer_reload_seconds", []).append(
                    time.perf_counter() - reload_started
                )
            elif self._loaded_loras != self.loras:
                reload_started = time.perf_counter()
                self._load_transformer()
                timings.setdefault("temporal_transformer_reload_seconds", []).append(
                    time.perf_counter() - reload_started
                )
            from .video_only import LTX25VideoOnlyX0Model

            video_only_model = LTX25VideoOnlyX0Model(self.dit)
            num_frames = 2 * (num_frames - 1) + 1
            current_fps *= 2.0
            seam_frames = tuple(2 * frame for frame in carry_frames)
            seam_lookup = {frame: index for index, frame in enumerate(seam_frames)}
            image_anchors = scale_dfr_temporal_image_anchors(image_anchors)
            tiles = plan_dfr_temporal_tiles(seam_frames, num_frames, 2**round_index)
            tile_outputs = []
            slot_frames_all: list[int] = []
            slot_latents_all: list[mx.array] = []
            tile_reports = []
            conditioning_fps = min(current_fps, 60.0)
            for tile_index, tile in enumerate(tiles):
                if check_interrupted is not None:
                    check_interrupted()
                tile_started = time.perf_counter()
                tile_video = video_latent[
                    :, :, tile.latent_start : tile.latent_end_exclusive
                ]
                local_latent_frames = tile_video.shape[2]
                local_frames = (local_latent_frames - 1) * 8 + 1
                conditionings = []
                tile_image_anchors = tuple(
                    anchor
                    for anchor in image_anchors
                    if tile.pixel_start <= anchor.pixel_frame <= tile.pixel_end
                )
                explicit_frames = {anchor.pixel_frame for anchor in tile_image_anchors}
                for anchor in tile_image_anchors:
                    local_frame = anchor.pixel_frame - tile.pixel_start
                    if anchor.replace and local_frame == 0:
                        conditionings.append(
                            VideoConditionByLatentIndex(
                                frame_indices=[0],
                                clean_latent=anchor.latent_tokens,
                                strength=anchor.strength,
                            )
                        )
                for anchor in tile.anchor_frames:
                    if anchor in explicit_frames:
                        continue
                    if anchor not in seam_lookup:
                        raise RuntimeError("A DFR temporal anchor is missing from the carry bag.")
                    keyframe = carry_keyframes[
                        :, :, seam_lookup[anchor] : seam_lookup[anchor] + 1
                    ]
                    keyframe_tokens, _ = patchifier.patchify(keyframe)
                    conditionings.append(
                        VideoConditionByKeyframeIndex(
                            frame_idx=anchor - tile.pixel_start,
                            keyframe_latent=keyframe_tokens,
                            spatial_dims=(local_latent_frames, latent_h, latent_w),
                            frame_rate=conditioning_fps,
                            strength=0.95,
                        )
                    )
                for anchor in tile_image_anchors:
                    local_frame = anchor.pixel_frame - tile.pixel_start
                    if anchor.replace and local_frame == 0:
                        continue
                    conditionings.append(
                        VideoConditionByKeyframeIndex(
                            frame_idx=local_frame,
                            keyframe_latent=anchor.latent_tokens,
                            spatial_dims=(local_latent_frames, latent_h, latent_w),
                            frame_rate=conditioning_fps,
                            strength=anchor.strength,
                        )
                    )
                local_slots = tuple(
                    frame - tile.pixel_start
                    for frame in tile.slot_frames
                    if frame not in explicit_frames
                )
                slots = None
                if local_slots:
                    slot_initials = mx.concatenate(
                        [
                            tile_video[
                                :,
                                :,
                                min(max(round(frame / 8), 0), local_latent_frames - 1) :
                                min(max(round(frame / 8), 0), local_latent_frames - 1) + 1,
                            ]
                            for frame in local_slots
                        ],
                        axis=2,
                    )
                    slots = GeneratedKeyframeSlots(
                        local_slots,
                        spatial_dims=(local_latent_frames, latent_h, latent_w),
                        frame_rate=conditioning_fps,
                        initial_keyframes=slot_initials,
                    )
                    # Generated slots must remain last because their learned marker
                    # is applied to the final appended projection rows.
                    conditionings.append(slots)
                tile_tokens, _ = patchifier.patchify(tile_video)
                state = create_noised_state(
                    base_shape=tile_tokens.shape,
                    conditionings=conditionings,
                    spatial_dims=(local_latent_frames, latent_h, latent_w),
                    positions=compute_video_positions(
                        local_latent_frames,
                        latent_h,
                        latent_w,
                        frame_rate=conditioning_fps,
                    ),
                    seed=seed + round_index * 1000 + tile_index,
                    sigma=temporal_sigmas[0],
                    initial_latent=tile_tokens,
                )
                slot_rows = slots.token_count if slots is not None else 0
                set_generated_keyframe_marker(self.dit, slot_rows)
                evaluation_times = []
                try:
                    result = euler_ancestral_denoise_loop(
                        video_only_model,
                        state,
                        None,
                        video_embeds,
                        audio_embeds,
                        sigmas=temporal_sigmas,
                        noise_seed=seed + round_index * 1000 + tile_index,
                        eta=0.5,
                        check_interrupted=check_interrupted,
                        step_callback=(
                            (
                                lambda completed, _total, offset=completed_offset: step_callback(
                                    offset + completed,
                                    total_progress,
                                )
                            )
                            if step_callback is not None
                            else None
                        ),
                        evaluation_timing_callback=(
                            lambda index, elapsed, records=evaluation_times: records.append(
                                {"evaluation": index, "seconds": elapsed}
                            )
                        ),
                    )
                finally:
                    set_generated_keyframe_marker(self.dit, 0)
                generated_rows = local_latent_frames * latent_h * latent_w
                tile_output = patchifier.unpatchify(
                    result.video_latent[:, :generated_rows],
                    (local_latent_frames, latent_h, latent_w),
                )
                slot_output = None
                if slot_rows:
                    slot_output = patchifier.unpatchify(
                        select_dfr_generated_slot_tokens(result.video_latent, slot_rows),
                        (len(local_slots), latent_h, latent_w),
                    )
                    mx.eval(slot_output)
                mx.eval(tile_output)
                tile_outputs.append(tile_output)
                if slot_output is not None:
                    slot_frames_all.extend(
                        frame for frame in tile.slot_frames if frame not in explicit_frames
                    )
                    slot_latents_all.extend(
                        slot_output[:, :, index : index + 1]
                        for index in range(slot_output.shape[2])
                    )
                tile_reports.append(
                    {
                        "tile": tile_index + 1,
                        "frames": local_frames,
                        "seconds": time.perf_counter() - tile_started,
                        "evaluations": evaluation_times,
                    }
                )
                completed_offset += len(temporal_sigmas) - 1
            video_latent = stitch_dfr_temporal_tiles(tile_outputs, tiles)
            first_slots: dict[int, mx.array] = {}
            for frame, latent in zip(slot_frames_all, slot_latents_all, strict=True):
                first_slots.setdefault(frame, latent)
            carry_map = {
                frame: carry_keyframes[:, :, index : index + 1]
                for index, frame in enumerate(seam_frames)
            }
            carry_map.update(first_slots)
            carry_frames = tuple(sorted(carry_map))
            carry_keyframes = mx.concatenate([carry_map[frame] for frame in carry_frames], axis=2)
            mx.eval(video_latent, carry_keyframes)
            temporal_reports.append(
                {
                    "round": round_index,
                    "output_frames": num_frames,
                    "conditioning_fps": conditioning_fps,
                    "playback_fps": current_fps,
                    "tiles": tile_reports,
                    "seconds": time.perf_counter() - round_started,
                }
            )
        target_frames = (requested_num_frames - 1) * 2**rounds + 1
        target_latents = (target_frames - 1) // 8 + 1
        video_latent = video_latent[:, :, :target_latents]
        timings["temporal_rounds"] = temporal_reports
        return video_latent, target_frames, current_fps

    def _predict_num_frames(
        self,
        video_embeds: mx.array,
        audio_embeds: mx.array,
        *,
        frame_rate: float,
        min_seconds: float,
        max_seconds: float,
    ) -> int:
        if self.duration_head_path is None:
            raise ValueError(
                "Automatic LTX 2.5 duration requires the official duration-head checkpoint."
            )
        from .duration_head import load_ltx25_duration_head, seconds_to_ltx25_frames

        if self.duration_head is None:
            self.duration_head = load_ltx25_duration_head(self.duration_head_path)
        seconds_array = self.duration_head(video_embeds, audio_embeds)
        mx.eval(seconds_array)
        if seconds_array.shape != (1,):
            raise ValueError(
                "LTX 2.5 automatic duration supports one prompt at a time; "
                f"got {seconds_array.shape}."
            )
        seconds = float(seconds_array.item())
        self.last_predicted_duration_seconds = seconds
        return seconds_to_ltx25_frames(
            seconds,
            frame_rate=frame_rate,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
        )

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
        video_references=None,
        msr_references=None,
        audio_reference=None,
        stage1_steps: int = 8,
        stage2_steps: int = 3,
        ic_lora_single_stage: bool = False,
        stage1_sampler: str = "euler_ancestral",
        cfg_pp_batched: bool = False,
        cfg_pp_schedule: str = "full",
        stage2_sampler: str = "euler",
        stage1_eta: float = 1.0,
        stage1_s_noise: float = 1.0,
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
        generated_keyframes: int = 0,
        pipeline_mode: str = "distilled",
        negative_prompt: str = "",
        video_cfg_scale: float = 1.0,
        audio_cfg_scale: float = 1.0,
        stg_scale: float = 0.0,
        video_rescale_scale: float = 0.0,
        audio_rescale_scale: float = 0.0,
        modality_scale: float = 1.0,
        stg_blocks: tuple[int, ...] = (),
        dfr_enabled: bool = False,
        dfr_official_recipe: bool = False,
        dfr_detailing_lora: tuple[str, float] | None = None,
        temporal_upsample_rounds: int = 0,
        auto_duration: bool = False,
        auto_duration_min_seconds: float = 1.0,
        auto_duration_max_seconds: float = 20.0,
        **_unused,
    ):
        """Generate synchronized latents using a validated LTX 2.5 two-stage recipe."""
        if pipeline_mode not in {"distilled", "guided", "guided_hq"}:
            raise ValueError(f"Unsupported LTX 2.5 pipeline mode: {pipeline_mode!r}.")
        expected_stage1 = {"distilled": 8, "guided": 30, "guided_hq": 15}[pipeline_mode]
        expected_stage2 = 0 if ic_lora_single_stage else 3
        if stage1_steps != expected_stage1 or stage2_steps != expected_stage2:
            raise ValueError(
                f"LTX 2.5 {pipeline_mode} generation requires exactly "
                f"{expected_stage1}+{expected_stage2} sampler iterations."
            )
        if pipeline_mode == "distilled" and stage1_sampler not in {
            "euler_ancestral",
            "euler_ancestral_cfg_pp",
        }:
            raise ValueError(f"Unsupported distilled stage-one sampler: {stage1_sampler!r}.")
        if stage2_sampler != "euler":
            raise ValueError(f"Unsupported LTX 2.5 stage-two sampler: {stage2_sampler!r}.")
        if ic_lora_single_stage and pipeline_mode != "distilled":
            raise ValueError("LTX 2.5 IC-LoRA single-stage mode requires distilled sampling.")
        if ic_lora_single_stage and (dfr_enabled or temporal_upsample_rounds):
            raise ValueError("LTX 2.5 IC-LoRA single-stage mode cannot be combined with DFR.")
        if pipeline_mode != "distilled" and self.distilled_lora_path is None:
            raise ValueError("Guided LTX 2.5 generation requires a stage-two distilled LoRA.")
        if temporal_upsample_rounds not in {0, 1, 2}:
            raise ValueError("LTX 2.5 DFR temporal rounds must be zero, one, or two.")
        if temporal_upsample_rounds and not dfr_enabled:
            raise ValueError("LTX 2.5 temporal rounds require DFR detailing.")
        if temporal_upsample_rounds and (continuation is not None or return_continuation):
            raise ValueError(
                "LTX 2.5 temporal DFR is not yet available for chained timelines."
            )
        if audio_reference is not None and continuation is not None:
            raise ValueError("Audio-driven conditioning is not yet available in chained mode.")
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
        negative_video_embeds = None
        negative_audio_embeds = None
        if encoded_prompt is None:
            self.prompt_encoder.load()
            video_embeds, audio_embeds, resolved_context = self.prompt_encoder.encode(
                prompt, prompt_context
            )
            mx.eval(video_embeds, audio_embeds)
            needs_negative_context = (
                pipeline_mode != "distilled"
                or stage1_sampler == "euler_ancestral_cfg_pp"
            )
            if needs_negative_context:
                if pipeline_mode != "distilled" and not negative_prompt.strip():
                    raise ValueError("Guided LTX 2.5 generation requires a negative prompt.")
                negative_video_embeds, negative_audio_embeds, _negative_context = (
                    self.prompt_encoder.encode(negative_prompt, prompt_context)
                )
                mx.eval(negative_video_embeds, negative_audio_embeds)
        else:
            if pipeline_mode != "distilled" or stage1_sampler == "euler_ancestral_cfg_pp":
                raise ValueError(
                    "Pre-encoded chained prompts currently support the one-forward "
                    "distilled sampler only."
                )
            video_embeds, audio_embeds, resolved_context = encoded_prompt
        timings["prompt_encode_seconds"] = (
            time.perf_counter() - prompt_started if encoded_prompt is None else 0.0
        )
        self.last_prompt_context = resolved_context
        self.last_passthrough_audio = None
        self.last_predicted_duration_seconds = None
        if auto_duration:
            duration_started = time.perf_counter()
            num_frames = self._predict_num_frames(
                video_embeds,
                audio_embeds,
                frame_rate=frame_rate,
                min_seconds=auto_duration_min_seconds,
                max_seconds=auto_duration_max_seconds,
            )
            timings["duration_prediction_seconds"] = time.perf_counter() - duration_started
            timings["predicted_duration_seconds"] = self.last_predicted_duration_seconds
            timings["resolved_num_frames"] = num_frames
        self.last_num_frames = num_frames
        self.last_output_frame_rate = frame_rate
        if self.low_memory and encoded_prompt is None:
            release_started = time.perf_counter()
            self.prompt_encoder.free()
            self.duration_head = None
            aggressive_cleanup()
            timings["prompt_release_seconds"] = time.perf_counter() - release_started
        resolved_video_references = list(video_references or ())
        resolved_msr_references = list(msr_references or ())
        if resolved_msr_references:
            if self.msr_lora_path is None:
                raise ValueError("LTX 2.5 MSR references require the dedicated MSR loader.")
            if resolved_video_references:
                raise ValueError("Use either MSR references or an ordinary IC-LoRA guide.")
            if not ic_lora_single_stage:
                raise ValueError(
                    "LTX 2.5 MSR currently requires full-resolution single-stage mode."
                )
        if resolved_video_references and not (self.ic_loras or self.baked_ic_loras):
            raise ValueError(
                "LTX 2.5 video-reference conditioning requires a dedicated IC-LoRA loader."
            )
        requested_num_frames = num_frames
        dfr_slot_frames: tuple[int, ...] = ()
        if dfr_enabled:
            from .dfr import resolve_dfr_canvas

            num_frames, _segment, dfr_slot_frames = resolve_dfr_canvas(num_frames)
        height, width = snap_output_dimensions(
            height, width, two_stage=not ic_lora_single_stage
        )
        half_h, half_w = (
            (height, width) if ic_lora_single_stage else (height // 2, width // 2)
        )
        latent_f, latent_h, latent_w = compute_video_latent_shape(num_frames, half_h, half_w)
        requested_latent_f, _requested_h, _requested_w = compute_video_latent_shape(
            requested_num_frames, half_h, half_w
        )
        video_shape = (1, latent_f * latent_h * latent_w, 128)
        audio_tokens = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        requested_audio_tokens = compute_audio_token_count(
            requested_num_frames, frame_rate=frame_rate
        )
        audio_shape = (1, audio_tokens, 128)
        video_positions = compute_video_positions(
            latent_f, latent_h, latent_w, frame_rate=frame_rate
        )
        audio_positions = compute_audio_positions(audio_tokens)

        audio_reference_tokens = None
        if audio_reference is not None:
            from .audio_driven import prepare_audio_driven_conditioning

            audio_reference_tokens, passthrough, audio_report = (
                prepare_audio_driven_conditioning(
                    audio=audio_reference,
                    audio_conditioner=self.audio_conditioner,
                    audio_patchifier=self.audio_patchifier,
                    target_tokens=audio_tokens,
                    duration_seconds=num_frames / frame_rate,
                )
            )
            self.last_passthrough_audio = (
                passthrough,
                int(audio_report.source_sample_rate),
            )
            timings["audio_driven_conditioning"] = audio_report.as_dict()
            if self.low_memory:
                self.audio_conditioner.free()
                aggressive_cleanup()

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
        ic_reference_reports = []
        msr_group_rows: tuple[int, ...] = ()
        if resolved_video_references:
            from .ic_lora import encode_reference_video_conditioning

            encoder = self.image_conditioner.load()
            from .transformer import inspect_ltx25_lora

            adapter_reports = (
                [inspect_ltx25_lora(path) for path, _ in self.ic_loras]
                if self.ic_loras
                else list(self.baked_ic_loras)
            )
            spatial_scales = {
                int(adapter["reference_downscale_factor"]) for adapter in adapter_reports
            }
            temporal_scales = {
                int(adapter["reference_temporal_scale_factor"]) for adapter in adapter_reports
            }
            if len(spatial_scales) != 1 or len(temporal_scales) != 1:
                raise ValueError(
                    "Stacked LTX 2.5 IC-LoRAs must use identical reference scale factors."
                )
            for reference in resolved_video_references:
                conditioning, reference_report = encode_reference_video_conditioning(
                    images=reference["images"],
                    video_encoder=encoder,
                    target_height=half_h,
                    target_width=half_w,
                    target_num_frames=num_frames,
                    frame_rate=frame_rate,
                    start_frame=int(reference["start_frame"]),
                    end_frame=int(reference["end_frame"]),
                    strength=float(reference["strength"]),
                    attention_strength=float(reference["attention_strength"]),
                    attention_mask=reference.get("mask"),
                    reference_downscale_factor=next(iter(spatial_scales)),
                    reference_temporal_scale_factor=next(iter(temporal_scales)),
                    control_type=str(
                        reference.get("control_type", "custom_preprocessed")
                    ),
                    reference_size_policy=str(
                        reference.get("reference_size_policy", "quality")
                    ),
                    compact_attention_mask=(
                        ic_lora_single_stage
                        and self.sol_attention_profile != "disabled"
                    ),
                )
                conditionings.append(conditioning)
                ic_reference_reports.append(reference_report.as_dict())
            timings["ic_lora_references"] = ic_reference_reports
        if resolved_msr_references:
            from .msr import encode_ltx25_msr_references

            encoder = self.image_conditioner.load()
            msr_conditioning, msr_reports = encode_ltx25_msr_references(
                references=tuple(resolved_msr_references),
                video_encoder=encoder,
                slot_checkpoint=self.msr_lora_path,
                target_height=half_h,
                target_width=half_w,
                frame_rate=frame_rate,
                compact_attention_mask=self.sol_attention_profile != "disabled",
            )
            conditionings.append(msr_conditioning)
            msr_group_rows = msr_conditioning.group_rows
            timings["msr_references"] = [report.as_dict() for report in msr_reports]
        if self.low_memory and (
            resolved_images or resolved_video_references or resolved_msr_references
        ):
            reference_release_started = time.perf_counter()
            self.image_conditioner.free()
            aggressive_cleanup()
            timings["conditioning_encoder_release_seconds"] = (
                time.perf_counter() - reference_release_started
            )
        sampling_load_started = time.perf_counter()
        self.load(
            include_ic_loras=bool(resolved_video_references or resolved_msr_references),
            load_spatial_upscaler=not ic_lora_single_stage,
        )
        timings["sampling_component_load_seconds"] = time.perf_counter() - sampling_load_started
        assert self.dit is not None
        if not ic_lora_single_stage:
            assert self.upsampler is not None
        stage1_generated_slot_rows = 0
        if generated_keyframes or dfr_slot_frames:
            from .generated_keyframes import (
                GeneratedKeyframeSlots,
                evenly_spaced_keyframe_positions,
                set_generated_keyframe_marker,
            )

            slot_frames = (
                dfr_slot_frames
                if dfr_slot_frames
                else evenly_spaced_keyframe_positions(generated_keyframes, num_frames)
            )
            slots = GeneratedKeyframeSlots(
                slot_frames,
                spatial_dims=(latent_f, latent_h, latent_w),
                frame_rate=frame_rate,
            )
            conditionings.append(slots)
            stage1_generated_slot_rows = slots.token_count
            set_generated_keyframe_marker(self.dit, stage1_generated_slot_rows)
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
        if audio_reference_tokens is not None:
            from ltx_core_mlx.conditioning.types.latent_cond import LatentState

            audio_state = LatentState(
                latent=audio_reference_tokens,
                clean_latent=audio_reference_tokens,
                denoise_mask=mx.zeros((1, audio_tokens, 1), dtype=audio_reference_tokens.dtype),
                positions=audio_positions,
            )
        else:
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
        if ic_lora_single_stage and self.sol_attention_profile != "disabled":
            from .sol_attention import set_ltx25_sol_context

            target_tokens = requested_latent_f * latent_h * latent_w
            set_ltx25_sol_context(
                model,
                step_index=0,
                total_steps=stage1_steps,
                exact_suffix_rows=max(0, int(video_state.latent.shape[1]) - target_tokens),
                exact_suffix_groups=msr_group_rows,
            )
        from .feed_forward import set_mpp_feed_forward_enabled

        set_mpp_feed_forward_enabled(self.dit, self.feed_forward_stage_scope in {"all", "stage1"})
        stage1_started = time.perf_counter()
        try:
            if pipeline_mode == "distilled":
                sampler_kwargs = {
                    "sigmas": LTX25_DISTILLED_SIGMAS,
                    "noise_seed": (
                        seed + 10000
                        if ancestral_noise_seed is None
                        else ancestral_noise_seed
                    ),
                    # Official DFR attaches the rank-450 distilled adapter to the
                    # development transformer and uses deterministic Euler in both
                    # spatial stages. Fused distilled checkpoints retain their
                    # ordinary ancestral first stage.
                    "eta": 0.0 if dfr_official_recipe else stage1_eta,
                    "s_noise": stage1_s_noise,
                    "check_interrupted": check_interrupted,
                    "step_callback": (
                        (
                            lambda completed, _total: step_callback(
                                completed, stage1_steps + stage2_steps
                            )
                        )
                        if step_callback is not None
                        else None
                    ),
                    "evaluation_timing_callback": (
                        lambda index, elapsed: timings["stage1_evaluations"].append(
                            {"evaluation": index, "seconds": elapsed}
                        )
                    ),
                }
                if stage1_sampler == "euler_ancestral_cfg_pp":
                    assert negative_video_embeds is not None
                    assert negative_audio_embeds is not None
                    timings["stage1_forwards"] = []
                    stage1 = euler_ancestral_cfg_pp_denoise_loop(
                        model,
                        video_state,
                        audio_state,
                        video_embeds,
                        audio_embeds,
                        negative_video_embeds,
                        negative_audio_embeds,
                        forward_timing_callback=(
                            lambda index, kind, elapsed: timings["stage1_forwards"].append(
                                {
                                    "forward": index,
                                    "conditioning": kind,
                                    "seconds": elapsed,
                                }
                            )
                        ),
                        batch_branches=cfg_pp_batched,
                        cfg_pp_step_indices=LTX25_CFG_PP_SCHEDULES[cfg_pp_schedule],
                        **sampler_kwargs,
                    )
                else:
                    stage1 = euler_ancestral_denoise_loop(
                        model,
                        video_state,
                        audio_state,
                        video_embeds,
                        audio_embeds,
                        **sampler_kwargs,
                    )
                timings["stage1_sampler"] = stage1_sampler
                timings["stage1_sampler_iterations"] = stage1_steps
                timings["stage1_real_forwards"] = (
                    stage1_steps + len(LTX25_CFG_PP_SCHEDULES[cfg_pp_schedule])
                    if stage1_sampler == "euler_ancestral_cfg_pp"
                    else stage1_steps
                )
                timings["stage1_physical_invocations"] = (
                    stage1_steps
                    if stage1_sampler == "euler_ancestral_cfg_pp" and cfg_pp_batched
                    else timings["stage1_real_forwards"]
                )
            else:
                from ltx_core_mlx.components.guiders import (
                    MultiModalGuiderParams,
                    create_multimodal_guider_factory,
                )
                from ltx_pipelines_mlx.scheduler import ltx2_schedule
                from ltx_pipelines_mlx.utils.samplers import (
                    guided_denoise_loop,
                    res2s_denoise_loop,
                )

                video_params = MultiModalGuiderParams(
                    cfg_scale=video_cfg_scale,
                    stg_scale=stg_scale,
                    rescale_scale=video_rescale_scale,
                    modality_scale=modality_scale,
                    stg_blocks=list(stg_blocks),
                )
                audio_params = MultiModalGuiderParams(
                    cfg_scale=audio_cfg_scale,
                    stg_scale=stg_scale,
                    rescale_scale=audio_rescale_scale,
                    modality_scale=modality_scale,
                    stg_blocks=list(stg_blocks),
                )
                video_factory = create_multimodal_guider_factory(
                    video_params, negative_context=negative_video_embeds
                )
                audio_factory = create_multimodal_guider_factory(
                    audio_params, negative_context=negative_audio_embeds
                )
                sigmas = ltx2_schedule(
                    stage1_steps, num_tokens=latent_f * latent_h * latent_w
                )
                sampler = (
                    res2s_denoise_loop
                    if pipeline_mode == "guided_hq"
                    else guided_denoise_loop
                )
                stage1 = sampler(
                    model=model,
                    video_state=video_state,
                    audio_state=audio_state,
                    video_text_embeds=video_embeds,
                    audio_text_embeds=audio_embeds,
                    video_guider_factory=video_factory,
                    audio_guider_factory=audio_factory,
                    sigmas=sigmas,
                    show_progress=True,
                )
                timings["stage1_sampler"] = (
                    "res_2s_guided" if pipeline_mode == "guided_hq" else "euler_guided"
                )
                timings["stage1_sampler_iterations"] = stage1_steps
        finally:
            if stage1_generated_slot_rows:
                set_generated_keyframe_marker(self.dit, 0)
        mx.eval(stage1.video_latent, stage1.audio_latent)
        timings["stage1_seconds"] = time.perf_counter() - stage1_started

        if ic_lora_single_stage:
            target_tokens = requested_latent_f * latent_h * latent_w
            generated_stage1 = stage1.video_latent[:, :target_tokens, :]
            generated_audio = stage1.audio_latent[:, :requested_audio_tokens, :]
            video_latent = self.video_patchifier.unpatchify(
                generated_stage1,
                (requested_latent_f, latent_h, latent_w),
            )
            audio_latent = self.audio_patchifier.unpatchify(
                generated_audio
            )
            mx.eval(video_latent, audio_latent)
            timings["ic_lora_stage_scope"] = "single_stage_full_resolution"
            from .sol_attention import ltx25_sol_attention_report

            self.sol_attention_report = ltx25_sol_attention_report(
                self.dit, self.sol_attention_report
            )
            timings["sol_attention"] = dict(self.sol_attention_report)
            timings["latent_upscale_seconds"] = 0.0
            timings["stage2_seconds"] = 0.0
            timings["sampling_total_seconds"] = sum(
                float(timings.get(name, 0.0))
                for name in (
                    "prompt_encode_seconds",
                    "prompt_release_seconds",
                    "sampling_component_load_seconds",
                    "stage1_seconds",
                )
            )
            self.last_timings = timings
            if return_continuation:
                if not output_video_context_frames or not output_audio_context_tokens:
                    raise ValueError(
                        "LTX 2.5 continuation output requires positive video and audio context."
                    )
                tail_tokens = output_video_context_frames * latent_h * latent_w
                if tail_tokens >= generated_stage1.shape[1]:
                    raise ValueError("LTX 2.5 continuation context is invalid for this window.")
                if output_audio_context_tokens >= generated_audio.shape[1]:
                    raise ValueError(
                        "LTX 2.5 audio continuation is longer than its source window."
                    )
                stage1_tail = mx.contiguous(generated_stage1[:, -tail_tokens:, :])
                audio_tail = mx.contiguous(
                    generated_audio[:, -output_audio_context_tokens:, :]
                )
                mx.eval(stage1_tail, audio_tail)
                continuation_result = LTX25LatentContinuation(
                    stage1_video_tokens=stage1_tail,
                    stage2_video_tokens=stage1_tail,
                    audio_tokens=audio_tail,
                    video_latent_frames=output_video_context_frames,
                    audio_token_count=output_audio_context_tokens,
                )
                return video_latent, audio_latent, continuation_result
            return video_latent, audio_latent

        upscale_started = time.perf_counter()
        generated = stage1.video_latent[:, : latent_f * latent_h * latent_w, :]
        stage1_audio_generated = stage1.audio_latent[:, :audio_tokens, :]
        stage1_slot_tokens = None
        if dfr_enabled:
            from .dfr import select_dfr_generated_slot_tokens

            stage1_slot_tokens = mx.contiguous(
                select_dfr_generated_slot_tokens(
                    stage1.video_latent, stage1_generated_slot_rows
                )
            )
        dfr_audio_tokens = (
            mx.contiguous(stage1_audio_generated[:, :requested_audio_tokens, :])
            if dfr_enabled
            else None
        )
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
        upscaled_slot_keyframes = None
        if stage1_slot_tokens is not None:
            slot_latent = self.video_patchifier.unpatchify(
                stage1_slot_tokens,
                (len(dfr_slot_frames), latent_h, latent_w),
            )
            denormalized_slots = self.latent_normalizer.denormalize_latent(
                slot_latent.transpose(0, 2, 3, 4, 1)
            ).transpose(0, 4, 1, 2, 3)
            upscaled_slot_keyframes = self.upsampler(denormalized_slots)
            upscaled_slot_keyframes = self.latent_normalizer.normalize_latent(
                upscaled_slot_keyframes.transpose(0, 2, 3, 4, 1)
            ).transpose(0, 4, 1, 2, 3)
            mx.eval(upscaled_slot_keyframes)
        timings["latent_upscale_seconds"] = time.perf_counter() - upscale_started
        full_h, full_w = latent_h * 2, latent_w * 2

        full_conditionings = []
        temporal_image_anchors = ()
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
            if temporal_upsample_rounds:
                from .dfr import extract_dfr_temporal_image_anchors

                temporal_image_anchors = extract_dfr_temporal_image_anchors(
                    full_conditionings,
                    latent_h=full_h,
                    latent_w=full_w,
                )
                timings["temporal_image_anchors"] = [
                    {
                        "frame": anchor.pixel_frame,
                        "strength": anchor.strength,
                        "replace": anchor.replace,
                    }
                    for anchor in temporal_image_anchors
                ]
        stage2_generated_slot_rows = 0
        if dfr_enabled:
            from ltx_core_mlx.conditioning.types.reference_video_cond import (
                VideoConditionByReferenceLatent,
            )

            from .generated_keyframes import GeneratedKeyframeSlots

            if upscaled_slot_keyframes is None:
                raise RuntimeError("DFR stage one did not produce seeded keyframe slots.")
            full_conditionings.append(
                VideoConditionByReferenceLatent(
                    reference_latent=generated,
                    reference_positions=video_positions,
                    downscale_factor=2,
                    strength=1.0,
                )
            )
            full_slots = GeneratedKeyframeSlots(
                dfr_slot_frames,
                spatial_dims=(latent_f, full_h, full_w),
                frame_rate=frame_rate,
                initial_keyframes=upscaled_slot_keyframes,
            )
            # Generated slots must be the final appended rows because the
            # learned keyframe marker is applied to the projection tail.
            full_conditionings.append(full_slots)
            stage2_generated_slot_rows = full_slots.token_count
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
        if audio_reference_tokens is not None:
            from ltx_core_mlx.conditioning.types.latent_cond import LatentState

            audio_state2 = LatentState(
                latent=audio_reference_tokens,
                clean_latent=audio_reference_tokens,
                denoise_mask=mx.zeros((1, audio_tokens, 1), dtype=audio_reference_tokens.dtype),
                positions=audio_positions,
            )
        else:
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
            stage1_slot_tokens,
            upscaled_slot_keyframes,
        )
        aggressive_cleanup()
        timings["stage_boundary_cleanup_seconds"] = time.perf_counter() - cleanup_started
        if resolved_video_references:
            self._load_transformer(
                extra_loras=(
                    ((str(self.distilled_lora_path), 1.0),)
                    if pipeline_mode != "distilled"
                    else ()
                ),
                include_ic_loras=False,
            )
            model = X0Model(self.dit)
            timings["ic_lora_stage_scope"] = "stage_1_only"
        elif pipeline_mode != "distilled":
            self._load_transformer(
                extra_loras=((str(self.distilled_lora_path), 1.0),)
            )
            model = X0Model(self.dit)
        if dfr_detailing_lora is not None:
            if self.dfr_stage2_transformer_path is not None:
                self._load_transformer(
                    transformer_path=self.dfr_stage2_transformer_path,
                )
            else:
                self.load(extra_loras=(dfr_detailing_lora,))
            model = X0Model(self.dit)
        if dfr_enabled:
            set_generated_keyframe_marker(self.dit, stage2_generated_slot_rows)
        set_mpp_feed_forward_enabled(self.dit, self.feed_forward_stage_scope in {"all", "stage2"})
        stage2_started = time.perf_counter()
        try:
            stage2 = euler_ancestral_denoise_loop(
                model,
                video_state2,
                audio_state2,
                video_embeds,
                audio_embeds,
                sigmas=list(LTX25_STAGE2_SIGMAS),
                noise_seed=seed + 2,
                # The three-evaluation refinement stage is deterministic in the official
                # pipeline. Fresh ancestral noise cannot be removed reliably this late.
                eta=0.0,
                s_noise=1.0,
                check_interrupted=check_interrupted,
                step_callback=(
                    (
                        lambda completed, _total: step_callback(
                            stage1_steps + completed, stage1_steps + stage2_steps
                        )
                    )
                    if step_callback is not None
                    else None
                ),
                evaluation_timing_callback=(
                    lambda index, elapsed: timings["stage2_evaluations"].append(
                        {"evaluation": index, "seconds": elapsed}
                    )
                ),
            )
        finally:
            if dfr_enabled:
                set_generated_keyframe_marker(self.dit, 0)
        mx.eval(stage2.video_latent, stage2.audio_latent)
        timings["stage2_seconds"] = time.perf_counter() - stage2_started
        stage2_generated = stage2.video_latent[:, : latent_f * full_h * full_w, :]
        video_latent = self.video_patchifier.unpatchify(
            stage2_generated,
            (latent_f, full_h, full_w),
        )
        if temporal_upsample_rounds:
            from .dfr import select_dfr_generated_slot_tokens

            stage2_slot_tokens = select_dfr_generated_slot_tokens(
                stage2.video_latent, stage2_generated_slot_rows
            )
            carry_keyframes = self.video_patchifier.unpatchify(
                stage2_slot_tokens,
                (len(dfr_slot_frames), full_h, full_w),
            )
            video_latent, output_frames, output_fps = self._run_dfr_temporal_rounds(
                video_latent=video_latent,
                carry_frames=dfr_slot_frames,
                carry_keyframes=carry_keyframes,
                image_anchors=temporal_image_anchors,
                num_frames=num_frames,
                requested_num_frames=requested_num_frames,
                frame_rate=frame_rate,
                rounds=temporal_upsample_rounds,
                latent_h=full_h,
                latent_w=full_w,
                video_embeds=video_embeds,
                audio_embeds=audio_embeds,
                seed=seed,
                check_interrupted=check_interrupted,
                step_callback=step_callback,
                timings=timings,
            )
            self.last_num_frames = output_frames
            self.last_output_frame_rate = output_fps
        else:
            video_latent = video_latent[:, :, :requested_latent_f]
        stage2_audio_generated = stage2.audio_latent[:, :audio_tokens, :]
        output_audio_tokens = (
            dfr_audio_tokens if dfr_audio_tokens is not None else stage2_audio_generated
        )
        audio_latent = self.audio_patchifier.unpatchify(
            output_audio_tokens[:, :requested_audio_tokens, :]
        )
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
        timings["sampling_total_seconds"] += sum(
            float(round_report["seconds"])
            for round_report in timings.get("temporal_rounds", [])
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
        ic_lora_single_stage: bool = False,
        stage1_sampler: str = "euler_ancestral",
        cfg_pp_batched: bool = False,
        cfg_pp_schedule: str = "full",
        stage1_steps: int = 8,
        stage2_steps: int = 3,
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
        stage1_forwards = (
            stage1_steps
            if stage1_sampler != "euler_ancestral_cfg_pp"
            else stage1_steps + len(LTX25_CFG_PP_SCHEDULES[cfg_pp_schedule])
        )
        forwards_per_window = stage1_forwards + stage2_steps
        total_evaluations = window_count * forwards_per_window
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
                        lambda completed, _total, offset=index * forwards_per_window: step_callback(
                            offset + completed, total_evaluations
                        )
                    )
                    if step_callback is not None
                    else None
                ),
                prompt_context=prompt_context,
                encoded_prompt=encoded,
                ic_lora_single_stage=ic_lora_single_stage,
                stage1_sampler=stage1_sampler,
                cfg_pp_batched=cfg_pp_batched,
                cfg_pp_schedule=cfg_pp_schedule,
                stage1_steps=stage1_steps,
                stage2_steps=stage2_steps,
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

        # Sigma arrays are fixed by the pipeline. Validated sampler names and
        # stochastic controls must cross this adapter boundary because the
        # official Ingredients parity mode selects CFG++ here.
        for ignored in (
            "stage1_sigmas",
            "stage2_sigmas",
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
        if self.last_passthrough_audio is None:
            result = decode_and_save_video(
                self.video_decoder_block,
                self.audio_decoder_block,
                video_latent,
                audio_latent,
                output_path,
                frame_rate=self.last_output_frame_rate or frame_rate,
                low_memory=self.low_memory,
            )
        else:
            import tempfile

            from ltx_pipelines_mlx.utils._orchestration import save_waveform

            waveform, sample_rate = self.last_passthrough_audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
                audio_path = Path(temporary.name)
            try:
                save_waveform(waveform, str(audio_path), sample_rate=sample_rate)
                result = self.video_decoder_block.decode_and_stream(
                    video_latent,
                    output_path,
                    frame_rate=self.last_output_frame_rate or frame_rate,
                    audio_path=str(audio_path),
                )
            finally:
                audio_path.unlink(missing_ok=True)
            self.last_timings["audio_publication"] = "original_comfy_audio"
            self.last_passthrough_audio = None
        self.last_timings["decode_publish_seconds"] = time.perf_counter() - decode_started
        if self.low_memory:
            self.video_decoder_block.free()
            self.audio_decoder_block.free()
        return result


__all__ = ["LTX25DistilledPipeline"]
