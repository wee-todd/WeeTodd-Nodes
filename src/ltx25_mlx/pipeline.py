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
        temporal_upsampler_path: str = "",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        feed_forward_backend: str = "reference_fp32",
        feed_forward_stage_scope: str = "all",
        diffvae_optimization: str = "combined",
        diffvae_query_chunk_size: int = 512,
        diffvae_context_width_chunks: int = 4,
        diffvae_stage4_tile_width: int = 0,
        loras: tuple[tuple[str, float], ...] = (),
        verbose: bool = True,
    ) -> None:
        self.transformer_path = Path(transformer_path)
        self.duration_head_path = Path(duration_head_path) if duration_head_path else None
        self.spatial_upscaler_path = Path(spatial_upscaler_path)
        self.temporal_upsampler_path = (
            Path(temporal_upsampler_path) if temporal_upsampler_path else None
        )
        self.low_memory = low_memory
        self.low_ram_streaming = low_ram_streaming
        self.feed_forward_backend = feed_forward_backend
        self.loras = tuple(loras)
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
        self.dit = None
        self._loaded_loras = None
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

        from ltx_core_mlx.components.patchifiers import (
            AudioPatchifier,
            VideoLatentPatchifier,
        )

        self.video_patchifier = VideoLatentPatchifier()
        self.audio_patchifier = AudioPatchifier()

    def load(self, *, extra_loras: tuple[tuple[str, float], ...] = ()) -> None:
        self._load_transformer(extra_loras=extra_loras)
        if self.upsampler is None:
            self.upsampler = load_ltx25_spatial_upsampler(self.spatial_upscaler_path)

    def _load_transformer(
        self,
        *,
        extra_loras: tuple[tuple[str, float], ...] = (),
    ) -> None:
        desired_loras = (*self.loras, *extra_loras)
        if self.dit is not None and self._loaded_loras != desired_loras:
            self._release_transformer()
        if self.dit is None:
            self.dit = load_ltx25_transformer(
                self.transformer_path,
                low_ram_streaming=self.low_ram_streaming,
                feed_forward_backend=self.feed_forward_backend,
                loras=desired_loras,
            )
            self._loaded_loras = desired_loras
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

    def _release_sampling(self) -> None:
        self._release_transformer()
        self.upsampler = None
        self.temporal_upsampler = None
        self.image_conditioner.free()
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
        generated_keyframes: int = 0,
        dfr_enabled: bool = False,
        dfr_detailing_lora: tuple[str, float] | None = None,
        temporal_upsample_rounds: int = 0,
        auto_duration: bool = False,
        auto_duration_min_seconds: float = 1.0,
        auto_duration_max_seconds: float = 20.0,
        **_unused,
    ):
        """Generate synchronized latents using official 8+3 stage semantics."""
        if stage1_steps != 8 or stage2_steps != 3:
            raise ValueError("LTX 2.5 distilled generation requires exactly 8+3 evaluations.")
        if temporal_upsample_rounds not in {0, 1, 2}:
            raise ValueError("LTX 2.5 DFR temporal rounds must be zero, one, or two.")
        if temporal_upsample_rounds and not dfr_enabled:
            raise ValueError("LTX 2.5 temporal rounds require DFR detailing.")
        if temporal_upsample_rounds and (continuation is not None or return_continuation):
            raise ValueError(
                "LTX 2.5 temporal DFR is not yet available for chained timelines."
            )
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
        sampling_load_started = time.perf_counter()
        self.load()
        timings["sampling_component_load_seconds"] = time.perf_counter() - sampling_load_started
        assert self.dit is not None and self.upsampler is not None
        requested_num_frames = num_frames
        dfr_slot_frames: tuple[int, ...] = ()
        if dfr_enabled:
            from .dfr import resolve_dfr_canvas

            num_frames, _segment, dfr_slot_frames = resolve_dfr_canvas(num_frames)
        height, width = snap_output_dimensions(height, width, two_stage=True)
        half_h, half_w = height // 2, width // 2
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
        try:
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
        finally:
            if stage1_generated_slot_rows:
                set_generated_keyframe_marker(self.dit, 0)
        mx.eval(stage1.video_latent, stage1.audio_latent)
        timings["stage1_seconds"] = time.perf_counter() - stage1_started

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
        if dfr_detailing_lora is not None:
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
            frame_rate=self.last_output_frame_rate or frame_rate,
            low_memory=self.low_memory,
        )
        self.last_timings["decode_publish_seconds"] = time.perf_counter() - decode_started
        if self.low_memory:
            self.video_decoder_block.free()
            self.audio_decoder_block.free()
        return result


__all__ = ["LTX25DistilledPipeline"]
