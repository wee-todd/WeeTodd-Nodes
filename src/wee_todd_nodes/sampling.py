"""Process-local transformer-only sampling lifecycle for ComfyUI nodes."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any

from .conditioning import H3Conditioning, H3TextEncoderSpec
from .continuation import H3ContinuationContext, validate_continuation_for_sample
from .lora import H3LoRAStack
from .phase_memory import measured_phase
from .preflight import H3ComponentSetSpec, validate_task_partition
from .preview import H3PreviewConfig, H3PreviewSession
from .runtime import H3GenerationConfig


@dataclass(frozen=True)
class H3TransformerSpec:
    """Immutable inputs needed to construct the transformer-only H3 sampler."""

    checkpoint: str
    transformer: str
    text_encoder: str
    processor: str
    tokenizer: str
    video_vae: str
    audio_vae: str
    task: str
    text_encoder_config: str | None = None
    allow_fl2va_weights_for_ref2va: bool = False

    @classmethod
    def from_components(cls, components: H3ComponentSetSpec) -> H3TransformerSpec:
        paths = components.resolved_paths()
        encoder_spec = H3TextEncoderSpec.from_components(components, load_vision=False)
        return cls(
            checkpoint=components.checkpoint,
            transformer=str(paths["transformer"]),
            text_encoder=str(paths["text_encoder"]),
            processor=str(paths["processor"]),
            tokenizer=str(paths["tokenizer"]),
            video_vae=str(paths["video_vae"]),
            audio_vae=str(paths["audio_vae"]),
            task=components.task,
            text_encoder_config=encoder_spec.config_path,
            allow_fl2va_weights_for_ref2va=components.allow_fl2va_weights_for_ref2va,
        )

    def validate(self) -> None:
        root = Path(self.checkpoint).expanduser()
        transformer = Path(self.transformer).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"MiniMax H3 checkpoint directory not found: {root}")
        if not (root / "model_index.json").is_file():
            manifest = root / "model_index.json"
            raise FileNotFoundError(f"MiniMax H3 model manifest not found: {manifest}")
        if not transformer.exists():
            raise FileNotFoundError(f"MiniMax H3 transformer not found: {transformer}")
        from minimax_h3_mlx.dt_h3_checkpoint import is_dt_checkpoint
        from minimax_h3_mlx.dt_source import dt_source
        if self.task != "t2va" and (
            is_dt_checkpoint(transformer)
            or any(dt_source(getattr(self, component), component)
                   for component in ("text_encoder", "video_vae", "audio_vae"))
        ):
            raise ValueError("DT direct components currently support text-to-video only.")
        manifest = json.loads((root / "model_index.json").read_text())
        metadata = manifest.get("_minimax_h3", {})
        tasks = metadata.get("tasks", [])
        if self.task not in {"t2va", "fl2va", "ref2va"}:
            raise ValueError(f"The transformer sampler does not support task {self.task!r}.")
        partition = metadata.get("partition")
        if not isinstance(partition, str) or not partition:
            raise ValueError("MiniMax H3 model manifest has no partition name.")
        validate_task_partition(
            task=self.task,
            tasks=tasks,
            partition=partition,
            allow_fl2va_weights_for_ref2va=self.allow_fl2va_weights_for_ref2va,
        )


@dataclass(frozen=True)
class H3LearnedLatentUpscalerSpec:
    """Deferred MLX checkpoint selection for learned H3 video-latent upscaling."""

    checkpoint: str

    def validate(self) -> Path:
        source = Path(self.checkpoint).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"H3 learned latent upscaler not found: {source}")
        if source.suffix.lower() != ".safetensors":
            raise ValueError("H3 learned latent upscaler must be a SafeTensors checkpoint.")
        return source


@dataclass(frozen=True)
class H3Latents:
    """Adapter contract for synchronized undecoded MLX video and audio latents."""

    video: Any
    audio: Any
    num_frames: int
    width: int
    height: int
    fps: int
    sample_rate: int
    transformer_evaluations: int
    seconds_per_evaluation: float
    total_seconds: float
    transformer_spec: H3TransformerSpec
    generation_config: H3GenerationConfig
    easycache_skipped_steps: int = 0
    easycache_resolved_threshold: float | None = None
    easycache_reuse_strategy: str | None = None
    easycache_cache_bytes: int = 0
    blockcache_hits: int = 0
    blockcache_resolved_threshold: float | None = None
    blockcache_cache_bytes: int = 0
    blockcache_segment_hits: tuple[int, ...] = ()
    blockcache_segment_thresholds: tuple[float | None, ...] = ()
    blockcache_executed_blocks: int = 0
    blockcache_skipped_blocks: int = 0
    trajectory_forecasts: int = 0
    trajectory_bootstrap_forecasts: int = 0
    trajectory_fallbacks: int = 0
    trajectory_history_bytes: int = 0
    trajectory_offline_replay: bool = False
    trajectory_replay_steps: int = 0
    trajectory_replay_anchor_steps: int = 0
    trajectory_replay_smoothed_steps: int = 0
    trajectory_capture_seconds: float = 0.0
    trajectory_replay_seconds: float = 0.0
    trajectory_replay_fallback_reason: str | None = None
    trajectory_conditioned_row_policy: str | None = None
    trajectory_excluded_video_rows: int = 0
    trajectory_excluded_audio_rows: int = 0
    lora_report: tuple[dict[str, Any], ...] = ()
    projection_backend_report: dict[str, Any] | None = None
    projection_backend_runtime: dict[str, Any] | None = None
    paging_report: dict[str, Any] | None = None
    block_residency_report: dict[str, Any] | None = None
    text_encoder_paging_report: dict[str, Any] | None = None
    refinement_source_width: int | None = None
    refinement_source_height: int | None = None
    refinement_strength: float | None = None
    refinement_audio_preserved: bool = False
    refinement_upscaler_report: dict[str, int | float | str | bool] | None = None
    refinement_condition_rows_resized: bool = False
    preview_report: tuple[dict[str, Any], ...] = ()
    prepared_state_report: dict[str, int | float | str | None] | None = None
    sol_attention_report: dict[str, Any] | None = None
    fast_h3_approximation_report: dict[str, Any] | None = None
    vdn_report: dict[str, Any] | None = None
    phase_memory: dict[str, Any] | None = None
    conditioning_cache_report: dict[str, Any] | None = None
    fun_control_report: dict[str, Any] | None = None


SamplerFactory = Callable[[H3TransformerSpec], Any]


def _default_sampler_factory(spec: H3TransformerSpec):
    from minimax_h3_mlx.config import PipelineConfig
    from minimax_h3_mlx.load import load_dit
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    pipeline_config = PipelineConfig.from_model_index(Path(spec.checkpoint) / "model_index.json")
    transformer = Path(spec.transformer)
    from minimax_h3_mlx.dt_h3_checkpoint import is_dt_checkpoint, load_dt_h3_dit

    if is_dt_checkpoint(transformer):
        dit = load_dt_h3_dit(transformer)
    elif (transformer / "paged_manifest.json").is_file():
        from minimax_h3_mlx.paged_checkpoint import load_paged_dit

        dit = load_paged_dit(transformer)
    else:
        dit = load_dit(transformer)
    try:
        return MiniMaxH3Pipeline(dit, None, None, None, pipeline_config)
    except BaseException:
        pager = getattr(dit, "paged_blocks", None)
        if pager is not None:
            pager.close()
        raise


class H3TransformerCache:
    """Cache one transformer-only sampler with schedule-safe reuse."""

    def __init__(self, factory: SamplerFactory | None = None) -> None:
        self._lock = RLock()
        self._factory = factory or _default_sampler_factory
        self._spec: H3TransformerSpec | None = None
        self._schedule_key: tuple | None = None
        self._lora_key = None
        self._vdn_key = None
        self._lora_report: tuple[dict[str, Any], ...] = ()
        self._projection_backend_report: dict[str, Any] | None = None
        self._fun_control_key = None
        self._fun_control_model: Any = None
        self._sampler: Any = None

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._sampler is not None

    @measured_phase("transformer")
    def sample(
        self,
        spec: H3TransformerSpec,
        conditioning: H3Conditioning,
        config: H3GenerationConfig,
        *,
        unload_after: bool = True,
        block_residency: str = "checkpoint_default",
        step_callback=None,
        easycache=None,
        blockcache=None,
        trajectory_forecast=None,
        sol_attention=None,
        fastvideo=None,
        vdn=None,
        continuation: H3ContinuationContext | None = None,
        refinement_source: H3Latents | None = None,
        refinement_strength: float = 1.0,
        refinement_mode: str = "spatial",
        refinement_evaluations: int | None = None,
        refinement_resize_method: str = "bilinear",
        refinement_learned_upscaler: H3LearnedLatentUpscalerSpec | None = None,
        refinement_upscaler_callback=None,
        loras: H3LoRAStack | None = None,
        preview_config: H3PreviewConfig | None = None,
        preview_callback=None,
        prepare_stage: Callable[[], None] | None = None,
        fun_control_spec=None,
        fun_control_latent=None,
    ) -> H3Latents:
        spec.validate()
        config.validate()
        config.validate_paging(spec.transformer, block_residency)
        if refinement_mode not in {"spatial", "motion"}:
            raise ValueError("Unknown H3 refinement mode.")
        if refinement_evaluations is not None:
            if refinement_mode != "motion":
                raise ValueError("Explicit refinement evaluations are supported for motion only.")
            if type(refinement_evaluations) is not int or not 1 <= refinement_evaluations <= 64:
                raise ValueError("Refinement evaluations must be an integer from 1 to 64.")
        if refinement_mode == "motion":
            if refinement_source is None or spec.task != "t2va":
                raise ValueError("Motion refinement requires initialized native T2VA latents.")
            if any(
                value is not None
                for value in (
                    continuation,
                    easycache,
                    blockcache,
                    trajectory_forecast,
                    sol_attention,
                    fastvideo,
                    vdn,
                    fun_control_spec,
                    refinement_learned_upscaler,
                )
            ):
                raise ValueError("Motion refinement does not support conditioning or accelerators.")
            if loras is not None:
                loras.validate_for_motion(config.steps)
        if (fun_control_spec is None) != (fun_control_latent is None):
            raise ValueError("H3 Fun ControlNet requires both a checkpoint spec and control latent")
        fun_control_key = None
        if fun_control_spec is not None:
            control_path = fun_control_spec.validate()
            fun_control_key = (str(control_path.resolve()), float(fun_control_spec.strength))
            if spec.task != "t2va" or conditioning.task != "t2va":
                raise ValueError("H3 Fun ControlNet currently requires the native T2VA base path")
        if block_residency not in {"checkpoint_default", "resident"}:
            raise ValueError("H3 block_residency must be checkpoint_default or resident.")
        if block_residency == "resident" and config.memory_mode == "low_memory_bf16":
            raise ValueError("Resident H3 blocks require normal memory mode.")
        if config.projection_backend == "mpp_resident_expanded_experimental" and (
            block_residency != "resident" or config.memory_mode != "normal"
        ):
            raise ValueError(
                "Expanded Q8 projections require explicit resident blocks and normal memory mode."
            )
        continuation_text_only_fl2va = bool(
            spec.task == "fl2va"
            and continuation is not None
            and conditioning.condition_video_rows is None
            and not conditioning.keyframe_anchors
        )
        expected_vision = (spec.task == "fl2va" and not continuation_text_only_fl2va) or (
            spec.task == "ref2va" and conditioning.condition_video_rows is not None
        )
        if conditioning.task != spec.task:
            raise ValueError(
                f"Conditioning task {conditioning.task!r} does not match "
                f"component task {spec.task!r}."
            )
        if conditioning.load_vision != expected_vision:
            requirement = "vision" if expected_vision else "text-only"
            raise ValueError(f"The {spec.task} sampler requires {requirement} conditioning.")
        if spec.task == "t2va" and (
            conditioning.condition_video_rows is not None or conditioning.keyframe_anchors
        ):
            raise ValueError("T2VA conditioning cannot contain first/last-frame rows.")
        if (
            spec.task == "fl2va"
            and not continuation_text_only_fl2va
            and (conditioning.condition_video_rows is None or not conditioning.keyframe_anchors)
        ):
            raise ValueError("FL2VA conditioning requires encoded first/last-frame rows.")
        if spec.task == "ref2va" and (
            not conditioning.references
            or (
                conditioning.condition_video_rows is None
                and conditioning.condition_audio_rows is None
            )
        ):
            raise ValueError("Ref2VA conditioning requires encoded visual or audio reference rows.")
        if spec.task == "ref2va" and conditioning.keyframe_anchors:
            raise ValueError("Ref2VA conditioning cannot contain first/last-frame anchors.")
        if continuation is not None:
            validate_continuation_for_sample(continuation, spec, config)
        if refinement_source is not None:
            if continuation is not None:
                raise ValueError("H3 Hi Res Fix cannot be combined with motion continuation.")
            if refinement_source.transformer_spec != spec:
                raise ValueError("H3 Hi Res Fix source latents use a different transformer.")
            from minimax_h3_mlx.packing import align_num_frames

            expected_frames = align_num_frames(round(config.duration_seconds * 24))
            if refinement_source.num_frames != expected_frames:
                raise ValueError(
                    "H3 Hi Res Fix duration does not match the source latent frame count."
                )
            if refinement_source.fps != 24 or refinement_source.sample_rate != 32000:
                raise ValueError("H3 Hi Res Fix requires native 24 fps and 32 kHz H3 latents.")
            if refinement_mode == "motion" and (
                config.width != refinement_source.width or config.height != refinement_source.height
            ):
                raise ValueError("Motion refinement must preserve source dimensions.")
            if refinement_mode == "spatial" and (
                config.width <= refinement_source.width or config.height <= refinement_source.height
            ):
                raise ValueError(
                    "H3 Hi Res Fix target dimensions must exceed the source dimensions."
                )
            if not 0.0 < refinement_strength <= 1.0:
                raise ValueError(
                    "H3 Hi Res Fix refinement strength must be greater than 0 and at most 1."
                )
        expected_encoder = H3TextEncoderSpec(
            text_encoder=spec.text_encoder,
            processor=spec.processor,
            tokenizer=spec.tokenizer,
            load_vision=expected_vision,
            config_path=spec.text_encoder_config,
        )
        if conditioning.encoder_spec != expected_encoder:
            raise ValueError(
                "Conditioning was produced by a different Qwen3-VL component specification."
            )
        refinement_condition_rows_resized = False
        if refinement_source is not None and spec.task == "fl2va":
            from minimax_h3_mlx.hires_fix import resize_fl2va_condition_rows

            conditioning = replace(
                conditioning,
                condition_video_rows=resize_fl2va_condition_rows(
                    conditioning.condition_video_rows,
                    conditioning.keyframe_anchors,
                    source_height=refinement_source.height // 16,
                    source_width=refinement_source.width // 16,
                    target_height=config.height // 16,
                    target_width=config.width // 16,
                    # Condition rows are geometry constraints, not the visual refinement source.
                    # Stable bilinear resizing matches the learned checkpoint's spatial stage.
                    method="bilinear",
                ),
            )
            refinement_condition_rows_resized = True
        condition_schedule_key = (
            refinement_mode,
            continuation is not None,
            refinement_source is not None,
            round(float(refinement_strength), 6) if refinement_source is not None else None,
            refinement_evaluations,
            conditioning.condition_video_rows is not None,
            conditioning.condition_audio_rows is not None,
            float(conditioning.visual_condition_strength),
            float(conditioning.audio_condition_strength),
        )
        schedule_key = (
            config.steps,
            config.drop_adaln,
            config.memory_mode,
            config.projection_backend,
            config.sampling_method,
            config.inference_optimization,
            block_residency,
            condition_schedule_key,
            sol_attention,
            fastvideo,
            vdn.cache_key if vdn is not None else None,
        )
        if vdn is not None:
            vdn.validate()
            vdn.validate_sampling(config, loras)
            if spec.task != "t2va":
                raise ValueError("VDN-H3 MLX currently supports T2VA only.")
            if config.steps != vdn.schedule_points:
                raise ValueError(
                    "VDN-H3 sampling schedule does not match the selected checkpoint; "
                    "use the config output from WeeTodd H3 VDN Checkpoint."
                )
            if any(
                value is not None
                for value in (
                    easycache,
                    blockcache,
                    trajectory_forecast,
                    sol_attention,
                    fastvideo,
                    continuation,
                    refinement_source,
                )
            ):
                raise ValueError(
                    "VDN-H3 must run in isolation from cache, forecast, sparse-attention, "
                    "FastVideo, continuation, and Hi Res Fix controls."
                )
        loras = loras or H3LoRAStack()
        loras.validate_for_steps(config.steps)
        staged_lora = any(spec.start_after_evaluations > 0 for spec in loras.adapters)
        if staged_lora and any(
            value is not None for value in (easycache, blockcache, trajectory_forecast)
        ):
            raise ValueError(
                "Staged LoRA activation requires dense transformer evaluations. Disconnect "
                "EasyCache, BlockCache, and Trajectory Forecast before sampling."
            )
        if (
            loras.has_turbo
            and easycache is not None
            and not getattr(easycache, "allow_turbo_experimental", False)
        ):
            raise ValueError(
                "Turbo LoRA with EasyCache requires the explicit experimental opt-in on the "
                "EasyCache node. This combination may change motion, detail, or audio."
            )
        if (
            loras.has_turbo
            and blockcache is not None
            and not getattr(blockcache, "allow_turbo_experimental", False)
        ):
            raise ValueError(
                "Turbo LoRA with BlockCache requires the explicit experimental opt-in on the "
                "BlockCache node. This combination may change motion, detail, or audio."
            )
        accelerators = sum(
            value is not None for value in (easycache, blockcache, trajectory_forecast)
        )
        if sol_attention is not None and accelerators:
            raise ValueError(
                "H3 Sol Attention must be validated without EasyCache, BlockCache, or "
                "Trajectory Forecast. Disconnect the other accelerator."
            )
        if fastvideo is not None and getattr(fastvideo, "enabled", False) and accelerators:
            raise ValueError(
                "FastH3 FastVideo approximations require an isolated baseline run. Disconnect "
                "EasyCache, BlockCache, and Trajectory Forecast."
            )
        if accelerators > 1:
            raise ValueError(
                "EasyCache, BlockCache, and Trajectory Forecast are mutually exclusive."
            )
        if fun_control_spec is not None and any(
            value is not None
            for value in (
                easycache,
                blockcache,
                trajectory_forecast,
                sol_attention,
                fastvideo,
                vdn,
                continuation,
                refinement_source,
            )
        ):
            raise ValueError(
                "H3 Fun ControlNet must run without cache, forecast, VDN, sparse-attention, "
                "or FastH3 approximation controls until those combinations are qualified"
            )
        if fun_control_spec is not None and loras.adapters:
            raise ValueError("H3 Fun ControlNet with LoRA stacks is not qualified yet")
        if continuation is not None and (easycache is not None or blockcache is not None):
            raise ValueError(
                "H3 continuation supports dense sampling or Trajectory Forecast. Disconnect "
                "EasyCache and BlockCache."
            )
        if spec.task == "fl2va" and accelerators:
            raise ValueError(
                "The first FL2VA baseline does not support cache or trajectory acceleration."
            )
        if spec.task == "ref2va" and (easycache is not None or blockcache is not None):
            raise ValueError(
                "Ref2VA supports Trajectory Forecast only; EasyCache and BlockCache remain "
                "disabled until their conditioned-row behavior is validated."
            )
        if prepare_stage is not None:
            prepare_stage()
        initial_video_latents = None
        initial_audio_latents = None
        refinement_upscaler_report = None
        if refinement_source is not None:
            import mlx.core as mx

            if refinement_mode == "motion":
                initial_video_latents = refinement_source.video
            elif refinement_learned_upscaler is not None:
                # A warm transformer from the first pass must not overlap the temporary learned
                # upscaler. Reloading it after this stage is cheaper than increasing peak memory.
                self.unload()
                from minimax_h3_mlx.learned_latent_upscaler import (
                    upscale_h3_video_latents_learned,
                )

                initial_video_latents, refinement_upscaler_report = (
                    upscale_h3_video_latents_learned(
                        refinement_source.video,
                        config.height // 16,
                        config.width // 16,
                        refinement_learned_upscaler.validate(),
                        progress_callback=refinement_upscaler_callback,
                    )
                )
            else:
                from minimax_h3_mlx.hires_fix import resize_video_latents

                initial_video_latents = resize_video_latents(
                    refinement_source.video,
                    config.height // 16,
                    config.width // 16,
                    method=refinement_resize_method,
                )
            initial_audio_latents = refinement_source.audio
            mx.eval(initial_video_latents, initial_audio_latents)
        lora_key = loras.cache_key
        vdn_key = vdn.cache_key if vdn is not None else None
        with self._lock:
            if (
                self._sampler is None
                or self._spec != spec
                or self._schedule_key != schedule_key
                or self._lora_key != lora_key
                or self._vdn_key != vdn_key
                or self._fun_control_key != fun_control_key
            ):
                self._release_locked()
                try:
                    self._sampler = self._factory(spec)
                    if block_residency == "resident":
                        from minimax_h3_mlx.paged_checkpoint import materialize_paged_blocks

                        self._sampler.dit.block_residency_report = materialize_paged_blocks(
                            self._sampler.dit
                        )
                    self._spec = spec
                    self._schedule_key = schedule_key
                    self._lora_key = lora_key
                    self._vdn_key = vdn_key
                    self._fun_control_key = fun_control_key
                    from minimax_h3_mlx.projection import configure_projection_backend

                    backend_report = configure_projection_backend(
                        self._sampler.dit, config.projection_backend
                    )
                    self._projection_backend_report = backend_report.to_dict()
                    if vdn is not None:
                        from minimax_h3_mlx.vdn import load_vdn_runtime

                        vdn_runtime = load_vdn_runtime(vdn.engine_request())
                        vdn_runtime.inference.mpp = backend_report.resolved == "mpp_experimental"
                        self._sampler.dit.set_vdn_runtime(vdn_runtime)
                    if fun_control_spec is not None:
                        from minimax_h3_mlx.controlnet import load_fun_controlnet

                        self._fun_control_model = load_fun_controlnet(fun_control_spec.validate())
                    if loras.adapters:
                        from minimax_h3_mlx.lora import apply_lora_stack

                        reports = apply_lora_stack(self._sampler.dit, loras.engine_requests())
                        sanitized = []
                        for report in reports:
                            item = asdict(report)
                            item["path"] = Path(item["path"]).name
                            sanitized.append(item)
                        self._lora_report = tuple(sanitized)
                    if config.inference_optimization != "off":
                        from minimax_h3_mlx.inference_optimizations import (
                            configure_inference_optimizations,
                        )

                        self._projection_backend_report["inference_optimization"] = (
                            configure_inference_optimizations(
                                self._sampler.dit, config.inference_optimization
                            )
                        )
                except BaseException:
                    self._release_locked()
                    raise
            try:
                self._sampler.dit.set_attention_query_chunk_size(config.attention_query_chunk_size)
                sol_setter = getattr(self._sampler.dit, "set_sol_attention_config", None)
                if sol_attention is not None and sol_setter is None:
                    raise RuntimeError("The active H3 engine does not support MLX Sol Attention.")
                if sol_attention is not None and config.attention_query_chunk_size is not None:
                    from minimax_h3_mlx.vsa_h3 import FastH3VSAConfig

                    if isinstance(sol_attention, FastH3VSAConfig):
                        # VSA owns its bounded gathered-query batches. The ordinary dense query
                        # chunk is neither used nor compatible with that route.
                        self._sampler.dit.set_attention_query_chunk_size(None)
                    else:
                        raise ValueError(
                            "MLX Sol Attention requires unchunked attention queries. Use normal "
                            "memory mode or disable attention query chunking."
                        )
                if sol_setter is not None:
                    sol_setter(sol_attention)
                fastvideo_setter = getattr(
                    self._sampler.dit, "set_fast_h3_approximation_config", None
                )
                if fastvideo is not None and fastvideo_setter is None:
                    raise RuntimeError(
                        "The active H3 engine does not support FastVideo approximation controls."
                    )
                if fastvideo_setter is not None:
                    fastvideo_setter(fastvideo)
                head_group_size = config.attention_head_group_size
                row_group_size = config.ffn_row_group_size
                head_setter = getattr(self._sampler.dit, "set_attention_head_chunk_size", None)
                row_setter = getattr(self._sampler.dit, "set_ffn_row_chunk_size", None)
                if head_group_size is not None and head_setter is None:
                    raise RuntimeError(
                        "The active H3 engine does not support attention-head chunking."
                    )
                if row_group_size is not None and row_setter is None:
                    raise RuntimeError("The active H3 engine does not support FFN row chunking.")
                if head_setter is not None:
                    head_setter(head_group_size)
                if row_setter is not None:
                    row_setter(row_group_size)
                if self._fun_control_model is not None:
                    self._fun_control_model.set_chunk_sizes(
                        query_rows=config.attention_query_chunk_size,
                        attention_heads=head_group_size,
                        ffn_rows=row_group_size,
                    )
                preview_session = (
                    H3PreviewSession(preview_config) if preview_config is not None else None
                )
                preview_reports: list[dict[str, Any]] = []

                def on_latent_preview(completed, total, video_latents):
                    if preview_session is None:
                        return
                    update = preview_session.update(video_latents, completed, total)
                    if update is None:
                        return
                    report = {
                        "completed": int(completed),
                        "total": int(total),
                        "backend": preview_session.backend,
                        "fallback_reason": preview_session.fallback_reason,
                        "statistics": (
                            asdict(update.statistics) if update.statistics is not None else None
                        ),
                        "rejected": update.reject_reason is not None,
                    }
                    preview_reports.append(report)
                    if preview_callback is not None:
                        preview_callback(update, completed, total)
                    if update.reject_reason is not None:
                        raise RuntimeError(update.reject_reason)

                try:
                    paging_executor = getattr(self._sampler.dit, "paged_blocks", None)
                    if config.paging_cache_gb > 0 and paging_executor is None:
                        raise ValueError("paging_cache_gb requires a paged H3 transformer.")
                    if paging_executor is not None:
                        paging_executor.store.configure_cache(int(config.paging_cache_gb * 1e9))
                    vdn_runtime = getattr(self._sampler.dit, "vdn_runtime", None)
                    if vdn_runtime is not None:
                        vdn_runtime.begin_run()
                    fun_control = None
                    if fun_control_spec is not None:
                        from minimax_h3_mlx.controlnet import H3FunControlCondition

                        fun_control = H3FunControlCondition(
                            self._fun_control_model,
                            fun_control_latent,
                            float(fun_control_spec.strength),
                        )
                    result = self._sampler.sample_latents(
                        conditioning.embeddings,
                        conditioning.token_tags,
                        duration_seconds=config.duration_seconds,
                        num_inference_steps=config.steps,
                        seed=config.seed,
                        height=config.height,
                        width=config.width,
                        drop_adaln=config.drop_adaln,
                        step_callback=step_callback,
                        latent_preview_callback=(
                            on_latent_preview if preview_session is not None else None
                        ),
                        easycache_config=easycache,
                        blockcache_config=blockcache,
                        trajectory_forecast_config=trajectory_forecast,
                        continuation_video_latents=(
                            continuation.video if continuation is not None else None
                        ),
                        continuation_audio_latents=(
                            continuation.audio if continuation is not None else None
                        ),
                        continuation_frames=(
                            continuation.context_frames if continuation is not None else 0
                        ),
                        condition_video_rows=conditioning.condition_video_rows,
                        condition_audio_rows=conditioning.condition_audio_rows,
                        keyframe_anchors=conditioning.keyframe_anchors,
                        references=conditioning.references,
                        sampling_method=config.sampling_method,
                        visual_condition_strength=conditioning.visual_condition_strength,
                        audio_condition_strength=conditioning.audio_condition_strength,
                        initial_video_latents=initial_video_latents,
                        initial_audio_latents=initial_audio_latents,
                        refinement_strength=refinement_strength,
                        refinement_start_sigma=(
                            refinement_strength if refinement_mode == "motion" else None
                        ),
                        refinement_evaluations=refinement_evaluations,
                        preserve_initial_audio=refinement_source is not None,
                        fun_control=fun_control,
                    )
                finally:
                    if paging_executor is not None:
                        paging_executor.store.clear_retained_cache()
                    if preview_session is not None:
                        preview_session.release()
                from minimax_h3_mlx.projection import mpp_runtime_status

                paged = getattr(self._sampler.dit, "paged_blocks", None)
                resolved_sol = getattr(self._sampler.dit, "last_sol_attention_config", None)
                sol_reporter = getattr(self._sampler.dit, "sol_attention_report", None)

                latents = H3Latents(
                    conditioning_cache_report=getattr(conditioning, "cache_report", None),
                    fun_control_report=(
                        {
                            "checkpoint": Path(fun_control_spec.checkpoint).name,
                            "strength": float(fun_control_spec.strength),
                            "injection_layers": list(self._fun_control_model.injection_layers),
                        }
                        if fun_control_spec is not None
                        else None
                    ),
                    video=result.video_latents,
                    audio=result.audio_latents,
                    num_frames=result.num_frames,
                    width=result.width,
                    height=result.height,
                    fps=result.fps,
                    sample_rate=result.sample_rate,
                    transformer_evaluations=result.transformer_evaluations,
                    easycache_skipped_steps=getattr(result, "easycache_skipped_steps", 0),
                    easycache_resolved_threshold=getattr(
                        result, "easycache_resolved_threshold", None
                    ),
                    easycache_reuse_strategy=getattr(result, "easycache_reuse_strategy", None),
                    easycache_cache_bytes=getattr(result, "easycache_cache_bytes", 0),
                    blockcache_hits=getattr(result, "blockcache_hits", 0),
                    blockcache_resolved_threshold=getattr(
                        result, "blockcache_resolved_threshold", None
                    ),
                    blockcache_cache_bytes=getattr(result, "blockcache_cache_bytes", 0),
                    blockcache_segment_hits=getattr(result, "blockcache_segment_hits", ()),
                    blockcache_segment_thresholds=getattr(
                        result, "blockcache_segment_thresholds", ()
                    ),
                    blockcache_executed_blocks=getattr(result, "blockcache_executed_blocks", 0),
                    blockcache_skipped_blocks=getattr(result, "blockcache_skipped_blocks", 0),
                    trajectory_forecasts=getattr(result, "trajectory_forecasts", 0),
                    trajectory_bootstrap_forecasts=getattr(
                        result, "trajectory_bootstrap_forecasts", 0
                    ),
                    trajectory_fallbacks=getattr(result, "trajectory_fallbacks", 0),
                    trajectory_history_bytes=getattr(result, "trajectory_history_bytes", 0),
                    trajectory_offline_replay=getattr(result, "trajectory_offline_replay", False),
                    trajectory_replay_steps=getattr(result, "trajectory_replay_steps", 0),
                    trajectory_replay_anchor_steps=getattr(
                        result, "trajectory_replay_anchor_steps", 0
                    ),
                    trajectory_replay_smoothed_steps=getattr(
                        result, "trajectory_replay_smoothed_steps", 0
                    ),
                    trajectory_capture_seconds=getattr(result, "trajectory_capture_seconds", 0.0),
                    trajectory_replay_seconds=getattr(result, "trajectory_replay_seconds", 0.0),
                    trajectory_replay_fallback_reason=getattr(
                        result, "trajectory_replay_fallback_reason", None
                    ),
                    trajectory_conditioned_row_policy=getattr(
                        result, "trajectory_conditioned_row_policy", None
                    ),
                    trajectory_excluded_video_rows=getattr(
                        result, "trajectory_excluded_video_rows", 0
                    ),
                    trajectory_excluded_audio_rows=getattr(
                        result, "trajectory_excluded_audio_rows", 0
                    ),
                    lora_report=self._lora_report,
                    projection_backend_report=self._projection_backend_report,
                    projection_backend_runtime=mpp_runtime_status(),
                    paging_report=paged.report() if paged is not None else None,
                    block_residency_report={
                        **getattr(self._sampler.dit, "block_residency_report", {}),
                        "requested": block_residency,
                        "mode": "paged" if paged is not None else "resident",
                        "keep_warm": not unload_after and config.memory_mode == "normal",
                    },
                    text_encoder_paging_report=conditioning.paging_report,
                    refinement_source_width=(
                        refinement_source.width if refinement_source is not None else None
                    ),
                    refinement_source_height=(
                        refinement_source.height if refinement_source is not None else None
                    ),
                    refinement_strength=getattr(result, "refinement_strength", None),
                    refinement_audio_preserved=getattr(result, "refinement_audio_preserved", False),
                    refinement_upscaler_report=refinement_upscaler_report,
                    refinement_condition_rows_resized=refinement_condition_rows_resized,
                    preview_report=tuple(preview_reports),
                    prepared_state_report={
                        "cache_hits": getattr(result, "prepared_state_cache_hits", 0),
                        "cache_builds": getattr(result, "prepared_state_cache_builds", 0),
                        "cache_bytes": getattr(result, "prepared_state_cache_bytes", 0),
                        "build_seconds": getattr(result, "prepared_state_build_seconds", 0.0),
                        "key": getattr(result, "prepared_state_key", None),
                    },
                    sol_attention_report=(
                        sol_reporter()
                        if sol_reporter is not None
                        else asdict(resolved_sol)
                        if resolved_sol is not None
                        else None
                    ),
                    fast_h3_approximation_report=getattr(
                        self._sampler.dit, "fast_h3_approximation_report", None
                    ),
                    vdn_report=(
                        self._sampler.dit.vdn_report()
                        if getattr(self._sampler.dit, "vdn_report", None) is not None
                        else None
                    ),
                    seconds_per_evaluation=result.seconds_per_evaluation,
                    total_seconds=result.total_seconds,
                    transformer_spec=spec,
                    generation_config=config,
                )
                try:
                    import mlx.core as mx

                    if type(latents.video).__module__.startswith("mlx."):
                        mx.eval(latents.video, latents.audio)
                except ImportError:
                    pass
            except BaseException:
                self._release_locked()
                raise
            if unload_after or config.memory_mode == "low_memory_bf16":
                self._release_locked()
            return latents

    def unload(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        dit = getattr(self._sampler, "dit", None)
        pager = getattr(dit, "paged_blocks", None)
        if pager is not None and hasattr(pager, "close"):
            pager.close()
        self._sampler = None
        self._spec = None
        self._schedule_key = None
        self._lora_key = None
        self._vdn_key = None
        if self._fun_control_model is not None:
            self._fun_control_model.release()
        self._fun_control_model = None
        self._fun_control_key = None
        self._lora_report = ()
        self._projection_backend_report = None
        try:
            from minimax_h3_mlx.projection import reset_mpp_runtime_status

            reset_mpp_runtime_status()
        except ImportError:
            pass
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass


TRANSFORMER_RUNTIME = H3TransformerCache()
