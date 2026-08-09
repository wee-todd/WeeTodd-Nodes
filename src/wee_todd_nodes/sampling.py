"""Process-local transformer-only sampling lifecycle for ComfyUI nodes."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from .conditioning import H3Conditioning, H3TextEncoderSpec
from .lora import H3LoRAStack
from .preflight import H3ComponentSetSpec, validate_task_partition
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
    lora_report: tuple[dict[str, Any], ...] = ()
    projection_backend_report: dict[str, Any] | None = None
    projection_backend_runtime: dict[str, Any] | None = None
    paging_report: dict[str, Any] | None = None
    text_encoder_paging_report: dict[str, Any] | None = None


SamplerFactory = Callable[[H3TransformerSpec], Any]


def _default_sampler_factory(spec: H3TransformerSpec):
    from minimax_h3_mlx.config import PipelineConfig
    from minimax_h3_mlx.load import load_dit
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    transformer = Path(spec.transformer)
    if (transformer / "paged_manifest.json").is_file():
        from minimax_h3_mlx.paged_checkpoint import load_paged_dit

        dit = load_paged_dit(transformer)
    else:
        dit = load_dit(transformer)
    pipeline_config = PipelineConfig.from_model_index(Path(spec.checkpoint) / "model_index.json")
    return MiniMaxH3Pipeline(dit, None, None, None, pipeline_config)


class H3TransformerCache:
    """Cache one transformer-only sampler with schedule-safe reuse."""

    def __init__(self, factory: SamplerFactory | None = None) -> None:
        self._lock = RLock()
        self._factory = factory or _default_sampler_factory
        self._spec: H3TransformerSpec | None = None
        self._schedule_key: tuple[int, bool, str, str, str] | None = None
        self._lora_key = None
        self._lora_report: tuple[dict[str, Any], ...] = ()
        self._projection_backend_report: dict[str, Any] | None = None
        self._sampler: Any = None

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._sampler is not None

    def sample(
        self,
        spec: H3TransformerSpec,
        conditioning: H3Conditioning,
        config: H3GenerationConfig,
        *,
        unload_after: bool = True,
        step_callback=None,
        easycache=None,
        blockcache=None,
        trajectory_forecast=None,
        loras: H3LoRAStack | None = None,
        prepare_stage: Callable[[], None] | None = None,
    ) -> H3Latents:
        spec.validate()
        config.validate()
        expected_vision = spec.task in {"fl2va", "ref2va"}
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
        if spec.task == "fl2va" and (
            conditioning.condition_video_rows is None or not conditioning.keyframe_anchors
        ):
            raise ValueError("FL2VA conditioning requires encoded first/last-frame rows.")
        if spec.task == "ref2va" and (
            conditioning.condition_video_rows is None or not conditioning.references
        ):
            raise ValueError("Ref2VA conditioning requires encoded visual reference rows.")
        if spec.task == "ref2va" and conditioning.keyframe_anchors:
            raise ValueError("Ref2VA conditioning cannot contain first/last-frame anchors.")
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
        schedule_key = (
            config.steps,
            config.drop_adaln,
            config.memory_mode,
            config.projection_backend,
            config.sampling_method,
        )
        loras = loras or H3LoRAStack()
        loras.validate_for_steps(config.steps)
        if loras.has_turbo and easycache is not None:
            raise ValueError(
                "Turbo LoRA sampling does not support EasyCache. Disconnect the cache node "
                "before sampling."
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
        if accelerators > 1:
            raise ValueError(
                "EasyCache, BlockCache, and Trajectory Forecast are mutually exclusive."
            )
        if spec.task != "t2va" and accelerators:
            raise ValueError(
                "The first FL2VA baseline does not support cache or trajectory acceleration."
            )
        if prepare_stage is not None:
            prepare_stage()
        lora_key = loras.cache_key
        with self._lock:
            if (
                self._sampler is None
                or self._spec != spec
                or self._schedule_key != schedule_key
                or self._lora_key != lora_key
            ):
                self._release_locked()
                try:
                    self._sampler = self._factory(spec)
                    self._spec = spec
                    self._schedule_key = schedule_key
                    self._lora_key = lora_key
                    from minimax_h3_mlx.projection import configure_projection_backend

                    backend_report = configure_projection_backend(
                        self._sampler.dit, config.projection_backend
                    )
                    self._projection_backend_report = backend_report.to_dict()
                    if loras.adapters:
                        from minimax_h3_mlx.lora import apply_lora_stack

                        reports = apply_lora_stack(self._sampler.dit, loras.engine_requests())
                        sanitized = []
                        for report in reports:
                            item = asdict(report)
                            item["path"] = Path(item["path"]).name
                            sanitized.append(item)
                        self._lora_report = tuple(sanitized)
                except BaseException:
                    self._release_locked()
                    raise
            try:
                self._sampler.dit.set_attention_query_chunk_size(config.attention_query_chunk_size)
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
                    easycache_config=easycache,
                    blockcache_config=blockcache,
                    trajectory_forecast_config=trajectory_forecast,
                    condition_video_rows=conditioning.condition_video_rows,
                    condition_audio_rows=conditioning.condition_audio_rows,
                    keyframe_anchors=conditioning.keyframe_anchors,
                    references=conditioning.references,
                    sampling_method=config.sampling_method,
                    visual_condition_strength=conditioning.visual_condition_strength,
                    audio_condition_strength=conditioning.audio_condition_strength,
                )
                from minimax_h3_mlx.projection import mpp_runtime_status

                paged = getattr(self._sampler.dit, "paged_blocks", None)

                latents = H3Latents(
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
                    lora_report=self._lora_report,
                    projection_backend_report=self._projection_backend_report,
                    projection_backend_runtime=mpp_runtime_status(),
                    paging_report=paged.report() if paged is not None else None,
                    text_encoder_paging_report=conditioning.paging_report,
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
        self._sampler = None
        self._spec = None
        self._schedule_key = None
        self._lora_key = None
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
