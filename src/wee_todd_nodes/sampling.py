"""Process-local transformer-only sampling lifecycle for ComfyUI nodes."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from .conditioning import H3Conditioning, H3TextEncoderSpec
from .preflight import H3ComponentSetSpec
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
        if self.task not in tasks:
            raise ValueError(
                f"Checkpoint does not support task {self.task!r}; supported tasks: {tasks}."
            )
        if self.task != "t2va":
            raise ValueError("The first transformer sampler supports the t2va task only.")


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


SamplerFactory = Callable[[H3TransformerSpec], Any]


def _default_sampler_factory(spec: H3TransformerSpec):
    from minimax_h3_mlx.config import PipelineConfig
    from minimax_h3_mlx.load import load_dit
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    dit = load_dit(spec.transformer)
    pipeline_config = PipelineConfig.from_model_index(Path(spec.checkpoint) / "model_index.json")
    return MiniMaxH3Pipeline(dit, None, None, None, pipeline_config)


class H3TransformerCache:
    """Cache one transformer-only sampler with schedule-safe reuse."""

    def __init__(self, factory: SamplerFactory | None = None) -> None:
        self._lock = RLock()
        self._factory = factory or _default_sampler_factory
        self._spec: H3TransformerSpec | None = None
        self._schedule_key: tuple[int, bool] | None = None
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
    ) -> H3Latents:
        spec.validate()
        config.validate()
        if conditioning.load_vision:
            raise ValueError("The first transformer sampler accepts text-only conditioning.")
        expected_encoder = H3TextEncoderSpec(
            text_encoder=spec.text_encoder,
            processor=spec.processor,
            tokenizer=spec.tokenizer,
            load_vision=False,
            config_path=spec.text_encoder_config,
        )
        if conditioning.encoder_spec != expected_encoder:
            raise ValueError(
                "Conditioning was produced by a different Qwen3-VL component specification."
            )
        schedule_key = (config.steps, config.drop_adaln)
        with self._lock:
            if (
                self._sampler is None
                or self._spec != spec
                or self._schedule_key != schedule_key
            ):
                self._release_locked()
                self._sampler = self._factory(spec)
                self._spec = spec
                self._schedule_key = schedule_key
            try:
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
                )
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
                    seconds_per_evaluation=result.seconds_per_evaluation,
                    total_seconds=result.total_seconds,
                    transformer_spec=spec,
                    generation_config=config,
                )
            except BaseException:
                self._release_locked()
                raise
            if unload_after:
                self._release_locked()
            return latents

    def unload(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        self._sampler = None
        self._spec = None
        self._schedule_key = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass


TRANSFORMER_RUNTIME = H3TransformerCache()
