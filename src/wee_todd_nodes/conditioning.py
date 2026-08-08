"""Process-local Qwen3-VL conditioning lifecycle for ComfyUI nodes."""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from .preflight import H3ComponentSetSpec


@dataclass(frozen=True)
class H3TextEncoderSpec:
    """Immutable inputs needed to construct the H3 Qwen3-VL conditioner."""

    text_encoder: str
    processor: str
    tokenizer: str
    load_vision: bool = False
    config_path: str | None = None

    @classmethod
    def from_components(
        cls,
        components: H3ComponentSetSpec,
        load_vision: bool = False,
    ) -> H3TextEncoderSpec:
        paths = components.resolved_paths()
        config_path = None
        encoder_config = paths["text_encoder"] / "config.json"
        packaged_architecture = paths["text_encoder"] / "architecture_config.json"
        if packaged_architecture.is_file():
            config_path = str(packaged_architecture)
        if encoder_config.is_file():
            import json

            raw = json.loads(encoder_config.read_text())
            if "text_config" not in raw and config_path is None:
                upstream_config = Path(components.checkpoint) / "text_encoder" / "config.json"
                if upstream_config.is_file():
                    config_path = str(upstream_config)
        return cls(
            text_encoder=str(paths["text_encoder"]),
            processor=str(paths["processor"]),
            tokenizer=str(paths["tokenizer"]),
            load_vision=load_vision,
            config_path=config_path,
        )

    def validate(self) -> None:
        for name in ("text_encoder", "processor", "tokenizer"):
            path = Path(getattr(self, name)).expanduser()
            if not path.is_dir():
                raise FileNotFoundError(f"H3 {name} directory not found: {path}")
        if not (Path(self.text_encoder).expanduser() / "config.json").is_file():
            raise FileNotFoundError(
                f"H3 text_encoder config file not found: {Path(self.text_encoder) / 'config.json'}"
            )
        if self.config_path and not Path(self.config_path).expanduser().is_file():
            raise FileNotFoundError(
                f"H3 text encoder architecture config not found: {self.config_path}"
            )


@dataclass(frozen=True)
class H3Conditioning:
    """Live MLX conditioning values for one H3 request."""

    embeddings: Any
    token_tags: Any
    token_count: int
    prompt: str
    load_vision: bool
    encoder_spec: H3TextEncoderSpec
    paging_report: dict[str, Any] | None = None


EncoderFactory = Callable[[H3TextEncoderSpec], Any]


def _default_encoder_factory(spec: H3TextEncoderSpec):
    from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

    return MiniMaxH3TextEncoder(
        spec.text_encoder,
        load_vision=spec.load_vision,
        processor_dir=spec.processor,
        tokenizer_dir=spec.tokenizer,
        config_path=spec.config_path,
    )


class H3TextEncoderCache:
    """Cache one compatible Qwen3-VL encoder and release it explicitly."""

    def __init__(self, factory: EncoderFactory | None = None) -> None:
        self._lock = RLock()
        self._factory = factory or _default_encoder_factory
        self._spec: H3TextEncoderSpec | None = None
        self._encoder: Any = None

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._encoder is not None

    @property
    def spec(self) -> H3TextEncoderSpec | None:
        with self._lock:
            return self._spec

    def encode(
        self,
        spec: H3TextEncoderSpec,
        prompt: str,
        *,
        unload_after: bool = True,
        prepare_stage: Callable[[], None] | None = None,
    ) -> H3Conditioning:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must contain text.")
        spec.validate()
        if prepare_stage is not None:
            prepare_stage()
        with self._lock:
            if self._encoder is None or self._spec != spec:
                self._release_locked()
                self._encoder = self._factory(spec)
                self._spec = spec
            try:
                embeddings, token_tags = self._encoder.encode(prompt)
                # Materialize the only live outputs before dropping the encoder. Otherwise MLX's
                # lazy graph can retain encoder parameters into the transformer stage.
                try:
                    import mlx.core as mx

                    if type(embeddings).__module__.startswith("mlx."):
                        mx.eval(embeddings, token_tags)
                except ImportError:
                    pass
                token_count = int(token_tags.shape[0])
                pager = getattr(self._encoder, "paged_layers", None)
                conditioning = H3Conditioning(
                    embeddings=embeddings,
                    token_tags=token_tags,
                    token_count=token_count,
                    prompt=prompt,
                    load_vision=spec.load_vision,
                    encoder_spec=spec,
                    paging_report=pager.report() if pager is not None else None,
                )
            except BaseException:
                self._release_locked()
                raise
            if unload_after:
                self._release_locked()
            return conditioning

    def unload(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        self._encoder = None
        self._spec = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass


TEXT_ENCODER_RUNTIME = H3TextEncoderCache()
