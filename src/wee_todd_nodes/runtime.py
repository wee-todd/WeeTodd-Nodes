"""Lazy, process-local MiniMax H3 MLX runtime management."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class H3ModelSpec:
    checkpoint: str
    transformer: str | None = None
    load_vision: bool = False

    def validate(self) -> None:
        if not Path(self.checkpoint).expanduser().is_dir():
            raise FileNotFoundError(f"MiniMax H3 checkpoint directory not found: {self.checkpoint}")
        if self.transformer and not Path(self.transformer).expanduser().is_dir():
            raise FileNotFoundError(f"Transformer directory not found: {self.transformer}")


@dataclass(frozen=True)
class H3GenerationConfig:
    duration_seconds: float = 5.0
    steps: int = 16
    seed: int = 0
    width: int = 640
    height: int = 384
    drop_adaln: bool = True

    def validate(self) -> None:
        if not 5.0 <= self.duration_seconds <= 15.0:
            raise ValueError("MiniMax H3 duration must be between 5 and 15 seconds")
        if self.steps < 2:
            raise ValueError("steps must be at least 2")
        if self.width % 32 or self.height % 32:
            raise ValueError("width and height must be divisible by 32")


class H3RuntimeCache:
    def __init__(self) -> None:
        self._lock = RLock()
        self._spec: H3ModelSpec | None = None
        self._pipeline: Any = None

    def get(self, spec: H3ModelSpec):
        spec.validate()
        with self._lock:
            if self._pipeline is None or self._spec != spec:
                from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

                self._pipeline = MiniMaxH3Pipeline.from_pretrained(
                    str(Path(spec.checkpoint).expanduser()),
                    transformer_dir=(
                        str(Path(spec.transformer).expanduser()) if spec.transformer else None
                    ),
                    load_vision=spec.load_vision,
                )
                self._spec = spec
            return self._pipeline

    def unload(self) -> None:
        with self._lock:
            self._pipeline = None
            self._spec = None
            try:
                import mlx.core as mx
                mx.clear_cache()
            except (ImportError, AttributeError):
                pass


RUNTIME = H3RuntimeCache()
