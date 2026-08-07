"""Coordinate exact, low-memory component staging at the ComfyUI adapter boundary."""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock
from typing import Any

_STAGES = frozenset({"text_encoder", "transformer", "video_vae", "audio_vae", "pipeline"})


class H3ResidencyCoordinator:
    """Release inactive H3 runtimes before a low-memory stage loads its weights."""

    def __init__(self) -> None:
        self._lock = RLock()

    def prepare(
        self,
        active_stage: str,
        runtimes: Mapping[str, Any],
        *,
        enabled: bool,
    ) -> tuple[str, ...]:
        if active_stage not in _STAGES:
            raise ValueError(f"Unknown H3 residency stage: {active_stage!r}.")
        if not enabled:
            return ()
        unknown = set(runtimes).difference(_STAGES)
        if unknown:
            raise ValueError(f"Unknown H3 runtime stages: {sorted(unknown)!r}.")

        released = []
        with self._lock:
            for stage, runtime in runtimes.items():
                if stage == active_stage or not runtime.loaded:
                    continue
                runtime.unload()
                released.append(stage)
        return tuple(released)


RESIDENCY = H3ResidencyCoordinator()


def prepare_low_memory_stage(active_stage: str, memory_mode: str) -> tuple[str, ...]:
    """Release stale weighted runtimes before constructing the active BF16 component."""
    from .conditioning import TEXT_ENCODER_RUNTIME
    from .decoding import AUDIO_VAE_RUNTIME, VIDEO_VAE_RUNTIME
    from .runtime import RUNTIME
    from .sampling import TRANSFORMER_RUNTIME

    return RESIDENCY.prepare(
        active_stage,
        {
            "text_encoder": TEXT_ENCODER_RUNTIME,
            "transformer": TRANSFORMER_RUNTIME,
            "video_vae": VIDEO_VAE_RUNTIME,
            "audio_vae": AUDIO_VAE_RUNTIME,
            "pipeline": RUNTIME,
        },
        enabled=memory_mode == "low_memory_bf16",
    )
