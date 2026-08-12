"""Compositional H3 chained-timeline contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .continuation import SUPPORTED_CONTEXT_FRAMES


@dataclass(frozen=True)
class H3ChainedTimeline:
    """Global-to-local timestamp map for equal-length overlapping H3 windows."""

    window_frames: int
    window_count: int
    context_frames: int
    fps: int = 24
    target_frames: int | None = None

    def validate(self) -> None:
        if self.window_count < 2 or self.window_count > 16:
            raise ValueError("An H3 chained timeline requires between two and sixteen windows.")
        if self.context_frames not in SUPPORTED_CONTEXT_FRAMES:
            raise ValueError(
                f"H3 chain context must be one of {SUPPORTED_CONTEXT_FRAMES} frames."
            )
        if self.window_frames <= self.context_frames:
            raise ValueError("Each H3 chain window must be longer than its overlap.")
        if self.fps != 24:
            raise ValueError("MiniMax H3 chained timelines run at 24 fps.")
        if self.target_frames is not None and not 1 <= self.target_frames <= self.total_frames:
            raise ValueError("The chain target must fit inside the assembled timeline.")

    @property
    def stride_frames(self) -> int:
        return self.window_frames - self.context_frames

    @property
    def total_frames(self) -> int:
        return self.window_frames + (self.window_count - 1) * self.stride_frames

    @property
    def published_frames(self) -> int:
        return self.target_frames or self.total_frames

    def window_start_frame(self, window_index: int) -> int:
        self.validate()
        if not 1 <= window_index <= self.window_count:
            raise ValueError(f"Window index must be 1..{self.window_count}.")
        return (window_index - 1) * self.stride_frames

    def local_timestamp(self, window_index: int, global_seconds: float) -> float:
        local_frame = round(global_seconds * self.fps) - self.window_start_frame(window_index)
        if not 0 <= local_frame < self.window_frames:
            lo = self.window_start_frame(window_index) / self.fps
            hi = (self.window_start_frame(window_index) + self.window_frames - 1) / self.fps
            raise ValueError(
                f"Global timestamp {global_seconds:g}s is outside window {window_index} "
                f"({lo:g}s..{hi:g}s)."
            )
        return local_frame / self.fps

    def metadata(self) -> dict[str, Any]:
        self.validate()
        return {
            "window_count": self.window_count,
            "window_frames": self.window_frames,
            "context_frames": self.context_frames,
            "stride_frames": self.stride_frames,
            "assembled_frames": self.total_frames,
            "published_frames": self.published_frames,
            "fps": self.fps,
            "window_start_frames": [
                self.window_start_frame(index) for index in range(1, self.window_count + 1)
            ],
        }


@dataclass(frozen=True)
class H3LatentChain:
    """Small latent-native list of synchronized H3 windows awaiting staged publication."""

    timeline: H3ChainedTimeline
    windows: tuple[Any, ...] = ()

    def append(self, latents: Any) -> H3LatentChain:
        self.timeline.validate()
        if len(self.windows) >= self.timeline.window_count:
            raise ValueError("The H3 latent chain already contains every configured window.")
        if latents.num_frames != self.timeline.window_frames:
            raise ValueError(
                f"Chain window has {latents.num_frames} frames; expected "
                f"{self.timeline.window_frames}."
            )
        if latents.fps != 24 or latents.sample_rate != 32000:
            raise ValueError("H3 chain windows must be 24 fps with 32 kHz stereo audio.")
        if self.windows:
            first = self.windows[0]
            if (latents.width, latents.height) != (first.width, first.height):
                raise ValueError("Every H3 chain window must use the same canvas.")
            if latents.transformer_spec != first.transformer_spec:
                raise ValueError("Every H3 chain window must use the same transformer.")
        return H3LatentChain(self.timeline, (*self.windows, latents))

    def validate_complete(self) -> None:
        self.timeline.validate()
        if len(self.windows) != self.timeline.window_count:
            raise ValueError(
                f"H3 latent chain has {len(self.windows)} windows; expected "
                f"{self.timeline.window_count}."
            )
