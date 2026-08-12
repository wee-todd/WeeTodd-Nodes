"""Core ML / Apple Neural Engine decoder for MiniMax H3 live previews."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .hires_fix import resize_video_latents_bilinear


class H3CoreMLPreviewDecoder:
    """Run a fixed-shape H3 TAE through Core ML without using the Metal GPU."""

    latent_channels = 24
    spatial_scale = 16
    temporal_frames = 10

    def __init__(self, model_path: str | Path):
        resolved = Path(model_path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"H3 Core ML preview model not found: {resolved}")
        try:
            import coremltools as ct
        except ImportError as exc:
            raise RuntimeError(
                "The Neural Engine preview backend requires the optional coremltools package."
            ) from exc

        self._ct = ct
        if resolved.suffix == ".mlmodelc":
            metadata = json.loads((resolved / "metadata.json").read_text(encoding="utf-8"))
            if not isinstance(metadata, list) or len(metadata) != 1:
                raise ValueError("H3 compiled Core ML preview metadata is invalid.")
            inputs = metadata[0].get("inputSchema", [])
            outputs = metadata[0].get("outputSchema", [])
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError("H3 Core ML preview model must expose one input and output.")
            shape = tuple(json.loads(inputs[0]["shape"]))
            self.input_name = inputs[0]["name"]
            self.output_name = outputs[0]["name"]
            self.model = ct.models.CompiledMLModel(
                str(resolved), compute_units=ct.ComputeUnit.CPU_AND_NE
            )
        else:
            self.model = ct.models.MLModel(str(resolved), compute_units=ct.ComputeUnit.CPU_AND_NE)
            specification = self.model.get_spec()
            inputs = list(specification.description.input)
            if len(inputs) != 1:
                raise ValueError("H3 Core ML preview model must expose exactly one input.")
            shape = tuple(int(value) for value in inputs[0].type.multiArrayType.shape)
            self.input_name = inputs[0].name
            self.output_name = specification.description.output[0].name
        if len(shape) != 5 or shape[:3] != (1, self.latent_channels, self.temporal_frames):
            raise ValueError("H3 Core ML preview input must have shape (1, 24, 10, edge, edge).")
        if shape[3] != shape[4] or shape[3] < 4:
            raise ValueError("H3 Core ML preview requires a fixed square latent input.")
        self.latent_edge = shape[3]
        self.output_edge = self.latent_edge * self.spatial_scale

    def decode(self, latents: mx.array, *, max_edge: int = 384) -> np.ndarray:
        if latents.ndim != 5 or int(latents.shape[1]) != self.latent_channels:
            raise ValueError(
                "H3 preview latents must have shape (batch, 24, frames, height, width)."
            )
        if int(latents.shape[2]) != self.temporal_frames:
            raise ValueError("H3 Neural Engine previews require exactly ten latent frames.")

        source_height, source_width = int(latents.shape[3]), int(latents.shape[4])
        requested_latent_edge = max(4, min(max_edge, self.output_edge) // self.spatial_scale)
        scale = min(1.0, requested_latent_edge / max(source_height, source_width))
        target_height = max(2, int(round(source_height * scale)))
        target_width = max(2, int(round(source_width * scale)))
        working = resize_video_latents_bilinear(
            latents.astype(mx.float16), target_height, target_width
        )
        working = mx.pad(
            working,
            [
                (0, 0),
                (0, 0),
                (0, 0),
                (0, self.latent_edge - target_height),
                (0, self.latent_edge - target_width),
            ],
        )
        mx.eval(working)
        host_latents = np.asarray(working, dtype=np.float16)
        result = self.model.predict({self.input_name: host_latents})[self.output_name]
        output = np.asarray(result, dtype=np.float32)[0]
        return output[
            :,
            : target_height * self.spatial_scale,
            : target_width * self.spatial_scale,
            :,
        ]

    def release(self) -> None:
        self.model = None
        self._ct = None
        gc.collect()
