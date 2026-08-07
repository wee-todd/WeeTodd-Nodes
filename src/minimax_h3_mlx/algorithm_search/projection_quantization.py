"""Packed-weight projection candidates for isolated H3 algorithm research."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class ProjectionQuantizationSpec:
    """One MLX-native packed-weight representation."""

    name: str
    mode: str
    group_size: int
    bits: int


PROJECTION_QUANTIZATION_SPECS = {
    spec.name: spec
    for spec in (
        ProjectionQuantizationSpec("affine4g32", "affine", 32, 4),
        ProjectionQuantizationSpec("affine4", "affine", 64, 4),
        ProjectionQuantizationSpec("affine5", "affine", 64, 5),
        ProjectionQuantizationSpec("affine6", "affine", 64, 6),
        ProjectionQuantizationSpec("affine8", "affine", 64, 8),
        ProjectionQuantizationSpec("mxfp4", "mxfp4", 32, 4),
        ProjectionQuantizationSpec("mxfp8", "mxfp8", 32, 8),
        ProjectionQuantizationSpec("nvfp4", "nvfp4", 16, 4),
    )
}


def quantize_projection(weight: mx.array, spec: ProjectionQuantizationSpec) -> tuple[mx.array, ...]:
    """Quantize one output-by-input projection weight with a declared MLX format."""
    if weight.ndim != 2:
        raise ValueError(f"projection weight must be rank 2, got shape {weight.shape}")
    if weight.shape[-1] % spec.group_size:
        raise ValueError(
            f"input width {weight.shape[-1]} is not divisible by group size {spec.group_size}"
        )
    values = mx.quantize(
        weight,
        group_size=spec.group_size,
        bits=spec.bits,
        mode=spec.mode,
    )
    return tuple(values)


def packed_projection_matmul(
    inputs: mx.array,
    packed: tuple[mx.array, ...],
    spec: ProjectionQuantizationSpec,
) -> mx.array:
    """Apply an MLX packed weight without explicitly expanding it."""
    if len(packed) not in (2, 3):
        raise ValueError("packed projection must contain weights/scales and optional biases")
    weights, scales = packed[:2]
    biases = packed[2] if len(packed) == 3 else None
    return mx.quantized_matmul(
        inputs,
        weights,
        scales,
        biases,
        transpose=True,
        group_size=spec.group_size,
        bits=spec.bits,
        mode=spec.mode,
    )


def packed_nbytes(packed: tuple[mx.array, ...]) -> int:
    """Return physical bytes occupied by a packed weight and its metadata."""
    return sum(int(value.nbytes) for value in packed)


def dynamic_input_probe(
    inputs: mx.array,
    packed: tuple[mx.array, ...],
    spec: ProjectionQuantizationSpec,
) -> tuple[bool, str | None]:
    """Probe MLX dynamic input quantization without hiding unsupported-device errors."""
    if spec.mode not in {"nvfp4", "mxfp8"}:
        return False, "mlx.qqmm only supports nvfp4 and mxfp8"
    try:
        output = mx.qqmm(
            inputs,
            packed[0],
            scales=packed[1],
            group_size=spec.group_size,
            bits=spec.bits,
            mode=spec.mode,
        )
        mx.eval(output)
    except RuntimeError as error:
        return False, str(error)
    return True, None
