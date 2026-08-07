"""Versioned MiniMax H3 video VAE checkpoint conversion.

Released checkpoints store 3D convolution weights in the PyTorch ``OIDHW`` layout. MLX consumes
those weights as ``ODHWI``. Converting that layout once produces an otherwise identical checkpoint
that avoids several gigabytes of CPU transpose traffic every time the decoder is loaded.
"""

from __future__ import annotations

import gc
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

VIDEO_VAE_METADATA_KEY = "minimax_h3_video_vae"
VIDEO_VAE_NATIVE_FORMAT = "minimax-h3-mlx-video-vae"
VIDEO_VAE_NATIVE_FORMAT_VERSION = 1
VIDEO_VAE_SOURCE_LAYOUT = "OIDHW"
VIDEO_VAE_MLX_LAYOUT = "ODHWI"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_video_vae_wrapper(wrapper: dict[str, Any]) -> str:
    """Return the stored tensor layout or reject an unsupported native format."""

    stored_format = wrapper.get("format")
    layout = wrapper.get("tensor_layout")
    version = wrapper.get("format_version")

    if stored_format is None and layout is None and version is None:
        return VIDEO_VAE_SOURCE_LAYOUT
    if stored_format != VIDEO_VAE_NATIVE_FORMAT:
        raise ValueError(f"Unsupported MiniMax H3 video VAE format: {stored_format!r}.")
    if version != VIDEO_VAE_NATIVE_FORMAT_VERSION:
        raise ValueError(
            "Unsupported MiniMax H3 MLX video VAE format version: "
            f"{version!r}; expected {VIDEO_VAE_NATIVE_FORMAT_VERSION}."
        )
    if layout != VIDEO_VAE_MLX_LAYOUT:
        raise ValueError(
            f"Unsupported MiniMax H3 video VAE tensor layout: {layout!r}; "
            f"expected {VIDEO_VAE_MLX_LAYOUT!r}."
        )
    return layout


def prepare_video_vae_tensor(tensor, layout: str):
    """Return one tensor in the MLX runtime layout, materialized safely on the CPU."""

    if layout == VIDEO_VAE_MLX_LAYOUT or tensor.ndim != 5:
        return tensor
    if layout != VIDEO_VAE_SOURCE_LAYOUT:
        raise ValueError(f"Unsupported MiniMax H3 video VAE tensor layout: {layout!r}.")

    import mlx.core as mx

    with mx.stream(mx.cpu):
        tensor = mx.contiguous(tensor.transpose(0, 2, 3, 4, 1))
        mx.eval(tensor)
    return tensor


def _source_description(source: Path) -> tuple[Path, dict[str, Any]]:
    if source.is_file():
        from .load import safetensor_metadata

        metadata = safetensor_metadata(source)
        value = metadata.get(VIDEO_VAE_METADATA_KEY)
        if value is None:
            raise ValueError(f"Single-file video VAE has no {VIDEO_VAE_METADATA_KEY!r} metadata.")
        return source, json.loads(value)

    wrapper_path = source / "config.json"
    source_config_path = source / "source" / "config.json"
    weights_path = source / "source" / "model.safetensors"
    for required in (wrapper_path, source_config_path, weights_path):
        if not required.is_file():
            raise FileNotFoundError(f"MiniMax H3 video VAE input is incomplete: {required}")
    with wrapper_path.open() as handle:
        wrapper = json.load(handle)
    with source_config_path.open() as handle:
        source_config = json.load(handle)
    return weights_path, {
        "source_config": source_config,
        "vae_clip_length": wrapper.get("vae_clip_length", 17),
        "vae_token_drop": wrapper.get("vae_token_drop", 3),
        "latents_mean": wrapper.get("latents_mean", []),
        "latents_std": wrapper.get("latents_std", []),
    }


def convert_video_vae_checkpoint(
    source: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a self-describing MLX-layout video VAE artifact.

    The destination is written through a neighboring temporary file and installed only after the
    safetensors writer succeeds. Existing outputs are preserved unless ``overwrite`` is explicit.
    """

    import mlx.core as mx

    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"MiniMax H3 video VAE input not found: {source}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    weights_path, wrapper = _source_description(source)
    source_layout = validate_video_vae_wrapper(wrapper)
    source_bytes = weights_path.stat().st_size
    source_sha256 = _sha256(weights_path)

    native_wrapper = dict(wrapper)
    native_wrapper.update(
        {
            "format": VIDEO_VAE_NATIVE_FORMAT,
            "format_version": VIDEO_VAE_NATIVE_FORMAT_VERSION,
            "tensor_layout": VIDEO_VAE_MLX_LAYOUT,
            "source_sha256": source_sha256,
            "source_bytes": source_bytes,
        }
    )
    metadata = {VIDEO_VAE_METADATA_KEY: json.dumps(native_wrapper, separators=(",", ":"))}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.tmp.safetensors")
    loaded = mx.load(str(weights_path))
    transposed_tensors = (
        sum(tensor.ndim == 5 for tensor in loaded.values())
        if source_layout == VIDEO_VAE_SOURCE_LAYOUT
        else 0
    )
    converted = {}
    try:
        converted = {
            key: prepare_video_vae_tensor(tensor, source_layout) for key, tensor in loaded.items()
        }
        mx.save_safetensors(str(temporary), converted, metadata=metadata)
        if output.exists() and not overwrite:
            raise FileExistsError(f"Output appeared while conversion was running: {output}")
        temporary.replace(output)
        tensor_count = len(converted)
    finally:
        if temporary.exists():
            temporary.unlink()
        del converted
        del loaded
        gc.collect()
        mx.clear_cache()

    return {
        "output": str(output),
        "source_sha256": source_sha256,
        "output_sha256": _sha256(output),
        "source_bytes": source_bytes,
        "output_bytes": output.stat().st_size,
        "tensor_count": tensor_count,
        "transposed_tensors": transposed_tensors,
        "source_layout": source_layout,
        "output_layout": VIDEO_VAE_MLX_LAYOUT,
    }
