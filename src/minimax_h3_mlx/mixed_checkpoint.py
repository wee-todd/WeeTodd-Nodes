"""Streamed mixed-precision checkpoint conversion for MiniMax H3."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
from safetensors import safe_open

from .config import DiTConfig
from .load import SKIP_KEYS, shard_paths
from .quantize import CORE_LINEARS, QuantConfig

MIXED_CHECKPOINT_FORMAT = "minimax-h3-mlx-mixed-quant"
MIXED_CHECKPOINT_VERSION = 1
DEFAULT_MAX_SHARD_BYTES = 1024**3


def block_core_paths(
    blocks: range | list[int] | tuple[int, ...],
    *,
    projections: tuple[str, ...] = CORE_LINEARS,
) -> tuple[str, ...]:
    """Return exact core-projection module paths for selected transformer blocks."""
    indices = tuple(sorted(set(int(index) for index in blocks)))
    if not indices or indices[0] < 0:
        raise ValueError("Mixed-precision block indices must be non-negative and non-empty.")
    invalid = [suffix for suffix in projections if suffix not in CORE_LINEARS]
    if invalid:
        raise ValueError(f"Unsupported mixed-precision projection suffixes: {invalid!r}.")
    return tuple(f"blocks.{index}{suffix}" for index in indices for suffix in projections)


def accepted_q8_blocks_38_49_recipe() -> QuantConfig:
    """Return the measured overrides-only recipe accepted for experimental use."""
    return QuantConfig(
        bits=8,
        group_size=64,
        overrides={path: 8 for path in block_core_paths(range(38, 50))},
        quantize_core=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_config(source: Path, shards: list[Path]) -> dict[str, Any]:
    if source.is_dir():
        config_path = source / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Transformer directory has no config.json: {source}")
        with config_path.open() as handle:
            return json.load(handle)

    with safe_open(str(shards[0]), framework="np") as handle:
        if "adaln_t_table" not in handle.keys():
            raise KeyError("Single-file transformer checkpoints must contain adaln_t_table.")
        grid, rank = handle.get_slice("adaln_t_table").get_shape()
    return asdict(DiTConfig(time_embed_dim=rank, adaln_curve_grid=grid))


def _validate_recipe(recipe: QuantConfig) -> dict[str, int]:
    if recipe.quantize_core:
        raise ValueError("Streamed mixed conversion requires an overrides-only recipe.")
    if recipe.quantize_adaln:
        raise ValueError("Streamed mixed conversion does not quantize AdaLN projections.")
    if recipe.group_size < 1:
        raise ValueError("Mixed-precision group size must be positive.")
    selected = {str(path): int(bits) for path, bits in recipe.overrides.items()}
    if not selected:
        raise ValueError("Mixed-precision conversion requires at least one module override.")
    for path, bits in selected.items():
        if not path.startswith("blocks.") or not path.endswith(CORE_LINEARS):
            raise ValueError(f"Unsupported mixed-precision module path: {path!r}.")
        if bits not in (4, 5, 6, 8):
            raise ValueError(f"Unsupported mixed-precision bit width for {path}: {bits}.")
    return selected


class _ShardWriter:
    def __init__(self, directory: Path, max_bytes: int) -> None:
        if max_bytes < 1:
            raise ValueError("Maximum checkpoint shard size must be positive.")
        self.directory = directory
        self.max_bytes = max_bytes
        self.pending: dict[str, mx.array] = {}
        self.pending_bytes = 0
        self.peak_pending_bytes = 0
        self.weight_map: dict[str, str] = {}
        self.shards: list[str] = []
        self.total_bytes = 0

    def add(self, values: dict[str, mx.array]) -> None:
        size = sum(int(value.nbytes) for value in values.values())
        if self.pending and self.pending_bytes + size > self.max_bytes:
            self.flush()
        duplicate = set(values).intersection(self.weight_map, self.pending)
        if duplicate:
            raise KeyError(f"Duplicate output tensor keys: {sorted(duplicate)!r}.")
        self.pending.update(values)
        self.pending_bytes += size
        self.peak_pending_bytes = max(self.peak_pending_bytes, self.pending_bytes)

    def flush(self) -> None:
        if not self.pending:
            return
        name = f"model-{len(self.shards) + 1:05d}.safetensors"
        target = self.directory / name
        mx.save_safetensors(str(target), self.pending, metadata={"format": "mlx"})
        for key in self.pending:
            self.weight_map[key] = name
        self.shards.append(name)
        self.total_bytes += self.pending_bytes
        self.pending = {}
        self.pending_bytes = 0
        gc.collect()
        mx.clear_cache()

    def finish(self, metadata: dict[str, Any]) -> None:
        self.flush()
        with (self.directory / "model.safetensors.index.json").open("w") as handle:
            json.dump(
                {
                    "metadata": {"total_size": self.total_bytes, **metadata},
                    "weight_map": self.weight_map,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")


def convert_mixed_checkpoint(
    source: str | Path,
    output: str | Path,
    recipe: QuantConfig,
    *,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> dict[str, Any]:
    """Convert selected weights without instantiating or retaining the complete transformer."""
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Transformer checkpoint not found: {source}")
    if output.exists():
        raise FileExistsError(f"Mixed-precision output already exists: {output}")

    selected = _validate_recipe(recipe)
    shards = shard_paths(source)
    config = _source_config(source, shards)
    source_records = [
        {"name": shard.name, "bytes": shard.stat().st_size, "sha256": _sha256(shard)}
        for shard in shards
    ]
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    writer = _ShardWriter(temporary, max_shard_bytes)
    found: set[str] = set()
    input_tensor_count = 0
    output_tensor_count = 0

    try:
        for shard in shards:
            loaded = mx.load(str(shard))
            for key in sorted(loaded):
                if key in SKIP_KEYS:
                    continue
                tensor = loaded[key]
                input_tensor_count += 1
                module = key.removesuffix(".weight") if key.endswith(".weight") else None
                bits = selected.get(module) if module is not None else None
                if bits is None:
                    values = {key: tensor}
                else:
                    if tensor.ndim != 2:
                        raise ValueError(f"Selected projection weight must be rank two: {key}.")
                    if tensor.shape[-1] % recipe.group_size:
                        raise ValueError(
                            "Selected projection input width is not divisible by group size: "
                            f"{key}."
                        )
                    packed, scales, biases = mx.quantize(
                        tensor,
                        group_size=recipe.group_size,
                        bits=bits,
                        mode="affine",
                    )
                    mx.eval(packed, scales, biases)
                    values = {
                        key: packed,
                        f"{module}.scales": scales,
                        f"{module}.biases": biases,
                    }
                    found.add(module)
                writer.add(values)
                output_tensor_count += len(values)
                del tensor
            del loaded
            gc.collect()
            mx.clear_cache()

        missing = sorted(set(selected).difference(found))
        if missing:
            raise KeyError(
                f"Mixed-precision recipe did not match {len(missing)} modules: {missing[:4]!r}."
            )

        quant_config = {
            "format": MIXED_CHECKPOINT_FORMAT,
            "format_version": MIXED_CHECKPOINT_VERSION,
            "bits": recipe.bits,
            "group_size": recipe.group_size,
            "quantize_core": False,
            "quantize_adaln": False,
            "adaln_bits": None,
            "overrides": dict(sorted(selected.items())),
            "source": source_records,
        }
        writer.finish({"quantization": json.dumps(quant_config, separators=(",", ":"))})
        with (temporary / "config.json").open("w") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with (temporary / "quant_config.json").open("w") as handle:
            json.dump(quant_config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        gc.collect()
        mx.clear_cache()

    return {
        "output": str(output),
        "format": MIXED_CHECKPOINT_FORMAT,
        "format_version": MIXED_CHECKPOINT_VERSION,
        "selected_modules": len(selected),
        "input_tensors": input_tensor_count,
        "output_tensors": output_tensor_count,
        "shards": len(writer.shards),
        "tensor_bytes": writer.total_bytes,
        "peak_buffered_output_bytes": writer.peak_pending_bytes,
        "source": source_records,
    }
