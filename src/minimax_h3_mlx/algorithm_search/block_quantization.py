"""Research-only selected-block quantization for full H3 evaluation probes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from minimax_h3_mlx.quantize import CORE_LINEARS

SUPPORTED_EXPERIMENTAL_BITS = frozenset({4, 5, 6, 8})


def parse_block_bit_overrides(values: Iterable[str]) -> dict[int, int]:
    """Parse repeatable ``BLOCK=BITS`` research CLI values with strict validation."""
    overrides: dict[int, int] = {}
    for value in values:
        try:
            block_text, bits_text = value.split("=", 1)
            block, bits = int(block_text), int(bits_text)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"block quantization override must be BLOCK=BITS, got {value!r}"
            ) from error
        if block < 0:
            raise ValueError("block quantization override index must be non-negative")
        if bits not in SUPPORTED_EXPERIMENTAL_BITS:
            raise ValueError(
                f"block quantization override bits must be one of "
                f"{sorted(SUPPORTED_EXPERIMENTAL_BITS)}"
            )
        if block in overrides:
            raise ValueError(f"duplicate block quantization override for block {block}")
        overrides[block] = bits
    return overrides


def selected_block_predicate(
    block_indices: Iterable[int],
    *,
    bits: int,
    group_size: int,
    selected_paths: list[str] | None = None,
) -> Callable[[str, nn.Module], bool | dict[str, int]]:
    """Create an MLX quantization predicate limited to core linears in selected H3 blocks."""
    blocks = frozenset(int(index) for index in block_indices)
    if not blocks or min(blocks) < 0:
        raise ValueError("selected block indices must be non-negative and non-empty")
    if bits not in SUPPORTED_EXPERIMENTAL_BITS:
        raise ValueError(
            f"experimental quantization bits must be one of {sorted(SUPPORTED_EXPERIMENTAL_BITS)}"
        )
    if group_size < 1:
        raise ValueError("experimental quantization group size must be positive")

    prefixes = tuple(f"blocks.{index}." for index in sorted(blocks))

    def predicate(path: str, module: nn.Module) -> bool | dict[str, int]:
        if not isinstance(module, nn.Linear):
            return False
        if not path.startswith(prefixes) or not path.endswith(CORE_LINEARS):
            return False
        if module.weight.shape[-1] % group_size:
            return False
        if selected_paths is not None and path not in selected_paths:
            selected_paths.append(path)
        return {"group_size": group_size, "bits": bits}

    return predicate


def _parameter_bytes(model: Any) -> int:
    return sum(int(value.nbytes) for _, value in tree_flatten(model.parameters()))


def quantize_selected_blocks(
    model: Any,
    block_indices: Iterable[int],
    *,
    bits: int = 5,
    group_size: int = 64,
) -> dict[str, Any]:
    """Quantize selected block projections in place and report the resident weight change."""
    selected_paths: list[str] = []
    before = _parameter_bytes(model)
    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        class_predicate=selected_block_predicate(
            block_indices,
            bits=bits,
            group_size=group_size,
            selected_paths=selected_paths,
        ),
    )
    if not selected_paths:
        raise ValueError("selected block quantization did not match any core linear modules")
    mx.eval(model.parameters())
    after = _parameter_bytes(model)
    return {
        "bits": bits,
        "group_size": group_size,
        "selected_paths": sorted(selected_paths),
        "parameter_bytes_before": before,
        "parameter_bytes_after": after,
        "parameter_bytes_saved": before - after,
    }
