"""Slow, independent Sol-style attention reference for H3 research.

This module is not a production attention backend. It preserves the complete packed multimodal
prefix exactly and applies query-dependent approximate sparsity only to target-video rows.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class SolReferenceConfig:
    """Configuration for the bounded numerical reference."""

    prefix_rows: int
    block_size: int = 64
    beta: float = 1.0
    threshold_mode: str = "diagonal"
    approximate_correction: bool = True
    force_all_exact: bool = False
    target_query_blocks: tuple[int, ...] = ()

    def validate(self, sequence_rows: int) -> None:
        if not 0 < self.prefix_rows < sequence_rows:
            raise ValueError("Sol attention prefix_rows must be inside the packed sequence")
        if self.block_size < 1:
            raise ValueError("Sol attention block_size must be positive")
        if not math.isfinite(self.beta):
            raise ValueError("Sol attention beta must be finite")
        if self.threshold_mode not in {"exact", "diagonal"}:
            raise ValueError("Sol attention threshold_mode must be exact or diagonal")
        target_blocks = math.ceil((sequence_rows - self.prefix_rows) / self.block_size)
        if any(index < 0 or index >= target_blocks for index in self.target_query_blocks):
            raise ValueError("Sol attention target query block index is out of range")


@dataclass(frozen=True)
class SolReferenceTelemetry:
    """Routing and execution facts from one reference call."""

    sequence_rows: int
    prefix_rows: int
    target_rows: int
    block_size: int
    target_key_blocks: int
    evaluated_query_blocks: int
    evaluated_query_rows: int
    exact_route_blocks: int
    total_route_blocks: int
    exact_route_density: float
    approximate_correction: bool
    threshold_mode: str
    beta: float
    force_all_exact: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_qkv(query: mx.array, key: mx.array, value: mx.array) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Sol attention requires [batch, heads, rows, head_dim] arrays")
    if key.shape != value.shape:
        raise ValueError("Sol attention requires matching key and value shapes")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("Sol attention query, key, and value batch/head dimensions must match")
    if query.shape[-1] < 1:
        raise ValueError("Sol attention head dimension must be positive")


def dense_attention_reference(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    scale: float,
) -> mx.array:
    """Calculate dense float32 attention for small or sampled numerical checks."""
    _validate_qkv(query, key, value)
    scores = (query.astype(mx.float32) @ key.astype(mx.float32).transpose(0, 1, 3, 2)) * scale
    return mx.softmax(scores, axis=-1) @ value.astype(mx.float32)


def _component_from_exact(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    scale: float,
) -> tuple[mx.array, mx.array, mx.array]:
    scores = (query @ key.transpose(0, 1, 3, 2)) * scale
    maximum = mx.max(scores, axis=-1, keepdims=True)
    weights = mx.exp(scores - maximum)
    denominator = mx.sum(weights, axis=-1, keepdims=True)
    numerator = weights @ value
    return maximum, denominator, numerator


def _merge_components(
    maximum: mx.array | None,
    denominator: mx.array | None,
    numerator: mx.array | None,
    component_maximum: mx.array,
    component_denominator: mx.array,
    component_numerator: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    if maximum is None or denominator is None or numerator is None:
        return component_maximum, component_denominator, component_numerator
    merged_maximum = mx.maximum(maximum, component_maximum)
    old_scale = mx.where(
        denominator > 0,
        mx.exp(maximum - merged_maximum),
        mx.zeros_like(denominator),
    )
    component_scale = mx.where(
        component_denominator > 0,
        mx.exp(component_maximum - merged_maximum),
        mx.zeros_like(component_denominator),
    )
    return (
        merged_maximum,
        denominator * old_scale + component_denominator * component_scale,
        numerator * old_scale + component_numerator * component_scale,
    )


def _thresholds(
    query: mx.array,
    pooled_keys: mx.array,
    proxy_scores: mx.array,
    *,
    scale: float,
    beta: float,
    mode: str,
) -> mx.array:
    if mode == "exact":
        mean = mx.mean(proxy_scores, axis=-1, keepdims=True)
        variance = mx.mean((proxy_scores - mean) ** 2, axis=-1, keepdims=True)
    else:
        key_mean = mx.mean(pooled_keys, axis=-2, keepdims=True)
        key_second = mx.mean(pooled_keys * pooled_keys, axis=-2, keepdims=True)
        mean = mx.sum(query * key_mean, axis=-1, keepdims=True) * scale
        diagonal_variance = mx.maximum(key_second - key_mean * key_mean, 0.0)
        variance = mx.sum(query * query * diagonal_variance, axis=-1, keepdims=True) * (
            scale * scale
        )
    return mean + beta * mx.sqrt(mx.maximum(variance, 0.0) + 1e-12)


def sol_reference_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    scale: float,
    config: SolReferenceConfig,
) -> tuple[mx.array, SolReferenceTelemetry]:
    """Run corrected block-sparse attention over selected H3 target-video query blocks.

    If ``target_query_blocks`` is empty, the output contains the dense prefix followed by every
    target-video row. Otherwise, the output contains only the requested target-video blocks in the
    requested order. The subset mode bounds real-QKV studies at long sequence lengths.
    """
    _validate_qkv(query, key, value)
    if query.shape != key.shape:
        raise ValueError("Sol attention full reference requires matching Q/K/V row counts")
    sequence_rows = int(query.shape[-2])
    config.validate(sequence_rows)
    q = query.astype(mx.float32)
    k = key.astype(mx.float32)
    v = value.astype(mx.float32)
    prefix = config.prefix_rows
    block_size = config.block_size
    target_rows = sequence_rows - prefix
    key_ranges = [
        (start, min(start + block_size, sequence_rows))
        for start in range(prefix, sequence_rows, block_size)
    ]
    pooled_keys = mx.stack(
        [mx.mean(k[..., start:stop, :], axis=-2) for start, stop in key_ranges],
        axis=-2,
    )
    summed_values = [mx.sum(v[..., start:stop, :], axis=-2) for start, stop in key_ranges]
    key_counts = [stop - start for start, stop in key_ranges]

    if config.target_query_blocks:
        query_blocks = list(config.target_query_blocks)
        output_parts: list[mx.array] = []
    else:
        query_blocks = list(range(len(key_ranges)))
        output_parts = [dense_attention_reference(q[..., :prefix, :], k, v, scale=scale)]

    selected_counts: list[mx.array] = []
    evaluated_rows = 0
    for query_block in query_blocks:
        query_start, query_stop = key_ranges[query_block]
        query_piece = q[..., query_start:query_stop, :]
        evaluated_rows += query_stop - query_start
        proxy = mx.einsum("bhqd,bhkd->bhqk", query_piece, pooled_keys) * scale
        threshold = _thresholds(
            query_piece,
            pooled_keys,
            proxy,
            scale=scale,
            beta=config.beta,
            mode=config.threshold_mode,
        )
        selected = mx.mean(proxy - threshold, axis=-2) >= 0
        if config.force_all_exact:
            selected = mx.ones(selected.shape, dtype=mx.bool_)
        selected_counts.append(mx.sum(selected))

        maximum, denominator, numerator = _component_from_exact(
            query_piece,
            k[..., :prefix, :],
            v[..., :prefix, :],
            scale=scale,
        )
        for key_block, (key_start, key_stop) in enumerate(key_ranges):
            exact_maximum, exact_denominator, exact_numerator = _component_from_exact(
                query_piece,
                k[..., key_start:key_stop, :],
                v[..., key_start:key_stop, :],
                scale=scale,
            )
            route = selected[..., key_block, None, None]
            proxy_maximum = proxy[..., key_block, None]
            if config.approximate_correction:
                approximate_denominator = mx.full_like(proxy_maximum, float(key_counts[key_block]))
                approximate_numerator = mx.broadcast_to(
                    summed_values[key_block][..., None, :], exact_numerator.shape
                )
            else:
                approximate_denominator = mx.zeros_like(proxy_maximum)
                approximate_numerator = mx.zeros_like(exact_numerator)
            component_maximum = mx.where(route, exact_maximum, proxy_maximum)
            component_denominator = mx.where(route, exact_denominator, approximate_denominator)
            component_numerator = mx.where(route, exact_numerator, approximate_numerator)
            maximum, denominator, numerator = _merge_components(
                maximum,
                denominator,
                numerator,
                component_maximum,
                component_denominator,
                component_numerator,
            )
        output_parts.append(numerator / mx.maximum(denominator, 1e-20))

    if selected_counts:
        mx.eval(*selected_counts)
        exact_routes = sum(int(np.asarray(count)) for count in selected_counts)
    else:
        exact_routes = 0
    batch_heads = int(query.shape[0]) * int(query.shape[1])
    total_routes = batch_heads * len(query_blocks) * len(key_ranges)
    telemetry = SolReferenceTelemetry(
        sequence_rows=sequence_rows,
        prefix_rows=prefix,
        target_rows=target_rows,
        block_size=block_size,
        target_key_blocks=len(key_ranges),
        evaluated_query_blocks=len(query_blocks),
        evaluated_query_rows=evaluated_rows,
        exact_route_blocks=exact_routes,
        total_route_blocks=total_routes,
        exact_route_density=(exact_routes / total_routes if total_routes else 0.0),
        approximate_correction=config.approximate_correction,
        threshold_mode=config.threshold_mode,
        beta=config.beta,
        force_all_exact=config.force_all_exact,
    )
    return mx.concatenate(output_parts, axis=-2), telemetry
