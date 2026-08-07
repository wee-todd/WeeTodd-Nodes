"""Experimental SDPA formulations for isolated H3 benchmarks."""

from __future__ import annotations

import mlx.core as mx


def query_chunked_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float,
    chunk_size: int,
    mask: mx.array | None = None,
    synchronize_chunks: bool = True,
) -> mx.array:
    """Apply fused SDPA to independent query slices with complete keys and values."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    query_rows = int(q.shape[-2])
    if query_rows <= chunk_size:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    pieces = []
    for start in range(0, query_rows, chunk_size):
        stop = min(start + chunk_size, query_rows)
        chunk_mask = mask
        if mask is not None and int(mask.shape[-2]) == query_rows:
            chunk_mask = mask[..., start:stop, :]
        piece = mx.fast.scaled_dot_product_attention(
            q[..., start:stop, :], k, v, scale=scale, mask=chunk_mask
        )
        if synchronize_chunks:
            mx.eval(piece)
        pieces.append(piece)
    return mx.concatenate(pieces, axis=-2)


def head_grouped_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float,
    heads_per_group: int,
    mask: mx.array | None = None,
    synchronize_groups: bool = False,
) -> mx.array:
    """Apply fused SDPA to independent groups along the attention-head axis."""
    if heads_per_group < 1:
        raise ValueError("heads_per_group must be positive")
    heads = int(q.shape[-3])
    if int(k.shape[-3]) != heads or int(v.shape[-3]) != heads:
        raise ValueError("q, k, and v must have the same head count")
    if heads <= heads_per_group:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    pieces = []
    for start in range(0, heads, heads_per_group):
        stop = min(start + heads_per_group, heads)
        group_mask = mask
        if mask is not None and int(mask.shape[-3]) == heads:
            group_mask = mask[..., start:stop, :, :]
        piece = mx.fast.scaled_dot_product_attention(
            q[..., start:stop, :, :],
            k[..., start:stop, :, :],
            v[..., start:stop, :, :],
            scale=scale,
            mask=group_mask,
        )
        if synchronize_groups:
            mx.eval(piece)
        pieces.append(piece)
    return mx.concatenate(pieces, axis=-3)
