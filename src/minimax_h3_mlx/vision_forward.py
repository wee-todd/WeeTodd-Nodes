"""H3 vision-stage orchestration using the loaded mlx-vlm Qwen3-VL modules.

Keep sequence lengths as host integers. MLX repeat requires an integer repeat
count, whereas some mlx-vlm releases pass a scalar MLX array. No global MLX or
site-package patch is installed here; learned layers and their math are reused.
"""

from __future__ import annotations

import mlx.core as mx


def encode_vision(vision, pixels: mx.array, grid: mx.array):
    rows = grid.tolist()
    if not rows or any(len(row) != 3 or any(int(n) <= 0 for n in row) for row in rows):
        raise ValueError("Qwen vision grid must contain positive temporal/height/width triples")
    boundaries = [0]
    for temporal, height, width in rows:
        for _ in range(int(temporal)):
            boundaries.append(boundaries[-1] + int(height) * int(width))
    hidden = vision.patch_embed(pixels) + vision.fast_pos_embed_interpolate(grid)
    if boundaries[-1] != hidden.shape[0]:
        raise ValueError("Qwen vision grid does not match the number of patch embeddings")
    hidden = hidden.reshape(boundaries[-1], -1)
    rotary = vision.rot_pos_emb(grid).reshape(boundaries[-1], -1)
    lengths = mx.array(boundaries, dtype=mx.int32)
    deep = []
    for index, block in enumerate(vision.blocks):
        hidden = block(hidden, cu_seqlens=lengths, rotary_pos_emb=rotary)
        if index in vision.deepstack_visual_indexes:
            merger = vision.deepstack_merger_list[vision.deepstack_visual_indexes.index(index)]
            deep.append(merger(hidden))
    return vision.merger(hidden), deep
