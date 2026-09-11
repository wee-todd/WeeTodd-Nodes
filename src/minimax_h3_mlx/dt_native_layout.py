"""Fuse DT int8 expansion and native H3 weight ordering without changing values.

The intermediate FP16 rounding is intentional. Casting the int8 product directly
to BF16 changes weights. This module does not import MLX until execution.
"""

from functools import lru_cache


@lru_cache(maxsize=2)
def _kernel(layout):
    import mlx.core as mx

    count = 3 if layout == "qkv" else 2
    indexing = r"""
        uint part = row / ROWS;
        uint source_row = row % ROWS;
    """
    if layout == "qkv":
        indexing = r"""
        uint part = (row / 128) % 3;
        uint channel = row % 128;
        if (part < 2 && channel < 96) {
            channel = channel < 48 ? channel * 2 : (channel - 48) * 2 + 1;
        }
        uint source_row = (row / 384) * 128 + channel;
        """
    branches = []
    for part in range(count):
        branches.append(
            f"{'if' if part == 0 else 'else if'} (part == {part}) "
            f"value = half(float(weight_{part}[source_index]) "
            f"* float(scale_{part}[source_row]));"
        )
    return mx.fast.metal_kernel(
        name=f"weetodd_dt_native_{layout}",
        input_names=[name for i in range(count) for name in (f"weight_{i}", f"scale_{i}")],
        output_names=["decoded"],
        source=r"""
        uint i = thread_position_in_grid.x;
        if (i >= COUNT) return;
        uint row = i / COLUMNS;
        uint column = i % COLUMNS;
        """
        + indexing
        + r"""
        uint source_index = source_row * COLUMNS + column;
        half value = half(0);
        """
        + "\n".join(branches)
        + r"""
        decoded[i] = bfloat16_t(float(value));
        """,
    )


def decode_group(inputs, shape, *, layout):
    """Decode a validated Q/K/V or gate/up group from owned MLX payload arrays."""
    import mlx.core as mx

    rows, columns = shape
    parts = 3 if layout == "qkv" else 2
    elements = rows * columns * parts
    return _kernel(layout)(
        inputs=inputs,
        template=[("COUNT", elements), ("ROWS", rows), ("COLUMNS", columns)],
        grid=(elements, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows * parts, columns)],
        output_dtypes=[mx.bfloat16],
    )[0]
