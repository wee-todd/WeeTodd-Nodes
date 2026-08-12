from __future__ import annotations

import mlx.core as mx
import pytest

from minimax_h3_mlx.algorithm_search.nax_attention import (
    NAXAttentionTile,
    apple_gpu_generation,
    nax_attention,
    nax_attention_available,
)


def test_nax_attention_tile_contract() -> None:
    NAXAttentionTile().validate()
    with pytest.raises(ValueError, match="sixteen rows per simdgroup"):
        NAXAttentionTile(query_rows=32).validate()
    with pytest.raises(ValueError, match="divisible by thirty-two"):
        NAXAttentionTile(key_rows=16).validate()


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("applegpu_g15d", 15),
        ("applegpu_g17g", 17),
        ("applegpu_g18p", 18),
        ("Apple M5", None),
    ],
)
def test_apple_gpu_generation(architecture: str, expected: int | None) -> None:
    assert apple_gpu_generation(architecture) == expected


@pytest.mark.parametrize(
    ("architecture", "macos_version", "expected"),
    [
        ("applegpu_g15d", "26.2", False),
        ("applegpu_g17g", "26.1", False),
        ("applegpu_g17g", "26.2", True),
        ("applegpu_g17d", "27.0", True),
        ("applegpu_g17p", "26.2", False),
        ("applegpu_g18p", "26.2", True),
        ("unknown", "26.2", False),
    ],
)
def test_nax_attention_capability_gate(
    architecture: str, macos_version: str, expected: bool
) -> None:
    assert (
        nax_attention_available(
            device_info={"architecture": architecture},
            macos_version=macos_version,
        )
        is expected
    )


def test_nax_attention_refuses_unsupported_device() -> None:
    if nax_attention_available():
        pytest.skip("This test exercises the unsupported-device guard")
    query = mx.ones((1, 2, 64, 128), dtype=mx.bfloat16)
    with pytest.raises(RuntimeError, match="NAX attention is unavailable"):
        nax_attention(query, query, query, scale=128**-0.5)
