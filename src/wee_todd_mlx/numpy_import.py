"""Zero-copy NumPy imports for the project-wide MLX 0.32.2 runtime."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def adopt_numpy_array(value: np.ndarray, *, dtype: np.dtype | None = None) -> mx.array:
    """Adopt an immutable, C-contiguous host array into MLX unified memory.

    The caller must not mutate ``value`` after this call. MLX retains the NumPy owner for the
    returned array's lifetime, so temporary arrays remain valid after their Python scope ends.
    """
    owner = np.ascontiguousarray(value, dtype=dtype)
    return mx.asarray(owner, copy=False)


__all__ = ["adopt_numpy_array"]
