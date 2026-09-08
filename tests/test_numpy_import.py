from __future__ import annotations

import gc

import mlx.core as mx
import numpy as np

from wee_todd_mlx.numpy_import import adopt_numpy_array


def test_adopt_numpy_array_shares_host_storage_without_copying():
    source = np.arange(16, dtype=np.float32)
    adopted = adopt_numpy_array(source)
    source[3] = 99.0
    assert float(adopted[3]) == 99.0


def test_adopt_numpy_array_keeps_temporary_owner_alive():
    def build():
        return adopt_numpy_array(np.arange(32, dtype=np.int32))

    adopted = build()
    gc.collect()
    assert int(mx.sum(adopted).item()) == sum(range(32))


def test_adopt_numpy_array_materializes_requested_contiguous_dtype():
    source = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]
    adopted = adopt_numpy_array(source, dtype=np.float32)
    assert adopted.dtype == mx.float32
    assert np.array_equal(np.asarray(adopted), source.astype(np.float32))
