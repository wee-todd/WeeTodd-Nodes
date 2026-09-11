"""Protect DT source rounding and native head/channel order during fused loading."""

import sqlite3
import struct
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Requires Metal")

from minimax_h3_mlx.dt_mlx_decode import DTMLXTensorStore  # noqa: E402
from minimax_h3_mlx.dt_tensor_store import DTTensorStore  # noqa: E402


def group_checkpoint(tmp_path, *, layout="qkv", external=True, codec=0x8A1E9B):
    names = ("q", "k", "v") if layout == "qkv" else ("gate", "up")
    shape = (256, 3)
    model = tmp_path / "group.ckpt"
    payloads = bytearray()
    with sqlite3.connect(model) as db:
        db.execute(
            "CREATE TABLE tensors(name TEXT PRIMARY KEY, type INTEGER, format INTEGER, "
            "datatype INTEGER, dim BLOB, data BLOB)"
        )
        for i, name in enumerate(names):
            quantized = (np.arange(np.prod(shape)) + 17 * i).astype(np.int8).reshape(shape)
            scales = np.resize(np.array([0.1, -0.3, 0, -0.0, 512, 0.000061], np.float16), shape[0])
            payload = (
                quantized.tobytes() + bytes((-quantized.size) % 128) + scales.tobytes()
                if codec
                else quantized.astype(np.float16).tobytes()
            )
            data = struct.pack("<QQ", len(payloads), len(payload)) if external else payload
            payloads.extend(payload)
            db.execute(
                "INSERT INTO tensors VALUES(?,?,?,?,?,?)",
                (
                    name,
                    ((codec | (0x10000000 if external else 0)) << 32) | 1,
                    0,
                    0x20000,
                    struct.pack("<12i", *shape, *([0] * 10)),
                    data,
                ),
            )
    if external:
        model.with_name(model.name + "-tensordata").write_bytes(payloads)
    return model, names


@pytest.mark.parametrize("layout", ["qkv", "fc1"])
@pytest.mark.parametrize("external", [False, True])
def test_native_group_matches_reference_rounding_order_and_survives_close(
    tmp_path, layout, external
):
    model, names = group_checkpoint(tmp_path, layout=layout, external=external)
    before = model.read_bytes()
    with DTTensorStore(model) as reference, np.errstate(over="ignore"):
        arrays = [reference.read(n) for n in names]
    if layout == "qkv":
        # Hand-defined native head order: split even/odd rotary channels, then tail.
        order = list(range(0, 96, 2)) + list(range(1, 96, 2)) + list(range(96, 128))
        q, k, v = [a.reshape(2, 128, 3) for a in arrays]
        expected = np.stack([q[:, order], k[:, order], v], axis=1).reshape(768, 3)
    else:
        expected = np.concatenate(arrays)
    expected = mx.array(expected).astype(mx.bfloat16)
    mx.eval(expected)
    with DTMLXTensorStore(model) as store:
        actual = store.materialize(lambda: {"w": store.read_native_group(names, layout=layout)})[
            "w"
        ]
        assert store.report()["native_layout_groups"] == 1
        assert store.report()["tensors_read"] == len(names)
        assert store.report()["persistent_weight_bytes_written"] == 0
    assert actual.dtype == mx.bfloat16
    assert bool(mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)))
    assert model.read_bytes() == before


@pytest.mark.parametrize("rounding,codec", [("toward_zero", 0x8A1E9B), ("nearest", 0)])
def test_unsupported_native_group_returns_none_without_reading_payloads(tmp_path, rounding, codec):
    model, names = group_checkpoint(tmp_path, codec=codec)
    with DTMLXTensorStore(model, rounding=rounding) as store:
        assert store.read_native_group(names, layout="qkv") is None
        assert store.payload_bytes_read == 0
        assert store.report()["native_layout_groups"] == 0


def test_native_group_rejects_invalid_sizes_limits_and_changed_source(tmp_path):
    model, names = group_checkpoint(tmp_path)
    with DTMLXTensorStore(model, max_tensor_bytes=64) as store:
        with pytest.raises(ValueError, match="limit"):
            store.read_native_group(names, layout="qkv")
        assert store.payload_bytes_read == 0
    with DTMLXTensorStore(model) as store:
        with pytest.raises(ValueError, match="group"):
            store.read_native_group(names[:2], layout="qkv")
        replacement = tmp_path / "replacement"
        replacement.write_bytes(model.read_bytes())
        replacement.replace(model)
        with pytest.raises(RuntimeError, match="changed"):
            store.read_native_group(names, layout="qkv")


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_native_group_failure_closes_payload_windows(tmp_path, monkeypatch, error):
    import mmap
    import weakref

    from minimax_h3_mlx import dt_native_layout

    model, names = group_checkpoint(tmp_path)
    opened = []
    real_map = mmap.mmap

    def track(*args, **kwargs):
        region = real_map(*args, **kwargs)
        opened.append(weakref.ref(region))
        return region

    def fail(*args, **kwargs):
        raise error("kernel failed")

    monkeypatch.setattr(mmap, "mmap", track)
    monkeypatch.setattr(dt_native_layout, "decode_group", fail)
    with DTMLXTensorStore(model) as store:
        with pytest.raises(error, match="kernel failed") as caught:
            store.materialize(lambda: {"w": store.read_native_group(names, layout="qkv")})
        assert caught.value
        assert opened and all(ref() is None or ref().closed for ref in opened)
        # Recovery must return to normal eager reads after a failed batch.
        assert store.read(names[0]).dtype == mx.float16


@pytest.mark.parametrize("rounding,groups", [("nearest", 2), ("toward_zero", 0)])
def test_native_block_routes_groups_and_preserves_reference_mapping(tmp_path, rounding, groups):
    from minimax_h3_mlx.dt_h3_checkpoint import DTH3Mapping, _load_native_record, _native_arrays

    model = tmp_path / "block.ckpt"
    shapes = {name: (7168, 3) for name in ("q", "k", "v")}
    shapes.update(
        {
            "gate": (4, 3),
            "up": (4, 3),
            "down": (3, 4),
            "o": (3, 7168),
            "norm_q": (128,),
            "norm_k": (128,),
            "norm1": (3,),
            "norm2": (3,),
        }
    )
    with sqlite3.connect(model) as db:
        db.execute(
            "CREATE TABLE tensors(name TEXT PRIMARY KEY, type INTEGER, format INTEGER, "
            "datatype INTEGER, dim BLOB, data BLOB)"
        )
        for i, (role, shape) in enumerate(shapes.items()):
            values = (np.arange(np.prod(shape)) + i * 17).astype(np.int8)
            if len(shape) == 2:
                scales = np.full(shape[0], 0.1, np.float16)
                codec = 0x8A1E9B
                payload = values.tobytes() + bytes((-values.size) % 128) + scales.tobytes()
            else:
                codec = 0
                payload = values.astype(np.float16).tobytes()
            db.execute(
                "INSERT INTO tensors VALUES(?,?,?,?,?,?)",
                (
                    f"__dit__[t-{role}-0-0]",
                    (codec << 32) | 1,
                    0,
                    0x20000,
                    struct.pack("<12i", *shape, *([0] * (12 - len(shape)))),
                    payload,
                ),
            )
    with DTTensorStore(model, rounding=rounding) as store:
        mapper = DTH3Mapping(store)
        expected = _native_arrays(
            {f"blocks.0.{k}": v for k, v in mapper.block(0, skip_adaln=True).items()}
        )
        consumed = mapper.consumed
    with DTMLXTensorStore(model, rounding=rounding) as store:
        mapper = DTH3Mapping(store, array_module=mx)
        actual = _load_native_record(mapper, "0", skip_adaln=True)
        assert actual.keys() == expected.keys()
        assert all(bool(mx.array_equal(actual[k], expected[k])) for k in expected)
        assert mapper.consumed == consumed
        assert store.report()["native_layout_groups"] == groups


def test_native_group_preserves_all_finite_half_scale_int8_products(tmp_path):
    from test_dt_tensor_store import checkpoint

    bits = np.concatenate(
        [np.arange(0x7C00, dtype=np.uint16), np.arange(0x8000, 0xFC00, dtype=np.uint16)]
    )
    scales = bits.view(np.float16)
    quantized = np.tile(np.arange(-128, 128, dtype=np.int8), (len(scales), 1))
    model = checkpoint(
        tmp_path, 0x8A1E9B, quantized.shape, quantized.tobytes() + scales.tobytes(), trailer=True
    )
    with DTTensorStore(model) as reference, np.errstate(over="ignore"):
        expected = mx.array(reference.read("w")).astype(mx.bfloat16)
        mx.eval(expected)
    with DTMLXTensorStore(model) as store:
        actual = store.read_native_group(("w", "w"), layout="fc1")
        rows = len(scales)
        for part in (actual[:rows], actual[rows:]):
            assert bool(mx.array_equal(part.view(mx.uint16), expected.view(mx.uint16)))
