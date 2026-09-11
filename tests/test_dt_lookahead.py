"""Bounded lookahead must transfer ownership and drain work before unloading."""

import gc
import threading
import weakref

import pytest


def lookahead(prepare):
    from minimax_h3_mlx import dt_h3_checkpoint

    cls = getattr(dt_h3_checkpoint, "BlockLookahead", None)
    assert cls is not None, "DT block lookahead is not implemented"
    return cls(prepare)


def test_only_one_prepared_block_is_retained_and_transferred():
    calls = []

    class Weights:
        pass

    def prepare(index):
        calls.append(index)
        return Weights()

    worker = lookahead(prepare)
    try:
        worker.start(1)
        with pytest.raises(RuntimeError, match="pending"):
            worker.start(2)
        value = worker.take(1)
        ref = weakref.ref(value)
        del value
        gc.collect()
        assert ref() is None
        assert worker.take(1) is None
        worker.start(2)
        assert isinstance(worker.take(2), Weights)
        assert calls == [1, 2]
    finally:
        worker.close()


def test_nonsequential_request_drains_and_discards_unneeded_weights():
    started = threading.Event()
    release = threading.Event()
    refs = []

    class Weights:
        pass

    def prepare(index):
        value = Weights()
        refs.append(weakref.ref(value))
        started.set()
        assert release.wait(5)
        return value

    worker = lookahead(prepare)
    worker.start(3)
    assert started.wait(5)
    release.set()
    assert worker.take(8) is None
    gc.collect()
    assert refs[0]() is None
    worker.close()


def test_close_waits_for_running_prepare_and_releases_result():
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    refs = []

    class Weights:
        pass

    def prepare(index):
        value = Weights()
        refs.append(weakref.ref(value))
        started.set()
        assert release.wait(5)
        return value

    worker = lookahead(prepare)
    worker.start(0)
    assert started.wait(5)
    closer = threading.Thread(target=lambda: (worker.close(), closed.set()))
    closer.start()
    assert not closed.wait(0.05)
    release.set()
    closer.join(5)
    assert closed.is_set()
    gc.collect()
    assert refs[0]() is None
    worker.close()
    with pytest.raises(RuntimeError, match="closed"):
        worker.start(1)


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_worker_failure_propagates_without_retaining_a_failed_slot(error):
    def prepare(index):
        raise error("decode failed")

    worker = lookahead(prepare)
    try:
        worker.start(2)
        with pytest.raises(error, match="decode failed"):
            worker.take(2)
        assert worker.take(2) is None
    finally:
        worker.close()


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, True),
        ({"enabled": False}, False),
        ({"decode_backend": "reference"}, False),
        ({"window_size": 2}, False),
        ({"skip_adaln": False}, False),
        ({"projection_backend": "mlx"}, False),
        ({"cache_budget_bytes": 1}, False),
    ],
)
def test_lookahead_is_limited_to_accelerated_single_block_uncached_sampling(changes, expected):
    from minimax_h3_mlx import dt_h3_checkpoint

    eligible = getattr(dt_h3_checkpoint, "weight_lookahead_eligible", None)
    assert eligible is not None, "Lookahead needs an explicit eligibility gate"
    options = dict(
        enabled=True,
        decode_backend="mlx",
        window_size=1,
        skip_adaln=True,
        projection_backend="mpp_experimental",
        cache_budget_bytes=0,
    )
    options.update(changes)
    assert eligible(**options) is expected


@pytest.fixture
def tiny_dt(tmp_path, monkeypatch):
    """Real SQLite reader, MLX weights and pager; shrink only the H3 architecture/mapping."""
    import sqlite3
    import struct

    import numpy as np

    mx = pytest.importorskip("mlx.core")
    from mlx.utils import tree_flatten
    from test_paged_checkpoint import _tiny_dit_config

    from minimax_h3_mlx import config as config_module
    from minimax_h3_mlx import dt_h3_checkpoint as dt
    from minimax_h3_mlx import dt_mlx_decode
    from minimax_h3_mlx.dit import MiniMaxH3DiT

    config = _tiny_dit_config()
    weights = dict(tree_flatten(MiniMaxH3DiT(config).parameters()))
    mx.eval(weights)
    path = tmp_path / "tiny.ckpt"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE tensors(name TEXT PRIMARY KEY, type INTEGER, format INTEGER, "
            "datatype INTEGER, dim BLOB, data BLOB)"
        )
        for key, value in weights.items():
            array = np.asarray(value.astype(mx.float16))
            db.execute(
                "INSERT INTO tensors VALUES(?,?,?,?,?,?)",
                (
                    key,
                    1,
                    0,
                    0x20000,
                    struct.pack("<12i", *array.shape, *([0] * (12 - array.ndim))),
                    array.tobytes(),
                ),
            )
    opened, reads = [], []
    real_store = dt_mlx_decode.DTMLXTensorStore

    class Store(real_store):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

    class Mapping:
        def __init__(self, store, *, array_module):
            self.store, self.xp = store, array_module

        def fixed(self):
            return {k: self.store.read(k) for k in weights if not k.startswith("blocks.")}

        def block(self, index, *, skip_adaln):
            reads.append((index, threading.get_ident()))
            prefix = f"blocks.{index}."
            return {
                k[len(prefix) :]: self.store.read(k)
                for k in weights
                if k.startswith(prefix) and not (skip_adaln and ".adaln_proj." in k)
            }

    monkeypatch.setattr(config_module, "DiTConfig", lambda: config)
    monkeypatch.setattr(dt, "describe_dt_h3", lambda path: None)
    monkeypatch.setattr(dt, "DTH3Mapping", Mapping)
    monkeypatch.setattr(dt_mlx_decode, "DTMLXTensorStore", Store)
    model = dt.load_dt_h3_dit(path)
    pager = model.paged_blocks
    pager.skip_adaln = True
    pager.projection_backend = "mpp_experimental"
    yield pager, path, opened, reads, weights
    pager.close()
    assert all(s._db is None and s._fd is None for s in opened)


def enable(pager):
    configure = getattr(pager, "configure_weight_lookahead", None)
    assert configure is not None, "Direct executor has no lookahead integration"
    configure(True)


def test_real_pager_lookahead_matches_every_weight_and_releases_worker_handles(tiny_dt):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from minimax_h3_mlx.load import is_fp32_key

    pager, _, opened, reads, weights = tiny_dt
    enable(pager)
    main_thread = threading.get_ident()
    for index in range(3):
        with pager.window(index) as blocks:
            for key, value in tree_flatten(blocks[0].parameters()):
                full_key = f"blocks.{index}.{key.replace('.base.', '.')}"
                expected = (
                    weights[full_key]
                    .astype(mx.float16)
                    .astype(mx.float32 if is_fp32_key(full_key) else mx.bfloat16)
                )
                assert bool(mx.array_equal(value, expected))
    assert [i for i, _ in reads] == [0, 1, 2]
    assert reads[0][1] == main_thread
    assert all(thread != main_thread for _, thread in reads[1:])
    assert all(s._db is None for s in opened[1:])
    report = pager.report()
    assert report["weight_lookahead_hits"] == 2
    assert report["weight_lookahead_max_blocks"] == 1
    assert report["weight_lookahead_peak_bytes"] > 0
    assert report["disk_page_loads"] == 4  # fixed plus each of three blocks


@pytest.mark.parametrize("route", ["disabled", "cache", "selected", "no_adaln_cache"])
def test_ineligible_or_selected_windows_do_not_decode_extra_blocks(tiny_dt, route):
    pager, _, _, reads, _ = tiny_dt
    enable(pager)
    if route == "disabled":
        pager.configure_weight_lookahead(False)
    elif route == "cache":
        pager.store.configure_cache(1000000)
        pager.store.begin_cache()
    elif route == "no_adaln_cache":
        pager.skip_adaln = False
    context = pager.selected_window((0,)) if route == "selected" else pager.window(0)
    with context:
        pass
    assert reads == [(0, threading.get_ident())]
    assert pager.report()["weight_lookahead_hits"] == 0


def test_source_replacement_rejects_already_prepared_weights(tiny_dt):
    pager, path, _, _, _ = tiny_dt
    enable(pager)
    with pager.window(0):
        pass
    pager.store.lookahead.future.result()
    replacement = path.with_name("replacement")
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    with pytest.raises(RuntimeError, match="changed"):
        with pager.window(1):
            pass
    assert pager.store.lookahead.future is None
    assert pager.store.active_page is None


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_pager_failure_drains_next_block_before_returning(tiny_dt, error):
    pager, _, opened, _, _ = tiny_dt
    enable(pager)
    with pytest.raises(error, match="cancel"):
        with pager.window(0):
            raise error("cancel")
    assert pager.store.lookahead.future is None
    assert pager.store.active_page is None
    assert all(s._db is None for s in opened[1:])


@pytest.mark.parametrize(
    "memory_mode,backend,enabled",
    [
        ("normal", "auto", True),
        ("normal", "mlx", False),
        ("low_memory_bf16", "auto", False),
    ],
)
def test_shared_sampler_configures_lookahead_from_existing_acceleration_controls(
    tiny_dt, tmp_path, memory_mode, backend, enabled
):
    from test_sampling import FakeSampler, _conditioning, _spec

    from wee_todd_nodes.runtime import H3GenerationConfig
    from wee_todd_nodes.sampling import H3TransformerCache

    pager = tiny_dt[0]

    class Sampler(FakeSampler):
        def __init__(self, spec):
            super().__init__(spec)
            self.dit.paged_blocks = pager

        def sample_latents(self, *args, **kwargs):
            assert pager.weight_lookahead_enabled is enabled
            return super().sample_latents(*args, **kwargs)

    cache = H3TransformerCache(Sampler)
    spec = _spec(tmp_path)
    config = H3GenerationConfig(steps=3, memory_mode=memory_mode, projection_backend=backend)
    cache.sample(spec, _conditioning(spec), config, unload_after=True)
    assert not cache.loaded


@pytest.mark.parametrize("stage", ["worker_validation", "consumer_validation"])
def test_source_validation_failure_does_not_pin_weights_in_retained_exception(
    tiny_dt, monkeypatch, stage
):
    from minimax_h3_mlx import dt_h3_checkpoint as dt

    pager = tiny_dt[0]
    enable(pager)
    refs = []
    original_load = dt._load_native_record
    original_check = pager.store.source._check
    main_thread = threading.get_ident()
    checks = 0

    def load(mapping, file, *, skip_adaln):
        values = original_load(mapping, file, skip_adaln=skip_adaln)
        if file == "1":
            refs.extend(weakref.ref(value) for value in values.values())
        return values

    def check():
        nonlocal checks
        on_worker = threading.get_ident() != main_thread
        if on_worker == (stage == "worker_validation"):
            checks += 1
            if checks == 2:
                raise RuntimeError("changed after preparation")
        return original_check()

    monkeypatch.setattr(dt, "_load_native_record", load)
    if stage == "worker_validation":
        monkeypatch.setattr(pager.store.source, "_check", check)
    with pager.window(0):
        pass
    if stage == "consumer_validation":
        pager.store.lookahead.future.result()
        monkeypatch.setattr(pager.store.source, "_check", check)
    with pytest.raises(RuntimeError, match="changed after preparation") as retained_exception:
        with pager.window(1):
            pass
    pager.close()
    gc.collect()
    assert refs and all(ref() is None for ref in refs)
    assert retained_exception.value is not None  # Intentionally keep the traceback alive.


def test_inherited_window_cleanup_failure_also_drains_lookahead(tiny_dt, monkeypatch):
    pager = tiny_dt[0]
    enable(pager)
    release = pager.store.release

    def fail_release():
        release()
        raise RuntimeError("release failed")

    with monkeypatch.context() as patch:
        patch.setattr(pager.store, "release", fail_release)
        with pytest.raises(RuntimeError, match="release failed"):
            with pager.window(0):
                pass
        assert pager.store.lookahead.future is None
