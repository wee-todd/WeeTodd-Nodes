"""H3 tensor-layout adapter for original Draw Things checkpoints.

Uses the existing H3 block executor and sampler. Source files remain read-only;
accelerated single-block execution may prepare one additional block ahead.
Native H3's BF16/FP32 arithmetic policy is retained, so this is not a claim of
Draw Things GPU numerical parity.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from threading import local

import numpy as np

from .dt_tensor_store import DTTensorStore


class BlockLookahead:
    """Own at most one prepared block; callers consume slots on the sampling thread.

    The prepare callback owns its worker-thread resources. Closing waits for an
    in-flight preparation before releasing its result, including on cancellation.
    """

    def __init__(self, prepare):
        self.prepare = prepare
        self.pool = None
        self.future = None
        self.index = None
        self.closed = False

    def start(self, index):
        if self.closed:
            raise RuntimeError("DT block lookahead is closed.")
        if self.future is not None:
            raise RuntimeError("DT block lookahead already has a pending block.")
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-dt-lookahead")
        self.index = index
        self.future = self.pool.submit(self.prepare, index)

    def take(self, index):
        if self.future is None:
            return None
        if index != self.index:
            self.discard()
            return None
        future, self.future = self.future, None
        self.index = None
        return future.result()

    def discard(self):
        future, self.future = self.future, None
        self.index = None
        if future is not None:
            future.cancel()
            wait((future,))

    def close(self):
        self.closed = True
        try:
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)
        finally:
            self.future = None
            self.index = None
            self.pool = None


def weight_lookahead_eligible(
    *, enabled, decode_backend, window_size, skip_adaln, projection_backend, cache_budget_bytes
):
    return (
        enabled
        and decode_backend == "mlx"
        and window_size == 1
        and skip_adaln
        and projection_backend == "mpp_experimental"
        and cache_budget_bytes == 0
    )


def unpair_rotary(value, *, heads=56, head_dim=128, rotary_dim=96, array_module=np):
    if value.shape[0] != heads * head_dim or rotary_dim % 2 or rotary_dim > head_dim:
        raise ValueError("Invalid DT rotary tensor shape.")
    order = array_module.array(
        np.concatenate(
            (
                np.arange(0, rotary_dim, 2),
                np.arange(1, rotary_dim, 2),
                np.arange(rotary_dim, head_dim),
            )
        )
    )
    return value.reshape(heads, head_dim, *value.shape[1:])[:, order].reshape(value.shape)


def fuse_qkv(q, k, v, *, heads=56, head_dim=128, array_module=np):
    if q.shape != k.shape or q.shape != v.shape or q.shape[0] != heads * head_dim:
        raise ValueError("Invalid DT Q/K/V shapes.")
    return array_module.stack(
        [a.reshape(heads, head_dim, *a.shape[1:]) for a in (q, k, v)], axis=1
    ).reshape(-1, *q.shape[1:])


class DTH3Mapping:
    def __init__(self, store, *, array_module=np):
        self.store = store
        self.xp = array_module
        self.consumed = set()

    def read(self, name, index=0, parameter=0):
        key = f"__dit__[t-{name}-{index}-{parameter}]"
        self.consumed.add(key)
        value = self.store.read(key)
        return value if self.xp is np else self.xp.array(value)

    def block(self, index, *, refiner=False, skip_adaln=False):
        prefix = "refiner_" if refiner else ""

        def read(name):
            return self.read(prefix + name, index)

        def native_group(roles, layout):
            reader = getattr(self.store, "read_native_group", None)
            if refiner or not skip_adaln or self.xp is np or reader is None:
                return None
            names = [f"__dit__[t-{role}-{index}-0]" for role in roles]
            value = reader(names, layout=layout)
            if value is not None:
                self.consumed.update(names)
            return value

        qkv = native_group(("q", "k", "v"), "qkv")
        if qkv is None:
            q, k, v = (read(name) for name in ("q", "k", "v"))
            if not refiner:
                q, k = (unpair_rotary(a, array_module=self.xp) for a in (q, k))
            qkv = fuse_qkv(q, k, v, array_module=self.xp)
        fc1 = native_group(("gate", "up"), "fc1")
        if fc1 is None:
            fc1 = self.xp.concatenate([read("gate"), read("up")])
        qn, kn = (read(name).reshape(-1) for name in ("norm_q", "norm_k"))
        if not refiner:
            qn, kn = (unpair_rotary(a, heads=1, array_module=self.xp) for a in (qn, kn))
        values = {
            "norm1.weight": read("norm1").reshape(-1),
            "norm2.weight": read("norm2").reshape(-1),
            "attn.qkv_proj.weight": qkv,
            "attn.q_norm.weight": qn,
            "attn.k_norm.weight": kn,
            "attn.out_proj.weight": read("o"),
            "mlp.fc1.weight": fc1,
            "mlp.fc2.weight": read("down"),
        }
        if not refiner and not skip_adaln:
            for parameter, suffix in ((0, "weight"), (1, "bias")):
                values[f"adaln_proj.linear.{suffix}"] = self.xp.concatenate(
                    [
                        self.read(f"adaln_{chunk}_{modality}", index, parameter)
                        for modality in range(3)
                        for chunk in range(6)
                    ]
                )
        return values

    def fixed(self):
        values = {}
        pairs = {
            "video_patch_proj": "proj_in",
            "audio_patch_proj": "audio_proj_in",
            "condition_proj": "context_embedder",
            "time_embedder.proj_in": "time_embedder_linear_1",
            "time_embedder.proj_out": "time_embedder_linear_2",
            "final_layer.video_out": "proj_out",
            "final_layer.audio_out": "audio_proj_out",
        }
        for target, source in pairs.items():
            for parameter, suffix in ((0, "weight"), (1, "bias")):
                values[f"{target}.{suffix}"] = self.read(source, parameter=parameter)
        for i in range(2):
            values.update(
                {f"token_refiner.blocks.{i}.{k}": v for k, v in self.block(i, refiner=True).items()}
            )
        values["token_refiner.final_norm.weight"] = self.read("refiner_final_norm").reshape(-1)
        values["final_layer.norm.weight"] = self.read("norm_out").reshape(-1)
        for parameter, suffix in ((0, "weight"), (1, "bias")):
            values[f"final_layer.adaln_proj.linear.{suffix}"] = self.xp.concatenate(
                [
                    self.read("norm_out_shift", parameter=parameter),
                    self.read("norm_out_scale", parameter=parameter),
                ]
            )
        return values


def _native_arrays(values, *, evaluate=True):
    import mlx.core as mx

    from .load import is_fp32_key

    out = {}
    for key in list(values):
        array = values.pop(key)
        out[key] = mx.array(array).astype(mx.float32 if is_fp32_key(key) else mx.bfloat16)
        if evaluate:
            mx.eval(out[key])
    return out


def _load_native_record(mapping, record_file, *, skip_adaln):
    import mlx.core as mx

    batch = mapping.xp is mx and record_file != "fixed" and skip_adaln

    def prepare():
        if record_file == "fixed":
            values = mapping.fixed()
        else:
            index = int(record_file)
            values = {
                f"blocks.{index}.{k}": v
                for k, v in mapping.block(index, skip_adaln=skip_adaln).items()
            }
        return _native_arrays(values, evaluate=not batch)

    return mapping.store.materialize(prepare) if batch else prepare()


def load_dt_h3_dit(path: str | Path, *, window_size=1, decode_backend="mlx"):
    """Load H3 fixed weights, leaving transformer blocks in their original DT file."""
    if decode_backend not in {"mlx", "reference"}:
        raise ValueError("DT weight decode backend must be mlx or reference.")
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from .config import DiTConfig
    from .dit import MiniMaxH3DiT
    from .paged_checkpoint import (
        PagedBlockExecutor,
        PagedCheckpointManifest,
        PagedTensorStore,
        PageRecord,
    )

    path = Path(path).expanduser().resolve(strict=True)
    describe_dt_h3(path)
    config = DiTConfig()
    # Logical records feed the existing executor; no page files are generated or opened.
    records = tuple(PageRecord(str(i), 0, 0, "") for i in range(config.num_layers))
    manifest = PagedCheckpointManifest(
        path.parent, config.num_layers, path.stat().st_size, PageRecord("fixed", 0, 0, ""), records
    )

    class DirectStore(PagedTensorStore):
        def __init__(self):
            super().__init__(manifest)
            if decode_backend == "mlx":
                from .dt_mlx_decode import DTMLXTensorStore

                self.source = DTMLXTensorStore(path)
            else:
                self.source = DTTensorStore(path)
            self.mapping = DTH3Mapping(
                self.source, array_module=mx if decode_backend == "mlx" else np
            )
            self.lookahead = BlockLookahead(self._prepare_block)
            self.worker_local = local()
            self.lookahead_hits = 0
            self.lookahead_wait_seconds = 0.0
            self.lookahead_peak_bytes = 0
            self.lookahead_source_report = {}

        def _prepare_block(self, index):
            from .dt_mlx_decode import DTMLXTensorStore

            # MLX stream registration and SQLite connections are thread-owned.
            # Complete evaluation before transferring the arrays to the sampler.
            if not hasattr(self.worker_local, "stream"):
                self.worker_local.stream = mx.new_stream(mx.gpu)
            self.source._check()
            started = time.perf_counter()
            with mx.stream(self.worker_local.stream), DTMLXTensorStore(path) as source:
                mapping = DTH3Mapping(source, array_module=mx)
                values = _load_native_record(mapping, str(index), skip_adaln=True)
                try:
                    self.source._check()
                except BaseException:
                    values.clear()
                    raise
                return values, source.report(), time.perf_counter() - started

        def _load_record(self, record, *, skip_adaln):
            self.source._check()
            if record.file == "fixed" or not skip_adaln or self.cache_budget_bytes:
                self.lookahead.discard()
            if self._cache_enabled and record.file in self._retained:
                self.raw_cache_hits += 1
                return self._retained[record.file]
            started = time.perf_counter()
            prepared = self.lookahead.take(record.file)
            if prepared is None:
                before = self.source.payload_bytes_read
                values = _load_native_record(self.mapping, record.file, skip_adaln=skip_adaln)
                self.file_tensor_bytes += self.source.payload_bytes_read - before
                self.disk_load_seconds += time.perf_counter() - started
            else:
                # Revalidate after waiting too: the file may have changed since preparation.
                try:
                    self.source._check()
                except BaseException:
                    prepared[0].clear()
                    raise
                values, source_report, elapsed = prepared
                self.lookahead_hits += 1
                self.lookahead_wait_seconds += time.perf_counter() - started
                self.lookahead_peak_bytes = max(
                    self.lookahead_peak_bytes, sum(v.nbytes for v in values.values())
                )
                for key, value in source_report.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        self.lookahead_source_report[key] = (
                            self.lookahead_source_report.get(key, 0) + value
                        )
                self.file_tensor_bytes += source_report["payload_bytes_read"]
                self.disk_load_seconds += elapsed
            self.disk_page_loads += 1
            size = sum(v.nbytes for v in values.values())
            if self._cache_enabled:
                self.raw_cache_misses += 1
                if record.file != "fixed" and self.retained_bytes + size <= self.cache_budget_bytes:
                    self._retained[record.file] = values
                    self._retained_adaln_bytes[record.file] = 0
                    self.retained_bytes += size
                    self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
            return values

    class DirectExecutor(PagedBlockExecutor):
        weight_lookahead_enabled = False

        def configure_weight_lookahead(self, enabled):
            self.store.lookahead.discard()
            self.weight_lookahead_enabled = bool(enabled)

        def _lookahead_eligible(self):
            return weight_lookahead_eligible(
                enabled=self.weight_lookahead_enabled,
                decode_backend=decode_backend,
                window_size=self.window_size,
                skip_adaln=self.skip_adaln,
                projection_backend=self.projection_backend,
                cache_budget_bytes=self.store.cache_budget_bytes,
            )

        @contextmanager
        def _selected_window(self, indices, prefetch_after):
            if prefetch_after is None or not self._lookahead_eligible():
                self.store.lookahead.discard()
            completed = False
            try:
                with super()._selected_window(indices, prefetch_after) as blocks:
                    if (
                        self._lookahead_eligible()
                        and prefetch_after is not None
                        and prefetch_after < self.num_blocks
                    ):
                        self.store.lookahead.start(str(prefetch_after))
                    yield blocks
                completed = True
            finally:
                if not completed:
                    self.store.lookahead.discard()

        def close(self):
            try:
                self.store.lookahead.close()
                super().close()
                self.store.release()
            finally:
                self.store.source.close()

        def report(self):
            source_report = self.store.source.report()
            for key, value in self.store.lookahead_source_report.items():
                source_report[key] += value
            return {
                **super().report(),
                **source_report,
                "weight_decode_backend": decode_backend,
                "execution_dtype": "native_bf16_fp32",
                "weight_lookahead_enabled": self.weight_lookahead_enabled,
                "weight_lookahead_eligible": self._lookahead_eligible(),
                "weight_lookahead_max_blocks": 1,
                "weight_lookahead_hits": self.store.lookahead_hits,
                "weight_lookahead_wait_seconds": self.store.lookahead_wait_seconds,
                "weight_lookahead_peak_bytes": self.store.lookahead_peak_bytes,
                "weight_lookahead_statistics_scope": (
                    "consumed blocks; preparation overlaps compute"
                ),
            }

    store = DirectStore()
    executor = None
    try:
        model = MiniMaxH3DiT(config)
        model.blocks = []
        expected = {k: v.shape for k, v in tree_flatten(model.parameters())}
        fixed = store.load_fixed()
        if {k: v.shape for k, v in fixed.items()} != expected:
            raise ValueError("DT H3 fixed tensor mapping does not match the native architecture.")
        model.update(tree_unflatten(list(fixed.items())))
        mx.eval(model.parameters())
        fixed.clear()
        store.release()
        executor = DirectExecutor(manifest, config, None, window_size, prefetch=False)
        executor.store = store
        model.paged_blocks = executor
        return model
    except BaseException:
        if executor is not None:
            executor.close()
        else:
            store.source.close()
        raise


def is_dt_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(16) == b"SQLite format 3\0"


def describe_dt_h3(path):
    """Validate the supported FL2VA tensor inventory without loading weights."""
    names = set()
    fixed = set()
    roles = ("q", "k", "v", "o", "gate", "up", "down", "norm1", "norm2", "norm_q", "norm_k")
    for i in range(50):
        names.update(f"__dit__[t-{role}-{i}-0]" for role in roles)
        names.update(
            f"__dit__[t-adaln_{chunk}_{modality}-{i}-{param}]"
            for chunk in range(6)
            for modality in range(3)
            for param in (0, 1)
        )
    for i in range(2):
        fixed.update(f"__dit__[t-refiner_{role}-{i}-0]" for role in roles)
    for role in (
        "proj_in",
        "audio_proj_in",
        "context_embedder",
        "time_embedder_linear_1",
        "time_embedder_linear_2",
        "proj_out",
        "audio_proj_out",
        "norm_out_shift",
        "norm_out_scale",
    ):
        fixed.update(f"__dit__[t-{role}-0-{param}]" for param in (0, 1))
    fixed.update({"__dit__[t-norm_out-0-0]", "__dit__[t-refiner_final_norm-0-0]"})
    from .dt_source import validate_inventory

    with DTTensorStore(path) as store:
        validate_inventory(store.records, "transformer")
        if set(store.records) != names | fixed:
            raise ValueError("DT checkpoint is not the supported H3 FL2VA transformer inventory.")
        for name in store.records:
            r = store.validate_tensor(name)
            if r.codec & 0x10000000:
                store._span(r, store._inline(r))
        block_sizes = []
        for i in range(50):
            block_sizes.append(
                sum(
                    r.elements * 2
                    for n, r in store.records.items()
                    if n not in fixed and n.endswith((f"-{i}-0]", f"-{i}-1]"))
                )
            )
        return {
            "tensor_count": len(store.records),
            "tensor_bytes": sum(r.elements * 2 for r in store.records.values()),
            "fixed_bytes": sum(store.records[n].elements * 2 for n in fixed),
            "window_bytes": max(block_sizes),
            "adaln_bytes": sum(r.elements * 2 for n, r in store.records.items() if "t-adaln_" in n),
        }
