"""Bounded file-cache read-ahead for sequential LTX 2.5 checkpoint pages."""

from __future__ import annotations

import fcntl
import os
import struct
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

_DARWIN_F_RDADVISE = 44


@dataclass(frozen=True)
class PagePrefetchResult:
    index: int
    bytes_advised: int
    seconds: float


class LTX25PagePrefetch:
    """Advise one future page through one worker without materializing MLX arrays."""

    def __init__(
        self,
        root: Path,
        records: Sequence,
        *,
        enabled: bool = True,
        reader: Callable[[int, Path], PagePrefetchResult] | None = None,
        thread_name: str = "ltx25-page-prefetch",
    ) -> None:
        self.root = Path(root)
        self.records = tuple(records)
        self.enabled = bool(enabled and sys.platform == "darwin")
        self._reader = reader or self._advise
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name)
            if self.enabled
            else None
        )
        self._future: Future[PagePrefetchResult] | None = None
        self._future_index: int | None = None
        self._lock = RLock()
        self._closed = False
        self.requests = 0
        self.hits = 0
        self.failures = 0
        self.bytes_advised = 0
        self.read_seconds = 0.0
        self.wait_seconds = 0.0

    @staticmethod
    def default_enabled() -> bool:
        # The first matched 1344x768 Q8-paged measurement was 3% slower with
        # advisory read-ahead. Keep the exact path available for other storage
        # and hardware combinations, but never impose it without an opt-in.
        value = os.environ.get("WEETODD_LTX25_PAGE_PREFETCH", "0").strip().lower()
        return value not in {"0", "false", "no", "off"}

    def _advise(self, index: int, path: Path) -> PagePrefetchResult:
        started = time.perf_counter()
        descriptor = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(descriptor).st_size
            offset = 0
            while offset < size:
                count = min(size - offset, (1 << 31) - 1)
                fcntl.fcntl(descriptor, _DARWIN_F_RDADVISE, struct.pack("=qi4x", offset, count))
                offset += count
        finally:
            os.close(descriptor)
        return PagePrefetchResult(index, size, time.perf_counter() - started)

    def start(self, index: int) -> None:
        if not self.enabled or not 0 <= index < len(self.records):
            return
        with self._lock:
            if self._closed:
                return
            if self._future is not None:
                if self._future_index == index:
                    return
                raise RuntimeError("LTX 2.5 page prefetch must be consumed before reuse.")
            self.requests += 1
            self._future_index = index
            path = self.root / self.records[index].file
            self._future = self._executor.submit(self._reader, index, path)

    def wait(self, index: int) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            if self._future is None or self._future_index != index:
                return False
            future = self._future
            self._future = None
            self._future_index = None
        started = time.perf_counter()
        try:
            result = future.result()
        except Exception:
            with self._lock:
                self.failures += 1
                self.wait_seconds += time.perf_counter() - started
            return False
        with self._lock:
            self.hits += 1
            self.bytes_advised += result.bytes_advised
            self.read_seconds += result.seconds
            self.wait_seconds += time.perf_counter() - started
        return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            future = self._future
            self._future = None
            self._future_index = None
            executor = self._executor
            self._executor = None
        if future is not None:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def report(self) -> dict[str, int | float | bool | str]:
        with self._lock:
            return {
                "prefetch_enabled": self.enabled,
                "prefetch_backend": "darwin_advisory" if self.enabled else "disabled",
                "prefetch_depth": 1 if self.enabled else 0,
                "prefetch_requests": self.requests,
                "prefetch_hits": self.hits,
                "prefetch_failures": self.failures,
                "prefetch_bytes": self.bytes_advised,
                "prefetch_read_seconds": self.read_seconds,
                "prefetch_wait_seconds": self.wait_seconds,
                "prefetch_hidden_seconds": max(self.read_seconds - self.wait_seconds, 0.0),
                "prefetch_buffer_bytes": 0,
            }


__all__ = ["LTX25PagePrefetch", "PagePrefetchResult"]
