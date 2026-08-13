"""Bounded file-cache read-ahead for sequential MLX checkpoint pages."""

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

PREFETCH_BUFFER_BYTES = 8 * 1024 * 1024
_DARWIN_F_RDADVISE = 44


@dataclass(frozen=True)
class PagePrefetchResult:
    start: int
    size: int
    bytes_read: int
    read_seconds: float


class SequentialPagePrefetch:
    """Warm one future page window through one worker and one reusable read buffer."""

    def __init__(
        self,
        root: Path,
        records: Sequence,
        *,
        enabled: bool = True,
        reader: Callable[[int, tuple[Path, ...]], PagePrefetchResult] | None = None,
        thread_name: str = "h3-page-prefetch",
        backend: str = "stream",
    ):
        self.root = root
        self.records = records
        self.enabled = bool(enabled)
        if backend not in {"stream", "darwin_advisory"}:
            raise ValueError(f"Unsupported H3 page-prefetch backend: {backend!r}.")
        if backend == "darwin_advisory" and sys.platform != "darwin":
            backend = "stream"
        self.backend = backend
        self._reader = reader or self._read_pages
        self._thread_name = thread_name
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name)
            if self.enabled
            else None
        )
        self._future: Future[PagePrefetchResult] | None = None
        self._future_key: tuple[int, int] | None = None
        self._lock = RLock()
        self.requests = 0
        self.hits = 0
        self.failures = 0
        self.bytes_read = 0
        self.read_seconds = 0.0
        self.wait_seconds = 0.0
        self.max_window_bytes = 0
        self._closed = False

    def _read_pages(self, start: int, paths: tuple[Path, ...]) -> PagePrefetchResult:
        started = time.perf_counter()
        total = 0
        if self.backend == "darwin_advisory":
            for path in paths:
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    size = os.fstat(descriptor).st_size
                    # Darwin's radvisory count is a signed 32-bit integer. H3 pages are below
                    # that bound, but split defensively so the helper remains correct.
                    offset = 0
                    while offset < size:
                        count = min(size - offset, (1 << 31) - 1)
                        advice = struct.pack("=qi4x", offset, count)
                        fcntl.fcntl(descriptor, _DARWIN_F_RDADVISE, advice)
                        offset += count
                    total += size
                finally:
                    os.close(descriptor)
        else:
            buffer = bytearray(PREFETCH_BUFFER_BYTES)
            for path in paths:
                with path.open("rb", buffering=0) as handle:
                    while True:
                        count = handle.readinto(buffer)
                        if not count:
                            break
                        total += count
        return PagePrefetchResult(start, len(paths), total, time.perf_counter() - started)

    def start(self, start: int, size: int = 1) -> None:
        if not self.enabled or size < 1 or not 0 <= start < len(self.records):
            return
        stop = min(start + size, len(self.records))
        key = (start, stop - start)
        with self._lock:
            if self._closed:
                return
            if self._future is not None:
                if self._future_key == key:
                    return
                raise RuntimeError(
                    "H3 page prefetch must consume the current future before starting another."
                )
            selected = self.records[start:stop]
            paths = tuple(self.root / record.file for record in selected)
            self.requests += 1
            self.max_window_bytes = max(
                self.max_window_bytes, sum(record.tensor_bytes for record in selected)
            )
            self._future_key = key
            self._future = self._executor.submit(self._reader, start, paths)

    def wait(self, start: int, size: int = 1) -> bool:
        """Wait for a matching read-ahead operation; return false for serial fallback."""
        if not self.enabled:
            return False
        size = min(size, len(self.records) - start)
        key = (start, size)
        with self._lock:
            if self._future is None or self._future_key != key:
                return False
            future = self._future
            self._future = None
            self._future_key = None
        started = time.perf_counter()
        try:
            result = future.result()
        except Exception:
            with self._lock:
                self.failures += 1
                self.wait_seconds += time.perf_counter() - started
            return False
        waited = time.perf_counter() - started
        with self._lock:
            self.hits += 1
            self.bytes_read += result.bytes_read
            self.read_seconds += result.read_seconds
            self.wait_seconds += waited
        return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            future = self._future
            self._future = None
            self._future_key = None
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
                "prefetch_backend": self.backend if self.enabled else "disabled",
                "prefetch_depth": 1 if self.enabled else 0,
                "prefetch_requests": self.requests,
                "prefetch_hits": self.hits,
                "prefetch_failures": self.failures,
                "prefetch_bytes": self.bytes_read,
                "prefetch_read_seconds": self.read_seconds,
                "prefetch_wait_seconds": self.wait_seconds,
                "prefetch_hidden_seconds": max(self.read_seconds - self.wait_seconds, 0.0),
                "prefetch_buffer_bytes": (
                    PREFETCH_BUFFER_BYTES
                    if self.enabled and self.backend == "stream"
                    else 0
                ),
                "prefetch_max_window_bytes": self.max_window_bytes,
            }
