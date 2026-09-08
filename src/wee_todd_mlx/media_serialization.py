"""Bounded, copy-free serialization helpers for evaluated host buffers."""

from __future__ import annotations

from typing import Any


def write_all_contiguous(buffer: Any, stream: Any) -> None:
    """Write one C-contiguous buffer completely without materializing ``bytes``."""
    view = memoryview(buffer).cast("B")
    try:
        while view:
            written = stream.write(view)
            if written is None or written <= 0:
                raise OSError(
                    f"stream.write() reported {written!r} bytes written; "
                    f"cannot make progress on {len(view)} remaining bytes"
                )
            view = view[written:]
    finally:
        view.release()


__all__ = ["write_all_contiguous"]
