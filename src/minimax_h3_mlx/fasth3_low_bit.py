"""Hardware gate for M5/NAX low-bit FastH3 core projections.

The gate intentionally does not infer support from macOS alone.  Metal 4 headers compile on older
Apple GPUs, but Apple's neural-accelerator tensor path is an M5 feature.  Callers must also supply
an actually quantized checkpoint; this module never silently quantizes BF16 weights at runtime.
"""

from __future__ import annotations

import platform
import re

import mlx.core as mx


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", value)
    return tuple(int(number) for number in numbers[:3])


def low_bit_tensorops_capability(
    device_info: dict[str, object] | None = None,
    *,
    system_name: str | None = None,
    macos_release: str | None = None,
    mlx_version: str | None = None,
) -> tuple[bool, str | None]:
    """Return whether MLX may dispatch quantized projections to the Apple M5 NAX path."""

    system = platform.system() if system_name is None else system_name
    if system != "Darwin":
        return False, "M5 low-bit tensor operations require macOS"
    release = platform.mac_ver()[0] if macos_release is None else macos_release
    if _version_tuple(release) < (26, 2):
        return False, "M5 low-bit tensor operations require macOS 26.2 or newer"
    version = getattr(mx, "__version__", "0") if mlx_version is None else mlx_version
    if _version_tuple(version) < (0, 30, 0):
        return False, "M5 neural-accelerator dispatch requires MLX 0.30.0 or newer"
    info = mx.device_info() if device_info is None else device_info
    name = str(info.get("device_name", ""))
    architecture = str(info.get("architecture", "")).lower()
    if not name.startswith("Apple M5") or re.fullmatch(r"applegpu_g17[sd]?", architecture) is None:
        return False, (
            "M5 low-bit tensor operations require an Apple M5-family g17 GPU; found "
            f"{name or 'unknown device'} ({architecture or 'unknown architecture'})"
        )
    if not hasattr(mx, "quantized_matmul"):
        return False, "the installed MLX build has no quantized_matmul operator"
    return True, None


__all__ = ["low_bit_tensorops_capability"]
