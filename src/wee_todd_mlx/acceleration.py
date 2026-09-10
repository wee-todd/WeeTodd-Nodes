"""Hardware advisories for native H3; never imports weights or changes creative settings."""

from __future__ import annotations

import platform
import subprocess


def _sysctl(name: str) -> str:
    try:
        return subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL, timeout=2
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_hardware() -> dict:
    """Read hardware without initializing Metal or importing an inference runtime."""
    memory = _sysctl("hw.memsize") if platform.system() == "Darwin" else ""
    return {
        "chip": _sysctl("machdep.cpu.brand_string") or platform.machine(),
        "memoryBytes": int(memory) if memory.isdecimal() else 0,
        "osVersion": platform.mac_ver()[0],
    }


def resolve_h3_acceleration(preferences: dict | None = None, hardware: dict | None = None) -> dict:
    """Resolve saved overrides, retaining conservative policies until hardware is qualified.

    Automatic projections delegate capability and numerical checks to the renderer's existing
    backend selector. A RAM figure alone is insufficient to qualify full transformer residency.
    The caller applies this only to explicit new selections; legacy recipes stay unchanged.
    """
    preferences = preferences or {}
    unknown = set(preferences) - {"h3MemoryPolicy", "h3ProjectionBackend"}
    if unknown:
        raise ValueError(f"Unsupported acceleration settings: {', '.join(sorted(unknown))}.")
    policy = preferences.get("h3MemoryPolicy", "automatic")
    backend = preferences.get("h3ProjectionBackend", "auto")
    if policy not in {"automatic", "paged", "pagedNormal", "resident"}:
        raise ValueError("H3 memory preference must be automatic, paged, pagedNormal, or resident.")
    if backend not in {"auto", "mlx"}:
        raise ValueError("H3 projection preference must be auto or mlx.")
    hardware = dict(detect_hardware() if hardware is None else hardware)
    memory = hardware.get("memoryBytes", 0)
    if not isinstance(memory, (int, float)) or memory < 0:
        memory = 0
    gib = memory / 1024**3
    if policy == "automatic":
        resolved = "paged" if gib <= 64 else "recipe"
        explanation = (
            f"{gib:g} GiB detected. Automatic uses the lower-memory execution policy."
            if gib and gib <= 64 else
            "Memory could not be detected. Automatic uses the lower-memory execution policy."
            if not gib else
            f"{gib:g} GiB detected. Automatic retains the recipe's memory policy; "
            "full residency needs separate qualification."
        )
    else:
        resolved = policy
        explanation = (
            "Resident sampling retains all transformer blocks until sampling completes. "
            "This is not a memory-fit guarantee; use only with ample free unified memory."
            if policy == "resident" else
            "Checkpoint paging uses normal working buffers with staged unloading. "
            "The checkpoint pagination and supported cache budget remain unchanged."
            if policy == "pagedNormal" else
            "Lower-memory execution uses the checkpoint's paging layout and staged unloading. "
            "A resident checkpoint is not converted to pages."
        )
    explanation += (
        " Automatic projections use verified hardware support and fall back to standard MLX."
        if backend == "auto" else " Standard MLX projections are selected explicitly."
    )
    return {
        "memoryPolicy": resolved,
        "projectionBackend": backend,
        "hardware": hardware,
        "explanation": explanation,
    }


def effective_h3_acceleration_report(report: dict, memory_policy: str, backend: str) -> dict:
    """Describe the final clip policy after preset and per-clip overrides have resolved."""
    memory = report.get("hardware", {}).get("memoryBytes", 0)
    gib = memory / 1024**3 if isinstance(memory, (int, float)) and memory > 0 else 0
    context = f"{gib:g} GiB detected. " if gib else "Memory could not be detected. "
    policies = {
        "paged": "Lower-memory execution uses the checkpoint's paging layout and staged "
        "unloading. A resident checkpoint is not converted to pages.",
        "pagedNormal": "Checkpoint paging uses normal working buffers with staged unloading. "
        "The checkpoint pagination and supported cache budget remain unchanged.",
        "resident": "Resident sampling retains all transformer blocks until sampling completes. "
        "This is not a memory-fit guarantee; use only with ample free unified memory.",
        "recipe": "The clip retains the recipe's memory policy. "
        "Full residency needs separate qualification.",
    }
    projections = {
        "auto": " Automatic projections use verified hardware support "
        "and fall back to standard MLX.",
        "mlx": " Standard MLX projections are selected explicitly.",
    }
    if memory_policy not in policies or backend not in projections:
        raise ValueError("Unsupported effective H3 acceleration policy.")
    return dict(
        report, memoryPolicy=memory_policy, projectionBackend=backend,
        explanation=context + policies[memory_policy] + projections[backend],
    )
