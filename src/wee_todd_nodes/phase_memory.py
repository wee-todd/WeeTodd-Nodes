"""Synchronized MLX phase peaks, isolated from earlier jobs in the same process."""

from __future__ import annotations

import copy
import inspect
import logging
import time
import uuid
from dataclasses import replace
from functools import wraps
from threading import RLock

_LOCK = RLock()  # MLX's peak counter is process-global.


def current_prompt_id():
    try:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
        return context.prompt_id if context is not None else None
    except ImportError:
        return None


def measured_phase(name):
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def measured(*args, **kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            upstream = bound.get("latents", bound.get("conditioning"))
            report = getattr(upstream, "phase_memory", None)
            prompt_id = current_prompt_id()
            if report and prompt_id and report.get("prompt_id") != prompt_id:
                # Comfy may reuse conditioning/latents from a previous cached graph.
                # Its historical encoder/sampler peak must not count in this job.
                if name == "transformer":
                    report = None
                else:
                    report.clear()
                    report.update(phases=[], run_peak_bytes=0, run_id=uuid.uuid4().hex)
            if name in {"text_encoder", "transformer"}:
                report = (
                    copy.deepcopy(report)
                    if report
                    else {
                        "phases": [],
                        "run_peak_bytes": 0,
                        "scope": "MLX instrumented H3 phases; not total system RAM",
                    }
                )
                report["run_id"] = uuid.uuid4().hex
            elif report is None:
                report = {"phases": [], "run_peak_bytes": 0, "run_id": uuid.uuid4().hex}
            report["prompt_id"] = prompt_id
            report["aggregation"] = (
                "current_comfy_prompt" if prompt_id else "explicit_phase_lineage"
            )
            try:
                import mlx.core as mx
            except ImportError:
                return function(*args, **kwargs)
            with _LOCK:
                mx.synchronize()
                mx.reset_peak_memory()
                started = time.perf_counter()
                status = "failed"
                try:
                    result = function(*args, **kwargs)
                    status = "success"
                finally:
                    mx.synchronize()
                    peak = int(mx.get_peak_memory())
                    report["phases"].append(
                        {
                            "phase": name,
                            "peak_bytes": peak,
                            "active_end_bytes": int(mx.get_active_memory()),
                            "seconds": time.perf_counter() - started,
                            "status": status,
                        }
                    )
                    report["run_peak_bytes"] = max(report["run_peak_bytes"], peak)
                    if status != "success":
                        logging.getLogger(__name__).warning("H3 phase failed: %s", report)
                if hasattr(result, "phase_memory"):
                    result = replace(result, phase_memory=report)
                return result

        return measured

    return decorate
