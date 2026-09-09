"""Portable Draw Things headless-job validation helpers."""

from __future__ import annotations

import copy
from typing import Any

from .contracts import validate_request


def validate_job_dependencies(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate remote jobs and return a stable dependency-first ordering."""
    if not isinstance(jobs, list):
        raise ValueError("remoteJobs must be an array")
    indexed: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, raw in enumerate(jobs):
        if not isinstance(raw, dict):
            raise ValueError("each remote job must be an object")
        job = copy.deepcopy(raw)
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("each remote job requires a non-empty id")
        if job_id in indexed:
            raise ValueError(f"duplicate remote job id {job_id}")
        dependencies = job.get("dependsOn", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(value, str) or not value for value in dependencies
        ):
            raise ValueError(f"remote job {job_id} dependsOn must contain IDs")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError(f"remote job {job_id} contains duplicate dependencies")
        job["dependsOn"] = dependencies
        if "request" in job:
            job["request"] = validate_request(job["request"])
        indexed[job_id] = (index, job)
    for _, job in indexed.values():
        missing = [value for value in job["dependsOn"] if value not in indexed]
        if missing:
            raise ValueError(f"remote job {job['id']} has missing dependency {missing[0]}")

    remaining = {key: set(value[1]["dependsOn"]) for key, value in indexed.items()}
    ordered: list[dict[str, Any]] = []
    while remaining:
        ready = sorted(
            (key for key, dependencies in remaining.items() if not dependencies),
            key=lambda key: indexed[key][0],
        )
        if not ready:
            raise ValueError("remote job dependency cycle detected")
        for key in ready:
            ordered.append(indexed[key][1])
            del remaining[key]
        for dependencies in remaining.values():
            dependencies.difference_update(ready)
    return ordered
