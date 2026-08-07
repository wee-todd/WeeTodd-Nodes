"""Append-only experiment history for algorithm-search candidates."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import CandidateResult


class ExperimentStore:
    """Persist candidate results as portable JSON Lines without a database dependency."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, result: CandidateResult, *, context: dict[str, Any] | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = result.to_dict()
        experiment_context = dict(context or {})
        experiment_context.setdefault("recorded_at_utc", datetime.now(UTC).isoformat())
        if "git_commit" not in experiment_context:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            experiment_context["git_commit"] = (
                completed.stdout.strip() if completed.returncode == 0 else "unknown"
            )
        record["context"] = experiment_context
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
