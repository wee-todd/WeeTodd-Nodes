#!/usr/bin/env python3
"""Create a bounded, restorable WeeTodd source snapshot.

The snapshot contains Git-tracked files, visible untracked source files, and a
binary Git patch. Ignored local state is never traversed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
from pathlib import Path

DEFAULT_MAX_FILE_MIB = 256
DEFAULT_MAX_TOTAL_MIB = 1024


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        capture_output=True,
    )


def _protected_prefixes(repo: Path) -> tuple[str, ...]:
    path = repo / ".source-backupignore"
    prefixes = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip().lstrip("/")
        if value and not value.startswith("#"):
            prefixes.append(value.rstrip("/") + "/")
    return tuple(prefixes)


def _is_protected(relative: str, prefixes: tuple[str, ...]) -> bool:
    normalized = relative.replace(os.sep, "/").lstrip("./")
    return any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in prefixes
    )


def _source_files(repo: Path, prefixes: tuple[str, ...]) -> list[Path]:
    result = _git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    relatives = [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    files = []
    for relative in relatives:
        if _is_protected(relative, prefixes):
            continue
        path = repo / relative
        if path.is_file() or path.is_symlink():
            files.append(Path(relative))
    return sorted(files)


def _size(path: Path) -> int:
    return path.lstat().st_size if path.is_symlink() else path.stat().st_size


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        target.symlink_to(os.readlink(source))
    else:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a source-only WeeTodd backup without traversing ignored local state."
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--name", default="weetodd-source")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-file-mib", type=int, default=DEFAULT_MAX_FILE_MIB)
    parser.add_argument("--max-total-mib", type=int, default=DEFAULT_MAX_TOTAL_MIB)
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"Not a Git checkout: {repo}")
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else repo.parent / f"{repo.name}-source-backups"
    )
    prefixes = _protected_prefixes(repo)
    files = _source_files(repo, prefixes)
    sizes = {str(relative): _size(repo / relative) for relative in files}
    total_bytes = sum(sizes.values())
    max_file_bytes = args.max_file_mib * 1024 * 1024
    max_total_bytes = args.max_total_mib * 1024 * 1024
    oversized = [name for name, size in sizes.items() if size > max_file_bytes]
    if not args.allow_large and (oversized or total_bytes > max_total_bytes):
        details = {
            "error": "source snapshot exceeds its safety ceiling",
            "total_bytes": total_bytes,
            "max_total_bytes": max_total_bytes,
            "oversized_files": oversized,
            "max_file_bytes": max_file_bytes,
        }
        raise SystemExit(json.dumps(details, indent=2))

    head = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    branch = _git(repo, "branch", "--show-current").stdout.decode().strip()
    status = _git(repo, "status", "--short").stdout.decode()
    patch = _git(repo, "diff", "--binary", "HEAD").stdout
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = output_root / f"{args.name}-{timestamp}"
    summary = {
        "format": "weetodd-source-backup-v1",
        "repo": str(repo),
        "backup_path": str(target),
        "base_commit": head,
        "branch": branch,
        "file_count": len(files),
        "source_bytes": total_bytes,
        "protected_prefixes": list(prefixes),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    target.mkdir(parents=True, exist_ok=False)
    snapshot = target / repo.name
    for relative in files:
        _copy_file(repo / relative, snapshot / relative)
    (target / "working-tree.patch").write_bytes(patch)
    (target / "status-short.txt").write_text(status, encoding="utf-8")
    (target / "MANIFEST.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    restore = f"""# Restore WeeTodd source snapshot

Start from a clean checkout at `{head}`, apply `working-tree.patch`, then copy the
contents of `{snapshot}` over the checkout. The snapshot intentionally excludes
models, renders, caches, benchmark captures, private knowledge, and nested backups.
"""
    (target / "RESTORE.md").write_text(restore, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
