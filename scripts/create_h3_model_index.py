#!/usr/bin/env python3
"""Create a WeeTodd-compatible MiniMax H3 model manifest without downloading weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIRECTORY = REPOSITORY_ROOT / "assets"
SUPPORTED_PARTITIONS = ("fl2va", "ref2va")


def create_model_index(
    checkpoint_root: str | Path,
    partition: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Copy a checked-in manifest template into one local checkpoint partition."""

    normalized_partition = partition.casefold()
    if normalized_partition not in SUPPORTED_PARTITIONS:
        raise ValueError(
            f"Partition must be one of {', '.join(SUPPORTED_PARTITIONS)}, "
            f"got {partition!r}."
        )
    root = Path(checkpoint_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")
    target = root / "model_index.json"
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace existing manifest: {target}. Use --force only after "
            "confirming the selected partition."
        )
    template = TEMPLATE_DIRECTORY / f"model_index.{normalized_partition}.json"
    manifest = json.loads(template.read_text(encoding="utf-8"))
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_root",
        help="existing FL2VA or Ref2VA directory that will receive model_index.json",
    )
    parser.add_argument("--partition", required=True, choices=SUPPORTED_PARTITIONS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing model_index.json after the partition was verified",
    )
    args = parser.parse_args(argv)
    target = create_model_index(args.checkpoint_root, args.partition, overwrite=args.force)
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
