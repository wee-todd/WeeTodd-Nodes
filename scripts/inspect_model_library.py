"""Inventory existing model folders without copying, deleting, or downloading weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-duplicates", action="store_true", help="Read full same-size files")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Inventory output exists; select a new filename")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from wee_todd_mlx.model_library import scan_model_library

    report = scan_model_library(args.root, hash_duplicates=args.hash_duplicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "files": report["distinct_physical_files"],
                "directory_aliases": len(report["directory_aliases"]),
                "errors": len(report["errors"]),
                "header_errors": sum(a.get("header_valid") is False for a in report["assets"]),
                "verified_duplicate_groups": len(report["verified_content_duplicates"]),
            }
        )
    )


if __name__ == "__main__":
    main()
