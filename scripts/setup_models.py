#!/usr/bin/env python3
"""Prepare shared Studio/headless model profiles without editing recipe JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from wee_todd_mlx.model_setup import prepare_recipe, scan_models, setup_catalog

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog")
    scan = commands.add_parser("scan")
    scan.add_argument("preset")
    scan.add_argument("roots", nargs="+")
    create = commands.add_parser("create")
    create.add_argument("preset")
    create.add_argument("--component", action="append", default=[], metavar="KEY=PATH")
    create.add_argument("--profiles-directory", required=True)
    create.add_argument(
        "--memory-mode", choices=["automatic", "lower_memory", "custom"], default="automatic"
    )
    create.add_argument("--memory-gb", type=float)
    commands.add_parser("downloads")
    download = commands.add_parser("download")
    download.add_argument("download_id")
    download.add_argument("--destination", required=True)
    download.add_argument("--existing-root", action="append", default=[])
    args = parser.parse_args()
    if args.command == "catalog":
        result = {"presets": setup_catalog()}
    elif args.command == "scan":
        result = scan_models(args.preset, args.roots)
    elif args.command == "create":
        components = {}
        for item in args.component:
            key, separator, value = item.partition("=")
            if not separator or not key or key in components:
                parser.error("Specify each component once as --component KEY=PATH")
            components[key] = value
        result = prepare_recipe(
            args.preset, components, args.profiles_directory, args.memory_mode, args.memory_gb
        )
    elif args.command == "downloads":
        from wee_todd_mlx.model_downloads import download_catalog

        result = {"downloads": download_catalog()}
    else:
        from wee_todd_mlx.model_downloads import prepare_download

        result = prepare_download(
            args.download_id,
            args.destination,
            progress=lambda message, fraction: print(message, file=sys.stderr),
            existing_roots=args.existing_root,
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
