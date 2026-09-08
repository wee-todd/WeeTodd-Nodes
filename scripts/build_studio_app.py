#!/usr/bin/env python3
"""Build an ad-hoc-signed local Swift app. Runtime paths live in the generated bundle only."""

from __future__ import annotations

import argparse
import plistlib
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=("debug", "release"), default="debug")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    package = root / "studio"
    subprocess.run(
        ["swift", "build", "--package-path", str(package), "-c", args.configuration], check=True
    )
    binaries = package / ".build" / args.configuration
    app = package / ".build" / "WeeTodd Studio.app"
    macos = app / "Contents" / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    for name in ("WeeToddStudio", "StudioMetal", "WeeToddCLI"):
        shutil.copy2(binaries / name, macos / name)
    resources = app / "Contents/Resources"
    source = resources / "RendererSource"
    source.mkdir(parents=True, exist_ok=True)
    for name in ("src", "scripts"):
        shutil.copytree(
            root / name,
            source / name,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"),
        )
    for name in ("pyproject.toml", "LICENSE", "README.md"):
        shutil.copy2(root / name, source / name)
    for relative in ("studio/runtime", ".agents/skills/python-environment-preflight"):
        shutil.copytree(
            root / relative,
            source / relative,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"),
        )
    info = {
        "CFBundleExecutable": "WeeToddStudio",
        "CFBundleIdentifier": "studio.weetodd.mac",
        "CFBundleName": "WeeTodd Studio",
        "CFBundleDisplayName": "WeeTodd Studio",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "14.0",
        "NSHighResolutionCapable": True,
        "WeeToddRuntimeRoot": str(root),
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "WeeTodd Movie Project",
                "CFBundleTypeRole": "Editor",
                "CFBundleTypeExtensions": ["weetodd"],
            }
        ],
    }
    with (app / "Contents" / "Info.plist").open("wb") as stream:
        plistlib.dump(info, stream)
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=True)
    print(app)


if __name__ == "__main__":
    main()
