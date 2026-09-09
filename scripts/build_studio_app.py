#!/usr/bin/env python3
"""Build an ad-hoc-signed local Swift app. Runtime paths live in the generated bundle only."""

from __future__ import annotations

import argparse
import plistlib
import shutil
import subprocess
import tempfile
from pathlib import Path


def build_bundle(root: Path, configuration: str, app: Path) -> None:
    """Assemble a fresh bundle without requiring ignored agent configuration."""
    binaries = root / "studio" / ".build" / configuration
    macos = app / "Contents" / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    for name in ("WeeToddStudio", "StudioMetal", "WeeToddCLI"):
        shutil.copy2(binaries / name, macos / name)
    resources = app / "Contents/Resources"
    source = resources / "RendererSource"
    source.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "studio/Resources/AppIcon.icns", resources / "AppIcon.icns")
    for name in ("src", "scripts"):
        shutil.copytree(
            root / name,
            source / name,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"),
        )
    for name in ("pyproject.toml", "LICENSE", "README.md"):
        shutil.copy2(root / name, source / name)
    for relative in ("studio/runtime",):
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
        "CFBundleIconFile": "AppIcon.icns",
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
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)


def package_app(root: Path, configuration: str) -> Path:
    build = root / "studio" / ".build"
    app = build / "WeeTodd Studio.app"
    # Finish and verify packaging before replacing a previously working build.
    with tempfile.TemporaryDirectory(prefix=".studio-package-", dir=build) as temporary:
        staging = Path(temporary) / app.name
        build_bundle(root, configuration, staging)
        previous = Path(temporary) / "Previous.app"
        if app.exists():
            app.rename(previous)
        try:
            staging.rename(app)
        except OSError:
            if previous.exists():
                previous.rename(app)
            raise
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=("debug", "release"), default="debug")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["swift", "build", "--package-path", str(root / "studio"), "-c", args.configuration],
        check=True,
    )
    print(package_app(root, args.configuration))


if __name__ == "__main__":
    main()
