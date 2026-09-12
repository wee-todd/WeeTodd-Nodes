#!/usr/bin/env python3
"""Build the optional Draw Things helper and its corresponding-source distribution."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


def sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_filter(info: tarfile.TarInfo):
    if any(part in {".git", ".build", ".swiftpm", "__pycache__"} for part in Path(info.name).parts):
        return None
    if info.issym() and Path(info.linkname).is_absolute():
        raise ValueError("Source archives cannot contain machine-specific absolute symlinks")
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def package_helper(root: Path, scratch: Path, destination: Path) -> Path:
    package = root / "integrations/drawthings-client"
    binary = scratch / "release/WeeToddDrawThings"
    pins = json.loads((package / "Package.resolved").read_text())["pins"]
    for pin in pins:
        if not (scratch / "checkouts" / pin["identity"] / "Package.swift").is_file():
            raise ValueError(f"Missing corresponding dependency source: {pin['identity']}")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "DrawThings-Corresponding-Source.tar.gz"
    rebuild = '''#!/usr/bin/env python3
"""Rebuild against the supplied, editable dependency sources (requires Swift 6)."""
import json, subprocess
from pathlib import Path
root = Path(__file__).resolve().parent
package = root / "helper"
subprocess.run(["swift", "package", "--package-path", str(package), "resolve"], check=True)
for pin in json.loads((package / "Package.resolved").read_text())["pins"]:
    subprocess.run(["swift", "package", "--package-path", str(package), "edit", pin["identity"],
                    "--path", str(root / "dependencies" / pin["identity"])], check=True)
subprocess.run(["swift", "build", "--package-path", str(package), "-c", "release",
                "--product", "WeeToddDrawThings"], check=True)
print(package / ".build/release/WeeToddDrawThings")
'''
    instructions = """Draw Things helper corresponding source

This separately linked helper uses draw-things-community's _MediaGenerationKit
under GPL-3.0 and dependencies under their accompanying licenses. The newer H3
revision is not covered by the older public wrapper's LGPL distribution grant.
Every dependency's source and
license files are included under dependencies/. WeeTodd helper source is under
helper/ and the accompanying Apache-2.0 license. No model weights are included.

To modify and relink: edit the supplied helper/dependency sources, then run
python3 rebuild.py on Apple Silicon with Swift 6 / Xcode command-line tools.
The initial SwiftPM resolve may access the network; subsequent package edits
point the build at the supplied local sources. Package.resolved records exact
upstream revisions. No account or model download is needed for this build.

Choose the rebuilt executable using Studio > Movie > Draw Things Connections >
Transport helper > Import. This supports replacement without modifying Studio.
Alternatively replace Contents/MacOS/WeeToddDrawThings in a writable app copy
and ad-hoc sign the copy with codesign --force --deep --sign - 'WeeTodd Studio.app'.
The application does not prohibit modification or reverse engineering of this
helper for debugging changes to covered libraries.

This package supports local builds and corresponding-source inspection. Before
publishing a Studio bundle containing this helper, resolve the GPL distribution
requirements or obtain an applicable upstream alternative license. Do not label
the combined helper binary Apache-2.0 or LGPL-3.0.
"""
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(package, arcname="helper", filter=source_filter)
        tar.add(root / "LICENSE", arcname="helper/LICENSE", filter=source_filter)
        for pin in pins:
            tar.add(scratch / "checkouts" / pin["identity"],
                    arcname="dependencies/" + pin["identity"], filter=source_filter)
        for name, value in (("rebuild.py", rebuild), ("README.txt", instructions)):
            data = value.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    shutil.copy2(binary, destination / binary.name)
    (destination / "DrawThings-Notices.txt").write_text(instructions)
    metadata = {"format": "weetodd-drawthings-distribution-v1", "helperSHA256": sha256(binary),
                "sourceSHA256": sha256(archive), "dependencies": pins}
    (destination / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return destination


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-path", type=Path)
    parser.add_argument("--output", type=Path, default=root / "studio/.build/drawthings")
    args = parser.parse_args()
    scratch = (args.scratch_path or root / "integrations/drawthings-client/.build").resolve()
    subprocess.run(
        ["swift", "build", "--package-path", str(root / "integrations/drawthings-client"),
         "--scratch-path", str(scratch), "-c", "release", "--product", "WeeToddDrawThings"],
        check=True,
    )
    print(package_helper(root, scratch, args.output.resolve()))


if __name__ == "__main__":
    main()
