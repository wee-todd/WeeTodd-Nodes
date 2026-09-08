"""Shipping must work without private skills and preserve the last usable bundle."""

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
packager = importlib.import_module("build_studio_app")
preflight = importlib.import_module("preflight_python_environment")
installer = importlib.import_module("install_studio_runtime")


@pytest.fixture
def source(tmp_path):
    for name in ("src", "scripts", "studio/runtime", "studio/.build/release"):
        (tmp_path / name).mkdir(parents=True)
    for name in ("WeeToddStudio", "StudioMetal", "WeeToddCLI"):
        (tmp_path / "studio/.build/release" / name).write_text("binary fixture")
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.11"\n')
    (tmp_path / "LICENSE").write_text("license fixture")
    (tmp_path / "README.md").write_text("readme fixture")
    shutil.copy2(SCRIPTS / "preflight_python_environment.py", tmp_path / "scripts")
    (tmp_path / "studio/runtime/requirements.lock").write_text("lock fixture")
    return tmp_path


def test_clean_source_packaging_removes_stale_files(source, monkeypatch):
    monkeypatch.setattr(packager.subprocess, "run", lambda *a, **kw: None)
    app = packager.package_app(source, "release")
    (app / "obsolete.py").write_text("stale")
    result = packager.package_app(source, "release")
    assert not (result / "obsolete.py").exists()
    shipped = result / "Contents/Resources/RendererSource"
    assert not (shipped / ".agents").exists()
    probe = subprocess.Popen(
        [sys.executable, str(shipped / "scripts/preflight_python_environment.py"),
         "--project", str(shipped), "--python", sys.executable],
        stdout=subprocess.PIPE, text=True,
    )
    stdout, _ = probe.communicate()
    assert probe.returncode == 0
    assert json.loads(stdout)["compatible"]


def test_failed_signature_preserves_previous_bundle(source, monkeypatch):
    app = source / "studio/.build/WeeTodd Studio.app"
    app.mkdir()
    (app / "working").write_text("previous build")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(packager.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        packager.package_app(source, "release")
    assert (app / "working").read_text() == "previous build"
    assert not list(app.parent.glob(".studio-package-*"))


def test_installer_runs_shipped_preflight_before_creating_runtime(source, monkeypatch):
    monkeypatch.setattr(installer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(installer.platform, "machine", lambda: "arm64")
    destination = source / "private-runtime"

    def reject(command):
        assert command[1] == source / "scripts/preflight_python_environment.py"
        assert command[1].is_file()
        assert not destination.exists()
        raise ValueError("incompatible interpreter")

    monkeypatch.setattr(installer, "run", reject)
    with pytest.raises(ValueError, match="incompatible interpreter"):
        installer.install(source, destination, source / "uv")
    assert not destination.exists()


@pytest.mark.parametrize("constraint", [">=3.11,<3.13", ">3.11", "==3.12"])
def test_preflight_does_not_ignore_unhandled_constraints(constraint):
    with pytest.raises(ValueError, match="one requires-python lower bound"):
        preflight._minimum_version(constraint)
