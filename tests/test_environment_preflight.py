import json
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1]
    / ".agents"
    / "skills"
    / "python-environment-preflight"
    / "scripts"
    / "preflight.py"
)


def _project(tmp_path: Path, requires_python: str) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        "name = \"environment-test\"\n"
        "version = \"0.0.0\"\n"
        f"requires-python = \"{requires_python}\"\n"
    )
    return tmp_path


def test_environment_preflight_accepts_compatible_interpreter(tmp_path):
    project = _project(tmp_path, ">=3.8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--project", str(project), "--python", sys.executable],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["compatible"] is True
    assert report["candidate"]["executable"] == sys.executable


def test_environment_preflight_rejects_incompatible_interpreter(tmp_path):
    project = _project(tmp_path, ">=99.0")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--project", str(project), "--python", sys.executable],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "does not satisfy >=99.0" in result.stderr
    assert json.loads(result.stdout)["compatible"] is False


def test_environment_preflight_rejects_wrong_architecture(tmp_path):
    project = _project(tmp_path, ">=3.8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--project",
            str(project),
            "--python",
            sys.executable,
            "--require-architecture",
            "not-a-real-architecture",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "does not satisfy not-a-real-architecture" in result.stderr
