import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "create_source_backup.py"


def _run(*args, check=True):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        check=check,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Backup Test"], cwd=root, check=True)
    (root / ".gitignore").write_text("backups/\noutput/\n")
    (root / ".source-backupignore").write_text("backups/\noutput/\n")
    (root / "source.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    return root


def test_source_backup_never_traverses_ignored_local_state(repo, tmp_path):
    (repo / "source.py").write_text("value = 2\n")
    (repo / "new_test.py").write_text("assert True\n")
    (repo / "backups").mkdir()
    (repo / "backups" / "huge.bin").write_bytes(b"x" * 1024 * 1024)
    output = tmp_path / "snapshots"

    result = _run("--repo", repo, "--output-root", output, "--name", "safe")
    manifest = json.loads(result.stdout)
    snapshot = Path(manifest["backup_path"]) / repo.name

    assert (snapshot / "source.py").read_text() == "value = 2\n"
    assert (snapshot / "new_test.py").is_file()
    assert not (snapshot / "backups").exists()
    assert manifest["source_bytes"] < 1024 * 1024


def test_source_backup_rejects_an_unignored_large_file(repo, tmp_path):
    (repo / "accidental.bin").write_bytes(b"x" * 2048)

    result = _run(
        "--repo",
        repo,
        "--output-root",
        tmp_path / "snapshots",
        "--max-file-mib",
        0,
        check=False,
    )

    assert result.returncode != 0
    assert "safety ceiling" in result.stderr
