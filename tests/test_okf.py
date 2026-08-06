from pathlib import Path

from scripts.validate_okf import validate_bundle


def test_project_knowledge_bundle_is_valid():
    assert validate_bundle(Path("knowledge")) == []


def test_validator_rejects_concept_without_type(tmp_path: Path):
    (tmp_path / "index.md").write_text("# Concepts\n")
    (tmp_path / "concept.md").write_text("---\ntitle: Missing type\n---\n\n# Body\n")

    errors = validate_bundle(tmp_path)

    assert any("non-empty type" in error for error in errors)


def test_validator_rejects_malformed_reserved_files(tmp_path: Path):
    (tmp_path / "index.md").write_text("not an index\n")
    (tmp_path / "log.md").write_text("# Log\n\n## tomorrow\n")

    errors = validate_bundle(tmp_path)

    assert any("index body" in error for error in errors)
    assert any("ISO date" in error for error in errors)
