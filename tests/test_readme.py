import re
import tomllib
from pathlib import Path

from wee_todd_nodes.nodes import NODE_CLASS_MAPPINGS

ROOT = Path(__file__).parents[1]
README = (ROOT / "README.md").read_text()
LOCAL_LINK = re.compile(r"\[[^\]]+\]\((?!https?://|mailto:|#)([^)]+)\)")


def test_readme_local_links_resolve():
    missing = []
    for target in LOCAL_LINK.findall(README):
        clean_target = target.split("#", 1)[0].strip("<>")
        if clean_target and not (ROOT / clean_target).exists():
            missing.append(clean_target)
    assert not missing


def test_readme_catalogs_every_shipped_ui_workflow():
    workflows = sorted((ROOT / "workflows").glob("*.json"))
    workflows += sorted((ROOT / "examples").glob("*_workflow.json"))
    missing = [path.name for path in workflows if path.name not in README]
    assert not missing


def test_readme_h3_node_count_matches_registration():
    match = re.search(r"- (\d+) composable nodes under `WeeTodd/H3`", README)
    assert match is not None
    registered = sum(name.startswith("WeeToddH3") for name in NODE_CLASS_MAPPINGS)
    assert int(match.group(1)) == registered


def test_readme_runtime_requirements_match_project_metadata():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    python_minimum = metadata["requires-python"].removeprefix(">=")
    mlx_requirement = next(item for item in metadata["dependencies"] if item.startswith("mlx>="))
    mlx_minimum = mlx_requirement.removeprefix("mlx>=")
    assert f"Python {python_minimum} or later" in README
    assert f"MLX {mlx_minimum} or later" in README


def test_readme_contains_current_portable_h3_component_paths():
    expected = {
        "MiniMax-H3/FL2VA",
        "MiniMax-H3/transformers/q8_extended_paged",
        "MiniMax-H3/text_encoders/q8-paged",
        "MiniMax-H3/FL2VA/processor",
        "MiniMax-H3/FL2VA/tokenizer",
        "MiniMax-H3/vae/q8/video_vae_affine_q8.safetensors",
        "MiniMax-H3/FL2VA/audio_vae",
    }
    assert all(f"`{path}`" in README for path in expected)


def test_readme_excludes_private_paths():
    assert "/Users/" not in README
    assert "/Volumes/" not in README
    assert "file://" not in README


def test_readme_documents_the_commit_validation_surface():
    assert "tests/test_readme.py" in README
    assert "tests/test_workflows.py" in README
    assert "scripts/lint_docs.py" in README
