import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_model_library import tensor_file
from wee_todd_mlx.asset_registry import (
    AssetRegistry,
    describe_asset,
    import_recipe,
    resolve_recipe,
)


def test_registry_persists_physical_aliases_without_copying(tmp_path):
    weight = tensor_file(tmp_path / "original.safetensors")
    alias = tmp_path / "alias.safetensors"
    alias.symlink_to(weight)
    hardlink = tmp_path / "hardlink.safetensors"
    hardlink.hardlink_to(weight)
    registry = AssetRegistry(tmp_path / "registry.json")
    ref = registry.register(weight)
    assert registry.register(alias)["asset_id"] == ref["asset_id"]
    assert registry.register(hardlink)["asset_id"] == ref["asset_id"]
    registry.save()
    loaded = AssetRegistry(registry.path)
    assert len(loaded.data["assets"]) == 1
    assert loaded.resolve(ref)[0] == str(weight)
    assert not loaded.resolve(ref)[1]["payload_hash_verified"]
    assert weight.read_bytes() == alias.read_bytes()


def test_rename_and_restart_keep_asset_reference(tmp_path):
    weight = tensor_file(tmp_path / "first.safetensors")
    registry = AssetRegistry(tmp_path / "registry.json")
    ref = registry.register(weight)
    registry.save()
    renamed = tmp_path / "community-download.safetensors"
    weight.rename(renamed)
    loaded = AssetRegistry(registry.path)
    moved = loaded.register(renamed)
    assert moved["asset_id"] == ref["asset_id"]
    assert moved["revision"] == ref["revision"]
    loaded.save()
    assert AssetRegistry(registry.path).resolve(ref)[0] == str(renamed)


def test_equal_layout_is_not_content_identity_or_compatibility(tmp_path):
    one = tensor_file(tmp_path / "one.safetensors")
    two = tensor_file(tmp_path / "two.safetensors", value=b"5678")
    registry = AssetRegistry(tmp_path / "registry.json")
    assert registry.register(one)["asset_id"] != registry.register(two)["asset_id"]
    first, second = describe_asset(one), describe_asset(two)
    assert first["structural_sha256"] == second["structural_sha256"]
    assert first["compatibility"] == "not_evaluated"


def test_payload_change_invalidates_recipe_even_with_identical_header(tmp_path):
    weight = tensor_file(tmp_path / "weight.safetensors")
    registry = AssetRegistry(tmp_path / "registry.json")
    ref = registry.register(weight)
    tensor_file(weight, value=b"5678")
    with pytest.raises(ValueError, match="changed since import"):
        registry.resolve(ref)
    registry.register(weight)
    with pytest.raises(ValueError, match="revision changed"):
        registry.resolve(ref)


def test_directory_rename_cycles_and_missing_members(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    weight = tensor_file(root / "weights.safetensors")
    (root / "config.json").write_text('{"model_type":"fixture"}')
    (root / "cycle").symlink_to(root, target_is_directory=True)
    registry = AssetRegistry(tmp_path / "registry.json")
    ref = registry.register(root)
    assert len(describe_asset(root)["members"]) == 2
    renamed = tmp_path / "renamed"
    root.rename(renamed)
    # Remove the now-broken cycle alias; it never contributed a data member.
    (renamed / "cycle").unlink()
    moved = registry.register(renamed)
    assert moved["asset_id"] == ref["asset_id"]
    assert moved["revision"] == ref["revision"]
    (renamed / weight.name).unlink()
    with pytest.raises(ValueError, match="changed"):
        registry.resolve(ref)


def test_writer_conflict_preserves_registry_and_releases_lock(tmp_path):
    path = tmp_path / "registry.json"
    first, stale = AssetRegistry(path), AssetRegistry(path)
    first.register(tensor_file(tmp_path / "a.safetensors"))
    first.save()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="concurrently"):
        stale.save()
    assert path.read_bytes() == before
    assert not path.with_name(path.name + ".lock").exists()


def test_existing_writer_lock_is_not_removed(tmp_path):
    registry = AssetRegistry(tmp_path / "registry.json")
    lock = tmp_path / "registry.json.lock"
    lock.write_text("other writer")
    with pytest.raises(FileExistsError):
        registry.save()
    assert lock.read_text() == "other writer"


def test_registry_cannot_make_its_own_component_snapshot_stale(tmp_path):
    tensor_file(tmp_path / "weights.safetensors")
    with pytest.raises(ValueError, match="outside selected component"):
        AssetRegistry(tmp_path / "library.json").register(tmp_path)


@pytest.mark.parametrize("suffix", [".pt", ".pth", ".ckpt", ".gguf"])
def test_no_untrusted_execution_or_unsupported_format_assumption(tmp_path, suffix):
    weight = tmp_path / ("untrusted" + suffix)
    weight.write_bytes(b"untrusted data")
    with pytest.raises(ValueError, match="Unsupported asset format"):
        describe_asset(weight)


def test_binding_roundtrip_preserves_prompt_settings_and_stack(tmp_path):
    weight = tensor_file(tmp_path / "community.safetensors")
    recipe = {
        "engine": "ltx25",
        "components": {"transformer_path": str(weight), "loras": [[str(weight), 0.75]]},
        "config": {"seed": 13},
        "prompt": "Keep /some/path and filenames unchanged",
    }
    registry = AssetRegistry(tmp_path / "registry.json")
    bound = import_recipe(recipe, registry)
    assert isinstance(bound["components"]["transformer_path"], dict)
    assert bound["components"]["loras"][0][1] == 0.75
    resolved, report = resolve_recipe(bound, registry)
    assert recipe == resolved
    assert len(registry.data["assets"]) == 1
    assert report["copied_weight_bytes"] == report["downloaded_bytes"] == 0
    with pytest.raises(ValueError, match="model-library"):
        resolve_recipe(bound)


def test_h3_manifest_root_does_not_index_unselected_weights(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "model_index.json").write_text("{}")
    (root / "broken.safetensors").write_bytes(b"not selected")
    assert len(describe_asset(root, manifest_only=True)["members"]) == 1


def test_missing_numbered_shard_fails_before_runtime(tmp_path):
    tensor_file(tmp_path / "model-00001-of-00002.safetensors")
    with pytest.raises(ValueError, match="Incomplete sharded"):
        describe_asset(tmp_path)


def test_stale_index_reported_for_engine_specific_validation(tmp_path):
    tensor_file(tmp_path / "part-one.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "missing.safetensors"}})
    )
    descriptor = describe_asset(tmp_path)
    index = next(m for m in descriptor["members"] if m["name"].endswith("index.json"))
    assert index["unresolved_index_files"] == ["missing.safetensors"]
    assert descriptor["compatibility"] == "not_evaluated"


def test_registry_import_does_not_import_weighted_engines():
    root = Path(__file__).parents[1]
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import wee_todd_mlx.asset_registry; "
            "import wee_todd_mlx.adapter_contract; "
            "assert not any(n.split('.')[0] in {'mlx','torch','comfy','wee_todd_nodes'} "
            "for n in sys.modules)",
        ],
        cwd="/",
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr


def test_import_freezes_source_independent_h3_default_before_model_is_renamed(tmp_path):
    from tests.test_adapter_contract import adapter_file

    root = tmp_path / "bundle"
    root.mkdir()
    (root / "model_index.json").write_text("{}")
    weight = tensor_file(tmp_path / "weight.safetensors")
    adapter = adapter_file(
        tmp_path / "turbo.safetensors",
        {"block.lora_A.weight": [2, 4], "block.lora_B.weight": [6, 2]},
    )
    recipe = {
        "engine": "h3",
        "components": {
            "checkpoint": str(root),
            **{
                role: str(weight)
                for role in (
                    "transformer",
                    "text_encoder",
                    "processor",
                    "tokenizer",
                    "video_vae",
                    "audio_vae",
                )
            },
        },
        "loras": {"adapters": [{"path": str(adapter)}]},
    }
    registry = AssetRegistry(tmp_path / "library.json")
    bound = import_recipe(recipe, registry)
    selected = bound["loras"]["adapters"][0]
    assert selected["profile"] == "standard"
    assert selected["qkv_layout"] == "native_interleaved"
    moved = tmp_path / "renamed.safetensors"
    adapter.rename(moved)
    registry.register(moved)
    resolved, _ = resolve_recipe(bound, registry)
    assert resolved["loras"]["adapters"][0] == {
        "path": str(moved),
        "profile": "standard",
        "qkv_layout": "native_interleaved",
    }


def test_runner_preflight_only_uses_resolution_and_never_renders(tmp_path, monkeypatch):
    from tests.test_headless_candidates import load_script
    from wee_todd_mlx import headless_preflight

    runner = load_script("render_headless", monkeypatch)
    monkeypatch.setattr(runner, "assert_isolated", lambda: {"fixture": True})
    weight = tensor_file(tmp_path / "weight.safetensors")
    original = {
        "format": "weetodd-headless-v2",
        "engine": "ltx25",
        "candidate": "fixture",
        "components": {"transformer_path": str(weight)},
        "config": {},
        "prompt": "hi",
    }
    registry = AssetRegistry(tmp_path / "registry.json")
    bound = import_recipe(original, registry)
    registry.save()
    recipe_file = tmp_path / "recipe.json"
    recipe_file.write_text(json.dumps(bound))
    checked = []
    monkeypatch.setattr(
        headless_preflight,
        "preflight_recipe",
        lambda r: (
            checked.append(r)
            or {
                "status": "preflight_passed",
                "conditioning": {"contract": {"version": 1, "task": "t2v", "inputs": []}},
            }
        ),
    )
    monkeypatch.setattr(runner, "render_ltx", lambda *a: pytest.fail("Must not render"))
    output = tmp_path / "check"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render",
            "--recipe",
            str(recipe_file),
            "--output-directory",
            str(output),
            "--preflight-only",
            "--model-library",
            str(registry.path),
        ],
    )
    original_meta = list(sys.meta_path)
    try:
        runner.main()
    finally:
        sys.meta_path[:] = original_meta
    assert checked == [original]
    assert json.loads((output / "resolved-recipe.json").read_text()) == original
    assert json.loads((output / "result.json").read_text())["status"] == "preflight_passed"
    assert not (output / "render.mp4").exists()
