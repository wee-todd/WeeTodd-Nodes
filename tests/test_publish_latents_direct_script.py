import importlib.util
from pathlib import Path


def test_direct_publish_script_imports_without_running():
    path = Path("scripts/publish_latents_direct.py")
    spec = importlib.util.spec_from_file_location("publish_latents_direct_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert callable(module.main)
