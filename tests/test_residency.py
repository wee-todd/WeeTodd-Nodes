from __future__ import annotations

from types import SimpleNamespace

import pytest

from wee_todd_nodes.residency import H3ResidencyCoordinator


class FakeRuntime:
    def __init__(self, loaded: bool) -> None:
        self.loaded = loaded
        self.unloads = 0

    def unload(self) -> None:
        self.unloads += 1
        self.loaded = False


def test_low_memory_stage_releases_only_loaded_inactive_components() -> None:
    runtimes = {
        "text_encoder": FakeRuntime(True),
        "transformer": FakeRuntime(True),
        "video_vae": FakeRuntime(False),
        "audio_vae": FakeRuntime(True),
        "pipeline": FakeRuntime(True),
    }

    released = H3ResidencyCoordinator().prepare(
        "transformer", runtimes, enabled=True
    )

    assert released == ("text_encoder", "audio_vae", "pipeline")
    assert runtimes["transformer"].loaded is True
    assert runtimes["transformer"].unloads == 0
    assert runtimes["video_vae"].unloads == 0
    assert all(not runtimes[name].loaded for name in released)


def test_normal_mode_preserves_keep_warm_components() -> None:
    runtime = FakeRuntime(True)

    released = H3ResidencyCoordinator().prepare(
        "transformer", {"text_encoder": runtime}, enabled=False
    )

    assert released == ()
    assert runtime.loaded is True
    assert runtime.unloads == 0


@pytest.mark.parametrize("stage", ["unknown", ""])
def test_residency_coordinator_rejects_unknown_active_stage(stage: str) -> None:
    with pytest.raises(ValueError, match="Unknown H3 residency stage"):
        H3ResidencyCoordinator().prepare(stage, {}, enabled=True)


def test_residency_coordinator_rejects_unknown_runtime_stage() -> None:
    with pytest.raises(ValueError, match="Unknown H3 runtime stages"):
        H3ResidencyCoordinator().prepare(
            "transformer", {"other": SimpleNamespace(loaded=False)}, enabled=True
        )
