import pytest

from wee_todd_mlx.acceleration import resolve_h3_acceleration


def hardware(gib):
    return {"chip": "Test Mac", "memoryBytes": gib * 1024**3, "osVersion": "26.0"}


def test_small_mac_automatic_keeps_paged_execution():
    result = resolve_h3_acceleration(hardware=hardware(36))
    assert result["memoryPolicy"] == "paged"
    assert result["projectionBackend"] == "auto"
    assert result["hardware"]["memoryBytes"] == 36 * 1024**3
    assert "36" in result["explanation"]


def test_large_mac_does_not_imply_unqualified_residency():
    assert resolve_h3_acceleration(hardware=hardware(256))["memoryPolicy"] == "recipe"


def test_explicit_preferences_override_hardware_recommendations():
    result = resolve_h3_acceleration(
        {"h3MemoryPolicy": "resident", "h3ProjectionBackend": "mlx"}, hardware(36)
    )
    assert result["memoryPolicy"] == "resident"
    assert result["projectionBackend"] == "mlx"
    assert "not a memory-fit guarantee" in result["explanation"]


def test_unknown_hardware_remains_conservative():
    assert resolve_h3_acceleration(hardware={})["memoryPolicy"] == "paged"


@pytest.mark.parametrize("preferences", [
    {"h3MemoryPolicy": "fast"}, {"h3ProjectionBackend": "cuda"}, {"flashAttention": True},
])
def test_unknown_preferences_rejected(preferences):
    with pytest.raises(ValueError):
        resolve_h3_acceleration(preferences, hardware(36))


def test_explicit_paged_normal_preserves_automatic_default_and_describes_buffers():
    report = resolve_h3_acceleration({'h3MemoryPolicy': 'pagedNormal'}, hardware(36))
    assert report['memoryPolicy'] == 'pagedNormal'
    assert 'normal working buffers' in report['explanation']
    assert 'Resident sampling retains' not in report['explanation']
    assert resolve_h3_acceleration(hardware=hardware(36))['memoryPolicy'] == 'paged'
    assert resolve_h3_acceleration(hardware=hardware(256))['memoryPolicy'] == 'recipe'
