from types import SimpleNamespace

import mlx.core as mx
import pytest

from minimax_h3_mlx.fasth3_approx import (
    FastH3ApproximationConfig,
    approximation_report,
    build_target_video_pairing,
    pair_target_video_rows,
    restore_target_video_rows,
    select_active_layers,
)
from minimax_h3_mlx.fasth3_low_bit import low_bit_tensorops_capability


def _modulation_cache(scores):
    tables = []
    for score in scores:
        zero = mx.zeros((2, 3), dtype=mx.bfloat16)
        gate = mx.full((2, 3), score, dtype=mx.bfloat16)
        tables.append((zero, zero, gate, zero, zero, gate))
    return SimpleNamespace(tables=tables)


def test_layer_thinning_protects_boundaries_and_ranks_schedule_wide_gates():
    config = FastH3ApproximationConfig(
        active_layers=3,
        protect_first_layers=1,
        protect_last_layers=1,
    )

    selected = select_active_layers(config, 5, _modulation_cache([1, 2, 3, 9, 4]))

    assert selected == (0, 3, 4)


def test_layer_thinning_requires_modulation_evidence():
    config = FastH3ApproximationConfig(active_layers=4)

    with pytest.raises(ValueError, match="AdaLN modulation cache"):
        select_active_layers(config, 6)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_layer_thinning_rejects_nonfinite_ranking_evidence(value):
    with pytest.raises(ValueError, match="finite AdaLN"):
        select_active_layers(
            FastH3ApproximationConfig(active_layers=4),
            6,
            _modulation_cache([1, 1, value, 1, 1, 1]),
        )


@pytest.mark.parametrize("value", [40.0, True])
def test_layer_thinning_rejects_noninteger_layer_counts(value):
    with pytest.raises(ValueError, match="active_layers must be an integer"):
        FastH3ApproximationConfig(active_layers=value).validate()


def test_layer_thinning_ties_are_deterministic_and_report_joint_scope():
    config = FastH3ApproximationConfig(active_layers=40)
    cache = _modulation_cache([1] * 50)
    selected = select_active_layers(config, 50, cache)
    assert selected == (*range(38), 48, 49)
    assert select_active_layers(config, 50, cache) == selected
    report = approximation_report(config, selected, 50, None)
    assert report["skipped_layer_indices"] == list(range(38, 48))
    assert report["layer_scope"] == "joint_video_audio_and_text_rows"
    assert report["selection_policy"] == "schedule_wide_adaln_gate_magnitude_v1"


def test_paged_layer_thinning_executes_only_selected_pages_and_changes_output():
    from contextlib import contextmanager

    from minimax_h3_mlx.dit import MiniMaxH3DiT

    class Pages:
        num_blocks = 50
        window_size = 4
        pages_avoided = 0

        def __init__(self):
            self.loaded = []

        @contextmanager
        def selected_window(self, indices):
            self.loaded.extend(indices)
            yield [lambda x, *args, index=index, **kwargs: x + index + 1 for index in indices]

        def window(self, start):
            return self.selected_window(range(start, min(start + self.window_size, 50)))

    def run(selected):
        pages = Pages()
        owner = SimpleNamespace(paged_blocks=pages)
        cache = SimpleNamespace(get=lambda index: ())
        output, _, _ = MiniMaxH3DiT._run_paged_blocks(
            owner,
            mx.zeros((1, 2, 1)),
            None,
            None,
            None,
            None,
            None,
            cache,
            None,
            None,
            0,
            4,
            None,
            selected,
            FastH3ApproximationConfig(active_layers=40),
            None,
        )
        return output, pages

    selected = select_active_layers(
        FastH3ApproximationConfig(active_layers=40), 50, _modulation_cache(range(50))
    )
    thinned, pages = run(selected)
    full, _ = run(tuple(range(50)))
    assert pages.loaded == list(selected)
    assert pages.pages_avoided == 10
    assert float(thinned[0, 0, 0].item()) == sum(index + 1 for index in selected)
    assert not mx.array_equal(thinned, full)


def test_target_video_pairing_preserves_prefix_and_restores_full_residual_updates():
    geometry = build_target_video_pairing(2, (1, 2, 3))
    x = mx.arange(8 * 2).reshape(1, 8, 2).astype(mx.float32)
    adaln = mx.arange(8, dtype=mx.int32)
    rotary = (
        mx.arange(8 * 2).reshape(8, 2).astype(mx.float32),
        mx.arange(8 * 2).reshape(8, 2).astype(mx.float32),
    )

    reduced, reduced_adaln, reduced_rotary, state = pair_target_video_rows(
        x, adaln, rotary, geometry
    )
    changed = reduced + 3
    restored = restore_target_video_rows(changed, state)
    mx.eval(reduced, restored)

    assert reduced.shape == (1, 6, 2)
    assert reduced_adaln.tolist() == [0, 1, 2, 4, 5, 7]
    assert reduced_rotary[0].shape == (6, 2)
    assert mx.array_equal(restored, x + 3)


def test_m5_low_bit_gate_rejects_current_m3_signature():
    supported, reason = low_bit_tensorops_capability(
        {
            "device_name": "Apple M3 Ultra",
            "architecture": "applegpu_g15d",
        },
        system_name="Darwin",
        macos_release="26.6",
        mlx_version="0.32.0",
    )

    assert not supported
    assert "M5-family g17" in reason


def test_m5_low_bit_gate_accepts_documented_m5_signature():
    supported, reason = low_bit_tensorops_capability(
        {
            "device_name": "Apple M5 Max",
            "architecture": "applegpu_g17s",
        },
        system_name="Darwin",
        macos_release="26.6",
        mlx_version="0.32.0",
    )

    assert supported
    assert reason is None
