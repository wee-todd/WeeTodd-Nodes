from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from ltx23_mlx.ic_lora import LTX23ICLoRASpec, install_ic_fusion
from ltx23_mlx.lora import (
    LTX23LoRASpec,
    LTX23StreamingLoRASource,
    apply_stack,
    install_loader,
    target_name,
    validate_stack,
)


def save_adapter(path, a, b, *, schema="ab", alpha=None, metadata=None, target="proj"):
    suffixes = {
        "ab": ("lora_A", "lora_B"),
        "down_up": ("lora_down", "lora_up"),
        "default": ("lora_A.default", "lora_B.default"),
    }[schema]
    weights = {f"{target}.{suffixes[0]}.weight": a, f"{target}.{suffixes[1]}.weight": b}
    if alpha is not None:
        weights[f"{target}.alpha"] = mx.array(alpha)
    mx.save_safetensors(str(path), weights, metadata=metadata or {})
    return LTX23LoRASpec(str(path), strength=0.75)


@pytest.mark.parametrize("bits", [None, 4, 8])
@pytest.mark.parametrize("schema", ["ab", "down_up", "default"])
def test_fusion_matches_explicit_math_preserves_names_and_quantization(tmp_path, bits, schema):
    mx.random.seed(17)
    linear = nn.Linear(64, 32)
    layer = nn.QuantizedLinear.from_linear(linear, group_size=32, bits=bits) if bits else linear
    a, b = mx.random.normal((2, 64)) * 0.03, mx.random.normal((32, 2)) * 0.03
    adapter = save_adapter(tmp_path / "any_source.safetensors", a, b, schema=schema, alpha=4.0)
    weight = (
        mx.dequantize(layer.weight, layer.scales, layer.biases, group_size=32, bits=bits)
        if bits
        else layer.weight
    )
    expected = weight.astype(mx.float32) + b @ a * 1.5
    expected = (
        mx.quantize(expected.astype(weight.dtype), group_size=32, bits=bits) if bits else expected
    )
    bias = layer.bias
    report = apply_stack(SimpleNamespace(proj=layer), (adapter,))
    if bits:
        assert all(
            mx.array_equal(x, y).item()
            for x, y in zip((layer.weight, layer.scales, layer.biases), expected, strict=True)
        )
        assert layer.bits == bits and layer.group_size == 32
    else:
        assert mx.array_equal(layer.weight, expected).item()
    assert mx.array_equal(layer.bias, bias).item()
    assert report[0]["nonzero_targets"] == 1
    assert not hasattr(layer, "base")


def test_zero_strength_is_exact_noop_for_quantized_base(tmp_path):
    layer = nn.QuantizedLinear(64, 32, group_size=32, bits=4)
    before = [layer.weight, layer.scales, layer.biases]
    spec = save_adapter(tmp_path / "adapter.safetensors", mx.ones((2, 64)), mx.ones((32, 2)))
    report = apply_stack(SimpleNamespace(proj=layer), (LTX23LoRASpec(spec.path, 0),))
    assert report[0]["nonzero_targets"] == 0
    assert all(
        mx.array_equal(x, y).item()
        for x, y in zip(before, [layer.weight, layer.scales, layer.biases], strict=True)
    )


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), -1])
def test_invalid_alpha_rejected_before_any_weight_mutation(tmp_path, alpha):
    layer = nn.Linear(4, 6)
    before = layer.weight
    spec = save_adapter(tmp_path / "bad.safetensors", mx.ones((2, 4)), mx.ones((6, 2)), alpha=alpha)
    with pytest.raises(ValueError, match="alpha"):
        apply_stack(SimpleNamespace(proj=layer), (spec,))
    assert mx.array_equal(layer.weight, before).item()


def test_all_targets_validated_before_mutating_model(tmp_path):
    layer = nn.Linear(4, 6)
    before = layer.weight
    valid = save_adapter(tmp_path / "valid.safetensors", mx.ones((2, 4)), mx.ones((6, 2)))
    invalid = save_adapter(
        tmp_path / "invalid.safetensors", mx.ones((2, 4)), mx.ones((6, 2)), target="missing"
    )
    with pytest.raises(ValueError, match="Unmatched"):
        apply_stack(SimpleNamespace(proj=layer), (valid, invalid))
    assert mx.array_equal(layer.weight, before).item()


def test_header_target_validation_is_filename_independent(tmp_path):
    mx.save_safetensors(
        str(tmp_path / "transformer-dev.safetensors"), {"transformer.proj.weight": mx.ones((6, 4))}
    )
    spec = save_adapter(
        tmp_path / "download.safetensors",
        mx.ones((2, 4)),
        mx.ones((6, 2)),
        target="diffusion_model.proj",
    )
    before = validate_stack((spec,), tmp_path, "one_stage")
    moved = tmp_path / "renamed.safetensors"
    Path(spec.path).rename(moved)
    assert before == validate_stack((LTX23LoRASpec(str(moved)),), tmp_path, "one_stage")
    bad = save_adapter(tmp_path / "bad.safetensors", mx.ones((2, 5)), mx.ones((6, 2)))
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_stack((bad,), tmp_path, "one_stage")


def test_streaming_stack_uses_the_same_header_validation(tmp_path):
    mx.save_safetensors(
        str(tmp_path / "transformer-dev.safetensors"),
        {"transformer.proj.weight": mx.ones((6, 4))},
    )
    spec = save_adapter(
        tmp_path / "download.safetensors",
        mx.ones((2, 4)),
        mx.ones((6, 2)),
    )
    report = validate_stack((spec,), tmp_path, "one_stage", low_ram_streaming=True)
    assert report[0]["pairs"][0]["mapped_target"] == "proj"


def test_streaming_source_normalizes_default_schema_and_global_alpha(tmp_path):
    from ltx_core_mlx.loader.block_streaming import BlockStreamer

    class TinyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(4, 4, bias=False)

    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = TinyAttention()

    transformer = tmp_path / "transformer.safetensors"
    mx.save_safetensors(
        str(transformer),
        {
            "transformer.transformer_blocks.0.attn1.to_q.weight": mx.zeros((4, 4))
        },
    )
    spec = save_adapter(
        tmp_path / "adapter.safetensors",
        mx.ones((2, 4)),
        mx.ones((4, 2)),
        schema="default",
        metadata={"model_version": "2.3", "lora_rank": "4", "lora_alpha": "2"},
        target="diffusion_model.transformer_blocks.0.attn1.to_q",
    )
    source = LTX23StreamingLoRASource(spec)
    streamer = BlockStreamer(
        transformer, block_prefix="transformer.transformer_blocks."
    )
    block = TinyBlock()
    streamer.bind(block, 0, lora_sources=[source])
    mx.eval(block.parameters())
    # B is scaled by alpha/rank (1/2), then upstream applies strength (3/4).
    assert mx.allclose(block.attn1.to_q.weight, mx.full((4, 4), 0.75))
    assert source.application_report([])["application"] == "normalized_per_block_streaming"
    source.close()
    streamer.close()


def test_ic_contract_is_explicit_and_filename_independent(tmp_path):
    metadata = {"model_version": "2.3.0", "reference_downscale_factor": "2"}
    ordinary_name = save_adapter(
        tmp_path / "completely-renamed.safetensors",
        mx.ones((2, 4)),
        mx.ones((6, 2)),
        metadata=metadata,
    )
    spec = LTX23ICLoRASpec(ordinary_name.path, "union_control", 0.8)
    report = spec.inspect()
    assert report["adapter_family"] == "union_control"
    assert report["family_provenance"] == "explicit_declaration"
    assert report["stages"] == "stage1_only"
    with pytest.raises(ValueError, match="reference_downscale_factor=1"):
        LTX23ICLoRASpec(ordinary_name.path, "ingredients_reference_sheet").inspect()


def test_ic_fusion_hooks_only_stage_one_without_replacing_clean_reload(tmp_path):
    adapter = save_adapter(
        tmp_path / "ic.safetensors",
        mx.ones((2, 4)),
        mx.ones((6, 2)),
        metadata={"model_version": "2.3", "reference_downscale_factor": "2"},
    )

    class Pipeline:
        def __init__(self):
            self.dit = SimpleNamespace(proj=nn.Linear(4, 6))

        def _fuse_loras(self):
            raise AssertionError("unvalidated upstream fusion must be replaced")

        def _reload_clean_transformer(self):
            self.dit = SimpleNamespace(proj=nn.Linear(4, 6))

    pipeline = Pipeline()
    original_reload = pipeline._reload_clean_transformer.__func__
    reports = install_ic_fusion(
        pipeline, (LTX23ICLoRASpec(adapter.path, "motion_track", 0),)
    )
    pipeline._fuse_loras()
    assert reports[0]["stages"] == "stage1_only"
    assert pipeline._reload_clean_transformer.__func__ is original_reload

    pipeline._weetodd_ic_stages = "stage1_and_control_refine"
    pipeline._fuse_loras()
    assert reports[0]["stages"] == "stage1_and_control_refine"


def test_ic_fusion_includes_audited_dev_mode_helper(tmp_path):
    ic = save_adapter(
        tmp_path / "ic.safetensors",
        mx.ones((2, 4)),
        mx.ones((6, 2)),
        metadata={"model_version": "2.3", "reference_downscale_factor": "1"},
    )
    helper = save_adapter(
        tmp_path / "distilled.safetensors",
        mx.ones((2, 4)),
        mx.ones((6, 2)),
    )

    class Pipeline:
        def __init__(self):
            self.dit = SimpleNamespace(proj=nn.Linear(4, 6))

        def _fuse_loras(self):
            raise AssertionError("unvalidated upstream fusion must be replaced")

        def _reload_clean_transformer(self):
            raise AssertionError("Dev mode must retain the fused transformer")

    pipeline = Pipeline()
    pipeline._weetodd_ic_stages = "all_two_stage"
    reports = install_ic_fusion(
        pipeline,
        (LTX23ICLoRASpec(ic.path, "ingredients_reference_sheet", 0),),
        (helper,),
    )
    pipeline._fuse_loras()
    assert reports[0]["stages"] == "all_two_stage"
    assert pipeline._weetodd_auxiliary_lora_reports[0]["adapter_role"] == (
        "distillation_helper"
    )
    assert pipeline._weetodd_auxiliary_lora_reports[0]["nonzero_targets"] == 1


def test_ingredients_auto_selects_dev_two_stage_topology():
    from ltx23_mlx.runtime import (
        LTX23GenerationConfig,
        LTX23ModelSpec,
        resolve_ic_topology,
    )

    model = LTX23ModelSpec(
        "/not-loaded",
        ic_loras=(
            LTX23ICLoRASpec(
                "/not-loaded/ingredients.safetensors", "ingredients_reference_sheet"
            ),
        ),
    )
    config = LTX23GenerationConfig(pipeline_mode="two_stage", ic_lora_topology="auto")
    assert resolve_ic_topology(model, config, "control") == "two_stage_dev"


def test_ingredients_dev_runtime_constructor_is_explicit(tmp_path, monkeypatch):
    from ltx23_mlx.runtime import LTX23GenerationConfig, LTX23ModelSpec, LTX23RuntimeCache

    constructor = {}
    installed = {}

    class Pipeline:
        def __init__(self, **kwargs):
            constructor.update(kwargs)

    def fake_install(pipeline, specs, auxiliary_specs=()):
        installed["task"] = specs
        installed["auxiliary"] = auxiliary_specs
        pipeline._weetodd_auxiliary_lora_reports = []
        return []

    monkeypatch.setattr("ltx23_mlx.runtime._pipeline_class", lambda _mode: Pipeline)
    monkeypatch.setattr("ltx23_mlx.runtime.validate_ic_stack", lambda *_args: None)
    monkeypatch.setattr("ltx23_mlx.runtime.install_ic_fusion", fake_install)
    monkeypatch.setattr(LTX23ModelSpec, "validate", lambda *_args: None)
    monkeypatch.setattr(LTX23ModelSpec, "gemma_root", lambda *_args: tmp_path)

    model = LTX23ModelSpec(
        str(tmp_path),
        ic_loras=(
            LTX23ICLoRASpec(
                str(tmp_path / "ingredients.safetensors"), "ingredients_reference_sheet"
            ),
        ),
    )
    config = LTX23GenerationConfig(pipeline_mode="two_stage")
    LTX23RuntimeCache().get(model, config, conditioning_task="control")
    assert constructor["dev_transformer"] == "transformer-dev.safetensors"
    assert constructor["distilled_lora"] == "ltx-2.3-22b-distilled-lora-384.safetensors"
    assert constructor["distilled_lora_strength"] == 0.5
    assert installed["auxiliary"][0].strength == 0.5


@pytest.mark.parametrize(
    ("topology", "expected"),
    [
        ("auto", (True, False, None)),
        ("two_stage_clean", (False, False, None)),
        ("control_refine", (False, True, 3)),
        ("upsample_only", (False, True, None)),
        ("single_stage", (True, False, None)),
    ],
)
def test_ic_topology_reaches_control_pipeline(tmp_path, monkeypatch, topology, expected):
    from ltx23_mlx.runtime import LTX23GenerationConfig, LTX23ModelSpec, LTX23RuntimeCache

    calls = []

    class Pipeline:
        def __init__(self, **_kwargs):
            pass

        def generate_and_save(
            self,
            output_path,
            video_conditioning,
            single_stage=False,
            upsample_only=False,
            refine_steps=None,
        ):
            calls.append((single_stage, upsample_only, refine_steps))
            return output_path

    monkeypatch.setattr("ltx23_mlx.runtime._pipeline_class", lambda _mode: Pipeline)
    monkeypatch.setattr("ltx23_mlx.runtime.validate_ic_stack", lambda *_args: None)
    monkeypatch.setattr("ltx23_mlx.runtime.validate_control_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("ltx23_mlx.runtime.install_ic_fusion", lambda *_args: [])
    monkeypatch.setattr(LTX23ModelSpec, "validate", lambda *_args: None)
    monkeypatch.setattr(LTX23ModelSpec, "gemma_root", lambda *_args: tmp_path)

    model = LTX23ModelSpec(
        str(tmp_path),
        ic_loras=(LTX23ICLoRASpec(str(tmp_path / "ic.safetensors"), "motion_track"),),
    )
    config = LTX23GenerationConfig(
        pipeline_mode="distilled",
        width=384,
        height=256,
        duration_seconds=1,
        ic_lora_topology=topology,
    )
    LTX23RuntimeCache().generate_to_file(
        model,
        config,
        "fixture",
        tmp_path / "out.mp4",
        control_inputs=[
            {
                "path": str(tmp_path / "guide.mp4"),
                "kind": "video",
                "control_type": "motion_track",
                "strength": 1.0,
            }
        ],
    )
    assert calls == [expected]


@pytest.mark.parametrize(
    "metadata",
    [
        {"model_version": "2.5"},
        {"reference_downscale_factor": "2"},
        {"ss_network_args": '{"use_rslora": true}'},
        {"use_dora": "True"},
        {"adapter_role": "distillation"},
        {"lora_alpha": "4", "network_alpha": "2"},
    ],
)
def test_unsupported_or_ambiguous_contracts_rejected(tmp_path, metadata):
    spec = save_adapter(
        tmp_path / "adapter.safetensors", mx.ones((2, 4)), mx.ones((6, 2)), metadata=metadata
    )
    with pytest.raises(ValueError):
        spec.inspect()


def test_loader_hook_reapplies_only_on_new_transformer_and_survives_refinement(tmp_path):
    spec = save_adapter(tmp_path / "adapter.safetensors", mx.ones((2, 4)), mx.ones((6, 2)))
    models = []

    class Pipeline:
        def _load_transformer_with_optional_streaming(self, path):
            layer = nn.Linear(4, 6, bias=False)
            layer.weight = mx.zeros((6, 4))
            model = SimpleNamespace(proj=layer)
            models.append(model)
            return model

    pipeline = Pipeline()
    report = install_loader(pipeline, (spec,))
    first = pipeline._load_transformer_with_optional_streaming("dev")
    assert mx.array_equal(first.proj.weight, mx.full((6, 4), 1.5)).item()
    # Simulate the upstream in-place refinement delta using unchanged parameter names.
    first.proj.weight = first.proj.weight + 0.25
    assert mx.array_equal(first.proj.weight, mx.full((6, 4), 1.75)).item()
    second = pipeline._load_transformer_with_optional_streaming("distilled")
    assert mx.array_equal(second.proj.weight, mx.full((6, 4), 1.5)).item()
    assert len(report) == 2


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            "base_model.model.diffusion_model.transformer_blocks.0.attn.to_out.0",
            "transformer_blocks.0.attn.to_out",
        ),
        (
            "diffusion_model.transformer_blocks.0.audio_ff.net.0.proj",
            "transformer_blocks.0.audio_ff.proj_in",
        ),
    ],
)
def test_supported_target_maps(source, expected):
    assert target_name(source) == expected


def test_exported_component_adapter_schema_and_top_level_agree():
    from ltx23_mlx.lora import recipe_specs

    item = {"path": "fixture.safetensors", "strength": 0.75}
    assert recipe_specs({"components": {"loras": [item]}}) == recipe_specs(
        {"loras": {"adapters": [item]}}
    )
    assert recipe_specs({"components": {"loras": []}}) == ()
    with pytest.raises(ValueError, match="not both"):
        recipe_specs({"components": {"loras": [item]}, "loras": {"adapters": [item]}})


def test_node_builds_lazy_ordered_stack(tmp_path):
    from ltx23_mlx.runtime import LTX23ModelSpec
    from wee_todd_nodes.ltx_nodes import WeeToddLTX23LoRALoader

    spec = save_adapter(tmp_path / "local.safetensors", mx.ones((2, 4)), mx.ones((6, 2)))
    node = WeeToddLTX23LoRALoader()
    model = LTX23ModelSpec("not_loaded")
    first, _ = node.attach(model, spec.path, 0.5)
    second, _ = node.attach(first, spec.path, -0.25, alpha=4)
    assert model.loras == ()
    assert [adapter.strength for adapter in second.loras] == [0.5, -0.25]
    assert second.loras[1].alpha == 4


def test_alpha_metadata_and_override_are_auditable(tmp_path):
    spec = save_adapter(
        tmp_path / "metadata.safetensors",
        mx.ones((2, 4)),
        mx.ones((6, 2)),
        metadata={"ss_network_alpha": "4"},
    )
    report = apply_stack(SimpleNamespace(proj=nn.Linear(4, 6)), (spec,))
    assert report[0]["target_details"][0]["effective_scale"] == 1.5
    overridden = LTX23LoRASpec(spec.path, 0.75, alpha=8)
    report = apply_stack(SimpleNamespace(proj=nn.Linear(4, 6)), (overridden,))
    assert report[0]["target_details"][0]["effective_scale"] == 3


def test_global_exporter_rank_controls_ltx23_metadata_alpha_denominator(tmp_path):
    spec = save_adapter(
        tmp_path / "global-scaling.safetensors",
        mx.ones((2, 4)),
        mx.ones((6, 2)),
        metadata={"ss_network_dim": "4", "ss_network_alpha": "1"},
    )

    report = apply_stack(SimpleNamespace(proj=nn.Linear(4, 6)), (spec,))
    detail = report[0]["target_details"][0]

    assert detail["declared_rank"] == 4
    assert detail["effective_scale"] == 0.1875


def test_cancel_before_fusion_does_not_mutate(tmp_path):
    spec = save_adapter(tmp_path / "adapter.safetensors", mx.ones((2, 4)), mx.ones((6, 2)))
    layer = nn.Linear(4, 6)
    before = layer.weight

    def cancel():
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        apply_stack(SimpleNamespace(proj=layer), (spec,), check_interrupted=cancel)
    assert mx.array_equal(layer.weight, before).item()


def test_lora_runtime_unloads_after_cancel_and_does_not_reuse_adapter_model(tmp_path, monkeypatch):
    from ltx23_mlx.runtime import LTX23GenerationConfig, LTX23ModelSpec, LTX23RuntimeCache

    spec = save_adapter(tmp_path / "adapter.safetensors", mx.ones((2, 4)), mx.ones((6, 2)))
    monkeypatch.setattr(LTX23ModelSpec, "validate", lambda *a: None)
    monkeypatch.setattr(LTX23ModelSpec, "gemma_root", lambda *a: tmp_path)
    instances = []

    class Pipeline:
        def __init__(self, **kwargs):
            instances.append(self)

        def _load_transformer_with_optional_streaming(self, path):
            return SimpleNamespace(proj=nn.Linear(4, 6))

        def generate_and_save(self, output_path):
            self._load_transformer_with_optional_streaming("fixture")
            return output_path

    monkeypatch.setattr("ltx23_mlx.runtime._pipeline_class", lambda mode: Pipeline)
    runtime = LTX23RuntimeCache()
    model = LTX23ModelSpec(str(tmp_path), loras=(spec,))
    for _ in range(2):
        report = runtime.generate_to_file(
            model, LTX23GenerationConfig(), "fixture", tmp_path / "out.mp4", unload_after=False
        )
        assert not runtime.loaded
        assert not report["runtime_cached"]
        assert len(report["loras"]) == 1
    assert len(instances) == 2

    def fail(_self, output_path):
        raise RuntimeError("cancelled")

    monkeypatch.setattr(Pipeline, "generate_and_save", fail)
    with pytest.raises(RuntimeError, match="cancelled"):
        runtime.generate_to_file(model, LTX23GenerationConfig(), "fixture", tmp_path / "out.mp4")
    assert not runtime.loaded
