import copy

import pytest

from wee_todd_mlx import generation_selection as selection


def profile(engine="h3", task="t2va", **config):
    return {
        "id": "renamed.json",
        "recipe": {
            "engine": engine,
            "components": {"task": task},
            "config": {"steps": 20, **config},
            "conditioning": {"task": "t2v", "inputs": []},
        },
    }


def resolve(value=None, item=None, **clip):
    return selection.resolve_generation_selection(
        value, {"engine": "h3", "attachments": [], **clip}, [item or profile()], {}
    )


def test_legacy_recipe_preserved_and_actual_evaluations_reported():
    item = profile()
    before = copy.deepcopy(item)
    result = resolve(item=item)
    assert result["recipe"] == item["recipe"]
    assert result["resolved_controls"]["evaluations"] == 19
    assert result["required_inputs"] == []
    assert item == before
    assert result["fingerprint"] == resolve(item=item)["fingerprint"]


def test_explicit_reference_recipe_conflicts_with_text_task():
    with pytest.raises(ValueError, match="t2v.*recipe"):
        resolve(
            {"task": "t2v", "preset": "balanced"}, profile(task="ref2va"), profileID="renamed.json"
        )


def test_fflf_names_missing_last_slot():
    with pytest.raises(ValueError, match="Last frame image"):
        resolve({"task": "fflf"}, profile(task="fl2va"), attachments=[{"role": "first"}])


def test_task_change_keeps_conflicting_attachments_and_fails():
    attachments = [{"role": "reference"}]
    with pytest.raises(ValueError, match="reference.*preserved"):
        resolve({"task": "t2v"}, attachments=attachments)
    assert attachments == [{"role": "reference"}]


@pytest.mark.parametrize("control", ["cfg", "shift", "refinementSteps"])
def test_native_h3_rejects_unsupported_override(control):
    with pytest.raises(ValueError, match=control):
        resolve({"task": "t2v", control: 3})


def test_euler_overrides_are_actual_evaluations():
    result = resolve({"task": "t2v", "steps": 12})
    assert result["recipe"]["config"]["steps"] == 13
    assert result["resolved_controls"]["evaluations"] == 12


def test_custom_schedule_not_translated():
    with pytest.raises(ValueError, match="steps"):
        resolve({"task": "t2v", "steps": 12}, profile(sampling_method="turbo"))


def test_ltx25_fixed_schedule_descriptors_and_override_rejection():
    item = profile("ltx25", pipeline_mode="distilled", stage1_steps=8, stage2_steps=3)
    result = resolve({"task": "t2v"}, item, engine="ltx25")
    assert result["resolved_controls"]["evaluations"] == 8
    assert result["resolved_controls"]["refinementSteps"] == 3
    assert not result["resolved_controls"]["stepsEditable"]
    with pytest.raises(ValueError, match="steps"):
        resolve({"task": "t2v", "steps": 9}, item, engine="ltx25")


def test_i2v_uses_first_only_native_fflf_transport():
    result = resolve({"task": "i2v"}, profile(task="fl2va"), attachments=[{"role": "first"}])
    assert result["recipe"]["conditioning"]["task"] == "fflf"
    assert result["required_inputs"] == ["first"]


def test_resident_explicit_policy_and_preserved_sampling():
    result = resolve({"task": "t2v", "memoryPolicy": "resident", "projectionBackend": "auto"})
    assert result["recipe"]["block_residency"] == "resident"
    assert result["recipe"]["config"]["memory_mode"] == "normal"
    assert result["recipe"]["config"]["steps"] == 20


def test_no_filename_capability_inference():
    item = profile("ltx25", pipeline_mode="distilled", stage1_steps=8, stage2_steps=3)
    result = resolve(
        {"task": "fflf"}, item, engine="ltx25", attachments=[{"role": "first"}, {"role": "last"}]
    )
    assert result["recipe"]["conditioning"]["task"] == "fflf"


def test_runtime_acceleration_applies_only_to_explicit_selection():
    capabilities = {"acceleration": {"h3ProjectionBackend": "auto", "h3MemoryPolicy": "paged"}}
    clip = {"engine": "h3", "attachments": []}
    item = profile(projection_backend="mlx", memory_mode="normal")
    legacy = selection.resolve_generation_selection(None, clip, [item], capabilities)
    assert legacy["recipe"] == item["recipe"]
    current = selection.resolve_generation_selection(
        {"task": "t2v", "preset": "balanced"}, clip, [item], capabilities
    )
    assert current["recipe"]["config"]["projection_backend"] == "auto"
    assert current["recipe"]["config"]["memory_mode"] == "low_memory_bf16"


def test_cfgpp_descriptor_counts_extra_forward_evaluations():
    item = profile(
        "ltx25",
        pipeline_mode="distilled",
        stage1_steps=8,
        stage2_steps=0,
        stage1_sampler="euler_ancestral_cfg_pp",
        cfg_pp_schedule="full",
        ic_lora_single_stage=True,
    )
    result = resolve({"task": "t2v"}, item, engine="ltx25")
    assert result["resolved_controls"]["evaluations"] == 15


def test_legacy_explicit_profile_is_not_replaced_by_other_task_match():
    item = profile(task="ref2va")
    with pytest.raises(ValueError, match="t2v.*recipe"):
        resolve(None, item, profileID=item["id"])


def test_new_balanced_auto_prefers_plain_recipe_over_approximation():
    approximate = profile()
    approximate["id"] = "first.json"
    approximate["recipe"]["fastvideo"] = {"enabled": True}
    ordinary = profile()
    result = selection.resolve_generation_selection(
        {"task": "t2v", "preset": "balanced"}, {"engine": "h3"}, [approximate, ordinary], {}
    )
    assert result["profileID"] == ordinary["id"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("vdn", {"schedule_points": 5}),
        ("fastvideo", {"enabled": True}),
        ("loras", {"adapters": [{"profile": "turbo", "path": "/unused"}]}),
    ],
)
def test_special_h3_recipe_rejects_generic_step_override(field, value):
    item = profile()
    item["recipe"][field] = value
    with pytest.raises(ValueError, match="steps"):
        resolve({"task": "t2v", "steps": 12}, item)


def test_explicit_image_task_reuses_known_fl2va_partition_from_text_recipe():
    result = resolve({"task": "i2v"}, attachments=[{"role": "first"}])
    assert result["recipe"]["components"]["task"] == "fl2va"
    assert result["recipe"]["conditioning"]["task"] == "fflf"


def test_editing_legacy_steps_as_custom_preserves_acceleration():
    item = profile(memory_mode="normal", projection_backend="mlx")
    result = selection.resolve_generation_selection(
        {"task": "t2v", "preset": "custom", "steps": 12},
        {"engine": "h3"},
        [item],
        {"acceleration": {"h3MemoryPolicy": "paged", "h3ProjectionBackend": "auto"}},
    )
    expected = copy.deepcopy(item["recipe"])
    expected["config"]["steps"] = 13
    assert result["recipe"] == expected


def test_ltx23_distilled_refinement_has_fixed_schedule():
    item = profile("ltx23", pipeline_mode="distilled", stage1_steps=8, stage2_steps=3)
    with pytest.raises(ValueError, match="refinementSteps"):
        resolve({"task": "t2v", "refinementSteps": 100}, item, engine="ltx23")


def test_attached_turbo_disables_steps_before_composition():
    with pytest.raises(ValueError, match="steps"):
        selection.resolve_generation_selection(
            {"task": "t2v", "steps": 12},
            {"engine": "h3"},
            [profile()],
            {"attached_loras": [{"path": "/unused", "profile": "turbo"}]},
        )


def test_checkpoint_manifest_sampling_contract_wins_over_renamed_profile(tmp_path):
    checkpoint = tmp_path / "neutral-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "paged_manifest.json").write_text(
        '{"format":"weetodd-h3-paged-v1",'
        '"source":"FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree",'
        '"sampling":{"schedule_points":5,"transformer_evaluations":4},"attention":"dense"}'
    )
    fixed = profile(steps=5)
    fixed["id"] = "alphabetically-first-neutral-name.json"
    fixed["recipe"]["components"]["transformer"] = str(checkpoint)
    ordinary = profile()
    selected = selection.resolve_generation_selection(
        {"task": "t2v", "preset": "balanced"}, {"engine": "h3"}, [fixed, ordinary], {}
    )
    assert selected["profileID"] == ordinary["id"]
    descriptor = selection.generation_descriptor(fixed["recipe"])
    assert descriptor["controls"]["evaluations"] == 4
    assert not descriptor["controls"]["stepsEditable"]
    exact = resolve({"task": "t2v", "preset": "custom"}, fixed, profileID=fixed["id"])
    assert exact["recipe"] == fixed["recipe"]
    with pytest.raises(ValueError, match="steps"):
        resolve({"task": "t2v", "preset": "custom", "steps": 12}, fixed, profileID=fixed["id"])


def test_checkpoint_fixed_contract_rejects_mismatched_recipe_steps(tmp_path):
    checkpoint = tmp_path / "neutral-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "paged_manifest.json").write_text(
        '{"sampling":{"schedule_points":5,"transformer_evaluations":4}}'
    )
    item = profile(steps=20)
    item["recipe"]["components"]["transformer"] = str(checkpoint)
    with pytest.raises(ValueError, match="fixed.*5"):
        resolve({"task": "t2v", "preset": "custom"}, item)


def test_checkpoint_provenance_read_is_bounded(tmp_path):
    checkpoint = tmp_path / "neutral"
    checkpoint.mkdir()
    manifest = checkpoint / "paged_manifest.json"
    with manifest.open("wb") as stream:
        stream.truncate(4 * 1024 * 1024 + 1)
    item = profile()
    item["recipe"]["components"]["transformer"] = str(checkpoint)
    with pytest.raises(ValueError, match="inspection limit"):
        selection.generation_descriptor(item["recipe"])


@pytest.mark.parametrize(
    "family,mode,topology,steps,refinement,expected_refinement",
    [
        ("ingredients_reference_sheet", "two_stage", "auto", 30, 3, 3),
        ("canny", "distilled", "two_stage_clean", 30, 20, 3),
        ("canny", "distilled", "control_refine", 30, 6, 6),
        ("canny", "distilled", "control_refine", 30, 20, 8),
        ("canny", "distilled", "single_stage", 30, 3, None),
        ("canny", "distilled", "upsample_only", 30, 3, None),
        ("motion_track", "distilled", "auto", 30, 3, None),
    ],
)
def test_ltx23_ic_counts_follow_executed_topology(
    family, mode, topology, steps, refinement, expected_refinement
):
    item = profile(
        "ltx23",
        pipeline_mode=mode,
        ic_lora_topology=topology,
        stage1_steps=steps,
        stage2_steps=refinement,
    )
    item["recipe"]["components"]["ic_loras"] = [{"family": family}]
    controls = selection.generation_descriptor(item["recipe"])["controls"]
    assert controls["evaluations"] == 8
    assert controls["refinementSteps"] == expected_refinement
    assert not controls["cfgEditable"]
    assert controls["cfg"] is None
    assert not controls["stepsEditable"]
    assert not controls["refinementStepsEditable"]


def test_ltx23_ingredients_rejects_ignored_cfg_override():
    item = profile("ltx23", pipeline_mode="two_stage", stage1_steps=30, stage2_steps=3)
    item["recipe"]["components"]["ic_loras"] = [{"family": "ingredients_reference_sheet"}]
    with pytest.raises(ValueError, match="cfg.*unsupported"):
        resolve(
            {"task": "ref2va", "cfg": 4}, item, engine="ltx23", attachments=[{"role": "reference"}]
        )


@pytest.mark.parametrize("app_policy", ["resident", "automatic"])
def test_low_memory_preset_precedes_app_residency_default(app_policy):
    item = profile(memory_mode="normal")
    item["recipe"]["block_residency"] = "resident"
    result = selection.resolve_generation_selection(
        {"task": "t2v", "preset": "lowMemory"},
        {"engine": "h3"},
        [item],
        {
            "acceleration": {"h3MemoryPolicy": app_policy, "h3ProjectionBackend": "auto"},
            "hardware": {"memoryBytes": 256 * 1024**3},
        },
    )
    assert result["recipe"]["block_residency"] == "checkpoint_default"
    assert result["recipe"]["config"]["memory_mode"] == "low_memory_bf16"


def test_explicit_clip_memory_override_precedes_low_memory_preset():
    result = resolve({"task": "t2v", "preset": "lowMemory", "memoryPolicy": "resident"})
    assert result["recipe"]["block_residency"] == "resident"
    assert result["recipe"]["config"]["memory_mode"] == "normal"


@pytest.mark.parametrize(
    "value,preferences,expected,absent",
    [
        (
            {"task": "t2v", "preset": "lowMemory"},
            {"h3MemoryPolicy": "resident", "h3ProjectionBackend": "auto"},
            "Lower-memory execution",
            "Resident sampling retains",
        ),
        (
            {
                "task": "t2v",
                "preset": "balanced",
                "memoryPolicy": "resident",
                "projectionBackend": "mlx",
            },
            {"h3MemoryPolicy": "automatic", "h3ProjectionBackend": "auto"},
            "Resident sampling retains",
            "Automatic retains",
        ),
    ],
)
def test_acceleration_explanation_describes_effective_clip_policy(
    value, preferences, expected, absent
):
    result = selection.resolve_generation_selection(
        value,
        {"engine": "h3"},
        [profile()],
        {"acceleration": preferences, "hardware": {"memoryBytes": 256 * 1024**3}},
    )
    explanation = result["generation"]["acceleration"]["explanation"]
    assert expected in explanation
    assert absent not in explanation
    assert "256 GiB" in explanation
    if value.get("projectionBackend") == "mlx":
        assert "Standard MLX projections" in explanation
        assert "Automatic projections" not in explanation


def test_h3_paged_normal_keeps_checkpoint_paging_and_cache():
    item = profile(memory_mode="low_memory_bf16", paging_cache_gb=2)
    item["recipe"]["block_residency"] = "resident"
    result = resolve({"task": "t2v", "memoryPolicy": "pagedNormal"}, item)
    assert result["recipe"]["config"]["memory_mode"] == "normal"
    assert result["recipe"]["block_residency"] == "checkpoint_default"
    assert result["recipe"]["config"]["paging_cache_gb"] == 2


def test_app_paged_normal_applies_to_balanced_but_not_low_memory():
    capabilities = {"acceleration": {"h3MemoryPolicy": "pagedNormal"}}
    balanced = selection.resolve_generation_selection(
        {"task": "t2v", "preset": "balanced"}, {"engine": "h3"}, [profile()], capabilities
    )
    assert balanced["recipe"]["config"]["memory_mode"] == "normal"
    assert balanced["recipe"]["block_residency"] == "checkpoint_default"
    assert "normal working buffers" in balanced["generation"]["acceleration"]["explanation"]
    bounded = selection.resolve_generation_selection(
        {"task": "t2v", "preset": "lowMemory"}, {"engine": "h3"}, [profile()], capabilities
    )
    assert bounded["recipe"]["config"]["memory_mode"] == "low_memory_bf16"


def test_paged_normal_rejects_non_h3():
    with pytest.raises(ValueError, match="H3 only"):
        resolve({"task": "t2v", "memoryPolicy": "pagedNormal"}, profile("ltx25"), engine="ltx25")
