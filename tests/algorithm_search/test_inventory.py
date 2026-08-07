from minimax_h3_mlx.algorithm_search.inventory import InventoryCase, build_operation_inventory
from minimax_h3_mlx.config import DiTConfig


def test_inventory_tracks_actual_fused_h3_operations_and_resolution_scaling():
    config = DiTConfig(num_layers=2, token_refiner_num_layers=1)
    smoke = InventoryCase("smoke", 640, 384, 5.0, prompt_rows=16)
    native = InventoryCase("native", 1344, 768, 5.0, prompt_rows=16)
    records = build_operation_inventory((smoke, native), config)

    smoke_qkv = [
        record
        for record in records
        if record.case == "smoke" and record.operation_type == "fused_qkv_projection"
    ]
    native_sdpa = [
        record
        for record in records
        if record.case == "native" and record.operation_type == "attention_scores_softmax"
    ]
    smoke_sdpa = [
        record
        for record in records
        if record.case == "smoke" and record.operation_type == "attention_scores_softmax"
    ]
    assert len(smoke_qkv) == config.num_layers
    assert smoke_qkv[0].weight_constant
    assert smoke_qkv[0].shared_input_group
    assert native_sdpa[0].approximate_flops > smoke_sdpa[0].approximate_flops
    assert smoke.geometry(config)["frames"] == 124

