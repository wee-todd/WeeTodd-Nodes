from wee_todd_nodes.nodes import NODE_CLASS_MAPPINGS, WeeToddH3GenerationConfig


def test_expected_nodes_are_registered():
    assert len(NODE_CLASS_MAPPINGS) == 4


def test_generation_config_node_returns_validated_value():
    (config,) = WeeToddH3GenerationConfig().configure(5.0, 8, 42, 640, 384, True)
    assert config.seed == 42
    assert config.drop_adaln is True
