import pytest

from sae_lens.synthetic import HierarchyConfig


def test_hierarchy_config_default_values():
    cfg = HierarchyConfig()
    assert cfg.total_root_nodes == 100
    assert cfg.branching_factor == 100
    assert cfg.max_depth == 2
    assert cfg.mutually_exclusive_portion == 0.0
    assert cfg.mutually_exclusive_min_depth == 0
    assert cfg.mutually_exclusive_max_depth is None
    assert cfg.compensate_probabilities is False


def test_hierarchy_config_validation_non_positive_parent_nodes():
    with pytest.raises(ValueError, match="total_root_nodes must be positive"):
        HierarchyConfig(total_root_nodes=-1)
    with pytest.raises(ValueError, match="total_root_nodes must be positive"):
        HierarchyConfig(total_root_nodes=0)


def test_hierarchy_config_validation_branching_factor_too_small():
    with pytest.raises(ValueError, match="branching_factor must be at least 2"):
        HierarchyConfig(total_root_nodes=5, branching_factor=1)


def test_hierarchy_config_validation_branching_range_min():
    with pytest.raises(ValueError, match="branching_factor minimum must be at least 2"):
        HierarchyConfig(total_root_nodes=5, branching_factor=(1, 4))


def test_hierarchy_config_validation_branching_range_order():
    with pytest.raises(
        ValueError,
        match=r"branching_factor\[0\] must be <= branching_factor\[1\]",
    ):
        HierarchyConfig(total_root_nodes=5, branching_factor=(5, 3))


def test_hierarchy_config_validation_max_depth():
    with pytest.raises(ValueError, match="max_depth must be at least 1"):
        HierarchyConfig(total_root_nodes=5, max_depth=0)


def test_hierarchy_config_validation_me_portion():
    with pytest.raises(
        ValueError, match="mutually_exclusive_portion must be between 0.0 and 1.0"
    ):
        HierarchyConfig(total_root_nodes=5, mutually_exclusive_portion=1.5)


def test_hierarchy_config_validation_me_min_depth_negative():
    with pytest.raises(
        ValueError, match="mutually_exclusive_min_depth must be non-negative"
    ):
        HierarchyConfig(total_root_nodes=5, mutually_exclusive_min_depth=-1)


def test_hierarchy_config_validation_me_max_depth_less_than_min():
    with pytest.raises(
        ValueError,
        match="mutually_exclusive_max_depth must be >= mutually_exclusive_min_depth",
    ):
        HierarchyConfig(
            total_root_nodes=5,
            mutually_exclusive_min_depth=2,
            mutually_exclusive_max_depth=1,
        )


def test_hierarchy_config_to_dict_from_dict_roundtrip():
    original = HierarchyConfig(
        total_root_nodes=10,
        branching_factor=3,
        max_depth=4,
        mutually_exclusive_portion=0.3,
        mutually_exclusive_min_depth=1,
        mutually_exclusive_max_depth=2,
    )
    d = original.to_dict()
    restored = HierarchyConfig.from_dict(d)
    assert restored.total_root_nodes == original.total_root_nodes
    assert restored.branching_factor == original.branching_factor
    assert restored.max_depth == original.max_depth
    assert restored.mutually_exclusive_portion == original.mutually_exclusive_portion
    assert (
        restored.mutually_exclusive_min_depth == original.mutually_exclusive_min_depth
    )
    assert (
        restored.mutually_exclusive_max_depth == original.mutually_exclusive_max_depth
    )


def test_hierarchy_config_compensate_probabilities_serialization():
    cfg = HierarchyConfig(
        total_root_nodes=5,
        branching_factor=3,
        max_depth=2,
        compensate_probabilities=True,
    )
    d = cfg.to_dict()
    assert d["compensate_probabilities"] is True

    restored = HierarchyConfig.from_dict(d)
    assert restored.compensate_probabilities is True


def test_hierarchy_config_compensate_probabilities_default_serialization():
    cfg = HierarchyConfig(total_root_nodes=5, branching_factor=3, max_depth=2)
    d = cfg.to_dict()
    assert d["compensate_probabilities"] is False

    restored = HierarchyConfig.from_dict(d)
    assert restored.compensate_probabilities is False
