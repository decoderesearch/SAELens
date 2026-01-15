from collections.abc import Sequence

import pytest

from sae_lens.synthetic import (
    Hierarchy,
    HierarchyConfig,
    HierarchyNode,
    generate_hierarchy,
)


def test_hierarchy_config_default_values():
    cfg = HierarchyConfig()
    assert cfg.total_root_nodes == 100
    assert cfg.branching_factor == 100
    assert cfg.max_depth == 2
    assert cfg.mutually_exclusive_portion == 0.0
    assert cfg.seed is None


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


def test_hierarchy_config_to_dict_from_dict_roundtrip():
    original = HierarchyConfig(
        total_root_nodes=10,
        branching_factor=3,
        max_depth=4,
        mutually_exclusive_portion=0.3,
        seed=42,
    )
    d = original.to_dict()
    restored = HierarchyConfig.from_dict(d)
    assert restored.total_root_nodes == original.total_root_nodes
    assert restored.branching_factor == original.branching_factor
    assert restored.max_depth == original.max_depth
    assert restored.mutually_exclusive_portion == original.mutually_exclusive_portion
    assert restored.seed == original.seed


def test_generate_hierarchy_creates_correct_number_of_roots():
    cfg = HierarchyConfig(total_root_nodes=5, branching_factor=2, max_depth=2, seed=42)
    result = generate_hierarchy(100, cfg)

    assert len(result.roots) == 5
    # Each root should have children (since max_depth=2, roots are parents)
    for root in result.roots:
        assert len(root.children) > 0


def test_generate_hierarchy_applies_mutual_exclusion():
    # max_depth=1 means only roots are parents (children are all leaves)
    cfg = HierarchyConfig(
        total_root_nodes=10,
        branching_factor=3,
        max_depth=1,
        mutually_exclusive_portion=1.0,
        seed=42,
    )
    result = generate_hierarchy(200, cfg)

    # Count ME parents
    def count_me_parents(nodes: Sequence[HierarchyNode]) -> int:
        count = 0
        for node in nodes:
            if node.mutually_exclusive_children:
                count += 1
            count += count_me_parents(node.children)
        return count

    me_count = count_me_parents(result.roots)
    # With max_depth=1, only the 10 roots are parents
    assert me_count == 10


def test_generate_hierarchy_no_mutual_exclusion_by_default():
    cfg = HierarchyConfig(
        total_root_nodes=5,
        branching_factor=2,
        max_depth=2,
        mutually_exclusive_portion=0.0,
        seed=42,
    )
    result = generate_hierarchy(100, cfg)

    def has_me_parents(nodes: Sequence[HierarchyNode]) -> bool:
        for node in nodes:
            if node.mutually_exclusive_children:
                return True
            if has_me_parents(node.children):
                return True
        return False

    assert not has_me_parents(result.roots)


def test_generate_hierarchy_uses_seed_for_reproducibility():
    cfg = HierarchyConfig(
        total_root_nodes=5, branching_factor=3, max_depth=2, seed=12345
    )
    result1 = generate_hierarchy(100, cfg)
    result2 = generate_hierarchy(100, cfg)

    # Same seed should produce same structure
    assert result1.feature_indices_used == result2.feature_indices_used


def test_generated_hierarchy_to_dict_from_dict_roundtrip():
    cfg = HierarchyConfig(total_root_nodes=3, branching_factor=2, max_depth=2, seed=42)
    original = generate_hierarchy(50, cfg)
    d = original.to_dict()
    restored = Hierarchy.from_dict(d)

    assert restored.feature_indices_used == original.feature_indices_used
    assert len(restored.roots) == len(original.roots)
    # Modifier should be recreated
    assert (restored.modifier is None) == (original.modifier is None)
