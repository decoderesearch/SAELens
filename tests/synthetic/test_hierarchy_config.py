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
    assert cfg.mutually_exclusive_min_depth == 0
    assert cfg.mutually_exclusive_max_depth is None
    assert cfg.compensate_probabilities is False
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
        seed=42,
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


def test_generate_hierarchy_me_depth_filtering_excludes_roots():
    # max_depth=2 means roots (depth 0) and their children (depth 1) are parents
    # Setting min_depth=1 should exclude roots from ME
    cfg = HierarchyConfig(
        total_root_nodes=5,
        branching_factor=3,
        max_depth=2,
        mutually_exclusive_portion=1.0,
        mutually_exclusive_min_depth=1,
        seed=42,
    )
    result = generate_hierarchy(200, cfg)

    # No roots should have ME
    for root in result.roots:
        assert not root.mutually_exclusive_children

    # But depth-1 parents should have ME (if they have >= 2 children)
    depth_1_parents_with_me = 0
    for root in result.roots:
        for child in root.children:
            if child.children and child.mutually_exclusive_children:
                depth_1_parents_with_me += 1
    assert depth_1_parents_with_me > 0


def test_generate_hierarchy_me_depth_filtering_only_roots():
    # max_depth=2 means roots (depth 0) and their children (depth 1) are parents
    # Setting max_depth=0 should only apply ME to roots
    cfg = HierarchyConfig(
        total_root_nodes=5,
        branching_factor=3,
        max_depth=2,
        mutually_exclusive_portion=1.0,
        mutually_exclusive_min_depth=0,
        mutually_exclusive_max_depth=0,
        seed=42,
    )
    result = generate_hierarchy(200, cfg)

    # All roots with >= 2 children should have ME
    for root in result.roots:
        if len(root.children) >= 2:
            assert root.mutually_exclusive_children

    # No depth-1 parents should have ME
    for root in result.roots:
        for child in root.children:
            if child.children:
                assert not child.mutually_exclusive_children


def test_generate_hierarchy_me_depth_filtering_middle_range():
    # max_depth=3 creates: roots (0), depth 1 parents, depth 2 parents
    # Only apply ME to depth 1
    cfg = HierarchyConfig(
        total_root_nodes=3,
        branching_factor=2,
        max_depth=3,
        mutually_exclusive_portion=1.0,
        mutually_exclusive_min_depth=1,
        mutually_exclusive_max_depth=1,
        seed=42,
    )
    result = generate_hierarchy(500, cfg)

    # Roots should not have ME
    for root in result.roots:
        assert not root.mutually_exclusive_children

    # Check depth 1 and depth 2 parents
    depth_1_me_count = 0
    depth_2_me_count = 0
    for root in result.roots:
        for child in root.children:
            if child.children:  # depth 1 parent
                if child.mutually_exclusive_children:
                    depth_1_me_count += 1
                for grandchild in child.children:
                    if grandchild.children and grandchild.mutually_exclusive_children:
                        depth_2_me_count += 1

    # Depth 1 should have ME, depth 2 should not
    assert depth_1_me_count > 0
    assert depth_2_me_count == 0


def test_hierarchy_config_compensate_probabilities_serialization():
    cfg = HierarchyConfig(
        total_root_nodes=5,
        branching_factor=3,
        max_depth=2,
        compensate_probabilities=True,
        seed=42,
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
