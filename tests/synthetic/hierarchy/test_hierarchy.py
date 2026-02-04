from collections.abc import Sequence

import pytest
import torch

from sae_lens.synthetic import (
    Hierarchy,
    HierarchyConfig,
    HierarchyNode,
    generate_hierarchy,
    hierarchy_modifier,
)


def test_generate_hierarchy_creates_correct_number_of_roots():
    cfg = HierarchyConfig(total_root_nodes=5, branching_factor=2, max_depth=2)
    result = generate_hierarchy(100, cfg, seed=42)

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
    )
    result = generate_hierarchy(200, cfg, seed=42)

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
    )
    result = generate_hierarchy(100, cfg, seed=42)

    def has_me_parents(nodes: Sequence[HierarchyNode]) -> bool:
        for node in nodes:
            if node.mutually_exclusive_children:
                return True
            if has_me_parents(node.children):
                return True
        return False

    assert not has_me_parents(result.roots)


def test_generate_hierarchy_uses_seed_for_reproducibility():
    cfg = HierarchyConfig(total_root_nodes=5, branching_factor=3, max_depth=2)
    result1 = generate_hierarchy(100, cfg, seed=12345)
    result2 = generate_hierarchy(100, cfg, seed=12345)

    # Same seed should produce same structure
    assert result1.feature_indices_used == result2.feature_indices_used


def test_generated_hierarchy_to_dict_from_dict_roundtrip():
    cfg = HierarchyConfig(total_root_nodes=3, branching_factor=2, max_depth=2)
    original = generate_hierarchy(50, cfg, seed=42)
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
    )
    result = generate_hierarchy(200, cfg, seed=42)

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
    )
    result = generate_hierarchy(200, cfg, seed=42)

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
    )
    result = generate_hierarchy(500, cfg, seed=42)

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


def test_generate_hierarchy_runs_out_of_features_mid_construction():
    cfg = HierarchyConfig(
        total_root_nodes=5,
        branching_factor=3,
        max_depth=2,
    )
    # With 5 roots, branching=3, depth=2: we need 5 + 5*3 + 5*3*3 = 5+15+45 = 65 features
    # Provide only 30 features - should stop mid-construction
    result = generate_hierarchy(30, cfg, seed=42)

    # Should have used exactly the available features
    assert len(result.feature_indices_used) <= 30

    # Structure should still be valid - modifier should work
    assert result.modifier is not None
    activations = torch.rand(10, 30)
    output = result.modifier(activations)
    assert output.shape == (10, 30)


def test_generate_hierarchy_runs_out_of_features_before_all_roots():
    cfg = HierarchyConfig(
        total_root_nodes=10,
        branching_factor=2,
        max_depth=1,
    )
    # Provide only 5 features - can't create all 10 roots
    result = generate_hierarchy(5, cfg, seed=42)

    # Should have created as many roots as possible
    assert len(result.roots) == 5
    assert len(result.feature_indices_used) == 5

    # No children because all features used for roots
    for root in result.roots:
        assert len(root.children) == 0


def test_generate_hierarchy_determinism_across_runs():
    cfg = HierarchyConfig(
        total_root_nodes=10,
        branching_factor=(2, 4),  # Variable branching
        max_depth=3,
        mutually_exclusive_portion=0.5,
    )

    # Run generation twice with same seed
    result1 = generate_hierarchy(500, cfg, seed=12345)
    result2 = generate_hierarchy(500, cfg, seed=12345)

    # Should produce identical results
    assert result1 == result2

    # Also verify the serialization matches
    assert result1.to_dict() == result2.to_dict()

    # Run with different seed - should be different
    result3 = generate_hierarchy(500, cfg, seed=54321)
    assert result1 != result3


def test_generate_hierarchy_assigns_indices_by_depth():
    cfg = HierarchyConfig(
        total_root_nodes=10,
        branching_factor=3,
        max_depth=2,
    )
    result = generate_hierarchy(500, cfg, seed=42)

    # Roots should be at indices 0-9
    depth_0_indices: list[int] = []
    for r in result.roots:
        assert r.feature_index is not None
        depth_0_indices.append(r.feature_index)
    assert depth_0_indices == list(range(10))

    # Collect all indices by depth
    depth_1_indices: list[int] = []
    depth_2_indices: list[int] = []

    for root in result.roots:
        for child in root.children:
            assert child.feature_index is not None
            depth_1_indices.append(child.feature_index)
            for grandchild in child.children:
                assert grandchild.feature_index is not None
                depth_2_indices.append(grandchild.feature_index)

    # All depth 0 indices should be less than all depth 1 indices
    if depth_1_indices:
        assert max(depth_0_indices) < min(depth_1_indices)

    # All depth 1 indices should be less than all depth 2 indices
    if depth_2_indices:
        assert max(depth_1_indices) < min(depth_2_indices)

    # Indices within each depth should be contiguous and increasing
    assert depth_0_indices == list(range(0, len(depth_0_indices)))
    if depth_1_indices:
        assert depth_1_indices == list(
            range(len(depth_0_indices), len(depth_0_indices) + len(depth_1_indices))
        )


class TestComputeProbabilityCorrectionFactors:
    def test_simple_parent_child(self):
        child = HierarchyNode(feature_index=1)
        root = HierarchyNode(feature_index=0, children=[child])
        hierarchy = Hierarchy(roots=[root], modifier=hierarchy_modifier([root]))

        base_probs = torch.tensor([0.5, 0.3])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        assert correction[0] == 1.0
        assert correction[1] == pytest.approx(1.0 / 0.5)

    def test_deep_hierarchy(self):
        grandchild = HierarchyNode(feature_index=2)
        child = HierarchyNode(feature_index=1, children=[grandchild])
        root = HierarchyNode(feature_index=0, children=[child])
        hierarchy = Hierarchy(roots=[root], modifier=hierarchy_modifier([root]))

        base_probs = torch.tensor([0.5, 0.4, 0.3])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        # Correction is 1/base[parent], not 1/product(all ancestors)
        assert correction[0] == 1.0
        assert correction[1] == pytest.approx(1.0 / 0.5)  # parent is root (0.5)
        assert correction[2] == pytest.approx(1.0 / 0.4)  # parent is child (0.4)

    def test_organizational_nodes(self):
        child1 = HierarchyNode(feature_index=0)
        child2 = HierarchyNode(feature_index=1)
        org_root = HierarchyNode(feature_index=None, children=[child1, child2])
        hierarchy = Hierarchy(roots=[org_root], modifier=hierarchy_modifier([org_root]))

        base_probs = torch.tensor([0.5, 0.3])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        assert correction[0] == 1.0
        assert correction[1] == 1.0

    def test_features_outside_hierarchy(self):
        child = HierarchyNode(feature_index=1)
        root = HierarchyNode(feature_index=0, children=[child])
        hierarchy = Hierarchy(roots=[root], modifier=hierarchy_modifier([root]))

        base_probs = torch.tensor([0.5, 0.3, 0.6, 0.4])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        assert correction[0] == 1.0
        assert correction[1] == pytest.approx(1.0 / 0.5)
        assert correction[2] == 1.0
        assert correction[3] == 1.0

    def test_multiple_trees(self):
        child1 = HierarchyNode(feature_index=1)
        tree1 = HierarchyNode(feature_index=0, children=[child1])
        child2 = HierarchyNode(feature_index=3)
        tree2 = HierarchyNode(feature_index=2, children=[child2])
        hierarchy = Hierarchy(
            roots=[tree1, tree2], modifier=hierarchy_modifier([tree1, tree2])
        )

        base_probs = torch.tensor([0.5, 0.3, 0.4, 0.2])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        assert correction[0] == 1.0
        assert correction[1] == pytest.approx(1.0 / 0.5)
        assert correction[2] == 1.0
        assert correction[3] == pytest.approx(1.0 / 0.4)

    def test_zero_ancestor_probability(self):
        child = HierarchyNode(feature_index=1)
        root = HierarchyNode(feature_index=0, children=[child])
        hierarchy = Hierarchy(roots=[root], modifier=hierarchy_modifier([root]))

        base_probs = torch.tensor([0.0, 0.3])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        assert correction[0] == 1.0
        assert correction[1] == 1.0

    def test_very_small_parent_probability(self):
        child = HierarchyNode(feature_index=1)
        root = HierarchyNode(feature_index=0, children=[child])
        hierarchy = Hierarchy(roots=[root], modifier=hierarchy_modifier([root]))

        # Very small but nonzero parent probability
        base_probs = torch.tensor([1e-8, 0.3])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        assert correction[0] == 1.0
        # Correction factor should be large but finite
        assert correction[1] == pytest.approx(1.0 / 1e-8, rel=1e-6)
        assert torch.isfinite(correction[1])

    def test_me_correction_with_zero_sibling_probability(self):
        child1 = HierarchyNode(feature_index=1)
        child2 = HierarchyNode(feature_index=2)
        root = HierarchyNode(
            feature_index=0, children=[child1, child2], mutually_exclusive_children=True
        )
        hierarchy = Hierarchy(roots=[root], modifier=hierarchy_modifier([root]))

        # One sibling has zero probability
        base_probs = torch.tensor([0.5, 0.3, 0.0])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        # Child with zero prob sibling: ME correction = 1 + 0/0.5 = 1
        # Child with nonzero sibling: ME correction = 1 + 0.3/0.5 = 1.6
        hierarchy_corr = 1.0 / 0.5
        assert correction[1] == pytest.approx(hierarchy_corr * 1.0)  # no ME boost
        assert correction[2] == pytest.approx(hierarchy_corr * (1 + 0.3 / 0.5))

    def test_me_correction_with_very_high_probabilities(self):
        child1 = HierarchyNode(feature_index=1)
        child2 = HierarchyNode(feature_index=2)
        child3 = HierarchyNode(feature_index=3)
        root = HierarchyNode(
            feature_index=0,
            children=[child1, child2, child3],
            mutually_exclusive_children=True,
        )
        hierarchy = Hierarchy(roots=[root], modifier=hierarchy_modifier([root]))

        # Very high base probabilities close to parent
        base_probs = torch.tensor([0.9, 0.8, 0.7, 0.6])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        hierarchy_corr = 1.0 / 0.9
        # ME correction = 1 + sum(other_probs) / parent_prob
        # For child1: 1 + (0.7 + 0.6) / 0.9 = 1 + 1.44... ≈ 2.44
        # For child2: 1 + (0.8 + 0.6) / 0.9 = 1 + 1.55... ≈ 2.55
        # For child3: 1 + (0.8 + 0.7) / 0.9 = 1 + 1.67... ≈ 2.67
        assert correction[1] == pytest.approx(hierarchy_corr * (1 + (0.7 + 0.6) / 0.9))
        assert correction[2] == pytest.approx(hierarchy_corr * (1 + (0.8 + 0.6) / 0.9))
        assert correction[3] == pytest.approx(hierarchy_corr * (1 + (0.8 + 0.7) / 0.9))

    def test_me_correction_numerical_stability_equal_probs(self):
        children = [HierarchyNode(feature_index=i) for i in range(1, 5)]
        root = HierarchyNode(
            feature_index=0, children=children, mutually_exclusive_children=True
        )
        hierarchy = Hierarchy(roots=[root], modifier=hierarchy_modifier([root]))

        # All children have same probability
        base_probs = torch.tensor([0.5, 0.2, 0.2, 0.2, 0.2])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        # All children should have same correction factor
        hierarchy_corr = 1.0 / 0.5
        # ME correction = 1 + (3 * 0.2) / 0.5 = 1 + 1.2 = 2.2
        expected = hierarchy_corr * (1 + 0.6 / 0.5)
        for i in range(1, 5):
            assert correction[i] == pytest.approx(expected)

    def test_preserves_dtype(self):
        child = HierarchyNode(feature_index=1)
        root = HierarchyNode(feature_index=0, children=[child])
        hierarchy = Hierarchy(roots=[root], modifier=hierarchy_modifier([root]))

        base_probs = torch.tensor([0.5, 0.3], dtype=torch.float64)
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        assert correction.dtype == torch.float64

    def test_correction_recovers_original_rates_without_mutual_exclusion(self):
        # Create deeper hierarchies without mutual exclusion:
        # Tree 1: Root (0) -> Child (1) -> Grandchild (2)
        #                  -> Child (3)
        # Tree 2: Root (4) -> Child (5) -> Grandchild (6)
        # Feature outside hierarchy (7)
        grandchild1 = HierarchyNode(feature_index=2)
        child1a = HierarchyNode(feature_index=1, children=[grandchild1])
        child1b = HierarchyNode(feature_index=3)
        tree1 = HierarchyNode(feature_index=0, children=[child1a, child1b])

        grandchild2 = HierarchyNode(feature_index=6)
        child2 = HierarchyNode(feature_index=5, children=[grandchild2])
        tree2 = HierarchyNode(feature_index=4, children=[child2])

        hierarchy = Hierarchy(
            roots=[tree1, tree2], modifier=hierarchy_modifier([tree1, tree2])
        )

        # Use probabilities that won't exceed 1.0 after correction
        # For child: correction = 1/base[parent], so base[child] <= base[parent]
        # For grandchild: correction = 1/base[child], so base[grandchild] <= base[child]
        #
        # Tree 1: base[0]=0.9, base[1]=0.8 (0.8 <= 0.9 ✓), base[2]=0.5 (0.5 <= 0.8 ✓)
        # Tree 2: base[4]=0.9, base[5]=0.8 (0.8 <= 0.9 ✓), base[6]=0.6 (0.6 <= 0.8 ✓)
        base_probs = torch.tensor([0.9, 0.8, 0.5, 0.7, 0.9, 0.8, 0.6, 0.5])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        # Verify correction factors: correction[i] = 1 / base[parent]
        assert correction[0] == 1.0  # root (no parent)
        assert correction[1] == pytest.approx(1.0 / 0.9)  # child of 0
        assert correction[2] == pytest.approx(1.0 / 0.8)  # grandchild, parent is 1
        assert correction[3] == pytest.approx(1.0 / 0.9)  # child of 0
        assert correction[4] == 1.0  # root (no parent)
        assert correction[5] == pytest.approx(1.0 / 0.9)  # child of 4
        assert correction[6] == pytest.approx(1.0 / 0.8)  # grandchild, parent is 5
        assert correction[7] == 1.0  # outside hierarchy

        # Corrected probabilities
        corrected_probs = base_probs * correction

        # Verify corrected probs are all <= 1.0 (no clamping needed)
        assert (
            corrected_probs <= 1.0
        ).all(), f"Corrected probs exceed 1.0: {corrected_probs.tolist()}"

        # Sample all features with corrected probabilities
        n_samples = 100_000
        random_vals = torch.rand(n_samples, len(base_probs))
        activations = (random_vals < corrected_probs).float()

        # Apply hierarchy constraints
        assert hierarchy.modifier is not None
        result = hierarchy.modifier(activations)

        # Measure effective firing rates
        effective_rates = result.mean(dim=0)

        # Compare to base probabilities - all should match now
        # With 100k samples, std ≈ sqrt(p(1-p)/n) ≈ 0.0016 max
        # 6 std deviations gives us ~0.01 margin
        tolerance = 0.015
        for i in range(len(base_probs)):
            expected = base_probs[i].item()
            actual = effective_rates[i].item()
            assert actual == pytest.approx(expected, abs=tolerance), (
                f"Feature {i}: expected rate {expected:.4f}, got {actual:.4f}, "
                f"(tolerance={tolerance})"
            )

    def test_simple_mutual_exclusion_correction(self):
        child1 = HierarchyNode(feature_index=1)
        child2 = HierarchyNode(feature_index=2)
        root = HierarchyNode(
            feature_index=0, children=[child1, child2], mutually_exclusive_children=True
        )
        hierarchy = Hierarchy(roots=[root], modifier=hierarchy_modifier([root]))

        base_probs = torch.tensor([0.5, 0.3, 0.2])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        # Root has no correction
        assert correction[0] == 1.0

        # Children get hierarchy correction (1/parent_prob) * ME correction
        # ME correction = 1 + sum(other_sibling_probs) / parent_prob
        p_parent = 0.5
        p1, p2 = 0.3, 0.2
        me_1 = 1 + p2 / p_parent  # 1 + 0.2/0.5 = 1.4
        me_2 = 1 + p1 / p_parent  # 1 + 0.3/0.5 = 1.6

        hierarchy_correction = 1.0 / p_parent
        assert correction[1] == pytest.approx(hierarchy_correction * me_1)
        assert correction[2] == pytest.approx(hierarchy_correction * me_2)

    def test_mutual_exclusion_three_siblings(self):
        child1 = HierarchyNode(feature_index=1)
        child2 = HierarchyNode(feature_index=2)
        child3 = HierarchyNode(feature_index=3)
        root = HierarchyNode(
            feature_index=0,
            children=[child1, child2, child3],
            mutually_exclusive_children=True,
        )
        hierarchy = Hierarchy(roots=[root], modifier=hierarchy_modifier([root]))

        base_probs = torch.tensor([0.6, 0.1, 0.2, 0.15])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        hierarchy_correction = 1.0 / 0.6
        parent_prob = 0.6

        # ME correction = 1 + sum(other_sibling_probs) / parent_prob
        # For child1 (0.1): others = 0.2 + 0.15 = 0.35
        # For child2 (0.2): others = 0.1 + 0.15 = 0.25
        # For child3 (0.15): others = 0.1 + 0.2 = 0.30
        me_1 = 1 + 0.35 / parent_prob
        me_2 = 1 + 0.25 / parent_prob
        me_3 = 1 + 0.30 / parent_prob

        assert correction[1] == pytest.approx(hierarchy_correction * me_1)
        assert correction[2] == pytest.approx(hierarchy_correction * me_2)
        assert correction[3] == pytest.approx(hierarchy_correction * me_3)

    def test_mutual_exclusion_root_level(self):
        child1 = HierarchyNode(feature_index=0)
        child2 = HierarchyNode(feature_index=1)
        # Organizational root with ME children (no feature_index)
        org_root = HierarchyNode(
            feature_index=None,
            children=[child1, child2],
            mutually_exclusive_children=True,
        )
        hierarchy = Hierarchy(roots=[org_root], modifier=hierarchy_modifier([org_root]))

        base_probs = torch.tensor([0.3, 0.2])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        # No hierarchy correction (parent is organizational with prob 1.0)
        # ME correction = 1 + other_prob / parent_prob = 1 + other_prob
        p_parent = 1.0
        p1, p2 = 0.3, 0.2
        me_1 = 1 + p2 / p_parent  # 1 + 0.2 = 1.2
        me_2 = 1 + p1 / p_parent  # 1 + 0.3 = 1.3

        assert correction[0] == pytest.approx(me_1)
        assert correction[1] == pytest.approx(me_2)

    def test_nested_mutual_exclusion(self):
        # Create nested ME: root has ME children, one child also has ME grandchildren
        grandchild1 = HierarchyNode(feature_index=3)
        grandchild2 = HierarchyNode(feature_index=4)
        child1 = HierarchyNode(
            feature_index=1,
            children=[grandchild1, grandchild2],
            mutually_exclusive_children=True,
        )
        child2 = HierarchyNode(feature_index=2)
        root = HierarchyNode(
            feature_index=0, children=[child1, child2], mutually_exclusive_children=True
        )
        hierarchy = Hierarchy(roots=[root], modifier=hierarchy_modifier([root]))

        base_probs = torch.tensor([0.8, 0.4, 0.3, 0.1, 0.15])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        assert correction[0] == 1.0  # root

        # Children of root (ME group under root with prob 0.8)
        root_prob = 0.8
        p1, p2 = 0.4, 0.3
        me_child1 = 1 + p2 / root_prob  # 1 + 0.3/0.8 = 1.375
        me_child2 = 1 + p1 / root_prob  # 1 + 0.4/0.8 = 1.5

        hier_corr_child = 1.0 / root_prob
        assert correction[1] == pytest.approx(hier_corr_child * me_child1)
        assert correction[2] == pytest.approx(hier_corr_child * me_child2)

        # Grandchildren (ME group under child1 with prob 0.4)
        child1_prob = 0.4
        gc1, gc2 = 0.1, 0.15
        me_gc1 = 1 + gc2 / child1_prob  # 1 + 0.15/0.4 = 1.375
        me_gc2 = 1 + gc1 / child1_prob  # 1 + 0.1/0.4 = 1.25

        hier_corr_grandchild = 1.0 / child1_prob
        assert correction[3] == pytest.approx(hier_corr_grandchild * me_gc1)
        assert correction[4] == pytest.approx(hier_corr_grandchild * me_gc2)

    def test_no_me_no_extra_correction(self):
        child1 = HierarchyNode(feature_index=1)
        child2 = HierarchyNode(feature_index=2)
        root = HierarchyNode(
            feature_index=0,
            children=[child1, child2],
            mutually_exclusive_children=False,
        )
        hierarchy = Hierarchy(roots=[root], modifier=hierarchy_modifier([root]))

        base_probs = torch.tensor([0.5, 0.3, 0.2])
        correction = hierarchy.compute_probability_correction_factors(base_probs)

        # Without ME, only hierarchy correction applies
        assert correction[0] == 1.0
        assert correction[1] == pytest.approx(1.0 / 0.5)
        assert correction[2] == pytest.approx(1.0 / 0.5)

    def test_me_correction_improves_effective_rates(self):
        # Create hierarchy with ME children
        children = [HierarchyNode(feature_index=i) for i in range(1, 5)]
        root = HierarchyNode(
            feature_index=0, children=children, mutually_exclusive_children=True
        )
        hierarchy = Hierarchy(roots=[root], modifier=hierarchy_modifier([root]))

        # Use moderate probabilities that won't exceed 1.0 after correction
        base_probs = torch.tensor([0.8, 0.15, 0.12, 0.1, 0.08])
        correction = hierarchy.compute_probability_correction_factors(base_probs)
        corrected_probs = base_probs * correction

        # Corrected probabilities must not exceed 1.0 - if they do, the test
        # setup is broken (base probs too high relative to parent)
        assert (
            corrected_probs <= 1.0
        ).all(), f"Corrected probs exceed 1.0: {corrected_probs.tolist()}"

        n_samples = 200_000

        # Sample with corrected probabilities
        random_vals = torch.rand(n_samples, len(base_probs))
        activations_corrected = (random_vals < corrected_probs).float()
        assert hierarchy.modifier is not None
        result_corrected = hierarchy.modifier(activations_corrected)
        rates_corrected = result_corrected.mean(dim=0)

        # Sample with uncorrected probabilities (hierarchy only)
        hierarchy_only_correction = torch.tensor(
            [1.0, 1.0 / 0.8, 1.0 / 0.8, 1.0 / 0.8, 1.0 / 0.8]
        )
        uncorrected_probs = base_probs * hierarchy_only_correction
        random_vals2 = torch.rand(n_samples, len(base_probs))
        activations_uncorrected = (random_vals2 < uncorrected_probs).float()
        result_uncorrected = hierarchy.modifier(activations_uncorrected)
        rates_uncorrected = result_uncorrected.mean(dim=0)

        # Verify basic properties for each ME child
        total_error_corrected = 0.0
        total_error_uncorrected = 0.0
        for i in range(1, 5):
            target = base_probs[i].item()
            error_corrected = abs(rates_corrected[i].item() - target)
            error_uncorrected = abs(rates_uncorrected[i].item() - target)
            total_error_corrected += error_corrected
            total_error_uncorrected += error_uncorrected

            # Uncorrected should have lower rates than target (due to ME filtering)
            assert rates_uncorrected[i].item() < target

            # Corrected should have higher rates than uncorrected
            assert rates_corrected[i].item() > rates_uncorrected[i].item()

        # Total error across all ME children should be reasonably small
        # The approximation isn't perfect, but should keep total error under 0.10
        # (average of 0.025 per feature, roughly 20% relative error)
        max_total_error = 0.10
        assert total_error_corrected < max_total_error, (
            f"Total corrected error {total_error_corrected:.4f} should be < "
            f"{max_total_error}"
        )
