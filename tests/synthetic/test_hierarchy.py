import pytest
import torch

from sae_lens.synthetic import HierarchyNode, hierarchy_modifier


def test_HierarchyNode_simple_construction():
    root = HierarchyNode(feature_index=0)
    assert root.feature_index == 0
    assert root.children == []
    assert not root.mutually_exclusive_children


def test_HierarchyNode_with_children():
    child1 = HierarchyNode(feature_index=1)
    child2 = HierarchyNode(feature_index=2)
    root = HierarchyNode(feature_index=0, children=[child1, child2])

    assert root.feature_index == 0
    assert len(root.children) == 2
    assert child1.feature_index == 1
    assert child2.feature_index == 2


def test_hierarchy_modifier_returns_correct_shape():
    child = HierarchyNode(feature_index=1)
    root = HierarchyNode(feature_index=0, children=[child])
    modifier = hierarchy_modifier([root])

    activations = torch.rand(100, 3)
    result = modifier(activations)
    assert result.shape == (100, 3)


def test_hierarchy_modifier_deactivates_children_when_parent_inactive():
    child = HierarchyNode(feature_index=1)
    root = HierarchyNode(feature_index=0, children=[child])
    modifier = hierarchy_modifier([root])

    # Parent inactive in all samples
    activations = torch.tensor(
        [
            [0.0, 1.0, 0.5],
            [0.0, 0.8, 0.3],
        ]
    )
    result = modifier(activations)

    # Child should be deactivated when parent is inactive
    assert torch.all(result[:, 1] == 0)


def test_hierarchy_modifier_keeps_children_when_parent_active():
    child = HierarchyNode(feature_index=1)
    root = HierarchyNode(feature_index=0, children=[child])
    modifier = hierarchy_modifier([root])

    # Parent active, child active
    activations = torch.tensor(
        [
            [1.0, 0.5, 0.3],
            [0.8, 0.3, 0.2],
        ]
    )
    result = modifier(activations)

    # Child values should be preserved when parent is active
    assert torch.allclose(result[:, 1], activations[:, 1])


def test_hierarchy_modifier_mixed_parent_states():
    child = HierarchyNode(feature_index=1)
    root = HierarchyNode(feature_index=0, children=[child])
    modifier = hierarchy_modifier([root])

    activations = torch.tensor(
        [
            [1.0, 0.5, 0.3],  # Parent active
            [0.0, 0.8, 0.2],  # Parent inactive
            [0.5, 0.0, 0.1],  # Parent active, child already inactive
        ]
    )
    result = modifier(activations)

    assert result[0, 1] == 0.5  # Preserved
    assert result[1, 1] == 0.0  # Deactivated
    assert result[2, 1] == 0.0  # Already inactive


def test_hierarchy_modifier_mutually_exclusive_children():
    child1 = HierarchyNode(feature_index=1)
    child2 = HierarchyNode(feature_index=2)
    root = HierarchyNode(
        feature_index=0,
        children=[child1, child2],
        mutually_exclusive_children=True,
    )
    modifier = hierarchy_modifier([root])

    # Parent active, both children active
    activations = torch.tensor(
        [
            [1.0, 0.5, 0.3],
            [1.0, 0.8, 0.6],
        ]
    )

    result = modifier(activations)

    # Both children should never be active simultaneously
    both_active = (result[:, 1] > 0) & (result[:, 2] > 0)
    assert torch.sum(both_active) == 0

    # At least one child should remain active (randomly selected)
    either_active = (result[:, 1] > 0) | (result[:, 2] > 0)
    assert torch.all(either_active)


def test_hierarchy_modifier_mutually_exclusive_allows_single_child():
    child1 = HierarchyNode(feature_index=1)
    child2 = HierarchyNode(feature_index=2)
    root = HierarchyNode(
        feature_index=0,
        children=[child1, child2],
        mutually_exclusive_children=True,
    )
    modifier = hierarchy_modifier([root])

    # Only one child active
    activations = torch.tensor(
        [
            [1.0, 0.5, 0.0],
            [1.0, 0.0, 0.3],
        ]
    )

    result = modifier(activations)

    # Single active child should remain
    assert result[0, 1] == 0.5
    assert result[0, 2] == 0.0
    assert result[1, 1] == 0.0
    assert result[1, 2] == 0.3


def test_hierarchy_modifier_non_readout_node():
    """Test organizational node with no feature_index."""
    child1 = HierarchyNode(feature_index=0)
    child2 = HierarchyNode(feature_index=1)
    root = HierarchyNode(
        feature_index=None,  # Organizational node
        children=[child1, child2],
    )
    modifier = hierarchy_modifier([root])

    # Both children active
    activations = torch.tensor(
        [
            [0.5, 0.3],
            [0.8, 0.6],
        ]
    )

    result = modifier(activations)

    # Children should be unaffected since organizational root is always "active"
    assert torch.allclose(result, activations)


def test_HierarchyNode_from_dict():
    tree_dict = {
        "feature_index": 0,
        "children": [
            {"feature_index": 1},
            {"feature_index": 2, "id": "child2"},
        ],
    }

    tree = HierarchyNode.from_dict(tree_dict)
    assert tree.feature_index == 0
    assert len(tree.children) == 2
    assert tree.children[0].feature_index == 1
    assert tree.children[1].feature_index == 2
    assert tree.children[1].feature_id == "child2"


def test_HierarchyNode_from_dict_mutually_exclusive():
    tree_dict = {
        "feature_index": 0,
        "mutually_exclusive_children": True,
        "children": [
            {"feature_index": 1},
            {"feature_index": 2},
        ],
    }

    tree = HierarchyNode.from_dict(tree_dict)
    assert tree.mutually_exclusive_children

    modifier = hierarchy_modifier([tree])
    activations = torch.tensor([[1.0, 0.5, 0.3]])
    result = modifier(activations)

    both_active = (result[:, 1] > 0) & (result[:, 2] > 0)
    assert torch.sum(both_active) == 0


def test_hierarchy_modifier_deep_hierarchy():
    grandchild = HierarchyNode(feature_index=2)
    child = HierarchyNode(feature_index=1, children=[grandchild])
    root = HierarchyNode(feature_index=0, children=[child])
    modifier = hierarchy_modifier([root])

    # All active
    activations = torch.tensor([[1.0, 0.5, 0.3]])
    result = modifier(activations)
    assert torch.allclose(result, activations)

    # Root inactive - all descendants should be inactive
    activations = torch.tensor([[0.0, 0.5, 0.3]])
    result = modifier(activations)
    assert result[0, 0] == 0.0
    assert result[0, 1] == 0.0
    assert result[0, 2] == 0.0

    # Root active, child inactive - grandchild should be inactive
    activations = torch.tensor([[1.0, 0.0, 0.3]])
    result = modifier(activations)
    assert result[0, 0] == 1.0
    assert result[0, 1] == 0.0
    assert result[0, 2] == 0.0


def test_hierarchy_modifier_does_not_modify_input():
    child = HierarchyNode(feature_index=1)
    root = HierarchyNode(feature_index=0, children=[child])
    modifier = hierarchy_modifier([root])

    activations = torch.tensor([[0.0, 0.5, 0.3]])
    original = activations.clone()

    _ = modifier(activations)

    # Original should be unchanged
    assert torch.allclose(activations, original)


def test_HierarchyNode_get_all_feature_indices():
    grandchild = HierarchyNode(feature_index=3)
    child1 = HierarchyNode(feature_index=1, children=[grandchild])
    child2 = HierarchyNode(feature_index=2)
    root = HierarchyNode(feature_index=0, children=[child1, child2])

    indices = root.get_all_feature_indices()
    assert sorted(indices) == [0, 1, 2, 3]


def test_HierarchyNode_get_all_feature_indices_with_non_readout():
    child1 = HierarchyNode(feature_index=0)
    child2 = HierarchyNode(feature_index=1)
    root = HierarchyNode(feature_index=None, children=[child1, child2])

    indices = root.get_all_feature_indices()
    assert sorted(indices) == [0, 1]


def test_HierarchyNode_repr():
    child = HierarchyNode(feature_index=1, feature_id="child")
    root = HierarchyNode(
        feature_index=0,
        children=[child],
        mutually_exclusive_children=False,
        feature_id="root",
    )

    repr_str = repr(root)
    assert "0" in repr_str
    assert "root" in repr_str
    assert "1" in repr_str
    assert "child" in repr_str


def test_HierarchyNode_repr_mutually_exclusive():
    child1 = HierarchyNode(feature_index=1)
    child2 = HierarchyNode(feature_index=2)
    root = HierarchyNode(
        feature_index=0,
        children=[child1, child2],
        mutually_exclusive_children=True,
    )

    repr_str = repr(root)
    assert "x" in repr_str  # Mutual exclusion marker


def test_HierarchyNode_requires_two_children_for_mutual_exclusion():
    child = HierarchyNode(feature_index=1)

    with pytest.raises(ValueError, match="Need at least 2 children"):
        HierarchyNode(
            feature_index=0,
            children=[child],
            mutually_exclusive_children=True,
        )


def test_hierarchy_modifier_with_activation_generator():
    """Integration test with ActivationGenerator."""
    from sae_lens.synthetic import ActivationGenerator

    child1 = HierarchyNode(feature_index=1)
    child2 = HierarchyNode(feature_index=2)
    root = HierarchyNode(
        feature_index=0,
        children=[child1, child2],
        mutually_exclusive_children=True,
    )
    modifier = hierarchy_modifier([root])

    generator = ActivationGenerator(
        num_features=3,
        firing_probabilities=torch.tensor([0.8, 0.5, 0.5]),
        modify_activations=modifier,
    )

    samples = generator.sample(batch_size=1000)

    # Check hierarchy: children inactive when parent inactive
    parent_inactive = samples[:, 0] == 0
    assert torch.all(samples[parent_inactive, 1] == 0)
    assert torch.all(samples[parent_inactive, 2] == 0)

    # Check mutual exclusion: never both active
    both_active = (samples[:, 1] > 0) & (samples[:, 2] > 0)
    assert torch.sum(both_active) == 0


def test_HierarchyNode_validate_valid_hierarchy():
    """Valid hierarchy should pass validation."""
    grandchild = HierarchyNode(feature_index=2)
    child1 = HierarchyNode(feature_index=1, children=[grandchild])
    child2 = HierarchyNode(feature_index=3)
    root = HierarchyNode(feature_index=0, children=[child1, child2])

    # Should not raise
    root.validate()


def test_HierarchyNode_validate_detects_loop():
    """Should detect when a node is its own ancestor."""
    child = HierarchyNode(feature_index=1)
    root = HierarchyNode(feature_index=0, children=[child])

    # Create a loop by making root a child of child
    child.children = [root]

    with pytest.raises(ValueError, match="Loop detected"):
        root.validate()


def test_HierarchyNode_validate_detects_self_loop():
    """Should detect when a node is its own child."""
    root = HierarchyNode(feature_index=0)
    root.children = [root]

    with pytest.raises(ValueError, match="Loop detected"):
        root.validate()


def test_HierarchyNode_validate_detects_multiple_parents():
    """Should detect when a node has multiple parents."""
    shared_child = HierarchyNode(feature_index=2)
    child1 = HierarchyNode(feature_index=1, children=[shared_child])
    child2 = HierarchyNode(feature_index=3, children=[shared_child])  # Same child!
    root = HierarchyNode(feature_index=0, children=[child1, child2])

    with pytest.raises(ValueError, match="multiple parents"):
        root.validate()


def test_HierarchyNode_validate_detects_node_as_sibling_of_itself():
    """Should detect when a node appears multiple times in the same children list."""
    child = HierarchyNode(feature_index=1)
    root = HierarchyNode(feature_index=0, children=[child, child])

    with pytest.raises(ValueError, match="multiple parents"):
        root.validate()


def test_HierarchyNode_validate_deep_loop():
    """Should detect loops in deep hierarchies."""
    node3 = HierarchyNode(feature_index=3)
    node2 = HierarchyNode(feature_index=2, children=[node3])
    node1 = HierarchyNode(feature_index=1, children=[node2])
    root = HierarchyNode(feature_index=0, children=[node1])

    # Create a deep loop: node3 -> root
    node3.children = [root]

    with pytest.raises(ValueError, match="Loop detected"):
        root.validate()


def test_HierarchyNode_validate_empty_hierarchy():
    """Single node hierarchy should be valid."""
    root = HierarchyNode(feature_index=0)
    root.validate()  # Should not raise


def test_HierarchyNode_validate_none_feature_index_nodes():
    """Validation should work with None feature_index nodes."""
    child1 = HierarchyNode(feature_index=0)
    child2 = HierarchyNode(feature_index=1)
    organizational = HierarchyNode(feature_index=None, children=[child1, child2])

    organizational.validate()  # Should not raise


# Tests for hierarchy_modifier


def test_hierarchy_modifier_empty_list_returns_identity():
    """Empty list should return identity function."""
    modifier = hierarchy_modifier([])
    activations = torch.randn(10, 5)
    result = modifier(activations)
    torch.testing.assert_close(result, activations)


def test_hierarchy_modifier_single_tree():
    """Single tree should work correctly."""
    child = HierarchyNode(feature_index=1)
    root = HierarchyNode(feature_index=0, children=[child])

    modifier = hierarchy_modifier([root])

    # Parent inactive - child should be deactivated
    activations = torch.tensor([[0.0, 1.0, 0.5]])
    result = modifier(activations)
    assert result[0, 1] == 0.0


def test_hierarchy_modifier_multiple_trees():
    """Multiple trees should all be applied."""
    # Tree 1: feature 0 -> feature 1
    tree1 = HierarchyNode(feature_index=0, children=[HierarchyNode(feature_index=1)])
    # Tree 2: feature 2 -> feature 3
    tree2 = HierarchyNode(feature_index=2, children=[HierarchyNode(feature_index=3)])

    modifier = hierarchy_modifier([tree1, tree2])

    # Both parents inactive - both children should be deactivated
    activations = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.5]])
    result = modifier(activations)

    assert result[0, 1] == 0.0  # child of tree1 deactivated
    assert result[0, 3] == 0.0  # child of tree2 deactivated
    assert result[0, 4] == 0.5  # unrelated feature unchanged


def test_hierarchy_modifier_validates_by_default():
    """Should validate hierarchies by default."""
    child = HierarchyNode(feature_index=1)
    root = HierarchyNode(feature_index=0, children=[child])
    # Create loop
    child.children = [root]

    with pytest.raises(ValueError, match="Loop detected"):
        hierarchy_modifier([root])


def test_hierarchy_modifier_detects_overlapping_features():
    """Should detect when same feature appears in multiple trees."""
    tree1 = HierarchyNode(feature_index=0, children=[HierarchyNode(feature_index=1)])
    tree2 = HierarchyNode(
        feature_index=2,
        children=[HierarchyNode(feature_index=1)],  # overlaps!
    )

    with pytest.raises(ValueError, match="appear in multiple hierarchy trees"):
        hierarchy_modifier([tree1, tree2])


def test_hierarchy_modifier_allows_disjoint_features():
    """Should allow multiple trees with disjoint feature indices."""
    tree1 = HierarchyNode(feature_index=0, children=[HierarchyNode(feature_index=1)])
    tree2 = HierarchyNode(feature_index=2, children=[HierarchyNode(feature_index=3)])

    # Should not raise
    modifier = hierarchy_modifier([tree1, tree2])
    assert callable(modifier)


def test_hierarchy_modifier_works_with_activation_generator():
    """Should integrate with ActivationGenerator."""
    from sae_lens.synthetic import ActivationGenerator

    tree = HierarchyNode(
        feature_index=0,
        children=[HierarchyNode(feature_index=1), HierarchyNode(feature_index=2)],
    )

    modifier = hierarchy_modifier([tree])

    gen = ActivationGenerator(
        num_features=5,
        firing_probabilities=0.5,
        modify_activations=modifier,
    )

    samples = gen.sample(100)
    assert samples.shape == (100, 5)

    # Check hierarchy is enforced: where parent is 0, children should be 0
    parent_inactive = samples[:, 0] == 0
    assert torch.all(samples[parent_inactive, 1] == 0)
    assert torch.all(samples[parent_inactive, 2] == 0)


def test_mutual_exclusion_statistical_distribution():
    """Verify mutual exclusion selects children with approximately uniform distribution."""
    child1 = HierarchyNode(feature_index=1)
    child2 = HierarchyNode(feature_index=2)
    root = HierarchyNode(
        feature_index=0,
        children=[child1, child2],
        mutually_exclusive_children=True,
    )
    modifier = hierarchy_modifier([root])

    # All samples have parent active and both children active
    n_samples = 2000
    activations = torch.ones(n_samples, 3)

    result = modifier(activations)

    # Count how often each child was kept
    child1_kept = (result[:, 1] > 0).sum().item()
    child2_kept = (result[:, 2] > 0).sum().item()

    # Verify mutual exclusion holds
    both_active = (result[:, 1] > 0) & (result[:, 2] > 0)
    assert torch.sum(both_active) == 0, "Both children should never be active"

    # Verify exactly one is kept per sample
    assert child1_kept + child2_kept == n_samples, "Exactly one child should be kept"

    # Statistical test: with 2000 samples and p=0.5, expect ~1000 each
    # Using a generous margin (4 standard deviations: sqrt(2000*0.5*0.5) * 4 ≈ 89)
    # This gives us a very low false positive rate while catching broken randomness
    expected = n_samples / 2
    margin = 120  # ~4 standard deviations, allows for statistical variation
    assert abs(child1_kept - expected) < margin, (
        f"Child 1 selected {child1_kept} times, expected ~{expected} "
        f"(within {margin}). Distribution may not be uniform."
    )
    assert abs(child2_kept - expected) < margin, (
        f"Child 2 selected {child2_kept} times, expected ~{expected} "
        f"(within {margin}). Distribution may not be uniform."
    )


def test_mutual_exclusion_three_or_more_children():
    """Verify mutual exclusion works with 3+ children."""
    children = [HierarchyNode(feature_index=i) for i in range(1, 5)]  # 4 children
    root = HierarchyNode(
        feature_index=0,
        children=children,
        mutually_exclusive_children=True,
    )
    modifier = hierarchy_modifier([root])

    # All samples have parent and all children active
    n_samples = 4000
    activations = torch.ones(n_samples, 5)

    result = modifier(activations)

    # Verify mutual exclusion: at most one child active per sample
    active_counts = (result[:, 1:5] > 0).sum(dim=1)
    assert torch.all(
        active_counts <= 1
    ), "At most one child should be active per sample"
    assert torch.all(
        active_counts == 1
    ), "Exactly one child should be active per sample"

    # Verify all children can be selected (each should appear at least sometimes)
    child_selections = [(result[:, i] > 0).sum().item() for i in range(1, 5)]
    for i, count in enumerate(child_selections):
        assert count > 0, f"Child {i+1} was never selected - randomness may be broken"

    # Statistical test: with 4 children and 4000 samples, expect ~1000 each
    expected = n_samples / 4
    margin = 150  # Allow for statistical variation
    for i, count in enumerate(child_selections):
        assert abs(count - expected) < margin, (
            f"Child {i+1} selected {count} times, expected ~{expected}. "
            f"Distribution may not be uniform."
        )


def test_mutual_exclusion_randomness_varies():
    """Verify that mutual exclusion produces different results on different calls."""
    child1 = HierarchyNode(feature_index=1)
    child2 = HierarchyNode(feature_index=2)
    root = HierarchyNode(
        feature_index=0,
        children=[child1, child2],
        mutually_exclusive_children=True,
    )
    modifier = hierarchy_modifier([root])

    # Run the same input multiple times
    activations = torch.ones(100, 3)

    results = []
    for _ in range(5):
        result = modifier(activations.clone())
        child1_count = (result[:, 1] > 0).sum().item()
        results.append(child1_count)

    # The results should not all be identical (would indicate broken randomness)
    # With 100 samples and 5 runs, the probability of all runs being identical
    # by chance is astronomically low
    unique_results = set(results)
    assert len(unique_results) > 1, (
        f"All 5 runs produced identical results ({results[0]} child1 selections). "
        "Randomness may be broken or deterministic."
    )


def test_multi_level_hierarchy_with_mutual_exclusion():
    """Verify hierarchy enforcement works correctly across multiple levels."""
    # Create a 3-level hierarchy:
    # Root (0) with mutual exclusion
    #   ├── Child A (1) with mutual exclusion
    #   │     ├── Grandchild A1 (3)
    #   │     └── Grandchild A2 (4)
    #   └── Child B (2) with mutual exclusion
    #         ├── Grandchild B1 (5)
    #         └── Grandchild B2 (6)

    grandchild_a1 = HierarchyNode(feature_index=3)
    grandchild_a2 = HierarchyNode(feature_index=4)
    grandchild_b1 = HierarchyNode(feature_index=5)
    grandchild_b2 = HierarchyNode(feature_index=6)

    child_a = HierarchyNode(
        feature_index=1,
        children=[grandchild_a1, grandchild_a2],
        mutually_exclusive_children=True,
    )
    child_b = HierarchyNode(
        feature_index=2,
        children=[grandchild_b1, grandchild_b2],
        mutually_exclusive_children=True,
    )
    root = HierarchyNode(
        feature_index=0,
        children=[child_a, child_b],
        mutually_exclusive_children=True,
    )

    modifier = hierarchy_modifier([root])

    # Test with all features initially active
    n_samples = 1000
    activations = torch.ones(n_samples, 7)
    result = modifier(activations)

    # 1. Root's children should be mutually exclusive
    both_children_active = (result[:, 1] > 0) & (result[:, 2] > 0)
    assert (
        both_children_active.sum() == 0
    ), "Root's children should be mutually exclusive"

    # 2. When Child A is active, its grandchildren should be mutually exclusive
    child_a_active = result[:, 1] > 0
    if child_a_active.any():
        a_grandchildren_both = (result[child_a_active, 3] > 0) & (
            result[child_a_active, 4] > 0
        )
        assert (
            a_grandchildren_both.sum() == 0
        ), "Child A's grandchildren should be exclusive"

    # 3. When Child B is active, its grandchildren should be mutually exclusive
    child_b_active = result[:, 2] > 0
    if child_b_active.any():
        b_grandchildren_both = (result[child_b_active, 5] > 0) & (
            result[child_b_active, 6] > 0
        )
        assert (
            b_grandchildren_both.sum() == 0
        ), "Child B's grandchildren should be exclusive"

    # 4. When Child A is inactive (because Child B was selected), its grandchildren should be 0
    child_a_inactive = result[:, 1] == 0
    assert torch.all(
        result[child_a_inactive, 3] == 0
    ), "Grandchild A1 should be 0 when Child A inactive"
    assert torch.all(
        result[child_a_inactive, 4] == 0
    ), "Grandchild A2 should be 0 when Child A inactive"

    # 5. When Child B is inactive (because Child A was selected), its grandchildren should be 0
    child_b_inactive = result[:, 2] == 0
    assert torch.all(
        result[child_b_inactive, 5] == 0
    ), "Grandchild B1 should be 0 when Child B inactive"
    assert torch.all(
        result[child_b_inactive, 6] == 0
    ), "Grandchild B2 should be 0 when Child B inactive"

    # 6. Verify distribution is reasonable (each path should be selected sometimes)
    child_a_count = child_a_active.sum().item()
    child_b_count = child_b_active.sum().item()
    assert (
        child_a_count > 100
    ), f"Child A selected only {child_a_count} times, expected ~500"
    assert (
        child_b_count > 100
    ), f"Child B selected only {child_b_count} times, expected ~500"


def test_multi_level_hierarchy_parent_deactivation_propagates():
    """Verify that parent deactivation propagates to all descendants."""
    # 4-level hierarchy without mutual exclusion
    # Root (0) -> Child (1) -> Grandchild (2) -> Great-grandchild (3)

    great_grandchild = HierarchyNode(feature_index=3)
    grandchild = HierarchyNode(feature_index=2, children=[great_grandchild])
    child = HierarchyNode(feature_index=1, children=[grandchild])
    root = HierarchyNode(feature_index=0, children=[child])

    modifier = hierarchy_modifier([root])

    # Test case 1: Root inactive - all descendants should be deactivated
    activations = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    result = modifier(activations)
    assert result[0, 0] == 0.0
    assert result[0, 1] == 0.0, "Child should be 0 when root inactive"
    assert result[0, 2] == 0.0, "Grandchild should be 0 when root inactive"
    assert result[0, 3] == 0.0, "Great-grandchild should be 0 when root inactive"

    # Test case 2: Root active, Child inactive - grandchildren should be deactivated
    activations = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    result = modifier(activations)
    assert result[0, 0] == 1.0
    assert result[0, 1] == 0.0
    assert result[0, 2] == 0.0, "Grandchild should be 0 when child inactive"
    assert result[0, 3] == 0.0, "Great-grandchild should be 0 when child inactive"

    # Test case 3: Root and Child active, Grandchild inactive
    activations = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    result = modifier(activations)
    assert result[0, 0] == 1.0
    assert result[0, 1] == 1.0
    assert result[0, 2] == 0.0
    assert result[0, 3] == 0.0, "Great-grandchild should be 0 when grandchild inactive"

    # Test case 4: All active - all should remain active
    activations = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    result = modifier(activations)
    assert torch.allclose(result, activations)
