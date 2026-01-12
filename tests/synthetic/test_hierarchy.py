import pytest
import torch

from sae_lens.synthetic import HierarchyNode, hierarchy_modifier
from tests.helpers import to_dense, to_sparse


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


@pytest.mark.parametrize("use_sparse_tensors", [False, True])
def test_hierarchy_modifier_returns_correct_shape(use_sparse_tensors: bool):
    child = HierarchyNode(feature_index=1)
    root = HierarchyNode(feature_index=0, children=[child])
    modifier = hierarchy_modifier([root])

    activations = torch.rand(100, 3)
    if use_sparse_tensors:
        activations = to_sparse(activations)
    result = to_dense(modifier(activations))
    assert result.shape == (100, 3)


@pytest.mark.parametrize("use_sparse_tensors", [False, True])
def test_hierarchy_modifier_deactivates_children_when_parent_inactive(
    use_sparse_tensors: bool,
):
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
    if use_sparse_tensors:
        activations = to_sparse(activations)
    result = to_dense(modifier(activations))

    # Child should be deactivated when parent is inactive
    assert torch.all(result[:, 1] == 0)


@pytest.mark.parametrize("use_sparse_tensors", [False, True])
def test_hierarchy_modifier_keeps_children_when_parent_active(use_sparse_tensors: bool):
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
    original_child_vals = activations[:, 1].clone()
    if use_sparse_tensors:
        activations = to_sparse(activations)
    result = to_dense(modifier(activations))

    # Child values should be preserved when parent is active
    assert torch.allclose(result[:, 1], original_child_vals)


@pytest.mark.parametrize("use_sparse_tensors", [False, True])
def test_hierarchy_modifier_mixed_parent_states(use_sparse_tensors: bool):
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
    if use_sparse_tensors:
        activations = to_sparse(activations)
    result = to_dense(modifier(activations))

    assert result[0, 1] == 0.5  # Preserved
    assert result[1, 1] == 0.0  # Deactivated
    assert result[2, 1] == 0.0  # Already inactive


@pytest.mark.parametrize("use_sparse_tensors", [False, True])
def test_hierarchy_modifier_mutually_exclusive_children(use_sparse_tensors: bool):
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
    if use_sparse_tensors:
        activations = to_sparse(activations)

    result = to_dense(modifier(activations))

    # Both children should never be active simultaneously
    both_active = (result[:, 1] > 0) & (result[:, 2] > 0)
    assert torch.sum(both_active) == 0

    # At least one child should remain active (randomly selected)
    either_active = (result[:, 1] > 0) | (result[:, 2] > 0)
    assert torch.all(either_active)


@pytest.mark.parametrize("use_sparse_tensors", [False, True])
def test_hierarchy_modifier_mutually_exclusive_allows_single_child(
    use_sparse_tensors: bool,
):
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
    if use_sparse_tensors:
        activations = to_sparse(activations)

    result = to_dense(modifier(activations))

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


def test_mutual_exclusion_root_level_groups_no_parent():
    """Test ME groups at root level (parent=-1) work correctly.

    This exercises the mixed-parent code path in _apply_me_for_groups where
    some groups have parents and some don't. The implementation uses
    `safe_parents = parents.clamp(min=0)` which reads activations[:, 0] for
    root-level groups, but the result is masked out since these groups are
    always considered "active" regardless of feature 0's value.
    """
    # Create two independent root-level ME groups (no parent feature)
    # Tree 1: organizational root with ME children at features 0, 1
    # Tree 2: organizational root with ME children at features 2, 3
    tree1 = HierarchyNode(
        feature_index=None,  # organizational node, no parent feature
        children=[HierarchyNode(feature_index=0), HierarchyNode(feature_index=1)],
        mutually_exclusive_children=True,
    )
    tree2 = HierarchyNode(
        feature_index=None,
        children=[HierarchyNode(feature_index=2), HierarchyNode(feature_index=3)],
        mutually_exclusive_children=True,
    )

    modifier = hierarchy_modifier([tree1, tree2])

    # All features active - ME should still be enforced
    n_samples = 500
    activations = torch.ones(n_samples, 4)
    result = modifier(activations)

    # Tree 1: features 0 and 1 should be mutually exclusive
    both_01_active = (result[:, 0] > 0) & (result[:, 1] > 0)
    assert both_01_active.sum() == 0, "Features 0 and 1 should be mutually exclusive"

    # Tree 2: features 2 and 3 should be mutually exclusive
    both_23_active = (result[:, 2] > 0) & (result[:, 3] > 0)
    assert both_23_active.sum() == 0, "Features 2 and 3 should be mutually exclusive"

    # Each ME group should have exactly one active
    tree1_active = (result[:, 0] > 0) | (result[:, 1] > 0)
    tree2_active = (result[:, 2] > 0) | (result[:, 3] > 0)
    assert tree1_active.all(), "Tree 1 should always have one child active"
    assert tree2_active.all(), "Tree 2 should always have one child active"


def test_mutual_exclusion_mixed_root_and_nested_groups():
    """Test hierarchy with both root-level and nested ME groups.

    This exercises the code path where some ME groups have parent=-1 (root level)
    and others have valid parent indices, ensuring the safe_parents.clamp(min=0)
    logic handles the mixed case correctly.
    """
    # Tree 1: Root-level ME (parent=-1 for the ME group)
    root_me = HierarchyNode(
        feature_index=None,
        children=[HierarchyNode(feature_index=0), HierarchyNode(feature_index=1)],
        mutually_exclusive_children=True,
    )

    # Tree 2: Nested ME (parent=2 for the ME group)
    nested_me = HierarchyNode(
        feature_index=2,
        children=[HierarchyNode(feature_index=3), HierarchyNode(feature_index=4)],
        mutually_exclusive_children=True,
    )

    modifier = hierarchy_modifier([root_me, nested_me])

    n_samples = 500
    activations = torch.ones(n_samples, 5)
    result = modifier(activations)

    # Root-level ME: 0 and 1 mutually exclusive, always one active
    both_01 = (result[:, 0] > 0) & (result[:, 1] > 0)
    assert both_01.sum() == 0, "Root ME: 0 and 1 should be exclusive"
    either_01 = (result[:, 0] > 0) | (result[:, 1] > 0)
    assert either_01.all(), "Root ME: one of 0 or 1 should always be active"

    # Nested ME: 3 and 4 mutually exclusive when parent (2) is active
    parent_active = result[:, 2] > 0
    both_34 = (result[:, 3] > 0) & (result[:, 4] > 0)
    assert both_34.sum() == 0, "Nested ME: 3 and 4 should be exclusive"

    # When parent is active, one child should be active
    either_34_when_parent = ((result[:, 3] > 0) | (result[:, 4] > 0)) & parent_active
    assert (
        either_34_when_parent.sum() == parent_active.sum()
    ), "Nested ME: one of 3 or 4 should be active when parent 2 is active"


def test_me_fallback_path_variable_sibling_counts_single_level():
    """ME groups with different sizes at same level use fallback path.

    When ME groups at the same level have different numbers of siblings,
    the optimized reshape path cannot be used and _apply_me_for_groups is called.
    """
    # Level 0: Two parents, each with ME children of different sizes
    # Parent A (feature 0): 2 ME children (features 1, 2)
    # Parent B (feature 3): 3 ME children (features 4, 5, 6)
    # At level 1, ME groups have sizes [2, 3] -> fallback path

    child_a1 = HierarchyNode(feature_index=1)
    child_a2 = HierarchyNode(feature_index=2)
    parent_a = HierarchyNode(
        feature_index=0,
        children=[child_a1, child_a2],
        mutually_exclusive_children=True,
    )

    child_b1 = HierarchyNode(feature_index=4)
    child_b2 = HierarchyNode(feature_index=5)
    child_b3 = HierarchyNode(feature_index=6)
    parent_b = HierarchyNode(
        feature_index=3,
        children=[child_b1, child_b2, child_b3],
        mutually_exclusive_children=True,
    )

    modifier = hierarchy_modifier([parent_a, parent_b])

    n_samples = 1000
    activations = torch.ones(n_samples, 7)
    result = modifier(activations)

    # Parent A's children should be mutually exclusive
    both_a_active = (result[:, 1] > 0) & (result[:, 2] > 0)
    assert both_a_active.sum() == 0, "Parent A's children should be exclusive"
    either_a_active = (result[:, 1] > 0) | (result[:, 2] > 0)
    assert either_a_active.all(), "One of Parent A's children should be active"

    # Parent B's children should be mutually exclusive
    all_b_active = (
        (result[:, 4] > 0).int() + (result[:, 5] > 0).int() + (result[:, 6] > 0).int()
    )
    assert (
        all_b_active <= 1
    ).all(), "At most one of Parent B's children should be active"
    assert (
        all_b_active == 1
    ).all(), "Exactly one of Parent B's children should be active"

    # Statistical check: all children should be selected sometimes
    for idx in [1, 2]:
        count = (result[:, idx] > 0).sum().item()
        assert count > 100, f"Child {idx} selected only {count} times, expected ~500"

    for idx in [4, 5, 6]:
        count = (result[:, idx] > 0).sum().item()
        assert count > 100, f"Child {idx} selected only {count} times, expected ~333"


def test_me_fallback_path_variable_sibling_counts_multi_level():
    """ME groups with different sizes across multiple levels use fallback path.

    Tests a 3-level hierarchy where each level has ME groups of varying sizes.
    """
    # Level 0: Root with 2 ME children
    # Level 1: Child A has 2 ME grandchildren, Child B has 4 ME grandchildren
    # Level 2: Mixed grandchildren counts

    # Grandchildren for Child A (2 grandchildren)
    gc_a1 = HierarchyNode(feature_index=3)
    gc_a2 = HierarchyNode(feature_index=4)

    # Grandchildren for Child B (4 grandchildren)
    gc_b1 = HierarchyNode(feature_index=5)
    gc_b2 = HierarchyNode(feature_index=6)
    gc_b3 = HierarchyNode(feature_index=7)
    gc_b4 = HierarchyNode(feature_index=8)

    child_a = HierarchyNode(
        feature_index=1,
        children=[gc_a1, gc_a2],
        mutually_exclusive_children=True,
    )
    child_b = HierarchyNode(
        feature_index=2,
        children=[gc_b1, gc_b2, gc_b3, gc_b4],
        mutually_exclusive_children=True,
    )

    root = HierarchyNode(
        feature_index=0,
        children=[child_a, child_b],
        mutually_exclusive_children=True,
    )

    modifier = hierarchy_modifier([root])

    n_samples = 2000
    activations = torch.ones(n_samples, 9)
    result = modifier(activations)

    # Level 0 ME: children 1 and 2 should be exclusive
    both_children = (result[:, 1] > 0) & (result[:, 2] > 0)
    assert both_children.sum() == 0, "Root's children should be exclusive"

    # When Child A is active, its grandchildren (3, 4) should be exclusive
    child_a_active = result[:, 1] > 0
    if child_a_active.any():
        gc_a_both = (result[child_a_active, 3] > 0) & (result[child_a_active, 4] > 0)
        assert gc_a_both.sum() == 0, "Child A's grandchildren should be exclusive"
        gc_a_either = (result[child_a_active, 3] > 0) | (result[child_a_active, 4] > 0)
        assert gc_a_either.all(), "One of Child A's grandchildren should be active"

    # When Child B is active, its grandchildren (5, 6, 7, 8) should be exclusive
    child_b_active = result[:, 2] > 0
    if child_b_active.any():
        gc_b_count = (
            (result[child_b_active, 5] > 0).int()
            + (result[child_b_active, 6] > 0).int()
            + (result[child_b_active, 7] > 0).int()
            + (result[child_b_active, 8] > 0).int()
        )
        assert (gc_b_count <= 1).all(), "At most one of Child B's grandchildren active"
        assert (gc_b_count == 1).all(), "Exactly one of Child B's grandchildren active"

    # When Child A is inactive, its grandchildren should be 0
    child_a_inactive = result[:, 1] == 0
    assert (result[child_a_inactive, 3] == 0).all()
    assert (result[child_a_inactive, 4] == 0).all()

    # When Child B is inactive, its grandchildren should be 0
    child_b_inactive = result[:, 2] == 0
    assert (result[child_b_inactive, 5] == 0).all()
    assert (result[child_b_inactive, 6] == 0).all()
    assert (result[child_b_inactive, 7] == 0).all()
    assert (result[child_b_inactive, 8] == 0).all()


def test_me_fallback_path_non_contiguous_siblings():
    """ME with non-contiguous sibling indices uses fallback path.

    When ME siblings don't have consecutive feature indices, the optimized
    path cannot use reshape and must fall back to _apply_me_for_groups.
    """
    # ME children at non-contiguous indices: 0, 5 (gap of 4)
    child1 = HierarchyNode(feature_index=0)
    child2 = HierarchyNode(feature_index=5)
    root = HierarchyNode(
        feature_index=None,  # organizational root
        children=[child1, child2],
        mutually_exclusive_children=True,
    )

    modifier = hierarchy_modifier([root])

    n_samples = 1000
    activations = torch.ones(n_samples, 6)
    result = modifier(activations)

    # ME should still be enforced
    both_active = (result[:, 0] > 0) & (result[:, 5] > 0)
    assert both_active.sum() == 0, "Non-contiguous ME siblings should be exclusive"

    either_active = (result[:, 0] > 0) | (result[:, 5] > 0)
    assert either_active.all(), "One ME sibling should always be active"

    # Features in the gap should be unchanged
    assert (result[:, 1] == 1.0).all()
    assert (result[:, 2] == 1.0).all()
    assert (result[:, 3] == 1.0).all()
    assert (result[:, 4] == 1.0).all()


def test_me_fallback_path_non_contiguous_multi_level():
    """Non-contiguous ME siblings across multiple levels use fallback path."""
    # Level 1: Parent with ME children at indices 1, 10 (non-contiguous)
    # Level 2: Each child has ME grandchildren, also non-contiguous

    # Grandchildren for child at index 1 (non-contiguous: 2, 7)
    gc_1a = HierarchyNode(feature_index=2)
    gc_1b = HierarchyNode(feature_index=7)

    # Grandchildren for child at index 10 (non-contiguous: 11, 15)
    gc_2a = HierarchyNode(feature_index=11)
    gc_2b = HierarchyNode(feature_index=15)

    child1 = HierarchyNode(
        feature_index=1,
        children=[gc_1a, gc_1b],
        mutually_exclusive_children=True,
    )
    child2 = HierarchyNode(
        feature_index=10,
        children=[gc_2a, gc_2b],
        mutually_exclusive_children=True,
    )

    root = HierarchyNode(
        feature_index=0,
        children=[child1, child2],
        mutually_exclusive_children=True,
    )

    modifier = hierarchy_modifier([root])

    n_samples = 1000
    activations = torch.ones(n_samples, 16)
    result = modifier(activations)

    # Level 1 ME: children at 1 and 10 should be exclusive
    both_children = (result[:, 1] > 0) & (result[:, 10] > 0)
    assert both_children.sum() == 0, "Children 1 and 10 should be exclusive"

    # When child 1 is active, grandchildren 2 and 7 should be exclusive
    child1_active = result[:, 1] > 0
    if child1_active.any():
        gc1_both = (result[child1_active, 2] > 0) & (result[child1_active, 7] > 0)
        assert gc1_both.sum() == 0, "Grandchildren 2 and 7 should be exclusive"

    # When child 10 is active, grandchildren 11 and 15 should be exclusive
    child2_active = result[:, 10] > 0
    if child2_active.any():
        gc2_both = (result[child2_active, 11] > 0) & (result[child2_active, 15] > 0)
        assert gc2_both.sum() == 0, "Grandchildren 11 and 15 should be exclusive"

    # When child 1 is inactive, its grandchildren should be 0
    child1_inactive = result[:, 1] == 0
    assert (result[child1_inactive, 2] == 0).all()
    assert (result[child1_inactive, 7] == 0).all()

    # When child 10 is inactive, its grandchildren should be 0
    child2_inactive = result[:, 10] == 0
    assert (result[child2_inactive, 11] == 0).all()
    assert (result[child2_inactive, 15] == 0).all()


def test_me_mixed_optimized_and_fallback_paths():
    """Hierarchy where some levels use optimized path and others use fallback.

    Level 1: Uniform ME groups (2 children each) -> optimized path
    Level 2: Variable ME groups (2 vs 3 children) -> fallback path
    """
    # Level 2 grandchildren
    # Group under child_a: 2 grandchildren (contiguous)
    gc_a1 = HierarchyNode(feature_index=3)
    gc_a2 = HierarchyNode(feature_index=4)

    # Group under child_b: 3 grandchildren (contiguous)
    gc_b1 = HierarchyNode(feature_index=5)
    gc_b2 = HierarchyNode(feature_index=6)
    gc_b3 = HierarchyNode(feature_index=7)

    # Level 1 children (uniform: 2 children each for root)
    child_a = HierarchyNode(
        feature_index=1,
        children=[gc_a1, gc_a2],
        mutually_exclusive_children=True,
    )
    child_b = HierarchyNode(
        feature_index=2,
        children=[gc_b1, gc_b2, gc_b3],
        mutually_exclusive_children=True,
    )

    # Root with uniform ME at level 1
    root = HierarchyNode(
        feature_index=0,
        children=[child_a, child_b],
        mutually_exclusive_children=True,
    )

    modifier = hierarchy_modifier([root])

    n_samples = 2000
    activations = torch.ones(n_samples, 8)
    result = modifier(activations)

    # Level 1: uniform ME should work (optimized path)
    both_level1 = (result[:, 1] > 0) & (result[:, 2] > 0)
    assert both_level1.sum() == 0, "Level 1 ME should be enforced"

    # Level 2: variable ME should work (fallback path)
    child_a_active = result[:, 1] > 0
    child_b_active = result[:, 2] > 0

    # Child A's grandchildren (size 2)
    if child_a_active.any():
        gc_a_both = (result[child_a_active, 3] > 0) & (result[child_a_active, 4] > 0)
        assert gc_a_both.sum() == 0

    # Child B's grandchildren (size 3)
    if child_b_active.any():
        gc_b_count = (
            (result[child_b_active, 5] > 0).int()
            + (result[child_b_active, 6] > 0).int()
            + (result[child_b_active, 7] > 0).int()
        )
        assert (gc_b_count <= 1).all()

    # Verify cascading deactivation still works
    root_inactive = result[:, 0] == 0
    for idx in range(1, 8):
        assert (
            result[root_inactive, idx] == 0
        ).all(), f"Feature {idx} should be 0 when root inactive"


def test_me_fallback_non_contiguous_groups():
    """ME groups that are not laid out contiguously use fallback path.

    Even if each group has the same size and contiguous siblings within,
    if the groups themselves are not contiguous, fallback is used.
    """
    # Two ME groups at same level, each with 2 contiguous siblings
    # But groups are not contiguous: group 1 at [1,2], group 2 at [5,6]

    # Parent A with children at 1, 2
    child_a1 = HierarchyNode(feature_index=1)
    child_a2 = HierarchyNode(feature_index=2)
    parent_a = HierarchyNode(
        feature_index=0,
        children=[child_a1, child_a2],
        mutually_exclusive_children=True,
    )

    # Parent B with children at 5, 6 (gap from parent A's children)
    child_b1 = HierarchyNode(feature_index=5)
    child_b2 = HierarchyNode(feature_index=6)
    parent_b = HierarchyNode(
        feature_index=4,
        children=[child_b1, child_b2],
        mutually_exclusive_children=True,
    )

    # Organizational root to put both at same level
    root = HierarchyNode(
        feature_index=None,
        children=[parent_a, parent_b],
    )

    modifier = hierarchy_modifier([root])

    n_samples = 1000
    activations = torch.ones(n_samples, 7)
    result = modifier(activations)

    # Parent A's children should be exclusive
    both_a = (result[:, 1] > 0) & (result[:, 2] > 0)
    assert both_a.sum() == 0

    # Parent B's children should be exclusive
    both_b = (result[:, 5] > 0) & (result[:, 6] > 0)
    assert both_b.sum() == 0

    # Features 3 (in gap) should be unchanged
    assert (result[:, 3] == 1.0).all()

    # Verify parent deactivation still works
    parent_a_inactive = result[:, 0] == 0
    assert (result[parent_a_inactive, 1] == 0).all()
    assert (result[parent_a_inactive, 2] == 0).all()

    parent_b_inactive = result[:, 4] == 0
    assert (result[parent_b_inactive, 5] == 0).all()
    assert (result[parent_b_inactive, 6] == 0).all()


def test_hierarchy_modifier_large_hierarchy_performance():
    """Test hierarchy modifier performance with a large hierarchy (50k nodes).

    Creates a hierarchy with:
    - 2500 root trees, each with 20 nodes (depth 2)
    - Mix of ME and non-ME nodes to exercise all code paths
    - Verifies correctness and that it completes in reasonable time
    """
    import time

    num_features = 50_000
    feature_idx = 0
    trees = []

    # Create 2500 trees, each with 20 features
    # Structure: root -> 4 children (ME) -> 4 grandchildren each (ME)
    # That's 1 + 4 + 16 = 21 features per tree, but we'll use 20 for simplicity
    features_per_tree = 20
    num_trees = num_features // features_per_tree

    for _ in range(num_trees):
        if feature_idx >= num_features - features_per_tree:
            break

        # Create grandchildren (leaf level) - 4 groups of 4
        grandchildren_groups = []
        for _ in range(4):
            gc_group = []
            for _ in range(4):
                if feature_idx < num_features:
                    gc_group.append(HierarchyNode(feature_index=feature_idx))
                    feature_idx += 1
            grandchildren_groups.append(gc_group)

        # Create children with ME grandchildren
        children = []
        for gc_group in grandchildren_groups:
            if feature_idx < num_features and len(gc_group) >= 2:
                child = HierarchyNode(
                    feature_index=feature_idx,
                    children=gc_group,
                    mutually_exclusive_children=True,
                )
                children.append(child)
                feature_idx += 1

        # Create root with ME children
        if feature_idx < num_features and len(children) >= 2:
            root = HierarchyNode(
                feature_index=feature_idx,
                children=children,
                mutually_exclusive_children=True,
            )
            trees.append(root)
            feature_idx += 1

    actual_features = feature_idx
    assert (
        actual_features > 40_000
    ), f"Should have created ~50k features, got {actual_features}"

    # Time the modifier creation
    start = time.perf_counter()
    modifier = hierarchy_modifier(trees)
    creation_time = time.perf_counter() - start

    # Time the modifier application
    batch_size = 1000
    activations = torch.rand(batch_size, actual_features)
    activations = (activations > 0.5).float()  # Binary activations

    start = time.perf_counter()
    result = modifier(activations)
    apply_time = time.perf_counter() - start

    # Verify basic correctness
    assert result.shape == activations.shape

    # Verify hierarchy is enforced: spot check a few trees
    for tree in trees[:10]:
        root_idx = tree.feature_index
        root_inactive = result[:, root_idx] == 0

        # All descendants should be 0 when root is inactive
        for child in tree.children:
            child_idx = child.feature_index
            assert torch.all(
                result[root_inactive, child_idx] == 0
            ), f"Child {child_idx} should be 0 when root {root_idx} inactive"

    # Verify ME is enforced for roots with ME children
    for tree in trees[:10]:
        if tree.mutually_exclusive_children and len(tree.children) >= 2:
            root_idx = tree.feature_index
            root_active = result[:, root_idx] > 0

            child_indices = [c.feature_index for c in tree.children]
            children_active = result[:, child_indices] > 0

            # Count active children per sample where root is active
            active_counts = children_active[root_active].sum(dim=1)
            assert torch.all(
                active_counts <= 1
            ), "ME should enforce at most one child active"

    # Performance assertions (generous bounds for CI variability)
    assert (
        creation_time < 2.0
    ), f"Modifier creation took {creation_time:.2f}s, expected < 2s"
    assert (
        apply_time < 2.0
    ), f"Modifier application took {apply_time:.2f}s, expected < 2s"


class TestHierarchyModifierSparseCOO:
    def test_sparse_parent_deactivation(self):
        child = HierarchyNode(feature_index=1)
        root = HierarchyNode(feature_index=0, children=[child])
        modifier = hierarchy_modifier([root])

        # Create sparse COO tensor: parent inactive in all samples
        # Features: 0 (parent), 1 (child), 2 (unrelated)
        # Sample 0: child=1.0 active but parent=0 inactive -> child should be deactivated
        # Sample 1: child=0.8 active but parent=0 inactive -> child should be deactivated
        indices = torch.tensor([[0, 1], [1, 1]])  # (batch, feature) pairs
        values = torch.tensor([1.0, 0.8])
        sparse_input = torch.sparse_coo_tensor(indices, values, size=(2, 3))

        result = modifier(sparse_input)

        # Result should be sparse and child should be deactivated
        assert result.is_sparse
        dense_result = result.to_dense()
        assert torch.all(dense_result[:, 1] == 0)  # Child deactivated

    def test_sparse_keeps_children_when_parent_active(self):
        child = HierarchyNode(feature_index=1)
        root = HierarchyNode(feature_index=0, children=[child])
        modifier = hierarchy_modifier([root])

        # Parent active in both samples, child active
        indices = torch.tensor(
            [[0, 0, 1, 1], [0, 1, 0, 1]]
        )  # both parent and child active
        values = torch.tensor([1.0, 0.5, 0.8, 0.3])
        sparse_input = torch.sparse_coo_tensor(indices, values, size=(2, 3))

        result = modifier(sparse_input)

        assert result.is_sparse
        dense_result = result.to_dense()
        # Child should be preserved
        assert dense_result[0, 1] == 0.5
        assert dense_result[1, 1] == 0.3

    def test_sparse_mutual_exclusion(self):
        child1 = HierarchyNode(feature_index=1)
        child2 = HierarchyNode(feature_index=2)
        root = HierarchyNode(
            feature_index=0,
            children=[child1, child2],
            mutually_exclusive_children=True,
        )
        modifier = hierarchy_modifier([root])

        # Parent active, both children active -> ME should pick one
        indices = torch.tensor([[0, 0, 0], [0, 1, 2]])
        values = torch.tensor([1.0, 0.5, 0.3])
        sparse_input = torch.sparse_coo_tensor(indices, values, size=(1, 4))

        result = modifier(sparse_input)

        assert result.is_sparse
        dense_result = result.to_dense()
        # Exactly one of child1 or child2 should be active
        active_children = (dense_result[0, 1:3] > 0).sum()
        assert active_children == 1

    def test_sparse_matches_dense_result(self):
        child1 = HierarchyNode(feature_index=1)
        child2 = HierarchyNode(feature_index=2)
        grandchild = HierarchyNode(feature_index=3)
        child1 = HierarchyNode(feature_index=1, children=[grandchild])
        root = HierarchyNode(feature_index=0, children=[child1, child2])
        modifier = hierarchy_modifier([root])

        # Create dense activations
        dense_input = torch.tensor(
            [
                [1.0, 0.5, 0.3, 0.2],  # All active
                [0.0, 0.8, 0.4, 0.6],  # Root inactive
                [
                    1.0,
                    0.0,
                    0.5,
                    0.7,
                ],  # Child1 inactive (grandchild should be deactivated)
            ]
        )

        # Convert to sparse
        sparse_input = dense_input.to_sparse()

        dense_result = modifier(dense_input)
        sparse_result = modifier(sparse_input)

        # Results should match (convert sparse to dense for comparison)
        sparse_as_dense = sparse_result.to_dense()
        torch.testing.assert_close(dense_result, sparse_as_dense)

    def test_sparse_empty_tensor(self):
        child = HierarchyNode(feature_index=1)
        root = HierarchyNode(feature_index=0, children=[child])
        modifier = hierarchy_modifier([root])

        # Empty sparse tensor
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty(0)
        sparse_input = torch.sparse_coo_tensor(indices, values, size=(10, 5))

        result = modifier(sparse_input)

        assert result.is_sparse
        assert result._nnz() == 0
        assert result.shape == (10, 5)

    def test_sparse_no_hierarchy_features_active(self):
        child = HierarchyNode(feature_index=1)
        root = HierarchyNode(feature_index=0, children=[child])
        modifier = hierarchy_modifier([root])

        # Only unrelated features active (index 3, 4)
        indices = torch.tensor([[0, 0, 1], [3, 4, 3]])
        values = torch.tensor([1.0, 0.5, 0.8])
        sparse_input = torch.sparse_coo_tensor(indices, values, size=(2, 5))

        result = modifier(sparse_input)

        assert result.is_sparse
        dense_result = result.to_dense()
        # Unrelated features should be unchanged
        assert dense_result[0, 3] == 1.0
        assert dense_result[0, 4] == 0.5
        assert dense_result[1, 3] == 0.8

    def test_sparse_all_features_deactivated(self):
        # Create hierarchy: parent (0) -> children (1, 2)
        # Only children are active, parent is inactive
        # All children should be deactivated, resulting in empty tensor
        child1 = HierarchyNode(feature_index=1)
        child2 = HierarchyNode(feature_index=2)
        root = HierarchyNode(feature_index=0, children=[child1, child2])
        modifier = hierarchy_modifier([root])

        # Only children active (parent inactive) - all should be deactivated
        indices = torch.tensor([[0, 0, 1, 1], [1, 2, 1, 2]])
        values = torch.tensor([1.0, 0.5, 0.8, 0.3])
        sparse_input = torch.sparse_coo_tensor(indices, values, size=(2, 3))

        result = modifier(sparse_input)

        assert result.is_sparse
        # All features should be deactivated since parent is inactive
        assert result._nnz() == 0
        dense_result = result.to_dense()
        assert torch.all(dense_result == 0)

    def test_sparse_all_features_deactivated_multi_level(self):
        # Multi-level hierarchy: grandparent (0) -> parent (1) -> child (2)
        # Only child is active - should propagate up and deactivate everything
        grandchild = HierarchyNode(feature_index=2)
        child = HierarchyNode(feature_index=1, children=[grandchild])
        root = HierarchyNode(feature_index=0, children=[child])
        modifier = hierarchy_modifier([root])

        # Only grandchild active (parent and grandparent inactive)
        indices = torch.tensor([[0, 1], [2, 2]])
        values = torch.tensor([1.0, 0.5])
        sparse_input = torch.sparse_coo_tensor(indices, values, size=(2, 3))

        result = modifier(sparse_input)

        assert result.is_sparse
        # Grandchild should be deactivated because parent (1) is inactive
        assert result._nnz() == 0
        dense_result = result.to_dense()
        assert torch.all(dense_result == 0)

    def test_sparse_partial_deactivation_then_empty(self):
        # Test that processing continues correctly after partial deactivation
        # at one level leads to empty tensor at next level
        # Level 0: root (0) -> child1 (1), child2 (2)
        # Level 1: child1 (1) -> grandchild (3)
        # If only grandchild and child2 are active:
        # - child2 gets deactivated (root inactive)
        # - grandchild gets deactivated (child1 inactive)
        grandchild = HierarchyNode(feature_index=3)
        child1 = HierarchyNode(feature_index=1, children=[grandchild])
        child2 = HierarchyNode(feature_index=2)
        root = HierarchyNode(feature_index=0, children=[child1, child2])
        modifier = hierarchy_modifier([root])

        # Only child2 and grandchild active (root and child1 inactive)
        indices = torch.tensor([[0, 0], [2, 3]])
        values = torch.tensor([1.0, 0.5])
        sparse_input = torch.sparse_coo_tensor(indices, values, size=(1, 4))

        result = modifier(sparse_input)

        assert result.is_sparse
        # Both should be deactivated due to inactive parents
        assert result._nnz() == 0
