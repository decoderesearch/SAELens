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
