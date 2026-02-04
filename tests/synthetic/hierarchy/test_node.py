import pytest

from sae_lens.synthetic import HierarchyNode


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


def test_HierarchyNode_validate_valid_hierarchy():
    grandchild = HierarchyNode(feature_index=2)
    child1 = HierarchyNode(feature_index=1, children=[grandchild])
    child2 = HierarchyNode(feature_index=3)
    root = HierarchyNode(feature_index=0, children=[child1, child2])

    # Should not raise
    root.validate()


def test_HierarchyNode_validate_detects_loop():
    child = HierarchyNode(feature_index=1)
    root = HierarchyNode(feature_index=0, children=[child])

    # Create a loop by making root a child of child
    child.children = [root]

    with pytest.raises(ValueError, match="Loop detected"):
        root.validate()


def test_HierarchyNode_validate_detects_self_loop():
    root = HierarchyNode(feature_index=0)
    root.children = [root]

    with pytest.raises(ValueError, match="Loop detected"):
        root.validate()


def test_HierarchyNode_validate_detects_multiple_parents():
    shared_child = HierarchyNode(feature_index=2)
    child1 = HierarchyNode(feature_index=1, children=[shared_child])
    child2 = HierarchyNode(feature_index=3, children=[shared_child])  # Same child!
    root = HierarchyNode(feature_index=0, children=[child1, child2])

    with pytest.raises(ValueError, match="multiple parents"):
        root.validate()


def test_HierarchyNode_validate_detects_node_as_sibling_of_itself():
    child = HierarchyNode(feature_index=1)
    root = HierarchyNode(feature_index=0, children=[child, child])

    with pytest.raises(ValueError, match="multiple parents"):
        root.validate()


def test_HierarchyNode_validate_deep_loop():
    node3 = HierarchyNode(feature_index=3)
    node2 = HierarchyNode(feature_index=2, children=[node3])
    node1 = HierarchyNode(feature_index=1, children=[node2])
    root = HierarchyNode(feature_index=0, children=[node1])

    # Create a deep loop: node3 -> root
    node3.children = [root]

    with pytest.raises(ValueError, match="Loop detected"):
        root.validate()


def test_HierarchyNode_validate_empty_hierarchy():
    root = HierarchyNode(feature_index=0)
    root.validate()  # Should not raise


def test_HierarchyNode_validate_none_feature_index_nodes():
    child1 = HierarchyNode(feature_index=0)
    child2 = HierarchyNode(feature_index=1)
    organizational = HierarchyNode(feature_index=None, children=[child1, child2])

    organizational.validate()  # Should not raise
