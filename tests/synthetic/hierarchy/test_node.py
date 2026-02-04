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


class TestHierarchyNodeEquality:
    def test_equal_nodes_simple(self):
        node1 = HierarchyNode(feature_index=0)
        node2 = HierarchyNode(feature_index=0)
        assert node1 == node2

    def test_equal_nodes_with_children(self):
        child1 = HierarchyNode(feature_index=1)
        child2 = HierarchyNode(feature_index=2)
        node1 = HierarchyNode(feature_index=0, children=[child1, child2])

        child1_copy = HierarchyNode(feature_index=1)
        child2_copy = HierarchyNode(feature_index=2)
        node2 = HierarchyNode(feature_index=0, children=[child1_copy, child2_copy])

        assert node1 == node2

    def test_not_equal_different_type(self):
        node = HierarchyNode(feature_index=0)
        # Should return NotImplemented for non-HierarchyNode types
        result = node.__eq__("not a node")
        assert result is NotImplemented

        result = node.__eq__(42)
        assert result is NotImplemented

        result = node.__eq__(None)
        assert result is NotImplemented

    def test_not_equal_different_feature_index(self):
        node1 = HierarchyNode(feature_index=0)
        node2 = HierarchyNode(feature_index=1)
        assert node1 != node2

    def test_not_equal_different_mutually_exclusive(self):
        child1 = HierarchyNode(feature_index=1)
        child2 = HierarchyNode(feature_index=2)
        node1 = HierarchyNode(
            feature_index=0, children=[child1, child2], mutually_exclusive_children=True
        )

        child1_copy = HierarchyNode(feature_index=1)
        child2_copy = HierarchyNode(feature_index=2)
        node2 = HierarchyNode(
            feature_index=0,
            children=[child1_copy, child2_copy],
            mutually_exclusive_children=False,
        )

        assert node1 != node2

    def test_not_equal_different_feature_id(self):
        node1 = HierarchyNode(feature_index=0, feature_id="first")
        node2 = HierarchyNode(feature_index=0, feature_id="second")
        assert node1 != node2

    def test_not_equal_different_feature_id_vs_none(self):
        node1 = HierarchyNode(feature_index=0, feature_id="named")
        node2 = HierarchyNode(feature_index=0, feature_id=None)
        assert node1 != node2

    def test_not_equal_different_children_count(self):
        child1 = HierarchyNode(feature_index=1)
        child2 = HierarchyNode(feature_index=2)
        node1 = HierarchyNode(feature_index=0, children=[child1, child2])
        node2 = HierarchyNode(feature_index=0, children=[child1])
        assert node1 != node2

    def test_not_equal_children_differ(self):
        child1 = HierarchyNode(feature_index=1)
        child2 = HierarchyNode(feature_index=2)
        node1 = HierarchyNode(feature_index=0, children=[child1])
        node2 = HierarchyNode(feature_index=0, children=[child2])
        assert node1 != node2

    def test_equal_with_none_feature_index(self):
        child1 = HierarchyNode(feature_index=0)
        child2 = HierarchyNode(feature_index=1)
        org1 = HierarchyNode(feature_index=None, children=[child1, child2])

        child1_copy = HierarchyNode(feature_index=0)
        child2_copy = HierarchyNode(feature_index=1)
        org2 = HierarchyNode(feature_index=None, children=[child1_copy, child2_copy])

        assert org1 == org2

    def test_deep_equality(self):
        # Create identical deep trees
        gc1_a = HierarchyNode(feature_index=3)
        gc1_b = HierarchyNode(feature_index=4)
        c1_a = HierarchyNode(feature_index=1, children=[gc1_a, gc1_b])
        c2_a = HierarchyNode(feature_index=2)
        root_a = HierarchyNode(feature_index=0, children=[c1_a, c2_a])

        gc2_a = HierarchyNode(feature_index=3)
        gc2_b = HierarchyNode(feature_index=4)
        c1_b = HierarchyNode(feature_index=1, children=[gc2_a, gc2_b])
        c2_b = HierarchyNode(feature_index=2)
        root_b = HierarchyNode(feature_index=0, children=[c1_b, c2_b])

        assert root_a == root_b

    def test_deep_inequality_at_leaf(self):
        # Identical except for one leaf's feature_index
        gc1_a = HierarchyNode(feature_index=3)
        gc1_b = HierarchyNode(feature_index=4)
        c1_a = HierarchyNode(feature_index=1, children=[gc1_a, gc1_b])
        c2_a = HierarchyNode(feature_index=2)
        root_a = HierarchyNode(feature_index=0, children=[c1_a, c2_a])

        gc2_a = HierarchyNode(feature_index=3)
        gc2_b = HierarchyNode(feature_index=99)  # Different!
        c1_b = HierarchyNode(feature_index=1, children=[gc2_a, gc2_b])
        c2_b = HierarchyNode(feature_index=2)
        root_b = HierarchyNode(feature_index=0, children=[c1_b, c2_b])

        assert root_a != root_b
