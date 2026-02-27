"""Validation helpers for hierarchy structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sae_lens.synthetic.hierarchy.node import HierarchyNode


@torch.no_grad()
def validate_hierarchy(roots: Sequence[HierarchyNode]) -> None:
    """
    Validate a forest of hierarchy trees.

    Treats the input as children of a virtual root node and validates the
    entire structure.

    Checks that:
    1. There are no loops (no node is its own ancestor)
    2. Each node has at most one parent (no node appears in multiple children lists)
    3. No feature index appears in multiple trees

    Args:
        roots: Root nodes of the hierarchy trees to validate

    Raises:
        ValueError: If the hierarchy is invalid
    """
    if not roots:
        return

    # Collect all nodes and check for loops, treating roots as children of virtual root
    all_nodes: list[HierarchyNode] = []
    virtual_root_id = id(roots)  # Use the list itself as virtual root identity

    for root in roots:
        all_nodes.append(root)
        _collect_nodes_and_check_loops(root, all_nodes, ancestors={virtual_root_id})

    # Check for multiple parents (same node appearing multiple times)
    seen_ids: set[int] = set()
    for node in all_nodes:
        node_id = id(node)
        if node_id in seen_ids:
            node_desc = _node_description(node)
            raise ValueError(
                f"Node ({node_desc}) has multiple parents. "
                "Each node must have at most one parent."
            )
        seen_ids.add(node_id)

    # Check for overlapping feature indices across trees
    if len(roots) > 1:
        all_indices: set[int] = set()
        for root in roots:
            tree_indices = root.get_all_feature_indices()
            overlap = all_indices & set(tree_indices)
            if overlap:
                raise ValueError(
                    f"Feature indices {overlap} appear in multiple hierarchy trees. "
                    "Each feature should belong to at most one hierarchy."
                )
            all_indices.update(tree_indices)


def _collect_nodes_and_check_loops(
    node: HierarchyNode,
    all_nodes: list[HierarchyNode],
    ancestors: set[int],
) -> None:
    """Recursively collect nodes and check for loops."""
    node_id = id(node)

    if node_id in ancestors:
        node_desc = _node_description(node)
        raise ValueError(f"Loop detected: node ({node_desc}) is its own ancestor.")

    # Add to ancestors for children traversal
    new_ancestors = ancestors | {node_id}

    for child in node.children:
        # Collect child (before recursing, so we can detect multiple parents)
        all_nodes.append(child)
        _collect_nodes_and_check_loops(child, all_nodes, new_ancestors)


def _node_description(node: HierarchyNode) -> str:
    """Get a human-readable description of a node for error messages."""
    if node.feature_index is not None:
        return f"feature_index={node.feature_index}"
    if node.feature_id:
        return f"id={node.feature_id}"
    return "unnamed node"
