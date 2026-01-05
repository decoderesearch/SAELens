"""
Hierarchical feature modifier for activation generators.

This module provides HierarchyNode, which enforces hierarchical dependencies
on feature activations. Child features are deactivated when their parent is inactive,
and children can optionally be mutually exclusive.

Based on Noa Nabeshima's Matryoshka SAEs:
https://github.com/noanabeshima/matryoshka-saes/blob/main/toy_model.py
"""

from collections.abc import Sequence
from typing import Any

import torch


class HierarchyNode:
    """
    Enforces hierarchical (tree) structure on feature activations.

    Works as an ActivationsModifier: takes a tensor of activations and returns
    a modified tensor where children are deactivated when parents are inactive.

    Example:
        >>> # Create a simple hierarchy:
        >>> # Feature 0 (root) -> Feature 1, Feature 2 (mutually exclusive)
        >>> child1 = HierarchyNode(feature_index=1)
        >>> child2 = HierarchyNode(feature_index=2)
        >>> root = HierarchyNode(
        ...     feature_index=0,
        ...     children=[child1, child2],
        ...     mutually_exclusive_children=True
        ... )
        >>> # Use directly as modifier
        >>> activations = torch.tensor([[1.0, 0.5, 0.3], [0.0, 0.5, 0.3]])
        >>> modified = root(activations)
        >>> # Row 0: root active, one child kept (mutual exclusion)
        >>> # Row 1: root inactive, both children deactivated

    Attributes:
        feature_index: Index of this feature in the activation tensor (None for non-readout nodes)
        children: Child HierarchyNode nodes
        mutually_exclusive_children: If True, at most one child is active per sample
        feature_id: Optional identifier for debugging
    """

    children: Sequence["HierarchyNode"]
    feature_index: int | None

    @classmethod
    def from_dict(cls, tree_dict: dict[str, Any]) -> "HierarchyNode":
        """
        Create a HierarchyNode from a dictionary specification.

        Args:
            tree_dict: Dictionary with keys:
                - feature_index (optional): Index in the activation tensor (None for non-readout)
                - children (optional): List of child tree dictionaries
                - mutually_exclusive_children (optional): Whether children are exclusive
                - id (optional): Identifier for this node

        Returns:
            HierarchyNode instance

        Example:
            >>> hierarchy = HierarchyNode.from_dict({
            ...     "feature_index": 0,
            ...     "children": [
            ...         {"feature_index": 1},
            ...         {"feature_index": 2}
            ...     ],
            ...     "mutually_exclusive_children": True
            ... })
        """
        children = [
            HierarchyNode.from_dict(child_dict)
            for child_dict in tree_dict.get("children", [])
        ]
        return cls(
            feature_index=tree_dict.get("feature_index"),
            children=children,
            mutually_exclusive_children=tree_dict.get(
                "mutually_exclusive_children", False
            ),
            feature_id=tree_dict.get("id"),
        )

    def __init__(
        self,
        feature_index: int | None = None,
        children: Sequence["HierarchyNode"] | None = None,
        mutually_exclusive_children: bool = False,
        feature_id: str | None = None,
    ):
        """
        Create a new HierarchyNode.

        Args:
            feature_index: Index of this feature in the activation tensor.
                Use None for organizational nodes that don't correspond to a feature.
            children: Child nodes that depend on this feature
            mutually_exclusive_children: If True, only one child can be active per sample
            feature_id: Optional identifier for debugging
        """
        self.feature_index = feature_index
        self.children = children or []
        self.mutually_exclusive_children = mutually_exclusive_children
        self.feature_id = feature_id

        if self.mutually_exclusive_children:
            assert (
                len(self.children) >= 2
            ), "Need at least 2 children for mutual exclusion"

    def __call__(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Apply hierarchical constraints to activations.

        Deactivates children when parents are inactive. For mutually exclusive
        children, randomly selects one active child when multiple are active.

        Args:
            activations: Tensor of shape [batch_size, num_features]

        Returns:
            Modified activations with hierarchical constraints applied
        """
        result = activations.clone()
        self._apply_hierarchy(result, parent_active_mask=None)
        return result

    def _apply_hierarchy(
        self,
        activations: torch.Tensor,
        parent_active_mask: torch.Tensor | None,
    ) -> None:
        """Recursively apply hierarchical constraints."""
        batch_size = activations.shape[0]

        # Determine which samples have this node active
        if self.feature_index is not None:
            is_active = activations[:, self.feature_index] > 0
        else:
            # Non-readout node: active if parent is active (or always if root)
            is_active = (
                parent_active_mask
                if parent_active_mask is not None
                else torch.ones(batch_size, dtype=torch.bool, device=activations.device)
            )

        # Deactivate this node if parent is inactive
        if parent_active_mask is not None and self.feature_index is not None:
            activations[~parent_active_mask, self.feature_index] = 0
            # Update is_active after deactivation
            is_active = activations[:, self.feature_index] > 0

        # Handle mutually exclusive children
        if self.mutually_exclusive_children and len(self.children) >= 2:
            self._enforce_mutual_exclusion(activations, is_active)

        # Recursively process children
        for child in self.children:
            child._apply_hierarchy(activations, parent_active_mask=is_active)

    def _enforce_mutual_exclusion(
        self,
        activations: torch.Tensor,
        parent_active_mask: torch.Tensor,
    ) -> None:
        """Ensure at most one child is active per sample."""
        batch_size = activations.shape[0]

        # Get indices of children that have feature indices
        child_indices = [
            child.feature_index
            for child in self.children
            if child.feature_index is not None
        ]

        if len(child_indices) < 2:
            return

        # For each sample where parent is active, enforce mutual exclusion
        for batch_idx in range(batch_size):
            if not parent_active_mask[batch_idx]:
                continue

            # Find which children are active
            active_children = [
                i
                for i, feat_idx in enumerate(child_indices)
                if activations[batch_idx, feat_idx] > 0
            ]

            if len(active_children) <= 1:
                continue

            # Randomly select one to keep active
            random_idx = int(torch.randint(len(active_children), (1,)).item())
            keep_idx = active_children[random_idx]

            # Deactivate all others
            for i, feat_idx in enumerate(child_indices):
                if i != keep_idx and i in active_children:
                    activations[batch_idx, feat_idx] = 0

    def get_all_feature_indices(self) -> list[int]:
        """Get all feature indices in this subtree."""
        indices = []
        if self.feature_index is not None:
            indices.append(self.feature_index)
        for child in self.children:
            indices.extend(child.get_all_feature_indices())
        return indices

    def validate(self) -> None:
        """
        Validate the hierarchy structure.

        Checks that:
        1. There are no loops (no node is its own ancestor)
        2. Each node has at most one parent (no node appears in multiple children lists)

        Raises:
            ValueError: If the hierarchy is invalid
        """
        # Check for loops and collect all nodes
        all_nodes: list[HierarchyNode] = []
        self._collect_nodes_and_check_loops(all_nodes, ancestors=set())

        # Check for multiple parents
        seen_ids: set[int] = set()
        for node in all_nodes:
            node_id = id(node)
            if node_id in seen_ids:
                node_desc = (
                    f"feature_index={node.feature_index}"
                    if node.feature_index is not None
                    else f"id={node.feature_id}" if node.feature_id else "unnamed node"
                )
                raise ValueError(
                    f"Node ({node_desc}) has multiple parents. "
                    "Each node must have at most one parent."
                )
            seen_ids.add(node_id)

    def _collect_nodes_and_check_loops(
        self,
        all_nodes: list["HierarchyNode"],
        ancestors: set[int],
    ) -> None:
        """Recursively collect nodes and check for loops."""
        node_id = id(self)

        if node_id in ancestors:
            node_desc = (
                f"feature_index={self.feature_index}"
                if self.feature_index is not None
                else f"id={self.feature_id}" if self.feature_id else "unnamed node"
            )
            raise ValueError(
                f"Loop detected: node ({node_desc}) is its own ancestor."
            )

        # Add to ancestors for children traversal
        new_ancestors = ancestors | {node_id}

        for child in self.children:
            # Collect child (before recursing, so we can detect multiple parents)
            all_nodes.append(child)
            child._collect_nodes_and_check_loops(all_nodes, new_ancestors)

    def __repr__(self, indent: int = 0) -> str:
        s = " " * (indent * 2)
        s += str(self.feature_index) if self.feature_index is not None else "-"
        s += "x" if self.mutually_exclusive_children else " "
        if self.feature_id:
            s += f" ({self.feature_id})"

        for child in self.children:
            s += "\n" + child.__repr__(indent + 2)
        return s
