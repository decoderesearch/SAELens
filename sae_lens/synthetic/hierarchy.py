"""
Hierarchical feature modifier for activation generators.

This module provides HierarchyNode, which enforces hierarchical dependencies
on feature activations. Child features are deactivated when their parent is inactive,
and children can optionally be mutually exclusive.

Based on Noa Nabeshima's Matryoshka SAEs:
https://github.com/noanabeshima/matryoshka-saes/blob/main/toy_model.py
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

ActivationsModifier = Callable[[torch.Tensor], torch.Tensor]


@torch.no_grad()
def _validate_hierarchy(roots: Sequence[HierarchyNode]) -> None:
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


# ---------------------------------------------------------------------------
# Vectorized hierarchy implementation
# ---------------------------------------------------------------------------


@dataclass
class _MutualExclusionGroup:
    """A group of children that are mutually exclusive."""

    parent_feature_idx: int  # Parent's feature index, or -1 if organizational
    child_feature_indices: torch.Tensor  # [num_children] feature indices


@dataclass
class _PrecomputedHierarchy:
    """Precomputed data structures for vectorized hierarchy processing."""

    # Per-level tensors: level_data[level] = (feature_indices, parent_features)
    #   feature_indices: [num_nodes_at_level] feature indices of nodes at this level
    #   parent_features: [num_nodes_at_level] effective parent feature (-1 if always active)
    level_data: list[tuple[torch.Tensor, torch.Tensor]]

    # Mutual exclusion groups organized by level
    mutual_exclusion_groups_by_level: list[list[_MutualExclusionGroup]]


def _build_precomputed_hierarchy(
    roots: Sequence[HierarchyNode],
) -> _PrecomputedHierarchy:
    """
    Build precomputed data structures for vectorized hierarchy processing.

    Uses BFS to assign levels and compute effective parent features.
    """
    if not roots:
        return _PrecomputedHierarchy(level_data=[], mutual_exclusion_groups_by_level=[])

    # BFS to assign levels and collect node info
    # Each entry: (node, level, effective_parent_feature)
    # effective_parent_feature is the feature index of the nearest ancestor with a feature
    node_info: list[tuple[HierarchyNode, int, int]] = []
    mutual_exclusion_info: list[
        tuple[int, int, list[int]]
    ] = []  # (level, parent_feat, child_feats)

    # Queue entries: (node, level, effective_parent_feature)
    queue: deque[tuple[HierarchyNode, int, int]] = deque()

    # Initialize with roots
    for root in roots:
        # Root's effective parent is -1 (always active) unless it has a feature
        # Actually, root has no parent, so its effective_parent is -1
        # But we need to track what effective_parent to pass to children
        if root.feature_index is not None:
            # This root has a feature, children will use it as effective parent
            effective_for_children = root.feature_index
        else:
            # Organizational root, children inherit -1 (always active)
            effective_for_children = -1

        queue.append((root, 0, -1))  # Root's own effective parent is -1

        # Process mutual exclusion for this root
        if root.mutually_exclusive_children:
            child_feats = [
                c.feature_index for c in root.children if c.feature_index is not None
            ]
            if len(child_feats) >= 2:
                parent_feat = (
                    root.feature_index if root.feature_index is not None else -1
                )
                mutual_exclusion_info.append((0, parent_feat, child_feats))

    while queue:
        node, level, effective_parent = queue.popleft()

        # Record this node if it has a feature
        if node.feature_index is not None:
            node_info.append((node, level, effective_parent))

        # Determine effective parent for children
        if node.feature_index is not None:
            effective_for_children = node.feature_index
        else:
            effective_for_children = effective_parent

        # Add children to queue
        for child in node.children:
            queue.append((child, level + 1, effective_for_children))

        # Process mutual exclusion for this node's children (already done for roots above)
        if node.mutually_exclusive_children and level > 0:
            # This was not a root, so we need to process it now
            pass  # Actually we process all nodes including roots in the loop

    # Rebuild mutual exclusion info by re-traversing (simpler than tracking above)
    mutual_exclusion_info = []
    queue2: deque[tuple[HierarchyNode, int]] = deque()
    for root in roots:
        queue2.append((root, 0))

    while queue2:
        node, level = queue2.popleft()

        if node.mutually_exclusive_children:
            child_feats = [
                c.feature_index for c in node.children if c.feature_index is not None
            ]
            if len(child_feats) >= 2:
                parent_feat = (
                    node.feature_index if node.feature_index is not None else -1
                )
                mutual_exclusion_info.append((level, parent_feat, child_feats))

        for child in node.children:
            queue2.append((child, level + 1))

    # Group nodes by level
    max_level = max((info[1] for info in node_info), default=-1)

    level_data: list[tuple[torch.Tensor, torch.Tensor]] = []
    for level in range(max_level + 1):
        nodes_at_level = [(n, ep) for (n, lv, ep) in node_info if lv == level]
        if nodes_at_level:
            feature_indices = torch.tensor(
                [n.feature_index for (n, _) in nodes_at_level], dtype=torch.long
            )
            parent_features = torch.tensor(
                [ep for (_, ep) in nodes_at_level], dtype=torch.long
            )
            level_data.append((feature_indices, parent_features))
        else:
            # Empty level (can happen if all nodes at this level are organizational)
            level_data.append(
                (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
            )

    # Group mutual exclusion by level
    max_me_level = max((info[0] for info in mutual_exclusion_info), default=-1)
    # Ensure we have at least as many levels as level_data
    num_levels = max(len(level_data), max_me_level + 1)

    mutual_exclusion_groups_by_level: list[list[_MutualExclusionGroup]] = [
        [] for _ in range(num_levels)
    ]
    for level, parent_feat, child_feats in mutual_exclusion_info:
        group = _MutualExclusionGroup(
            parent_feature_idx=parent_feat,
            child_feature_indices=torch.tensor(child_feats, dtype=torch.long),
        )
        mutual_exclusion_groups_by_level[level].append(group)

    return _PrecomputedHierarchy(
        level_data=level_data,
        mutual_exclusion_groups_by_level=mutual_exclusion_groups_by_level,
    )


def _apply_mutual_exclusion_vectorized(
    activations: torch.Tensor,
    group: _MutualExclusionGroup,
) -> None:
    """Apply mutual exclusion using vectorized random-score selection."""
    batch_size = activations.shape[0]
    device = activations.device
    child_feats = group.child_feature_indices.to(device)
    num_children = len(child_feats)

    # Get parent active mask
    if group.parent_feature_idx >= 0:
        parent_active = activations[:, group.parent_feature_idx] > 0
    else:
        parent_active = torch.ones(batch_size, dtype=torch.bool, device=device)

    # Get children active mask: [batch_size, num_children]
    active_mask = activations[:, child_feats] > 0

    # Count active children where parent is active
    active_with_parent = active_mask & parent_active.unsqueeze(1)
    active_counts = active_with_parent.sum(dim=1)
    needs_exclusion = active_counts > 1

    if not needs_exclusion.any():
        return

    # Random-score selection: assign random scores, pick highest
    random_scores = torch.rand(batch_size, num_children, device=device)
    random_scores[~active_with_parent] = -float("inf")

    # Winner per sample
    winner_local_idx = random_scores.argmax(dim=1)

    # Vectorized deactivation using scatter
    should_deactivate = (
        needs_exclusion.unsqueeze(1)
        & active_mask
        & (torch.arange(num_children, device=device) != winner_local_idx.unsqueeze(1))
    )

    batch_idx, child_idx = torch.where(should_deactivate)
    feat_idx = child_feats[child_idx]
    activations[batch_idx, feat_idx] = 0


@torch.no_grad()
def hierarchy_modifier(
    roots: Sequence[HierarchyNode] | HierarchyNode,
) -> ActivationsModifier:
    """
    Create an activations modifier from one or more hierarchy trees.

    This is the recommended way to use hierarchies with ActivationGenerator.
    It validates the hierarchy structure and returns a modifier function that
    applies all hierarchy constraints.

    Args:
        roots: One or more root HierarchyNode objects. Each root defines an
            independent hierarchy tree. All trees are validated and applied.

    Returns:
        An ActivationsModifier function that can be passed to ActivationGenerator.

    Raises:
        ValueError: If validate=True and any hierarchy contains loops or
            nodes with multiple parents.
    """
    if not roots:
        # No hierarchies - return identity function
        def identity(activations: torch.Tensor) -> torch.Tensor:
            return activations

        return identity

    if isinstance(roots, HierarchyNode):
        roots = [roots]
    _validate_hierarchy(roots)

    # Precompute hierarchy structure for vectorized processing
    precomputed = _build_precomputed_hierarchy(roots)

    # Cache for device-specific tensors
    device_cache: dict[torch.device, _PrecomputedHierarchy] = {}

    def _get_precomputed_for_device(device: torch.device) -> _PrecomputedHierarchy:
        """Get or create device-specific precomputed hierarchy."""
        if device not in device_cache:
            # Move tensors to device
            level_data = [
                (feats.to(device), parents.to(device))
                for feats, parents in precomputed.level_data
            ]
            groups_by_level = [
                [
                    _MutualExclusionGroup(
                        parent_feature_idx=g.parent_feature_idx,
                        child_feature_indices=g.child_feature_indices.to(device),
                    )
                    for g in groups
                ]
                for groups in precomputed.mutual_exclusion_groups_by_level
            ]
            device_cache[device] = _PrecomputedHierarchy(
                level_data=level_data,
                mutual_exclusion_groups_by_level=groups_by_level,
            )
        return device_cache[device]

    def modifier(activations: torch.Tensor) -> torch.Tensor:
        result = activations.clone()
        device = result.device
        cached = _get_precomputed_for_device(device)

        # Process level by level
        for level, (level_features, level_parents) in enumerate(cached.level_data):
            # Vectorized parent deactivation for all nodes at this level
            has_parent = level_parents >= 0
            if has_parent.any():
                nodes_feats = level_features[has_parent]
                parent_feats = level_parents[has_parent]

                # Get parent activations: [batch_size, num_nodes_with_parent]
                parent_activations = result[:, parent_feats]
                parent_inactive = parent_activations <= 0

                # Find (batch, node) pairs to deactivate
                batch_idx, node_idx = torch.where(parent_inactive)
                feat_idx = nodes_feats[node_idx]

                # Single scatter operation
                result[batch_idx, feat_idx] = 0

            # Apply mutual exclusion for groups at this level
            if level < len(cached.mutual_exclusion_groups_by_level):
                for group in cached.mutual_exclusion_groups_by_level[level]:
                    _apply_mutual_exclusion_vectorized(result, group)

        return result

    return modifier


class HierarchyNode:
    """
    Represents a node in a feature hierarchy tree.

    Used to define hierarchical dependencies between features. Children are
    deactivated when their parent is inactive, and children can optionally
    be mutually exclusive.

    Use `hierarchy_modifier()` to create an ActivationsModifier from one or
    more HierarchyNode trees.


    Attributes:
        feature_index: Index of this feature in the activation tensor
        children: Child HierarchyNode nodes
        mutually_exclusive_children: If True, at most one child is active per sample
        feature_id: Optional identifier for debugging
    """

    children: Sequence[HierarchyNode]
    feature_index: int | None

    @classmethod
    def from_dict(cls, tree_dict: dict[str, Any]) -> HierarchyNode:
        """
        Create a HierarchyNode from a dictionary specification.

        Args:
            tree_dict: Dictionary with keys:

                - feature_index (optional): Index in the activation tensor
                - children (optional): List of child tree dictionaries
                - mutually_exclusive_children (optional): Whether children are exclusive
                - id (optional): Identifier for this node

        Returns:
            HierarchyNode instance
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
        children: Sequence[HierarchyNode] | None = None,
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

        if self.mutually_exclusive_children and len(self.children) < 2:
            raise ValueError("Need at least 2 children for mutual exclusion")

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

        # For each sample where parent is active, enforce mutual exclusion.
        # Note: This loop is not vectorized because we need to randomly select
        # which child to keep active per sample. Vectorizing would require either
        # a deterministic selection (losing randomness) or complex gather/scatter
        # operations that aren't more efficient for typical batch sizes.
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
        _validate_hierarchy([self])

    def __repr__(self, indent: int = 0) -> str:
        s = " " * (indent * 2)
        s += str(self.feature_index) if self.feature_index is not None else "-"
        s += "x" if self.mutually_exclusive_children else " "
        if self.feature_id:
            s += f" ({self.feature_id})"

        for child in self.children:
            s += "\n" + child.__repr__(indent + 2)
        return s
