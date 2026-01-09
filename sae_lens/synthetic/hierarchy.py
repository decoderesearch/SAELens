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
class _LevelData:
    """Data for a single level in the hierarchy."""

    # Features at this level and their parents (for parent deactivation)
    features: torch.Tensor  # [num_features_at_level]
    parents: torch.Tensor  # [num_features_at_level]

    # ME group indices to process AFTER this level's parent deactivation
    # These are groups whose parent node is at this level
    # ME must be applied here before processing next level's parent deactivation
    me_group_indices: torch.Tensor  # [num_groups_at_level], may be empty


@dataclass
class _SparseHierarchyData:
    """Precomputed data for sparse hierarchy processing.

    This structure enables O(active_features) processing instead of O(all_groups).
    ME is applied at each level after parent deactivation to ensure cascading works.
    """

    # Per-level data for parent deactivation and ME (processed in order)
    level_data: list[_LevelData]

    # ME group data (shared across levels, indexed by me_group_indices)
    me_group_siblings: torch.Tensor  # [num_groups, max_siblings]
    me_group_sizes: torch.Tensor  # [num_groups]
    me_group_parents: (
        torch.Tensor
    )  # [num_groups] - parent feature index (-1 if no parent)

    # Total number of ME groups
    num_groups: int


def _build_sparse_hierarchy(
    roots: Sequence[HierarchyNode],
) -> _SparseHierarchyData:
    """
    Build sparse hierarchy data structure for O(active_features) processing.

    The key insight is that ME groups must be applied at the level of their parent node,
    AFTER parent deactivation at that level, but BEFORE processing the next level.
    This ensures that when a child is deactivated by ME, its grandchildren are also
    deactivated during the next level's parent deactivation.
    """
    # Collect feature info by level using BFS
    # Each entry: (feature_index, effective_parent, level)
    feature_info: list[tuple[int, int, int]] = []

    # ME groups: list of (parent_level, parent_feature, child_feature_indices)
    me_groups: list[tuple[int, int, list[int]]] = []

    # BFS queue: (node, effective_parent, level)
    queue: deque[tuple[HierarchyNode, int, int]] = deque()
    for root in roots:
        queue.append((root, -1, 0))

    while queue:
        node, effective_parent, level = queue.popleft()

        if node.feature_index is not None:
            feature_info.append((node.feature_index, effective_parent, level))
            new_effective_parent = node.feature_index
        else:
            new_effective_parent = effective_parent

        # Handle mutual exclusion children - record the parent's level and feature
        if node.mutually_exclusive_children and len(node.children) >= 2:
            child_feats = [
                c.feature_index for c in node.children if c.feature_index is not None
            ]
            if len(child_feats) >= 2:
                # ME group belongs to the parent's level (current level)
                # Parent feature is the node's feature_index (-1 if organizational node)
                parent_feat = (
                    node.feature_index if node.feature_index is not None else -1
                )
                me_groups.append((level, parent_feat, child_feats))

        for child in node.children:
            queue.append((child, new_effective_parent, level + 1))

    # Determine max level for both features and ME groups
    max_feature_level = max((info[2] for info in feature_info), default=-1)
    max_me_level = max((lvl for lvl, _, _ in me_groups), default=-1)
    max_level = max(max_feature_level, max_me_level)

    # Build level data with ME group indices per level
    level_data: list[_LevelData] = []

    # Group ME groups by their parent level
    me_groups_by_level: dict[int, list[int]] = {}
    for g_idx, (parent_level, _, _) in enumerate(me_groups):
        if parent_level not in me_groups_by_level:
            me_groups_by_level[parent_level] = []
        me_groups_by_level[parent_level].append(g_idx)

    for level in range(max_level + 1):
        # Get features at this level that have parents
        features_at_level = [
            (feat, parent) for feat, parent, lv in feature_info if lv == level
        ]
        with_parents = [(f, p) for f, p in features_at_level if p >= 0]

        if with_parents:
            feats = torch.tensor([f for f, _ in with_parents], dtype=torch.long)
            parents = torch.tensor([p for _, p in with_parents], dtype=torch.long)
        else:
            feats = torch.empty(0, dtype=torch.long)
            parents = torch.empty(0, dtype=torch.long)

        # Get ME group indices for this level
        if level in me_groups_by_level:
            me_indices = torch.tensor(me_groups_by_level[level], dtype=torch.long)
        else:
            me_indices = torch.empty(0, dtype=torch.long)

        level_data.append(
            _LevelData(features=feats, parents=parents, me_group_indices=me_indices)
        )

    # Build group siblings and parents tensors
    if me_groups:
        max_siblings = max(len(children) for _, _, children in me_groups)
        num_groups = len(me_groups)
        me_group_siblings = torch.full((num_groups, max_siblings), -1, dtype=torch.long)
        me_group_sizes = torch.zeros(num_groups, dtype=torch.long)
        me_group_parents = torch.full((num_groups,), -1, dtype=torch.long)
        for g_idx, (_, parent_feat, siblings) in enumerate(me_groups):
            me_group_sizes[g_idx] = len(siblings)
            me_group_parents[g_idx] = parent_feat
            me_group_siblings[g_idx, : len(siblings)] = torch.tensor(
                siblings, dtype=torch.long
            )
    else:
        me_group_siblings = torch.empty((0, 0), dtype=torch.long)
        me_group_sizes = torch.empty(0, dtype=torch.long)
        me_group_parents = torch.empty(0, dtype=torch.long)
        num_groups = 0

    return _SparseHierarchyData(
        level_data=level_data,
        me_group_siblings=me_group_siblings,
        me_group_sizes=me_group_sizes,
        me_group_parents=me_group_parents,
        num_groups=num_groups,
    )


def _apply_hierarchy_sparse(
    activations: torch.Tensor,
    sparse_data: _SparseHierarchyData,
) -> torch.Tensor:
    """
    Apply hierarchy constraints using precomputed sparse indices.

    Processes level by level:
    1. Apply parent deactivation for features at this level
    2. Apply mutual exclusion for groups whose parent is at this level
    3. Move to next level

    This ensures that ME at level L affects parent deactivation at level L+1.
    """
    result = activations.clone()

    # Data is already on correct device from cache
    me_group_siblings = sparse_data.me_group_siblings
    me_group_sizes = sparse_data.me_group_sizes
    me_group_parents = sparse_data.me_group_parents

    for level_data in sparse_data.level_data:
        # Step 1: Deactivate children where parent is inactive
        if level_data.features.numel() > 0:
            parent_vals = result[:, level_data.parents]
            child_vals = result[:, level_data.features]
            result[:, level_data.features] = child_vals * (parent_vals > 0)

        # Step 2: Apply ME for groups whose parent is at this level
        if level_data.me_group_indices.numel() > 0:
            _apply_me_for_groups(
                result,
                level_data.me_group_indices,
                me_group_siblings,
                me_group_sizes,
                me_group_parents,
            )

    return result


def _apply_me_for_groups(
    activations: torch.Tensor,
    group_indices: torch.Tensor,
    me_group_siblings: torch.Tensor,
    me_group_sizes: torch.Tensor,
    me_group_parents: torch.Tensor,
) -> None:
    """
    Apply mutual exclusion for the specified groups.

    Only processes groups where the parent is active (or has no parent).
    This is a key optimization since most groups are skipped when parent is inactive.

    Args:
        activations: [batch_size, num_features] - modified in place
        group_indices: [num_groups_to_process] - which groups to apply ME for
        me_group_siblings: [total_groups, max_siblings] - sibling indices per group
        me_group_sizes: [total_groups] - number of valid siblings per group
        me_group_parents: [total_groups] - parent feature index (-1 if no parent)
    """
    batch_size = activations.shape[0]
    device = activations.device
    num_groups = group_indices.numel()

    if num_groups == 0:
        return

    # Get parent indices for these groups
    parents = me_group_parents[group_indices]  # [num_groups]

    # Check which parents are active: [batch_size, num_groups]
    # Groups with parent=-1 are always active (root-level ME)
    has_parent = parents >= 0
    if has_parent.all():
        # All groups have parents - check their activation directly
        parent_active = activations[:, parents] > 0  # [batch, num_groups]
        if not parent_active.any():
            return
    elif has_parent.any():
        # Mixed case: some groups have parents, some don't
        # Use clamp to avoid indexing with -1 (reads feature 0, but result is masked out)
        safe_parents = parents.clamp(min=0)
        parent_active = activations[:, safe_parents] > 0  # [batch, num_groups]
        # Groups without parent are always "active"
        parent_active = parent_active | ~has_parent
    else:
        # No groups have parents - all are always active, skip parent check
        parent_active = None

    # Get siblings for the groups we're processing
    siblings = me_group_siblings[group_indices]  # [num_groups, max_siblings]
    sizes = me_group_sizes[group_indices]  # [num_groups]
    max_siblings = siblings.shape[1]

    # Get activations for all siblings: [batch_size, num_groups, max_siblings]
    safe_siblings = siblings.clamp(min=0)
    sibling_activations = activations[:, safe_siblings.view(-1)].view(
        batch_size, num_groups, max_siblings
    )

    # Create validity mask for padding: [num_groups, max_siblings]
    sibling_range = torch.arange(max_siblings, device=device)
    valid_mask = sibling_range < sizes.unsqueeze(1)

    # Find active valid siblings, but only where parent is active: [batch, groups, siblings]
    sibling_active = (sibling_activations > 0) & valid_mask
    if parent_active is not None:
        sibling_active = sibling_active & parent_active.unsqueeze(2)

    # Count active per group and check for conflicts: [batch_size, num_groups]
    active_counts = sibling_active.sum(dim=2)
    needs_exclusion = active_counts > 1

    if not needs_exclusion.any():
        return

    # Get (batch, group) pairs needing exclusion
    batch_with_conflict, groups_with_conflict = torch.where(needs_exclusion)
    num_conflicts = batch_with_conflict.numel()

    if num_conflicts == 0:
        return

    # Get siblings and activations for conflicts
    conflict_siblings = siblings[groups_with_conflict]  # [num_conflicts, max_siblings]
    conflict_active = sibling_active[
        batch_with_conflict, groups_with_conflict
    ]  # [num_conflicts, max_siblings]

    # Random selection for winner
    # Use -1e9 instead of -inf to avoid creating a tensor (torch.tensor(-float("inf")))
    # on every call. Since random scores are in [0,1], -1e9 is effectively -inf for argmax.
    random_scores = torch.rand(num_conflicts, max_siblings, device=device)
    random_scores[~conflict_active] = -1e9

    winner_idx = random_scores.argmax(dim=1)

    # Determine losers using scatter for efficiency
    is_winner = torch.zeros(
        num_conflicts, max_siblings, dtype=torch.bool, device=device
    )
    is_winner.scatter_(1, winner_idx.unsqueeze(1), True)
    should_deactivate = conflict_active & ~is_winner

    # Get (conflict, sibling) pairs to deactivate
    conflict_idx, sib_idx = torch.where(should_deactivate)

    if conflict_idx.numel() == 0:
        return

    # Map back to (batch, feature) and deactivate
    deact_batch = batch_with_conflict[conflict_idx]
    deact_feat = conflict_siblings[conflict_idx, sib_idx]
    activations[deact_batch, deact_feat] = 0


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

    # Build sparse hierarchy data
    sparse_data = _build_sparse_hierarchy(roots)

    # Cache for device-specific tensors
    device_cache: dict[torch.device, _SparseHierarchyData] = {}

    def _get_sparse_for_device(device: torch.device) -> _SparseHierarchyData:
        """Get or create device-specific sparse hierarchy data."""
        if device not in device_cache:
            device_cache[device] = _SparseHierarchyData(
                level_data=[
                    _LevelData(
                        features=ld.features.to(device),
                        parents=ld.parents.to(device),
                        me_group_indices=ld.me_group_indices.to(device),
                    )
                    for ld in sparse_data.level_data
                ],
                me_group_siblings=sparse_data.me_group_siblings.to(device),
                me_group_sizes=sparse_data.me_group_sizes.to(device),
                me_group_parents=sparse_data.me_group_parents.to(device),
                num_groups=sparse_data.num_groups,
            )
        return device_cache[device]

    def modifier(activations: torch.Tensor) -> torch.Tensor:
        device = activations.device
        cached = _get_sparse_for_device(device)
        return _apply_hierarchy_sparse(activations, cached)

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
