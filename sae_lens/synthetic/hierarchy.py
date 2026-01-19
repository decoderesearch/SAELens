"""
Hierarchical feature modifier for activation generators.

This module provides HierarchyNode, which enforces hierarchical dependencies
on feature activations. Child features are deactivated when their parent is inactive,
and children can optionally be mutually exclusive.

Based on Noa Nabeshima's Matryoshka SAEs:
https://github.com/noanabeshima/matryoshka-saes/blob/main/toy_model.py
"""

from __future__ import annotations

import random
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

    # Sparse COO support: Feature-to-parent mapping
    # feat_to_parent[f] = parent feature index, or -1 if root/no parent
    feat_to_parent: torch.Tensor | None = None  # [num_features]

    # Sparse COO support: Feature-to-ME-group mapping
    # feat_to_me_group[f] = group index, or -1 if not in any ME group
    feat_to_me_group: torch.Tensor | None = None  # [num_features]


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
            _LevelData(
                features=feats,
                parents=parents,
                me_group_indices=me_indices,
            )
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

    # Build sparse COO support: feat_to_parent and feat_to_me_group mappings
    # First determine num_features (max feature index + 1)
    all_features = [f for f, _, _ in feature_info]
    num_features = max(all_features) + 1 if all_features else 0

    # Build feature-to-parent mapping
    feat_to_parent = torch.full((num_features,), -1, dtype=torch.long)
    for feat, parent, _ in feature_info:
        feat_to_parent[feat] = parent

    # Build feature-to-ME-group mapping
    feat_to_me_group = torch.full((num_features,), -1, dtype=torch.long)
    for g_idx, (_, _, siblings) in enumerate(me_groups):
        for sib in siblings:
            feat_to_me_group[sib] = g_idx

    return _SparseHierarchyData(
        level_data=level_data,
        me_group_siblings=me_group_siblings,
        me_group_sizes=me_group_sizes,
        me_group_parents=me_group_parents,
        num_groups=num_groups,
        feat_to_parent=feat_to_parent,
        feat_to_me_group=feat_to_me_group,
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
    _INACTIVE_SCORE = -1e9
    random_scores = torch.rand(num_conflicts, max_siblings, device=device)
    random_scores[~conflict_active] = _INACTIVE_SCORE

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


# ---------------------------------------------------------------------------
# Sparse COO hierarchy implementation
# ---------------------------------------------------------------------------


def _apply_hierarchy_sparse_coo(
    sparse_tensor: torch.Tensor,
    sparse_data: _SparseHierarchyData,
) -> torch.Tensor:
    """
    Apply hierarchy constraints to a sparse COO tensor.

    This is the sparse analog of _apply_hierarchy_sparse. It processes
    level-by-level, applying parent deactivation then mutual exclusion.
    """
    if sparse_tensor._nnz() == 0:
        return sparse_tensor

    sparse_tensor = sparse_tensor.coalesce()

    for level_data in sparse_data.level_data:
        # Step 1: Apply parent deactivation for features at this level
        if level_data.features.numel() > 0:
            sparse_tensor = _apply_parent_deactivation_coo(
                sparse_tensor, level_data, sparse_data
            )

        # Step 2: Apply ME for groups whose parent is at this level
        if level_data.me_group_indices.numel() > 0:
            sparse_tensor = _apply_me_coo(
                sparse_tensor, level_data.me_group_indices, sparse_data
            )

    return sparse_tensor


def _apply_parent_deactivation_coo(
    sparse_tensor: torch.Tensor,
    level_data: _LevelData,
    sparse_data: _SparseHierarchyData,
) -> torch.Tensor:
    """
    Remove children from sparse COO tensor when their parent is inactive.

    Uses searchsorted for efficient membership testing of parent activity.
    """
    if sparse_tensor._nnz() == 0 or level_data.features.numel() == 0:
        return sparse_tensor

    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices()  # [2, nnz]
    values = sparse_tensor.values()  # [nnz]
    batch_indices = indices[0]
    feat_indices = indices[1]

    _, num_features = sparse_tensor.shape
    device = sparse_tensor.device
    nnz = indices.shape[1]

    # Build set of active (batch, feature) pairs for efficient lookup
    # Encode as: batch_idx * num_features + feat_idx
    active_pairs = batch_indices * num_features + feat_indices
    active_pairs_sorted, _ = active_pairs.sort()

    # Use the precomputed feat_to_parent mapping
    assert sparse_data.feat_to_parent is not None
    hierarchy_num_features = sparse_data.feat_to_parent.numel()

    # Handle features outside the hierarchy (they have no parent, pass through)
    in_hierarchy = feat_indices < hierarchy_num_features
    parent_of_feat = torch.full((nnz,), -1, dtype=torch.long, device=device)
    parent_of_feat[in_hierarchy] = sparse_data.feat_to_parent[
        feat_indices[in_hierarchy]
    ]

    # Find entries that have a parent (parent >= 0 means this feature has a parent)
    has_parent = parent_of_feat >= 0

    if not has_parent.any():
        return sparse_tensor

    # For entries with parents, check if parent is active
    child_entry_indices = torch.where(has_parent)[0]
    child_batch = batch_indices[has_parent]
    child_parents = parent_of_feat[has_parent]

    # Look up parent activity using searchsorted
    parent_pairs = child_batch * num_features + child_parents
    search_pos = torch.searchsorted(active_pairs_sorted, parent_pairs)
    search_pos = search_pos.clamp(max=active_pairs_sorted.numel() - 1)
    parent_active = active_pairs_sorted[search_pos] == parent_pairs

    # Handle empty case
    if active_pairs_sorted.numel() == 0:
        parent_active = torch.zeros_like(parent_pairs, dtype=torch.bool)

    # Build keep mask: keep entry if it's a root OR its parent is active
    keep_mask = torch.ones(nnz, dtype=torch.bool, device=device)
    keep_mask[child_entry_indices[~parent_active]] = False

    if keep_mask.all():
        return sparse_tensor

    return torch.sparse_coo_tensor(
        indices[:, keep_mask],
        values[keep_mask],
        sparse_tensor.shape,
        device=device,
        dtype=sparse_tensor.dtype,
    )


def _apply_me_coo(
    sparse_tensor: torch.Tensor,
    group_indices: torch.Tensor,
    sparse_data: _SparseHierarchyData,
) -> torch.Tensor:
    """
    Apply mutual exclusion to sparse COO tensor.

    For each ME group with multiple active siblings in the same batch,
    randomly selects one winner and removes the rest.
    """
    if sparse_tensor._nnz() == 0 or group_indices.numel() == 0:
        return sparse_tensor

    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices()  # [2, nnz]
    values = sparse_tensor.values()  # [nnz]
    batch_indices = indices[0]
    feat_indices = indices[1]

    _, num_features = sparse_tensor.shape
    device = sparse_tensor.device
    nnz = indices.shape[1]

    # Use precomputed feat_to_me_group mapping
    assert sparse_data.feat_to_me_group is not None
    hierarchy_num_features = sparse_data.feat_to_me_group.numel()

    # Handle features outside the hierarchy (they are not in any ME group)
    in_hierarchy = feat_indices < hierarchy_num_features
    me_group_of_feat = torch.full((nnz,), -1, dtype=torch.long, device=device)
    me_group_of_feat[in_hierarchy] = sparse_data.feat_to_me_group[
        feat_indices[in_hierarchy]
    ]

    # Find entries that belong to ME groups we're processing (vectorized)
    in_relevant_group = torch.isin(me_group_of_feat, group_indices)

    if not in_relevant_group.any():
        return sparse_tensor

    # Get the ME entries
    me_entry_indices = torch.where(in_relevant_group)[0]
    me_batch = batch_indices[in_relevant_group]
    me_group = me_group_of_feat[in_relevant_group]

    # Check parent activity for ME groups (only apply ME if parent is active)
    me_group_parents = sparse_data.me_group_parents[me_group]
    has_parent = me_group_parents >= 0

    if has_parent.any():
        # Build active pairs for parent lookup
        active_pairs = batch_indices * num_features + feat_indices
        active_pairs_sorted, _ = active_pairs.sort()

        parent_pairs = (
            me_batch[has_parent] * num_features + me_group_parents[has_parent]
        )
        search_pos = torch.searchsorted(active_pairs_sorted, parent_pairs)
        search_pos = search_pos.clamp(max=active_pairs_sorted.numel() - 1)
        parent_active_for_has_parent = active_pairs_sorted[search_pos] == parent_pairs

        # Build full parent_active mask
        parent_active = torch.ones(
            me_entry_indices.numel(), dtype=torch.bool, device=device
        )
        parent_active[has_parent] = parent_active_for_has_parent

        # Filter to only ME entries where parent is active
        valid_me = parent_active
        me_entry_indices = me_entry_indices[valid_me]
        me_batch = me_batch[valid_me]
        me_group = me_group[valid_me]

    if me_entry_indices.numel() == 0:
        return sparse_tensor

    # Encode (batch, group) pairs
    num_groups = sparse_data.num_groups
    batch_group_pairs = me_batch * num_groups + me_group

    # Find unique (batch, group) pairs and count occurrences
    unique_bg, inverse, counts = torch.unique(
        batch_group_pairs, return_inverse=True, return_counts=True
    )

    # Only process pairs with count > 1 (conflicts)
    has_conflict = counts > 1

    if not has_conflict.any():
        return sparse_tensor

    # For efficiency, we process all conflicts together
    # Assign random scores to each ME entry
    random_scores = torch.rand(me_entry_indices.numel(), device=device)

    # For each (batch, group) pair, we want the entry with highest score to be winner
    # Use scatter_reduce to find max score per (batch, group)
    bg_to_dense = torch.zeros(unique_bg.numel(), dtype=torch.long, device=device)
    bg_to_dense[has_conflict.nonzero(as_tuple=True)[0]] = torch.arange(
        has_conflict.sum(), device=device
    )

    # Map each ME entry to its dense conflict index
    entry_has_conflict = has_conflict[inverse]

    if not entry_has_conflict.any():
        return sparse_tensor

    conflict_entries_mask = entry_has_conflict
    conflict_entry_indices = me_entry_indices[conflict_entries_mask]
    conflict_random_scores = random_scores[conflict_entries_mask]
    conflict_inverse = inverse[conflict_entries_mask]
    conflict_dense_idx = bg_to_dense[conflict_inverse]

    # Vectorized winner selection using sorting
    # Sort entries by (group_idx, -random_score) so highest score comes first per group
    # Use group * 2 - score to sort by group ascending, then score descending
    sort_keys = conflict_dense_idx.float() * 2.0 - conflict_random_scores
    sorted_order = sort_keys.argsort()
    sorted_dense_idx = conflict_dense_idx[sorted_order]

    # Find first entry of each group in sorted order (these are winners)
    group_starts = torch.cat(
        [
            torch.tensor([True], device=device),
            sorted_dense_idx[1:] != sorted_dense_idx[:-1],
        ]
    )

    # Winners are entries at group starts in sorted order
    winner_positions_in_sorted = torch.where(group_starts)[0]
    winner_original_positions = sorted_order[winner_positions_in_sorted]

    # Create winner mask (vectorized)
    is_winner = torch.zeros(
        conflict_entry_indices.numel(), dtype=torch.bool, device=device
    )
    is_winner[winner_original_positions] = True

    # Build keep mask (vectorized)
    keep_mask = torch.ones(nnz, dtype=torch.bool, device=device)
    loser_entry_indices = conflict_entry_indices[~is_winner]
    keep_mask[loser_entry_indices] = False

    if keep_mask.all():
        return sparse_tensor

    return torch.sparse_coo_tensor(
        indices[:, keep_mask],
        values[keep_mask],
        sparse_tensor.shape,
        device=device,
        dtype=sparse_tensor.dtype,
    )


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
                feat_to_parent=(
                    sparse_data.feat_to_parent.to(device)
                    if sparse_data.feat_to_parent is not None
                    else None
                ),
                feat_to_me_group=(
                    sparse_data.feat_to_me_group.to(device)
                    if sparse_data.feat_to_me_group is not None
                    else None
                ),
            )
        return device_cache[device]

    def modifier(activations: torch.Tensor) -> torch.Tensor:
        device = activations.device
        cached = _get_sparse_for_device(device)
        if activations.is_sparse:
            return _apply_hierarchy_sparse_coo(activations, cached)
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HierarchyNode):
            return NotImplemented
        if self.feature_index != other.feature_index:
            return False
        if self.mutually_exclusive_children != other.mutually_exclusive_children:
            return False
        if self.feature_id != other.feature_id:
            return False
        if len(self.children) != len(other.children):
            return False
        return all(a == b for a, b in zip(self.children, other.children, strict=True))


@dataclass
class HierarchyConfig:
    """
    Configuration for automatic hierarchy generation.

    Creates a forest of hierarchy trees with controlled structure.

    Attributes:
        total_root_nodes: Number of root nodes (trees).
        branching_factor: Number of children per parent. Can be:

            - An int for fixed branching (e.g., 3 means exactly 3 children)
            - A tuple (min, max) for random branching in that range

        max_depth: Maximum depth of hierarchy trees. Depth 1 = parent with leaves.
        mutually_exclusive_portion: Fraction of eligible parent nodes whose children
            should be mutually exclusive (0.0 to 1.0). Default 0.0.
        mutually_exclusive_min_depth: Minimum depth at which ME can be applied.
            Depth 0 = root nodes. Default 0.
        mutually_exclusive_max_depth: Maximum depth at which ME can be applied.
            None means no upper limit. Default None.
        compensate_probabilities: If True, scale up firing probabilities to compensate
            for the probability reduction caused by hierarchy constraints. When children
            can only fire when parents fire, effective probability becomes the product
            of all ancestor probabilities. This option corrects for that. Default False.
    """

    total_root_nodes: int = 100
    branching_factor: int | tuple[int, int] = 100
    max_depth: int = 2
    mutually_exclusive_portion: float = 0.0
    mutually_exclusive_min_depth: int = 0
    mutually_exclusive_max_depth: int | None = None
    compensate_probabilities: bool = False

    def __post_init__(self) -> None:
        if self.total_root_nodes <= 0:
            raise ValueError("total_root_nodes must be positive")
        if isinstance(self.branching_factor, int):
            if self.branching_factor < 2:
                raise ValueError("branching_factor must be at least 2")
        else:
            if self.branching_factor[0] < 2:
                raise ValueError("branching_factor minimum must be at least 2")
            if self.branching_factor[0] > self.branching_factor[1]:
                raise ValueError("branching_factor[0] must be <= branching_factor[1]")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if not 0.0 <= self.mutually_exclusive_portion <= 1.0:
            raise ValueError("mutually_exclusive_portion must be between 0.0 and 1.0")
        if self.mutually_exclusive_min_depth < 0:
            raise ValueError("mutually_exclusive_min_depth must be non-negative")
        if (
            self.mutually_exclusive_max_depth is not None
            and self.mutually_exclusive_max_depth < self.mutually_exclusive_min_depth
        ):
            raise ValueError(
                "mutually_exclusive_max_depth must be >= mutually_exclusive_min_depth"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary."""
        branching = (
            self.branching_factor
            if isinstance(self.branching_factor, int)
            else list(self.branching_factor)
        )
        return {
            "total_root_nodes": self.total_root_nodes,
            "branching_factor": branching,
            "max_depth": self.max_depth,
            "mutually_exclusive_portion": self.mutually_exclusive_portion,
            "mutually_exclusive_min_depth": self.mutually_exclusive_min_depth,
            "mutually_exclusive_max_depth": self.mutually_exclusive_max_depth,
            "compensate_probabilities": self.compensate_probabilities,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HierarchyConfig:
        """Deserialize config from dictionary."""
        d = dict(d)  # Make a copy
        # Convert list to tuple (JSON doesn't have tuples)
        if "branching_factor" in d and isinstance(d["branching_factor"], list):
            d["branching_factor"] = tuple(d["branching_factor"])
        # Remove deprecated seed field if present (for backward compatibility)
        d.pop("seed", None)
        return cls(**d)


@dataclass
class Hierarchy:
    """Result of hierarchy generation."""

    roots: list[HierarchyNode]
    modifier: ActivationsModifier | None

    @property
    def feature_indices_used(self) -> set[int]:
        """Compute set of feature indices used in the hierarchy."""
        indices: set[int] = set()

        def collect_indices(nodes: list[HierarchyNode]) -> None:
            for node in nodes:
                if node.feature_index is not None:
                    indices.add(node.feature_index)
                collect_indices(list(node.children))

        collect_indices(self.roots)
        return indices

    def to_dict(self) -> dict[str, Any]:
        """Serialize hierarchy to dict for persistence."""

        def node_to_dict(node: HierarchyNode) -> dict[str, Any]:
            return {
                "feature_index": node.feature_index,
                "mutually_exclusive_children": node.mutually_exclusive_children,
                "feature_id": node.feature_id,
                "children": [node_to_dict(c) for c in node.children],
            }

        return {
            "roots": [node_to_dict(r) for r in self.roots],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Hierarchy:
        """Deserialize hierarchy from dict."""

        def dict_to_node(node_dict: dict[str, Any]) -> HierarchyNode:
            children = [dict_to_node(c) for c in node_dict.get("children", [])]
            return HierarchyNode(
                feature_index=node_dict.get("feature_index"),
                children=children,
                mutually_exclusive_children=node_dict.get(
                    "mutually_exclusive_children", False
                ),
                feature_id=node_dict.get("feature_id"),
            )

        roots = [dict_to_node(r) for r in d["roots"]]
        modifier = hierarchy_modifier(roots) if roots else None
        return cls(roots=roots, modifier=modifier)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hierarchy):
            return NotImplemented
        if len(self.roots) != len(other.roots):
            return False
        return all(a == b for a, b in zip(self.roots, other.roots, strict=True))

    def compute_probability_correction_factors(
        self,
        base_firing_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute correction factors to compensate for hierarchy probability reduction.

        When hierarchy is enabled, children can only fire when their parents fire.
        This reduces effective firing probabilities. The correction factor for each
        feature compensates for this reduction.

        For mutually exclusive children, the correction also accounts for the
        probability of winning the ME selection (when multiple siblings are active,
        one is randomly chosen).

        When all features are sampled with corrected probabilities and hierarchy
        is applied, the effective firing rate for each feature equals its base
        probability.

        Args:
            base_firing_probabilities: Original firing probabilities of shape
                (num_features,)

        Returns:
            Tensor of shape (num_features,) where correction_factors[i] compensates
            for both parent deactivation and ME selection. Features not in the
            hierarchy (roots and features outside any tree) get correction factor
            of 1.0.
        """
        num_features = base_firing_probabilities.shape[0]
        correction_factors = torch.ones(
            num_features, dtype=base_firing_probabilities.dtype
        )

        # Track which features are in ME groups (their corrections are set by _adjust_me_corrections)
        me_features: set[int] = set()

        def traverse(node: HierarchyNode, parent_base_prob: float) -> None:
            if node.feature_index is not None:
                # Only set basic correction if not in an ME group
                if node.feature_index not in me_features and parent_base_prob > 0:
                    correction_factors[node.feature_index] = 1.0 / parent_base_prob
                node_prob = base_firing_probabilities[node.feature_index].item()
            else:
                # Organizational node without feature_index - children see parent's prob
                node_prob = parent_base_prob

            # Adjust for ME groups - this sets corrections for ME children
            if node.mutually_exclusive_children and len(node.children) >= 2:
                me_indices = [
                    c.feature_index
                    for c in node.children
                    if c.feature_index is not None
                ]
                if len(me_indices) >= 2:
                    # Mark these features as ME so traverse won't overwrite their corrections
                    me_features.update(me_indices)
                    _adjust_me_corrections(
                        me_indices,
                        node_prob,
                        base_firing_probabilities,
                        correction_factors,
                    )

            for child in node.children:
                traverse(child, node_prob)

        for root in self.roots:
            traverse(root, 1.0)

        return correction_factors


def _adjust_me_corrections(
    sibling_indices: list[int],
    parent_base_prob: float,
    base_probs: torch.Tensor,
    correction_factors: torch.Tensor,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> None:
    """
    Adjust correction factors for an ME group using iterative refinement.

    For ME siblings, P(child fires after ME) = corrected × parent_prob × win_prob,
    where win_prob is the probability of winning the ME selection given the child
    is initially active. This creates a system of equations since win_prob depends
    on all siblings' corrected probabilities.

    Args:
        sibling_indices: Feature indices of ME siblings
        parent_base_prob: Base probability of the parent (effective rate after correction)
        base_probs: Base firing probabilities tensor
        correction_factors: Correction factors tensor (modified in place)
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence tolerance
    """
    if parent_base_prob <= 0:
        return

    n = len(sibling_indices)
    base = [base_probs[i].item() for i in sibling_indices]

    # Initialize corrected probs (without ME adjustment)
    corrected = [b / parent_base_prob for b in base]

    for _ in range(max_iterations):
        # Compute win probability for each sibling
        win_probs = _compute_me_win_probs(corrected)

        # Update corrected probs
        new_corrected = []
        max_diff = 0.0
        for i in range(n):
            if win_probs[i] > 0 and parent_base_prob > 0:
                c = base[i] / (parent_base_prob * win_probs[i])
                # Clamp to valid probability
                c = min(c, 1.0)
            else:
                c = corrected[i]
            max_diff = max(max_diff, abs(c - corrected[i]))
            new_corrected.append(c)

        corrected = new_corrected
        if max_diff < tolerance:
            break

    # Update correction factors
    for i, idx in enumerate(sibling_indices):
        if base[i] > 0:
            correction_factors[idx] = corrected[i] / base[i]


def _compute_me_win_probs(corrected: list[float]) -> list[float]:
    """
    Compute P(sibling i wins ME | sibling i is active) for each sibling.

    When sibling i is active and k total siblings are active (including i),
    sibling i wins with probability 1/k. This computes E[1/K | i is active].

    Args:
        corrected: List of corrected probabilities for each sibling

    Returns:
        List of win probabilities for each sibling
    """
    n = len(corrected)
    win_probs = []

    for i in range(n):
        # For sibling i, compute E[1/K | i is active]
        # where K = 1 + (number of other active siblings)
        # Enumerate all 2^(n-1) subsets of other siblings
        other_indices = [j for j in range(n) if j != i]
        total_prob = 0.0

        for mask in range(1 << (n - 1)):  # 2^(n-1) subsets
            # Compute P(this subset of others is active) × 1/(1 + |subset|)
            subset_prob = 1.0
            num_active = 1  # sibling i is always active in this conditional

            for bit, j in enumerate(other_indices):
                if mask & (1 << bit):
                    subset_prob *= corrected[j]
                    num_active += 1
                else:
                    subset_prob *= 1 - corrected[j]

            total_prob += subset_prob / num_active

        win_probs.append(total_prob)

    return win_probs


def generate_hierarchy(
    num_features: int,
    config: HierarchyConfig,
    seed: int | None = None,
) -> Hierarchy:
    """
    Generate a hierarchy forest based on config using breadth-first construction.

    Algorithm:
        1. Create total_root_nodes root nodes
        2. Build each tree breadth-first up to max_depth
        3. Each non-leaf node gets branching_factor children
        4. Apply mutually_exclusive flag to portion of parent nodes

    Args:
        num_features: Total number of features available
        config: Hierarchy configuration
        seed: Random seed for reproducibility

    Returns:
        Hierarchy with roots and modifier function
    """
    if config.total_root_nodes == 0:
        return Hierarchy(roots=[], modifier=None)

    rng = random.Random(seed)

    feature_idx = 0
    roots: list[HierarchyNode] = []
    # Track all parent nodes with their depths for ME assignment
    all_parents_with_depth: list[tuple[HierarchyNode, int]] = []

    def get_branching() -> int:
        if isinstance(config.branching_factor, int):
            return config.branching_factor
        return rng.randint(config.branching_factor[0], config.branching_factor[1])

    def next_feature() -> int | None:
        nonlocal feature_idx
        if feature_idx >= num_features:
            return None
        idx = feature_idx
        feature_idx += 1
        return idx

    # Create root nodes
    for _ in range(config.total_root_nodes):
        feat_idx = next_feature()
        if feat_idx is None:
            break

        node = HierarchyNode(
            feature_index=feat_idx,
            children=[],
            mutually_exclusive_children=False,
        )
        roots.append(node)
        all_parents_with_depth.append((node, 0))

    # Process level by level (breadth-first), building complete trees
    # depth 0 = roots, we build children for depths 0 to max_depth-1
    parents_at_current_level: list[tuple[HierarchyNode, int]] = [(r, 0) for r in roots]

    while parents_at_current_level:
        next_level: list[tuple[HierarchyNode, int]] = []

        for parent_node, depth in parents_at_current_level:
            # Don't add children if we're at max_depth
            if depth >= config.max_depth:
                continue

            branching = get_branching()
            children: list[HierarchyNode] = []

            for _ in range(branching):
                feat_idx = next_feature()
                if feat_idx is None:
                    break

                # Children at depth+1 < max_depth are parents, otherwise leaves
                is_parent = depth + 1 < config.max_depth

                if is_parent:
                    child = HierarchyNode(
                        feature_index=feat_idx,
                        children=[],
                        mutually_exclusive_children=False,
                    )
                    next_level.append((child, depth + 1))
                    all_parents_with_depth.append((child, depth + 1))
                else:
                    child = HierarchyNode(feature_index=feat_idx)

                children.append(child)

            parent_node.children = children

        parents_at_current_level = next_level

    # Filter parents eligible for ME based on depth constraints
    me_min_depth = config.mutually_exclusive_min_depth
    me_max_depth = config.mutually_exclusive_max_depth
    eligible_parents = [
        (node, depth)
        for node, depth in all_parents_with_depth
        if depth >= me_min_depth
        and (me_max_depth is None or depth <= me_max_depth)
        and len(node.children) >= 2
    ]

    # Assign ME flag to portion of eligible parents
    num_me_parents = int(len(eligible_parents) * config.mutually_exclusive_portion)
    if num_me_parents > 0:
        me_indices = set(rng.sample(range(len(eligible_parents)), num_me_parents))
        for i in me_indices:
            parent, _ = eligible_parents[i]
            parent.mutually_exclusive_children = True

    modifier = hierarchy_modifier(roots) if roots else None

    return Hierarchy(
        roots=roots,
        modifier=modifier,
    )
