"""
Hierarchy modifier functions for applying hierarchy constraints to activations.

This module provides the hierarchy_modifier function and all supporting sparse
machinery for efficiently applying hierarchical dependencies and mutual exclusion
to feature activations.
"""

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from sae_lens.synthetic.activation_generator import ActivationGenerator
from sae_lens.synthetic.hierarchy.node import HierarchyNode
from sae_lens.synthetic.hierarchy.validation import validate_hierarchy

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

    # Per-feature bool mask: True where the feature's parent has scale_children_by_parent
    rescale_mask: torch.Tensor | None = None  # [num_features_at_level]


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

    # Per-feature bool: True if parent has scale_children_by_parent
    feat_rescale: torch.Tensor | None = None  # [num_features]

    # Unique parent feature indices that have rescaling children
    rescale_parent_indices: torch.Tensor | None = None  # [num_rescale_parents]

    # Whether any node uses rescale (for fast-path optimization)
    has_rescale: bool = False


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
    # Each entry: (feature_index, effective_parent, level, parent_rescales)
    feature_info: list[tuple[int, int, int, bool]] = []

    # ME groups: list of (parent_level, parent_feature, child_feature_indices)
    me_groups: list[tuple[int, int, list[int]]] = []

    # BFS queue: (node, effective_parent, level, effective_parent_rescales)
    queue: deque[tuple[HierarchyNode, int, int, bool]] = deque()
    for root in roots:
        queue.append((root, -1, 0, False))

    while queue:
        node, effective_parent, level, effective_parent_rescales = queue.popleft()

        if node.feature_index is not None:
            feature_info.append(
                (node.feature_index, effective_parent, level, effective_parent_rescales)
            )
            new_effective_parent = node.feature_index
            new_rescales = node.scale_children_by_parent
        else:
            new_effective_parent = effective_parent
            new_rescales = node.scale_children_by_parent

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
            queue.append((child, new_effective_parent, level + 1, new_rescales))

    # Determine max level for both features and ME groups
    max_feature_level = max((info[2] for info in feature_info), default=-1)
    max_me_level = max((lvl for lvl, _, _ in me_groups), default=-1)
    max_level = max(max_feature_level, max_me_level)

    # Check if any node uses rescale
    any_rescale = any(rescale for _, _, _, rescale in feature_info)

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
            (feat, parent, rescale)
            for feat, parent, lv, rescale in feature_info
            if lv == level
        ]
        with_parents = [(f, p, r) for f, p, r in features_at_level if p >= 0]

        if with_parents:
            feats = torch.tensor([f for f, _, _ in with_parents], dtype=torch.long)
            parents = torch.tensor([p for _, p, _ in with_parents], dtype=torch.long)
            rescale_mask: torch.Tensor | None = (
                torch.tensor([r for _, _, r in with_parents], dtype=torch.bool)
                if any_rescale
                else None
            )
        else:
            feats = torch.empty(0, dtype=torch.long)
            parents = torch.empty(0, dtype=torch.long)
            rescale_mask = None

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
                rescale_mask=rescale_mask,
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
    all_features = [f for f, _, _, _ in feature_info]
    num_features = max(all_features) + 1 if all_features else 0

    # Build feature-to-parent mapping
    feat_to_parent = torch.full((num_features,), -1, dtype=torch.long)
    for feat, parent, _, _ in feature_info:
        feat_to_parent[feat] = parent

    # Build feature-to-ME-group mapping
    feat_to_me_group = torch.full((num_features,), -1, dtype=torch.long)
    for g_idx, (_, _, siblings) in enumerate(me_groups):
        for sib in siblings:
            feat_to_me_group[sib] = g_idx

    # Build per-feature rescale mask for COO path
    feat_rescale: torch.Tensor | None = None
    rescale_parent_indices: torch.Tensor | None = None
    if any_rescale:
        feat_rescale = torch.zeros(num_features, dtype=torch.bool)
        rescale_parents: set[int] = set()
        for feat, parent, _, rescale in feature_info:
            feat_rescale[feat] = rescale
            if rescale and parent >= 0:
                rescale_parents.add(parent)
        rescale_parent_indices = torch.tensor(sorted(rescale_parents), dtype=torch.long)

    return _SparseHierarchyData(
        level_data=level_data,
        me_group_siblings=me_group_siblings,
        me_group_sizes=me_group_sizes,
        me_group_parents=me_group_parents,
        num_groups=num_groups,
        feat_to_parent=feat_to_parent,
        feat_to_me_group=feat_to_me_group,
        feat_rescale=feat_rescale,
        rescale_parent_indices=rescale_parent_indices,
        has_rescale=any_rescale,
    )


def _apply_hierarchy_sparse(
    activations: torch.Tensor,
    sparse_data: _SparseHierarchyData,
    mean_firing_magnitudes: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply hierarchy constraints using precomputed sparse indices.

    Processes level by level:
    1. Apply parent deactivation for features at this level
    2. Apply mutual exclusion for groups whose parent is at this level
    3. Move to next level

    This ensures that ME at level L affects parent deactivation at level L+1.

    Args:
        activations: (batch_size, num_features) tensor of activations
        sparse_data: Precomputed sparse hierarchy data
        mean_firing_magnitudes: Required when sparse_data.has_rescale is True.
            Shape (num_features,).
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
            if level_data.rescale_mask is not None and level_data.rescale_mask.any():
                assert mean_firing_magnitudes is not None
                parent_means = mean_firing_magnitudes[level_data.parents]
                parent_active = parent_vals > 0
                scale = torch.where(
                    parent_active,
                    parent_vals / parent_means,
                    torch.zeros_like(parent_vals),
                )
                binary_gate = parent_active.to(child_vals.dtype)
                factor = torch.where(level_data.rescale_mask, scale, binary_gate)
                result[:, level_data.features] = child_vals * factor
            else:
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
    mean_firing_magnitudes: torch.Tensor | None = None,
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
                sparse_tensor, level_data, sparse_data, mean_firing_magnitudes
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
    mean_firing_magnitudes: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Remove or rescale children in sparse COO tensor based on parent activity.

    Uses the per-feature rescale_mask from level_data. Features whose parent has
    scale_children_by_parent=True are rescaled by parent_val/parent_mean; others
    are binary-gated.

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
    active_pairs_sorted, sort_order = active_pairs.sort()

    # Only process features at this level (not all features with parents)
    level_features_set = level_data.features
    if level_features_set.numel() == 0:
        return sparse_tensor

    # Build a mask of entries whose feature is in this level's feature set
    is_level_feature = torch.isin(feat_indices, level_features_set)

    # Use the precomputed feat_to_parent mapping
    assert sparse_data.feat_to_parent is not None
    hierarchy_num_features = sparse_data.feat_to_parent.numel()

    # Get parent for level features only
    parent_of_feat = torch.full((nnz,), -1, dtype=torch.long, device=device)
    level_and_in_hierarchy = is_level_feature & (feat_indices < hierarchy_num_features)
    parent_of_feat[level_and_in_hierarchy] = sparse_data.feat_to_parent[
        feat_indices[level_and_in_hierarchy]
    ]

    # Find entries that are at this level and have a parent
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

    has_rescale_at_level = (
        level_data.rescale_mask is not None and level_data.rescale_mask.any()
    )

    if has_rescale_at_level:
        assert mean_firing_magnitudes is not None
        assert level_data.rescale_mask is not None

        # Build per-child-entry rescale flag from feat_rescale
        assert sparse_data.feat_rescale is not None
        child_feat_indices = feat_indices[has_parent]
        child_in_hierarchy = child_feat_indices < sparse_data.feat_rescale.numel()
        entry_rescale = torch.zeros(
            child_entry_indices.numel(), dtype=torch.bool, device=device
        )
        entry_rescale[child_in_hierarchy] = sparse_data.feat_rescale[
            child_feat_indices[child_in_hierarchy]
        ]

        new_values = values.clone()

        # For children with active parents that need rescaling
        rescale_and_active = parent_active & entry_rescale
        if rescale_and_active.any():
            rescale_entries = child_entry_indices[rescale_and_active]
            rescale_parent_pairs = parent_pairs[rescale_and_active]

            # Find parent values via the sorted active_pairs
            # Parents were already rescaled in the previous level's call
            parent_positions_in_sorted = torch.searchsorted(
                active_pairs_sorted, rescale_parent_pairs
            )
            parent_original_indices = sort_order[parent_positions_in_sorted]
            parent_vals = new_values[parent_original_indices]
            parent_feat_ids = child_parents[rescale_and_active]
            parent_means = mean_firing_magnitudes[parent_feat_ids]
            scale = parent_vals / parent_means
            new_values[rescale_entries] *= scale

        # For children with inactive parents (both rescale and binary), zero them out
        inactive_child_entries = child_entry_indices[~parent_active]
        new_values[inactive_child_entries] = 0

        # Remove zero entries
        nonzero_mask = new_values != 0
        if nonzero_mask.all():
            return torch.sparse_coo_tensor(
                indices,
                new_values,
                sparse_tensor.shape,
                device=device,
                dtype=sparse_tensor.dtype,
            )
        return torch.sparse_coo_tensor(
            indices[:, nonzero_mask],
            new_values[nonzero_mask],
            sparse_tensor.shape,
            device=device,
            dtype=sparse_tensor.dtype,
        )

    # Binary mode: just remove children with inactive parents
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
) -> Callable[..., torch.Tensor]:
    """
    Create an activations modifier from one or more hierarchy trees.

    This is the recommended way to use hierarchies with ActivationGenerator.
    It validates the hierarchy structure and returns a modifier function that
    applies all hierarchy constraints.

    If any node in the hierarchy has ``scale_children_by_parent=True``, a
    2-arg modifier is returned that receives the ``ActivationGenerator`` as its
    second argument (to read ``mean_firing_magnitudes``). Otherwise a simple
    1-arg modifier is returned.

    Args:
        roots: One or more root HierarchyNode objects. Each root defines an
            independent hierarchy tree. All trees are validated and applied.

    Returns:
        An ActivationsModifier function that can be passed to ActivationGenerator.

    Raises:
        ValueError: If the hierarchy contains loops or nodes with multiple parents.
    """
    if not roots:
        # No hierarchies - return identity function
        def identity(activations: torch.Tensor) -> torch.Tensor:
            return activations

        return identity

    if isinstance(roots, HierarchyNode):
        roots = [roots]
    validate_hierarchy(roots)

    # Build sparse hierarchy data
    sparse_data = _build_sparse_hierarchy(roots)

    # Cache for device-specific tensors
    device_cache: dict[torch.device, _SparseHierarchyData] = {}
    mean_mags_cache: dict[torch.device, torch.Tensor] = {}

    def _get_sparse_for_device(device: torch.device) -> _SparseHierarchyData:
        """Get or create device-specific sparse hierarchy data."""
        if device not in device_cache:
            device_cache[device] = _SparseHierarchyData(
                level_data=[
                    _LevelData(
                        features=ld.features.to(device),
                        parents=ld.parents.to(device),
                        me_group_indices=ld.me_group_indices.to(device),
                        rescale_mask=(
                            ld.rescale_mask.to(device)
                            if ld.rescale_mask is not None
                            else None
                        ),
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
                feat_rescale=(
                    sparse_data.feat_rescale.to(device)
                    if sparse_data.feat_rescale is not None
                    else None
                ),
                rescale_parent_indices=(
                    sparse_data.rescale_parent_indices.to(device)
                    if sparse_data.rescale_parent_indices is not None
                    else None
                ),
                has_rescale=sparse_data.has_rescale,
            )
        return device_cache[device]

    if sparse_data.has_rescale:

        def modifier_with_rescale(
            activations: torch.Tensor, generator: ActivationGenerator
        ) -> torch.Tensor:
            device = activations.device
            cached = _get_sparse_for_device(device)
            if device not in mean_mags_cache:
                mean_mags = generator.mean_firing_magnitudes.to(device)
                assert cached.rescale_parent_indices is not None
                if not (mean_mags[cached.rescale_parent_indices] > 0).all():
                    raise ValueError(
                        "mean_firing_magnitudes must be > 0 for parents with"
                        " scale_children_by_parent=True"
                    )
                mean_mags_cache[device] = mean_mags
            mean_mags = mean_mags_cache[device]
            if activations.is_sparse:
                return _apply_hierarchy_sparse_coo(activations, cached, mean_mags)
            return _apply_hierarchy_sparse(activations, cached, mean_mags)

        return modifier_with_rescale

    def modifier(activations: torch.Tensor) -> torch.Tensor:
        device = activations.device
        cached = _get_sparse_for_device(device)
        if activations.is_sparse:
            return _apply_hierarchy_sparse_coo(activations, cached)
        return _apply_hierarchy_sparse(activations, cached)

    return modifier
