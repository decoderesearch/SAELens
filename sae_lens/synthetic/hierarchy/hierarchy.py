"""Hierarchy dataclass and generate_hierarchy function."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch

from sae_lens.synthetic.hierarchy.config import HierarchyConfig
from sae_lens.synthetic.hierarchy.modifier import (
    ActivationsModifier,
    hierarchy_modifier,
)
from sae_lens.synthetic.hierarchy.node import HierarchyNode


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

        When mutual exclusion is enabled for a group of siblings, only one can
        remain active at a time. This further reduces effective probabilities.
        For feature i in an ME group under parent P, the expected number of
        competing siblings (given parent fires) is sum(other_base_probs) / P.
        The ME correction is (1 + expected_competitors).

        When all features are sampled with corrected probabilities and hierarchy
        is applied, the effective firing rate for each feature approximately
        equals its base probability.

        Args:
            base_firing_probabilities: Original firing probabilities of shape
                (num_features,)

        Returns:
            Tensor of shape (num_features,) with correction factors. For hierarchy,
            this is 1 / base_prob[parent]. For ME groups, an additional multiplicative
            factor of (1 + sum_other_sibling_probs) is applied. Features not in the
            hierarchy (roots and features outside any tree) get correction factor
            of 1.0.
        """
        num_features = base_firing_probabilities.shape[0]
        correction_factors = torch.ones(
            num_features, dtype=base_firing_probabilities.dtype
        )

        def traverse(node: HierarchyNode, parent_base_prob: float) -> None:
            # Set hierarchy correction for this node
            if node.feature_index is not None:
                if parent_base_prob > 0:
                    correction_factors[node.feature_index] = 1.0 / parent_base_prob
                node_prob = base_firing_probabilities[node.feature_index].item()
            else:
                # Organizational node without feature_index - children see parent's prob
                node_prob = parent_base_prob

            # Recurse to children first so they get their hierarchy corrections
            for child in node.children:
                traverse(child, node_prob)

            # After children have hierarchy corrections, apply ME correction
            # The ME correction is multiplicative on top of hierarchy correction
            if node.mutually_exclusive_children and len(node.children) >= 2:
                # Collect base probabilities of children in this ME group
                child_probs: list[tuple[int, float]] = []
                for child in node.children:
                    if child.feature_index is not None:
                        p = base_firing_probabilities[child.feature_index].item()
                        child_probs.append((child.feature_index, p))

                if len(child_probs) >= 2 and node_prob > 0:
                    # For each child, ME correction accounts for competition with siblings.
                    # Given parent fires, sibling j fires with prob base_prob[j] / node_prob
                    # (after hierarchy correction). Expected competing siblings is
                    # sum(other_base_probs) / node_prob.
                    # ME correction = 1 + expected_other_active
                    total_prob = sum(p for _, p in child_probs)
                    for feat_idx, p in child_probs:
                        # Use max(0, ...) to handle floating-point errors when total_prob ≈ p
                        other_probs_sum = max(0.0, total_prob - p)
                        me_correction = 1.0 + other_probs_sum / node_prob
                        correction_factors[feat_idx] *= me_correction

        for root in self.roots:
            traverse(root, 1.0)

        return correction_factors


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
