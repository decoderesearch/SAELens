"""
Hierarchical (tree-structured) feature generator.

This module provides TreeFeatureGenerator, which generates features with
hierarchical dependencies. Child features can only fire when their parent fires,
and children can optionally be mutually exclusive.

Based on Noa Nabeshima's Matryoshka SAEs:
https://github.com/noanabeshima/matryoshka-saes/blob/main/toy_model.py
"""

from collections.abc import Sequence
from typing import Any

import torch


class TreeFeatureGenerator:
    """
    Generates feature activations with hierarchical (tree) structure.

    Features are organized in a tree where child features can only fire
    when their parent fires. Children can optionally be mutually exclusive.

    Example:
        >>> # Create a simple hierarchy:
        >>> # Root (always fires) -> Child1 (50%) -> GrandChild (30%)
        >>> #                     -> Child2 (40%)
        >>> grandchild = TreeFeatureGenerator(active_prob=0.3)
        >>> child1 = TreeFeatureGenerator(active_prob=0.5, children=[grandchild])
        >>> child2 = TreeFeatureGenerator(active_prob=0.4)
        >>> root = TreeFeatureGenerator(
        ...     active_prob=1.0,
        ...     children=[child1, child2],
        ...     mutually_exclusive_children=True  # Only one child fires
        ... )
        >>> activations = root.sample(batch_size=1000)
        >>> activations.shape
        torch.Size([1000, 4])

    Attributes:
        active_prob: Probability this feature fires (conditional on parent firing)
        children: Child TreeFeatureGenerator nodes
        mutually_exclusive_children: If True, at most one child fires per sample
        is_read_out: If True, this feature is included in the output
        index: Index of this feature in the output tensor (None if not read out)
        magnitude_mean: Mean magnitude when firing
        magnitude_std: Standard deviation of magnitude when firing
    """

    children: Sequence["TreeFeatureGenerator"]
    index: int | None

    @classmethod
    def from_dict(
        cls,
        tree_dict: dict[str, Any],
        default_magnitude_mean: float = 1.0,
        default_magnitude_std: float = 0.0,
    ) -> "TreeFeatureGenerator":
        """
        Create a TreeFeatureGenerator from a dictionary specification.

        Args:
            tree_dict: Dictionary with keys:
                - active_prob (required): Probability this feature fires
                - children (optional): List of child tree dictionaries
                - mutually_exclusive_children (optional): Whether children are exclusive
                - is_read_out (optional): Whether to include in output (default True)
                - id (optional): Identifier for this feature
                - magnitude_mean (optional): Mean firing magnitude
                - magnitude_std (optional): Std dev of firing magnitude
            default_magnitude_mean: Default mean magnitude if not specified
            default_magnitude_std: Default std dev if not specified

        Returns:
            TreeFeatureGenerator instance
        """
        children = [
            TreeFeatureGenerator.from_dict(
                child_dict,
                default_magnitude_mean=default_magnitude_mean,
                default_magnitude_std=default_magnitude_std,
            )
            for child_dict in tree_dict.get("children", [])
        ]
        return cls(
            active_prob=tree_dict["active_prob"],
            children=children,
            mutually_exclusive_children=tree_dict.get(
                "mutually_exclusive_children", False
            ),
            is_read_out=tree_dict.get("is_read_out", True),
            feature_id=tree_dict.get("id"),
            magnitude_mean=tree_dict.get("magnitude_mean", default_magnitude_mean),
            magnitude_std=tree_dict.get("magnitude_std", default_magnitude_std),
        )

    def __init__(
        self,
        active_prob: float,
        children: Sequence["TreeFeatureGenerator"] | None = None,
        mutually_exclusive_children: bool = False,
        is_read_out: bool = True,
        feature_id: str | None = None,
        magnitude_mean: float = 1.0,
        magnitude_std: float = 0.0,
    ):
        """
        Create a new TreeFeatureGenerator node.

        Args:
            active_prob: Probability this feature fires (conditional on parent)
            children: Child generators that depend on this feature
            mutually_exclusive_children: If True, only one child can fire per sample
            is_read_out: If True, include this feature in the output tensor
            feature_id: Optional identifier for this feature
            magnitude_mean: Mean magnitude when feature fires
            magnitude_std: Standard deviation of magnitude
        """
        self.active_prob = active_prob
        self.children = children or []
        self.mutually_exclusive_children = mutually_exclusive_children
        self.is_read_out = is_read_out
        self.index = None
        self.feature_id = feature_id
        self.magnitude_mean = magnitude_mean
        self.magnitude_std = magnitude_std

        if self.mutually_exclusive_children:
            assert (
                len(self.children) >= 2
            ), "Need at least 2 children for mutual exclusion"

        self._init_indices()

    def _init_indices(self, start_idx: int = 0) -> int:
        """Initialize feature indices for all nodes in the tree."""
        if self.is_read_out:
            self.index = start_idx
            start_idx += 1
        else:
            self.index = None

        for child in self.children:
            start_idx = child._init_indices(start_idx)

        return start_idx

    @property
    def n_features(self) -> int:
        """Total number of read-out features in this subtree."""
        count = 1 if self.is_read_out else 0
        for child in self.children:
            count += child.n_features
        return count

    @property
    def _child_probs(self) -> torch.Tensor:
        """Get firing probabilities of children as tensor."""
        return torch.tensor([child.active_prob for child in self.children])

    @torch.no_grad()
    def sample(
        self, batch_size: int, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Generate a batch of feature activations.

        Args:
            batch_size: Number of samples to generate
            device: Device to generate samples on

        Returns:
            Tensor of shape [batch_size, n_features] with activations
        """
        if device is None:
            device = torch.device("cpu")

        batch = torch.zeros((batch_size, self.n_features))
        self._fill_batch(batch)
        return batch.to(device)

    def _fill_batch(
        self,
        batch: torch.Tensor,
        parent_active_mask: torch.Tensor | None = None,
        force_active_mask: torch.Tensor | None = None,
        force_inactive_mask: torch.Tensor | None = None,
    ) -> None:
        """Fill the batch tensor with activations for this subtree."""
        batch_size = batch.shape[0]
        is_active = _sample_is_active(
            self.active_prob,
            batch_size=batch_size,
            parent_active_mask=parent_active_mask,
            force_active_mask=force_active_mask,
            force_inactive_mask=force_inactive_mask,
        )

        if self.is_read_out:
            magnitudes = _sample_magnitudes(
                is_active, self.magnitude_mean, self.magnitude_std
            )
            assert self.index is not None
            batch[:, self.index] = magnitudes

        # Handle children
        active_child = None
        if self.mutually_exclusive_children:
            active_child = torch.multinomial(
                self._child_probs.expand(batch_size, -1), 1
            ).squeeze(-1)

        for child_idx, child in enumerate(self.children):
            child_force_inactive = (
                None if active_child is None else active_child != child_idx
            )
            child_force_active = (
                None if active_child is None else active_child == child_idx
            )
            child._fill_batch(
                batch,
                parent_active_mask=is_active,
                force_active_mask=child_force_active,
                force_inactive_mask=child_force_inactive,
            )

    def __repr__(self, indent: int = 0) -> str:
        s = " " * (indent * 2)
        s += str(self.index) if self.index is not None else "-"
        s += "x" if self.mutually_exclusive_children else " "
        s += f" {self.active_prob}"
        if self.feature_id:
            s += f" ({self.feature_id})"

        for child in self.children:
            s += "\n" + child.__repr__(indent + 2)
        return s


def _sample_is_active(
    active_prob: float,
    batch_size: int,
    parent_active_mask: torch.Tensor | None,
    force_active_mask: torch.Tensor | None,
    force_inactive_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Sample binary activation mask."""
    is_active = torch.bernoulli(torch.tensor(active_prob).expand(batch_size))
    if force_active_mask is not None:
        is_active[force_active_mask] = 1
    if force_inactive_mask is not None:
        is_active[force_inactive_mask] = 0
    if parent_active_mask is not None:
        is_active[parent_active_mask == 0] = 0
    return is_active


def _sample_magnitudes(
    is_active: torch.Tensor,
    magnitude_mean: float,
    magnitude_std: float,
) -> torch.Tensor:
    """Sample magnitudes for active features."""
    batch_size = is_active.shape[0]
    magnitudes = torch.zeros_like(is_active, dtype=torch.float32)

    if magnitude_std > 0:
        active_magnitudes = torch.abs(
            torch.normal(mean=magnitude_mean, std=magnitude_std, size=(batch_size,))
        )
    else:
        active_magnitudes = torch.full((batch_size,), magnitude_mean)

    magnitudes[is_active == 1] = active_magnitudes[is_active == 1]
    return magnitudes
