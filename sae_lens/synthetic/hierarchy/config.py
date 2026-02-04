"""HierarchyConfig dataclass for automatic hierarchy generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


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
            **asdict(self),
            "branching_factor": branching,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HierarchyConfig:
        """Deserialize config from dictionary."""
        d = dict(d)  # Make a copy
        # Convert list to tuple (JSON doesn't have tuples)
        if "branching_factor" in d and isinstance(d["branching_factor"], list):
            d["branching_factor"] = tuple(d["branching_factor"])
        return cls(**d)
