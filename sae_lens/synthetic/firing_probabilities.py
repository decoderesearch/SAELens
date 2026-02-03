"""
Firing probability distributions and registry.

This module provides functions for generating firing probability distributions
and a registry pattern for extensibility.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import torch

from sae_lens.synthetic.registry import FIRING_PROB_REGISTRY

# =============================================================================
# Base classes
# =============================================================================


@dataclass
class FiringProbabilityConfig(ABC):
    """Base config for firing probability generators."""

    @classmethod
    @abstractmethod
    def generator_name(cls) -> str:
        """Return the registered name for this generator type."""
        ...

    @abstractmethod
    def generate(self, num_features: int, seed: int | None = None) -> torch.Tensor:
        """
        Generate firing probabilities.

        Args:
            num_features: Number of features to generate probabilities for
            seed: Optional random seed for reproducibility (for stochastic generators)

        Returns:
            Tensor of shape (num_features,) with firing probabilities
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary."""
        result = asdict(self)
        result["generator_name"] = self.generator_name()
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FiringProbabilityConfig":
        """
        Deserialize config from dictionary.

        Uses the registry to find the correct config class.
        """
        d = dict(d)  # Make a copy
        name = d.pop("generator_name", None)
        if name is None:
            raise ValueError("generator_name required in config dict")
        cfg_class = FIRING_PROB_REGISTRY.get_or_raise(name)
        return cfg_class(**d)


def register_firing_probability(
    name: str,
    config_class: type[FiringProbabilityConfig],
) -> None:
    """
    Register a firing probability config.

    Args:
        name: Unique name for this firing probability type
        config_class: Config dataclass for this generator
    """
    FIRING_PROB_REGISTRY.register(name, config_class)


def get_firing_probability_class(
    name: str,
) -> type[FiringProbabilityConfig]:
    """
    Get the config class for a firing probability type.

    Args:
        name: Name of the firing probability type

    Returns:
        The config class
    """
    return FIRING_PROB_REGISTRY.get_or_raise(name)


# =============================================================================
# Helper functions
# =============================================================================


def zipfian_firing_probabilities(
    num_features: int,
    exponent: float = 1.0,
    max_prob: float = 0.3,
    min_prob: float = 0.01,
) -> torch.Tensor:
    """
    Generate firing probabilities following a Zipfian (power-law) distribution.

    Creates probabilities where a few features fire frequently and most fire rarely,
    which mirrors the distribution often observed in real neural network features.

    Args:
        num_features: Number of features to generate probabilities for
        exponent: Zipf exponent (higher = steeper dropoff). Default 1.0.
        max_prob: Maximum firing probability (for the most frequent feature)
        min_prob: Minimum firing probability (for the least frequent feature)

    Returns:
        Tensor of shape [num_features] with firing probabilities in descending order
    """
    if num_features < 1:
        raise ValueError("num_features must be at least 1")
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    if not 0 < min_prob < max_prob <= 1:
        raise ValueError("Must have 0 < min_prob < max_prob <= 1")

    ranks = torch.arange(1, num_features + 1, dtype=torch.float32)
    probs = 1.0 / ranks**exponent

    # Scale to [min_prob, max_prob]
    if num_features == 1:
        return torch.tensor([max_prob])

    probs_min, probs_max = probs.min(), probs.max()
    return min_prob + (max_prob - min_prob) * (probs - probs_min) / (
        probs_max - probs_min
    )


def linear_firing_probabilities(
    num_features: int,
    max_prob: float = 0.3,
    min_prob: float = 0.01,
) -> torch.Tensor:
    """
    Generate firing probabilities that decay linearly from max to min.

    Args:
        num_features: Number of features to generate probabilities for
        max_prob: Firing probability for the first feature
        min_prob: Firing probability for the last feature

    Returns:
        Tensor of shape [num_features] with linearly decaying probabilities
    """
    if num_features < 1:
        raise ValueError("num_features must be at least 1")
    if not 0 < min_prob <= max_prob <= 1:
        raise ValueError("Must have 0 < min_prob <= max_prob <= 1")

    if num_features == 1:
        return torch.tensor([max_prob])

    return torch.linspace(max_prob, min_prob, num_features)


def random_firing_probabilities(
    num_features: int,
    max_prob: float = 0.5,
    min_prob: float = 0.01,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Generate random firing probabilities uniformly sampled from a range.

    Args:
        num_features: Number of features to generate probabilities for
        max_prob: Maximum firing probability
        min_prob: Minimum firing probability
        seed: Optional random seed for reproducibility

    Returns:
        Tensor of shape [num_features] with random firing probabilities
    """
    if num_features < 1:
        raise ValueError("num_features must be at least 1")
    if not 0 < min_prob < max_prob <= 1:
        raise ValueError("Must have 0 < min_prob < max_prob <= 1")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    probs = torch.rand(num_features, generator=generator, dtype=torch.float32)
    return min_prob + (max_prob - min_prob) * probs


# =============================================================================
# Built-in implementations
# =============================================================================


@dataclass
class ZipfianFiringProbabilityConfig(FiringProbabilityConfig):
    """
    Config for Zipfian (power-law) firing probability distribution.

    Creates probabilities where a few features fire frequently and most fire rarely,
    which mirrors the distribution often observed in real neural network features.
    """

    exponent: float = 1.0
    max_prob: float = 0.3
    min_prob: float = 0.01

    @classmethod
    def generator_name(cls) -> str:
        return "zipfian"

    def generate(self, num_features: int, seed: int | None = None) -> torch.Tensor:
        return zipfian_firing_probabilities(
            num_features,
            exponent=self.exponent,
            max_prob=self.max_prob,
            min_prob=self.min_prob,
        )


@dataclass
class LinearFiringProbabilityConfig(FiringProbabilityConfig):
    """
    Config for linearly decaying firing probability distribution.

    Probabilities decay linearly from max_prob to min_prob.
    """

    max_prob: float = 0.3
    min_prob: float = 0.01

    @classmethod
    def generator_name(cls) -> str:
        return "linear"

    def generate(self, num_features: int, seed: int | None = None) -> torch.Tensor:
        return linear_firing_probabilities(
            num_features,
            max_prob=self.max_prob,
            min_prob=self.min_prob,
        )


@dataclass
class RandomFiringProbabilityConfig(FiringProbabilityConfig):
    """
    Config for random firing probability distribution.

    Probabilities are uniformly sampled from [min_prob, max_prob].
    """

    max_prob: float = 0.5
    min_prob: float = 0.01

    @classmethod
    def generator_name(cls) -> str:
        return "random"

    def generate(self, num_features: int, seed: int | None = None) -> torch.Tensor:
        return random_firing_probabilities(
            num_features,
            max_prob=self.max_prob,
            min_prob=self.min_prob,
            seed=seed,
        )


@dataclass
class ConstantFiringProbabilityConfig(FiringProbabilityConfig):
    """
    Config for constant firing probability.

    All features have the same firing probability.
    """

    probability: float = 0.1

    @classmethod
    def generator_name(cls) -> str:
        return "constant"

    def generate(self, num_features: int, seed: int | None = None) -> torch.Tensor:
        return torch.full((num_features,), self.probability, dtype=torch.float32)


# =============================================================================
# Register built-in generators
# =============================================================================

register_firing_probability("zipfian", ZipfianFiringProbabilityConfig)
register_firing_probability("linear", LinearFiringProbabilityConfig)
register_firing_probability("random", RandomFiringProbabilityConfig)
register_firing_probability("constant", ConstantFiringProbabilityConfig)
