"""
Firing magnitude configuration and generation.

This module provides configuration and generation for per-feature magnitude values
(mean and std) that vary across features, using a registry pattern for extensibility.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import torch

from sae_lens.synthetic.registry import MAGNITUDE_REGISTRY

# =============================================================================
# Base classes
# =============================================================================


@dataclass
class MagnitudeConfig(ABC):
    """Base config for magnitude generators."""

    @classmethod
    @abstractmethod
    def generator_name(cls) -> str:
        """Return the registered name for this generator type."""
        ...

    @abstractmethod
    def generate(self, num_features: int, seed: int | None = None) -> torch.Tensor:
        """
        Generate magnitude values.

        Args:
            num_features: Number of features to generate magnitudes for
            seed: Optional random seed for reproducibility (for stochastic generators)

        Returns:
            Tensor of shape (num_features,) with magnitude values
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary."""
        result = asdict(self)
        result["generator_name"] = self.generator_name()
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MagnitudeConfig":
        """
        Deserialize config from dictionary.

        Uses the registry to find the correct config class.
        """
        d = dict(d)  # Make a copy
        name = d.pop("generator_name", None)
        if name is None:
            raise ValueError("generator_name required in config dict")
        cfg_class = MAGNITUDE_REGISTRY.get_or_raise(name)
        return cfg_class(**d)


def register_magnitude(
    name: str,
    config_class: type[MagnitudeConfig],
) -> None:
    """
    Register a magnitude config.

    Args:
        name: Unique name for this magnitude type
        config_class: Config dataclass for this generator
    """
    MAGNITUDE_REGISTRY.register(name, config_class)


def get_magnitude_class(
    name: str,
) -> type[MagnitudeConfig]:
    """
    Get the config class for a magnitude type.

    Args:
        name: Name of the magnitude type

    Returns:
        The config class
    """
    return MAGNITUDE_REGISTRY.get_or_raise(name)


def generate_magnitudes(
    num_features: int,
    config: float | MagnitudeConfig,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Generate per-feature magnitude values.

    Args:
        num_features: Number of features
        config: Either a float (constant for all features) or MagnitudeConfig
        seed: Optional random seed for reproducibility (for stochastic generators)

    Returns:
        Tensor of shape (num_features,) with magnitude values
    """
    if isinstance(config, (int, float)):
        return torch.full((num_features,), config, dtype=torch.float32)

    return config.generate(num_features, seed=seed)


# =============================================================================
# Built-in implementations
# =============================================================================


@dataclass
class ConstantMagnitudeConfig(MagnitudeConfig):
    """
    Config for constant magnitude values.

    All features have the same magnitude value.
    """

    value: float = 1.0

    @classmethod
    def generator_name(cls) -> str:
        return "constant"

    def generate(self, num_features: int, seed: int | None = None) -> torch.Tensor:
        return torch.full((num_features,), self.value, dtype=torch.float32)


@dataclass
class LinearMagnitudeConfig(MagnitudeConfig):
    """
    Config for linearly interpolated magnitude values.

    Values interpolate linearly from `start` to `end` across features.
    value_i = start + (end - start) * i / (n-1)

    Both start and end must be positive.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start <= 0 or self.end <= 0:
            raise ValueError("start and end must be positive")

    @classmethod
    def generator_name(cls) -> str:
        return "linear"

    def generate(self, num_features: int, seed: int | None = None) -> torch.Tensor:
        if num_features == 1:
            return torch.tensor([self.start], dtype=torch.float32)
        return torch.linspace(self.start, self.end, num_features)


@dataclass
class ExponentialMagnitudeConfig(MagnitudeConfig):
    """
    Config for exponentially interpolated magnitude values.

    Values interpolate exponentially from `start` to `end` across features.
    value_i = start * (end/start)^(i/(n-1))

    Both start and end must be positive.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start <= 0 or self.end <= 0:
            raise ValueError("start and end must be positive for exponential scale")

    @classmethod
    def generator_name(cls) -> str:
        return "exponential"

    def generate(self, num_features: int, seed: int | None = None) -> torch.Tensor:
        if num_features == 1:
            return torch.tensor([self.start], dtype=torch.float32)
        t = torch.linspace(0, 1, num_features)
        return self.start * (self.end / self.start) ** t


@dataclass
class FoldedNormalMagnitudeConfig(MagnitudeConfig):
    """
    Config for folded normal distributed magnitude values.

    Each feature gets a magnitude sampled from |N(mean, std^2)|.
    The folded normal distribution is the absolute value of a normal distribution,
    producing only positive values.

    Optionally clamp values to [min_value, max_value].
    """

    mean: float = 0.0
    std: float = 0.1
    min_value: float | None = None
    max_value: float | None = None

    def __post_init__(self) -> None:
        if self.std <= 0:
            raise ValueError("std must be positive")
        if self.min_value is not None and self.min_value < 0:
            raise ValueError("min_value must be non-negative")
        if self.max_value is not None and self.max_value <= 0:
            raise ValueError("max_value must be positive")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value must be <= max_value")

    @classmethod
    def generator_name(cls) -> str:
        return "folded_normal"

    def generate(self, num_features: int, seed: int | None = None) -> torch.Tensor:
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        samples = self.mean + torch.randn(num_features, generator=generator) * self.std
        samples = torch.abs(samples)
        if self.min_value is not None or self.max_value is not None:
            samples = torch.clamp(
                samples,
                min=self.min_value,
                max=self.max_value,
            )
        return samples


# =============================================================================
# Register built-in generators
# =============================================================================

register_magnitude("constant", ConstantMagnitudeConfig)
register_magnitude("linear", LinearMagnitudeConfig)
register_magnitude("exponential", ExponentialMagnitudeConfig)
register_magnitude("folded_normal", FoldedNormalMagnitudeConfig)
