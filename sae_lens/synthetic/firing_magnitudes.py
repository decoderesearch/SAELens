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
        cfg_class, _ = MAGNITUDE_REGISTRY.get_or_raise(name)
        return cfg_class(**d)


class MagnitudeGenerator(ABC):
    """Base class for generating magnitude values."""

    @abstractmethod
    def generate(self, num_features: int) -> torch.Tensor:
        """
        Generate magnitude values.

        Args:
            num_features: Number of features to generate magnitudes for

        Returns:
            Tensor of shape (num_features,) with magnitude values
        """
        ...


def register_magnitude(
    name: str,
    config_class: type[MagnitudeConfig],
    generator_class: type[MagnitudeGenerator],
) -> None:
    """
    Register a magnitude generator with its config.

    Args:
        name: Unique name for this magnitude type
        config_class: Config dataclass for this generator
        generator_class: Generator class that produces magnitudes
    """
    MAGNITUDE_REGISTRY.register(name, (config_class, generator_class))


def get_magnitude_class(
    name: str,
) -> tuple[type[MagnitudeConfig], type[MagnitudeGenerator]]:
    """
    Get the config and generator classes for a magnitude type.

    Args:
        name: Name of the magnitude type

    Returns:
        Tuple of (config_class, generator_class)
    """
    return MAGNITUDE_REGISTRY.get_or_raise(name)


def generate_magnitudes(
    num_features: int,
    config: float | MagnitudeConfig,
) -> torch.Tensor:
    """
    Generate per-feature magnitude values.

    Args:
        num_features: Number of features
        config: Either a float (constant for all features) or MagnitudeConfig

    Returns:
        Tensor of shape (num_features,) with magnitude values
    """
    if isinstance(config, (int, float)):
        return torch.full((num_features,), float(config), dtype=torch.float32)

    _, generator_class = get_magnitude_class(config.generator_name())
    generator = generator_class(config)  # type: ignore[call-arg]
    return generator.generate(num_features)


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


class ConstantMagnitudeGenerator(MagnitudeGenerator):
    """Generator for constant magnitude values."""

    def __init__(self, cfg: ConstantMagnitudeConfig):
        self.cfg = cfg

    def generate(self, num_features: int) -> torch.Tensor:
        return torch.full((num_features,), self.cfg.value, dtype=torch.float32)


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


class LinearMagnitudeGenerator(MagnitudeGenerator):
    """Generator for linearly interpolated magnitude values."""

    def __init__(self, cfg: LinearMagnitudeConfig):
        self.cfg = cfg

    def generate(self, num_features: int) -> torch.Tensor:
        if num_features == 1:
            return torch.tensor([self.cfg.start], dtype=torch.float32)
        return torch.linspace(self.cfg.start, self.cfg.end, num_features)


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


class ExponentialMagnitudeGenerator(MagnitudeGenerator):
    """Generator for exponentially interpolated magnitude values."""

    def __init__(self, cfg: ExponentialMagnitudeConfig):
        self.cfg = cfg

    def generate(self, num_features: int) -> torch.Tensor:
        if num_features == 1:
            return torch.tensor([self.cfg.start], dtype=torch.float32)
        t = torch.linspace(0, 1, num_features)
        return self.cfg.start * (self.cfg.end / self.cfg.start) ** t


# =============================================================================
# Register built-in generators
# =============================================================================

register_magnitude("constant", ConstantMagnitudeConfig, ConstantMagnitudeGenerator)
register_magnitude("linear", LinearMagnitudeConfig, LinearMagnitudeGenerator)
register_magnitude(
    "exponential", ExponentialMagnitudeConfig, ExponentialMagnitudeGenerator
)
