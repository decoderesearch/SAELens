"""
Functions for generating synthetic feature activations.
"""

from collections.abc import Callable

import torch
from scipy.stats import norm
from torch import nn
from torch.distributions import MultivariateNormal

from sae_lens.util import str_to_dtype

ActivationsModifier = Callable[[torch.Tensor], torch.Tensor]


class ActivationGenerator(nn.Module):
    """
    Generator for synthetic feature activations.

    This module provides a generator for synthetic feature activations with controlled properties.
    """

    num_features: int
    firing_probabilities: torch.Tensor
    std_firing_magnitudes: torch.Tensor
    mean_firing_magnitudes: torch.Tensor
    modify_activations: ActivationsModifier | None
    correlation_matrix: torch.Tensor | None

    def __init__(
        self,
        num_features: int,
        firing_probabilities: torch.Tensor | float,
        std_firing_magnitudes: torch.Tensor | float = 0.0,
        mean_firing_magnitudes: torch.Tensor | float = 1.0,
        modify_activations: ActivationsModifier | None = None,
        correlation_matrix: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | str = "float32",
    ):
        super().__init__()
        self.num_features = num_features
        self.firing_probabilities = _to_tensor(
            firing_probabilities, num_features, device, dtype
        )
        self.std_firing_magnitudes = _to_tensor(
            std_firing_magnitudes, num_features, device, dtype
        )
        self.mean_firing_magnitudes = _to_tensor(
            mean_firing_magnitudes, num_features, device, dtype
        )
        self.modify_activations = modify_activations
        self.correlation_matrix = correlation_matrix

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Generate a batch of feature activations with controlled properties.

        This is the main function for generating synthetic training data for SAEs.
        Features fire independently according to their firing probabilities unless
        a correlation matrix is provided.

        Args:
            batch_size: Number of samples to generate

        Returns:
            Tensor of shape [batch_size, num_features] with non-negative activations
        """
        device = self.firing_probabilities.device

        if self.correlation_matrix is not None:
            firing_features = _generate_correlated_features(
                batch_size, self.firing_probabilities, self.correlation_matrix, device
            )
        else:
            firing_features = torch.bernoulli(
                self.firing_probabilities.unsqueeze(0).expand(batch_size, -1)
            )

        mean_firing_magnitudes = self.mean_firing_magnitudes.to(device)

        if self.modify_activations is not None:
            firing_features = self.modify_activations(firing_features)

        firing_features = firing_features.to(device)
        firing_magnitude_delta = torch.normal(
            torch.zeros_like(self.firing_probabilities)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(device),
            self.std_firing_magnitudes.unsqueeze(0).expand(batch_size, -1).to(device),
        )
        firing_magnitude_delta[firing_features == 0] = 0
        return (
            firing_features * mean_firing_magnitudes + firing_magnitude_delta
        ).relu()

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.sample(batch_size)


def _generate_correlated_features(
    batch_size: int,
    firing_probabilities: torch.Tensor,
    correlation_matrix: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate correlated binary features using multivariate Gaussian sampling.

    Uses the Gaussian copula approach: sample from a multivariate normal
    distribution, then threshold to get binary features.

    Args:
        batch_size: Number of samples to generate
        firing_probabilities: Marginal probabilities for each feature
        correlation_matrix: Correlation matrix between features
        device: Device to generate samples on

    Returns:
        Binary feature matrix of shape [batch_size, num_features]
    """
    num_features = firing_probabilities.shape[0]

    # Convert marginal probabilities to thresholds using inverse normal CDF
    thresholds = torch.tensor(
        [norm.ppf(1 - p.item()) for p in firing_probabilities], device=device
    )

    mvn = MultivariateNormal(
        loc=torch.zeros(num_features, device=device),
        covariance_matrix=correlation_matrix.to(device),
    )

    gaussian_samples = mvn.sample((batch_size,))
    return (gaussian_samples > thresholds.unsqueeze(0)).float()


def _to_tensor(
    value: torch.Tensor | float,
    num_features: int,
    device: torch.device | str,
    dtype: torch.dtype | str,
) -> torch.Tensor:
    dtype = str_to_dtype(dtype)
    device = torch.device(device)
    if not isinstance(value, torch.Tensor):
        value = value * torch.ones(num_features, device=device, dtype=dtype)
    if value.shape != (num_features,):
        raise ValueError(
            f"Value must be a tensor of shape ({num_features},) or a float. Got {value.shape}"
        )
    return value.to(device, dtype)


# Standalone functions for backward compatibility


def generate_activations(
    batch_size: int,
    firing_probabilities: torch.Tensor,
    std_firing_magnitudes: torch.Tensor | None = None,
    mean_firing_magnitudes: torch.Tensor | None = None,
    modify_activations: ActivationsModifier | None = None,
    correlation_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generate a batch of feature activations with controlled properties.

    This is a standalone function that creates a temporary ActivationGenerator
    and samples from it. For repeated sampling, use the ActivationGenerator class
    directly for better performance.

    Args:
        batch_size: Number of samples to generate
        firing_probabilities: Per-feature firing probability tensor of shape [num_features]
        std_firing_magnitudes: Standard deviation of firing magnitudes (default: 0)
        mean_firing_magnitudes: Mean firing magnitudes when features fire (default: 1)
        modify_activations: Optional function to modify activations after generation
        correlation_matrix: Optional correlation matrix for correlated feature firing

    Returns:
        Tensor of shape [batch_size, num_features] with non-negative activations

    Example:
        >>> probs = torch.tensor([0.3, 0.2, 0.1])
        >>> activations = generate_activations(1000, probs)
        >>> activations.shape
        torch.Size([1000, 3])
    """
    num_features = firing_probabilities.shape[0]
    generator = ActivationGenerator(
        num_features=num_features,
        firing_probabilities=firing_probabilities,
        std_firing_magnitudes=std_firing_magnitudes
        if std_firing_magnitudes is not None
        else 0.0,
        mean_firing_magnitudes=mean_firing_magnitudes
        if mean_firing_magnitudes is not None
        else 1.0,
        modify_activations=modify_activations,
        correlation_matrix=correlation_matrix,
    )
    return generator.sample(batch_size)


def suppress_features(
    dominant_feature: int,
    suppressed_features: list[int],
) -> ActivationsModifier:
    """
    Create a modifier that suppresses certain features when a dominant feature fires.

    When the dominant feature is active, the suppressed features are set to zero.
    This simulates inhibitory relationships between features.

    Args:
        dominant_feature: Index of the feature that dominates
        suppressed_features: Indices of features to suppress when dominant fires

    Returns:
        A modifier function that can be passed to generate_activations

    Example:
        >>> modifier = suppress_features(dominant_feature=0, suppressed_features=[1, 2])
        >>> activations = generate_activations(1000, probs, modify_activations=modifier)
    """

    def modifier(activations: torch.Tensor) -> torch.Tensor:
        result = activations.clone()
        dominant_active = activations[:, dominant_feature] > 0
        for feat_idx in suppressed_features:
            result[dominant_active, feat_idx] = 0
        return result

    return modifier


def absorb_features(
    parent_child_pairs: list[tuple[int, int]],
) -> ActivationsModifier:
    """
    Create a modifier that activates parent features when child features fire.

    For each (parent, child) pair, when the child fires, the parent is also
    activated with magnitude 1. This simulates hierarchical feature relationships.

    Args:
        parent_child_pairs: List of (parent_index, child_index) tuples

    Returns:
        A modifier function that can be passed to generate_activations

    Example:
        >>> modifier = absorb_features([(0, 2), (1, 3)])
        >>> # When feature 2 fires, feature 0 will also fire
        >>> # When feature 3 fires, feature 1 will also fire
    """

    def modifier(activations: torch.Tensor) -> torch.Tensor:
        result = activations.clone()
        for parent_idx, child_idx in parent_child_pairs:
            child_active = activations[:, child_idx] > 0
            result[child_active, parent_idx] = 1.0
        return result

    return modifier


def chain_modifiers(
    modifiers: list[ActivationsModifier],
) -> ActivationsModifier:
    """
    Chain multiple modifiers to apply them in sequence.

    Args:
        modifiers: List of modifier functions to apply in order

    Returns:
        A single modifier function that applies all modifiers in sequence

    Example:
        >>> modifier = chain_modifiers([
        ...     absorb_features([(1, 2)]),
        ...     suppress_features(0, [1]),
        ... ])
    """

    def chained(activations: torch.Tensor) -> torch.Tensor:
        result = activations
        for modifier in modifiers:
            result = modifier(result)
        return result

    return chained
