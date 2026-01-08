"""
Functions for generating synthetic feature activations.
"""

from collections.abc import Callable, Sequence

import torch
from scipy.stats import norm
from torch import nn
from torch.distributions import MultivariateNormal

from sae_lens.util import str_to_dtype

ActivationsModifier = Callable[[torch.Tensor], torch.Tensor]
ActivationsModifierInput = ActivationsModifier | Sequence[ActivationsModifier] | None


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
    correlation_thresholds: torch.Tensor | None

    def __init__(
        self,
        num_features: int,
        firing_probabilities: torch.Tensor | float,
        std_firing_magnitudes: torch.Tensor | float = 0.0,
        mean_firing_magnitudes: torch.Tensor | float = 1.0,
        modify_activations: ActivationsModifierInput = None,
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
        self.modify_activations = _normalize_modifiers(modify_activations)
        self.correlation_thresholds = None
        if correlation_matrix is not None:
            _validate_correlation_matrix(correlation_matrix, num_features)
            self.correlation_thresholds = torch.tensor(
                [norm.ppf(1 - p.item()) for p in self.firing_probabilities],
                device=device,
                dtype=self.firing_probabilities.dtype,
            )
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
        # All tensors (firing_probabilities, std_firing_magnitudes, mean_firing_magnitudes)
        # are on the same device from __init__ via _to_tensor()
        device = self.firing_probabilities.device

        if self.correlation_matrix is not None:
            assert self.correlation_thresholds is not None
            firing_features = _generate_correlated_features(
                batch_size,
                self.correlation_matrix,
                self.correlation_thresholds,
                device,
            )
        else:
            firing_features = torch.bernoulli(
                self.firing_probabilities.unsqueeze(0).expand(batch_size, -1)
            )

        firing_magnitude_delta = torch.normal(
            torch.zeros_like(self.firing_probabilities)
            .unsqueeze(0)
            .expand(batch_size, -1),
            self.std_firing_magnitudes.unsqueeze(0).expand(batch_size, -1),
        )
        firing_magnitude_delta[firing_features == 0] = 0
        feature_activations = (
            firing_features * self.mean_firing_magnitudes + firing_magnitude_delta
        ).relu()

        if self.modify_activations is not None:
            feature_activations = self.modify_activations(feature_activations).relu()
        return feature_activations

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.sample(batch_size)


def _generate_correlated_features(
    batch_size: int,
    correlation_matrix: torch.Tensor,
    thresholds: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate correlated binary features using multivariate Gaussian sampling.

    Uses the Gaussian copula approach: sample from a multivariate normal
    distribution, then threshold to get binary features.

    Args:
        batch_size: Number of samples to generate
        correlation_matrix: Correlation matrix between features
        thresholds: Pre-computed thresholds for each feature (from inverse normal CDF)
        device: Device to generate samples on

    Returns:
        Binary feature matrix of shape [batch_size, num_features]
    """
    num_features = correlation_matrix.shape[0]

    mvn = MultivariateNormal(
        loc=torch.zeros(num_features, device=device, dtype=thresholds.dtype),
        covariance_matrix=correlation_matrix.to(device=device, dtype=thresholds.dtype),
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


def _normalize_modifiers(
    modify_activations: ActivationsModifierInput,
) -> ActivationsModifier | None:
    """Convert modifier input to a single modifier or None."""
    if modify_activations is None:
        return None
    if callable(modify_activations):
        return modify_activations
    # It's a sequence of modifiers - chain them
    modifiers = list(modify_activations)
    if len(modifiers) == 0:
        return None
    if len(modifiers) == 1:
        return modifiers[0]

    def chained(activations: torch.Tensor) -> torch.Tensor:
        result = activations
        for modifier in modifiers:
            result = modifier(result)
        return result

    return chained


def _validate_correlation_matrix(
    correlation_matrix: torch.Tensor, num_features: int
) -> None:
    """Validate that a correlation matrix has correct properties.

    Args:
        correlation_matrix: The matrix to validate
        num_features: Expected number of features (matrix should be [num_features, num_features])

    Raises:
        ValueError: If the matrix has incorrect shape, non-unit diagonal, or is not positive definite
    """
    expected_shape = (num_features, num_features)
    if correlation_matrix.shape != expected_shape:
        raise ValueError(
            f"Correlation matrix must have shape {expected_shape}, "
            f"got {tuple(correlation_matrix.shape)}"
        )

    diagonal = torch.diag(correlation_matrix)
    if not torch.allclose(diagonal, torch.ones_like(diagonal)):
        raise ValueError("Correlation matrix diagonal must be all 1s")

    try:
        torch.linalg.cholesky(correlation_matrix)
    except RuntimeError as e:
        raise ValueError("Correlation matrix must be positive definite") from e
