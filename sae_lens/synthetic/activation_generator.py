"""
Functions for generating synthetic feature activations.
"""

import math
from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch.distributions import MultivariateNormal

from sae_lens.synthetic.correlation import LowRankCorrelationMatrix
from sae_lens.util import str_to_dtype

ActivationsModifier = Callable[[torch.Tensor], torch.Tensor]
ActivationsModifierInput = ActivationsModifier | Sequence[ActivationsModifier] | None
CorrelationMatrixInput = (
    torch.Tensor | LowRankCorrelationMatrix | tuple[torch.Tensor, torch.Tensor]
)


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
    low_rank_correlation: tuple[torch.Tensor, torch.Tensor] | None
    correlation_thresholds: torch.Tensor | None
    use_sparse_tensors: bool

    def __init__(
        self,
        num_features: int,
        firing_probabilities: torch.Tensor | float,
        std_firing_magnitudes: torch.Tensor | float = 0.0,
        mean_firing_magnitudes: torch.Tensor | float = 1.0,
        modify_activations: ActivationsModifierInput = None,
        correlation_matrix: CorrelationMatrixInput | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | str = "float32",
        use_sparse_tensors: bool = False,
    ):
        """
        Create a new ActivationGenerator.

        Args:
            num_features: Number of features to generate activations for.
            firing_probabilities: Probability of each feature firing. Can be a single
                float (applied to all features) or a tensor of shape (num_features,).
            std_firing_magnitudes: Standard deviation of firing magnitudes. Can be a
                single float or a tensor of shape (num_features,). Defaults to 0.0
                (deterministic magnitudes).
            mean_firing_magnitudes: Mean firing magnitude when a feature fires. Can be
                a single float or a tensor of shape (num_features,). Defaults to 1.0.
            modify_activations: Optional function(s) to modify activations after
                generation. Can be a single callable, a sequence of callables (applied
                in order), or None. Useful for applying hierarchy constraints.
            correlation_matrix: Optional correlation structure between features. Can be:

                - A full correlation matrix tensor of shape (num_features, num_features)
                - A LowRankCorrelationMatrix for memory-efficient large-scale correlations
                - A tuple of (factor, diag) tensors representing low-rank structure

            device: Device to place tensors on. Defaults to "cpu".
            dtype: Data type for tensors. Defaults to "float32".
            use_sparse_tensors: If True, return sparse COO tensors from sample().
                Only recommended when using massive numbers of features. Defaults to False.
        """
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
        self.correlation_matrix = None
        self.low_rank_correlation = None
        self.use_sparse_tensors = use_sparse_tensors

        if correlation_matrix is not None:
            if isinstance(correlation_matrix, torch.Tensor):
                # Full correlation matrix
                _validate_correlation_matrix(correlation_matrix, num_features)
                self.correlation_matrix = correlation_matrix
            else:
                # Low-rank correlation (tuple or LowRankCorrelationMatrix)
                correlation_factor, correlation_diag = (
                    correlation_matrix[0],
                    correlation_matrix[1],
                )
                _validate_low_rank_correlation(
                    correlation_factor, correlation_diag, num_features
                )
                # Pre-compute sqrt for efficiency (used every sample call)
                self.low_rank_correlation = (
                    correlation_factor,
                    correlation_diag.sqrt(),
                )

            # Vectorized inverse normal CDF: norm.ppf(1-p) = sqrt(2) * erfinv(1 - 2*p)
            self.correlation_thresholds = math.sqrt(2) * torch.erfinv(
                1 - 2 * self.firing_probabilities
            )

    @torch.no_grad()
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
            firing_indices = _generate_correlated_features(
                batch_size,
                self.correlation_matrix,
                self.correlation_thresholds,
                device,
            )
        elif self.low_rank_correlation is not None:
            assert self.correlation_thresholds is not None
            firing_indices = _generate_low_rank_correlated_features(
                batch_size,
                self.low_rank_correlation[0],
                self.low_rank_correlation[1],
                self.correlation_thresholds,
                device,
            )
        else:
            firing_indices = torch.bernoulli(
                self.firing_probabilities.unsqueeze(0).expand(batch_size, -1)
            ).nonzero(as_tuple=True)

        # Compute activations only at firing positions (sparse optimization)
        feature_indices = firing_indices[1]
        num_firing = feature_indices.shape[0]
        mean_at_firing = self.mean_firing_magnitudes[feature_indices]
        std_at_firing = self.std_firing_magnitudes[feature_indices]
        random_deltas = (
            torch.randn(
                num_firing, device=device, dtype=self.mean_firing_magnitudes.dtype
            )
            * std_at_firing
        )
        activations_at_firing = (mean_at_firing + random_deltas).relu()

        if self.use_sparse_tensors:
            # Return sparse COO tensor
            indices = torch.stack(firing_indices)  # [2, nnz]
            feature_activations = torch.sparse_coo_tensor(
                indices,
                activations_at_firing,
                size=(batch_size, self.num_features),
                device=device,
                dtype=self.mean_firing_magnitudes.dtype,
            )
        else:
            # Dense tensor path
            feature_activations = torch.zeros(
                batch_size,
                self.num_features,
                device=device,
                dtype=self.mean_firing_magnitudes.dtype,
            )
            feature_activations[firing_indices] = activations_at_firing

        if self.modify_activations is not None:
            feature_activations = self.modify_activations(feature_activations)
            if feature_activations.is_sparse:
                # Apply relu to sparse values
                feature_activations = feature_activations.coalesce()
                feature_activations = torch.sparse_coo_tensor(
                    feature_activations.indices(),
                    feature_activations.values().relu(),
                    feature_activations.shape,
                    device=feature_activations.device,
                    dtype=feature_activations.dtype,
                )
            else:
                feature_activations = feature_activations.relu()

        return feature_activations

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.sample(batch_size)


def _generate_correlated_features(
    batch_size: int,
    correlation_matrix: torch.Tensor,
    thresholds: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        Tuple of (row_indices, col_indices) for firing features
    """
    num_features = correlation_matrix.shape[0]

    mvn = MultivariateNormal(
        loc=torch.zeros(num_features, device=device, dtype=thresholds.dtype),
        covariance_matrix=correlation_matrix.to(device=device, dtype=thresholds.dtype),
    )

    gaussian_samples = mvn.sample((batch_size,))
    indices = (gaussian_samples > thresholds.unsqueeze(0)).nonzero(as_tuple=True)
    return indices[0], indices[1]


def _generate_low_rank_correlated_features(
    batch_size: int,
    correlation_factor: torch.Tensor,
    cov_diag_sqrt: torch.Tensor,
    thresholds: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate correlated binary features using low-rank multivariate Gaussian sampling.

    Uses the Gaussian copula approach with a low-rank covariance structure for scalability.
    The covariance is represented as: cov = factor @ factor.T + diag(diag_term)

    Args:
        batch_size: Number of samples to generate
        correlation_factor: Factor matrix of shape (num_features, rank)
        cov_diag_sqrt: Pre-computed sqrt of diagonal term, shape (num_features,)
        thresholds: Pre-computed thresholds for each feature (from inverse normal CDF)
        device: Device to generate samples on

    Returns:
        Tuple of (row_indices, col_indices) for firing features
    """
    # Manual low-rank MVN sampling to enable autocast for the expensive matmul
    # samples = eps @ cov_factor.T + eta * sqrt(cov_diag)
    # where eps ~ N(0, I_rank) and eta ~ N(0, I_n)

    num_features, rank = correlation_factor.shape

    # Generate random samples in float32 for numerical stability
    eps = torch.randn(batch_size, rank, device=device, dtype=correlation_factor.dtype)
    eta = torch.randn(
        batch_size, num_features, device=device, dtype=cov_diag_sqrt.dtype
    )

    gaussian_samples = eps @ correlation_factor.T + eta * cov_diag_sqrt

    indices = (gaussian_samples > thresholds.unsqueeze(0)).nonzero(as_tuple=True)
    return indices[0], indices[1]


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
        num_features: Expected number of features (matrix should be (num_features, num_features))

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


def _validate_low_rank_correlation(
    correlation_factor: torch.Tensor,
    correlation_diag: torch.Tensor,
    num_features: int,
) -> None:
    """Validate that low-rank correlation parameters have correct properties.

    Args:
        correlation_factor: Factor matrix of shape (num_features, rank)
        correlation_diag: Diagonal term of shape (num_features,)
        num_features: Expected number of features

    Raises:
        ValueError: If shapes are incorrect or diagonal terms are not positive
    """
    if correlation_factor.ndim != 2:
        raise ValueError(
            f"correlation_factor must be 2D, got {correlation_factor.ndim}D"
        )
    if correlation_factor.shape[0] != num_features:
        raise ValueError(
            f"correlation_factor must have shape ({num_features}, rank), "
            f"got {tuple(correlation_factor.shape)}"
        )
    if correlation_diag.shape != (num_features,):
        raise ValueError(
            f"correlation_diag must have shape ({num_features},), "
            f"got {tuple(correlation_diag.shape)}"
        )
    if torch.any(correlation_diag <= 0):
        raise ValueError("correlation_diag must have all positive values")
