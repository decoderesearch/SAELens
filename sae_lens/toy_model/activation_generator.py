"""
Functions for generating synthetic feature activations.

This module provides utilities for generating batches of feature activations
with controlled properties like:
- Firing probabilities (sparsity)
- Firing magnitudes
- Feature correlations
- Feature suppression and absorption patterns
"""

from collections.abc import Callable

import torch
from scipy.stats import norm
from torch.distributions import MultivariateNormal


def generate_activations(
    batch_size: int,
    firing_probabilities: torch.Tensor,
    std_firing_magnitudes: torch.Tensor | None = None,
    mean_firing_magnitudes: torch.Tensor | None = None,
    device: torch.device | None = None,
    modify_activations: Callable[[torch.Tensor], torch.Tensor] | None = None,
    correlation_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generate a batch of feature activations with controlled properties.

    This is the main function for generating synthetic training data for SAEs.
    Features fire independently according to their firing probabilities unless
    a correlation matrix is provided.

    Args:
        batch_size: Number of samples to generate
        firing_probabilities: Tensor of shape [num_features] with marginal
            probability each feature fires (between 0 and 1)
        std_firing_magnitudes: Optional tensor of shape [num_features] with
            standard deviation of firing magnitudes. Default is 0 (deterministic).
        mean_firing_magnitudes: Optional tensor of shape [num_features] with
            mean firing magnitudes. Default is 1.0 for all features.
        device: Device to generate activations on. Default uses CPU.
        modify_activations: Optional callback to modify activations after generation.
            Useful for implementing feature suppression, absorption, etc.
        correlation_matrix: Optional correlation matrix of shape [num_features, num_features].
            If provided, features are sampled with these correlations.
            Must be positive semi-definite.

    Returns:
        Tensor of shape [batch_size, num_features] with non-negative activations

    Example:
        >>> # Generate 1000 samples with 5 features
        >>> probs = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.05])
        >>> activations = generate_activations(1000, probs)
        >>> activations.shape
        torch.Size([1000, 5])

        >>> # With varying magnitudes
        >>> means = torch.tensor([1.0, 2.0, 1.5, 0.8, 1.0])
        >>> stds = torch.tensor([0.1, 0.2, 0.1, 0.05, 0.1])
        >>> activations = generate_activations(1000, probs, stds, means)

        >>> # With correlated features
        >>> corr = create_correlation_matrix(5, correlations={(0, 1): 0.8})
        >>> activations = generate_activations(1000, probs, correlation_matrix=corr)
    """
    if device is None:
        device = firing_probabilities.device

    if correlation_matrix is not None:
        firing_features = _generate_correlated_features(
            batch_size, firing_probabilities, correlation_matrix, device
        )
    else:
        firing_features = torch.bernoulli(
            firing_probabilities.unsqueeze(0).expand(batch_size, -1).to(device)
        )

    if std_firing_magnitudes is None:
        std_firing_magnitudes = torch.zeros_like(firing_probabilities)
    if mean_firing_magnitudes is None:
        mean_firing_magnitudes = torch.ones_like(firing_probabilities)

    mean_firing_magnitudes = mean_firing_magnitudes.to(device)

    if modify_activations is not None:
        firing_features = modify_activations(firing_features)

    firing_features = firing_features.to(device)
    firing_magnitude_delta = torch.normal(
        torch.zeros_like(firing_probabilities)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .to(device),
        std_firing_magnitudes.unsqueeze(0).expand(batch_size, -1).to(device),
    )
    firing_magnitude_delta[firing_features == 0] = 0
    return (firing_features * mean_firing_magnitudes + firing_magnitude_delta).relu()


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


def create_correlation_matrix(
    num_features: int,
    correlations: dict[tuple[int, int], float] | None = None,
    default_correlation: float = 0.0,
) -> torch.Tensor:
    """
    Create a correlation matrix with specified pairwise correlations.

    Args:
        num_features: Number of features
        correlations: Dict mapping (i, j) pairs to correlation values.
            Pairs should have i < j.
        default_correlation: Default correlation for unspecified pairs

    Returns:
        Correlation matrix of shape [num_features, num_features]

    Example:
        >>> # Create matrix where features 0,1 are correlated and 2,3 are anti-correlated
        >>> corr = create_correlation_matrix(
        ...     num_features=4,
        ...     correlations={(0, 1): 0.8, (2, 3): -0.5}
        ... )
    """
    matrix = torch.eye(num_features) + default_correlation * (
        1 - torch.eye(num_features)
    )

    if correlations is not None:
        for (i, j), corr in correlations.items():
            matrix[i, j] = corr
            matrix[j, i] = corr

    # Check positive definiteness and fix if necessary
    eigenvals = torch.linalg.eigvals(matrix)
    if torch.any(eigenvals.real < -1e-6):
        matrix = _fix_correlation_matrix(matrix)

    return matrix


def _fix_correlation_matrix(
    matrix: torch.Tensor, min_eigenval: float = 1e-6
) -> torch.Tensor:
    """Fix a correlation matrix to be positive semi-definite."""
    eigenvals, eigenvecs = torch.linalg.eigh(matrix)
    eigenvals = torch.clamp(eigenvals, min=min_eigenval)
    fixed_matrix = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T

    diag_vals = torch.diag(fixed_matrix)
    fixed_matrix = fixed_matrix / torch.sqrt(
        diag_vals.unsqueeze(0) * diag_vals.unsqueeze(1)
    )
    fixed_matrix.fill_diagonal_(1.0)

    return fixed_matrix


def create_block_correlation_matrix(
    block_sizes: list[int],
    within_block_correlation: float = 0.8,
    between_block_correlation: float = 0.1,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Create a block-structured correlation matrix.

    Useful for modeling groups of related features that are internally
    correlated but less correlated with other groups.

    Args:
        block_sizes: List of sizes for each block (e.g., [3, 4, 2] for 3 blocks)
        within_block_correlation: Correlation strength within blocks
        between_block_correlation: Correlation strength between blocks
        seed: Random seed for reproducibility

    Returns:
        Block-structured correlation matrix

    Example:
        >>> # Create 3 groups of features: 4, 3, and 3 features each
        >>> matrix = create_block_correlation_matrix(
        ...     block_sizes=[4, 3, 3],
        ...     within_block_correlation=0.8,
        ...     between_block_correlation=0.1
        ... )
    """
    if seed is not None:
        torch.manual_seed(seed)

    total_features = sum(block_sizes)
    matrix = torch.eye(total_features)

    # Fill in within-block correlations
    start_idx = 0
    for block_size in block_sizes:
        end_idx = start_idx + block_size
        for i in range(start_idx, end_idx):
            for j in range(i + 1, end_idx):
                noise = torch.randn(1).item() * 0.05
                corr = max(-0.99, min(0.99, within_block_correlation + noise))
                matrix[i, j] = corr
                matrix[j, i] = corr
        start_idx = end_idx

    # Fill in between-block correlations
    block_starts = [0] + [
        sum(block_sizes[: i + 1]) for i in range(len(block_sizes) - 1)
    ]

    for i, (start_i, size_i) in enumerate(zip(block_starts, block_sizes)):
        for j, (start_j, size_j) in enumerate(zip(block_starts, block_sizes)):
            if i < j:
                for fi in range(start_i, start_i + size_i):
                    for fj in range(start_j, start_j + size_j):
                        noise = torch.randn(1).item() * 0.02
                        corr = max(-0.99, min(0.99, between_block_correlation + noise))
                        matrix[fi, fj] = corr
                        matrix[fj, fi] = corr

    # Ensure positive semi-definiteness
    eigenvals, eigenvecs = torch.linalg.eigh(matrix)
    eigenvals = torch.clamp(eigenvals, min=1e-6)
    matrix = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T

    # Renormalize to correlation matrix
    diag_sqrt = torch.sqrt(torch.diag(matrix))
    matrix = matrix / (diag_sqrt.unsqueeze(0) * diag_sqrt.unsqueeze(1))
    matrix.fill_diagonal_(1.0)

    return matrix


# Feature modifiers for creating structured activation patterns


def suppress_features(
    dominant_feature: int, suppressed_features: list[int]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a modifier that suppresses certain features when a dominant feature fires.

    Useful for modeling mutually exclusive or antagonistic features.

    Args:
        dominant_feature: Index of the dominant feature
        suppressed_features: Indices of features to suppress when dominant fires

    Returns:
        Modifier function to pass to generate_activations

    Example:
        >>> # When feature 0 fires, suppress features 1 and 2
        >>> modifier = suppress_features(0, [1, 2])
        >>> activations = generate_activations(
        ...     1000, probs, modify_activations=modifier
        ... )
    """

    def suppress_fn(feats: torch.Tensor) -> torch.Tensor:
        for suppressed in suppressed_features:
            feats[:, suppressed] = torch.where(
                feats[:, dominant_feature] == 1,
                torch.tensor(0.0, device=feats.device),
                feats[:, suppressed],
            )
        return feats

    return suppress_fn


def absorb_features(
    absorption_pairs: list[tuple[int, int]],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a modifier that makes certain features absorb others.

    When a child feature fires, the parent is also set to fire.
    This models hierarchical feature relationships.

    Args:
        absorption_pairs: List of (parent, child) tuples. When child fires,
            parent is set to fire.

    Returns:
        Modifier function to pass to generate_activations

    Example:
        >>> # When feature 2 fires, feature 0 also fires (feature 0 absorbs 2)
        >>> modifier = absorb_features([(0, 2), (1, 3)])
        >>> activations = generate_activations(
        ...     1000, probs, modify_activations=modifier
        ... )
    """

    def absorb_fn(feats: torch.Tensor) -> torch.Tensor:
        for parent, child in absorption_pairs:
            feats[:, parent] = torch.where(
                feats[:, child] == 1,
                torch.tensor(1.0, device=feats.device),
                feats[:, parent],
            )
        return feats

    return absorb_fn


def chain_modifiers(
    modifiers: list[Callable[[torch.Tensor], torch.Tensor]],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Chain multiple modifiers together.

    Modifiers are applied in order.

    Args:
        modifiers: List of modifier functions

    Returns:
        Combined modifier function

    Example:
        >>> modifier = chain_modifiers([
        ...     absorb_features([(0, 2)]),
        ...     suppress_features(0, [1])
        ... ])
    """

    def chain_fn(feats: torch.Tensor) -> torch.Tensor:
        for modifier in modifiers:
            feats = modifier(feats)
        return feats

    return chain_fn
