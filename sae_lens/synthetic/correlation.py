import random
from typing import NamedTuple

import torch

from sae_lens.util import str_to_dtype


class LowRankCorrelationMatrix(NamedTuple):
    """
    Low-rank representation of a correlation matrix for scalable correlated sampling.

    The correlation structure is represented as:
        correlation = correlation_factor @ correlation_factor.T + diag(correlation_diag)

    This requires O(num_features * rank) storage instead of O(num_features^2),
    making it suitable for very large numbers of features (e.g., 1M+).

    Attributes:
        correlation_factor: Factor matrix of shape (num_features, rank) that captures
            correlations through shared latent factors.
        correlation_diag: Diagonal variance term of shape (num_features,). Should be
            chosen such that the diagonal of the full correlation matrix equals 1.
            Typically: correlation_diag[i] = 1 - sum(correlation_factor[i, :]^2)
    """

    correlation_factor: torch.Tensor
    correlation_diag: torch.Tensor


def create_correlation_matrix_from_correlations(
    num_features: int,
    correlations: dict[tuple[int, int], float] | None = None,
    default_correlation: float = 0.0,
) -> torch.Tensor:
    """
    Create a correlation matrix with specified pairwise correlations.

    Note: If the resulting matrix is not positive definite, it will be adjusted
    to ensure validity. This adjustment may change the specified correlation
    values. To minimize this effect, use smaller correlation magnitudes.

    Args:
        num_features: Number of features
        correlations: Dict mapping (i, j) pairs to correlation values.
            Pairs should have i < j. Pairs not specified will use default_correlation.
        default_correlation: Default correlation for unspecified pairs

    Returns:
        Correlation matrix of shape (num_features, num_features)
    """
    matrix = torch.eye(num_features) + default_correlation * (
        1 - torch.eye(num_features)
    )

    if correlations is not None:
        for (i, j), corr in correlations.items():
            matrix[i, j] = corr
            matrix[j, i] = corr

    # Ensure matrix is symmetric (numerical precision)
    matrix = (matrix + matrix.T) / 2

    # Check positive definiteness and fix if necessary
    # Use eigvalsh for symmetric matrices (returns real eigenvalues)
    eigenvals = torch.linalg.eigvalsh(matrix)
    if torch.any(eigenvals < -1e-6):
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
    diag_vals = torch.clamp(diag_vals, min=1e-8)  # Prevent division by zero
    fixed_matrix = fixed_matrix / torch.sqrt(
        diag_vals.unsqueeze(0) * diag_vals.unsqueeze(1)
    )
    fixed_matrix.fill_diagonal_(1.0)

    return fixed_matrix


def _validate_correlation_params(
    positive_ratio: float,
    uncorrelated_ratio: float,
    min_correlation_strength: float,
    max_correlation_strength: float,
) -> None:
    """Validate parameters for correlation generation."""
    if not 0.0 <= positive_ratio <= 1.0:
        raise ValueError("positive_ratio must be between 0.0 and 1.0")
    if not 0.0 <= uncorrelated_ratio <= 1.0:
        raise ValueError("uncorrelated_ratio must be between 0.0 and 1.0")
    if min_correlation_strength < 0:
        raise ValueError("min_correlation_strength must be non-negative")
    if max_correlation_strength > 1.0:
        raise ValueError("max_correlation_strength must be <= 1.0")
    if min_correlation_strength > max_correlation_strength:
        raise ValueError("min_correlation_strength must be <= max_correlation_strength")


def generate_random_correlations(
    num_features: int,
    positive_ratio: float = 0.5,
    uncorrelated_ratio: float = 0.3,
    min_correlation_strength: float = 0.1,
    max_correlation_strength: float = 0.8,
    seed: int | None = None,
) -> dict[tuple[int, int], float]:
    """
    Generate random correlations between features with specified constraints.

    Args:
        num_features: Number of features
        positive_ratio: Fraction of correlated pairs that should be positive (0.0 to 1.0)
        uncorrelated_ratio: Fraction of feature pairs that should have zero correlation
            (0.0 to 1.0). These pairs are omitted from the returned dictionary.
        min_correlation_strength: Minimum absolute correlation strength for correlated pairs
        max_correlation_strength: Maximum absolute correlation strength for correlated pairs
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping (i, j) pairs to correlation values. Pairs with zero
        correlation (determined by uncorrelated_ratio) are not included.
    """
    # Use local random number generator to avoid side effects on global state
    rng = random.Random(seed)

    _validate_correlation_params(
        positive_ratio,
        uncorrelated_ratio,
        min_correlation_strength,
        max_correlation_strength,
    )

    # Generate all possible feature pairs (i, j) where i < j
    all_pairs = [
        (i, j) for i in range(num_features) for j in range(i + 1, num_features)
    ]
    total_pairs = len(all_pairs)

    if total_pairs == 0:
        return {}

    # Determine how many pairs to correlate vs leave uncorrelated
    num_uncorrelated = int(total_pairs * uncorrelated_ratio)
    num_correlated = total_pairs - num_uncorrelated

    # Randomly select which pairs to correlate
    correlated_pairs = rng.sample(all_pairs, num_correlated)

    # For correlated pairs, determine positive vs negative
    num_positive = int(num_correlated * positive_ratio)
    num_negative = num_correlated - num_positive

    # Assign signs
    signs = [1] * num_positive + [-1] * num_negative
    rng.shuffle(signs)

    # Generate correlation strengths
    correlations = {}
    for pair, sign in zip(correlated_pairs, signs):
        # Sample correlation strength uniformly from range
        strength = rng.uniform(min_correlation_strength, max_correlation_strength)
        correlations[pair] = sign * strength

    return correlations


def generate_random_correlation_matrix(
    num_features: int,
    positive_ratio: float = 0.5,
    uncorrelated_ratio: float = 0.3,
    min_correlation_strength: float = 0.1,
    max_correlation_strength: float = 0.8,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | str = torch.float32,
) -> torch.Tensor:
    """
    Generate a random correlation matrix with specified constraints.

    Uses vectorized torch operations for efficiency with large numbers of features.

    Note: If the randomly generated matrix is not positive definite, it will be
    adjusted to ensure validity. This adjustment may change correlation values,
    including turning some zero correlations into non-zero values. To minimize
    this effect, use smaller correlation strengths (e.g., 0.01-0.1).

    Args:
        num_features: Number of features
        positive_ratio: Fraction of correlated pairs that should be positive (0.0 to 1.0)
        uncorrelated_ratio: Fraction of feature pairs that should have zero correlation
            (0.0 to 1.0). Note that matrix fixing for positive definiteness may reduce
            the actual number of zero correlations.
        min_correlation_strength: Minimum absolute correlation strength for correlated pairs
        max_correlation_strength: Maximum absolute correlation strength for correlated pairs
        seed: Random seed for reproducibility
        device: Device to create the matrix on
        dtype: Data type for the matrix

    Returns:
        Random correlation matrix of shape (num_features, num_features)
    """
    dtype = str_to_dtype(dtype)
    _validate_correlation_params(
        positive_ratio,
        uncorrelated_ratio,
        min_correlation_strength,
        max_correlation_strength,
    )

    if num_features <= 1:
        return torch.eye(num_features, device=device, dtype=dtype)

    # Set random seed if provided
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    # Get upper triangular indices (i < j)
    row_idx, col_idx = torch.triu_indices(num_features, num_features, offset=1)
    num_pairs = row_idx.shape[0]

    # Generate random values for all pairs at once
    # is_correlated: 1 if this pair should have a correlation, 0 otherwise
    is_correlated = (
        torch.rand(num_pairs, generator=generator, device=device) >= uncorrelated_ratio
    )

    # signs: +1 for positive correlation, -1 for negative
    is_positive = (
        torch.rand(num_pairs, generator=generator, device=device) < positive_ratio
    )
    signs = torch.where(is_positive, 1.0, -1.0)

    # strengths: uniform in [min_strength, max_strength]
    strengths = (
        torch.rand(num_pairs, generator=generator, device=device, dtype=dtype)
        * (max_correlation_strength - min_correlation_strength)
        + min_correlation_strength
    )

    # Combine: correlation = is_correlated * sign * strength
    correlations = is_correlated.to(dtype) * signs.to(dtype) * strengths

    # Build the symmetric matrix
    matrix = torch.eye(num_features, device=device, dtype=dtype)
    matrix[row_idx, col_idx] = correlations
    matrix[col_idx, row_idx] = correlations

    # Check positive definiteness and fix if necessary
    eigenvals = torch.linalg.eigvalsh(matrix)
    if torch.any(eigenvals < -1e-6):
        matrix = _fix_correlation_matrix(matrix)

    return matrix


def generate_random_low_rank_correlation_matrix(
    num_features: int,
    rank: int,
    correlation_scale: float = 0.1,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | str = torch.float32,
) -> LowRankCorrelationMatrix:
    """
    Generate a random low-rank correlation structure for scalable correlated sampling.

    The correlation structure is represented as:
        correlation = factor @ factor.T + diag(diag_term)

    This requires O(num_features * rank) storage instead of O(num_features^2),
    making it suitable for very large numbers of features (e.g., 1M+).

    The factor matrix is initialized with random values scaled by correlation_scale,
    and the diagonal term is computed to ensure the implied correlation matrix has
    unit diagonal.

    Args:
        num_features: Number of features
        rank: Rank of the low-rank approximation. Higher rank allows more complex
            correlation structures but uses more memory. Typical values: 10-100.
        correlation_scale: Scale factor for random correlations. Larger values produce
            stronger correlations between features. Use 0 for no correlations (identity
            matrix). Should be small enough that rank * correlation_scale^2 < 1 to
            ensure valid diagonal terms.
        seed: Random seed for reproducibility
        device: Device to create tensors on
        dtype: Data type for tensors

    Returns:
        LowRankCorrelationMatrix containing the factor matrix and diagonal term
    """
    # Minimum diagonal value to ensure numerical stability in the covariance matrix.
    # This limits how much variance can come from the low-rank factor.
    _MIN_DIAG = 0.01

    dtype = str_to_dtype(dtype)
    device = torch.device(device)

    if rank <= 0:
        raise ValueError("rank must be positive")
    if correlation_scale < 0:
        raise ValueError("correlation_scale must be non-negative")

    # Set random seed if provided
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    # Generate random factor matrix
    # Each row has norm roughly sqrt(rank) * correlation_scale
    factor = (
        torch.randn(num_features, rank, generator=generator, device=device, dtype=dtype)
        * correlation_scale
    )

    # Compute diagonal term to ensure unit diagonal in implied correlation matrix
    # diag(factor @ factor.T) + diag_term = 1
    # diag_term = 1 - sum(factor[i, :]^2)
    factor_sq_sum = (factor**2).sum(dim=1)
    diag_term = 1 - factor_sq_sum

    # Ensure diagonal terms are at least _MIN_DIAG for numerical stability
    # If any diagonal term is too small, scale down the factor matrix
    if torch.any(diag_term < _MIN_DIAG):
        # Scale factor so max row norm squared is at most (1 - _MIN_DIAG)
        # This ensures all diagonal terms are >= _MIN_DIAG
        max_factor_contribution = 1 - _MIN_DIAG
        max_sq_sum = factor_sq_sum.max()
        scale = torch.sqrt(
            torch.tensor(max_factor_contribution, device=device, dtype=dtype)
            / max_sq_sum
        )
        factor = factor * scale
        factor_sq_sum = (factor**2).sum(dim=1)
        diag_term = 1 - factor_sq_sum

    return LowRankCorrelationMatrix(
        correlation_factor=factor, correlation_diag=diag_term
    )
