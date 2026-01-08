import random

import torch


def create_correlation_matrix_from_correlations(
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
    fixed_matrix = fixed_matrix / torch.sqrt(
        diag_vals.unsqueeze(0) * diag_vals.unsqueeze(1)
    )
    fixed_matrix.fill_diagonal_(1.0)

    return fixed_matrix


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
        positive_ratio: Fraction of correlations that should be positive (0.0 to 1.0)
        uncorrelated_ratio: Fraction of feature pairs that should remain uncorrelated (0.0 to 1.0)
        min_correlation_strength: Minimum absolute correlation strength
        max_correlation_strength: Maximum absolute correlation strength
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping (i, j) pairs to correlation values
    """
    # Use local random number generator to avoid side effects on global state
    rng = random.Random(seed)

    # Validate inputs
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
) -> torch.Tensor:
    """
    Generate a random correlation matrix with specified constraints.

    This is a convenience function that combines generate_random_correlations()
    and create_correlation_matrix_from_correlations() into a single call.

    Args:
        num_features: Number of features
        positive_ratio: Fraction of correlations that should be positive (0.0 to 1.0)
        uncorrelated_ratio: Fraction of feature pairs that should remain uncorrelated (0.0 to 1.0)
        min_correlation_strength: Minimum absolute correlation strength
        max_correlation_strength: Maximum absolute correlation strength
        seed: Random seed for reproducibility

    Returns:
        Random correlation matrix of shape [num_features, num_features]
    """
    # Generate random correlations
    correlations = generate_random_correlations(
        num_features=num_features,
        positive_ratio=positive_ratio,
        uncorrelated_ratio=uncorrelated_ratio,
        min_correlation_strength=min_correlation_strength,
        max_correlation_strength=max_correlation_strength,
        seed=seed,
    )

    # Create and return correlation matrix
    return create_correlation_matrix_from_correlations(
        num_features=num_features, correlations=correlations
    )
