import random

import torch


def create_correlation_matrix(
    num_features: int,
    correlations: dict[tuple[int, int], float] | None = None,
    default_correlation: float = 0.0,
) -> torch.Tensor:
    """
    Create a correlation matrix with specified pairwise correlations.

    This is an alias for create_correlation_matrix_from_correlations.

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
    return create_correlation_matrix_from_correlations(
        num_features, correlations, default_correlation
    )


def create_block_correlation_matrix(
    block_sizes: list[int],
    within_block_correlation: float = 0.5,
    between_block_correlation: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Create a correlation matrix with block structure.

    Features are grouped into blocks. Within each block, features have the
    specified within-block correlation. Between blocks, features have the
    specified between-block correlation.

    Args:
        block_sizes: List of block sizes. Total features = sum(block_sizes).
        within_block_correlation: Correlation between features in the same block
        between_block_correlation: Correlation between features in different blocks
        seed: Random seed for reproducibility

    Returns:
        Correlation matrix of shape [num_features, num_features]

    Example:
        >>> # Create 2 blocks of 3 features each
        >>> corr = create_block_correlation_matrix(
        ...     block_sizes=[3, 3],
        ...     within_block_correlation=0.8,
        ...     between_block_correlation=0.1
        ... )
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    num_features = sum(block_sizes)
    correlations: dict[tuple[int, int], float] = {}

    # Assign feature indices to blocks
    block_start = 0
    feature_to_block = {}
    for block_idx, block_size in enumerate(block_sizes):
        for i in range(block_start, block_start + block_size):
            feature_to_block[i] = block_idx
        block_start += block_size

    # Generate correlations for all pairs
    for i in range(num_features):
        for j in range(i + 1, num_features):
            if feature_to_block[i] == feature_to_block[j]:
                # Within block - add some noise to avoid singular matrices
                noise = random.uniform(-0.05, 0.05)
                correlations[(i, j)] = within_block_correlation + noise
            else:
                # Between blocks - add some noise
                noise = random.uniform(-0.05, 0.05)
                correlations[(i, j)] = between_block_correlation + noise

    return create_correlation_matrix_from_correlations(num_features, correlations)


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

    Example:
        >>> # Create matrix where features 0,1 are correlated and 2,3 are anti-correlated
        >>> corr = create_correlation_matrix_from_correlations(
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

    Example:
        # Generate correlations where 70% are positive, 20% uncorrelated,
        # and correlation strengths range from 0.2 to 0.9
        correlations = generate_random_correlations(
            num_features=6,
            positive_ratio=0.7,
            uncorrelated_ratio=0.2,
            min_correlation_strength=0.2,
            max_correlation_strength=0.9,
            seed=42
        )
        correlation_matrix = create_correlation_matrix_from_correlations(6, correlations)
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

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
    correlated_pairs = random.sample(all_pairs, num_correlated)

    # For correlated pairs, determine positive vs negative
    num_positive = int(num_correlated * positive_ratio)
    num_negative = num_correlated - num_positive

    # Assign signs
    signs = [1] * num_positive + [-1] * num_negative
    random.shuffle(signs)

    # Generate correlation strengths
    correlations = {}
    for pair, sign in zip(correlated_pairs, signs):
        # Sample correlation strength uniformly from range
        strength = random.uniform(min_correlation_strength, max_correlation_strength)
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

    Example:
        # Generate a random 10x10 correlation matrix
        correlation_matrix = generate_random_correlation_matrix(
            num_features=10,
            positive_ratio=0.7,
            uncorrelated_ratio=0.2,
            min_correlation_strength=0.3,
            max_correlation_strength=0.8,
            seed=42
        )

        # Use directly in get_training_batch
        batch = get_training_batch(
            batch_size=1000,
            firing_probabilities=torch.rand(10) * 0.5 + 0.2,
            correlation_matrix=correlation_matrix
        )
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
