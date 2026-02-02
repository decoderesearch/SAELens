from dataclasses import dataclass

import torch

from sae_lens.synthetic.correlation import LowRankCorrelationMatrix
from sae_lens.synthetic.feature_dictionary import FeatureDictionary


@dataclass
class CorrelationMatrixStats:
    """Statistics computed from a correlation matrix."""

    rms_correlation: float  # Root mean square of off-diagonal correlations
    mean_correlation: float  # Mean of off-diagonal correlations (not absolute)
    num_features: int


@torch.no_grad()
def compute_correlation_matrix_stats(
    correlation_matrix: torch.Tensor,
) -> CorrelationMatrixStats:
    """Compute correlation statistics from a dense correlation matrix.

    Args:
        correlation_matrix: Dense correlation matrix of shape (n, n)

    Returns:
        CorrelationMatrixStats with correlation statistics
    """
    num_features = correlation_matrix.shape[0]

    # Extract off-diagonal elements
    mask = ~torch.eye(num_features, dtype=torch.bool, device=correlation_matrix.device)
    off_diag = correlation_matrix[mask]

    rms_correlation = (off_diag**2).mean().sqrt().item()
    mean_correlation = off_diag.mean().item()

    return CorrelationMatrixStats(
        rms_correlation=rms_correlation,
        mean_correlation=mean_correlation,
        num_features=num_features,
    )


@torch.no_grad()
def compute_low_rank_correlation_matrix_stats(
    correlation_matrix: LowRankCorrelationMatrix,
) -> CorrelationMatrixStats:
    """Compute correlation statistics from a LowRankCorrelationMatrix.

    The correlation matrix is represented as:
        correlation = factor @ factor.T + diag(diag_term)

    The off-diagonal elements are simply factor @ factor.T (the diagonal term
    only affects the diagonal).

    All statistics are computed efficiently in O(n*r²) time and O(r²) memory
    without materializing the full n×n correlation matrix.

    Args:
        correlation_matrix: Low-rank correlation matrix

    Returns:
        CorrelationMatrixStats with correlation statistics
    """

    factor = correlation_matrix.correlation_factor
    num_features = factor.shape[0]
    num_off_diag = num_features * (num_features - 1)

    # RMS correlation: uses ||F @ F.T||_F² = ||F.T @ F||_F²
    # This avoids computing the (num_features, num_features) matrix
    G = factor.T @ factor  # (rank, rank) - small!
    frobenius_sq = (G**2).sum()
    row_norms_sq = (factor**2).sum(dim=1)  # ||F[i]||² for each row
    diag_sq_sum = (row_norms_sq**2).sum()  # Σᵢ ||F[i]||⁴
    off_diag_sq_sum = frobenius_sq - diag_sq_sum
    rms_correlation = (off_diag_sq_sum / num_off_diag).sqrt().item()

    # Mean correlation (not absolute): sum(C) = ||col_sums(F)||², trace(C) = Σ||F[i]||²
    col_sums = factor.sum(dim=0)  # (rank,)
    sum_all = (col_sums**2).sum()  # 1ᵀ C 1
    trace_C = row_norms_sq.sum()
    mean_correlation = ((sum_all - trace_C) / num_off_diag).item()

    return CorrelationMatrixStats(
        rms_correlation=rms_correlation,
        mean_correlation=mean_correlation,
        num_features=num_features,
    )


@dataclass
class SuperpositionStats:
    """Statistics measuring superposition in a feature dictionary."""

    # Per-latent statistics: for each latent, max and percentile of |cos_sim| with others
    max_abs_cos_sims: torch.Tensor  # Shape: (num_features,)
    percentile_abs_cos_sims: dict[int, torch.Tensor]  # {percentile: (num_features,)}

    # Summary statistics (means of the per-latent values)
    mean_max_abs_cos_sim: float
    mean_percentile_abs_cos_sim: dict[int, float]
    mean_abs_cos_sim: float  # Mean |cos_sim| across all pairs

    # Metadata
    num_features: int
    hidden_dim: int


@torch.no_grad()
def compute_superposition_stats(
    feature_dictionary: FeatureDictionary,
    batch_size: int = 1024,
    device: str | torch.device | None = None,
    percentiles: list[int] | None = None,
) -> SuperpositionStats:
    """Compute superposition statistics for a feature dictionary.

    Computes pairwise cosine similarities in batches to handle large dictionaries.

    For each latent i, computes:

    - max |cos_sim(i, j)| over all j != i
    - kth percentile of |cos_sim(i, j)| over all j != i (for each k in percentiles)

    Args:
        feature_dictionary: FeatureDictionary containing the feature vectors
        batch_size: Number of features to process per batch
        device: Device for computation (defaults to feature dictionary's device)
        percentiles: List of percentiles to compute per latent (default: [95, 99])

    Returns:
        SuperpositionStats with superposition metrics
    """
    if percentiles is None:
        percentiles = [95, 99]

    feature_vectors = feature_dictionary.feature_vectors
    num_features, hidden_dim = feature_vectors.shape

    if num_features < 2:
        raise ValueError("Need at least 2 features to compute superposition stats")
    if device is None:
        device = feature_vectors.device

    # Normalize features to unit norm for cosine similarity
    features_normalized = feature_vectors.to(device).float()
    norms = torch.linalg.norm(features_normalized, dim=1, keepdim=True)
    features_normalized = features_normalized / norms.clamp(min=1e-8)

    # Track per-latent statistics
    max_abs_cos_sims = torch.zeros(num_features, device=device)
    percentile_abs_cos_sims = {
        p: torch.zeros(num_features, device=device) for p in percentiles
    }
    sum_abs_cos_sim = 0.0
    n_pairs = 0

    # Process in batches: for each batch of features, compute similarities with all others
    for i in range(0, num_features, batch_size):
        batch_end = min(i + batch_size, num_features)
        batch = features_normalized[i:batch_end]  # (batch_size, hidden_dim)

        # Compute cosine similarities with all features: (batch_size, num_features)
        cos_sims = batch @ features_normalized.T

        # Absolute cosine similarities
        abs_cos_sims = cos_sims.abs()

        # Process each latent in the batch
        for j, idx in enumerate(range(i, batch_end)):
            # Get similarities with all other features (exclude self)
            row = abs_cos_sims[j].clone()
            row[idx] = 0.0  # Exclude self for max
            max_abs_cos_sims[idx] = row.max()

            # For percentiles, exclude self and compute
            other_sims = torch.cat([abs_cos_sims[j, :idx], abs_cos_sims[j, idx + 1 :]])
            for p in percentiles:
                percentile_abs_cos_sims[p][idx] = torch.quantile(other_sims, p / 100.0)

            # Sum for mean computation (only count pairs once - with features after this one)
            sum_abs_cos_sim += abs_cos_sims[j, idx + 1 :].sum().item()
            n_pairs += num_features - idx - 1

    # Compute summary statistics
    mean_max_abs_cos_sim = max_abs_cos_sims.mean().item()
    mean_percentile_abs_cos_sim = {
        p: percentile_abs_cos_sims[p].mean().item() for p in percentiles
    }
    mean_abs_cos_sim = sum_abs_cos_sim / n_pairs if n_pairs > 0 else 0.0

    return SuperpositionStats(
        max_abs_cos_sims=max_abs_cos_sims.cpu(),
        percentile_abs_cos_sims={
            p: v.cpu() for p, v in percentile_abs_cos_sims.items()
        },
        mean_max_abs_cos_sim=mean_max_abs_cos_sim,
        mean_percentile_abs_cos_sim=mean_percentile_abs_cos_sim,
        mean_abs_cos_sim=mean_abs_cos_sim,
        num_features=num_features,
        hidden_dim=hidden_dim,
    )
