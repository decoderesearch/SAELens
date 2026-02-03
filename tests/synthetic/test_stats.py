import pytest
import torch

from sae_lens.synthetic.correlation import LowRankCorrelationMatrix
from sae_lens.synthetic.feature_dictionary import FeatureDictionary
from sae_lens.synthetic.stats import (
    compute_correlation_matrix_stats,
    compute_low_rank_correlation_matrix_stats,
    compute_superposition_stats,
)


def _make_feature_dict(vectors: torch.Tensor) -> FeatureDictionary:
    num_features, hidden_dim = vectors.shape
    fd = FeatureDictionary(num_features, hidden_dim, initializer=None)
    fd.feature_vectors.data = vectors
    return fd


def test_compute_superposition_stats_batch_size_independence():
    num_features, hidden_dim = 100, 32
    feature_dict = _make_feature_dict(torch.randn(num_features, hidden_dim))

    stats_batch_1 = compute_superposition_stats(feature_dict, batch_size=1)
    stats_batch_10 = compute_superposition_stats(feature_dict, batch_size=10)
    stats_batch_32 = compute_superposition_stats(feature_dict, batch_size=32)
    stats_batch_100 = compute_superposition_stats(feature_dict, batch_size=100)
    stats_batch_1000 = compute_superposition_stats(feature_dict, batch_size=1000)

    # All results should match
    for stats in [stats_batch_10, stats_batch_32, stats_batch_100, stats_batch_1000]:
        assert torch.allclose(
            stats.max_abs_cos_sims, stats_batch_1.max_abs_cos_sims, atol=1e-5
        )
        assert stats.mean_max_abs_cos_sim == pytest.approx(
            stats_batch_1.mean_max_abs_cos_sim, abs=1e-5
        )
        assert stats.mean_abs_cos_sim == pytest.approx(
            stats_batch_1.mean_abs_cos_sim, abs=1e-5
        )
        for p in stats.percentile_abs_cos_sims:
            assert torch.allclose(
                stats.percentile_abs_cos_sims[p],
                stats_batch_1.percentile_abs_cos_sims[p],
                atol=1e-5,
            )


def test_compute_superposition_stats_orthogonal_latents():
    hidden_dim = 8
    # Create 4 orthogonal unit vectors
    feature_dict = _make_feature_dict(torch.eye(hidden_dim)[:4])

    stats = compute_superposition_stats(feature_dict)

    assert stats.num_features == 4
    assert stats.hidden_dim == hidden_dim
    # All pairwise cosine similarities should be 0
    assert torch.allclose(stats.max_abs_cos_sims, torch.zeros(4), atol=1e-6)
    assert stats.mean_max_abs_cos_sim == pytest.approx(0, abs=1e-6)
    assert stats.mean_abs_cos_sim == pytest.approx(0, abs=1e-6)


def test_compute_superposition_stats_parallel_latents():
    hidden_dim = 8
    base_vector = torch.randn(hidden_dim)
    # Create 3 parallel vectors with different magnitudes
    vectors = torch.stack(
        [
            base_vector,
            base_vector * 2.0,
            base_vector * 0.5,
        ]
    )

    stats = compute_superposition_stats(_make_feature_dict(vectors))

    assert stats.num_features == 3
    # All pairs should have |cos_sim| = 1
    assert torch.allclose(stats.max_abs_cos_sims, torch.ones(3), atol=1e-5)
    assert stats.mean_max_abs_cos_sim == pytest.approx(1.0, abs=1e-5)
    assert stats.mean_abs_cos_sim == pytest.approx(1.0, abs=1e-5)


def test_compute_superposition_stats_antiparallel_latents():
    hidden_dim = 8
    base_vector = torch.randn(hidden_dim)
    vectors = torch.stack(
        [
            base_vector,
            -base_vector,
        ]
    )

    stats = compute_superposition_stats(_make_feature_dict(vectors))

    # |cos_sim| should be 1 for antiparallel vectors
    assert torch.allclose(stats.max_abs_cos_sims, torch.ones(2), atol=1e-5)
    assert stats.mean_abs_cos_sim == pytest.approx(1.0, abs=1e-5)


def test_compute_superposition_stats_known_case_3_latents():
    # Create 3 vectors:
    # v0 = [1, 0, 0]
    # v1 = [1, 1, 0] / sqrt(2)  -> cos(v0, v1) = 1/sqrt(2)
    # v2 = [0, 1, 0]            -> cos(v0, v2) = 0, cos(v1, v2) = 1/sqrt(2)
    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],  # Will be normalized
            [0.0, 1.0, 0.0],
        ]
    )

    stats = compute_superposition_stats(_make_feature_dict(vectors), percentiles=[50])

    # Expected pairwise |cos_sim|:
    # |cos(v0, v1)| = 1/sqrt(2) ≈ 0.7071
    # |cos(v0, v2)| = 0
    # |cos(v1, v2)| = 1/sqrt(2) ≈ 0.7071
    cos_01 = 1 / (2**0.5)
    cos_02 = 0.0
    cos_12 = 1 / (2**0.5)

    # Max |cos_sim| for each latent:
    # v0: max(|cos_01|, |cos_02|) = 1/sqrt(2)
    # v1: max(|cos_01|, |cos_12|) = 1/sqrt(2)
    # v2: max(|cos_02|, |cos_12|) = 1/sqrt(2)
    expected_max = torch.tensor([cos_01, cos_01, cos_12])
    assert torch.allclose(stats.max_abs_cos_sims, expected_max, atol=1e-5)

    # Mean of max |cos_sim| = 1/sqrt(2)
    assert stats.mean_max_abs_cos_sim == pytest.approx(cos_01, abs=1e-5)

    # Mean |cos_sim| across all pairs = (1/sqrt(2) + 0 + 1/sqrt(2)) / 3
    expected_mean = (cos_01 + cos_02 + cos_12) / 3
    assert stats.mean_abs_cos_sim == pytest.approx(expected_mean, abs=1e-5)

    # 50th percentile for each latent (median of 2 values = mean of 2 values):
    # v0: median([0, 1/sqrt(2)]) = (0 + 1/sqrt(2))/2
    # v1: median([1/sqrt(2), 1/sqrt(2)]) = 1/sqrt(2)
    # v2: median([0, 1/sqrt(2)]) = (0 + 1/sqrt(2))/2
    expected_p50_v0 = (cos_02 + cos_01) / 2
    expected_p50_v1 = cos_01  # Both values are the same
    expected_p50_v2 = (cos_02 + cos_12) / 2
    expected_p50 = torch.tensor([expected_p50_v0, expected_p50_v1, expected_p50_v2])
    assert torch.allclose(stats.percentile_abs_cos_sims[50], expected_p50, atol=1e-5)


def test_compute_superposition_stats_custom_percentiles():
    feature_dict = _make_feature_dict(torch.randn(20, 8))

    stats = compute_superposition_stats(feature_dict, percentiles=[25, 50, 75, 99])

    assert set(stats.percentile_abs_cos_sims.keys()) == {25, 50, 75, 99}
    assert set(stats.mean_percentile_abs_cos_sim.keys()) == {25, 50, 75, 99}
    for p in [25, 50, 75, 99]:
        assert stats.percentile_abs_cos_sims[p].shape == (20,)

    # Percentiles should be monotonically increasing
    for i in range(20):
        assert (
            stats.percentile_abs_cos_sims[25][i] <= stats.percentile_abs_cos_sims[50][i]
        )
        assert (
            stats.percentile_abs_cos_sims[50][i] <= stats.percentile_abs_cos_sims[75][i]
        )
        assert (
            stats.percentile_abs_cos_sims[75][i] <= stats.percentile_abs_cos_sims[99][i]
        )


def test_compute_superposition_stats_single_latent_raises():
    feature_dict = _make_feature_dict(torch.randn(1, 8))

    with pytest.raises(ValueError, match="at least 2 features"):
        compute_superposition_stats(feature_dict)


def test_compute_superposition_stats_two_latents():
    # Two vectors at 60 degrees
    vectors = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, (3**0.5) / 2],  # cos(60) = 0.5
        ]
    )

    stats = compute_superposition_stats(_make_feature_dict(vectors))

    expected_cos = 0.5
    assert torch.allclose(
        stats.max_abs_cos_sims, torch.tensor([expected_cos, expected_cos]), atol=1e-5
    )
    assert stats.mean_max_abs_cos_sim == pytest.approx(expected_cos, abs=1e-5)
    assert stats.mean_abs_cos_sim == pytest.approx(expected_cos, abs=1e-5)


def test_compute_superposition_stats_handles_zero_norm_latents():
    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # Zero vector
            [0.0, 1.0, 0.0],
        ]
    )

    stats = compute_superposition_stats(_make_feature_dict(vectors))

    # Should not contain NaN
    assert not torch.isnan(stats.max_abs_cos_sims).any()
    assert not torch.isnan(stats.percentile_abs_cos_sims[95]).any()


# Tests for compute_low_rank_correlation_matrix_stats


def test_compute_low_rank_correlation_matrix_stats_basic():
    # Create a simple low-rank correlation matrix
    # With factor [[0.5], [0.5], [0.0]], the off-diagonal of F @ F.T is:
    # (0,1): 0.25, (0,2): 0.0, (1,2): 0.0
    factor = torch.tensor([[0.5], [0.5], [0.0]])
    diag = torch.tensor([0.75, 0.75, 1.0])  # 1 - 0.25, 1 - 0.25, 1 - 0
    corr_matrix = LowRankCorrelationMatrix(factor, diag)

    stats = compute_low_rank_correlation_matrix_stats(corr_matrix)

    assert stats.num_features == 3

    # Off-diagonal values: 0.25, 0.0, 0.0, 0.25, 0.0, 0.0 (symmetric)
    # Mean (not abs) = (0.25 + 0 + 0 + 0.25 + 0 + 0) / 6 = 0.5 / 6 ≈ 0.0833
    # RMS = sqrt((0.25² * 2) / 6) = sqrt(0.125 / 6) ≈ 0.1443
    assert stats.mean_correlation == pytest.approx(0.5 / 6, abs=1e-5)
    assert stats.rms_correlation == pytest.approx((0.125 / 6) ** 0.5, abs=1e-5)


def test_compute_low_rank_correlation_matrix_stats_higher_rank():
    # 4 features, rank 2
    factor = torch.tensor(
        [
            [0.5, 0.3],
            [0.5, -0.3],
            [0.0, 0.6],
            [0.0, 0.0],
        ]
    )
    diag = 1 - (factor**2).sum(dim=1)
    corr_matrix = LowRankCorrelationMatrix(factor, diag)

    stats = compute_low_rank_correlation_matrix_stats(corr_matrix)

    assert stats.num_features == 4

    # Manually compute expected values
    off_diag_matrix = factor @ factor.T
    mask = ~torch.eye(4, dtype=torch.bool)
    off_diag_values = off_diag_matrix[mask]
    expected_mean = off_diag_values.mean().item()
    expected_rms = (off_diag_values**2).mean().sqrt().item()

    assert stats.mean_correlation == pytest.approx(expected_mean, abs=1e-5)
    assert stats.rms_correlation == pytest.approx(expected_rms, abs=1e-5)


def test_compute_low_rank_correlation_matrix_stats_no_correlation():
    # All zeros in factor means no off-diagonal correlations
    factor = torch.zeros(5, 2)
    diag = torch.ones(5)
    corr_matrix = LowRankCorrelationMatrix(factor, diag)

    stats = compute_low_rank_correlation_matrix_stats(corr_matrix)

    assert stats.mean_correlation == 0.0
    assert stats.rms_correlation == 0.0


# Tests for compute_correlation_matrix_stats


def test_compute_correlation_matrix_stats_identity():
    corr_matrix = torch.eye(5)

    stats = compute_correlation_matrix_stats(corr_matrix)

    assert stats.num_features == 5
    assert stats.mean_correlation == 0.0
    assert stats.rms_correlation == 0.0


def test_compute_correlation_matrix_stats_basic():
    # 3x3 matrix with known off-diagonal values
    corr_matrix = torch.tensor(
        [
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    stats = compute_correlation_matrix_stats(corr_matrix)

    assert stats.num_features == 3
    # Off-diagonal: 0.5, 0.0, 0.5, 0.0, 0.0, 0.0
    # Mean = (0.5 + 0 + 0.5 + 0 + 0 + 0) / 6 = 1/6
    # RMS = sqrt((0.25 + 0 + 0.25 + 0 + 0 + 0) / 6) = sqrt(0.5/6)
    assert stats.mean_correlation == pytest.approx(1 / 6, abs=1e-5)
    assert stats.rms_correlation == pytest.approx((0.5 / 6) ** 0.5, abs=1e-5)


def test_compute_correlation_matrix_stats_matches_low_rank():
    factor = torch.tensor(
        [
            [0.5, 0.3],
            [0.5, -0.3],
            [0.0, 0.6],
            [0.0, 0.0],
        ]
    )
    diag = 1 - (factor**2).sum(dim=1)

    # Compute dense correlation matrix
    dense_corr = factor @ factor.T + torch.diag(diag)

    # Compute stats both ways
    dense_stats = compute_correlation_matrix_stats(dense_corr)
    low_rank_stats = compute_low_rank_correlation_matrix_stats(
        LowRankCorrelationMatrix(factor, diag)
    )

    assert dense_stats.rms_correlation == pytest.approx(
        low_rank_stats.rms_correlation, abs=1e-5
    )
    assert dense_stats.mean_correlation == pytest.approx(
        low_rank_stats.mean_correlation, abs=1e-5
    )
