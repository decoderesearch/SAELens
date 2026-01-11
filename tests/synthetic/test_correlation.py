import pytest
import torch

from sae_lens.synthetic.correlation import (
    _fix_correlation_matrix,
    create_correlation_matrix_from_correlations,
    generate_random_correlation_matrix,
    generate_random_correlations,
)


class TestCreateCorrelationMatrixFromCorrelations:
    def test_identity_matrix_with_no_correlations(self):
        matrix = create_correlation_matrix_from_correlations(num_features=3)
        expected = torch.eye(3)
        torch.testing.assert_close(matrix, expected)

    def test_default_correlation_fills_off_diagonal(self):
        matrix = create_correlation_matrix_from_correlations(
            num_features=3, default_correlation=0.5
        )
        assert matrix[0, 0] == 1.0
        assert matrix[0, 1] == pytest.approx(0.5, abs=0.01)
        assert matrix[1, 2] == pytest.approx(0.5, abs=0.01)

    def test_custom_correlations_override_default(self):
        correlations = {(0, 1): 0.8, (1, 2): -0.3}
        matrix = create_correlation_matrix_from_correlations(
            num_features=3, correlations=correlations
        )
        assert matrix[0, 1] == pytest.approx(0.8, abs=0.01)
        assert matrix[1, 0] == pytest.approx(0.8, abs=0.01)
        assert matrix[1, 2] == pytest.approx(-0.3, abs=0.01)
        assert matrix[2, 1] == pytest.approx(-0.3, abs=0.01)

    def test_matrix_is_symmetric(self):
        correlations = {(0, 1): 0.5, (0, 2): 0.3, (1, 2): -0.2}
        matrix = create_correlation_matrix_from_correlations(
            num_features=3, correlations=correlations
        )
        torch.testing.assert_close(matrix, matrix.T)

    def test_diagonal_is_ones(self):
        matrix = create_correlation_matrix_from_correlations(
            num_features=5, default_correlation=0.3
        )
        diagonal = torch.diag(matrix)
        torch.testing.assert_close(diagonal, torch.ones(5))

    def test_fixes_non_positive_definite_matrix(self):
        # Create correlations that would result in non-positive definite matrix
        # A matrix with all off-diagonal = 0.99 is not positive definite for n > 2
        matrix = create_correlation_matrix_from_correlations(
            num_features=5, default_correlation=0.99
        )
        # Should be fixed to be positive semi-definite
        eigenvals = torch.linalg.eigvalsh(matrix)
        assert torch.all(eigenvals >= -1e-6)


class TestFixCorrelationMatrix:
    def test_fixes_negative_eigenvalues(self):
        # Create a matrix with negative eigenvalues
        # A correlation matrix with conflicting high correlations can be non-PSD
        bad_matrix = torch.tensor([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
        eigenvals_before = torch.linalg.eigvalsh(bad_matrix)
        assert torch.any(
            eigenvals_before < 0
        ), f"Expected negative eigenvalues, got {eigenvals_before}"

        fixed = _fix_correlation_matrix(bad_matrix)
        eigenvals_after = torch.linalg.eigvalsh(fixed)
        assert torch.all(eigenvals_after >= 0)

    def test_preserves_diagonal_ones(self):
        bad_matrix = torch.tensor([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
        fixed = _fix_correlation_matrix(bad_matrix)
        torch.testing.assert_close(torch.diag(fixed), torch.ones(3))

    def test_result_is_symmetric(self):
        bad_matrix = torch.tensor([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
        fixed = _fix_correlation_matrix(bad_matrix)
        torch.testing.assert_close(fixed, fixed.T)


class TestGenerateRandomCorrelations:
    def test_returns_dict(self):
        result = generate_random_correlations(num_features=5, seed=42)
        assert isinstance(result, dict)

    def test_seed_reproducibility(self):
        result1 = generate_random_correlations(num_features=5, seed=42)
        result2 = generate_random_correlations(num_features=5, seed=42)
        assert result1 == result2

    def test_different_seeds_different_results(self):
        result1 = generate_random_correlations(num_features=5, seed=42)
        result2 = generate_random_correlations(num_features=5, seed=123)
        assert result1 != result2

    def test_uncorrelated_ratio_reduces_pairs(self):
        # With uncorrelated_ratio=0.5, roughly half the pairs should be uncorrelated
        result = generate_random_correlations(
            num_features=5, uncorrelated_ratio=0.5, seed=42
        )
        total_possible_pairs = 5 * 4 // 2  # 10 pairs
        assert len(result) <= total_possible_pairs * 0.6  # Allow some margin

    def test_positive_ratio_affects_signs(self):
        # With positive_ratio=1.0, all correlations should be positive
        result = generate_random_correlations(
            num_features=10, positive_ratio=1.0, uncorrelated_ratio=0.0, seed=42
        )
        for corr in result.values():
            assert corr > 0

    def test_negative_correlations_with_low_positive_ratio(self):
        result = generate_random_correlations(
            num_features=10, positive_ratio=0.0, uncorrelated_ratio=0.0, seed=42
        )
        for corr in result.values():
            assert corr < 0

    def test_correlation_strength_in_range(self):
        result = generate_random_correlations(
            num_features=10,
            min_correlation_strength=0.2,
            max_correlation_strength=0.5,
            uncorrelated_ratio=0.0,
            seed=42,
        )
        for corr in result.values():
            assert 0.2 <= abs(corr) <= 0.5

    def test_single_feature_returns_empty(self):
        result = generate_random_correlations(num_features=1)
        assert result == {}

    def test_raises_on_invalid_positive_ratio(self):
        with pytest.raises(ValueError, match="positive_ratio"):
            generate_random_correlations(num_features=5, positive_ratio=1.5)

    def test_raises_on_invalid_uncorrelated_ratio(self):
        with pytest.raises(ValueError, match="uncorrelated_ratio"):
            generate_random_correlations(num_features=5, uncorrelated_ratio=-0.1)

    def test_raises_on_negative_min_strength(self):
        with pytest.raises(ValueError, match="min_correlation_strength"):
            generate_random_correlations(num_features=5, min_correlation_strength=-0.1)

    def test_raises_on_max_strength_above_one(self):
        with pytest.raises(ValueError, match="max_correlation_strength"):
            generate_random_correlations(num_features=5, max_correlation_strength=1.5)

    def test_raises_on_min_greater_than_max(self):
        with pytest.raises(ValueError, match="min_correlation_strength must be <="):
            generate_random_correlations(
                num_features=5,
                min_correlation_strength=0.8,
                max_correlation_strength=0.2,
            )


class TestGenerateRandomCorrelationMatrix:
    def test_returns_tensor(self):
        result = generate_random_correlation_matrix(num_features=5, seed=42)
        assert isinstance(result, torch.Tensor)

    def test_correct_shape(self):
        result = generate_random_correlation_matrix(num_features=5, seed=42)
        assert result.shape == (5, 5)

    def test_seed_reproducibility(self):
        result1 = generate_random_correlation_matrix(num_features=5, seed=42)
        result2 = generate_random_correlation_matrix(num_features=5, seed=42)
        torch.testing.assert_close(result1, result2)

    def test_is_symmetric(self):
        result = generate_random_correlation_matrix(num_features=5, seed=42)
        torch.testing.assert_close(result, result.T)

    def test_diagonal_is_ones(self):
        result = generate_random_correlation_matrix(num_features=5, seed=42)
        torch.testing.assert_close(torch.diag(result), torch.ones(5))

    def test_is_positive_semi_definite(self):
        result = generate_random_correlation_matrix(num_features=5, seed=42)
        eigenvals = torch.linalg.eigvalsh(result)
        assert torch.all(eigenvals >= -1e-6)

    def test_respects_device_parameter(self):
        result = generate_random_correlation_matrix(num_features=5, device="cpu")
        assert result.device == torch.device("cpu")

    def test_respects_dtype_parameter(self):
        result = generate_random_correlation_matrix(num_features=5, dtype=torch.float64)
        assert result.dtype == torch.float64

    def test_single_feature_returns_identity(self):
        result = generate_random_correlation_matrix(num_features=1)
        torch.testing.assert_close(result, torch.eye(1))

    def test_zero_features_returns_empty(self):
        result = generate_random_correlation_matrix(num_features=0)
        assert result.shape == (0, 0)


class TestGenerateRandomCorrelationMatrixStatistics:
    def test_higher_positive_ratio_produces_more_positive_correlations(self):
        num_features = 30
        num_samples = 20

        def avg_positive_fraction(positive_ratio: float) -> float:
            total_positive = 0
            total_pairs = num_features * (num_features - 1) // 2
            for i in range(num_samples):
                matrix = generate_random_correlation_matrix(
                    num_features=num_features,
                    positive_ratio=positive_ratio,
                    uncorrelated_ratio=0.0,
                    min_correlation_strength=0.05,
                    max_correlation_strength=0.15,
                    seed=i * 54321,
                )
                row_idx, col_idx = torch.triu_indices(
                    num_features, num_features, offset=1
                )
                off_diag = matrix[row_idx, col_idx]
                total_positive += (off_diag > 0).sum().item()
            return total_positive / (num_samples * total_pairs)

        low_positive = avg_positive_fraction(0.2)
        high_positive = avg_positive_fraction(0.8)
        assert high_positive > low_positive + 0.3

    def test_higher_uncorrelated_ratio_produces_smaller_correlations(self):
        num_features = 30
        num_samples = 20

        def avg_abs_correlation(uncorrelated_ratio: float) -> float:
            total = 0.0
            count = 0
            for i in range(num_samples):
                matrix = generate_random_correlation_matrix(
                    num_features=num_features,
                    uncorrelated_ratio=uncorrelated_ratio,
                    min_correlation_strength=0.1,
                    max_correlation_strength=0.2,
                    seed=i * 12345,
                )
                row_idx, col_idx = torch.triu_indices(
                    num_features, num_features, offset=1
                )
                off_diag = matrix[row_idx, col_idx]
                total += off_diag.abs().sum().item()
                count += off_diag.numel()
            return total / count

        low_uncorrelated = avg_abs_correlation(0.2)
        high_uncorrelated = avg_abs_correlation(0.8)
        assert low_uncorrelated > high_uncorrelated

    def test_higher_strength_range_produces_larger_correlations(self):
        num_features = 30
        num_samples = 20

        def avg_abs_correlation(min_s: float, max_s: float) -> float:
            total = 0.0
            count = 0
            for i in range(num_samples):
                matrix = generate_random_correlation_matrix(
                    num_features=num_features,
                    uncorrelated_ratio=0.0,
                    min_correlation_strength=min_s,
                    max_correlation_strength=max_s,
                    seed=i * 11111,
                )
                row_idx, col_idx = torch.triu_indices(
                    num_features, num_features, offset=1
                )
                off_diag = matrix[row_idx, col_idx]
                total += off_diag.abs().sum().item()
                count += off_diag.numel()
            return total / count

        low_strength = avg_abs_correlation(0.05, 0.1)
        high_strength = avg_abs_correlation(0.2, 0.3)
        assert high_strength > low_strength

    def test_all_zero_correlations_when_uncorrelated_ratio_is_one(self):
        matrix = generate_random_correlation_matrix(
            num_features=20,
            uncorrelated_ratio=1.0,
        )
        expected = torch.eye(20)
        torch.testing.assert_close(matrix, expected)

    def test_small_correlations_preserve_positive_ratio_exactly(self):
        num_features = 20
        matrix = generate_random_correlation_matrix(
            num_features=num_features,
            positive_ratio=1.0,
            uncorrelated_ratio=0.0,
            min_correlation_strength=0.01,
            max_correlation_strength=0.02,
        )
        row_idx, col_idx = torch.triu_indices(num_features, num_features, offset=1)
        off_diag = matrix[row_idx, col_idx]
        assert torch.all(off_diag > 0)

    def test_small_negative_correlations_preserve_sign(self):
        num_features = 20
        matrix = generate_random_correlation_matrix(
            num_features=num_features,
            positive_ratio=0.0,
            uncorrelated_ratio=0.0,
            min_correlation_strength=0.01,
            max_correlation_strength=0.02,
        )
        row_idx, col_idx = torch.triu_indices(num_features, num_features, offset=1)
        off_diag = matrix[row_idx, col_idx]
        assert torch.all(off_diag < 0)
