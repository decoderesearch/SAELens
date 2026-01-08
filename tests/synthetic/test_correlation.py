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
