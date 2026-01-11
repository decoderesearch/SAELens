import pytest
import torch

from sae_lens.synthetic import ActivationGenerator, LowRankCorrelationMatrix
from sae_lens.synthetic.activation_generator import (
    _normalize_modifiers,
    _to_tensor,
    _validate_correlation_matrix,
    _validate_low_rank_correlation,
)
from sae_lens.synthetic.correlation import generate_random_correlation_matrix


class TestActivationGeneratorBasic:
    def test_respects_firing_probabilities(self):
        firing_probs = torch.tensor([0.3, 0.2, 0.1])
        batch_size = 2000
        generator = ActivationGenerator(
            num_features=3,
            firing_probabilities=firing_probs,
        )
        activations = generator.sample(batch_size)

        actual_probs = (activations > 0).float().mean(dim=0)
        torch.testing.assert_close(actual_probs, firing_probs, atol=0.05, rtol=0)

    def test_respects_std_magnitudes(self):
        firing_probs = torch.tensor([1.0, 1.0, 1.0])
        std_magnitudes = torch.tensor([0.1, 0.2, 0.3])
        batch_size = 2000
        generator = ActivationGenerator(
            num_features=3,
            firing_probabilities=firing_probs,
            std_firing_magnitudes=std_magnitudes,
        )
        activations = generator.sample(batch_size)

        actual_stds = activations.std(dim=0)
        torch.testing.assert_close(actual_stds, std_magnitudes, atol=0.05, rtol=0)

    def test_respects_mean_magnitudes(self):
        firing_probs = torch.tensor([0.5, 0.5, 1.0])
        mean_magnitudes = torch.tensor([1.5, 2.5, 3.5])
        batch_size = 2000
        generator = ActivationGenerator(
            num_features=3,
            firing_probabilities=firing_probs,
            mean_firing_magnitudes=mean_magnitudes,
        )
        activations = generator.sample(batch_size)

        assert set(activations[:, 0].tolist()) == {0, 1.5}
        assert set(activations[:, 1].tolist()) == {0, 2.5}
        assert set(activations[:, 2].tolist()) == {3.5}

    def test_never_returns_negative(self):
        firing_probs = torch.tensor([1.0, 1.0, 1.0])
        std_magnitudes = torch.tensor([0.5, 1.0, 2.0])
        batch_size = 2000
        generator = ActivationGenerator(
            num_features=3,
            firing_probabilities=firing_probs,
            std_firing_magnitudes=std_magnitudes,
        )
        activations = generator.sample(batch_size)

        assert torch.all(activations >= 0)


class TestActivationGeneratorModifiers:
    def test_with_empty_list_of_modifiers(self):
        generator = ActivationGenerator(
            num_features=3,
            firing_probabilities=0.5,
            modify_activations=[],
        )

        assert generator.modify_activations is None
        samples = generator.sample(batch_size=10)
        assert samples.shape == (10, 3)

    def test_with_single_modifier(self):
        def double_activations(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        generator = ActivationGenerator(
            num_features=3,
            firing_probabilities=1.0,  # Always fire
            mean_firing_magnitudes=1.0,
            std_firing_magnitudes=0.0,
            modify_activations=double_activations,
        )

        samples = generator.sample(batch_size=10)
        # After doubling, values should be 2.0 (and then relu keeps them positive)
        assert torch.allclose(samples, torch.ones_like(samples) * 2.0)

    def test_with_multiple_modifiers(self):
        def add_one(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        def multiply_two(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        generator = ActivationGenerator(
            num_features=3,
            firing_probabilities=1.0,
            mean_firing_magnitudes=1.0,
            std_firing_magnitudes=0.0,
            modify_activations=[add_one, multiply_two],
        )

        samples = generator.sample(batch_size=10)
        # (1 + 1) * 2 = 4
        assert torch.allclose(samples, torch.ones_like(samples) * 4.0)

    def test_modifier_result_is_relu_applied(self):
        def make_negative(x: torch.Tensor) -> torch.Tensor:
            return -x

        generator = ActivationGenerator(
            num_features=3,
            firing_probabilities=1.0,
            mean_firing_magnitudes=1.0,
            std_firing_magnitudes=0.0,
            modify_activations=make_negative,
        )

        samples = generator.sample(batch_size=10)
        # After negation and relu, should be 0
        assert torch.all(samples == 0)


class TestActivationGeneratorCorrelatedFeatures:
    def test_with_correlation_matrix(self):
        correlation_matrix = generate_random_correlation_matrix(num_features=5, seed=42)

        generator = ActivationGenerator(
            num_features=5,
            firing_probabilities=0.5,
            correlation_matrix=correlation_matrix,
        )

        samples = generator.sample(batch_size=100)
        assert samples.shape == (100, 5)
        assert torch.all(samples >= 0)

    def test_correlated_features_preserve_marginal_probabilities(self):
        correlation_matrix = generate_random_correlation_matrix(
            num_features=5,
            min_correlation_strength=0.3,
            max_correlation_strength=0.5,
            seed=42,
        )
        firing_probs = torch.tensor([0.4, 0.3, 0.2, 0.3, 0.4])

        generator = ActivationGenerator(
            num_features=5,
            firing_probabilities=firing_probs,
            correlation_matrix=correlation_matrix,
        )

        samples = generator.sample(batch_size=5000)
        actual_probs = (samples > 0).float().mean(dim=0)

        # Marginal probabilities should be approximately preserved
        torch.testing.assert_close(actual_probs, firing_probs, atol=0.1, rtol=0)


class TestActivationGeneratorForward:
    def test_forward_equals_sample(self):
        generator = ActivationGenerator(
            num_features=5,
            firing_probabilities=0.5,
        )

        torch.manual_seed(123)
        sample1 = generator.sample(batch_size=10)

        torch.manual_seed(123)
        sample2 = generator(batch_size=10)

        torch.testing.assert_close(sample1, sample2)


class TestActivationGeneratorDeviceAndDtype:
    def test_accepts_float_firing_probability(self):
        generator = ActivationGenerator(
            num_features=5,
            firing_probabilities=0.3,  # float, not tensor
        )

        samples = generator.sample(batch_size=100)
        assert samples.shape == (100, 5)

    def test_accepts_string_dtype(self):
        generator = ActivationGenerator(
            num_features=5,
            firing_probabilities=0.5,
            dtype="float32",
        )

        samples = generator.sample(batch_size=10)
        assert samples.dtype == torch.float32


class TestToTensor:
    def test_converts_float_to_tensor(self):
        result = _to_tensor(0.5, num_features=3, device="cpu", dtype="float32")
        expected = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_raises_on_wrong_shape(self):
        wrong_shape = torch.tensor([0.5, 0.5])  # Shape (2,) but num_features=3
        with pytest.raises(ValueError, match="must be a tensor of shape"):
            _to_tensor(wrong_shape, num_features=3, device="cpu", dtype="float32")

    def test_preserves_correct_tensor(self):
        tensor = torch.tensor([0.1, 0.2, 0.3])
        result = _to_tensor(tensor, num_features=3, device="cpu", dtype="float32")
        torch.testing.assert_close(result, tensor.float())


class TestNormalizeModifiers:
    def test_none_returns_none(self):
        assert _normalize_modifiers(None) is None

    def test_callable_returns_callable(self):
        def my_modifier(x: torch.Tensor) -> torch.Tensor:
            return x

        result = _normalize_modifiers(my_modifier)
        assert result is my_modifier

    def test_empty_list_returns_none(self):
        assert _normalize_modifiers([]) is None

    def test_single_item_list_returns_item(self):
        def my_modifier(x: torch.Tensor) -> torch.Tensor:
            return x

        result = _normalize_modifiers([my_modifier])
        assert result is my_modifier

    def test_multiple_items_returns_chained(self):
        def add_one(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        def multiply_two(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        result = _normalize_modifiers([add_one, multiply_two])
        assert result is not None

        # Test the chained function
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output = result(input_tensor)
        expected = (input_tensor + 1) * 2
        torch.testing.assert_close(output, expected)


class TestValidateCorrelationMatrix:
    def test_valid_correlation_matrix(self):
        correlation_matrix = generate_random_correlation_matrix(num_features=3, seed=42)
        _validate_correlation_matrix(correlation_matrix, num_features=3)

    def test_identity_matrix_is_valid(self):
        identity = torch.eye(5)
        _validate_correlation_matrix(identity, num_features=5)

    def test_raises_on_wrong_shape(self):
        wrong_shape = torch.eye(3)
        with pytest.raises(ValueError, match="must have shape"):
            _validate_correlation_matrix(wrong_shape, num_features=5)

    def test_raises_on_non_unit_diagonal(self):
        matrix = torch.eye(3)
        matrix[0, 0] = 2.0
        with pytest.raises(ValueError, match="diagonal must be all 1s"):
            _validate_correlation_matrix(matrix, num_features=3)

    def test_raises_on_not_positive_definite(self):
        # Matrix with off-diagonal values > 1 is not positive definite
        matrix = torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        with pytest.raises(ValueError, match="must be positive definite"):
            _validate_correlation_matrix(matrix, num_features=3)


class TestActivationGeneratorCorrelationMatrixValidation:
    def test_raises_on_invalid_shape(self):
        wrong_shape = torch.eye(3)
        with pytest.raises(ValueError, match="must have shape"):
            ActivationGenerator(
                num_features=5,
                firing_probabilities=0.5,
                correlation_matrix=wrong_shape,
            )

    def test_raises_on_non_unit_diagonal(self):
        matrix = torch.eye(3)
        matrix[1, 1] = 0.5
        with pytest.raises(ValueError, match="diagonal must be all 1s"):
            ActivationGenerator(
                num_features=3,
                firing_probabilities=0.5,
                correlation_matrix=matrix,
            )

    def test_raises_on_not_positive_definite(self):
        matrix = torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        with pytest.raises(ValueError, match="must be positive definite"):
            ActivationGenerator(
                num_features=3,
                firing_probabilities=0.5,
                correlation_matrix=matrix,
            )


class TestValidateLowRankCorrelationMatrix:
    def test_valid_low_rank_correlation(self):
        num_features = 5
        rank = 2
        factor = torch.randn(num_features, rank) * 0.1
        diag = 1 - (factor**2).sum(dim=1)
        diag = diag.clamp(min=0.1)
        _validate_low_rank_correlation(factor, diag, num_features)

    def test_raises_on_wrong_factor_dims(self):
        factor = torch.randn(5)
        diag = torch.ones(5)
        with pytest.raises(ValueError, match="must be 2D"):
            _validate_low_rank_correlation(factor, diag, num_features=5)

    def test_raises_on_wrong_factor_num_features(self):
        factor = torch.randn(3, 2)
        diag = torch.ones(5)
        with pytest.raises(ValueError, match=r"must have shape \(5, rank\)"):
            _validate_low_rank_correlation(factor, diag, num_features=5)

    def test_raises_on_wrong_diag_shape(self):
        factor = torch.randn(5, 2)
        diag = torch.ones(3)
        with pytest.raises(ValueError, match=r"must have shape \(5,\)"):
            _validate_low_rank_correlation(factor, diag, num_features=5)

    def test_raises_on_non_positive_diag(self):
        factor = torch.randn(5, 2)
        diag = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="must have all positive values"):
            _validate_low_rank_correlation(factor, diag, num_features=5)


class TestActivationGeneratorLowRankCorrelationMatrix:
    def test_with_low_rank_correlation_tuple(self):
        """Test that low-rank correlation as tuple works."""
        num_features = 5
        rank = 2
        factor = torch.randn(num_features, rank) * 0.1
        diag = 1 - (factor**2).sum(dim=1)
        diag = diag.clamp(min=0.1)

        generator = ActivationGenerator(
            num_features=num_features,
            firing_probabilities=0.5,
            correlation_matrix=(factor, diag),
        )

        samples = generator.sample(batch_size=100)
        assert samples.shape == (100, num_features)
        assert torch.all(samples >= 0)

    def test_with_low_rank_correlation_named_tuple(self):
        """Test that LowRankCorrelationMatrix NamedTuple works."""
        num_features = 5
        rank = 2
        factor = torch.randn(num_features, rank) * 0.1
        diag = 1 - (factor**2).sum(dim=1)
        diag = diag.clamp(min=0.1)

        generator = ActivationGenerator(
            num_features=num_features,
            firing_probabilities=0.5,
            correlation_matrix=LowRankCorrelationMatrix(factor, diag),
        )

        samples = generator.sample(batch_size=100)
        assert samples.shape == (100, num_features)
        assert torch.all(samples >= 0)

    def test_low_rank_preserves_marginal_probabilities(self):
        """Test that low-rank correlation preserves marginal firing probabilities."""
        num_features = 5
        rank = 2
        factor = torch.randn(num_features, rank) * 0.3
        diag = 1 - (factor**2).sum(dim=1)
        diag = diag.clamp(min=0.1)

        firing_probs = torch.tensor([0.4, 0.3, 0.2, 0.3, 0.4])

        generator = ActivationGenerator(
            num_features=num_features,
            firing_probabilities=firing_probs,
            correlation_matrix=(factor, diag),
        )

        samples = generator.sample(batch_size=5000)
        actual_probs = (samples > 0).float().mean(dim=0)

        torch.testing.assert_close(actual_probs, firing_probs, atol=0.02, rtol=0)

    def test_full_matrix_and_low_rank_equivalent_produce_same_statistics(self):
        """Test that a full correlation matrix and its low-rank representation produce
        statistically equivalent results.

        This is a key test: if we construct a correlation matrix from low-rank factors
        (C = F @ F.T + diag(D)), then using either the full matrix or the low-rank
        representation should yield the same firing statistics.
        """
        num_features = 5
        rank = 3
        batch_size = 50000

        # Create low-rank factors small enough that diag stays positive without clamping
        # This ensures the full matrix and low-rank representation are mathematically equivalent
        factor = torch.randn(num_features, rank) * 0.1
        factor_sq_sum = (factor**2).sum(dim=1)
        assert torch.all(
            factor_sq_sum < 1
        ), "Factor norms too large for valid correlation"
        diag = 1 - factor_sq_sum

        # Construct the full correlation matrix: C = F @ F.T + diag(D)
        # With our construction, this already has unit diagonal (no normalization needed)
        full_matrix = factor @ factor.T + torch.diag(diag)
        # Verify unit diagonal
        torch.testing.assert_close(
            torch.diag(full_matrix), torch.ones(num_features), atol=1e-6, rtol=0
        )

        firing_probs = torch.tensor([0.3, 0.4, 0.2, 0.5, 0.35])

        generator_full = ActivationGenerator(
            num_features=num_features,
            firing_probabilities=firing_probs,
            correlation_matrix=full_matrix,
        )
        generator_low_rank = ActivationGenerator(
            num_features=num_features,
            firing_probabilities=firing_probs,
            correlation_matrix=(factor, diag),
        )

        # Sample from both generators
        samples_full = generator_full.sample(batch_size)
        samples_low_rank = generator_low_rank.sample(batch_size)

        # Compare marginal firing probabilities
        probs_full = (samples_full > 0).float().mean(dim=0)
        probs_low_rank = (samples_low_rank > 0).float().mean(dim=0)

        torch.testing.assert_close(probs_full, firing_probs, atol=0.05, rtol=0)
        torch.testing.assert_close(probs_low_rank, firing_probs, atol=0.05, rtol=0)

        # Compare pairwise correlations of firing patterns
        def compute_firing_correlations(samples: torch.Tensor) -> torch.Tensor:
            firing = (samples > 0).float()
            firing_centered = firing - firing.mean(dim=0, keepdim=True)
            cov = (firing_centered.T @ firing_centered) / (samples.shape[0] - 1)
            std = firing.std(dim=0)
            std = std.clamp(min=1e-6)
            return cov / (std.unsqueeze(0) * std.unsqueeze(1))

        corr_full = compute_firing_correlations(samples_full)
        corr_low_rank = compute_firing_correlations(samples_low_rank)

        # The correlations should be similar (not identical due to sampling variance)
        torch.testing.assert_close(corr_full, corr_low_rank, atol=0.02, rtol=0)

    def test_exact_low_rank_matches_full_matrix_statistics(self):
        """Test that when we use the exact same covariance in both forms,
        the statistics match closely.

        We construct a low-rank covariance, compute the equivalent full matrix,
        and verify both produce the same firing statistics.
        """
        num_features = 4
        rank = 2
        batch_size = 20000

        # Create low-rank factors small enough that diag stays positive without clamping
        factor = torch.randn(num_features, rank) * 0.1
        factor_sq_sum = (factor**2).sum(dim=1)
        assert torch.all(
            factor_sq_sum < 1
        ), "Factor norms too large for valid correlation"
        diag = 1 - factor_sq_sum

        # Construct the full covariance matrix: C = F @ F.T + diag(D)
        full_cov = factor @ factor.T + torch.diag(diag)

        # Verify it has unit diagonal (guaranteed by construction)
        torch.testing.assert_close(
            torch.diag(full_cov), torch.ones(num_features), atol=1e-6, rtol=0
        )

        firing_probs = torch.tensor([0.25, 0.35, 0.45, 0.30])

        generator_full = ActivationGenerator(
            num_features=num_features,
            firing_probabilities=firing_probs,
            correlation_matrix=full_cov,
        )

        generator_low_rank = ActivationGenerator(
            num_features=num_features,
            firing_probabilities=firing_probs,
            correlation_matrix=(factor, diag),
        )

        # Generate many samples and compare statistics
        samples_full = generator_full.sample(batch_size)
        samples_low_rank = generator_low_rank.sample(batch_size)

        # Marginal probabilities should match
        probs_full = (samples_full > 0).float().mean(dim=0)
        probs_low_rank = (samples_low_rank > 0).float().mean(dim=0)

        torch.testing.assert_close(probs_full, probs_low_rank, atol=0.02, rtol=0)

        # Joint firing probabilities for pairs should match
        firing_full = (samples_full > 0).float()
        firing_low_rank = (samples_low_rank > 0).float()

        for i in range(num_features):
            for j in range(i + 1, num_features):
                joint_full = (firing_full[:, i] * firing_full[:, j]).mean()
                joint_low_rank = (firing_low_rank[:, i] * firing_low_rank[:, j]).mean()
                torch.testing.assert_close(
                    joint_full, joint_low_rank, atol=0.02, rtol=0
                )


class TestActivationGeneratorLowRankCorrelationMatrixValidation:
    def test_raises_on_wrong_factor_shape(self):
        factor = torch.randn(3, 2)
        diag = torch.ones(5)
        with pytest.raises(ValueError, match=r"must have shape \(5, rank\)"):
            ActivationGenerator(
                num_features=5,
                firing_probabilities=0.5,
                correlation_matrix=(factor, diag),
            )

    def test_raises_on_wrong_diag_shape(self):
        factor = torch.randn(5, 2)
        diag = torch.ones(3)
        with pytest.raises(ValueError, match=r"must have shape \(5,\)"):
            ActivationGenerator(
                num_features=5,
                firing_probabilities=0.5,
                correlation_matrix=(factor, diag),
            )

    def test_raises_on_non_positive_diag(self):
        factor = torch.randn(5, 2)
        diag = torch.tensor([1.0, 1.0, -0.1, 1.0, 1.0])
        with pytest.raises(ValueError, match="must have all positive values"):
            ActivationGenerator(
                num_features=5,
                firing_probabilities=0.5,
                correlation_matrix=(factor, diag),
            )
