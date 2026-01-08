import pytest
import torch

from sae_lens.synthetic import ActivationGenerator
from sae_lens.synthetic.activation_generator import (
    _normalize_modifiers,
    _to_tensor,
    _validate_correlation_matrix,
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
        """Test that empty list of modifiers works."""
        generator = ActivationGenerator(
            num_features=3,
            firing_probabilities=0.5,
            modify_activations=[],
        )

        assert generator.modify_activations is None
        samples = generator.sample(batch_size=10)
        assert samples.shape == (10, 3)

    def test_with_single_modifier(self):
        """Test with a single modifier function."""

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
        """Test chained modifiers."""

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
        """Test that modifiers that produce negative values get relu applied."""

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
        """Test that correlation matrix affects feature firing patterns."""
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
        """Test that correlation doesn't change marginal firing probabilities much."""
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
        """Test that forward() returns same as sample()."""
        torch.manual_seed(42)
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
        """Test that a float can be passed instead of tensor."""
        generator = ActivationGenerator(
            num_features=5,
            firing_probabilities=0.3,  # float, not tensor
        )

        samples = generator.sample(batch_size=100)
        assert samples.shape == (100, 5)

    def test_accepts_string_dtype(self):
        """Test that string dtype works."""
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
        """Test that valid correlation matrix passes validation."""
        correlation_matrix = generate_random_correlation_matrix(num_features=3, seed=42)
        _validate_correlation_matrix(correlation_matrix, num_features=3)

    def test_identity_matrix_is_valid(self):
        """Test that identity matrix is a valid correlation matrix."""
        identity = torch.eye(5)
        _validate_correlation_matrix(identity, num_features=5)

    def test_raises_on_wrong_shape(self):
        """Test that wrong shape raises ValueError."""
        wrong_shape = torch.eye(3)
        with pytest.raises(ValueError, match="must have shape"):
            _validate_correlation_matrix(wrong_shape, num_features=5)

    def test_raises_on_non_unit_diagonal(self):
        """Test that non-unit diagonal raises ValueError."""
        matrix = torch.eye(3)
        matrix[0, 0] = 2.0
        with pytest.raises(ValueError, match="diagonal must be all 1s"):
            _validate_correlation_matrix(matrix, num_features=3)

    def test_raises_on_not_positive_definite(self):
        """Test that non-positive-definite matrix raises ValueError."""
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
        """Test that ActivationGenerator raises on wrong shape correlation matrix."""
        wrong_shape = torch.eye(3)
        with pytest.raises(ValueError, match="must have shape"):
            ActivationGenerator(
                num_features=5,
                firing_probabilities=0.5,
                correlation_matrix=wrong_shape,
            )

    def test_raises_on_non_unit_diagonal(self):
        """Test that ActivationGenerator raises on non-unit diagonal."""
        matrix = torch.eye(3)
        matrix[1, 1] = 0.5
        with pytest.raises(ValueError, match="diagonal must be all 1s"):
            ActivationGenerator(
                num_features=3,
                firing_probabilities=0.5,
                correlation_matrix=matrix,
            )

    def test_raises_on_not_positive_definite(self):
        """Test that ActivationGenerator raises on non-positive-definite matrix."""
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
