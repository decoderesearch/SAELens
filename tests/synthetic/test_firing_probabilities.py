import pytest
import torch

from sae_lens.synthetic import (
    linear_firing_probabilities,
    random_firing_probabilities,
    zipfian_firing_probabilities,
)


class TestZipfianFiringProbabilities:
    def test_returns_correct_shape(self) -> None:
        probs = zipfian_firing_probabilities(100)
        assert probs.shape == (100,)

    def test_first_element_is_max_prob(self) -> None:
        probs = zipfian_firing_probabilities(10, max_prob=0.5, min_prob=0.01)
        assert torch.isclose(probs[0], torch.tensor(0.5))

    def test_last_element_is_min_prob(self) -> None:
        probs = zipfian_firing_probabilities(10, max_prob=0.5, min_prob=0.01)
        assert torch.isclose(probs[-1], torch.tensor(0.01))

    def test_probabilities_are_descending(self) -> None:
        probs = zipfian_firing_probabilities(20)
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1]

    def test_higher_exponent_steeper_dropoff(self) -> None:
        probs_low = zipfian_firing_probabilities(10, exponent=0.5)
        probs_high = zipfian_firing_probabilities(10, exponent=2.0)

        # With higher exponent, the ratio between first and second should be larger
        ratio_low = probs_low[0] / probs_low[1]
        ratio_high = probs_high[0] / probs_high[1]
        assert ratio_high > ratio_low

    def test_single_feature_returns_max_prob(self) -> None:
        probs = zipfian_firing_probabilities(1, max_prob=0.7)
        assert torch.isclose(probs[0], torch.tensor(0.7))

    def test_all_values_in_range(self) -> None:
        probs = zipfian_firing_probabilities(100, max_prob=0.4, min_prob=0.05)
        assert torch.all(probs >= 0.05)
        assert torch.all(probs <= 0.4)

    def test_raises_on_invalid_num_features(self) -> None:
        with pytest.raises(ValueError, match="num_features must be at least 1"):
            zipfian_firing_probabilities(0)

    def test_raises_on_invalid_exponent(self) -> None:
        with pytest.raises(ValueError, match="exponent must be positive"):
            zipfian_firing_probabilities(10, exponent=0)

        with pytest.raises(ValueError, match="exponent must be positive"):
            zipfian_firing_probabilities(10, exponent=-1)

    def test_raises_on_invalid_prob_range(self) -> None:
        with pytest.raises(ValueError, match="Must have 0 < min_prob < max_prob <= 1"):
            zipfian_firing_probabilities(10, max_prob=0.1, min_prob=0.5)

        with pytest.raises(ValueError, match="Must have 0 < min_prob < max_prob <= 1"):
            zipfian_firing_probabilities(10, max_prob=1.5, min_prob=0.1)

        with pytest.raises(ValueError, match="Must have 0 < min_prob < max_prob <= 1"):
            zipfian_firing_probabilities(10, max_prob=0.5, min_prob=0)


class TestLinearFiringProbabilities:
    def test_returns_correct_shape(self) -> None:
        probs = linear_firing_probabilities(100)
        assert probs.shape == (100,)

    def test_first_element_is_max_prob(self) -> None:
        probs = linear_firing_probabilities(10, max_prob=0.5, min_prob=0.1)
        assert torch.isclose(probs[0], torch.tensor(0.5))

    def test_last_element_is_min_prob(self) -> None:
        probs = linear_firing_probabilities(10, max_prob=0.5, min_prob=0.1)
        assert torch.isclose(probs[-1], torch.tensor(0.1))

    def test_probabilities_are_evenly_spaced(self) -> None:
        probs = linear_firing_probabilities(5, max_prob=0.5, min_prob=0.1)
        expected = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1])
        torch.testing.assert_close(probs, expected)

    def test_intervals_are_equal(self) -> None:
        probs = linear_firing_probabilities(10, max_prob=0.9, min_prob=0.1)
        diffs = probs[:-1] - probs[1:]
        # All differences should be the same
        assert torch.allclose(diffs, diffs[0] * torch.ones_like(diffs))

    def test_single_feature_returns_max_prob(self) -> None:
        probs = linear_firing_probabilities(1, max_prob=0.7)
        assert torch.isclose(probs[0], torch.tensor(0.7))

    def test_equal_min_max_returns_constant(self) -> None:
        probs = linear_firing_probabilities(5, max_prob=0.3, min_prob=0.3)
        expected = torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3])
        torch.testing.assert_close(probs, expected)

    def test_all_values_in_range(self) -> None:
        probs = linear_firing_probabilities(100, max_prob=0.4, min_prob=0.05)
        assert torch.all(probs >= 0.05)
        assert torch.all(probs <= 0.4)

    def test_raises_on_invalid_num_features(self) -> None:
        with pytest.raises(ValueError, match="num_features must be at least 1"):
            linear_firing_probabilities(0)

    def test_raises_on_invalid_prob_range(self) -> None:
        with pytest.raises(ValueError, match="Must have 0 < min_prob <= max_prob <= 1"):
            linear_firing_probabilities(10, max_prob=0.1, min_prob=0.5)

        with pytest.raises(ValueError, match="Must have 0 < min_prob <= max_prob <= 1"):
            linear_firing_probabilities(10, max_prob=1.5, min_prob=0.1)


class TestRandomFiringProbabilities:
    def test_returns_correct_shape(self) -> None:
        probs = random_firing_probabilities(100)
        assert probs.shape == (100,)

    def test_all_values_in_range(self) -> None:
        probs = random_firing_probabilities(1000, max_prob=0.4, min_prob=0.05)
        assert torch.all(probs >= 0.05)
        assert torch.all(probs <= 0.4)

    def test_seed_produces_reproducible_results(self) -> None:
        probs1 = random_firing_probabilities(50, seed=42)
        probs2 = random_firing_probabilities(50, seed=42)
        torch.testing.assert_close(probs1, probs2)

    def test_different_seeds_produce_different_results(self) -> None:
        probs1 = random_firing_probabilities(50, seed=42)
        probs2 = random_firing_probabilities(50, seed=123)
        assert not torch.allclose(probs1, probs2)

    def test_values_are_uniformly_distributed(self) -> None:
        # With enough samples, mean should be close to midpoint
        probs = random_firing_probabilities(10000, max_prob=0.8, min_prob=0.2, seed=42)
        expected_mean = (0.8 + 0.2) / 2
        assert abs(probs.mean().item() - expected_mean) < 0.02

    def test_raises_on_invalid_num_features(self) -> None:
        with pytest.raises(ValueError, match="num_features must be at least 1"):
            random_firing_probabilities(0)

    def test_raises_on_invalid_prob_range(self) -> None:
        with pytest.raises(ValueError, match="Must have 0 < min_prob < max_prob <= 1"):
            random_firing_probabilities(10, max_prob=0.1, min_prob=0.5)

        with pytest.raises(ValueError, match="Must have 0 < min_prob < max_prob <= 1"):
            random_firing_probabilities(10, max_prob=0.5, min_prob=0.5)
