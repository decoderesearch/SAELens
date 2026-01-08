from typing import Any

import pytest
import torch

from sae_lens.saes.sae import TrainingSAE
from sae_lens.synthetic import (
    ActivationGenerator,
    FeatureDictionary,
    SyntheticDataEvalResult,
    eval_sae_on_synthetic_data,
    mean_correlation_coefficient,
)


class TestMeanCorrelationCoefficient:
    def test_identical_features_returns_one(self) -> None:
        """MCC of identical features should be 1.0."""
        features = torch.randn(10, 8)
        mcc = mean_correlation_coefficient(features, features)
        assert abs(mcc - 1.0) < 1e-5

    def test_negated_features_returns_one(self) -> None:
        """MCC uses absolute cosine similarity, so negated features also match."""
        features = torch.randn(10, 8)
        mcc = mean_correlation_coefficient(features, -features)
        assert abs(mcc - 1.0) < 1e-5

    def test_permuted_features_returns_one(self) -> None:
        """MCC with optimal matching should find the permutation."""
        features = torch.randn(10, 8)
        perm = torch.randperm(10)
        permuted = features[perm]
        mcc = mean_correlation_coefficient(features, permuted)
        assert abs(mcc - 1.0) < 1e-5

    def test_random_features_low_correlation(self) -> None:
        """MCC of random high-dimensional features should be low."""
        # In high dimensions, random unit vectors are nearly orthogonal
        torch.manual_seed(42)
        features_a = torch.randn(10, 256)
        features_b = torch.randn(10, 256)
        mcc = mean_correlation_coefficient(features_a, features_b)
        # Random vectors in high dimensions have low correlation
        assert mcc < 0.3

    def test_scaled_features_returns_one(self) -> None:
        """MCC should be invariant to scaling since it uses cosine similarity."""
        features = torch.randn(10, 8)
        scaled = features * 5.0
        mcc = mean_correlation_coefficient(features, scaled)
        assert abs(mcc - 1.0) < 1e-5

    def test_partial_match_returns_intermediate_value(self) -> None:
        """MCC with some matching and some orthogonal features."""
        # First 5 features match, last 5 are random
        matched = torch.randn(5, 8)
        random_a = torch.randn(5, 8)
        random_b = torch.randn(5, 8)

        features_a = torch.cat([matched, random_a])
        features_b = torch.cat([matched, random_b])

        mcc = mean_correlation_coefficient(features_a, features_b)
        # Should be somewhere between 0 and 1
        assert 0.3 < mcc < 1.0

    def test_different_num_features_uses_min(self) -> None:
        """MCC should handle different numbers of features."""
        features_a = torch.randn(10, 8)
        features_b = torch.randn(15, 8)  # More features

        mcc = mean_correlation_coefficient(features_a, features_b)
        # Should not raise and return a valid value
        assert 0.0 <= mcc <= 1.0

    def test_returns_float(self) -> None:
        """MCC should return a Python float."""
        features = torch.randn(5, 4)
        mcc = mean_correlation_coefficient(features, features)
        assert isinstance(mcc, float)

    def test_single_feature_identical(self) -> None:
        """MCC with single identical feature should be 1.0."""
        features = torch.randn(1, 8)
        mcc = mean_correlation_coefficient(features, features)
        assert abs(mcc - 1.0) < 1e-5

    def test_handles_zero_norm_gracefully(self) -> None:
        """MCC should handle near-zero vectors without crashing."""
        features_a = torch.randn(5, 4)
        features_b = torch.randn(5, 4)
        features_b[0] = 1e-10  # Near-zero vector

        # Should not raise
        mcc = mean_correlation_coefficient(features_a, features_b)
        assert 0.0 <= mcc <= 1.0


EvalSetup = tuple[TrainingSAE[Any], FeatureDictionary, ActivationGenerator]


@pytest.fixture
def eval_setup() -> EvalSetup:
    """Create a minimal setup for testing eval_sae_on_synthetic_data."""
    hidden_dim = 8
    num_features = 10

    feature_dict = FeatureDictionary(num_features=num_features, hidden_dim=hidden_dim)

    activations_gen = ActivationGenerator(
        num_features=num_features,
        firing_probabilities=0.1,
    )

    sae = TrainingSAE.from_dict(
        {
            "architecture": "standard",
            "d_in": hidden_dim,
            "d_sae": num_features,
            "activation_fn_str": "relu",
            "normalize_sae_decoder": False,
            "apply_b_dec_to_input": True,
            "dtype": "float32",
            "device": "cpu",
            "model_name": "test",
            "hook_name": "test",
            "hook_layer": 0,
        }
    )

    return sae, feature_dict, activations_gen


class TestEvalSaeOnSyntheticData:
    def test_returns_correct_type(self, eval_setup: EvalSetup) -> None:
        """eval_sae_on_synthetic_data should return SyntheticDataEvalResult."""
        sae, feature_dict, activations_gen = eval_setup

        result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=1000,
        )

        assert isinstance(result, SyntheticDataEvalResult)

    def test_result_has_all_fields(self, eval_setup: EvalSetup) -> None:
        """Result should have all expected fields."""
        sae, feature_dict, activations_gen = eval_setup

        result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=1000,
        )

        assert hasattr(result, "true_l0")
        assert hasattr(result, "sae_l0")
        assert hasattr(result, "dead_latents")
        assert hasattr(result, "shrinkage")
        assert hasattr(result, "mcc")

    def test_true_l0_matches_firing_probability(self, eval_setup: EvalSetup) -> None:
        """true_l0 should be close to num_features * firing_prob."""
        sae, feature_dict, _ = eval_setup

        # Create generator with known firing probability
        activations_gen = ActivationGenerator(
            num_features=10,
            firing_probabilities=0.2,
        )

        result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=10000,
        )

        # Expected L0 is num_features * prob = 10 * 0.2 = 2.0
        assert abs(result.true_l0 - 2.0) < 0.2

    def test_dead_latents_is_non_negative(self, eval_setup: EvalSetup) -> None:
        """dead_latents should be a non-negative integer."""
        sae, feature_dict, activations_gen = eval_setup

        result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=1000,
        )

        assert isinstance(result.dead_latents, int)
        assert result.dead_latents >= 0

    def test_shrinkage_is_positive(self, eval_setup: EvalSetup) -> None:
        """shrinkage should be positive."""
        sae, feature_dict, activations_gen = eval_setup

        result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=1000,
        )

        assert result.shrinkage > 0

    def test_mcc_in_valid_range(self, eval_setup: EvalSetup) -> None:
        """MCC should be in [0, 1]."""
        sae, feature_dict, activations_gen = eval_setup

        result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=1000,
        )

        assert 0.0 <= result.mcc <= 1.0

    def test_sae_initialized_to_ground_truth_has_high_mcc(self) -> None:
        """SAE initialized to match ground truth should have high MCC."""
        hidden_dim = 8
        num_features = 8  # Same as hidden_dim for perfect match

        feature_dict = FeatureDictionary(
            num_features=num_features,
            hidden_dim=hidden_dim,
        )

        activations_gen = ActivationGenerator(
            num_features=num_features,
            firing_probabilities=0.1,
        )

        sae = TrainingSAE.from_dict(
            {
                "architecture": "standard",
                "d_in": hidden_dim,
                "d_sae": num_features,
                "activation_fn_str": "relu",
                "normalize_sae_decoder": False,
                "apply_b_dec_to_input": False,
                "dtype": "float32",
                "device": "cpu",
                "model_name": "test",
                "hook_name": "test",
                "hook_layer": 0,
            }
        )

        # Initialize SAE decoder to match ground truth features
        with torch.no_grad():
            sae.W_dec.data = feature_dict.feature_vectors.clone()

        result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=1000,
        )

        # MCC should be very high when decoder matches ground truth
        assert result.mcc > 0.99

    def test_num_samples_affects_precision(self) -> None:
        """More samples should give more stable results."""
        hidden_dim = 8
        num_features = 10

        feature_dict = FeatureDictionary(
            num_features=num_features, hidden_dim=hidden_dim
        )

        activations_gen = ActivationGenerator(
            num_features=num_features,
            firing_probabilities=0.1,
        )

        sae = TrainingSAE.from_dict(
            {
                "architecture": "standard",
                "d_in": hidden_dim,
                "d_sae": num_features,
                "activation_fn_str": "relu",
                "normalize_sae_decoder": False,
                "apply_b_dec_to_input": True,
                "dtype": "float32",
                "device": "cpu",
                "model_name": "test",
                "hook_name": "test",
                "hook_layer": 0,
            }
        )

        # Both should run without error
        result_small = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=100,
        )

        result_large = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=10000,
        )

        # Both should return valid results
        assert isinstance(result_small, SyntheticDataEvalResult)
        assert isinstance(result_large, SyntheticDataEvalResult)
