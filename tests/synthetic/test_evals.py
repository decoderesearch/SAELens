from typing import Any

import pytest
import torch

from sae_lens.saes.sae import TrainingSAE
from sae_lens.synthetic import (
    ActivationGenerator,
    ClassificationMetrics,
    FeatureDictionary,
    SyntheticDataEvalResult,
    compute_classification_metrics,
    eval_sae_on_synthetic_data,
    feature_uniqueness,
    mean_correlation_coefficient,
)
from sae_lens.synthetic.evals import ExplainedVarianceCalculator


class TestSyntheticDataEvalResultToLogDict:
    def test_returns_all_fields_with_prefix(self) -> None:
        result = SyntheticDataEvalResult(
            true_l0=1.5,
            sae_l0=2.0,
            dead_latents=5,
            shrinkage=0.95,
            explained_variance=0.9,
            mcc=0.85,
            uniqueness=0.7,
            classification=ClassificationMetrics(
                precision=0.8, recall=0.75, f1_score=0.77, accuracy=0.9
            ),
        )

        log_dict = result.to_log_dict(prefix="test/")

        assert log_dict == {
            "test/true_l0": 1.5,
            "test/sae_l0": 2.0,
            "test/dead_latents": 5,
            "test/shrinkage": 0.95,
            "test/explained_variance": 0.9,
            "test/mcc": 0.85,
            "test/uniqueness": 0.7,
            "test/classification/precision": 0.8,
            "test/classification/recall": 0.75,
            "test/classification/f1_score": 0.77,
            "test/classification/accuracy": 0.9,
        }

    def test_empty_prefix(self) -> None:
        result = SyntheticDataEvalResult(
            true_l0=1.0,
            sae_l0=2.0,
            dead_latents=0,
            shrinkage=1.0,
            explained_variance=1.0,
            mcc=1.0,
            uniqueness=1.0,
            classification=ClassificationMetrics(
                precision=1.0, recall=1.0, f1_score=1.0, accuracy=1.0
            ),
        )

        log_dict = result.to_log_dict()

        assert "true_l0" in log_dict
        assert "classification/precision" in log_dict


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

    def test_duplicate_values(self) -> None:
        features = torch.randn(10, 800)
        repeated = features[0].expand(10, -1).clone()
        mcc = mean_correlation_coefficient(features, repeated)
        assert mcc < 0.3

    def test_duplicate_values_with_different_sizes(self) -> None:
        features = torch.randn(10, 800)
        repeated = features[0].expand(2, -1).clone()
        mcc = mean_correlation_coefficient(features, repeated)
        assert 0.4 < mcc < 0.6

    def test_parameter_order_does_not_matter(self) -> None:
        features = torch.randn(10, 800)
        repeated = features[0].expand(2, -1).clone()
        mcc1 = mean_correlation_coefficient(features, repeated)
        mcc2 = mean_correlation_coefficient(repeated, features)
        assert mcc1 == pytest.approx(mcc2)

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
        assert hasattr(result, "explained_variance")
        assert hasattr(result, "mcc")
        assert hasattr(result, "uniqueness")
        assert hasattr(result, "classification")

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

    def test_explained_variance_in_valid_range(self, eval_setup: EvalSetup) -> None:
        sae, feature_dict, activations_gen = eval_setup

        result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=1000,
        )

        # Explained variance can theoretically be negative if reconstruction is very bad,
        # but should typically be between 0 and 1
        assert result.explained_variance <= 1.0

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

    def test_uniqueness_in_valid_range(self, eval_setup: EvalSetup) -> None:
        sae, feature_dict, activations_gen = eval_setup

        result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=1000,
        )

        assert 0.0 <= result.uniqueness <= 1.0

    def test_classification_metrics_in_valid_range(self, eval_setup: EvalSetup) -> None:
        sae, feature_dict, activations_gen = eval_setup

        result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=1000,
        )

        assert isinstance(result.classification, ClassificationMetrics)
        assert 0.0 <= result.classification.precision <= 1.0
        assert 0.0 <= result.classification.recall <= 1.0
        assert 0.0 <= result.classification.f1_score <= 1.0
        assert 0.0 <= result.classification.accuracy <= 1.0

    def test_batch_size_produces_valid_results(self, eval_setup: EvalSetup) -> None:
        sae, feature_dict, activations_gen = eval_setup

        result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=1000,
            batch_size=100,
        )

        assert isinstance(result, SyntheticDataEvalResult)
        assert 0.0 <= result.mcc <= 1.0
        assert 0.0 <= result.uniqueness <= 1.0
        assert 0.0 <= result.classification.precision <= 1.0

    def test_batch_size_matches_unbatched_statistically(self) -> None:
        hidden_dim = 8
        num_features = 10

        feature_dict = FeatureDictionary(
            num_features=num_features, hidden_dim=hidden_dim
        )

        activations_gen = ActivationGenerator(
            num_features=num_features,
            firing_probabilities=0.2,
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

        # Run multiple times and check statistical consistency
        num_samples = 10000
        batch_sizes = [None, 100, 500, 1000]
        results = []

        for batch_size in batch_sizes:
            result = eval_sae_on_synthetic_data(
                sae=sae,
                feature_dict=feature_dict,
                activations_generator=activations_gen,
                num_samples=num_samples,
                batch_size=batch_size,
            )
            results.append(result)

        # All results should have similar true_l0 (expected: num_features * prob = 2.0)
        for result in results:
            assert abs(result.true_l0 - 2.0) < 0.2

        # MCC and uniqueness should be identical (only depend on decoder weights)
        for result in results:
            assert result.mcc == results[0].mcc
            assert result.uniqueness == results[0].uniqueness


class TestFeatureUniqueness:
    def test_identical_features_returns_one(self) -> None:
        features = torch.randn(10, 8)
        score = feature_uniqueness(features, features)
        assert abs(score - 1.0) < 1e-5

    def test_all_same_feature_returns_one_over_n(self) -> None:
        gt_features = torch.randn(10, 8)
        # All SAE features are the same (copies of first GT feature)
        sae_features = gt_features[0:1].expand(5, -1).clone()
        score = feature_uniqueness(sae_features, gt_features)
        # All 5 SAE latents match the same GT feature, so uniqueness = 1/5
        assert abs(score - 0.2) < 1e-5

    def test_empty_sae_features_returns_zero(self) -> None:
        gt_features = torch.randn(10, 8)
        sae_features = torch.empty(0, 8)
        score = feature_uniqueness(sae_features, gt_features)
        assert score == 0.0

    def test_partial_uniqueness(self) -> None:
        gt_features = torch.eye(8)  # 8 orthogonal features
        # 4 SAE latents: 2 unique, 2 duplicates
        sae_features = torch.stack(
            [
                gt_features[0],  # matches GT 0
                gt_features[1],  # matches GT 1
                gt_features[0],  # matches GT 0 (duplicate)
                gt_features[2],  # matches GT 2
            ]
        )
        score = feature_uniqueness(sae_features, gt_features)
        # 3 unique GT features matched / 4 SAE latents = 0.75
        assert abs(score - 0.75) < 1e-5

    def test_negated_features_still_match(self) -> None:
        gt_features = torch.randn(10, 8)
        sae_features = -gt_features  # Negated but should still match
        score = feature_uniqueness(sae_features, gt_features)
        assert abs(score - 1.0) < 1e-5

    def test_scaled_features_still_match(self) -> None:
        gt_features = torch.randn(10, 8)
        sae_features = gt_features * 5.0  # Scaled but should still match
        score = feature_uniqueness(sae_features, gt_features)
        assert abs(score - 1.0) < 1e-5

    def test_returns_float(self) -> None:
        features = torch.randn(5, 4)
        score = feature_uniqueness(features, features)
        assert isinstance(score, float)


class TestComputeClassificationMetrics:
    def test_perfect_classifier_has_perfect_metrics(self) -> None:
        gt_features = torch.eye(4)  # 4 orthogonal features
        sae_decoder = gt_features.clone()  # SAE decoder matches GT perfectly
        num_samples = 1000

        # Generate samples where SAE latents perfectly predict GT features
        gt_feature_acts = torch.zeros(num_samples, 4)
        sae_latents = torch.zeros(num_samples, 4)
        for i in range(num_samples):
            # Randomly activate one feature
            active_feature = i % 4
            gt_feature_acts[i, active_feature] = 1.0
            sae_latents[i, active_feature] = 1.0

        metrics = compute_classification_metrics(
            sae_latents=sae_latents,
            gt_feature_acts=gt_feature_acts,
            sae_decoder=sae_decoder,
            gt_features=gt_features,
        )

        assert abs(metrics.precision - 1.0) < 1e-5
        assert abs(metrics.recall - 1.0) < 1e-5
        assert abs(metrics.f1_score - 1.0) < 1e-5
        assert abs(metrics.accuracy - 1.0) < 1e-5

    def test_no_true_positives_has_zero_precision_recall_f1(self) -> None:
        gt_features = torch.eye(4)
        sae_decoder = gt_features.clone()
        num_samples = 100

        # SAE fires but GT never does (all false positives)
        gt_feature_acts = torch.zeros(num_samples, 4)
        sae_latents = torch.ones(num_samples, 4)

        metrics = compute_classification_metrics(
            sae_latents=sae_latents,
            gt_feature_acts=gt_feature_acts,
            sae_decoder=sae_decoder,
            gt_features=gt_features,
        )

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0

    def test_empty_sae_latents_returns_zeros(self) -> None:
        gt_features = torch.eye(4)
        sae_decoder = torch.empty(0, 4)
        gt_feature_acts = torch.zeros(100, 4)
        sae_latents = torch.empty(100, 0)

        metrics = compute_classification_metrics(
            sae_latents=sae_latents,
            gt_feature_acts=gt_feature_acts,
            sae_decoder=sae_decoder,
            gt_features=gt_features,
        )

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.accuracy == 0.0

    def test_returns_classification_metrics(self) -> None:
        gt_features = torch.randn(4, 8)
        sae_decoder = torch.randn(4, 8)
        gt_feature_acts = torch.rand(100, 4) > 0.5
        sae_latents = torch.rand(100, 4) > 0.5

        metrics = compute_classification_metrics(
            sae_latents=sae_latents.float(),
            gt_feature_acts=gt_feature_acts.float(),
            sae_decoder=sae_decoder,
            gt_features=gt_features,
        )

        assert isinstance(metrics, ClassificationMetrics)
        assert isinstance(metrics.precision, float)
        assert isinstance(metrics.recall, float)
        assert isinstance(metrics.f1_score, float)
        assert isinstance(metrics.accuracy, float)

    def test_partial_overlap_gives_intermediate_metrics(self) -> None:
        gt_features = torch.eye(2)
        sae_decoder = gt_features.clone()
        num_samples = 100

        # Half the samples: perfect match. Other half: SAE fires but GT doesn't
        gt_feature_acts = torch.zeros(num_samples, 2)
        sae_latents = torch.zeros(num_samples, 2)

        # First half: both fire for feature 0
        gt_feature_acts[:50, 0] = 1.0
        sae_latents[:50, 0] = 1.0

        # Second half: SAE fires but GT doesn't (false positive)
        sae_latents[50:, 0] = 1.0

        metrics = compute_classification_metrics(
            sae_latents=sae_latents,
            gt_feature_acts=gt_feature_acts,
            sae_decoder=sae_decoder,
            gt_features=gt_features,
        )

        # For feature 0: TP=50, FP=50, FN=0, TN=0
        # Precision = 50/100 = 0.5, Recall = 50/50 = 1.0
        # F1 = 2 * 0.5 * 1.0 / (0.5 + 1.0) = 2/3
        # For feature 1: TP=0, FP=0, FN=0, TN=100
        # Precision = 0 (undefined, treated as 0), Recall = 0, F1 = 0, Accuracy = 1.0
        # Mean precision = (0.5 + 0) / 2 = 0.25
        # Mean recall = (1.0 + 0) / 2 = 0.5
        assert 0.2 < metrics.precision < 0.3
        assert 0.4 < metrics.recall < 0.6


class TestExplainedVarianceCalculator:
    def test_perfect_reconstruction_returns_one(self) -> None:
        hidden_dim = 8
        calc = ExplainedVarianceCalculator(hidden_dim)

        # Perfect reconstruction: output = input
        input_data = torch.randn(1000, hidden_dim)
        output_data = input_data.clone()

        calc.add_batch(output_data, input_data)
        result = calc.compute()

        assert abs(result - 1.0) < 1e-5

    def test_zero_reconstruction_returns_near_zero(self) -> None:
        hidden_dim = 8
        calc = ExplainedVarianceCalculator(hidden_dim)

        # Zero reconstruction: output = 0, input has variance
        input_data = torch.randn(10000, hidden_dim)
        output_data = torch.zeros_like(input_data)

        calc.add_batch(output_data, input_data)
        result = calc.compute()

        # If output is zeros, MSE equals sum of squared inputs
        # For zero-mean input, explained variance should be near 0
        # For non-zero-mean input, it could be slightly different
        assert result < 0.1

    def test_constant_input_with_perfect_reconstruction(self) -> None:
        hidden_dim = 4
        calc = ExplainedVarianceCalculator(hidden_dim)

        # Constant input (zero variance) with perfect reconstruction
        input_data = torch.ones(100, hidden_dim)
        output_data = input_data.clone()

        calc.add_batch(output_data, input_data)
        result = calc.compute()

        # Zero variance case should return 1.0 (perfect reconstruction)
        assert abs(result - 1.0) < 1e-5

    def test_batched_computation_matches_single_batch(self) -> None:
        hidden_dim = 8
        num_samples = 1000

        # Generate data
        input_data = torch.randn(num_samples, hidden_dim)
        output_data = input_data + 0.1 * torch.randn(num_samples, hidden_dim)

        # Single batch
        calc_single = ExplainedVarianceCalculator(hidden_dim)
        calc_single.add_batch(output_data, input_data)
        result_single = calc_single.compute()

        # Multiple batches
        calc_batched = ExplainedVarianceCalculator(hidden_dim)
        batch_size = 100
        for i in range(0, num_samples, batch_size):
            calc_batched.add_batch(
                output_data[i : i + batch_size], input_data[i : i + batch_size]
            )
        result_batched = calc_batched.compute()

        assert abs(result_single - result_batched) < 1e-5

    def test_returns_float(self) -> None:
        hidden_dim = 4
        calc = ExplainedVarianceCalculator(hidden_dim)

        input_data = torch.randn(100, hidden_dim)
        output_data = input_data.clone()

        calc.add_batch(output_data, input_data)
        result = calc.compute()

        assert isinstance(result, float)

    def test_no_samples_returns_zero(self) -> None:
        hidden_dim = 8
        calc = ExplainedVarianceCalculator(hidden_dim)

        result = calc.compute()

        assert result == 0.0

    def test_partial_reconstruction_gives_intermediate_value(self) -> None:
        hidden_dim = 8
        calc = ExplainedVarianceCalculator(hidden_dim)

        # Add noise that explains away some variance
        input_data = torch.randn(10000, hidden_dim)
        noise_level = 0.5
        output_data = input_data + noise_level * torch.randn(10000, hidden_dim)

        calc.add_batch(output_data, input_data)
        result = calc.compute()

        # Should be between 0 and 1, closer to 1 since noise is smaller than signal
        assert 0.3 < result < 1.0

    def test_known_explained_variance(self) -> None:
        hidden_dim = 1
        num_samples = 100000

        # Create input with known variance
        input_data = torch.randn(num_samples, hidden_dim)

        # Create reconstruction with known MSE
        # MSE = 0.25 * Var(input), so explained variance = 1 - 0.25 = 0.75
        noise_std = 0.5  # MSE = noise_std^2 = 0.25
        output_data = input_data + noise_std * torch.randn(num_samples, hidden_dim)

        calc = ExplainedVarianceCalculator(hidden_dim)
        calc.add_batch(output_data, input_data)
        result = calc.compute()

        # The input has variance ~1 (standard normal), noise has variance 0.25
        # So explained_variance = 1 - 0.25/1.0 = 0.75
        assert abs(result - 0.75) < 0.02
