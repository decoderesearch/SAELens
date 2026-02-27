import pytest
import torch

from sae_lens.saes.standard_sae import StandardTrainingSAE, StandardTrainingSAEConfig
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
from sae_lens.training.activation_scaler import ActivationScaler
from tests.helpers import random_params


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
        features = torch.randn(10, 8)
        mcc = mean_correlation_coefficient(features, features)
        assert mcc == pytest.approx(1.0, abs=1e-5)

    def test_negated_features_returns_one(self) -> None:
        features = torch.randn(10, 8)
        mcc = mean_correlation_coefficient(features, -features)
        assert mcc == pytest.approx(1.0, abs=1e-5)

    def test_permuted_features_returns_one(self) -> None:
        features = torch.randn(10, 8)
        perm = torch.randperm(10)
        permuted = features[perm]
        mcc = mean_correlation_coefficient(features, permuted)
        assert mcc == pytest.approx(1.0, abs=1e-5)

    def test_random_features_low_correlation(self) -> None:
        torch.manual_seed(42)
        features_a = torch.randn(10, 256)
        features_b = torch.randn(10, 256)
        mcc = mean_correlation_coefficient(features_a, features_b)
        assert mcc < 0.3

    def test_scaled_features_returns_one(self) -> None:
        features = torch.randn(10, 8)
        scaled = features * 5.0
        mcc = mean_correlation_coefficient(features, scaled)
        assert mcc == pytest.approx(1.0, abs=1e-5)

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
        matched = torch.randn(5, 8)
        random_a = torch.randn(5, 8)
        random_b = torch.randn(5, 8)

        features_a = torch.cat([matched, random_a])
        features_b = torch.cat([matched, random_b])

        mcc = mean_correlation_coefficient(features_a, features_b)
        assert 0.3 < mcc < 1.0

    def test_different_num_features_uses_min(self) -> None:
        features_a = torch.randn(10, 8)
        features_b = torch.randn(15, 8)

        mcc = mean_correlation_coefficient(features_a, features_b)
        assert 0.0 <= mcc <= 1.0

    def test_returns_float(self) -> None:
        features = torch.randn(5, 4)
        mcc = mean_correlation_coefficient(features, features)
        assert isinstance(mcc, float)

    def test_single_feature_identical(self) -> None:
        features = torch.randn(1, 8)
        mcc = mean_correlation_coefficient(features, features)
        assert mcc == pytest.approx(1.0, abs=1e-5)

    def test_handles_zero_norm_gracefully(self) -> None:
        features_a = torch.randn(5, 4)
        features_b = torch.randn(5, 4)
        features_b[0] = 1e-10

        mcc = mean_correlation_coefficient(features_a, features_b)
        assert 0.0 <= mcc <= 1.0


EvalSetup = tuple[StandardTrainingSAE, FeatureDictionary, ActivationGenerator]


@pytest.fixture
def eval_setup() -> EvalSetup:
    hidden_dim = 8
    num_features = 10

    feature_dict = FeatureDictionary(num_features=num_features, hidden_dim=hidden_dim)

    activations_gen = ActivationGenerator(
        num_features=num_features,
        firing_probabilities=0.1,
    )

    cfg = StandardTrainingSAEConfig(
        d_in=hidden_dim,
        d_sae=num_features,
        apply_b_dec_to_input=True,
    )
    sae = StandardTrainingSAE(cfg)

    return sae, feature_dict, activations_gen


class TestEvalSaeOnSyntheticData:
    def test_returns_correct_type(self, eval_setup: EvalSetup) -> None:
        sae, feature_dict, activations_gen = eval_setup

        result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=1000,
        )

        assert isinstance(result, SyntheticDataEvalResult)

    def test_result_has_all_fields(self, eval_setup: EvalSetup) -> None:
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
        sae, feature_dict, _ = eval_setup

        activations_gen = ActivationGenerator(
            num_features=10,
            firing_probabilities=0.2,
        )

        result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=100000,
        )

        # Expected L0 = 10 features * 0.2 probability = 2.0
        # Std per sample = sqrt(10 * 0.2 * 0.8) ≈ 1.26
        # Standard error of mean = 1.26 / sqrt(100000) ≈ 0.004
        assert result.true_l0 == pytest.approx(2.0, abs=0.03)

    def test_dead_latents_is_non_negative(self, eval_setup: EvalSetup) -> None:
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

        assert result.explained_variance <= 1.0

    def test_mcc_in_valid_range(self, eval_setup: EvalSetup) -> None:
        sae, feature_dict, activations_gen = eval_setup

        result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=1000,
        )

        assert 0.0 <= result.mcc <= 1.0

    def test_sae_initialized_to_ground_truth_has_high_mcc(self) -> None:
        hidden_dim = 8
        num_features = 8

        feature_dict = FeatureDictionary(
            num_features=num_features,
            hidden_dim=hidden_dim,
        )

        activations_gen = ActivationGenerator(
            num_features=num_features,
            firing_probabilities=0.1,
        )

        cfg = StandardTrainingSAEConfig(
            d_in=hidden_dim,
            d_sae=num_features,
            apply_b_dec_to_input=False,
        )
        sae = StandardTrainingSAE(cfg)

        with torch.no_grad():
            sae.W_dec.data = feature_dict.feature_vectors.clone()

        result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=1000,
        )

        assert result.mcc > 0.99

    def test_num_samples_affects_precision(self) -> None:
        hidden_dim = 8
        num_features = 10

        feature_dict = FeatureDictionary(
            num_features=num_features, hidden_dim=hidden_dim
        )

        activations_gen = ActivationGenerator(
            num_features=num_features,
            firing_probabilities=0.1,
        )

        cfg = StandardTrainingSAEConfig(d_in=hidden_dim, d_sae=num_features)
        sae = StandardTrainingSAE(cfg)

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

        cfg = StandardTrainingSAEConfig(d_in=hidden_dim, d_sae=num_features)
        sae = StandardTrainingSAE(cfg)

        num_samples = 100000
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

        # Expected L0 = 10 features * 0.2 probability = 2.0
        for result in results:
            assert result.true_l0 == pytest.approx(2.0, abs=0.03)

        for result in results:
            assert result.mcc == results[0].mcc
            assert result.uniqueness == results[0].uniqueness


class TestEvalSaeOnSyntheticDataWithActivationScaling:
    def test_scaled_eval_matches_folded_sae_eval(self) -> None:
        hidden_dim = 6
        num_features = 8
        scaling_factor = 3.0

        feature_dict = FeatureDictionary(
            num_features=num_features,
            hidden_dim=hidden_dim,
        )
        activations_gen = ActivationGenerator(
            num_features=num_features,
            firing_probabilities=0.2,
        )

        cfg = StandardTrainingSAEConfig(d_in=hidden_dim, d_sae=num_features)
        sae = StandardTrainingSAE(cfg)
        random_params(sae)

        folded_cfg = StandardTrainingSAEConfig(d_in=hidden_dim, d_sae=num_features)
        folded_sae = StandardTrainingSAE(folded_cfg)
        folded_sae.load_state_dict(sae.state_dict())

        folded_sae.fold_activation_norm_scaling_factor(scaling_factor)

        result_with_scaling = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=2_000_000,
            batch_size=10000,
            activation_scaler=ActivationScaler(scaling_factor=scaling_factor),
        )

        result_folded = eval_sae_on_synthetic_data(
            sae=folded_sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=2_000_000,
            batch_size=10000,
            activation_scaler=None,
        )

        assert result_with_scaling.sae_l0 == pytest.approx(
            result_folded.sae_l0, rel=0.02
        )
        assert result_with_scaling.shrinkage == pytest.approx(
            result_folded.shrinkage, rel=0.02
        )
        assert result_with_scaling.explained_variance == pytest.approx(
            result_folded.explained_variance, rel=0.02
        )
        assert result_with_scaling.classification.precision == pytest.approx(
            result_folded.classification.precision, rel=0.02
        )
        assert result_with_scaling.classification.recall == pytest.approx(
            result_folded.classification.recall, rel=0.02
        )


class TestFeatureUniqueness:
    def test_identical_features_returns_one(self) -> None:
        features = torch.randn(10, 8)
        score = feature_uniqueness(features, features)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_all_same_feature_returns_one_over_n(self) -> None:
        gt_features = torch.randn(10, 8)
        sae_features = gt_features[0:1].expand(5, -1).clone()
        score = feature_uniqueness(sae_features, gt_features)
        assert score == pytest.approx(0.2, abs=1e-5)

    def test_empty_sae_features_returns_zero(self) -> None:
        gt_features = torch.randn(10, 8)
        sae_features = torch.empty(0, 8)
        score = feature_uniqueness(sae_features, gt_features)
        assert score == 0.0

    def test_partial_uniqueness(self) -> None:
        gt_features = torch.eye(8)
        sae_features = torch.stack(
            [
                gt_features[0],
                gt_features[1],
                gt_features[0],
                gt_features[2],
            ]
        )
        score = feature_uniqueness(sae_features, gt_features)
        assert score == pytest.approx(0.75, abs=1e-5)

    def test_negated_features_still_match(self) -> None:
        gt_features = torch.randn(10, 8)
        sae_features = -gt_features
        score = feature_uniqueness(sae_features, gt_features)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_scaled_features_still_match(self) -> None:
        gt_features = torch.randn(10, 8)
        sae_features = gt_features * 5.0
        score = feature_uniqueness(sae_features, gt_features)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_returns_float(self) -> None:
        features = torch.randn(5, 4)
        score = feature_uniqueness(features, features)
        assert isinstance(score, float)


class TestComputeClassificationMetrics:
    def test_perfect_classifier_has_perfect_metrics(self) -> None:
        gt_features = torch.eye(4)
        sae_decoder = gt_features.clone()
        num_samples = 1000

        gt_feature_acts = torch.zeros(num_samples, 4)
        sae_latents = torch.zeros(num_samples, 4)
        for i in range(num_samples):
            active_feature = i % 4
            gt_feature_acts[i, active_feature] = 1.0
            sae_latents[i, active_feature] = 1.0

        metrics = compute_classification_metrics(
            sae_latents=sae_latents,
            gt_feature_acts=gt_feature_acts,
            sae_decoder=sae_decoder,
            gt_features=gt_features,
        )

        assert metrics.precision == pytest.approx(1.0, abs=1e-5)
        assert metrics.recall == pytest.approx(1.0, abs=1e-5)
        assert metrics.f1_score == pytest.approx(1.0, abs=1e-5)
        assert metrics.accuracy == pytest.approx(1.0, abs=1e-5)

    def test_no_true_positives_has_zero_precision_recall_f1(self) -> None:
        gt_features = torch.eye(4)
        sae_decoder = gt_features.clone()
        num_samples = 100

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

        gt_feature_acts = torch.zeros(num_samples, 2)
        sae_latents = torch.zeros(num_samples, 2)

        gt_feature_acts[:50, 0] = 1.0
        sae_latents[:50, 0] = 1.0

        sae_latents[50:, 0] = 1.0

        metrics = compute_classification_metrics(
            sae_latents=sae_latents,
            gt_feature_acts=gt_feature_acts,
            sae_decoder=sae_decoder,
            gt_features=gt_features,
        )

        assert 0.2 < metrics.precision < 0.3
        assert 0.4 < metrics.recall < 0.6


class TestExplainedVarianceCalculator:
    def test_perfect_reconstruction_returns_one(self) -> None:
        hidden_dim = 8
        calc = ExplainedVarianceCalculator(hidden_dim)

        input_data = torch.randn(1000, hidden_dim)
        output_data = input_data.clone()

        calc.add_batch(output_data, input_data)
        result = calc.compute()

        assert result == pytest.approx(1.0, abs=1e-5)

    def test_zero_reconstruction_returns_near_zero(self) -> None:
        hidden_dim = 8
        calc = ExplainedVarianceCalculator(hidden_dim)

        input_data = torch.randn(10000, hidden_dim)
        output_data = torch.zeros_like(input_data)

        calc.add_batch(output_data, input_data)
        result = calc.compute()

        assert result < 0.1

    def test_constant_input_with_perfect_reconstruction(self) -> None:
        hidden_dim = 4
        calc = ExplainedVarianceCalculator(hidden_dim)

        input_data = torch.ones(100, hidden_dim)
        output_data = input_data.clone()

        calc.add_batch(output_data, input_data)
        result = calc.compute()

        assert result == pytest.approx(1.0, abs=1e-5)

    def test_batched_computation_matches_single_batch(self) -> None:
        hidden_dim = 8
        num_samples = 1000

        input_data = torch.randn(num_samples, hidden_dim)
        output_data = input_data + 0.1 * torch.randn(num_samples, hidden_dim)

        calc_single = ExplainedVarianceCalculator(hidden_dim)
        calc_single.add_batch(output_data, input_data)
        result_single = calc_single.compute()

        calc_batched = ExplainedVarianceCalculator(hidden_dim)
        batch_size = 100
        for i in range(0, num_samples, batch_size):
            calc_batched.add_batch(
                output_data[i : i + batch_size], input_data[i : i + batch_size]
            )
        result_batched = calc_batched.compute()

        assert result_single == pytest.approx(result_batched, abs=1e-5)

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

        input_data = torch.randn(10000, hidden_dim)
        noise_level = 0.5
        output_data = input_data + noise_level * torch.randn(10000, hidden_dim)

        calc.add_batch(output_data, input_data)
        result = calc.compute()

        assert 0.3 < result < 1.0

    def test_known_explained_variance(self) -> None:
        hidden_dim = 1
        num_samples = 100000

        input_data = torch.randn(num_samples, hidden_dim)

        noise_std = 0.5
        output_data = input_data + noise_std * torch.randn(num_samples, hidden_dim)

        calc = ExplainedVarianceCalculator(hidden_dim)
        calc.add_batch(output_data, input_data)
        result = calc.compute()

        assert result == pytest.approx(0.75, abs=0.02)
