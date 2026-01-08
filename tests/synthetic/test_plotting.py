from pathlib import Path

import pytest
import torch

from sae_lens.saes.standard_sae import StandardTrainingSAE, StandardTrainingSAEConfig
from sae_lens.synthetic import FeatureDictionary
from sae_lens.synthetic.initialization import init_sae_to_match_feature_dict
from sae_lens.synthetic.plotting import (
    find_best_feature_ordering,
    find_best_feature_ordering_across_saes,
    find_best_feature_ordering_from_sae,
    plot_sae_feature_similarity,
)
from sae_lens.util import cosine_similarities


def _create_sae(d_in: int, d_sae: int) -> StandardTrainingSAE:
    cfg = StandardTrainingSAEConfig(d_in=d_in, d_sae=d_sae)
    return StandardTrainingSAE(cfg)


class TestFindBestFeatureOrdering:
    def test_returns_permutation(self):
        sae_features = torch.randn(5, 10)
        true_features = torch.randn(5, 10)

        ordering = find_best_feature_ordering(sae_features, true_features)

        assert ordering.shape == (5,)
        assert set(ordering.tolist()) == {0, 1, 2, 3, 4}

    def test_aligns_identical_features(self):
        # Create features where sae[i] = true[4-i] (reversed order)
        true_features = torch.eye(5, 10)
        sae_features = torch.flip(true_features, dims=[0])

        ordering = find_best_feature_ordering(sae_features, true_features)
        reordered_sae = sae_features[ordering]

        # After reordering, should match true features
        cos_sims = cosine_similarities(reordered_sae, true_features)
        diagonal = torch.diag(cos_sims)

        # Each reordered SAE feature should match its corresponding true feature
        assert torch.all(diagonal > 0.99)

    def test_with_feature_dict(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        true_features = feature_dict.feature_vectors.detach()

        # Create shuffled SAE features
        perm = torch.randperm(5)
        sae_features = true_features[perm]

        ordering = find_best_feature_ordering(sae_features, true_features)

        # After reordering, should approximately recover original order
        reordered = sae_features[ordering]
        cos_sims = cosine_similarities(reordered, true_features)
        diagonal = torch.diag(cos_sims)

        assert torch.all(diagonal > 0.9)


class TestFindBestFeatureOrderingFromSae:
    def test_returns_permutation(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)

        ordering = find_best_feature_ordering_from_sae(sae, feature_dict)

        assert ordering.shape == (5,)
        assert set(ordering.tolist()) == {0, 1, 2, 3, 4}

    def test_with_initialized_sae(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)

        # Initialize SAE to match feature dict
        init_sae_to_match_feature_dict(sae, feature_dict, noise_level=0.0)

        ordering = find_best_feature_ordering_from_sae(sae, feature_dict)

        # Reordered decoder should align with true features
        reordered_dec = sae.W_dec.detach()[ordering]
        cos_sims = cosine_similarities(reordered_dec, feature_dict.feature_vectors)
        diagonal = torch.diag(cos_sims)

        assert torch.all(diagonal > 0.9)


class TestFindBestFeatureOrderingAcrossSaes:
    def test_returns_permutation(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae1 = _create_sae(d_in=8, d_sae=5)
        sae2 = _create_sae(d_in=8, d_sae=5)

        ordering = find_best_feature_ordering_across_saes([sae1, sae2], feature_dict)

        assert ordering.shape == (5,)
        assert set(ordering.tolist()) == {0, 1, 2, 3, 4}

    def test_raises_on_empty_list(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)

        with pytest.raises(ValueError, match="No SAEs provided"):
            find_best_feature_ordering_across_saes([], feature_dict)

    def test_selects_best_ordering(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)

        # Create one good SAE (initialized to match) and one random SAE
        good_sae = _create_sae(d_in=8, d_sae=5)
        init_sae_to_match_feature_dict(good_sae, feature_dict, noise_level=0.0)

        bad_sae = _create_sae(d_in=8, d_sae=5)  # Random weights

        ordering = find_best_feature_ordering_across_saes(
            [bad_sae, good_sae], feature_dict
        )

        # Should select ordering that works for the good SAE
        reordered_dec = good_sae.W_dec.detach()[ordering]
        cos_sims = cosine_similarities(reordered_dec, feature_dict.feature_vectors)
        diagonal = torch.diag(cos_sims)

        assert torch.all(diagonal > 0.9)


class TestPlotSaeFeatureSimilarity:
    def test_runs_without_error_encoder_decoder(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)

        # Should not raise
        plot_sae_feature_similarity(
            sae, feature_dict, show_plot=False, decoder_only=False
        )

    def test_runs_without_error_decoder_only(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)

        # Should not raise
        plot_sae_feature_similarity(
            sae, feature_dict, show_plot=False, decoder_only=True
        )

    def test_with_custom_title(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)

        # Should not raise
        plot_sae_feature_similarity(
            sae, feature_dict, title="Custom Title", show_plot=False
        )

    def test_with_reorder_features_true(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)

        # Should not raise
        plot_sae_feature_similarity(
            sae, feature_dict, reorder_features=True, show_plot=False
        )

    def test_with_reorder_features_tensor(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)
        ordering = torch.tensor([4, 3, 2, 1, 0])

        # Should not raise
        plot_sae_feature_similarity(
            sae, feature_dict, reorder_features=ordering, show_plot=False
        )

    def test_with_show_values(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)

        # Should not raise
        plot_sae_feature_similarity(
            sae, feature_dict, show_values=True, show_plot=False
        )

    def test_with_show_values_decoder_only(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)

        # Should not raise
        plot_sae_feature_similarity(
            sae, feature_dict, show_values=True, decoder_only=True, show_plot=False
        )

    def test_save_to_file(self, tmp_path: Path):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)
        save_path = tmp_path / "test_plot.png"

        plot_sae_feature_similarity(
            sae, feature_dict, save_path=str(save_path), show_plot=False
        )

        assert save_path.exists()

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)
        save_path = tmp_path / "nested" / "dir" / "test_plot.png"

        plot_sae_feature_similarity(
            sae, feature_dict, save_path=str(save_path), show_plot=False
        )

        assert save_path.exists()

    def test_custom_dimensions(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)

        # Should not raise with custom dimensions
        plot_sae_feature_similarity(
            sae, feature_dict, height=600, width=1000, show_plot=False
        )
