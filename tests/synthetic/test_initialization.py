import torch

from sae_lens.saes.standard_sae import StandardTrainingSAE, StandardTrainingSAEConfig
from sae_lens.synthetic import FeatureDictionary
from sae_lens.synthetic.initialization import init_sae_to_match_feature_dict
from sae_lens.util import cosine_similarities


def _create_sae(d_in: int, d_sae: int) -> StandardTrainingSAE:
    cfg = StandardTrainingSAEConfig(d_in=d_in, d_sae=d_sae)
    return StandardTrainingSAE(cfg)


class TestInitSaeToMatchFeatureDict:
    def test_initializes_encoder_to_match_features(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)

        init_sae_to_match_feature_dict(sae, feature_dict)

        # W_enc columns should match feature vectors
        features = feature_dict.feature_vectors  # [5, 8]
        w_enc = sae.W_enc.data[:, :5]  # [8, 5]

        cos_sims = cosine_similarities(features, w_enc.T)
        diagonal = torch.diag(cos_sims)
        assert torch.all(diagonal > 0.99)

    def test_initializes_decoder_as_encoder_transpose(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)

        init_sae_to_match_feature_dict(sae, feature_dict)

        torch.testing.assert_close(sae.W_dec.data, sae.W_enc.data.T)

    def test_noise_level_adds_perturbation(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)

        init_sae_to_match_feature_dict(sae, feature_dict, noise_level=0.5)

        # With noise, cosine similarity should be lower but still reasonable
        features = feature_dict.feature_vectors
        w_enc = sae.W_enc.data[:, :5]

        cos_sims = cosine_similarities(features, w_enc.T)
        diagonal = torch.diag(cos_sims)
        # Should still be correlated but not perfectly
        assert torch.all(diagonal > 0.5)
        assert not torch.all(diagonal > 0.99)  # Should have some noise

    def test_zero_noise_is_exact_match(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)

        init_sae_to_match_feature_dict(sae, feature_dict, noise_level=0.0)

        features = feature_dict.feature_vectors  # [5, 8]
        w_enc = sae.W_enc.data[:, :5].T  # [5, 8]

        torch.testing.assert_close(w_enc, features)

    def test_feature_ordering_permutes_features(self):
        feature_dict = FeatureDictionary(num_features=5, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=5)

        # Reverse the feature order
        ordering = torch.tensor([4, 3, 2, 1, 0])
        init_sae_to_match_feature_dict(
            sae, feature_dict, feature_ordering=ordering, noise_level=0.0
        )

        features = feature_dict.feature_vectors  # [5, 8]
        reordered_features = features[ordering]
        w_enc = sae.W_enc.data[:, :5].T  # [5, 8]

        torch.testing.assert_close(w_enc, reordered_features)

    def test_handles_more_sae_latents_than_features(self):
        feature_dict = FeatureDictionary(num_features=3, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=10)  # More SAE latents than features

        init_sae_to_match_feature_dict(sae, feature_dict, noise_level=0.0)

        # First 3 columns should match features
        features = feature_dict.feature_vectors  # [3, 8]
        w_enc = sae.W_enc.data[:, :3].T  # [3, 8]

        torch.testing.assert_close(w_enc, features)

    def test_handles_fewer_sae_latents_than_features(self):
        feature_dict = FeatureDictionary(num_features=10, hidden_dim=8)
        sae = _create_sae(d_in=8, d_sae=3)  # Fewer SAE latents than features

        init_sae_to_match_feature_dict(sae, feature_dict, noise_level=0.0)

        # All 3 columns should match first 3 features
        features = feature_dict.feature_vectors[:3]  # [3, 8]
        w_enc = sae.W_enc.data.T  # [3, 8]

        torch.testing.assert_close(w_enc, features)
