import torch

from sae_lens.synthetic import FeatureDictionary, orthogonalize_vectors


def test_orthogonalize_vectors_creates_unit_vectors():
    vectors = orthogonalize_vectors(num_vectors=10, vector_dim=10, num_steps=500)
    norms = torch.norm(vectors, dim=1)
    assert torch.allclose(norms, torch.ones(10), atol=1e-5)


def test_orthogonalize_vectors_creates_orthogonal_vectors():
    vectors = orthogonalize_vectors(
        num_vectors=10, vector_dim=10, target_cos_sim=0, num_steps=500
    )
    dot_products = vectors @ vectors.T
    expected = torch.eye(10)
    assert torch.allclose(dot_products, expected, atol=1e-5)


def test_orthogonalize_vectors_respects_target_cos_sim():
    target_cos_sim = 0.3
    vectors = orthogonalize_vectors(
        num_vectors=5, vector_dim=20, target_cos_sim=target_cos_sim, num_steps=500
    )
    dot_products = vectors @ vectors.T
    off_diagonal = dot_products[~torch.eye(5, dtype=bool)]
    assert torch.allclose(off_diagonal, torch.tensor(target_cos_sim), atol=0.05)


def test_FeatureDictionary_creates_correct_shape():
    feature_dict = FeatureDictionary(num_features=10, hidden_dim=8, ortho_num_steps=100)
    assert feature_dict.embed.weight.shape == (8, 10)
    assert feature_dict.feature_vectors.shape == (10, 8)


def test_FeatureDictionary_creates_orthogonal_features():
    feature_dict = FeatureDictionary(
        num_features=8, hidden_dim=8, target_cos_sim=0, ortho_num_steps=500
    )
    features = feature_dict.feature_vectors
    dot_products = features @ features.T
    expected = torch.eye(8)
    assert torch.allclose(dot_products, expected, atol=1e-5)


def test_FeatureDictionary_creates_unit_norm_features():
    feature_dict = FeatureDictionary(num_features=10, hidden_dim=8, ortho_num_steps=100)
    features = feature_dict.feature_vectors
    norms = torch.norm(features, dim=1)
    assert torch.allclose(norms, torch.ones(10), atol=1e-5)


def test_FeatureDictionary_forward_pass():
    feature_dict = FeatureDictionary(num_features=5, hidden_dim=4, ortho_num_steps=100)
    feature_activations = torch.zeros(3, 5)
    feature_activations[0, 0] = 1.0
    feature_activations[1, 2] = 0.5
    feature_activations[2, [0, 3]] = torch.tensor([1.0, 0.8])

    hidden = feature_dict(feature_activations)
    assert hidden.shape == (3, 4)


def test_FeatureDictionary_forward_produces_linear_combination():
    feature_dict = FeatureDictionary(num_features=3, hidden_dim=4, ortho_num_steps=100)
    features = feature_dict.feature_vectors

    activations = torch.tensor([[1.0, 0.0, 0.0]])
    hidden = feature_dict(activations)
    expected = features[0].unsqueeze(0)
    assert torch.allclose(hidden, expected, atol=1e-5)

    activations = torch.tensor([[1.0, 1.0, 0.0]])
    hidden = feature_dict(activations)
    expected = (features[0] + features[1]).unsqueeze(0)
    assert torch.allclose(hidden, expected, atol=1e-5)
