import torch

from sae_lens.toy_model import FeatureDictionary
from sae_lens.toy_model.plotting import (
    cosine_similarities,
    find_best_feature_ordering,
)


def test_cosine_similarities_identity_for_same_matrix():
    mat = torch.randn(5, 10)
    mat = mat / mat.norm(dim=1, keepdim=True)

    cos_sims = cosine_similarities(mat, mat)

    # Diagonal should be 1
    torch.testing.assert_close(torch.diag(cos_sims), torch.ones(5), atol=1e-5, rtol=0)


def test_cosine_similarities_orthogonal_vectors():
    mat1 = torch.eye(3)
    mat2 = torch.eye(3)

    cos_sims = cosine_similarities(mat1, mat2)

    expected = torch.eye(3)
    torch.testing.assert_close(cos_sims, expected, atol=1e-5, rtol=0)


def test_cosine_similarities_shape():
    mat1 = torch.randn(4, 10)
    mat2 = torch.randn(6, 10)

    cos_sims = cosine_similarities(mat1, mat2)

    assert cos_sims.shape == (4, 6)


def test_cosine_similarities_range():
    mat1 = torch.randn(5, 10)
    mat2 = torch.randn(7, 10)

    cos_sims = cosine_similarities(mat1, mat2)

    assert torch.all(cos_sims >= -1.0 - 1e-5)
    assert torch.all(cos_sims <= 1.0 + 1e-5)


def test_find_best_feature_ordering_returns_permutation():
    sae_features = torch.randn(5, 10)
    true_features = torch.randn(5, 10)

    ordering = find_best_feature_ordering(sae_features, true_features)

    assert ordering.shape == (5,)
    assert set(ordering.tolist()) == {0, 1, 2, 3, 4}


def test_find_best_feature_ordering_aligns_identical_features():
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


def test_find_best_feature_ordering_with_feature_dict():
    feature_dict = FeatureDictionary(num_features=5, hidden_dim=8, ortho_num_steps=100)
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
