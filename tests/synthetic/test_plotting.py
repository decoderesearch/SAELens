import torch

from sae_lens.synthetic import FeatureDictionary
from sae_lens.synthetic.plotting import find_best_feature_ordering
from sae_lens.util import cosine_similarities


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
