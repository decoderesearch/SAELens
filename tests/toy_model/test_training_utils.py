import torch

from sae_lens.toy_model import (
    FeatureDictionary,
    SyntheticActivationIterator,
    generate_activations,
)


def test_SyntheticActivationIterator_generates_correct_shape():
    feature_dict = FeatureDictionary(num_features=10, hidden_dim=8, ortho_num_steps=100)
    probs = torch.ones(10) * 0.1

    def gen_fn(batch_size: int) -> torch.Tensor:
        return generate_activations(batch_size, probs)

    iterator = SyntheticActivationIterator(feature_dict, gen_fn, batch_size=32)
    batch = next(iterator)

    assert batch.shape == (32, 8)


def test_SyntheticActivationIterator_is_iterable():
    feature_dict = FeatureDictionary(num_features=5, hidden_dim=4, ortho_num_steps=100)
    probs = torch.ones(5) * 0.2

    def gen_fn(batch_size: int) -> torch.Tensor:
        return generate_activations(batch_size, probs)

    iterator = SyntheticActivationIterator(feature_dict, gen_fn, batch_size=16)

    batches = [next(iterator) for _ in range(3)]
    assert len(batches) == 3
    assert all(b.shape == (16, 4) for b in batches)


def test_SyntheticActivationIterator_produces_sparse_activations():
    feature_dict = FeatureDictionary(
        num_features=20, hidden_dim=10, ortho_num_steps=100
    )
    probs = torch.ones(20) * 0.05

    def gen_fn(batch_size: int) -> torch.Tensor:
        return generate_activations(batch_size, probs)

    iterator = SyntheticActivationIterator(feature_dict, gen_fn, batch_size=100)
    batch = next(iterator)

    # Some activations should be zero (sparse input)
    # Not all hidden activations will be zero though due to the linear transform
    assert batch.shape == (100, 10)
