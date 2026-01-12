import pytest
import torch

from sae_lens.synthetic import (
    FeatureDictionary,
    orthogonal_initializer,
    orthogonalize_embeddings,
)
from tests.helpers import to_sparse


def test_FeatureDictionary_creates_correct_shape():
    feature_dict = FeatureDictionary(num_features=10, hidden_dim=8)
    assert feature_dict.feature_vectors.shape == (10, 8)
    assert feature_dict.bias.shape == (8,)


def test_FeatureDictionary_creates_orthogonal_features():
    feature_dict = FeatureDictionary(num_features=8, hidden_dim=8)
    features = feature_dict.feature_vectors
    dot_products = features @ features.T
    expected = torch.eye(8)
    assert torch.allclose(dot_products, expected, atol=1e-4)


def test_FeatureDictionary_creates_unit_norm_features():
    feature_dict = FeatureDictionary(num_features=10, hidden_dim=8)
    features = feature_dict.feature_vectors
    norms = torch.norm(features, dim=1)
    assert torch.allclose(norms, torch.ones(10), atol=1e-5)


@pytest.mark.parametrize("use_sparse_tensors", [False, True])
def test_FeatureDictionary_forward_pass(use_sparse_tensors: bool):
    feature_dict = FeatureDictionary(num_features=5, hidden_dim=4)
    feature_activations = torch.zeros(3, 5)
    feature_activations[0, 0] = 1.0
    feature_activations[1, 2] = 0.5
    feature_activations[2, [0, 3]] = torch.tensor([1.0, 0.8])

    if use_sparse_tensors:
        feature_activations = to_sparse(feature_activations)

    hidden = feature_dict(feature_activations)
    assert hidden.shape == (3, 4)
    assert not hidden.is_sparse


@pytest.mark.parametrize("use_sparse_tensors", [False, True])
def test_FeatureDictionary_forward_produces_linear_combination(
    use_sparse_tensors: bool,
):
    feature_dict = FeatureDictionary(num_features=3, hidden_dim=4)
    features = feature_dict.feature_vectors

    activations = torch.tensor([[1.0, 0.0, 0.0]])
    if use_sparse_tensors:
        activations = to_sparse(activations)
    hidden = feature_dict(activations)
    expected = features[0].unsqueeze(0)
    assert torch.allclose(hidden, expected, atol=1e-5)

    activations = torch.tensor([[1.0, 1.0, 0.0]])
    if use_sparse_tensors:
        activations = to_sparse(activations)
    hidden = feature_dict(activations)
    expected = (features[0] + features[1]).unsqueeze(0)
    assert torch.allclose(hidden, expected, atol=1e-5)


@pytest.mark.parametrize("chunk_size", [1024, 3])
def test_FeatureDictionary_overcomplete_minimizes_cosine_similarity(chunk_size: int):
    """Test that overcomplete dictionaries minimize off-diagonal cosine similarities."""
    num_features = 16
    hidden_dim = 14

    feature_dict = FeatureDictionary(
        num_features=num_features,
        hidden_dim=hidden_dim,
        initializer=orthogonal_initializer(chunk_size=chunk_size),
    )
    features = feature_dict.feature_vectors
    dot_products = features @ features.T

    # Extract off-diagonal elements
    off_diagonal_mask = ~torch.eye(num_features, dtype=torch.bool)
    off_diagonal = dot_products[off_diagonal_mask]

    # Off-diagonal cosine similarities should be small (well below 1)
    max_off_diag = off_diagonal.abs().max().item()
    mean_off_diag = off_diagonal.abs().mean().item()

    # Compare against random unit vectors (no optimization)
    random_vecs = torch.randn(num_features, hidden_dim)
    random_vecs = random_vecs / random_vecs.norm(dim=1, keepdim=True)
    random_dots = random_vecs @ random_vecs.T
    random_off_diag = random_dots[off_diagonal_mask]
    random_mean = random_off_diag.abs().mean().item()

    # Optimized vectors should have lower mean cosine similarity than random
    assert mean_off_diag < random_mean, (
        f"Optimized mean {mean_off_diag:.4f} should be smaller than "
        f"random mean {random_mean:.4f}"
    )

    # Max absolute cosine similarity should be reasonable for 2x overcomplete
    assert (
        max_off_diag < 0.15
    ), f"Max off-diagonal cosine sim {max_off_diag:.4f} too high"


def test_orthogonalize_embeddings_identical_results_across_chunk_sizes():
    """Test that different chunk sizes produce identical results."""
    embeddings = torch.randn(16, 8)
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

    # Run with different chunk sizes using same initial embeddings
    result_large_chunk = orthogonalize_embeddings(
        embeddings.clone(), num_steps=10, chunk_size=1024
    )

    result_small_chunk = orthogonalize_embeddings(
        embeddings.clone(), num_steps=10, chunk_size=3
    )

    result_tiny_chunk = orthogonalize_embeddings(
        embeddings.clone(), num_steps=10, chunk_size=1
    )

    # Results should be identical regardless of chunk size
    assert torch.allclose(result_large_chunk, result_small_chunk, atol=1e-6), (
        f"Large vs small chunk max diff: "
        f"{(result_large_chunk - result_small_chunk).abs().max().item()}"
    )
    assert torch.allclose(result_large_chunk, result_tiny_chunk, atol=1e-6), (
        f"Large vs tiny chunk max diff: "
        f"{(result_large_chunk - result_tiny_chunk).abs().max().item()}"
    )


def test_FeatureDictionary_sparse_and_dense_produce_identical_outputs():
    feature_dict = FeatureDictionary(num_features=10, hidden_dim=8)

    # Create a sparse activation pattern
    dense_activations = torch.zeros(5, 10)
    dense_activations[0, 0] = 1.0
    dense_activations[1, [2, 5]] = torch.tensor([0.5, 0.3])
    dense_activations[2, 7] = 2.0
    dense_activations[3, [1, 3, 9]] = torch.tensor([0.1, 0.2, 0.8])
    # Row 4 is all zeros

    sparse_activations = to_sparse(dense_activations)

    dense_output = feature_dict(dense_activations)
    sparse_output = feature_dict(sparse_activations)

    assert not sparse_output.is_sparse
    assert torch.allclose(dense_output, sparse_output, atol=1e-6)


def test_FeatureDictionary_sparse_with_bias():
    feature_dict = FeatureDictionary(num_features=5, hidden_dim=4, bias=True)
    # Manually set bias for testing
    feature_dict.bias.data = torch.tensor([0.1, 0.2, 0.3, 0.4])

    activations = torch.zeros(2, 5)
    activations[0, 0] = 1.0
    activations[1, [1, 2]] = torch.tensor([0.5, 0.5])

    dense_output = feature_dict(activations)
    sparse_output = feature_dict(to_sparse(activations))

    assert torch.allclose(dense_output, sparse_output, atol=1e-6)

    # Verify bias is applied
    zero_activation = torch.zeros(1, 5)
    output_with_zero = feature_dict(to_sparse(zero_activation))
    assert torch.allclose(output_with_zero, feature_dict.bias.unsqueeze(0), atol=1e-6)
