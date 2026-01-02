import pytest
import torch

from sae_lens.synthetic import (
    absorb_features,
    chain_modifiers,
    create_block_correlation_matrix,
    create_correlation_matrix,
    generate_activations,
    suppress_features,
)


def test_generate_activations_respects_firing_probabilities():
    firing_probs = torch.tensor([0.3, 0.2, 0.1])
    batch_size = 2000
    activations = generate_activations(batch_size, firing_probs)

    actual_probs = (activations > 0).float().mean(dim=0)
    torch.testing.assert_close(actual_probs, firing_probs, atol=0.05, rtol=0)


def test_generate_activations_respects_std_magnitudes():
    firing_probs = torch.tensor([1.0, 1.0, 1.0])
    std_magnitudes = torch.tensor([0.1, 0.2, 0.3])
    batch_size = 2000
    activations = generate_activations(batch_size, firing_probs, std_magnitudes)

    actual_stds = activations.std(dim=0)
    torch.testing.assert_close(actual_stds, std_magnitudes, atol=0.05, rtol=0)


def test_generate_activations_respects_mean_magnitudes():
    firing_probs = torch.tensor([0.5, 0.5, 1.0])
    mean_magnitudes = torch.tensor([1.5, 2.5, 3.5])
    batch_size = 2000
    activations = generate_activations(
        batch_size, firing_probs, mean_firing_magnitudes=mean_magnitudes
    )

    assert set(activations[:, 0].tolist()) == {0, 1.5}
    assert set(activations[:, 1].tolist()) == {0, 2.5}
    assert set(activations[:, 2].tolist()) == {3.5}


def test_generate_activations_never_returns_negative():
    firing_probs = torch.tensor([1.0, 1.0, 1.0])
    std_magnitudes = torch.tensor([0.5, 1.0, 2.0])
    batch_size = 2000
    activations = generate_activations(batch_size, firing_probs, std_magnitudes)

    assert torch.all(activations >= 0)


def test_generate_activations_with_correlation_matrix():
    batch_size = 2000
    firing_probs = torch.tensor([0.2, 0.4, 0.3])
    correlation_matrix = create_correlation_matrix(
        num_features=3, correlations={(0, 1): 0.7, (0, 2): -0.3}
    )

    activations = generate_activations(
        batch_size=batch_size,
        firing_probabilities=firing_probs,
        correlation_matrix=correlation_matrix,
    )

    assert activations.shape == (batch_size, 3)
    assert torch.all(activations >= 0)

    # Check marginal probabilities
    firing_rates = (activations > 0).float().mean(dim=0)
    torch.testing.assert_close(firing_rates, firing_probs, atol=0.05, rtol=0.1)

    # Check correlations
    binary_features = (activations > 0).float()
    correlations = torch.corrcoef(binary_features.T)
    assert correlations[0, 1] > 0.3
    assert correlations[0, 2] < 0


def test_create_correlation_matrix_identity_by_default():
    matrix = create_correlation_matrix(num_features=3)
    expected = torch.eye(3)
    torch.testing.assert_close(matrix, expected)


def test_create_correlation_matrix_with_correlations():
    matrix = create_correlation_matrix(
        num_features=4, correlations={(0, 1): 0.8, (1, 2): -0.6}
    )

    torch.testing.assert_close(matrix, matrix.T)
    torch.testing.assert_close(torch.diag(matrix), torch.ones(4))
    assert abs(matrix[0, 1] - 0.8) < 0.1
    assert abs(matrix[1, 2] - (-0.6)) < 0.1


def test_create_correlation_matrix_with_default_correlation():
    matrix = create_correlation_matrix(num_features=3, default_correlation=0.2)
    expected = torch.tensor([[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0]])
    torch.testing.assert_close(matrix, expected)


def test_create_block_correlation_matrix_structure():
    matrix = create_block_correlation_matrix(
        block_sizes=[2, 2],
        within_block_correlation=0.8,
        between_block_correlation=0.1,
        seed=42,
    )

    assert matrix.shape == (4, 4)
    torch.testing.assert_close(matrix, matrix.T)
    torch.testing.assert_close(torch.diag(matrix), torch.ones(4))

    # Within block 1 (features 0,1)
    assert abs(matrix[0, 1] - 0.8) < 0.15
    # Within block 2 (features 2,3)
    assert abs(matrix[2, 3] - 0.8) < 0.15
    # Between blocks
    assert abs(matrix[0, 2] - 0.1) < 0.15


def test_suppress_features_suppresses_when_dominant_fires():
    feats = torch.zeros(5, 4)
    feats[0, 0] = 1  # Dominant fires
    feats[0, 1] = 1  # Should be suppressed
    feats[0, 2] = 1  # Should be suppressed
    feats[1, 1] = 1  # Should not be suppressed (dominant not firing)

    modifier = suppress_features(dominant_feature=0, suppressed_features=[1, 2])
    modified = modifier(feats)

    assert modified[0, 0] == 1
    assert modified[0, 1] == 0
    assert modified[0, 2] == 0
    assert modified[1, 1] == 1


def test_absorb_features_sets_parent_when_child_fires():
    feats = torch.zeros(5, 4)
    feats[0, 2] = 1  # Child fires
    feats[1, 3] = 1  # Another child fires
    feats[2, 1] = 1  # Not a child

    modifier = absorb_features([(0, 2), (1, 3)])
    modified = modifier(feats)

    assert modified[0, 0] == 1  # Parent absorbed
    assert modified[0, 2] == 1  # Child still fires
    assert modified[1, 1] == 1  # Parent absorbed
    assert modified[1, 3] == 1  # Child still fires
    assert modified[2, 0] == 0  # No absorption
    assert modified[2, 1] == 1  # Original value


def test_chain_modifiers_applies_in_order():
    feats = torch.zeros(3, 4)
    feats[0, 0] = 1  # Dominant
    feats[0, 2] = 1  # Child that causes absorption into 1
    feats[0, 1] = 1  # Will be suppressed

    # Absorption first: feature 2 causes feature 1 to fire
    # Suppression second: feature 0 suppresses feature 1
    modifier = chain_modifiers(
        [
            absorb_features([(1, 2)]),
            suppress_features(0, [1]),
        ]
    )
    modified = modifier(feats)

    assert modified[0, 0] == 1
    assert modified[0, 1] == 0  # Suppressed after absorption
    assert modified[0, 2] == 1


@pytest.mark.parametrize(
    "num_features,correlations",
    [
        (3, {}),
        (4, {(0, 1): 0.5}),
        (5, {(0, 1): 0.8, (2, 3): -0.3}),
    ],
)
def test_create_correlation_matrix_is_valid(
    num_features: int, correlations: dict[tuple[int, int], float]
):
    matrix = create_correlation_matrix(num_features, correlations)

    # Check symmetry
    torch.testing.assert_close(matrix, matrix.T)
    # Check diagonal
    torch.testing.assert_close(torch.diag(matrix), torch.ones(num_features))
    # Check positive semi-definiteness
    eigenvals = torch.linalg.eigvals(matrix)
    assert torch.all(eigenvals.real >= -1e-6)
