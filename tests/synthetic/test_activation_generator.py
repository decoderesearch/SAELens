import torch

from sae_lens.synthetic import (
    generate_activations,
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


def test_generate_activations_with_empty_list_of_modifiers():
    """Test that empty list of modifiers works."""
    from sae_lens.synthetic import ActivationGenerator

    generator = ActivationGenerator(
        num_features=3,
        firing_probabilities=0.5,
        modify_activations=[],
    )

    assert generator.modify_activations is None
    samples = generator.sample(batch_size=10)
    assert samples.shape == (10, 3)
