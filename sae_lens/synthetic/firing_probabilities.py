"""
Helper functions for generating firing probability distributions.
"""

import torch


def zipfian_firing_probabilities(
    num_features: int,
    exponent: float = 1.0,
    max_prob: float = 0.3,
    min_prob: float = 0.01,
) -> torch.Tensor:
    """
    Generate firing probabilities following a Zipfian (power-law) distribution.

    Creates probabilities where a few features fire frequently and most fire rarely,
    which mirrors the distribution often observed in real neural network features.

    Args:
        num_features: Number of features to generate probabilities for
        exponent: Zipf exponent (higher = steeper dropoff). Default 1.0.
        max_prob: Maximum firing probability (for the most frequent feature)
        min_prob: Minimum firing probability (for the least frequent feature)

    Returns:
        Tensor of shape [num_features] with firing probabilities in descending order
    """
    if num_features < 1:
        raise ValueError("num_features must be at least 1")
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    if not 0 < min_prob < max_prob <= 1:
        raise ValueError("Must have 0 < min_prob < max_prob <= 1")

    ranks = torch.arange(1, num_features + 1, dtype=torch.float32)
    probs = 1.0 / ranks**exponent

    # Scale to [min_prob, max_prob]
    if num_features == 1:
        return torch.tensor([max_prob])

    probs_min, probs_max = probs.min(), probs.max()
    return min_prob + (max_prob - min_prob) * (probs - probs_min) / (
        probs_max - probs_min
    )


def linear_firing_probabilities(
    num_features: int,
    max_prob: float = 0.3,
    min_prob: float = 0.01,
) -> torch.Tensor:
    """
    Generate firing probabilities that decay linearly from max to min.

    Args:
        num_features: Number of features to generate probabilities for
        max_prob: Firing probability for the first feature
        min_prob: Firing probability for the last feature

    Returns:
        Tensor of shape [num_features] with linearly decaying probabilities
    """
    if num_features < 1:
        raise ValueError("num_features must be at least 1")
    if not 0 < min_prob <= max_prob <= 1:
        raise ValueError("Must have 0 < min_prob <= max_prob <= 1")

    if num_features == 1:
        return torch.tensor([max_prob])

    return torch.linspace(max_prob, min_prob, num_features)


def random_firing_probabilities(
    num_features: int,
    max_prob: float = 0.5,
    min_prob: float = 0.01,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Generate random firing probabilities uniformly sampled from a range.

    Args:
        num_features: Number of features to generate probabilities for
        max_prob: Maximum firing probability
        min_prob: Minimum firing probability
        seed: Optional random seed for reproducibility

    Returns:
        Tensor of shape [num_features] with random firing probabilities
    """
    if num_features < 1:
        raise ValueError("num_features must be at least 1")
    if not 0 < min_prob < max_prob <= 1:
        raise ValueError("Must have 0 < min_prob < max_prob <= 1")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    probs = torch.rand(num_features, generator=generator, dtype=torch.float32)
    return min_prob + (max_prob - min_prob) * probs
