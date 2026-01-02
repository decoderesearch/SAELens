"""
Toy model utilities for synthetic SAE experiments.

This module provides tools for creating feature dictionaries and generating
synthetic activations for testing and experimenting with SAEs.

Main components:
- FeatureDictionary: Maps sparse feature activations to dense hidden activations
- generate_activations: Generates batches of synthetic feature activations
- TreeFeatureGenerator: Generates hierarchical feature activations
- Training utilities: Helpers for training and evaluating SAEs on synthetic data
- Plotting utilities: Visualization helpers for understanding SAE behavior

Example:
    >>> import torch
    >>> from sae_lens.toy_model import (
    ...     FeatureDictionary,
    ...     generate_activations,
    ...     create_correlation_matrix,
    ...     SyntheticActivationIterator,
    ...     plot_sae_feature_similarity,
    ... )
    >>>
    >>> # Create a feature dictionary with 100 features in 64-dim space
    >>> feature_dict = FeatureDictionary(num_features=100, hidden_dim=64)
    >>>
    >>> # Generate sparse feature activations
    >>> probs = torch.ones(100) * 0.1  # 10% firing probability
    >>> feature_acts = generate_activations(batch_size=1000, firing_probabilities=probs)
    >>>
    >>> # Convert to hidden activations
    >>> hidden_acts = feature_dict(feature_acts)
    >>>
    >>> # For training, create an iterator
    >>> def gen_fn(batch_size):
    ...     return generate_activations(batch_size, probs)
    >>> iterator = SyntheticActivationIterator(feature_dict, gen_fn, batch_size=1024)
"""

from sae_lens.toy_model.activation_generator import (
    absorb_features,
    chain_modifiers,
    create_block_correlation_matrix,
    create_correlation_matrix,
    generate_activations,
    suppress_features,
)
from sae_lens.toy_model.feature_dictionary import (
    FeatureDictionary,
    orthogonalize_vectors,
)
from sae_lens.toy_model.plotting import (
    cosine_similarities,
    find_best_feature_ordering,
    find_best_feature_ordering_across_saes,
    find_best_feature_ordering_from_sae,
    plot_decoder_bias_similarity,
    plot_latent_similarities,
    plot_sae_feature_similarity,
)
from sae_lens.toy_model.training_utils import (
    SyntheticActivationIterator,
    SyntheticEvalResult,
    eval_sae_on_synthetic,
    init_sae_from_feature_dict,
    train_sae_on_synthetic,
)
from sae_lens.toy_model.tree_feature_generator import TreeFeatureGenerator

__all__ = [
    # Main classes
    "FeatureDictionary",
    "TreeFeatureGenerator",
    # Activation generation
    "generate_activations",
    "create_correlation_matrix",
    "create_block_correlation_matrix",
    # Feature modifiers
    "suppress_features",
    "absorb_features",
    "chain_modifiers",
    # Utilities
    "orthogonalize_vectors",
    # Training utilities
    "SyntheticActivationIterator",
    "SyntheticEvalResult",
    "train_sae_on_synthetic",
    "eval_sae_on_synthetic",
    "init_sae_from_feature_dict",
    # Plotting utilities
    "cosine_similarities",
    "find_best_feature_ordering",
    "find_best_feature_ordering_from_sae",
    "find_best_feature_ordering_across_saes",
    "plot_sae_feature_similarity",
    "plot_decoder_bias_similarity",
    "plot_latent_similarities",
]
