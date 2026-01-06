"""
Synthetic data utilities for SAE experiments.

This module provides tools for creating feature dictionaries and generating
synthetic activations for testing and experimenting with SAEs.

Main components:
- FeatureDictionary: Maps sparse feature activations to dense hidden activations
- generate_activations: Generates batches of synthetic feature activations
- HierarchyNode: Enforces hierarchical structure on feature activations
- Training utilities: Helpers for training and evaluating SAEs on synthetic data
- Plotting utilities: Visualization helpers for understanding SAE behavior

Example:
    >>> import torch
    >>> from sae_lens.synthetic import (
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

from sae_lens.synthetic.activation_generator import (
    ActivationGenerator,
    ActivationsModifier,
    ActivationsModifierInput,
    absorb_features,
    chain_modifiers,
    generate_activations,
    suppress_features,
)
from sae_lens.synthetic.correlation import (
    create_block_correlation_matrix,
    create_correlation_matrix,
    create_correlation_matrix_from_correlations,
    generate_random_correlation_matrix,
    generate_random_correlations,
)
from sae_lens.synthetic.evals import (
    SyntheticDataEvalResult,
    eval_sae_on_synthetic_data,
    mean_correlation_coefficient,
)
from sae_lens.synthetic.feature_dictionary import (
    FeatureDictionary,
    FeatureDictionaryInitializer,
    orthogonal_initializer,
    orthogonalize_embeddings,
    orthogonalize_vectors,
)
from sae_lens.synthetic.firing_probabilities import (
    linear_firing_probabilities,
    random_firing_probabilities,
    zipfian_firing_probabilities,
)
from sae_lens.synthetic.hierarchy import HierarchyNode
from sae_lens.synthetic.initialization import init_sae_to_match_feature_dict
from sae_lens.synthetic.plotting import (
    find_best_feature_ordering,
    find_best_feature_ordering_across_saes,
    find_best_feature_ordering_from_sae,
    plot_sae_feature_similarity,
)
from sae_lens.synthetic.train_sae_on_synthetic_data import (
    SyntheticActivationIterator,
    train_sae_on_synthetic_data,
)
from sae_lens.util import cosine_similarities

__all__ = [
    # Main classes
    "FeatureDictionary",
    "HierarchyNode",
    "ActivationGenerator",
    # Activation generation
    "generate_activations",
    "zipfian_firing_probabilities",
    "linear_firing_probabilities",
    "random_firing_probabilities",
    "create_correlation_matrix",
    "create_correlation_matrix_from_correlations",
    "create_block_correlation_matrix",
    "generate_random_correlations",
    "generate_random_correlation_matrix",
    # Feature modifiers
    "suppress_features",
    "absorb_features",
    "chain_modifiers",
    "ActivationsModifier",
    "ActivationsModifierInput",
    # Utilities
    "orthogonalize_vectors",
    "orthogonalize_embeddings",
    "orthogonal_initializer",
    "FeatureDictionaryInitializer",
    "cosine_similarities",
    # Training utilities
    "SyntheticActivationIterator",
    "SyntheticDataEvalResult",
    "train_sae_on_synthetic_data",
    "eval_sae_on_synthetic_data",
    "mean_correlation_coefficient",
    "init_sae_to_match_feature_dict",
    # Plotting utilities
    "find_best_feature_ordering",
    "find_best_feature_ordering_from_sae",
    "find_best_feature_ordering_across_saes",
    "plot_sae_feature_similarity",
]
