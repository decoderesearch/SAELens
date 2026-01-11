"""
Synthetic data utilities for SAE experiments.

This module provides tools for creating feature dictionaries and generating
synthetic activations for testing and experimenting with SAEs.

Main components:

- FeatureDictionary: Maps sparse feature activations to dense hidden activations
- ActivationGenerator: Generates batches of synthetic feature activations
- HierarchyNode: Enforces hierarchical structure on feature activations
- Training utilities: Helpers for training and evaluating SAEs on synthetic data
- Plotting utilities: Visualization helpers for understanding SAE behavior
"""

from sae_lens.synthetic.activation_generator import (
    ActivationGenerator,
    ActivationsModifier,
    ActivationsModifierInput,
    CorrelationMatrixInput,
)
from sae_lens.synthetic.correlation import (
    LowRankCorrelationMatrix,
    create_correlation_matrix_from_correlations,
    generate_random_correlation_matrix,
    generate_random_correlations,
    generate_random_low_rank_correlation_matrix,
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
)
from sae_lens.synthetic.firing_probabilities import (
    linear_firing_probabilities,
    random_firing_probabilities,
    zipfian_firing_probabilities,
)
from sae_lens.synthetic.hierarchy import HierarchyNode, hierarchy_modifier
from sae_lens.synthetic.initialization import init_sae_to_match_feature_dict
from sae_lens.synthetic.plotting import (
    find_best_feature_ordering,
    find_best_feature_ordering_across_saes,
    find_best_feature_ordering_from_sae,
    plot_sae_feature_similarity,
)
from sae_lens.synthetic.training import (
    SyntheticActivationIterator,
    train_toy_sae,
)
from sae_lens.util import cosine_similarities

__all__ = [
    # Main classes
    "FeatureDictionary",
    "HierarchyNode",
    "hierarchy_modifier",
    "ActivationGenerator",
    # Activation generation
    "zipfian_firing_probabilities",
    "linear_firing_probabilities",
    "random_firing_probabilities",
    "create_correlation_matrix_from_correlations",
    "generate_random_correlations",
    "generate_random_correlation_matrix",
    "generate_random_low_rank_correlation_matrix",
    "LowRankCorrelationMatrix",
    "CorrelationMatrixInput",
    # Feature modifiers
    "ActivationsModifier",
    "ActivationsModifierInput",
    # Utilities
    "orthogonalize_embeddings",
    "orthogonal_initializer",
    "FeatureDictionaryInitializer",
    "cosine_similarities",
    # Training utilities
    "SyntheticActivationIterator",
    "SyntheticDataEvalResult",
    "train_toy_sae",
    "eval_sae_on_synthetic_data",
    "mean_correlation_coefficient",
    "init_sae_to_match_feature_dict",
    # Plotting utilities
    "find_best_feature_ordering",
    "find_best_feature_ordering_from_sae",
    "find_best_feature_ordering_across_saes",
    "plot_sae_feature_similarity",
]
