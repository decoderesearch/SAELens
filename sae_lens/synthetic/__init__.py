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
    ClassificationMetrics,
    ClassificationMetricsCalculator,
    DeadLatentsCalculator,
    L0Calculator,
    ShrinkageCalculator,
    SyntheticDataEvalResult,
    compute_classification_metrics,
    eval_sae_on_synthetic_data,
    feature_uniqueness,
    mean_correlation_coefficient,
)
from sae_lens.synthetic.feature_dictionary import (
    FeatureDictionary,
    FeatureDictionaryInitializer,
    orthogonal_initializer,
    orthogonalize_embeddings,
)
from sae_lens.synthetic.firing_magnitudes import (
    ConstantMagnitudeConfig,
    ExponentialMagnitudeConfig,
    FoldedNormalMagnitudeConfig,
    LinearMagnitudeConfig,
    MagnitudeConfig,
    generate_magnitudes,
    get_magnitude_class,
    register_magnitude,
)
from sae_lens.synthetic.firing_probabilities import (
    ConstantFiringProbabilityConfig,
    FiringProbabilityConfig,
    LinearFiringProbabilityConfig,
    RandomFiringProbabilityConfig,
    ZipfianFiringProbabilityConfig,
    get_firing_probability_class,
    linear_firing_probabilities,
    random_firing_probabilities,
    register_firing_probability,
    zipfian_firing_probabilities,
)
from sae_lens.synthetic.hierarchy import (
    Hierarchy,
    HierarchyConfig,
    HierarchyNode,
    generate_hierarchy,
    hierarchy_modifier,
)
from sae_lens.synthetic.initialization import init_sae_to_match_feature_dict
from sae_lens.synthetic.plotting import (
    find_best_feature_ordering,
    find_best_feature_ordering_across_saes,
    find_best_feature_ordering_from_sae,
    plot_sae_feature_similarity,
)
from sae_lens.synthetic.stats import (
    CorrelationMatrixStats,
    SuperpositionStats,
    compute_correlation_matrix_stats,
    compute_low_rank_correlation_matrix_stats,
    compute_superposition_stats,
)
from sae_lens.synthetic.synthetic_model import (
    SYNTHETIC_MODEL_CONFIG_FILENAME,
    SYNTHETIC_MODEL_HIERARCHY_FILENAME,
    SYNTHETIC_MODEL_WEIGHTS_FILENAME,
    LowRankCorrelationConfig,
    OrthogonalizationConfig,
    SyntheticModel,
    SyntheticModelConfig,
)
from sae_lens.synthetic.synthetic_sae_runner import (
    RUNNER_CONFIG_FILENAME,
    SyntheticSAERunner,
    SyntheticSAERunnerConfig,
    SyntheticSAERunnerResult,
)
from sae_lens.synthetic.training import (
    SyntheticActivationIterator,
    train_toy_sae,
)
from sae_lens.synthetic.upload_synthetic_model import (
    upload_synthetic_model_to_huggingface,
)
from sae_lens.util import cosine_similarities

__all__ = [
    # Main classes
    "FeatureDictionary",
    "HierarchyNode",
    "hierarchy_modifier",
    "ActivationGenerator",
    # SyntheticModel
    "SyntheticModel",
    "SyntheticModelConfig",
    "OrthogonalizationConfig",
    "LowRankCorrelationConfig",
    "SYNTHETIC_MODEL_CONFIG_FILENAME",
    "SYNTHETIC_MODEL_WEIGHTS_FILENAME",
    "SYNTHETIC_MODEL_HIERARCHY_FILENAME",
    # Firing probability
    "FiringProbabilityConfig",
    "ZipfianFiringProbabilityConfig",
    "LinearFiringProbabilityConfig",
    "RandomFiringProbabilityConfig",
    "ConstantFiringProbabilityConfig",
    "register_firing_probability",
    "get_firing_probability_class",
    # Magnitude
    "MagnitudeConfig",
    "ConstantMagnitudeConfig",
    "LinearMagnitudeConfig",
    "ExponentialMagnitudeConfig",
    "FoldedNormalMagnitudeConfig",
    "register_magnitude",
    "get_magnitude_class",
    "generate_magnitudes",
    # Hierarchy generation
    "HierarchyConfig",
    "Hierarchy",
    "generate_hierarchy",
    # Runner
    "SyntheticSAERunner",
    "SyntheticSAERunnerConfig",
    "SyntheticSAERunnerResult",
    "RUNNER_CONFIG_FILENAME",
    # Activation generation (legacy functions)
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
    # Statistics
    "compute_correlation_matrix_stats",
    "compute_low_rank_correlation_matrix_stats",
    "compute_superposition_stats",
    "CorrelationMatrixStats",
    "SuperpositionStats",
    # Training utilities
    "SyntheticActivationIterator",
    "SyntheticDataEvalResult",
    "ClassificationMetrics",
    "ClassificationMetricsCalculator",
    "L0Calculator",
    "DeadLatentsCalculator",
    "ShrinkageCalculator",
    "train_toy_sae",
    "eval_sae_on_synthetic_data",
    "compute_classification_metrics",
    "mean_correlation_coefficient",
    "feature_uniqueness",
    "init_sae_to_match_feature_dict",
    # Plotting utilities
    "find_best_feature_ordering",
    "find_best_feature_ordering_from_sae",
    "find_best_feature_ordering_across_saes",
    "plot_sae_feature_similarity",
    # HuggingFace utilities
    "upload_synthetic_model_to_huggingface",
]
