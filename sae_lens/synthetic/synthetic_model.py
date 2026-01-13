"""
SyntheticModel class for large-scale SAE training on synthetic data.

This module provides SyntheticModel, which encapsulates ActivationGenerator
and FeatureDictionary with configuration, hierarchy, correlation, and persistence.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from sae_lens.synthetic.activation_generator import ActivationGenerator
from sae_lens.synthetic.correlation import (
    LowRankCorrelationMatrix,
    generate_random_low_rank_correlation_matrix,
)
from sae_lens.synthetic.feature_dictionary import (
    FeatureDictionary,
    orthogonal_initializer,
)
from sae_lens.synthetic.firing_magnitudes import (
    MagnitudeConfig,
    generate_magnitudes,
)
from sae_lens.synthetic.firing_probabilities import (
    FiringProbabilityConfig,
    ZipfianFiringProbabilityConfig,
    get_firing_probability_class,
)
from sae_lens.synthetic.hierarchy import (
    Hierarchy,
    HierarchyConfig,
    generate_hierarchy,
)
from sae_lens.util import str_to_dtype


@dataclass
class OrthogonalizationConfig:
    """
    Configuration for feature dictionary orthogonalization.

    Attributes:
        num_steps: Number of optimization steps for orthogonalization.
        lr: Learning rate for orthogonalization optimization.
        chunk_size: Chunk size for memory-efficient orthogonalization.
    """

    num_steps: int = 200
    lr: float = 0.01
    chunk_size: int = 1024

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "num_steps": self.num_steps,
            "lr": self.lr,
            "chunk_size": self.chunk_size,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OrthogonalizationConfig":
        """Deserialize config from dictionary."""
        return cls(**d)


@dataclass
class LowRankCorrelationConfig:
    """
    Configuration for feature correlation structure.

    Uses low-rank correlation matrices for memory efficiency with large feature counts.

    Attributes:
        rank: Rank of the low-rank correlation matrix.
        correlation_scale: Scale of correlations (higher = stronger correlations).
        seed: Random seed for reproducibility.
    """

    rank: int = 32
    correlation_scale: float = 0.1
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "rank": self.rank,
            "correlation_scale": self.correlation_scale,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LowRankCorrelationConfig":
        """Deserialize config from dictionary."""
        return cls(**d)


def _deserialize_magnitude(value: Any) -> float | MagnitudeConfig:
    """Deserialize a magnitude value from dict or float."""
    if isinstance(value, dict):
        return MagnitudeConfig.from_dict(value)
    return float(value)


@dataclass
class SyntheticModelConfig:
    """
    Complete configuration for a SyntheticModel.

    This config encapsulates all settings needed to create a synthetic data
    generator for SAE training experiments.

    Attributes:
        num_features: Number of ground-truth features in the model.
        hidden_dim: Dimensionality of the hidden/activation space.
        firing_probability: Config for firing probability distribution.
        hierarchy: Config for automatic hierarchy generation.
        orthogonalization: Config for feature dictionary orthogonalization.
        correlation: Config for low-rank correlation structure.
        std_firing_magnitudes: Std dev of firing magnitudes (0 = deterministic).
            Can be a float for constant value, or MagnitudeConfig for
            per-feature values.
        mean_firing_magnitudes: Mean firing magnitude when active. Can be a float
            for constant value, or MagnitudeConfig for per-feature values.
        feature_dict_bias: Whether feature dictionary has a bias term.
        device: Device for tensors.
        dtype: Data type for tensors.
        use_sparse_tensors: Whether to use sparse COO tensors for activations.
        seed: Global random seed for reproducibility.
    """

    num_features: int
    hidden_dim: int
    firing_probability: FiringProbabilityConfig = field(
        default_factory=ZipfianFiringProbabilityConfig
    )
    hierarchy: HierarchyConfig | None = None
    orthogonalization: OrthogonalizationConfig | None = field(
        default_factory=OrthogonalizationConfig
    )
    correlation: LowRankCorrelationConfig | None = None
    std_firing_magnitudes: float | MagnitudeConfig = 0.0
    mean_firing_magnitudes: float | MagnitudeConfig = 1.0
    feature_dict_bias: bool = False
    device: str = "cpu"
    dtype: str = "float32"
    use_sparse_tensors: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.num_features < 1:
            raise ValueError("num_features must be at least 1")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "num_features": self.num_features,
            "hidden_dim": self.hidden_dim,
            "firing_probability": self.firing_probability.to_dict(),
            "hierarchy": (
                self.hierarchy.to_dict() if self.hierarchy is not None else None
            ),
            "orthogonalization": (
                self.orthogonalization.to_dict()
                if self.orthogonalization is not None
                else None
            ),
            "correlation": (
                self.correlation.to_dict() if self.correlation is not None else None
            ),
            "std_firing_magnitudes": (
                self.std_firing_magnitudes.to_dict()
                if isinstance(self.std_firing_magnitudes, MagnitudeConfig)
                else self.std_firing_magnitudes
            ),
            "mean_firing_magnitudes": (
                self.mean_firing_magnitudes.to_dict()
                if isinstance(self.mean_firing_magnitudes, MagnitudeConfig)
                else self.mean_firing_magnitudes
            ),
            "feature_dict_bias": self.feature_dict_bias,
            "device": self.device,
            "dtype": self.dtype,
            "use_sparse_tensors": self.use_sparse_tensors,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SyntheticModelConfig":
        """Deserialize config from dictionary."""
        ortho_dict = d.get("orthogonalization")
        orthogonalization = (
            OrthogonalizationConfig.from_dict(ortho_dict)
            if ortho_dict is not None
            else None
        )
        corr_dict = d.get("correlation")
        correlation = (
            LowRankCorrelationConfig.from_dict(corr_dict)
            if corr_dict is not None
            else None
        )
        hierarchy_dict = d.get("hierarchy")
        hierarchy = (
            HierarchyConfig.from_dict(hierarchy_dict)
            if hierarchy_dict is not None
            else None
        )
        return cls(
            num_features=d["num_features"],
            hidden_dim=d["hidden_dim"],
            firing_probability=FiringProbabilityConfig.from_dict(
                d["firing_probability"]
            ),
            hierarchy=hierarchy,
            orthogonalization=orthogonalization,
            correlation=correlation,
            std_firing_magnitudes=_deserialize_magnitude(
                d.get("std_firing_magnitudes", 0.0)
            ),
            mean_firing_magnitudes=_deserialize_magnitude(
                d.get("mean_firing_magnitudes", 1.0)
            ),
            feature_dict_bias=d.get("feature_dict_bias", False),
            device=d.get("device", "cpu"),
            dtype=d.get("dtype", "float32"),
            use_sparse_tensors=d.get("use_sparse_tensors", False),
            seed=d.get("seed"),
        )


# File names for persistence
SYNTHETIC_MODEL_CONFIG_FILENAME = "synthetic_model_config.json"
SYNTHETIC_MODEL_WEIGHTS_FILENAME = "synthetic_model.safetensors"
SYNTHETIC_MODEL_HIERARCHY_FILENAME = "hierarchy.json"


class SyntheticModel(nn.Module):
    """
    A complete synthetic data generator for SAE experiments.

    Encapsulates:

    - FeatureDictionary: Maps sparse features to dense activations
    - ActivationGenerator: Generates sparse feature activations
    - Hierarchy: Optional hierarchical structure on features
    - Correlation: Optional correlation structure between features

    Main method is `sample(batch_size)` which returns hidden activations
    ready for SAE training.
    """

    cfg: SyntheticModelConfig
    feature_dict: FeatureDictionary
    activation_generator: ActivationGenerator
    hierarchy: Hierarchy | None
    correlation_matrix: LowRankCorrelationMatrix | None

    def __init__(
        self,
        cfg: SyntheticModelConfig,
        feature_dict: FeatureDictionary | None = None,
        activation_generator: ActivationGenerator | None = None,
        hierarchy: Hierarchy | None = None,
        correlation_matrix: LowRankCorrelationMatrix | None = None,
    ):
        """
        Create a SyntheticModel.

        Typically, use `SyntheticModel.from_config(cfg)` to create a new model
        from configuration, or `SyntheticModel.load(path)` to load a saved model.

        Direct initialization is mainly for advanced use cases like loading
        with custom components.
        """
        super().__init__()
        self.cfg = cfg

        # Store correlation matrix (may be None)
        self.correlation_matrix = correlation_matrix

        # Store hierarchy (may be None)
        self.hierarchy = hierarchy

        # Feature dictionary
        if feature_dict is None:
            feature_dict = self._create_feature_dict()
        self.feature_dict = feature_dict

        # Activation generator
        if activation_generator is None:
            activation_generator = self._create_activation_generator()
        self.activation_generator = activation_generator

    @classmethod
    def from_config(cls, cfg: SyntheticModelConfig) -> "SyntheticModel":
        """
        Create a new SyntheticModel from configuration.

        This is the recommended way to create a new model. It:

        1. Generates hierarchy (if configured)
        2. Generates correlation matrix (if configured)
        3. Creates feature dictionary with orthogonalization
        4. Creates activation generator with all modifiers

        Args:
            cfg: Complete model configuration

        Returns:
            Fully initialized SyntheticModel
        """
        # Set random seed if specified
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        # Generate hierarchy
        hierarchy = None
        if cfg.hierarchy is not None and cfg.hierarchy.total_parent_nodes > 0:
            hierarchy = generate_hierarchy(cfg.num_features, cfg.hierarchy)

        # Generate correlation matrix
        correlation_matrix = None
        if cfg.correlation is not None:
            correlation_matrix = generate_random_low_rank_correlation_matrix(
                num_features=cfg.num_features,
                rank=cfg.correlation.rank,
                correlation_scale=cfg.correlation.correlation_scale,
                seed=cfg.correlation.seed,
                device=cfg.device,
                dtype=str_to_dtype(cfg.dtype),
            )

        return cls(
            cfg=cfg,
            feature_dict=None,  # Will be created in __init__
            activation_generator=None,  # Will be created in __init__
            hierarchy=hierarchy,
            correlation_matrix=correlation_matrix,
        )

    def _create_feature_dict(self) -> FeatureDictionary:
        """Create feature dictionary from config."""
        initializer = None
        if self.cfg.orthogonalization is not None:
            initializer = orthogonal_initializer(
                num_steps=self.cfg.orthogonalization.num_steps,
                lr=self.cfg.orthogonalization.lr,
                chunk_size=self.cfg.orthogonalization.chunk_size,
            )

        return FeatureDictionary(
            num_features=self.cfg.num_features,
            hidden_dim=self.cfg.hidden_dim,
            bias=self.cfg.feature_dict_bias,
            initializer=initializer,
            device=self.cfg.device,
        )

    def _create_activation_generator(self) -> ActivationGenerator:
        """Create activation generator from config."""
        # Generate firing probabilities
        _, generator_class = get_firing_probability_class(
            self.cfg.firing_probability.generator_name()
        )
        generator = generator_class(self.cfg.firing_probability)  # type: ignore[call-arg]
        firing_probs = generator.generate(self.cfg.num_features)

        # Generate firing magnitudes
        std_magnitudes = generate_magnitudes(
            self.cfg.num_features, self.cfg.std_firing_magnitudes
        )
        mean_magnitudes = generate_magnitudes(
            self.cfg.num_features, self.cfg.mean_firing_magnitudes
        )

        # Get correlation matrix input
        correlation_input = None
        if self.correlation_matrix is not None:
            correlation_input = self.correlation_matrix

        # Get hierarchy modifier
        modifier = None
        if self.hierarchy is not None:
            modifier = self.hierarchy.modifier

        return ActivationGenerator(
            num_features=self.cfg.num_features,
            firing_probabilities=firing_probs,
            std_firing_magnitudes=std_magnitudes,
            mean_firing_magnitudes=mean_magnitudes,
            modify_activations=modifier,
            correlation_matrix=correlation_input,
            device=self.cfg.device,
            dtype=self.cfg.dtype,
            use_sparse_tensors=self.cfg.use_sparse_tensors,
        )

    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Generate a batch of synthetic hidden activations.

        This is the main method for generating training data. It:

        1. Samples sparse feature activations from ActivationGenerator
        2. Transforms them through FeatureDictionary to get dense activations

        Args:
            batch_size: Number of samples to generate

        Returns:
            Tensor of shape (batch_size, hidden_dim) with hidden activations
        """
        feature_acts = self.activation_generator.sample(batch_size)
        return self.feature_dict(feature_acts)

    @torch.no_grad()
    def sample_with_features(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate both hidden activations and their ground-truth feature activations.

        Useful for evaluation and debugging.

        Args:
            batch_size: Number of samples to generate

        Returns:
            Tuple of (hidden_activations, feature_activations)

            - hidden_activations: (batch_size, hidden_dim)
            - feature_activations: (batch_size, num_features)
        """
        feature_acts = self.activation_generator.sample(batch_size)
        hidden_acts = self.feature_dict(feature_acts)
        return hidden_acts, feature_acts

    def forward(self, batch_size: int) -> torch.Tensor:
        """Forward pass equivalent to sample()."""
        return self.sample(batch_size)

    # =========================================================================
    # Persistence Methods
    # =========================================================================

    def save(self, path: str | Path) -> None:
        """
        Save the SyntheticModel to disk.

        Saves:

        - Config as JSON
        - Feature dictionary weights as safetensors
        - Hierarchy structure as JSON (if present)
        - Correlation matrix as safetensors (if present)

        Args:
            path: Directory to save model to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = path / SYNTHETIC_MODEL_CONFIG_FILENAME
        with open(config_path, "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)

        # Save weights (feature dict + correlation if present)
        weights: dict[str, torch.Tensor] = {
            "feature_vectors": self.feature_dict.feature_vectors.data,
            "bias": self.feature_dict.bias.data,
            "firing_probabilities": self.activation_generator.firing_probabilities,
        }

        if self.correlation_matrix is not None:
            weights["correlation_factor"] = self.correlation_matrix.correlation_factor
            weights["correlation_diag"] = self.correlation_matrix.correlation_diag

        weights_path = path / SYNTHETIC_MODEL_WEIGHTS_FILENAME
        save_file(weights, weights_path)

        # Save hierarchy if present
        if self.hierarchy is not None:
            hierarchy_path = path / SYNTHETIC_MODEL_HIERARCHY_FILENAME
            with open(hierarchy_path, "w") as f:
                json.dump(self.hierarchy.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "SyntheticModel":
        """
        Load a SyntheticModel from disk.

        Args:
            path: Directory containing saved model
            device: Override device (uses saved config device if None)

        Returns:
            Loaded SyntheticModel
        """
        path = Path(path)

        # Load config
        config_path = path / SYNTHETIC_MODEL_CONFIG_FILENAME
        with open(config_path) as f:
            cfg_dict = json.load(f)

        cfg = SyntheticModelConfig.from_dict(cfg_dict)
        if device is not None:
            cfg.device = device

        # Load weights
        weights_path = path / SYNTHETIC_MODEL_WEIGHTS_FILENAME
        weights = load_file(weights_path, device=cfg.device)

        # Reconstruct correlation matrix if present
        correlation_matrix = None
        if "correlation_factor" in weights:
            correlation_matrix = LowRankCorrelationMatrix(
                correlation_factor=weights["correlation_factor"],
                correlation_diag=weights["correlation_diag"],
            )

        # Load hierarchy if present
        hierarchy = None
        hierarchy_path = path / SYNTHETIC_MODEL_HIERARCHY_FILENAME
        if hierarchy_path.exists():
            with open(hierarchy_path) as f:
                hierarchy_dict = json.load(f)
            hierarchy = Hierarchy.from_dict(hierarchy_dict)

        # Create feature dictionary with loaded weights
        feature_dict = FeatureDictionary(
            num_features=cfg.num_features,
            hidden_dim=cfg.hidden_dim,
            bias=cfg.feature_dict_bias,
            initializer=None,  # Don't re-orthogonalize
            device=cfg.device,
        )
        feature_dict.feature_vectors.data = weights["feature_vectors"]
        feature_dict.bias.data = weights["bias"]

        # Create model (will create activation_generator in __init__)
        model = cls(
            cfg=cfg,
            feature_dict=feature_dict,
            activation_generator=None,  # Will be created
            hierarchy=hierarchy,
            correlation_matrix=correlation_matrix,
        )

        # Override firing probabilities with saved values
        model.activation_generator.firing_probabilities = weights[
            "firing_probabilities"
        ]

        return model

    def to(  # type: ignore[override]
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> "SyntheticModel":
        """
        Move model to device.

        Note: Only device is supported for SyntheticModel. dtype parameter is ignored.
        """
        if device is not None:
            device_str = str(device) if isinstance(device, torch.device) else device
            self.cfg.device = device_str
            self.feature_dict = self.feature_dict.to(device)

            # Recreate activation generator on new device
            self.activation_generator = self._create_activation_generator()

        return self
