"""
SyntheticModel class for large-scale SAE training on synthetic data.

This module provides SyntheticModel, which encapsulates ActivationGenerator
and FeatureDictionary with configuration, hierarchy, correlation, and persistence.
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
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
)
from sae_lens.synthetic.hierarchy import (
    Hierarchy,
    HierarchyConfig,
    generate_hierarchy,
)
from sae_lens.util import str_to_dtype

logger = logging.getLogger(__name__)


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
    generator for SAE training experiments. It contains only model-defining
    parameters, not runtime options like device or sparse tensor usage.

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
        bias: Feature dictionary bias. False for no bias, True for bias with
            norm 1.0, or a float for bias with that norm.
        dtype: Data type for model tensors.
        seed: Global random seed for reproducibility.
    """

    num_features: int
    hidden_dim: int
    firing_probability: FiringProbabilityConfig = field(
        default_factory=ZipfianFiringProbabilityConfig
    )
    hierarchy: HierarchyConfig | None = None
    orthogonalization: OrthogonalizationConfig | None = None
    correlation: LowRankCorrelationConfig | None = None
    std_firing_magnitudes: float | MagnitudeConfig = 0.0
    mean_firing_magnitudes: float | MagnitudeConfig = 1.0
    bias: bool | float = 1.0
    dtype: str = "float32"
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.num_features < 1:
            raise ValueError("num_features must be at least 1")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary."""
        d = asdict(self)

        # Handle nested configs with their own to_dict methods
        d["firing_probability"] = self.firing_probability.to_dict()
        if self.hierarchy is not None:
            d["hierarchy"] = self.hierarchy.to_dict()
        if self.orthogonalization is not None:
            d["orthogonalization"] = self.orthogonalization.to_dict()
        if self.correlation is not None:
            d["correlation"] = self.correlation.to_dict()
        if isinstance(self.std_firing_magnitudes, MagnitudeConfig):
            d["std_firing_magnitudes"] = self.std_firing_magnitudes.to_dict()
        if isinstance(self.mean_firing_magnitudes, MagnitudeConfig):
            d["mean_firing_magnitudes"] = self.mean_firing_magnitudes.to_dict()

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SyntheticModelConfig":
        """Deserialize config from dictionary."""
        # Parse nested configs
        ortho_dict = d.get("orthogonalization")
        corr_dict = d.get("correlation")
        hierarchy_dict = d.get("hierarchy")

        updated_dict: dict[str, Any] = {
            **d,
            "firing_probability": FiringProbabilityConfig.from_dict(
                d["firing_probability"]
            ),
            "orthogonalization": (
                OrthogonalizationConfig.from_dict(ortho_dict)
                if ortho_dict is not None
                else None
            ),
            "correlation": (
                LowRankCorrelationConfig.from_dict(corr_dict)
                if corr_dict is not None
                else None
            ),
            "hierarchy": (
                HierarchyConfig.from_dict(hierarchy_dict)
                if hierarchy_dict is not None
                else None
            ),
            "std_firing_magnitudes": _deserialize_magnitude(
                d.get("std_firing_magnitudes", 0.0)
            ),
            "mean_firing_magnitudes": _deserialize_magnitude(
                d.get("mean_firing_magnitudes", 1.0)
            ),
        }
        return cls(**updated_dict)


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

    Runtime options (device, use_sparse_tensors) are stored as instance
    attributes, not in the config, since they don't define the model itself.
    """

    cfg: SyntheticModelConfig
    feature_dict: FeatureDictionary
    activation_generator: ActivationGenerator
    hierarchy: Hierarchy | None
    correlation_matrix: LowRankCorrelationMatrix | None
    device: str

    def __init__(
        self,
        cfg: SyntheticModelConfig,
        feature_dict: FeatureDictionary | None = None,
        activation_generator: ActivationGenerator | None = None,
        hierarchy: Hierarchy | None = None,
        correlation_matrix: LowRankCorrelationMatrix | None = None,
        device: str = "cpu",
    ):
        """
        Create a SyntheticModel.

        All components (feature_dict, activation_generator, hierarchy, correlation_matrix)
        are automatically generated from the config if not explicitly provided.

        Args:
            cfg: Model configuration (defines the model structure)
            feature_dict: Optional pre-created feature dictionary
            activation_generator: Optional pre-created activation generator
            hierarchy: Optional hierarchy structure (generated from cfg.hierarchy if None)
            correlation_matrix: Optional correlation matrix (generated from cfg.correlation if None)
            device: Device for tensors (runtime option, not saved)
        """
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Set random seed if specified (for any operations that don't handle seeding internally)
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        self.hierarchy = hierarchy or self._create_hierarchy()
        self.correlation_matrix = (
            correlation_matrix or self._create_correlation_matrix()
        )
        self.feature_dict = feature_dict or self._create_feature_dict()
        self.activation_generator = (
            activation_generator or self._create_activation_generator()
        )

    @property
    def use_sparse_tensors(self) -> bool:
        """Whether to use sparse tensors for activations."""
        return self.activation_generator.use_sparse_tensors

    @use_sparse_tensors.setter
    def use_sparse_tensors(self, value: bool) -> None:
        """Set whether to use sparse tensors for activations."""
        self.activation_generator.use_sparse_tensors = value

    def _create_hierarchy(self) -> Hierarchy | None:
        """Create hierarchy from config if configured."""
        if self.cfg.hierarchy is None or self.cfg.hierarchy.total_root_nodes == 0:
            return None

        # Compute seed offset to avoid coupling with other random generators
        base_seed = self.cfg.seed
        hierarchy_seed = base_seed + 3 if base_seed is not None else None

        return generate_hierarchy(
            self.cfg.num_features, self.cfg.hierarchy, seed=hierarchy_seed
        )

    def _create_correlation_matrix(self) -> LowRankCorrelationMatrix | None:
        """Create correlation matrix from config if configured."""
        if self.cfg.correlation is None:
            return None

        # Compute seed offset to avoid coupling with other random generators
        base_seed = self.cfg.seed
        correlation_seed = base_seed + 4 if base_seed is not None else None

        return generate_random_low_rank_correlation_matrix(
            num_features=self.cfg.num_features,
            rank=self.cfg.correlation.rank,
            correlation_scale=self.cfg.correlation.correlation_scale,
            seed=correlation_seed,
            device=self.device,
            dtype=str_to_dtype(self.cfg.dtype),
        )

    def _create_feature_dict(self) -> FeatureDictionary:
        """Create feature dictionary from config."""
        initializer = None
        if self.cfg.orthogonalization is not None:
            initializer = orthogonal_initializer(
                num_steps=self.cfg.orthogonalization.num_steps,
                lr=self.cfg.orthogonalization.lr,
                chunk_size=self.cfg.orthogonalization.chunk_size,
                show_progress=True,
            )

        return FeatureDictionary(
            num_features=self.cfg.num_features,
            hidden_dim=self.cfg.hidden_dim,
            bias=self.cfg.bias,
            initializer=initializer,
            device=self.device,
            seed=self.cfg.seed,
        )

    def _create_activation_generator(
        self, use_sparse_tensors: bool = False
    ) -> ActivationGenerator:
        """Create activation generator from config."""
        # Compute seed offsets to avoid coupling between random generators
        base_seed = self.cfg.seed
        firing_prob_seed = base_seed
        std_mag_seed = base_seed + 1 if base_seed is not None else None
        mean_mag_seed = base_seed + 2 if base_seed is not None else None

        # Generate firing probabilities
        firing_probs = self.cfg.firing_probability.generate(
            self.cfg.num_features, seed=firing_prob_seed
        )

        # Apply hierarchy probability compensation if enabled
        if (
            self.hierarchy is not None
            and self.cfg.hierarchy is not None
            and self.cfg.hierarchy.compensate_probabilities
        ):
            correction_factors = self.hierarchy.compute_probability_correction_factors(
                firing_probs
            )
            original_probs = firing_probs
            corrected_probs = firing_probs * correction_factors
            firing_probs = corrected_probs.clamp(max=1.0)

            # Log warning if correction caused probabilities to reach/exceed 1.0
            # (but not if the user explicitly set them to 1.0)
            clamped_by_correction = (corrected_probs >= 1.0) & (original_probs < 1.0)
            if clamped_by_correction.any():
                num_clamped = clamped_by_correction.sum().item()
                logger.warning(
                    f"Hierarchy probability compensation clamped {num_clamped} features "
                    f"(max before clamp: {corrected_probs.max().item():.3f}). "
                    "Consider using lower base probabilities."
                )

        # Generate firing magnitudes
        std_magnitudes = generate_magnitudes(
            self.cfg.num_features, self.cfg.std_firing_magnitudes, seed=std_mag_seed
        )
        mean_magnitudes = generate_magnitudes(
            self.cfg.num_features, self.cfg.mean_firing_magnitudes, seed=mean_mag_seed
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
            device=self.device,
            dtype=self.cfg.dtype,
            use_sparse_tensors=use_sparse_tensors,
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
    def load_from_disk(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> "SyntheticModel":
        """
        Load a SyntheticModel from disk.

        Args:
            path: Directory containing saved model
            device: Device for tensors (runtime option)

        Returns:
            Loaded SyntheticModel
        """
        path = Path(path)

        # Load config
        config_path = path / SYNTHETIC_MODEL_CONFIG_FILENAME
        with open(config_path) as f:
            cfg_dict = json.load(f)

        cfg = SyntheticModelConfig.from_dict(cfg_dict)

        # Load weights
        weights_path = path / SYNTHETIC_MODEL_WEIGHTS_FILENAME
        weights = load_file(weights_path, device=device)

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
            bias=cfg.bias,
            initializer=None,  # Don't re-orthogonalize
            device=device,
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
            device=device,
        )

        # Override firing probabilities with saved values
        model.activation_generator.firing_probabilities = weights[
            "firing_probabilities"
        ]

        return model

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        model_path: str | None = None,
        device: str = "cpu",
        force_download: bool = False,
    ) -> "SyntheticModel":
        """
        Load a SyntheticModel from the Hugging Face model hub.

        Args:
            repo_id: The Hugging Face repository ID (e.g., "username/repo-name")
            model_path: Optional subfolder within the repo. If None, loads from repo root.
            device: Device for tensors (runtime option)
            force_download: Whether to force re-download even if cached

        Returns:
            Loaded SyntheticModel
        """
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Build filename prefixes
            prefix = f"{model_path}/" if model_path else ""

            # Download config
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{prefix}{SYNTHETIC_MODEL_CONFIG_FILENAME}",
                force_download=force_download,
            )

            # Download weights
            weights_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{prefix}{SYNTHETIC_MODEL_WEIGHTS_FILENAME}",
                force_download=force_download,
            )

            # Copy to temp directory for loading
            shutil.copy(config_path, tmp_path / SYNTHETIC_MODEL_CONFIG_FILENAME)
            shutil.copy(weights_path, tmp_path / SYNTHETIC_MODEL_WEIGHTS_FILENAME)

            # Try to download hierarchy if it exists
            try:
                hierarchy_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{prefix}{SYNTHETIC_MODEL_HIERARCHY_FILENAME}",
                    force_download=force_download,
                )
                shutil.copy(
                    hierarchy_path, tmp_path / SYNTHETIC_MODEL_HIERARCHY_FILENAME
                )
            except EntryNotFoundError:
                # Hierarchy is optional - models can be saved without one
                pass

            return cls.load_from_disk(tmp_path, device=device)

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
            self.device = device_str
            self.feature_dict = self.feature_dict.to(device)

            # Recreate activation generator on new device
            self.activation_generator = self._create_activation_generator()

        return self

    @classmethod
    def load_from_source(
        cls,
        source: "SyntheticModelConfig | str",
        device: str = "cpu",
    ) -> "SyntheticModel":
        """
        Load a SyntheticModel from various sources with smart detection.

        This is the recommended way to load a SyntheticModel when the source type
        is not known in advance.

        Args:
            source: One of:

                - SyntheticModelConfig: Create a new model from config
                - Local path string: Load from disk (if path exists or starts with
                  "/", "./", "~", or contains backslash)
                - HuggingFace string: Load from HuggingFace Hub. Format is
                  "repo_id" or "repo_id:model_path" for models in subfolders

            device: Device for tensors (runtime option)

        Returns:
            Loaded or created SyntheticModel
        """
        if isinstance(source, SyntheticModelConfig):
            return cls(source, device=device)

        # String source - determine if local path or HuggingFace
        if _is_local_path(source):
            return cls.load_from_disk(source, device=device)

        # Parse HuggingFace format: "repo_id" or "repo_id:model_path"
        if ":" in source:
            repo_id, model_path = source.split(":", 1)
        else:
            repo_id, model_path = source, None

        return cls.from_pretrained(repo_id, model_path=model_path, device=device)


def _is_local_path(path: str) -> bool:
    """
    Determine if a string represents a local path vs a HuggingFace repo ID.

    Returns True if the path:

    - Exists on the filesystem
    - Starts with "/", "./", "../", or "~"
    - Contains a backslash (Windows path)
    """
    # Check for explicit path indicators
    if path.startswith(("/", "./", "../", "~")) or "\\" in path:
        return True

    # Check if path exists on filesystem
    return Path(path).exists()
