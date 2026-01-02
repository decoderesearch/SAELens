"""
Utilities for training SAEs on synthetic data.

This module provides helpers for:
- Generating training data from feature dictionaries
- Training SAEs on synthetic data
- Evaluating SAEs against known ground truth features
- Initializing SAEs to match feature dictionaries
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

from sae_lens.saes.sae import TrainingSAE
from sae_lens.toy_model.feature_dictionary import FeatureDictionary


class SyntheticActivationIterator(Iterator[torch.Tensor]):
    """
    An iterator that generates synthetic activations for SAE training.

    This iterator wraps a FeatureDictionary and a function that generates
    feature activations, producing hidden activations that can be used
    to train an SAE.

    Example:
        >>> from sae_lens.toy_model import FeatureDictionary, generate_activations
        >>> feature_dict = FeatureDictionary(num_features=50, hidden_dim=32)
        >>> probs = torch.ones(50) * 0.1
        >>>
        >>> def generate_batch(batch_size):
        ...     return generate_activations(batch_size, probs)
        >>>
        >>> iterator = SyntheticActivationIterator(
        ...     feature_dict, generate_batch, batch_size=1024
        ... )
        >>> batch = next(iterator)  # Returns hidden activations
    """

    def __init__(
        self,
        feature_dict: FeatureDictionary,
        generate_features_fn: Callable[[int], torch.Tensor],
        batch_size: int,
    ):
        """
        Create a new SyntheticActivationIterator.

        Args:
            feature_dict: The feature dictionary to use for generating hidden activations
            generate_features_fn: A function that takes a batch size and returns
                feature activations of shape [batch_size, num_features]
            batch_size: Number of samples per batch
        """
        self.feature_dict = feature_dict
        self.generate_features_fn = generate_features_fn
        self.batch_size = batch_size

    @torch.no_grad()
    def next_batch(self) -> torch.Tensor:
        """Generate the next batch of hidden activations."""
        features = self.generate_features_fn(self.batch_size)
        return self.feature_dict(features)

    def __iter__(self) -> "SyntheticActivationIterator":
        return self

    def __next__(self) -> torch.Tensor:
        return self.next_batch()


@dataclass
class SyntheticEvalResult:
    """Results from evaluating an SAE on synthetic data."""

    true_l0: float
    """Average L0 of the true feature activations"""

    sae_l0: float
    """Average L0 of the SAE's latent activations"""

    dead_latents: int
    """Number of SAE latents that never fired"""

    shrinkage: float
    """Average ratio of SAE output norm to input norm (1.0 = no shrinkage)"""


@torch.no_grad()
def eval_sae_on_synthetic(
    sae: torch.nn.Module,
    feature_dict: FeatureDictionary,
    generate_features_fn: Callable[[int], torch.Tensor],
    num_samples: int = 100_000,
) -> SyntheticEvalResult:
    """
    Evaluate an SAE on synthetic data with known ground truth.

    Args:
        sae: The SAE to evaluate. Must have encode() and decode() methods.
        feature_dict: The feature dictionary used to generate activations
        generate_features_fn: Function that generates feature activations
        num_samples: Number of samples to use for evaluation

    Returns:
        SyntheticEvalResult containing evaluation metrics
    """
    sae.eval()

    # Generate samples
    feature_acts = generate_features_fn(num_samples)
    true_l0 = (feature_acts > 0).float().sum(dim=-1).mean().item()
    hidden_acts = feature_dict(feature_acts)

    # Filter out entries where no features fire
    non_zero_mask = hidden_acts.norm(dim=-1) > 0
    hidden_acts_filtered = hidden_acts[non_zero_mask]

    # Get SAE reconstructions
    sae_latents = sae.encode(hidden_acts_filtered)  # type: ignore[attr-defined]
    sae_output = sae.decode(sae_latents)  # type: ignore[attr-defined]

    sae_l0 = (sae_latents > 0).float().sum(dim=-1).mean().item()
    dead_latents = int(
        ((sae_latents == 0).sum(dim=0) == sae_latents.shape[0]).sum().item()
    )
    shrinkage = (
        (sae_output.norm(dim=-1) / hidden_acts_filtered.norm(dim=-1)).mean().item()
    )

    return SyntheticEvalResult(
        true_l0=true_l0,
        sae_l0=sae_l0,
        dead_latents=dead_latents,
        shrinkage=shrinkage,
    )


@torch.no_grad()
def init_sae_from_feature_dict(
    sae: torch.nn.Module,
    feature_dict: FeatureDictionary,
    noise_level: float = 0.0,
    feature_ordering: torch.Tensor | None = None,
) -> None:
    """
    Initialize an SAE's weights to match a feature dictionary.

    This can be useful for:
    - Starting training from a known good initialization
    - Testing SAE evaluation code with ground truth
    - Ablation studies on initialization

    Args:
        sae: The SAE to initialize. Must have W_enc and W_dec attributes.
        feature_dict: The feature dictionary to match
        noise_level: Standard deviation of Gaussian noise to add (0 = exact match)
        feature_ordering: Optional permutation of feature indices

    Example:
        >>> feature_dict = FeatureDictionary(num_features=10, hidden_dim=8)
        >>> sae = TrainingSAE(cfg)
        >>> init_sae_from_feature_dict(sae, feature_dict, noise_level=0.01)
    """
    features = feature_dict.feature_vectors  # [num_features, hidden_dim]
    min_dim = min(sae.W_enc.shape[1], features.shape[0])  # type: ignore[attr-defined]

    if feature_ordering is not None:
        features = features[feature_ordering]

    features = features[:min_dim]

    # W_enc is [hidden_dim, d_sae], feature vectors are [num_features, hidden_dim]
    sae.W_enc.data[:, :min_dim] = (  # type: ignore[index]
        features.T + torch.randn_like(features.T) * noise_level
    )
    sae.W_dec.data = sae.W_enc.data.T.clone()  # type: ignore[union-attr]


def train_sae_on_synthetic(
    sae: TrainingSAE[Any],
    feature_dict: FeatureDictionary,
    generate_features_fn: Callable[[int], torch.Tensor],
    training_samples: int = 10_000_000,
    batch_size: int = 1024,
    lr: float = 3e-4,
    lr_warm_up_steps: int = 0,
    lr_decay_steps: int = 0,
    device: str | torch.device = "cpu",
    n_checkpoints: int = 0,
    checkpoint_path: str | None = None,
    log_to_wandb: bool = False,
    wandb_project: str = "sae_synthetic_training",
) -> TrainingSAE[Any]:
    """
    Train an SAE on synthetic activations from a feature dictionary.

    This is a convenience function that sets up the training loop with
    sensible defaults for synthetic data experiments.

    Args:
        sae: The TrainingSAE to train
        feature_dict: The feature dictionary to generate activations from
        generate_features_fn: Function that generates feature activations.
            Takes batch_size as argument, returns tensor of shape [batch_size, num_features]
        training_samples: Total number of training samples
        batch_size: Batch size for training
        lr: Learning rate
        lr_warm_up_steps: Number of warmup steps for learning rate
        lr_decay_steps: Number of steps over which to decay learning rate
        device: Device to train on
        n_checkpoints: Number of checkpoints to save during training
        checkpoint_path: Path to save checkpoints (required if n_checkpoints > 0)
        log_to_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name if logging

    Returns:
        The trained SAE

    Example:
        >>> from sae_lens import StandardTrainingSAE
        >>> from sae_lens.toy_model import (
        ...     FeatureDictionary,
        ...     generate_activations,
        ...     train_sae_on_synthetic,
        ... )
        >>>
        >>> # Create feature dictionary
        >>> feature_dict = FeatureDictionary(num_features=100, hidden_dim=64)
        >>>
        >>> # Create SAE
        >>> cfg = StandardTrainingSAEConfig(d_in=64, d_sae=100, ...)
        >>> sae = StandardTrainingSAE(cfg)
        >>>
        >>> # Define feature generation
        >>> probs = torch.ones(100) * 0.1
        >>> def gen_fn(batch_size):
        ...     return generate_activations(batch_size, probs)
        >>>
        >>> # Train
        >>> trained_sae = train_sae_on_synthetic(
        ...     sae, feature_dict, gen_fn,
        ...     training_samples=1_000_000,
        ...     lr=1e-3,
        ... )
    """
    # Import here to avoid circular imports and heavy dependencies at module load time
    from sae_lens.config import LoggingConfig, SAETrainerConfig
    from sae_lens.training.sae_trainer import SAETrainer

    device_str = str(device) if isinstance(device, torch.device) else device

    # Create data iterator
    data_iterator = SyntheticActivationIterator(
        feature_dict=feature_dict,
        generate_features_fn=generate_features_fn,
        batch_size=batch_size,
    )

    # Create trainer config
    trainer_cfg = SAETrainerConfig(
        n_checkpoints=n_checkpoints,
        checkpoint_path=checkpoint_path,
        save_final_checkpoint=False,
        total_training_samples=training_samples,
        device=device_str,
        autocast=False,
        lr=lr,
        lr_end=lr,
        lr_scheduler_name="constant",
        lr_warm_up_steps=lr_warm_up_steps,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_decay_steps=lr_decay_steps,
        n_restart_cycles=1,
        train_batch_size_samples=batch_size,
        dead_feature_window=1000,
        feature_sampling_window=2000,
        logger=LoggingConfig(
            log_to_wandb=log_to_wandb,
            wandb_project=wandb_project,
        ),
    )

    # Create trainer and train
    feature_dict.eval()
    trainer = SAETrainer(
        cfg=trainer_cfg,
        sae=sae,
        data_provider=data_iterator,
    )

    return trainer.fit()
