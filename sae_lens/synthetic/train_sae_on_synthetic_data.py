from collections.abc import Iterator
from typing import Any

import torch

from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.sae import TrainingSAE
from sae_lens.synthetic.activation_generator import ActivationGenerator
from sae_lens.synthetic.feature_dictionary import FeatureDictionary
from sae_lens.training.sae_trainer import SAETrainer


def train_sae_on_synthetic_data(
    sae: TrainingSAE[Any],
    feature_dict: FeatureDictionary,
    activations_generator: ActivationGenerator,
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

    device_str = str(device) if isinstance(device, torch.device) else device

    # Create data iterator
    data_iterator = SyntheticActivationIterator(
        feature_dict=feature_dict,
        activations_generator=activations_generator,
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


class SyntheticActivationIterator(Iterator[torch.Tensor]):
    """
    An iterator that generates synthetic activations for SAE training.

    This iterator wraps a FeatureDictionary and a function that generates
    feature activations, producing hidden activations that can be used
    to train an SAE.
    """

    def __init__(
        self,
        feature_dict: FeatureDictionary,
        activations_generator: ActivationGenerator,
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
        self.activations_generator = activations_generator
        self.batch_size = batch_size

    @torch.no_grad()
    def next_batch(self) -> torch.Tensor:
        """Generate the next batch of hidden activations."""
        features = self.activations_generator(self.batch_size)
        return self.feature_dict(features)

    def __iter__(self) -> "SyntheticActivationIterator":
        return self

    def __next__(self) -> torch.Tensor:
        return self.next_batch()
