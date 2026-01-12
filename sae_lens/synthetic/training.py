from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

import torch

from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.sae import TrainingSAE
from sae_lens.synthetic.activation_generator import ActivationGenerator
from sae_lens.synthetic.feature_dictionary import FeatureDictionary
from sae_lens.training.sae_trainer import SAETrainer, SaveCheckpointFn


def train_toy_sae(
    sae: TrainingSAE[Any],
    feature_dict: FeatureDictionary,
    activations_generator: ActivationGenerator,
    training_samples: int = 10_000_000,
    batch_size: int = 1024,
    lr: float = 3e-4,
    lr_warm_up_steps: int = 0,
    lr_decay_steps: int = 0,
    device: str | torch.device = "cpu",
    n_snapshots: int = 0,
    snapshot_fn: Callable[[SAETrainer[Any, Any]], None] | None = None,
    autocast_sae: bool = False,
    autocast_data: bool = False,
) -> None:
    """
    Train an SAE on synthetic activations from a feature dictionary.

    This is a convenience function that sets up the training loop with
    sensible defaults for small-scale synthetic data experiments.

    Args:
        sae: The TrainingSAE to train
        feature_dict: The feature dictionary that maps feature activations to
            hidden activations
        activations_generator: Generator that produces feature activations
        training_samples: Total number of training samples
        batch_size: Batch size for training
        lr: Learning rate
        lr_warm_up_steps: Number of warmup steps for learning rate
        lr_decay_steps: Number of steps over which to decay learning rate
        device: Device to train on
        n_snapshots: Number of snapshots to take during training. Snapshots are
            evenly spaced throughout training.
        snapshot_fn: Callback function called at each snapshot point. Receives
            the SAETrainer instance, allowing access to the SAE, training step,
            and other training state. Required if n_snapshots > 0.
        autocast_sae: Whether to autocast the SAE to bfloat16. Only recommend for large SAEs on CUDA
        autocast_data: Whether to autocast the activations generator and feature dictionary to bfloat16. Only recommend for large data on CUDA.
    """

    device_str = str(device) if isinstance(device, torch.device) else device

    # Create data iterator
    data_iterator = SyntheticActivationIterator(
        feature_dict=feature_dict,
        activations_generator=activations_generator,
        batch_size=batch_size,
        autocast=autocast_data,
    )

    # Create trainer config
    trainer_cfg = SAETrainerConfig(
        n_checkpoints=n_snapshots,
        checkpoint_path=None,
        save_final_checkpoint=False,
        total_training_samples=training_samples,
        device=device_str,
        autocast=autocast_sae,
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
            log_to_wandb=False,
            # hacky way to disable evals, but works for now
            eval_every_n_wandb_logs=2**31 - 1,
        ),
    )

    def snapshot_wrapper(
        snapshot_fn: Callable[[SAETrainer[Any, Any]], None] | None,
    ) -> SaveCheckpointFn:
        def save_checkpoint(checkpoint_path: Path | None) -> None:  # noqa: ARG001
            if snapshot_fn is None:
                raise ValueError("snapshot_fn must be provided to take snapshots")
            snapshot_fn(trainer)

        return save_checkpoint

    # Create trainer and train
    feature_dict.eval()
    trainer = SAETrainer(
        cfg=trainer_cfg,
        sae=sae,
        data_provider=data_iterator,
        save_checkpoint_fn=snapshot_wrapper(snapshot_fn),
    )

    trainer.fit()


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
        autocast: bool = False,
    ):
        """
        Create a new SyntheticActivationIterator.

        Args:
            feature_dict: The feature dictionary to use for generating hidden activations
            activations_generator: Generator that produces feature activations
            batch_size: Number of samples per batch
            autocast: Whether to autocast the activations generator and feature dictionary to bfloat16.
        """
        self.feature_dict = feature_dict
        self.activations_generator = activations_generator
        self.batch_size = batch_size
        self.autocast = autocast

    @torch.no_grad()
    def next_batch(self) -> torch.Tensor:
        """Generate the next batch of hidden activations."""
        with torch.autocast(
            device_type=self.feature_dict.feature_vectors.device.type,
            dtype=torch.bfloat16,
            enabled=self.autocast,
        ):
            features = self.activations_generator(self.batch_size)
            return self.feature_dict(features)

    def __iter__(self) -> "SyntheticActivationIterator":
        return self

    def __next__(self) -> torch.Tensor:
        return self.next_batch()
