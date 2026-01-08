from typing import Any

import pytest
import torch

from sae_lens.saes.sae import TrainingSAE
from sae_lens.synthetic import (
    ActivationGenerator,
    FeatureDictionary,
    SyntheticActivationIterator,
    train_toy_sae,
)
from sae_lens.training.sae_trainer import SAETrainer


def test_SyntheticActivationIterator_generates_correct_shape():
    feature_dict = FeatureDictionary(num_features=10, hidden_dim=8)
    probs = torch.ones(10) * 0.1

    activations_gen = ActivationGenerator(
        num_features=10,
        firing_probabilities=probs,
    )

    iterator = SyntheticActivationIterator(feature_dict, activations_gen, batch_size=32)
    batch = next(iterator)

    assert batch.shape == (32, 8)


def test_SyntheticActivationIterator_is_iterable():
    feature_dict = FeatureDictionary(num_features=5, hidden_dim=4)
    probs = torch.ones(5) * 0.2

    activations_gen = ActivationGenerator(
        num_features=5,
        firing_probabilities=probs,
    )

    iterator = SyntheticActivationIterator(feature_dict, activations_gen, batch_size=16)

    batches = [next(iterator) for _ in range(3)]
    assert len(batches) == 3
    assert all(b.shape == (16, 4) for b in batches)


def test_SyntheticActivationIterator_produces_sparse_activations():
    feature_dict = FeatureDictionary(num_features=20, hidden_dim=10)
    probs = torch.ones(20) * 0.05

    activations_gen = ActivationGenerator(
        num_features=20,
        firing_probabilities=probs,
    )

    iterator = SyntheticActivationIterator(
        feature_dict, activations_gen, batch_size=100
    )
    batch = next(iterator)

    # Some activations should be zero (sparse input)
    # Not all hidden activations will be zero though due to the linear transform
    assert batch.shape == (100, 10)


SyntheticSetup = tuple[TrainingSAE[Any], FeatureDictionary, ActivationGenerator]


@pytest.fixture
def synthetic_training_setup() -> SyntheticSetup:
    """Create a minimal setup for testing train_toy_sae."""
    hidden_dim = 8
    num_features = 10

    feature_dict = FeatureDictionary(num_features=num_features, hidden_dim=hidden_dim)

    activations_gen = ActivationGenerator(
        num_features=num_features,
        firing_probabilities=0.1,
    )

    sae = TrainingSAE.from_dict(
        {
            "architecture": "standard",
            "d_in": hidden_dim,
            "d_sae": num_features,
            "activation_fn_str": "relu",
            "normalize_sae_decoder": False,
            "apply_b_dec_to_input": True,
            "dtype": "float32",
            "device": "cpu",
            "model_name": "test",
            "hook_name": "test",
            "hook_layer": 0,
        }
    )

    return sae, feature_dict, activations_gen


def test_train_toy_sae_runs(
    synthetic_training_setup: SyntheticSetup,
) -> None:
    """Test that training runs without errors."""
    sae, feature_dict, activations_gen = synthetic_training_setup

    train_toy_sae(
        sae=sae,
        feature_dict=feature_dict,
        activations_generator=activations_gen,
        training_samples=1024,
        batch_size=128,
    )


def test_train_toy_sae_snapshot_fn_called_correct_times(
    synthetic_training_setup: SyntheticSetup,
) -> None:
    """Test that snapshot_fn is called n_snapshots times."""
    sae, feature_dict, activations_gen = synthetic_training_setup

    snapshot_calls: list[int] = []

    def on_snapshot(trainer: SAETrainer[Any, Any]) -> None:
        snapshot_calls.append(trainer.n_training_steps)

    train_toy_sae(
        sae=sae,
        feature_dict=feature_dict,
        activations_generator=activations_gen,
        training_samples=2048,
        batch_size=128,
        n_snapshots=4,
        snapshot_fn=on_snapshot,
    )

    assert len(snapshot_calls) == 4


def test_train_toy_sae_snapshot_fn_receives_trainer(
    synthetic_training_setup: SyntheticSetup,
) -> None:
    """Test that snapshot_fn receives the trainer with expected attributes."""
    sae, feature_dict, activations_gen = synthetic_training_setup

    received_trainers: list[SAETrainer[Any, Any]] = []

    def on_snapshot(trainer: SAETrainer[Any, Any]) -> None:
        received_trainers.append(trainer)

    train_toy_sae(
        sae=sae,
        feature_dict=feature_dict,
        activations_generator=activations_gen,
        training_samples=1024,
        batch_size=128,
        n_snapshots=2,
        snapshot_fn=on_snapshot,
    )

    assert len(received_trainers) == 2
    for trainer in received_trainers:
        # Trainer should have these attributes accessible
        assert hasattr(trainer, "n_training_steps")
        assert hasattr(trainer, "n_training_samples")
        assert hasattr(trainer, "sae")


def test_train_toy_sae_snapshots_evenly_spaced(
    synthetic_training_setup: SyntheticSetup,
) -> None:
    """Test that snapshots are taken at evenly spaced intervals."""
    sae, feature_dict, activations_gen = synthetic_training_setup

    snapshot_steps: list[int] = []

    def on_snapshot(trainer: SAETrainer[Any, Any]) -> None:
        snapshot_steps.append(trainer.n_training_steps)

    # 4096 samples / 128 batch = 32 steps total
    # 4 snapshots should be at steps 8, 16, 24, 32
    train_toy_sae(
        sae=sae,
        feature_dict=feature_dict,
        activations_generator=activations_gen,
        training_samples=4096,
        batch_size=128,
        n_snapshots=4,
        snapshot_fn=on_snapshot,
    )

    assert len(snapshot_steps) == 4
    # Check that steps are increasing
    assert snapshot_steps == sorted(snapshot_steps)
    # Check roughly even spacing (within 1 step tolerance due to rounding)
    intervals = [
        snapshot_steps[i + 1] - snapshot_steps[i]
        for i in range(len(snapshot_steps) - 1)
    ]
    assert max(intervals) - min(intervals) <= 1


def test_train_toy_sae_no_snapshots_by_default(
    synthetic_training_setup: SyntheticSetup,
) -> None:
    """Test that no snapshot_fn is needed when n_snapshots=0."""
    sae, feature_dict, activations_gen = synthetic_training_setup

    # Should not raise even without snapshot_fn
    train_toy_sae(
        sae=sae,
        feature_dict=feature_dict,
        activations_generator=activations_gen,
        training_samples=1024,
        batch_size=128,
        n_snapshots=0,
    )


def test_train_toy_sae_snapshot_fn_required_when_n_snapshots_positive(
    synthetic_training_setup: SyntheticSetup,
) -> None:
    """Test that snapshot_fn is required when n_snapshots > 0."""
    sae, feature_dict, activations_gen = synthetic_training_setup

    with pytest.raises(ValueError, match="snapshot_fn must be provided"):
        train_toy_sae(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            training_samples=1024,
            batch_size=128,
            n_snapshots=2,
            snapshot_fn=None,
        )
