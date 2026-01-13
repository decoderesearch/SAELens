import tempfile
from pathlib import Path

import pytest
import torch

from sae_lens.synthetic import (
    ExponentialMagnitudeConfig,
    HierarchyConfig,
    LinearMagnitudeConfig,
    LowRankCorrelationConfig,
    MagnitudeConfig,
    OrthogonalizationConfig,
    SyntheticModel,
    SyntheticModelConfig,
    ZipfianFiringProbabilityConfig,
)


def test_synthetic_model_config_default_values():
    cfg = SyntheticModelConfig(num_features=64, hidden_dim=32)
    assert cfg.num_features == 64
    assert cfg.hidden_dim == 32
    assert isinstance(cfg.firing_probability, ZipfianFiringProbabilityConfig)
    assert cfg.hierarchy is None
    assert cfg.orthogonalization is not None
    assert cfg.correlation is None


def test_synthetic_model_config_validation_num_features():
    with pytest.raises(ValueError, match="num_features must be at least 1"):
        SyntheticModelConfig(num_features=0, hidden_dim=32)


def test_synthetic_model_config_validation_hidden_dim():
    with pytest.raises(ValueError, match="hidden_dim must be at least 1"):
        SyntheticModelConfig(num_features=64, hidden_dim=0)


def test_synthetic_model_config_to_dict_from_dict_roundtrip():
    original = SyntheticModelConfig(
        num_features=128,
        hidden_dim=64,
        hierarchy=HierarchyConfig(total_parent_nodes=5, branching_factor=3),
        orthogonalization=OrthogonalizationConfig(num_steps=100),
        correlation=LowRankCorrelationConfig(rank=16),
        seed=42,
    )
    d = original.to_dict()
    restored = SyntheticModelConfig.from_dict(d)
    assert restored.num_features == original.num_features
    assert restored.hidden_dim == original.hidden_dim
    assert original.hierarchy is not None
    assert restored.hierarchy is not None
    assert (
        restored.hierarchy.total_parent_nodes == original.hierarchy.total_parent_nodes
    )
    assert original.orthogonalization is not None
    assert restored.orthogonalization is not None
    assert restored.orthogonalization.num_steps == original.orthogonalization.num_steps
    assert original.correlation is not None
    assert restored.correlation is not None
    assert restored.correlation.rank == original.correlation.rank
    assert restored.seed == original.seed


def test_synthetic_model_from_config_creates_model():
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
        seed=42,
    )
    model = SyntheticModel.from_config(cfg)
    assert model.cfg == cfg
    assert model.feature_dict is not None
    assert model.activation_generator is not None
    assert model.feature_dict.num_features == 32
    assert model.feature_dict.hidden_dim == 16


def test_synthetic_model_sample_returns_correct_shape():
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)
    samples = model.sample(100)
    assert samples.shape == (100, 16)


def test_synthetic_model_sample_with_features_returns_both():
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)
    hidden_acts, feature_acts = model.sample_with_features(100)
    assert hidden_acts.shape == (100, 16)
    assert feature_acts.shape == (100, 32)


def test_synthetic_model_with_hierarchy():
    cfg = SyntheticModelConfig(
        num_features=64,
        hidden_dim=32,
        hierarchy=HierarchyConfig(
            total_parent_nodes=5, branching_factor=2, max_depth=2, seed=42
        ),
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)
    assert model.hierarchy is not None
    assert len(model.hierarchy.roots) > 0
    samples = model.sample(50)
    assert samples.shape == (50, 32)


def test_synthetic_model_with_correlation():
    cfg = SyntheticModelConfig(
        num_features=64,
        hidden_dim=32,
        correlation=LowRankCorrelationConfig(rank=8, correlation_scale=0.1),
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)
    assert model.correlation_matrix is not None
    samples = model.sample(50)
    assert samples.shape == (50, 32)


def test_synthetic_model_save_load_roundtrip():
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
        seed=42,
    )
    model = SyntheticModel.from_config(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        model.save(save_path)

        # Check files exist
        assert (save_path / "synthetic_model_config.json").exists()
        assert (save_path / "synthetic_model.safetensors").exists()

        # Load and compare
        loaded = SyntheticModel.load(save_path)
        assert loaded.cfg.num_features == cfg.num_features
        assert loaded.cfg.hidden_dim == cfg.hidden_dim

        # Feature vectors should be identical
        assert torch.allclose(
            loaded.feature_dict.feature_vectors, model.feature_dict.feature_vectors
        )


def test_synthetic_model_save_load_with_hierarchy():
    cfg = SyntheticModelConfig(
        num_features=64,
        hidden_dim=32,
        hierarchy=HierarchyConfig(
            total_parent_nodes=3, branching_factor=2, max_depth=2, seed=42
        ),
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        model.save(save_path)

        # Hierarchy file should exist
        assert (save_path / "hierarchy.json").exists()

        loaded = SyntheticModel.load(save_path)
        assert loaded.hierarchy is not None
        assert model.hierarchy is not None
        assert (
            loaded.hierarchy.feature_indices_used
            == model.hierarchy.feature_indices_used
        )


def test_synthetic_model_save_load_with_correlation():
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        correlation=LowRankCorrelationConfig(rank=8, seed=42),
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        model.save(save_path)

        loaded = SyntheticModel.load(save_path)
        assert loaded.correlation_matrix is not None
        assert model.correlation_matrix is not None
        assert torch.allclose(
            loaded.correlation_matrix.correlation_factor,
            model.correlation_matrix.correlation_factor,
        )


def test_synthetic_model_to_device():
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)

    # Test moving to same device (cpu)
    model.to("cpu")
    assert model.cfg.device == "cpu"
    samples = model.sample(10)
    assert samples.device.type == "cpu"


def test_synthetic_model_config_with_magnitude_configs():
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        std_firing_magnitudes=LinearMagnitudeConfig(start=0.5, end=0.1),
        mean_firing_magnitudes=ExponentialMagnitudeConfig(start=2.0, end=0.5),
        orthogonalization=None,
    )
    d = cfg.to_dict()
    restored = SyntheticModelConfig.from_dict(d)

    # Check std_firing_magnitudes
    assert isinstance(restored.std_firing_magnitudes, MagnitudeConfig)
    assert isinstance(restored.std_firing_magnitudes, LinearMagnitudeConfig)
    assert restored.std_firing_magnitudes.start == 0.5
    assert restored.std_firing_magnitudes.end == 0.1

    # Check mean_firing_magnitudes
    assert isinstance(restored.mean_firing_magnitudes, MagnitudeConfig)
    assert isinstance(restored.mean_firing_magnitudes, ExponentialMagnitudeConfig)
    assert restored.mean_firing_magnitudes.start == 2.0
    assert restored.mean_firing_magnitudes.end == 0.5


def test_synthetic_model_config_with_float_magnitudes():
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        std_firing_magnitudes=0.5,
        mean_firing_magnitudes=2.0,
        orthogonalization=None,
    )
    d = cfg.to_dict()
    restored = SyntheticModelConfig.from_dict(d)

    assert restored.std_firing_magnitudes == 0.5
    assert restored.mean_firing_magnitudes == 2.0


def test_synthetic_model_with_magnitude_configs_generates_samples():
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        std_firing_magnitudes=LinearMagnitudeConfig(start=0.2, end=0.05),
        mean_firing_magnitudes=LinearMagnitudeConfig(start=2.0, end=0.5),
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)
    samples = model.sample(100)
    assert samples.shape == (100, 16)
