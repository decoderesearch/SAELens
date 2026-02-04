import logging
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from huggingface_hub.utils import EntryNotFoundError

import sae_lens.synthetic.synthetic_model as synthetic_model_module
from sae_lens.synthetic import (
    ConstantFiringProbabilityConfig,
    ExponentialMagnitudeConfig,
    FoldedNormalMagnitudeConfig,
    HierarchyConfig,
    LinearMagnitudeConfig,
    LowRankCorrelationConfig,
    MagnitudeConfig,
    OrthogonalizationConfig,
    RandomFiringProbabilityConfig,
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
    assert cfg.orthogonalization is None
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
        hierarchy=HierarchyConfig(total_root_nodes=5, branching_factor=3),
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
    assert restored.hierarchy.total_root_nodes == original.hierarchy.total_root_nodes
    assert original.orthogonalization is not None
    assert restored.orthogonalization is not None
    assert restored.orthogonalization.num_steps == original.orthogonalization.num_steps
    assert original.correlation is not None
    assert restored.correlation is not None
    assert restored.correlation.rank == original.correlation.rank
    assert restored.seed == original.seed


def test_synthetic_model_config_from_dict_with_all_none_optional_fields():
    d = {
        "num_features": 64,
        "hidden_dim": 32,
        "firing_probability": {
            "generator_name": "zipfian",
            "exponent": 1.5,
            "max_prob": 0.5,
        },
        "hierarchy": None,
        "orthogonalization": None,
        "correlation": None,
        "std_firing_magnitudes": 0.0,
        "mean_firing_magnitudes": 1.0,
        "bias": True,
        "dtype": "float32",
        "seed": None,
    }
    cfg = SyntheticModelConfig.from_dict(d)

    assert cfg.num_features == 64
    assert cfg.hidden_dim == 32
    assert cfg.hierarchy is None
    assert cfg.orthogonalization is None
    assert cfg.correlation is None
    assert cfg.seed is None


def test_synthetic_model_config_from_dict_missing_optional_fields():
    d = {
        "num_features": 64,
        "hidden_dim": 32,
        "firing_probability": {"generator_name": "constant", "probability": 0.5},
    }
    cfg = SyntheticModelConfig.from_dict(d)

    assert cfg.num_features == 64
    assert cfg.hierarchy is None
    assert cfg.orthogonalization is None
    assert cfg.correlation is None
    # Default values for magnitudes
    assert cfg.std_firing_magnitudes == 0.0
    assert cfg.mean_firing_magnitudes == 1.0


def test_synthetic_model_config_from_dict_with_different_firing_probability_types():
    # Test with ConstantFiringProbabilityConfig
    d_constant = {
        "num_features": 32,
        "hidden_dim": 16,
        "firing_probability": {"generator_name": "constant", "probability": 0.3},
    }
    cfg_constant = SyntheticModelConfig.from_dict(d_constant)
    assert isinstance(cfg_constant.firing_probability, ConstantFiringProbabilityConfig)
    assert cfg_constant.firing_probability.probability == 0.3

    # Test with RandomFiringProbabilityConfig
    d_random = {
        "num_features": 32,
        "hidden_dim": 16,
        "firing_probability": {
            "generator_name": "random",
            "min_prob": 0.1,
            "max_prob": 0.5,
        },
    }
    cfg_random = SyntheticModelConfig.from_dict(d_random)
    assert isinstance(cfg_random.firing_probability, RandomFiringProbabilityConfig)
    assert cfg_random.firing_probability.min_prob == 0.1
    assert cfg_random.firing_probability.max_prob == 0.5

    # Test with ZipfianFiringProbabilityConfig
    d_zipfian = {
        "num_features": 32,
        "hidden_dim": 16,
        "firing_probability": {
            "generator_name": "zipfian",
            "exponent": 2.0,
            "max_prob": 0.8,
        },
    }
    cfg_zipfian = SyntheticModelConfig.from_dict(d_zipfian)
    assert isinstance(cfg_zipfian.firing_probability, ZipfianFiringProbabilityConfig)
    assert cfg_zipfian.firing_probability.exponent == 2.0
    assert cfg_zipfian.firing_probability.max_prob == 0.8


def test_synthetic_model_config_from_dict_with_magnitude_configs():
    d = {
        "num_features": 32,
        "hidden_dim": 16,
        "firing_probability": {"generator_name": "constant", "probability": 0.5},
        "std_firing_magnitudes": {
            "generator_name": "linear",
            "start": 0.5,
            "end": 0.1,
        },
        "mean_firing_magnitudes": {
            "generator_name": "exponential",
            "start": 2.0,
            "end": 0.5,
        },
    }
    cfg = SyntheticModelConfig.from_dict(d)

    assert isinstance(cfg.std_firing_magnitudes, LinearMagnitudeConfig)
    assert cfg.std_firing_magnitudes.start == 0.5
    assert cfg.std_firing_magnitudes.end == 0.1

    assert isinstance(cfg.mean_firing_magnitudes, ExponentialMagnitudeConfig)
    assert cfg.mean_firing_magnitudes.start == 2.0
    assert cfg.mean_firing_magnitudes.end == 0.5


def test_synthetic_model_from_config_creates_model():
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
        seed=42,
    )
    model = SyntheticModel(cfg)
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
    model = SyntheticModel(cfg)
    samples = model.sample(100)
    assert samples.shape == (100, 16)


def test_synthetic_model_sample_with_features_returns_both():
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel(cfg)
    hidden_acts, feature_acts = model.sample_with_features(100)
    assert hidden_acts.shape == (100, 16)
    assert feature_acts.shape == (100, 32)


def test_synthetic_model_with_bias():
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        bias=0.5,
        orthogonalization=None,
        seed=42,
    )
    model = SyntheticModel(cfg)
    assert model.feature_dict.bias.norm().item() == pytest.approx(0.5, abs=1e-5)
    assert model.feature_dict.bias.requires_grad


def test_synthetic_model_with_hierarchy():
    cfg = SyntheticModelConfig(
        num_features=64,
        hidden_dim=32,
        hierarchy=HierarchyConfig(total_root_nodes=5, branching_factor=2, max_depth=2),
        orthogonalization=None,
        seed=42,
    )
    model = SyntheticModel(cfg)
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
    model = SyntheticModel(cfg)
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
    model = SyntheticModel(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        model.save(save_path)

        # Check files exist
        assert (save_path / "synthetic_model_config.json").exists()
        assert (save_path / "synthetic_model.safetensors").exists()

        # Load and compare
        loaded = SyntheticModel.load_from_disk(save_path)
        assert loaded.cfg == cfg

        # Feature vectors should be identical
        assert torch.allclose(
            loaded.feature_dict.feature_vectors, model.feature_dict.feature_vectors
        )


def test_synthetic_model_save_load_with_hierarchy():
    cfg = SyntheticModelConfig(
        num_features=64,
        hidden_dim=32,
        hierarchy=HierarchyConfig(
            total_root_nodes=3, branching_factor=(2, 4), max_depth=2
        ),
        orthogonalization=None,
        seed=42,
    )
    model = SyntheticModel(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        model.save(save_path)

        # Hierarchy file should exist
        assert (save_path / "hierarchy.json").exists()

        loaded = SyntheticModel.load_from_disk(save_path)
        assert loaded.cfg == cfg
        assert loaded.hierarchy is not None
        assert model.hierarchy is not None
        assert loaded.hierarchy == model.hierarchy


def test_synthetic_model_save_load_with_correlation():
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        correlation=LowRankCorrelationConfig(rank=8),
        orthogonalization=None,
    )
    model = SyntheticModel(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        model.save(save_path)

        loaded = SyntheticModel.load_from_disk(save_path)
        assert loaded.cfg == cfg
        assert loaded.correlation_matrix is not None
        assert model.correlation_matrix is not None
        assert torch.allclose(
            loaded.correlation_matrix.correlation_factor,
            model.correlation_matrix.correlation_factor,
        )
        assert torch.allclose(
            loaded.correlation_matrix.correlation_diag,
            model.correlation_matrix.correlation_diag,
        )


def test_synthetic_model_to_device():
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel(cfg)

    # Test moving to same device (cpu)
    model.to("cpu")
    assert model.device == "cpu"
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
    model = SyntheticModel(cfg)
    samples = model.sample(100)
    assert samples.shape == (100, 16)


def test_synthetic_model_from_pretrained(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        model.save(save_path)

        # Mock hf_hub_download to return local files (raise error if file doesn't exist)
        def mock_download(
            repo_id: str | None = None,  # noqa: ARG001
            filename: str | None = None,
            **_kwargs: Any,
        ) -> str:
            assert filename is not None
            local_filename = filename.split("/")[-1]
            local_path = save_path / local_filename
            if not local_path.exists():
                raise EntryNotFoundError(f"File not found: {filename}")
            return str(local_path)

        monkeypatch.setattr(synthetic_model_module, "hf_hub_download", mock_download)

        loaded = SyntheticModel.from_pretrained("test/repo")
        assert loaded.cfg == cfg
        assert torch.allclose(
            loaded.feature_dict.feature_vectors, model.feature_dict.feature_vectors
        )


def test_synthetic_model_from_pretrained_with_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        hierarchy=HierarchyConfig(total_root_nodes=2, branching_factor=2, max_depth=1),
        orthogonalization=None,
        seed=42,
    )
    model = SyntheticModel(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        model.save(save_path)

        # Mock hf_hub_download to return local files (raise error if file doesn't exist)
        def mock_download(
            repo_id: str | None = None,  # noqa: ARG001
            filename: str | None = None,
            **_kwargs: Any,
        ) -> str:
            assert filename is not None
            local_filename = filename.split("/")[-1]
            local_path = save_path / local_filename
            if not local_path.exists():
                raise EntryNotFoundError(f"File not found: {filename}")
            return str(local_path)

        monkeypatch.setattr(synthetic_model_module, "hf_hub_download", mock_download)

        loaded = SyntheticModel.from_pretrained("test/repo", model_path="my_model")
        assert loaded.cfg == cfg
        assert loaded.hierarchy is not None
        assert loaded.hierarchy == model.hierarchy


def test_load_from_source_with_config() -> None:
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel.load_from_source(cfg)
    assert model.cfg == cfg
    assert model.feature_dict.feature_vectors.shape == (32, 16)


def test_load_from_source_with_local_path() -> None:
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        model.save(save_path)

        # Load using load_from_source with local path
        loaded = SyntheticModel.load_from_source(str(save_path))
        assert loaded.cfg == cfg
        assert torch.allclose(
            loaded.feature_dict.feature_vectors, model.feature_dict.feature_vectors
        )


def test_load_from_source_with_relative_path() -> None:
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        model.save(save_path)

        # Test that paths starting with "./" are treated as local
        # We can't easily test this without mocking, but we can test the detection
        from sae_lens.synthetic.synthetic_model import _is_local_path

        assert _is_local_path("./some/path")
        assert _is_local_path("/absolute/path")
        assert _is_local_path("~/home/path")
        assert _is_local_path("../relative/path")
        assert _is_local_path("C:\\windows\\path")
        # These should be detected as HuggingFace
        assert not _is_local_path("username/repo")
        assert not _is_local_path("org/model-name")


def test_load_from_source_with_huggingface(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        model.save(save_path)

        def mock_download(
            repo_id: str | None = None,  # noqa: ARG001
            filename: str | None = None,
            **_kwargs: Any,
        ) -> str:
            assert filename is not None
            local_filename = filename.split("/")[-1]
            local_path = save_path / local_filename
            if not local_path.exists():
                raise EntryNotFoundError(f"File not found: {filename}")
            return str(local_path)

        monkeypatch.setattr(synthetic_model_module, "hf_hub_download", mock_download)

        # Load using HuggingFace format (no colon = repo root)
        loaded = SyntheticModel.load_from_source("username/repo")
        assert loaded.cfg == cfg


def test_load_from_source_with_huggingface_subpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huggingface_hub.utils import EntryNotFoundError

    import sae_lens.synthetic.synthetic_model as synthetic_model_module

    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        model.save(save_path)

        def mock_download(
            repo_id: str | None = None,  # noqa: ARG001
            filename: str | None = None,
            **_kwargs: Any,
        ) -> str:
            assert filename is not None
            local_filename = filename.split("/")[-1]
            local_path = save_path / local_filename
            if not local_path.exists():
                raise EntryNotFoundError(f"File not found: {filename}")
            return str(local_path)

        monkeypatch.setattr(synthetic_model_module, "hf_hub_download", mock_download)

        # Load using HuggingFace format with colon for subpath
        loaded = SyntheticModel.load_from_source("username/repo:subfolder/model")
        assert loaded.cfg == cfg


def test_synthetic_model_seed_produces_identical_models():
    cfg = SyntheticModelConfig(
        num_features=64,
        hidden_dim=32,
        firing_probability=RandomFiringProbabilityConfig(max_prob=0.5, min_prob=0.1),
        std_firing_magnitudes=FoldedNormalMagnitudeConfig(mean=0.0, std=0.5),
        mean_firing_magnitudes=FoldedNormalMagnitudeConfig(mean=1.0, std=0.3),
        hierarchy=HierarchyConfig(
            total_root_nodes=5,
            branching_factor=3,
            max_depth=2,
            mutually_exclusive_portion=0.3,
        ),
        correlation=LowRankCorrelationConfig(rank=8, correlation_scale=0.1),
        orthogonalization=None,
        seed=42,
    )

    model1 = SyntheticModel(cfg)
    model2 = SyntheticModel(cfg)

    # Feature dictionary should be identical
    assert torch.allclose(
        model1.feature_dict.feature_vectors, model2.feature_dict.feature_vectors
    )
    assert torch.allclose(model1.feature_dict.bias, model2.feature_dict.bias)

    # Firing probabilities should be identical
    assert torch.allclose(
        model1.activation_generator.firing_probabilities,
        model2.activation_generator.firing_probabilities,
    )

    # Firing magnitudes should be identical
    assert torch.allclose(
        model1.activation_generator.std_firing_magnitudes,
        model2.activation_generator.std_firing_magnitudes,
    )
    assert torch.allclose(
        model1.activation_generator.mean_firing_magnitudes,
        model2.activation_generator.mean_firing_magnitudes,
    )

    # Hierarchy should be identical
    assert model1.hierarchy is not None
    assert model2.hierarchy is not None
    assert model1.hierarchy == model2.hierarchy

    # Correlation matrix should be identical
    assert model1.correlation_matrix is not None
    assert model2.correlation_matrix is not None
    assert torch.allclose(
        model1.correlation_matrix.correlation_factor,
        model2.correlation_matrix.correlation_factor,
    )
    assert torch.allclose(
        model1.correlation_matrix.correlation_diag,
        model2.correlation_matrix.correlation_diag,
    )


def test_synthetic_model_different_seeds_produce_different_models():
    cfg1 = SyntheticModelConfig(
        num_features=64,
        hidden_dim=32,
        firing_probability=RandomFiringProbabilityConfig(max_prob=0.5, min_prob=0.1),
        std_firing_magnitudes=FoldedNormalMagnitudeConfig(mean=0.0, std=0.5),
        mean_firing_magnitudes=FoldedNormalMagnitudeConfig(mean=1.0, std=0.3),
        hierarchy=HierarchyConfig(
            total_root_nodes=5,
            branching_factor=3,
            max_depth=2,
            mutually_exclusive_portion=0.3,
        ),
        correlation=LowRankCorrelationConfig(rank=8, correlation_scale=0.1),
        orthogonalization=None,
        seed=42,
    )
    cfg2 = SyntheticModelConfig(
        num_features=64,
        hidden_dim=32,
        firing_probability=RandomFiringProbabilityConfig(max_prob=0.5, min_prob=0.1),
        std_firing_magnitudes=FoldedNormalMagnitudeConfig(mean=0.0, std=0.5),
        mean_firing_magnitudes=FoldedNormalMagnitudeConfig(mean=1.0, std=0.3),
        hierarchy=HierarchyConfig(
            total_root_nodes=5,
            branching_factor=3,
            max_depth=2,
            mutually_exclusive_portion=0.3,
        ),
        correlation=LowRankCorrelationConfig(rank=8, correlation_scale=0.1),
        orthogonalization=None,
        seed=123,
    )

    model1 = SyntheticModel(cfg1)
    model2 = SyntheticModel(cfg2)

    # Feature dictionary should be different
    assert not torch.allclose(
        model1.feature_dict.feature_vectors, model2.feature_dict.feature_vectors
    )

    # Firing probabilities should be different
    assert not torch.allclose(
        model1.activation_generator.firing_probabilities,
        model2.activation_generator.firing_probabilities,
    )

    # Firing magnitudes should be different
    assert not torch.allclose(
        model1.activation_generator.std_firing_magnitudes,
        model2.activation_generator.std_firing_magnitudes,
    )
    assert not torch.allclose(
        model1.activation_generator.mean_firing_magnitudes,
        model2.activation_generator.mean_firing_magnitudes,
    )

    # Hierarchy should be different (different feature assignments)
    assert model1.hierarchy is not None
    assert model2.hierarchy is not None
    assert model1.hierarchy != model2.hierarchy

    # Correlation matrix should be different
    assert model1.correlation_matrix is not None
    assert model2.correlation_matrix is not None
    assert not torch.allclose(
        model1.correlation_matrix.correlation_factor,
        model2.correlation_matrix.correlation_factor,
    )


def test_synthetic_model_hierarchy_compensation_integration():
    cfg = SyntheticModelConfig(
        num_features=10,
        hidden_dim=8,
        firing_probability=ConstantFiringProbabilityConfig(probability=0.3),
        hierarchy=HierarchyConfig(
            total_root_nodes=2,
            branching_factor=2,
            max_depth=2,
            compensate_probabilities=True,
        ),
        orthogonalization=None,
        seed=42,
    )
    model = SyntheticModel(cfg)

    n_samples = 100000
    _, features = model.sample_with_features(n_samples)

    assert model.hierarchy is not None
    hierarchy_indices = model.hierarchy.feature_indices_used
    outside_indices = set(range(10)) - hierarchy_indices

    observed_rates = (features > 0).float().mean(dim=0)

    # Standard error for proportion with n=100k, p=0.3: sqrt(0.3*0.7/100000) ≈ 0.00145
    for idx in outside_indices:
        assert observed_rates[idx].item() == pytest.approx(0.3, abs=0.01)

    for root in model.hierarchy.roots:
        if root.feature_index is not None:
            assert observed_rates[root.feature_index].item() == pytest.approx(
                0.3, abs=0.01
            )


def test_synthetic_model_hierarchy_compensation_clamping(
    caplog: pytest.LogCaptureFixture,
):
    cfg = SyntheticModelConfig(
        num_features=10,
        hidden_dim=8,
        firing_probability=ConstantFiringProbabilityConfig(probability=0.9),
        hierarchy=HierarchyConfig(
            total_root_nodes=1,
            branching_factor=2,
            max_depth=3,
            compensate_probabilities=True,
        ),
        orthogonalization=None,
        seed=42,
    )

    with caplog.at_level(logging.WARNING, logger="sae_lens.synthetic.synthetic_model"):
        model = SyntheticModel(cfg)

    assert "clamped" in caplog.text.lower()
    assert model.activation_generator.firing_probabilities.max().item() <= 1.0


def test_synthetic_model_hierarchy_compensation_disabled_by_default():
    cfg = SyntheticModelConfig(
        num_features=10,
        hidden_dim=8,
        firing_probability=ConstantFiringProbabilityConfig(probability=0.5),
        hierarchy=HierarchyConfig(
            total_root_nodes=2,
            branching_factor=2,
            max_depth=2,
        ),
        orthogonalization=None,
        seed=42,
    )
    model = SyntheticModel(cfg)

    n_samples = 10000
    _, features = model.sample_with_features(n_samples)

    assert model.hierarchy is not None

    for root in model.hierarchy.roots:
        if root.feature_index is not None:
            for child in root.children:
                if child.feature_index is not None:
                    child_rate = (features[:, child.feature_index] > 0).float().mean()
                    assert child_rate.item() < 0.35
