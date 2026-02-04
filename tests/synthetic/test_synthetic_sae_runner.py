import json
from pathlib import Path

import pytest
import torch

from sae_lens.config import LoggingConfig
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig
from sae_lens.synthetic import (
    ConstantFiringProbabilityConfig,
    HierarchyConfig,
    SyntheticModel,
    SyntheticModelConfig,
    SyntheticSAERunner,
    SyntheticSAERunnerConfig,
)
from sae_lens.synthetic.synthetic_sae_runner import RUNNER_CONFIG_FILENAME
from sae_lens.training.activation_scaler import ActivationScaler


def test_runner_config_default_values():
    model_cfg = SyntheticModelConfig(num_features=32, hidden_dim=16)
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32)
    runner_cfg = SyntheticSAERunnerConfig(synthetic_model=model_cfg, sae=sae_cfg)

    assert runner_cfg.training_samples == 100_000_000
    assert runner_cfg.batch_size == 1024
    assert runner_cfg.lr == 3e-4
    assert runner_cfg.device == "cpu"


def test_runner_config_to_dict_from_dict_roundtrip():
    model_cfg = SyntheticModelConfig(num_features=32, hidden_dim=16)
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32, l1_coefficient=0.01)
    original = SyntheticSAERunnerConfig(
        synthetic_model=model_cfg,
        sae=sae_cfg,
        training_samples=1000,
        batch_size=100,
        lr=1e-3,
        device="cpu",
    )
    d = original.to_dict()
    restored = SyntheticSAERunnerConfig.from_dict(d)

    assert restored.training_samples == original.training_samples
    assert restored.batch_size == original.batch_size
    assert restored.lr == original.lr


def test_runner_config_with_path_synthetic_model():
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32)
    runner_cfg = SyntheticSAERunnerConfig(
        synthetic_model="/some/path/to/model", sae=sae_cfg
    )
    d = runner_cfg.to_dict()
    assert d["synthetic_model"] == "/some/path/to/model"


def test_runner_config_total_training_steps():
    model_cfg = SyntheticModelConfig(num_features=32, hidden_dim=16)
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32)
    runner_cfg = SyntheticSAERunnerConfig(
        synthetic_model=model_cfg,
        sae=sae_cfg,
        training_samples=10000,
        batch_size=100,
    )
    assert runner_cfg.total_training_steps == 100


def test_runner_initializes_with_config():
    model_cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32)
    runner_cfg = SyntheticSAERunnerConfig(
        synthetic_model=model_cfg,
        sae=sae_cfg,
        training_samples=100,
        batch_size=10,
        logger=LoggingConfig(log_to_wandb=False),
    )
    runner = SyntheticSAERunner(runner_cfg)

    assert runner.synthetic_model is not None
    assert runner.sae is not None
    assert runner.synthetic_model.cfg.num_features == 32


def test_runner_initializes_with_override_synthetic_model():
    model_cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel(model_cfg)

    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32)
    runner_cfg = SyntheticSAERunnerConfig(
        synthetic_model=model_cfg,  # Will be ignored
        sae=sae_cfg,
        logger=LoggingConfig(log_to_wandb=False),
    )
    runner = SyntheticSAERunner(runner_cfg, override_synthetic_model=model)

    assert runner.synthetic_model is model


def test_runner_run_completes_without_error():
    model_cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        hierarchy=HierarchyConfig(total_root_nodes=5, max_depth=2),
        orthogonalization=None,
    )
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32, l1_coefficient=0.01)
    runner_cfg = SyntheticSAERunnerConfig(
        synthetic_model=model_cfg,
        sae=sae_cfg,
        training_samples=100,
        batch_size=10,
        eval_samples=50,
        output_path=None,  # Don't save output
        logger=LoggingConfig(log_to_wandb=False),
    )
    runner = SyntheticSAERunner(runner_cfg)
    result = runner.run()

    assert result.sae is not None
    assert result.synthetic_model is not None
    assert result.final_eval is not None
    assert result.final_eval.mcc >= 0.0
    assert result.final_eval.mcc <= 1.0


def test_runner_saves_outputs(tmp_path: Path):
    model_cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32, l1_coefficient=0.01)

    output_path = tmp_path / "output"
    runner_cfg = SyntheticSAERunnerConfig(
        synthetic_model=model_cfg,
        sae=sae_cfg,
        training_samples=100,
        batch_size=10,
        eval_samples=50,
        output_path=str(output_path),
        save_synthetic_model=True,
        logger=LoggingConfig(log_to_wandb=False),
    )
    runner = SyntheticSAERunner(runner_cfg)
    result = runner.run()

    # Check outputs exist
    assert output_path.exists()
    assert (output_path / "sae_weights.safetensors").exists()
    assert (output_path / "cfg.json").exists()
    assert (output_path / "runner_config.json").exists()
    assert (output_path / "synthetic_model").exists()
    assert result.output_path == output_path


def test_runner_updates_sae_d_in_if_mismatched():
    model_cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    # Intentionally wrong d_in
    sae_cfg = StandardTrainingSAEConfig(d_in=999, d_sae=32)
    runner_cfg = SyntheticSAERunnerConfig(
        synthetic_model=model_cfg,
        sae=sae_cfg,
        logger=LoggingConfig(log_to_wandb=False),
    )
    runner = SyntheticSAERunner(runner_cfg)

    # Should have been corrected
    assert runner.sae.cfg.d_in == 16


def test_runner_create_evaluator_returns_callable():
    model_cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32)
    runner_cfg = SyntheticSAERunnerConfig(
        synthetic_model=model_cfg,
        sae=sae_cfg,
        eval_frequency=10,
        eval_samples=100,
        logger=LoggingConfig(log_to_wandb=False),
    )
    runner = SyntheticSAERunner(runner_cfg)
    evaluator = runner._create_evaluator()

    assert callable(evaluator)


def test_runner_create_evaluator_returns_metrics():
    model_cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32)
    runner_cfg = SyntheticSAERunnerConfig(
        synthetic_model=model_cfg,
        sae=sae_cfg,
        eval_frequency=10,
        eval_samples=100,
        logger=LoggingConfig(log_to_wandb=False),
    )
    runner = SyntheticSAERunner(runner_cfg)
    evaluator = runner._create_evaluator()

    # Call the evaluator
    activation_scaler = ActivationScaler(scaling_factor=None)
    result = evaluator(runner.sae, None, activation_scaler)

    # Should return a dict with synthetic/ prefixed keys
    assert isinstance(result, dict)
    assert "synthetic/mcc" in result
    assert "synthetic/explained_variance" in result
    assert 0.0 <= result["synthetic/mcc"] <= 1.0


def test_runner_save_checkpoint_creates_config_file(tmp_path: Path):
    model_cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32)
    runner_cfg = SyntheticSAERunnerConfig(
        synthetic_model=model_cfg,
        sae=sae_cfg,
        logger=LoggingConfig(log_to_wandb=False),
    )
    runner = SyntheticSAERunner(runner_cfg)

    checkpoint_path = tmp_path / "checkpoint_0"
    runner._save_checkpoint(checkpoint_path)

    # Check that runner config was saved
    config_file = checkpoint_path / RUNNER_CONFIG_FILENAME
    assert config_file.exists()

    # Verify it's valid JSON with expected content
    with open(config_file) as f:
        saved_config = json.load(f)
    assert saved_config["batch_size"] == runner_cfg.batch_size
    assert saved_config["lr"] == runner_cfg.lr


def test_runner_save_checkpoint_handles_none_path():
    model_cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32)
    runner_cfg = SyntheticSAERunnerConfig(
        synthetic_model=model_cfg,
        sae=sae_cfg,
        logger=LoggingConfig(log_to_wandb=False),
    )
    runner = SyntheticSAERunner(runner_cfg)

    # Should not raise
    runner._save_checkpoint(None)


def test_runner_end_to_end_with_checkpoints(tmp_path: Path):
    model_cfg = SyntheticModelConfig(
        num_features=16,
        hidden_dim=8,
        firing_probability=ConstantFiringProbabilityConfig(probability=0.3),
        orthogonalization=None,
        seed=42,
    )
    sae_cfg = StandardTrainingSAEConfig(d_in=8, d_sae=16, l1_coefficient=0.01)

    checkpoint_path = tmp_path / "checkpoints"
    output_path = tmp_path / "output"

    runner_cfg = SyntheticSAERunnerConfig(
        synthetic_model=model_cfg,
        sae=sae_cfg,
        training_samples=200,  # Very small for fast test
        batch_size=20,
        n_checkpoints=2,  # Create 2 checkpoints during training
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path),
        eval_frequency=5,  # Evaluate every 5 steps
        eval_samples=50,
        logger=LoggingConfig(log_to_wandb=False),
    )
    runner = SyntheticSAERunner(runner_cfg)
    result = runner.run()

    # Training should complete
    assert result.sae is not None
    assert result.final_eval is not None

    # Output should be saved
    assert output_path.exists()
    assert (output_path / "sae_weights.safetensors").exists()
    assert (output_path / "cfg.json").exists()
    assert (output_path / RUNNER_CONFIG_FILENAME).exists()

    # Verify checkpoint config contains expected fields
    runner_config_path = output_path / RUNNER_CONFIG_FILENAME
    with open(runner_config_path) as f:
        saved_runner_cfg = json.load(f)
    assert saved_runner_cfg["training_samples"] == 200
    assert saved_runner_cfg["batch_size"] == 20

    # Verify eval stats were saved and match the returned result
    eval_stats_path = output_path / "eval_stats.json"
    assert eval_stats_path.exists()

    with open(eval_stats_path) as f:
        saved_eval_stats = json.load(f)

    # Compare saved eval stats with the returned final_eval
    expected_eval_dict = result.final_eval.to_dict()
    assert saved_eval_stats["mcc"] == expected_eval_dict["mcc"]
    assert (
        saved_eval_stats["explained_variance"]
        == expected_eval_dict["explained_variance"]
    )
    assert saved_eval_stats["sae_l0"] == expected_eval_dict["sae_l0"]
    assert saved_eval_stats["true_l0"] == expected_eval_dict["true_l0"]
    assert saved_eval_stats["dead_latents"] == expected_eval_dict["dead_latents"]
    assert saved_eval_stats["shrinkage"] == expected_eval_dict["shrinkage"]
    assert saved_eval_stats["uniqueness"] == expected_eval_dict["uniqueness"]
    assert saved_eval_stats["classification"] == expected_eval_dict["classification"]


def test_runner_config_from_dict_with_string_synthetic_model():
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32)
    original = SyntheticSAERunnerConfig(
        synthetic_model="/path/to/model",
        sae=sae_cfg,
    )
    d = original.to_dict()

    # Should serialize as string
    assert d["synthetic_model"] == "/path/to/model"

    # Should deserialize back as string
    restored = SyntheticSAERunnerConfig.from_dict(d)
    assert restored.synthetic_model == "/path/to/model"
    assert isinstance(restored.synthetic_model, str)


def test_runner_config_from_dict_with_config_synthetic_model():
    model_cfg = SyntheticModelConfig(num_features=32, hidden_dim=16)
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32)
    original = SyntheticSAERunnerConfig(
        synthetic_model=model_cfg,
        sae=sae_cfg,
    )
    d = original.to_dict()

    # Should serialize as dict
    assert isinstance(d["synthetic_model"], dict)
    assert d["synthetic_model"]["num_features"] == 32

    # Should deserialize back as SyntheticModelConfig
    restored = SyntheticSAERunnerConfig.from_dict(d)
    assert isinstance(restored.synthetic_model, SyntheticModelConfig)
    assert restored.synthetic_model.num_features == 32


def test_runner_config_from_dict_validates_required_fields():
    # Missing sae field
    with pytest.raises(ValueError, match="sae field is required"):
        SyntheticSAERunnerConfig.from_dict({"synthetic_model": {}})

    # Missing architecture in sae
    with pytest.raises(ValueError, match="architecture field is required"):
        SyntheticSAERunnerConfig.from_dict(
            {
                "synthetic_model": {"num_features": 32, "hidden_dim": 16},
                "sae": {"d_in": 16, "d_sae": 32},
            }
        )

    # Missing synthetic_model field
    with pytest.raises(ValueError, match="synthetic_model field is required"):
        SyntheticSAERunnerConfig.from_dict(
            {
                "sae": {"architecture": "standard", "d_in": 16, "d_sae": 32},
            }
        )


def test_runner_loads_from_disk_path(tmp_path: Path):
    model_cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
        seed=42,
    )
    model = SyntheticModel(model_cfg)

    model_path = tmp_path / "model"
    model.save(model_path)

    # Create runner with path to saved model
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32)
    runner_cfg = SyntheticSAERunnerConfig(
        synthetic_model=str(model_path),
        sae=sae_cfg,
        logger=LoggingConfig(log_to_wandb=False),
    )
    runner = SyntheticSAERunner(runner_cfg)

    # Should have loaded the model from disk
    assert runner.synthetic_model.cfg.num_features == 32
    assert torch.allclose(
        runner.synthetic_model.feature_dict.feature_vectors,
        model.feature_dict.feature_vectors,
    )
