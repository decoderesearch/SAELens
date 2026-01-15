import tempfile
from pathlib import Path

from sae_lens.config import LoggingConfig
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig
from sae_lens.synthetic import (
    HierarchyConfig,
    SyntheticModel,
    SyntheticModelConfig,
    SyntheticSAERunner,
    SyntheticSAERunnerConfig,
)


def test_runner_config_default_values():
    model_cfg = SyntheticModelConfig(num_features=32, hidden_dim=16)
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32)
    runner_cfg = SyntheticSAERunnerConfig(synthetic_model=model_cfg, sae=sae_cfg)

    assert runner_cfg.training_samples == 10_000_000
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
    model = SyntheticModel.from_config(model_cfg)

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


def test_runner_saves_outputs():
    model_cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    sae_cfg = StandardTrainingSAEConfig(d_in=16, d_sae=32, l1_coefficient=0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output"
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
