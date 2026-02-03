"""
Runner for training SAEs on synthetic data.

This module provides SyntheticSAERunner and SyntheticSAERunnerConfig for
training SAEs on synthetic data with full support for all SAE architectures.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
import wandb
from safetensors.torch import save_file

from sae_lens import __version__, logger
from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.constants import SPARSITY_FILENAME
from sae_lens.registry import get_sae_training_class
from sae_lens.saes.sae import TrainingSAE, TrainingSAEConfig
from sae_lens.synthetic.evals import SyntheticDataEvalResult, eval_sae_on_synthetic_data
from sae_lens.synthetic.synthetic_model import SyntheticModel, SyntheticModelConfig
from sae_lens.synthetic.training import SyntheticActivationIterator
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.types import DataProvider

T_TRAINING_SAE_CONFIG = TypeVar("T_TRAINING_SAE_CONFIG", bound=TrainingSAEConfig)

RUNNER_CONFIG_FILENAME = "runner_config.json"


@dataclass
class SyntheticSAERunnerConfig(Generic[T_TRAINING_SAE_CONFIG]):
    """
    Configuration for training an SAE on synthetic data.

    Combines synthetic model config with SAE training config.

    Attributes:
        synthetic_model: Source for the synthetic data generator. Can be:

            - SyntheticModelConfig: Create a new model from config
            - Local path (str): Load from disk. Detected if path exists or starts
              with "/", "./", "~"
            - HuggingFace (str): Load from HuggingFace Hub. Format is "repo_id"
              or "repo_id:model_path" for models in subfolders

        sae: Config for the SAE being trained.
        training_samples: Total training samples (activations) to generate.
        batch_size: Batch size for training.
        lr: Learning rate.
        lr_warm_up_steps: Learning rate warmup steps.
        lr_decay_steps: Learning rate decay steps.
        lr_scheduler_name: Name of LR scheduler.
        adam_beta1: Adam beta1.
        adam_beta2: Adam beta2.
        device: Device for training.
        autocast_sae: Whether to autocast SAE to bfloat16.
        autocast_data: Whether to autocast data generation to bfloat16.
        n_checkpoints: Number of checkpoints during training.
        checkpoint_path: Path for checkpoints.
        output_path: Path for final output.
        eval_frequency: Evaluate MCC every N steps (0 = no eval).
        eval_samples: Number of samples for evaluation.
        logger: Logging config.
    """

    synthetic_model: SyntheticModelConfig | str  # Config or path to saved model
    sae: T_TRAINING_SAE_CONFIG

    # Training params
    training_samples: int = 100_000_000
    batch_size: int = 1024
    lr: float = 3e-4
    lr_warm_up_steps: int = 0
    lr_decay_steps: int = 0
    lr_scheduler_name: str = "constant"
    lr_end: float | None = None
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    n_restart_cycles: int = 1

    # Device/performance
    device: str = "cpu"
    autocast_sae: bool = False
    autocast_data: bool = False

    # Checkpoints/outputs
    n_checkpoints: int = 0
    checkpoint_path: str | None = "checkpoints"
    save_final_checkpoint: bool = False
    output_path: str | None = "output"
    save_synthetic_model: bool = False  # Save synthetic model with output

    # Evaluation
    eval_frequency: int = 0  # eval every N training steps (0 = disabled)
    eval_samples: int = 500_000
    run_final_eval: bool = True

    # Misc
    dead_feature_window: int = 1000
    feature_sampling_window: int = 2000
    sae_lens_version: str = field(default_factory=lambda: __version__)

    logger: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        if self.lr_end is None:
            self.lr_end = self.lr / 10

        # Set default run name
        if self.logger.run_name is None:
            arch = self.sae.architecture()
            d_sae = self.sae.d_sae
            self.logger.run_name = f"synthetic-{arch}-{d_sae}-LR-{self.lr}"

    @property
    def total_training_steps(self) -> int:
        return self.training_samples // self.batch_size

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary."""
        d = asdict(self)

        # Handle synthetic_model (may be config or path string)
        sm = self.synthetic_model
        if isinstance(sm, SyntheticModelConfig):
            d["synthetic_model"] = sm.to_dict()
        else:
            d["synthetic_model"] = str(sm)

        # Handle nested configs with their own to_dict methods
        d["sae"] = self.sae.to_dict()
        d["logger"] = asdict(self.logger)

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SyntheticSAERunnerConfig[Any]":
        """Deserialize config from dictionary."""
        if "sae" not in d:
            raise ValueError("sae field is required in the config dictionary")
        if "architecture" not in d["sae"]:
            raise ValueError("architecture field is required in the sae dictionary")
        if "synthetic_model" not in d:
            raise ValueError(
                "synthetic_model field is required in the config dictionary"
            )

        # Parse synthetic_model
        sm = d["synthetic_model"]
        if isinstance(sm, dict):
            synthetic_model: SyntheticModelConfig | str = (
                SyntheticModelConfig.from_dict(sm)
            )
        else:
            synthetic_model = str(sm)  # Path

        # Parse SAE config
        sae_cfg_class = get_sae_training_class(d["sae"]["architecture"])[1]
        sae_cfg = sae_cfg_class.from_dict(d["sae"])

        # Parse logger
        logger_cfg = LoggingConfig(**d.get("logger", {}))

        updated_dict: dict[str, Any] = {
            **d,
            "synthetic_model": synthetic_model,
            "sae": sae_cfg,
            "logger": logger_cfg,
        }
        return cls(**updated_dict)

    def to_sae_trainer_config(self) -> SAETrainerConfig:
        """Convert to SAETrainerConfig for use with SAETrainer."""

        return SAETrainerConfig(
            n_checkpoints=self.n_checkpoints,
            checkpoint_path=self.checkpoint_path,
            save_final_checkpoint=self.save_final_checkpoint,
            total_training_samples=self.training_samples,
            device=self.device,
            autocast=self.autocast_sae,
            lr=self.lr,
            lr_end=self.lr_end,
            lr_scheduler_name=self.lr_scheduler_name,
            lr_warm_up_steps=self.lr_warm_up_steps,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            lr_decay_steps=self.lr_decay_steps,
            n_restart_cycles=self.n_restart_cycles,
            train_batch_size_samples=self.batch_size,
            dead_feature_window=self.dead_feature_window,
            feature_sampling_window=self.feature_sampling_window,
            logger=self.logger,
        )


@dataclass
class SyntheticSAERunnerResult:
    """Result from SyntheticSAERunner."""

    sae: TrainingSAE[Any]
    synthetic_model: SyntheticModel
    final_eval: SyntheticDataEvalResult | None
    output_path: Path | None


class SyntheticSAERunner(Generic[T_TRAINING_SAE_CONFIG]):
    """
    Runner for training SAEs on synthetic data.

    Similar to LanguageModelSAETrainingRunner but for synthetic data.
    Supports all SAE architectures via the registry.
    """

    cfg: SyntheticSAERunnerConfig[T_TRAINING_SAE_CONFIG]
    synthetic_model: SyntheticModel
    sae: TrainingSAE[T_TRAINING_SAE_CONFIG]

    def __init__(
        self,
        cfg: SyntheticSAERunnerConfig[T_TRAINING_SAE_CONFIG],
        override_synthetic_model: SyntheticModel | None = None,
        override_sae: TrainingSAE[T_TRAINING_SAE_CONFIG] | None = None,
    ):
        """
        Initialize the runner.

        Args:
            cfg: Runner configuration
            override_synthetic_model: Use this synthetic model instead of creating from config
            override_sae: Use this SAE instead of creating from config
        """
        self.cfg = cfg

        # Create or load synthetic model
        if override_synthetic_model is not None:
            self.synthetic_model = override_synthetic_model
        else:
            self.synthetic_model = SyntheticModel.load_from_source(
                cfg.synthetic_model,
                device=cfg.device,
            )

        # Ensure SAE dimensions match synthetic model
        expected_d_in = self.synthetic_model.cfg.hidden_dim
        if cfg.sae.d_in != expected_d_in:
            logger.warning(
                f"SAE d_in ({cfg.sae.d_in}) doesn't match synthetic model "
                f"hidden_dim ({expected_d_in}). Updating SAE config."
            )
            cfg.sae.d_in = expected_d_in

        # Create or use provided SAE
        if override_sae is not None:
            self.sae = override_sae
        else:
            sae_class, _ = get_sae_training_class(cfg.sae.architecture())
            self.sae = sae_class(cfg.sae)

        self.sae.to(cfg.device)

    def run(self) -> SyntheticSAERunnerResult:
        """
        Run the training loop.

        Returns:
            SyntheticSAERunnerResult with trained SAE and evaluation
        """
        # Initialize wandb if configured
        if self.cfg.logger.log_to_wandb:
            wandb.init(
                project=self.cfg.logger.wandb_project,
                entity=self.cfg.logger.wandb_entity,
                config=self.cfg.to_dict(),
                name=self.cfg.logger.run_name,
            )

        # Create data iterator
        data_iterator = SyntheticActivationIterator(
            feature_dict=self.synthetic_model.feature_dict,
            activations_generator=self.synthetic_model.activation_generator,
            batch_size=self.cfg.batch_size,
            autocast=self.cfg.autocast_data,
        )

        # Create evaluator if eval_frequency > 0
        evaluator = None
        if self.cfg.eval_frequency > 0:
            evaluator = self._create_evaluator()

        # Create trainer
        trainer = SAETrainer(
            cfg=self.cfg.to_sae_trainer_config(),
            sae=self.sae,
            data_provider=data_iterator,
            evaluator=evaluator,
            save_checkpoint_fn=self._save_checkpoint,
        )

        # Train
        logger.info(f"Starting training for {self.cfg.training_samples:,} samples")
        sae = trainer.fit()

        # Final evaluation
        final_eval = None
        if self.cfg.eval_samples > 0 and self.cfg.run_final_eval:
            logger.info("Running final evaluation...")
            final_eval = eval_sae_on_synthetic_data(
                sae=sae,
                feature_dict=self.synthetic_model.feature_dict,
                activations_generator=self.synthetic_model.activation_generator,
                num_samples=self.cfg.eval_samples,
                batch_size=self.cfg.batch_size,
            )
            if self.cfg.logger.log_to_wandb:
                wandb.log(final_eval.to_log_dict(prefix="final/"))

        # Save outputs
        output_path = None
        if self.cfg.output_path is not None:
            output_path = Path(self.cfg.output_path)
            self._save_outputs(
                output_path, sae, trainer.log_feature_sparsity, final_eval
            )

        if self.cfg.logger.log_to_wandb:
            wandb.finish()

        return SyntheticSAERunnerResult(
            sae=sae,
            synthetic_model=self.synthetic_model,
            final_eval=final_eval,
            output_path=output_path,
        )

    def _create_evaluator(self) -> Any:
        """Create evaluator function for periodic MCC evaluation."""

        def evaluator(
            sae: TrainingSAE[Any],
            data_provider: DataProvider,  # noqa: ARG001
            activation_scaler: ActivationScaler,
        ) -> dict[str, Any]:
            result = eval_sae_on_synthetic_data(
                sae=sae,
                feature_dict=self.synthetic_model.feature_dict,
                activations_generator=self.synthetic_model.activation_generator,
                num_samples=self.cfg.eval_samples,
                batch_size=self.cfg.batch_size,
                activation_scaler=activation_scaler,
            )
            return result.to_log_dict(prefix="synthetic/")

        return evaluator

    def _save_checkpoint(self, checkpoint_path: Path | None) -> None:
        """Save checkpoint (called by trainer)."""
        if checkpoint_path is None:
            return

        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save runner config
        with open(checkpoint_path / RUNNER_CONFIG_FILENAME, "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)

    def _save_outputs(
        self,
        output_path: Path,
        sae: TrainingSAE[Any],
        log_feature_sparsity: torch.Tensor | None,
        eval_stats: SyntheticDataEvalResult | None,
    ) -> None:
        """Save final outputs."""
        output_path.mkdir(parents=True, exist_ok=True)

        # Save SAE
        sae.save_inference_model(str(output_path))

        # Save sparsity
        if log_feature_sparsity is not None:
            save_file(
                {"sparsity": log_feature_sparsity}, output_path / SPARSITY_FILENAME
            )

        # Save runner config
        with open(output_path / RUNNER_CONFIG_FILENAME, "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)

        # Save synthetic model if configured
        if self.cfg.save_synthetic_model:
            synthetic_model_path = output_path / "synthetic_model"
            self.synthetic_model.save(synthetic_model_path)

        if eval_stats is not None:
            with open(output_path / "eval_stats.json", "w") as f:
                json.dump(eval_stats.to_dict(), f, indent=2)

        logger.info(f"Saved outputs to {output_path}")
