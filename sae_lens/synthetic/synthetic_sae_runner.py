"""
Runner for training SAEs on synthetic data.

This module provides SyntheticSAERunner and SyntheticSAERunnerConfig for
training SAEs on synthetic data with full support for all SAE architectures.
"""

import json
from dataclasses import dataclass, field
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
        synthetic_model: Config for the synthetic data generator, or path to
            a pre-saved SyntheticModel directory.
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
    training_samples: int = 10_000_000
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
    output_path: str | None = "output"
    save_synthetic_model: bool = True  # Save synthetic model with output

    # Evaluation
    eval_frequency: int = 0  # MCC eval every N training steps (0 = disabled)
    eval_samples: int = 100_000

    # Misc
    dead_feature_window: int = 1000
    feature_sampling_window: int = 2000

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
        sm = self.synthetic_model
        if isinstance(sm, SyntheticModelConfig):
            sm_dict: dict[str, Any] | str = sm.to_dict()
        else:
            sm_dict = str(sm)  # Path string

        return {
            "synthetic_model": sm_dict,
            "sae": self.sae.to_dict(),
            "training_samples": self.training_samples,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "lr_warm_up_steps": self.lr_warm_up_steps,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_scheduler_name": self.lr_scheduler_name,
            "lr_end": self.lr_end,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "n_restart_cycles": self.n_restart_cycles,
            "device": self.device,
            "autocast_sae": self.autocast_sae,
            "autocast_data": self.autocast_data,
            "n_checkpoints": self.n_checkpoints,
            "checkpoint_path": self.checkpoint_path,
            "output_path": self.output_path,
            "save_synthetic_model": self.save_synthetic_model,
            "eval_frequency": self.eval_frequency,
            "eval_samples": self.eval_samples,
            "dead_feature_window": self.dead_feature_window,
            "feature_sampling_window": self.feature_sampling_window,
            "logger": {
                "log_to_wandb": self.logger.log_to_wandb,
                "wandb_project": self.logger.wandb_project,
                "run_name": self.logger.run_name,
                "wandb_log_frequency": self.logger.wandb_log_frequency,
                "wandb_entity": self.logger.wandb_entity,
            },
            "sae_lens_version": __version__,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SyntheticSAERunnerConfig[Any]":
        """Deserialize config from dictionary."""
        # Parse synthetic_model
        sm = d["synthetic_model"]
        if isinstance(sm, dict):
            synthetic_model: SyntheticModelConfig | str = (
                SyntheticModelConfig.from_dict(sm)
            )
        else:
            synthetic_model = str(sm)  # Path

        # Parse SAE config
        sae_dict = d["sae"]
        sae_cfg_class = get_sae_training_class(sae_dict["architecture"])[1]
        sae = sae_cfg_class.from_dict(sae_dict)

        # Parse logger
        logger_dict = d.get("logger", {})
        logger_cfg = LoggingConfig(
            log_to_wandb=logger_dict.get("log_to_wandb", True),
            wandb_project=logger_dict.get("wandb_project", "sae_lens_training"),
            run_name=logger_dict.get("run_name"),
            wandb_log_frequency=logger_dict.get("wandb_log_frequency", 10),
            wandb_entity=logger_dict.get("wandb_entity"),
        )

        return cls(
            synthetic_model=synthetic_model,
            sae=sae,
            training_samples=d.get("training_samples", 10_000_000),
            batch_size=d.get("batch_size", 1024),
            lr=d.get("lr", 3e-4),
            lr_warm_up_steps=d.get("lr_warm_up_steps", 0),
            lr_decay_steps=d.get("lr_decay_steps", 0),
            lr_scheduler_name=d.get("lr_scheduler_name", "constant"),
            lr_end=d.get("lr_end"),
            adam_beta1=d.get("adam_beta1", 0.9),
            adam_beta2=d.get("adam_beta2", 0.999),
            n_restart_cycles=d.get("n_restart_cycles", 1),
            device=d.get("device", "cpu"),
            autocast_sae=d.get("autocast_sae", False),
            autocast_data=d.get("autocast_data", False),
            n_checkpoints=d.get("n_checkpoints", 0),
            checkpoint_path=d.get("checkpoint_path", "checkpoints"),
            output_path=d.get("output_path", "output"),
            save_synthetic_model=d.get("save_synthetic_model", True),
            eval_frequency=d.get("eval_frequency", 0),
            eval_samples=d.get("eval_samples", 100_000),
            dead_feature_window=d.get("dead_feature_window", 1000),
            feature_sampling_window=d.get("feature_sampling_window", 2000),
            logger=logger_cfg,
        )

    def to_sae_trainer_config(self) -> SAETrainerConfig:
        """Convert to SAETrainerConfig for use with SAETrainer."""
        # Calculate eval frequency in terms of wandb logs
        # eval_every_n_wandb_logs controls when evals run
        if self.eval_frequency > 0:
            wandb_logs_per_eval = max(
                1, self.eval_frequency // self.logger.wandb_log_frequency
            )
        else:
            # Very large number to effectively disable evals
            wandb_logs_per_eval = 2**31 - 1

        logger_cfg = LoggingConfig(
            log_to_wandb=self.logger.log_to_wandb,
            wandb_project=self.logger.wandb_project,
            run_name=self.logger.run_name,
            wandb_log_frequency=self.logger.wandb_log_frequency,
            wandb_entity=self.logger.wandb_entity,
            eval_every_n_wandb_logs=wandb_logs_per_eval,
        )

        return SAETrainerConfig(
            n_checkpoints=self.n_checkpoints,
            checkpoint_path=self.checkpoint_path,
            save_final_checkpoint=False,
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
            logger=logger_cfg,
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
        elif isinstance(cfg.synthetic_model, str):
            # Load from path
            self.synthetic_model = SyntheticModel.load(
                cfg.synthetic_model, device=cfg.device
            )
        else:
            # Create from config
            model_cfg = cfg.synthetic_model
            model_cfg.device = cfg.device
            self.synthetic_model = SyntheticModel.from_config(model_cfg)

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
        if self.cfg.eval_samples > 0:
            logger.info("Running final evaluation...")
            final_eval = eval_sae_on_synthetic_data(
                sae=sae,
                feature_dict=self.synthetic_model.feature_dict,
                activations_generator=self.synthetic_model.activation_generator,
                num_samples=self.cfg.eval_samples,
            )
            logger.info(f"Final MCC: {final_eval.mcc:.4f}")

            if self.cfg.logger.log_to_wandb:
                wandb.log(
                    {
                        "final/mcc": final_eval.mcc,
                        "final/sae_l0": final_eval.sae_l0,
                        "final/true_l0": final_eval.true_l0,
                        "final/dead_latents": final_eval.dead_latents,
                        "final/shrinkage": final_eval.shrinkage,
                    }
                )

        # Save outputs
        output_path = None
        if self.cfg.output_path is not None:
            output_path = Path(self.cfg.output_path)
            self._save_outputs(output_path, sae, trainer.log_feature_sparsity)

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
            activation_scaler: ActivationScaler,  # noqa: ARG001
        ) -> dict[str, Any]:
            result = eval_sae_on_synthetic_data(
                sae=sae,
                feature_dict=self.synthetic_model.feature_dict,
                activations_generator=self.synthetic_model.activation_generator,
                num_samples=self.cfg.eval_samples,
            )
            return {
                "synthetic/mcc": result.mcc,
                "synthetic/sae_l0": result.sae_l0,
                "synthetic/true_l0": result.true_l0,
                "synthetic/dead_latents": result.dead_latents,
                "synthetic/shrinkage": result.shrinkage,
            }

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

        logger.info(f"Saved outputs to {output_path}")
