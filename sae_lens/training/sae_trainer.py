import contextlib
import math
from pathlib import Path
from typing import Any, Callable, Generic, Protocol

import torch
import wandb
from safetensors.torch import save_file
from torch.optim import Adam
from tqdm.auto import tqdm

from sae_lens import __version__
from sae_lens.config import SAETrainerConfig
from sae_lens.constants import (
    ACTIVATION_SCALER_CFG_FILENAME,
    SPARSITY_FILENAME,
    TRAINER_STATE_FILENAME,
)
from sae_lens.saes.sae import (
    T_TRAINING_SAE,
    T_TRAINING_SAE_CONFIG,
    TrainCoefficientConfig,
    TrainingSAE,
    TrainStepInput,
    TrainStepOutput,
)
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.optim import CoefficientScheduler, get_lr_scheduler
from sae_lens.training.types import DataProvider
from sae_lens.util import path_or_tmp_dir


def _log_feature_sparsity(
    feature_sparsity: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    return torch.log10(feature_sparsity + eps).detach().cpu()


def _update_sae_lens_training_version(sae: TrainingSAE[Any]) -> None:
    """
    Make sure we record the version of SAELens used for the training run
    """
    sae.cfg.sae_lens_training_version = str(__version__)


class SaveCheckpointFn(Protocol):
    def __call__(
        self,
        checkpoint_path: Path | None,
    ) -> None: ...


Evaluator = Callable[[T_TRAINING_SAE, DataProvider, ActivationScaler], dict[str, Any]]


class SAETrainer(Generic[T_TRAINING_SAE, T_TRAINING_SAE_CONFIG]):
    """
    Trainer for Sparse Autoencoder (SAE) models.
    """

    data_provider: DataProvider
    activation_scaler: ActivationScaler
    evaluator: Evaluator[T_TRAINING_SAE] | None
    coefficient_schedulers: dict[str, CoefficientScheduler]

    def __init__(
        self,
        cfg: SAETrainerConfig,
        sae: T_TRAINING_SAE,
        data_provider: DataProvider,
        evaluator: Evaluator[T_TRAINING_SAE] | None = None,
        save_checkpoint_fn: SaveCheckpointFn | None = None,
    ) -> None:
        self.sae = sae
        self.data_provider = data_provider
        self.evaluator = evaluator
        self.activation_scaler = ActivationScaler()
        self.save_checkpoint_fn = save_checkpoint_fn
        self.cfg = cfg

        self.n_training_steps: int = 0
        self.n_training_samples: int = 0
        self.started_fine_tuning: bool = False

        _update_sae_lens_training_version(self.sae)

        self.checkpoint_thresholds = []
        if self.cfg.n_checkpoints > 0:
            self.checkpoint_thresholds = list(
                range(
                    0,
                    cfg.total_training_samples,
                    math.ceil(
                        cfg.total_training_samples / (self.cfg.n_checkpoints + 1)
                    ),
                )
            )[1:]

        self.act_freq_scores = torch.zeros(sae.cfg.d_sae, device=cfg.device)
        self.n_forward_passes_since_fired = torch.zeros(
            sae.cfg.d_sae, device=cfg.device
        )
        self.n_frac_active_samples = 0

        self.optimizer = Adam(
            sae.parameters(),
            lr=cfg.lr,
            betas=(
                cfg.adam_beta1,
                cfg.adam_beta2,
            ),
        )
        assert cfg.lr_end is not None  # this is set in config post-init
        self.lr_scheduler = get_lr_scheduler(
            cfg.lr_scheduler_name,
            lr=cfg.lr,
            optimizer=self.optimizer,
            warm_up_steps=cfg.lr_warm_up_steps,
            decay_steps=cfg.lr_decay_steps,
            training_steps=self.cfg.total_training_steps,
            lr_end=cfg.lr_end,
            num_cycles=cfg.n_restart_cycles,
        )
        self.coefficient_schedulers = {}
        for name, coeff_cfg in self.sae.get_coefficients().items():
            if not isinstance(coeff_cfg, TrainCoefficientConfig):
                coeff_cfg = TrainCoefficientConfig(value=coeff_cfg, warm_up_steps=0)
            self.coefficient_schedulers[name] = CoefficientScheduler(
                warm_up_steps=coeff_cfg.warm_up_steps,
                final_value=coeff_cfg.value,
            )

        # Setup autocast if using
        self.grad_scaler = torch.amp.GradScaler(
            device=self.cfg.device, enabled=self.cfg.autocast
        )

        if self.cfg.autocast:
            self.autocast_if_enabled = torch.autocast(
                device_type=self.cfg.device,
                dtype=torch.bfloat16,
                enabled=self.cfg.autocast,
            )
        else:
            self.autocast_if_enabled = contextlib.nullcontext()

    @property
    def feature_sparsity(self) -> torch.Tensor:
        return self.act_freq_scores / self.n_frac_active_samples

    @property
    def log_feature_sparsity(self) -> torch.Tensor:
        return _log_feature_sparsity(self.feature_sparsity)

    @property
    def dead_neurons(self) -> torch.Tensor:
        return (self.n_forward_passes_since_fired > self.cfg.dead_feature_window).bool()

    def fit(self) -> T_TRAINING_SAE:
        self.sae.to(self.cfg.device)
        pbar = tqdm(total=self.cfg.total_training_samples, desc="Training SAE")

        if self.sae.cfg.normalize_activations == "expected_average_only_in":
            self.activation_scaler.estimate_scaling_factor(
                d_in=self.sae.cfg.d_in,
                data_provider=self.data_provider,
                n_batches_for_norm_estimate=int(1e3),
            )

        # Train loop
        while self.n_training_samples < self.cfg.total_training_samples:
            # Do a training step.
            batch = next(self.data_provider).to(self.sae.device)
            self.n_training_samples += batch.shape[0]
            scaled_batch = self.activation_scaler(batch)

            step_output = self._train_step(sae=self.sae, sae_in=scaled_batch)

            if self.cfg.logger.log_to_wandb:
                self._log_train_step(step_output)
                self._run_and_log_evals()

            self._checkpoint_if_needed()
            self.n_training_steps += 1
            self._update_pbar(step_output, pbar)

        # fold the estimated norm scaling factor into the sae weights
        if self.activation_scaler.scaling_factor is not None:
            self.sae.fold_activation_norm_scaling_factor(
                self.activation_scaler.scaling_factor
            )
            self.activation_scaler.scaling_factor = None

        if self.cfg.save_final_checkpoint:
            self.save_checkpoint(checkpoint_name=f"final_{self.n_training_samples}")

        pbar.close()
        return self.sae

    def save_checkpoint(
        self,
        checkpoint_name: str,
        wandb_aliases: list[str] | None = None,
    ) -> None:
        checkpoint_path = None
        if self.cfg.checkpoint_path is not None or self.cfg.logger.log_to_wandb:
            with path_or_tmp_dir(self.cfg.checkpoint_path) as base_checkpoint_path:
                checkpoint_path = base_checkpoint_path / checkpoint_name
                checkpoint_path.mkdir(exist_ok=True, parents=True)

                weights_path, cfg_path = self.sae.save_model(str(checkpoint_path))

                sparsity_path = checkpoint_path / SPARSITY_FILENAME
                save_file({"sparsity": self.log_feature_sparsity}, sparsity_path)

                self.save_trainer_state(checkpoint_path)

                if self.cfg.logger.log_to_wandb:
                    self.cfg.logger.log(
                        self,
                        weights_path,
                        cfg_path,
                        sparsity_path=sparsity_path,
                        wandb_aliases=wandb_aliases,
                    )

        if self.save_checkpoint_fn is not None:
            self.save_checkpoint_fn(checkpoint_path=checkpoint_path)

    def save_trainer_state(self, checkpoint_path: Path) -> None:
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        scheduler_state_dicts = {
            name: scheduler.state_dict()
            for name, scheduler in self.coefficient_schedulers.items()
        }
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "n_training_samples": self.n_training_samples,
                "n_training_steps": self.n_training_steps,
                "act_freq_scores": self.act_freq_scores,
                "n_forward_passes_since_fired": self.n_forward_passes_since_fired,
                "n_frac_active_samples": self.n_frac_active_samples,
                "started_fine_tuning": self.started_fine_tuning,
                "coefficient_schedulers": scheduler_state_dicts,
            },
            str(checkpoint_path / TRAINER_STATE_FILENAME),
        )
        activation_scaler_path = checkpoint_path / ACTIVATION_SCALER_CFG_FILENAME
        self.activation_scaler.save(str(activation_scaler_path))

    def load_trainer_state(self, checkpoint_path: Path | str) -> None:
        checkpoint_path = Path(checkpoint_path)
        self.activation_scaler.load(checkpoint_path / ACTIVATION_SCALER_CFG_FILENAME)
        state_dict = torch.load(checkpoint_path / TRAINER_STATE_FILENAME)
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        self.n_training_samples = state_dict["n_training_samples"]
        self.n_training_steps = state_dict["n_training_steps"]
        self.act_freq_scores = state_dict["act_freq_scores"]
        self.n_forward_passes_since_fired = state_dict["n_forward_passes_since_fired"]
        self.n_frac_active_samples = state_dict["n_frac_active_samples"]
        self.started_fine_tuning = state_dict["started_fine_tuning"]
        for name, scheduler_state_dict in state_dict["coefficient_schedulers"].items():
            self.coefficient_schedulers[name].load_state_dict(scheduler_state_dict)

    def _train_step(
        self,
        sae: T_TRAINING_SAE,
        sae_in: torch.Tensor,
    ) -> TrainStepOutput:
        sae.train()

        # log and then reset the feature sparsity every feature_sampling_window steps
        if (self.n_training_steps + 1) % self.cfg.feature_sampling_window == 0:
            if self.cfg.logger.log_to_wandb:
                sparsity_log_dict = self._build_sparsity_log_dict()
                wandb.log(sparsity_log_dict, step=self.n_training_steps)
            self._reset_running_sparsity_stats()

        # for documentation on autocasting see:
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        with self.autocast_if_enabled:
            train_step_output = self.sae.training_forward_pass(
                step_input=TrainStepInput(
                    sae_in=sae_in,
                    dead_neuron_mask=self.dead_neurons,
                    coefficients=self.get_coefficients(),
                    n_training_steps=self.n_training_steps,
                ),
            )

            with torch.no_grad():
                # calling .bool() should be equivalent to .abs() > 0, and work with coo tensors
                firing_feats = train_step_output.feature_acts.bool().float()
                did_fire = firing_feats.sum(-2).bool()
                if did_fire.is_sparse:
                    did_fire = did_fire.to_dense()
                self.n_forward_passes_since_fired += 1
                self.n_forward_passes_since_fired[did_fire] = 0
                self.act_freq_scores += firing_feats.sum(0)
                self.n_frac_active_samples += self.cfg.train_batch_size_samples

        # Grad scaler will rescale gradients if autocast is enabled
        self.grad_scaler.scale(
            train_step_output.loss
        ).backward()  # loss.backward() if not autocasting
        self.grad_scaler.unscale_(self.optimizer)  # needed to clip correctly
        # TODO: Work out if grad norm clipping should be in config / how to test it.
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        self.grad_scaler.step(
            self.optimizer
        )  # just ctx.optimizer.step() if not autocasting
        self.grad_scaler.update()

        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        for scheduler in self.coefficient_schedulers.values():
            scheduler.step()

        return train_step_output

    @torch.no_grad()
    def _log_train_step(self, step_output: TrainStepOutput):
        if (self.n_training_steps + 1) % self.cfg.logger.wandb_log_frequency == 0:
            wandb.log(
                self._build_train_step_log_dict(
                    output=step_output,
                    n_training_samples=self.n_training_samples,
                ),
                step=self.n_training_steps,
            )

    @torch.no_grad()
    def get_coefficients(self) -> dict[str, float]:
        return {
            name: scheduler.value
            for name, scheduler in self.coefficient_schedulers.items()
        }

    @torch.no_grad()
    def _build_train_step_log_dict(
        self,
        output: TrainStepOutput,
        n_training_samples: int,
    ) -> dict[str, Any]:
        sae_in = output.sae_in
        sae_out = output.sae_out
        feature_acts = output.feature_acts
        loss = output.loss.item()

        # metrics for currents acts
        l0 = feature_acts.bool().float().sum(-1).to_dense().mean()
        current_learning_rate = self.optimizer.param_groups[0]["lr"]

        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
        total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
        explained_variance_legacy = 1 - per_token_l2_loss / total_variance
        explained_variance = 1 - per_token_l2_loss.mean() / total_variance.mean()

        log_dict = {
            # losses
            "losses/overall_loss": loss,
            # variance explained
            "metrics/explained_variance_legacy": explained_variance_legacy.mean().item(),
            "metrics/explained_variance_legacy_std": explained_variance_legacy.std().item(),
            "metrics/explained_variance": explained_variance.item(),
            "metrics/l0": l0.item(),
            # sparsity
            "sparsity/mean_passes_since_fired": self.n_forward_passes_since_fired.mean().item(),
            "sparsity/dead_features": self.dead_neurons.sum().item(),
            "details/current_learning_rate": current_learning_rate,
            "details/n_training_samples": n_training_samples,
            **{
                f"details/{name}_coefficient": scheduler.value
                for name, scheduler in self.coefficient_schedulers.items()
            },
        }
        for loss_name, loss_value in output.losses.items():
            log_dict[f"losses/{loss_name}"] = _unwrap_item(loss_value)

        for metric_name, metric_value in output.metrics.items():
            log_dict[f"metrics/{metric_name}"] = _unwrap_item(metric_value)

        return log_dict

    @torch.no_grad()
    def _run_and_log_evals(self):
        # record loss frequently, but not all the time.
        if (self.n_training_steps + 1) % (
            self.cfg.logger.wandb_log_frequency
            * self.cfg.logger.eval_every_n_wandb_logs
        ) == 0:
            self.sae.eval()
            eval_metrics = (
                self.evaluator(self.sae, self.data_provider, self.activation_scaler)
                if self.evaluator is not None
                else {}
            )
            for key, value in self.sae.log_histograms().items():
                eval_metrics[key] = wandb.Histogram(value)  # type: ignore

            wandb.log(
                eval_metrics,
                step=self.n_training_steps,
            )
            self.sae.train()

    @torch.no_grad()
    def _build_sparsity_log_dict(self) -> dict[str, Any]:
        log_feature_sparsity = _log_feature_sparsity(self.feature_sparsity)
        wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())  # type: ignore
        return {
            "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
            "plots/feature_density_line_chart": wandb_histogram,
            "sparsity/below_1e-5": (self.feature_sparsity < 1e-5).sum().item(),
            "sparsity/below_1e-6": (self.feature_sparsity < 1e-6).sum().item(),
        }

    @torch.no_grad()
    def _reset_running_sparsity_stats(self) -> None:
        self.act_freq_scores = torch.zeros(
            self.sae.cfg.d_sae,  # type: ignore
            device=self.cfg.device,
        )
        self.n_frac_active_samples = 0

    @torch.no_grad()
    def _checkpoint_if_needed(self):
        if (
            self.checkpoint_thresholds
            and self.n_training_samples > self.checkpoint_thresholds[0]
        ):
            self.save_checkpoint(checkpoint_name=str(self.n_training_samples))
            self.checkpoint_thresholds.pop(0)

    @torch.no_grad()
    def _update_pbar(
        self,
        step_output: TrainStepOutput,
        pbar: tqdm,  # type: ignore
        update_interval: int = 100,
    ):
        if self.n_training_steps % update_interval == 0:
            loss_strs = " | ".join(
                f"{loss_name}: {_unwrap_item(loss_value):.5f}"
                for loss_name, loss_value in step_output.losses.items()
            )
            pbar.set_description(f"{self.n_training_steps}| {loss_strs}")
            pbar.update(update_interval * self.cfg.train_batch_size_samples)


def _unwrap_item(item: float | torch.Tensor) -> float:
    return item.item() if isinstance(item, torch.Tensor) else item
