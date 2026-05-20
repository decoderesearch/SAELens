import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import torch
import wandb
from tqdm.auto import tqdm

from sae_lens.config import SAETrainerConfig
from sae_lens.saes.sae import T_TRAINING_SAE_CONFIG, TrainingSAE, TrainStepOutput
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.sae_trainer import SAETrainer, SaveCheckpointFn
from sae_lens.training.types import MultiHookDataProvider
from sae_lens.util import path_or_tmp_dir


class MultiSAEEvaluatorProtocol(Protocol):
    def __call__(
        self,
        saes: dict[str, TrainingSAE[Any]],
        data_provider: MultiHookDataProvider,
        activation_scalers: dict[str, ActivationScaler],
        hook_names: dict[str, str],
    ) -> dict[str, Any]: ...


class MultiSAETrainer:
    """
    Coordinator that trains multiple `TrainingSAE`s in lockstep on a shared
    multi-hook activation stream.

    Internally holds one `SAETrainer` per SAE (composition, not duplication)
    and drives them via the public `step` / `maybe_reset_sparsity` /
    `save_checkpoint` / `load_trainer_state` seams. Each per-SAE trainer
    owns its own optimizer, lr scheduler, coefficient schedulers,
    activation scaler, sparsity tracking, etc. — exactly as in single-SAE
    mode. The coordinator only handles batch routing, log aggregation, and
    multi-SAE checkpoint layout.
    """

    saes: dict[str, TrainingSAE[Any]]
    hook_names: dict[str, str]
    trainers: dict[str, SAETrainer[Any, Any]]
    data_provider: MultiHookDataProvider
    evaluator: MultiSAEEvaluatorProtocol | None
    save_checkpoint_fn: SaveCheckpointFn | None
    cfg: SAETrainerConfig
    n_training_steps: int
    n_training_samples: int

    def __init__(
        self,
        cfg: SAETrainerConfig,
        saes: dict[str, TrainingSAE[T_TRAINING_SAE_CONFIG]],
        hook_names: dict[str, str],
        data_provider: MultiHookDataProvider,
        evaluator: MultiSAEEvaluatorProtocol | None = None,
        save_checkpoint_fn: SaveCheckpointFn | None = None,
    ) -> None:
        if set(saes.keys()) != set(hook_names.keys()):
            raise ValueError(
                f"saes and hook_names must have identical keys; got "
                f"{sorted(saes.keys())} vs {sorted(hook_names.keys())}"
            )

        self.cfg = cfg
        self.saes = dict(saes)
        self.hook_names = dict(hook_names)
        self.data_provider = data_provider
        self.evaluator = evaluator
        self.save_checkpoint_fn = save_checkpoint_fn

        self.n_training_steps = 0
        self.n_training_samples = 0

        # Each per-SAE trainer's config: same training hyperparams, but
        # checkpointing/eval/save-final disabled — the coordinator drives those.
        # Using replace() to avoid mutating the shared cfg.
        per_sae_cfg = replace(
            cfg,
            n_checkpoints=0,
            checkpoint_path=None,
            save_final_checkpoint=False,
        )
        # Per-SAE SAETrainer instances. data_provider here is never consumed
        # (the coordinator drives them via step()), so an empty iter satisfies
        # the type without keeping refs to anything heavy.
        self.trainers = {
            name: SAETrainer(
                cfg=per_sae_cfg,
                sae=sae,
                data_provider=iter(()),
                evaluator=None,
                save_checkpoint_fn=None,
            )
            for name, sae in self.saes.items()
        }

        self.checkpoint_thresholds: list[int] = []
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

    def fit(self) -> dict[str, TrainingSAE[Any]]:
        for sae in self.saes.values():
            sae.to(self.cfg.device)

        pbar = tqdm(total=self.cfg.total_training_samples, desc="Training Multi-SAE")

        self._estimate_scaling_factors()

        while self.n_training_samples < self.cfg.total_training_samples:
            self._maybe_log_sparsity()

            multi_batch = next(self.data_provider)
            step_outputs: dict[str, TrainStepOutput] = {}
            for name, trainer in self.trainers.items():
                step_outputs[name] = trainer.step(multi_batch[self.hook_names[name]])

            # All per-SAE trainers advance in lockstep; mirror onto outer counter.
            self.n_training_samples = next(
                iter(self.trainers.values())
            ).n_training_samples

            if self.cfg.logger.log_to_wandb:
                self._log_train_step(step_outputs)
                self._run_and_log_evals()

            self._checkpoint_if_needed()

            for trainer in self.trainers.values():
                trainer.n_training_steps += 1
            self.n_training_steps += 1

            self._update_pbar(step_outputs, pbar)

        for name, trainer in self.trainers.items():
            if trainer.activation_scaler.scaling_factor is not None:
                self.saes[name].fold_activation_norm_scaling_factor(
                    trainer.activation_scaler.scaling_factor
                )
                trainer.activation_scaler.scaling_factor = None

        if self.cfg.save_final_checkpoint:
            self.save_checkpoint(checkpoint_name=f"final_{self.n_training_samples}")

        pbar.close()
        return self.saes

    @torch.no_grad()
    def _estimate_scaling_factors(self) -> None:
        """Estimate per-SAE activation scaling from a shared set of multi-hook batches."""
        needs = [
            name
            for name, sae in self.saes.items()
            if sae.cfg.normalize_activations == "expected_average_only_in"
        ]
        if not needs:
            return
        n = self.cfg.n_batches_for_norm_estimate
        cached: list[dict[str, torch.Tensor]] = [
            next(self.data_provider) for _ in range(n)
        ]
        for name in needs:
            hook = self.hook_names[name]
            self.trainers[name].activation_scaler.estimate_scaling_factor(
                d_in=self.saes[name].cfg.d_in,
                data_provider=iter([batch[hook] for batch in cached]),
                n_batches_for_norm_estimate=n,
            )

    def _is_logging_step(self) -> bool:
        return (
            self.cfg.logger.log_to_wandb
            and (self.n_training_steps + 1) % self.cfg.logger.wandb_log_frequency == 0
        )

    def _maybe_log_sparsity(self) -> None:
        sparsity_logs: dict[str, dict[str, Any]] = {}
        for name, trainer in self.trainers.items():
            d = trainer.maybe_reset_sparsity()
            if d:
                sparsity_logs[name] = d
        if sparsity_logs and self.cfg.logger.log_to_wandb:
            combined: dict[str, Any] = {
                f"{name}/{k}": v
                for name, log_dict in sparsity_logs.items()
                for k, v in log_dict.items()
            }
            wandb.log(combined, step=self.n_training_steps)

    @torch.no_grad()
    def _log_train_step(self, step_outputs: dict[str, TrainStepOutput]) -> None:
        if not self._is_logging_step():
            return
        combined: dict[str, Any] = {
            "details/n_training_samples": self.n_training_samples
        }
        for name, trainer in self.trainers.items():
            log_dict = trainer.build_train_step_log_dict(
                output=step_outputs[name],
                n_training_samples=self.n_training_samples,
            )
            for k, v in log_dict.items():
                combined[f"{name}/{k}"] = v
        wandb.log(combined, step=self.n_training_steps)

    @torch.no_grad()
    def _run_and_log_evals(self) -> None:
        if (self.n_training_steps + 1) % (
            self.cfg.logger.wandb_log_frequency
            * self.cfg.logger.eval_every_n_wandb_logs
        ) != 0:
            return
        for sae in self.saes.values():
            sae.eval()
        try:
            if self.evaluator is not None:
                metrics = self.evaluator(
                    saes=self.saes,
                    data_provider=self.data_provider,
                    activation_scalers={
                        name: t.activation_scaler for name, t in self.trainers.items()
                    },
                    hook_names=self.hook_names,
                )
                wandb.log(metrics, step=self.n_training_steps)
            # Per-SAE histograms (as in single-SAE _run_and_log_evals).
            hist_metrics: dict[str, Any] = {}
            for name, sae in self.saes.items():
                for k, v in sae.log_histograms().items():
                    hist_metrics[f"{name}/{k}"] = wandb.Histogram(v)  # type: ignore
            if hist_metrics:
                wandb.log(hist_metrics, step=self.n_training_steps)
        finally:
            for sae in self.saes.values():
                sae.train()

    def _checkpoint_if_needed(self) -> None:
        if (
            self.checkpoint_thresholds
            and self.n_training_samples > self.checkpoint_thresholds[0]
        ):
            self.save_checkpoint(checkpoint_name=str(self.n_training_samples))
            self.checkpoint_thresholds.pop(0)

    def save_checkpoint(
        self,
        checkpoint_name: str,
        wandb_aliases: list[str] | None = None,
    ) -> None:
        """
        Write a per-checkpoint directory at
        `<checkpoint_path>/<checkpoint_name>/`, with a per-SAE subdirectory
        for each SAE containing its weights, sparsity, activation scaler,
        and trainer state. The user-supplied `save_checkpoint_fn` (if any)
        is invoked with the checkpoint directory so the runner can drop
        runner-level files alongside.
        """
        base = self.cfg.checkpoint_path
        if base is None and not self.cfg.logger.log_to_wandb:
            if self.save_checkpoint_fn is not None:
                self.save_checkpoint_fn(checkpoint_path=None)
            return

        with path_or_tmp_dir(base) as base_path:
            checkpoint_path = base_path / checkpoint_name
            checkpoint_path.mkdir(exist_ok=True, parents=True)

            for name, trainer in self.trainers.items():
                trainer.save_checkpoint(
                    checkpoint_name=name,
                    base_path=checkpoint_path,
                    wandb_aliases=wandb_aliases,
                )

            if self.save_checkpoint_fn is not None:
                self.save_checkpoint_fn(checkpoint_path=checkpoint_path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore each per-SAE trainer's state and SAE weights from a multi-SAE checkpoint dir."""
        path = Path(path)
        missing = [name for name in self.saes if not (path / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"checkpoint at {path} is missing per-SAE subdirectories for {missing}; "
                f"expected one subdirectory per key in cfg.saes"
            )
        for name, trainer in self.trainers.items():
            per_sae = path / name
            trainer.load_trainer_state(per_sae)
            trainer.sae.load_weights_from_checkpoint(per_sae)
        any_trainer = next(iter(self.trainers.values()))
        self.n_training_steps = any_trainer.n_training_steps
        self.n_training_samples = any_trainer.n_training_samples

    @torch.no_grad()
    def _update_pbar(
        self,
        step_outputs: dict[str, TrainStepOutput],
        pbar: tqdm,  # type: ignore
        update_interval: int = 100,
    ) -> None:
        if self.n_training_steps % update_interval == 0:
            losses = " | ".join(
                f"{name}: {out.loss.item():.4f}" for name, out in step_outputs.items()
            )
            pbar.set_description(f"{self.n_training_steps}| {losses}")
            pbar.update(update_interval * self.cfg.train_batch_size_samples)
