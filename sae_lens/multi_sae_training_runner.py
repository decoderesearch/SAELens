"""
Public entrypoint for training multiple SAEs in parallel from a shared LLM
forward pass.

Use `MultiSAETrainingRunner(MultiSAETrainingRunnerConfig(...)).run()` to train
N SAEs that share a model and capture activations from one or more hook points
in a single forward. Each SAE has its own optimizer, learning-rate schedule,
coefficient schedulers, activation scaler, and sparsity tracking — exactly as
in single-SAE training. The runner just orchestrates the shared model and
multi-hook activations.

V1 limitations:

- Argparse/CLI not supported (programmatic config only).
- `from_pretrained_path` per entry not supported (resume from a multi-SAE
  checkpoint instead).
- Cached activations not supported in multi-hook mode.
- `compile_sae=True` not supported (`compile_llm=True` works fine — model is
  shared).
"""

from __future__ import annotations

import json
import signal
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import wandb
from safetensors.torch import save_file
from transformer_lens.hook_points import HookedRootModule

from sae_lens import __version__, logger
from sae_lens.config import HfDataset, LoggingConfig, SAETrainerConfig
from sae_lens.constants import RUNNER_CFG_FILENAME, SPARSITY_FILENAME
from sae_lens.evals import EvalConfig, run_evals
from sae_lens.load_model import load_model
from sae_lens.registry import get_sae_training_class
from sae_lens.saes.sae import TrainingSAE, TrainingSAEConfig
from sae_lens.training._interruption import InterruptedException, interrupt_callback
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from sae_lens.training.prefetch import PrefetchingIterator
from sae_lens.training.types import MultiHookDataProvider
from sae_lens.util import get_special_token_ids

# A user-supplied custom evaluator. Mirrors the single-SAE Evaluator signature
# but applied per SAE: `(sae, single_hook_data_provider_view, sae's scaler)`.
# The data-provider view yields one hook's activation tensors per step.
# Returns a flat dict that the runner prefixes with `{sae_name}/` when merging.
PerSAEEvaluator = Callable[
    [TrainingSAE[Any], Iterator[torch.Tensor], ActivationScaler], dict[str, Any]
]


@dataclass
class MultiSAETrainingRunnerConfig:
    """
    Configuration for parallel multi-SAE training.

    `saes` is a dict from a user-chosen name to a `TrainingSAEConfig`
    (different SAEs can use different architectures). `hook_names` maps the
    same names to TransformerLens hook points; pass a single string to share
    one hook across every SAE. SAEs sharing a hook in V1 must agree on `d_in`
    and `hook_head_index`.
    """

    saes: Mapping[str, TrainingSAEConfig]
    hook_names: Mapping[str, str] | str

    hook_head_indices: dict[str, int | None] | int | None = None

    # Data Generating Function
    model_name: str = "gelu-2l"
    model_class_name: str = "HookedTransformer"
    dataset_path: str = ""
    dataset_trust_remote_code: bool = True
    streaming: bool = True
    use_chat_formatting: bool = False
    context_size: int = 128

    # Activation Store Parameters
    n_batches_in_buffer: int = 20
    training_tokens: int = 2_000_000
    store_batch_size_prompts: int = 32
    seqpos_slice: tuple[int | None, ...] = (None,)
    disable_concat_sequences: bool = False
    sequence_separator_token: int | Literal["bos", "eos", "sep"] | None = "bos"
    activations_mixing_fraction: float = 0.5

    # Devices
    device: str = "cpu"
    llm_device: str | None = None
    act_store_device: str | None = None
    prefetch_llm_batches: bool | int = False

    # Performance
    seed: int = 42
    dtype: str = "float32"
    prepend_bos: bool = True
    autocast: bool = False
    autocast_lm: bool = False
    compile_llm: bool = False
    llm_compilation_mode: str | None = None

    # Training Parameters
    train_batch_size_tokens: int = 4096
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    lr: float = 3e-4
    lr_scheduler_name: str = "constant"
    lr_warm_up_steps: int = 0
    lr_end: float | None = None
    lr_decay_steps: int = 0
    n_restart_cycles: int = 1

    # Resampling protocol
    dead_feature_window: int = 1000
    feature_sampling_window: int = 2000
    dead_feature_threshold: float = 1e-8

    # Evals
    n_eval_batches: int = 10
    eval_batch_size_prompts: int | None = None
    evaluator: PerSAEEvaluator | None = None

    # Logging
    logger: LoggingConfig = field(default_factory=LoggingConfig)

    # Outputs / Checkpoints
    n_checkpoints: int = 0
    checkpoint_path: str | None = "checkpoints"
    save_final_checkpoint: bool = False
    output_path: str | None = "output"
    resume_from_checkpoint: str | None = None

    # Misc
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    model_from_pretrained_kwargs: dict[str, Any] | None = None
    sae_lens_version: str = field(default_factory=lambda: __version__)
    sae_lens_training_version: str = field(default_factory=lambda: __version__)
    exclude_special_tokens: bool | list[int] = False
    # Norm estimation caches this many multi-hook batches in memory at once;
    # peak memory scales with n_hooks * d_in, not just d_in as in single-SAE.
    n_batches_for_norm_estimate: int = 1000

    # Internal: populated by __post_init__
    _hook_names_per_sae: dict[str, str] = field(default_factory=dict)
    _hook_head_indices_per_sae: dict[str, int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.saes:
            raise ValueError("saes must contain at least one entry")

        # Normalize hook_names to a per-SAE dict.
        if isinstance(self.hook_names, str):
            self._hook_names_per_sae = {name: self.hook_names for name in self.saes}
        else:
            extra = set(self.hook_names) - set(self.saes)
            missing = set(self.saes) - set(self.hook_names)
            if extra or missing:
                raise ValueError(
                    f"hook_names keys must match saes keys; "
                    f"extra: {sorted(extra)}; missing: {sorted(missing)}"
                )
            self._hook_names_per_sae = dict(self.hook_names)

        # Normalize hook_head_indices to a per-SAE dict.
        if self.hook_head_indices is None or isinstance(self.hook_head_indices, int):
            shared_head = self.hook_head_indices
            self._hook_head_indices_per_sae = {name: shared_head for name in self.saes}
        else:
            extra = set(self.hook_head_indices) - set(self.saes)
            if extra:
                raise ValueError(
                    f"hook_head_indices has unknown keys (not in saes): {sorted(extra)}"
                )
            self._hook_head_indices_per_sae = {
                name: self.hook_head_indices.get(name) for name in self.saes
            }

        # SAEs sharing a hook must agree on d_in and hook_head_index in V1.
        per_hook_d_in: dict[str, int] = {}
        per_hook_head_idx: dict[str, int | None] = {}
        for name, hook in self._hook_names_per_sae.items():
            d_in = self.saes[name].d_in
            head_idx = self._hook_head_indices_per_sae[name]
            if hook in per_hook_d_in:
                if per_hook_d_in[hook] != d_in:
                    raise ValueError(
                        f"SAEs sharing hook {hook!r} must have the same d_in; "
                        f"saw {per_hook_d_in[hook]} and {d_in}"
                    )
                if per_hook_head_idx[hook] != head_idx:
                    raise ValueError(
                        f"SAEs sharing hook {hook!r} must have the same hook_head_index in V1; "
                        f"saw {per_hook_head_idx[hook]} and {head_idx}"
                    )
            else:
                per_hook_d_in[hook] = d_in
                per_hook_head_idx[hook] = head_idx

        if self.logger.run_name is None:
            archs = sorted({sae.architecture() for sae in self.saes.values()})
            self.logger.run_name = (
                f"multi-{'+'.join(archs)}-N{len(self.saes)}-LR-{self.lr}"
                f"-Tokens-{self.training_tokens:.3e}"
            )

        if self.model_from_pretrained_kwargs is None:
            if self.model_class_name == "HookedTransformer":
                self.model_from_pretrained_kwargs = {"center_writing_weights": False}
            else:
                self.model_from_pretrained_kwargs = {}

        if self.llm_device is None:
            self.llm_device = self.device
        if self.act_store_device is None:
            self.act_store_device = self.device
        elif self.act_store_device == "with_model":
            self.act_store_device = self.llm_device

        if (
            not isinstance(self.prefetch_llm_batches, bool)
            and self.prefetch_llm_batches < 0
        ):
            raise ValueError(
                "prefetch_llm_batches must be a bool or a non-negative int "
                f"(0 / False disables prefetching), got {self.prefetch_llm_batches}"
            )

        if self.lr_end is None:
            self.lr_end = self.lr / 10

        if self.checkpoint_path is not None:
            unique_id = self.logger.wandb_id or uuid.uuid4().hex[:8]
            self.checkpoint_path = f"{self.checkpoint_path}/{unique_id}"

        if isinstance(self.exclude_special_tokens, list) and not all(
            isinstance(x, int) for x in self.exclude_special_tokens
        ):
            raise ValueError("exclude_special_tokens list must contain only integers")

    @property
    def hook_names_per_sae(self) -> dict[str, str]:
        return self._hook_names_per_sae

    @property
    def hook_head_indices_per_sae(self) -> dict[str, int | None]:
        return self._hook_head_indices_per_sae

    @property
    def total_training_tokens(self) -> int:
        return self.training_tokens

    @property
    def total_training_steps(self) -> int:
        return self.total_training_tokens // self.train_batch_size_tokens

    def to_sae_trainer_config(self) -> SAETrainerConfig:
        return SAETrainerConfig(
            n_checkpoints=self.n_checkpoints,
            checkpoint_path=self.checkpoint_path,
            save_final_checkpoint=self.save_final_checkpoint,
            total_training_samples=self.total_training_tokens,
            device=self.device,
            autocast=self.autocast,
            lr=self.lr,
            lr_end=self.lr_end,
            lr_scheduler_name=self.lr_scheduler_name,
            lr_warm_up_steps=self.lr_warm_up_steps,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            lr_decay_steps=self.lr_decay_steps,
            n_restart_cycles=self.n_restart_cycles,
            train_batch_size_samples=self.train_batch_size_tokens,
            dead_feature_window=self.dead_feature_window,
            feature_sampling_window=self.feature_sampling_window,
            logger=self.logger,
            n_batches_for_norm_estimate=self.n_batches_for_norm_estimate,
        )

    def to_dict(self) -> dict[str, Any]:
        # `evaluator` is not serializable; `_hook_*_per_sae` are internal
        # derived fields, redundant with hook_names / hook_head_indices.
        excluded = {
            "evaluator",
            "_hook_names_per_sae",
            "_hook_head_indices_per_sae",
        }
        d = {k: v for k, v in asdict(self).items() if k not in excluded}
        d["saes"] = {name: cfg.to_dict() for name, cfg in self.saes.items()}
        return d


class _SingleHookDataProviderView:
    """Adapter exposing one hook's slice of a multi-hook DataProvider as `Iterator[Tensor]`."""

    def __init__(self, source: MultiHookDataProvider, hook_name: str) -> None:
        self._source = source
        self._hook_name = hook_name

    def __iter__(self) -> _SingleHookDataProviderView:
        return self

    def __next__(self) -> torch.Tensor:
        return next(self._source)[self._hook_name]


@dataclass
class MultiSAEEvaluator:
    """
    Per-SAE built-in evaluator wrapper. Built-in evals (`run_evals`) run once
    per SAE — sequential, with model substitution for KL / CE recovered.
    Optionally also runs a user-supplied per-SAE evaluator on a single-hook
    view of the multi-hook DataProvider.
    """

    model: HookedRootModule
    activations_store: ActivationsStore
    eval_batch_size_prompts: int | None
    n_eval_batches: int
    model_kwargs: Mapping[str, Any]
    user_evaluator: PerSAEEvaluator | None = None

    def __call__(
        self,
        saes: dict[str, TrainingSAE[Any]],
        data_provider: MultiHookDataProvider,
        activation_scalers: dict[str, ActivationScaler],
        hook_names: dict[str, str],
    ) -> dict[str, Any]:
        exclude_special_tokens: bool | list[int] = False
        if self.activations_store.exclude_special_tokens is not None:
            exclude_special_tokens = (
                self.activations_store.exclude_special_tokens.tolist()
            )

        eval_config = EvalConfig(
            batch_size_prompts=self.eval_batch_size_prompts,
            n_eval_reconstruction_batches=self.n_eval_batches,
            n_eval_sparsity_variance_batches=self.n_eval_batches,
            compute_ce_loss=True,
            compute_l2_norms=True,
            compute_sparsity_metrics=True,
            compute_variance_metrics=True,
        )

        # Pause the prefetcher (if any) for the whole eval cycle; both built-in
        # and user evaluators may pull from the underlying generator state.
        pause_ctx: AbstractContextManager[None] = (
            data_provider.paused()
            if isinstance(data_provider, PrefetchingIterator)
            else nullcontext()
        )
        out: dict[str, Any] = {}
        with pause_ctx:
            for name, sae in saes.items():
                # `run_evals` reads sae.cfg.metadata.hook_name to know which hook
                # to substitute; the activation_store is just a token source so
                # passing the multi-hook store directly is correct.
                metrics, _ = run_evals(
                    sae=sae,
                    activation_store=self.activations_store,
                    model=self.model,
                    activation_scaler=activation_scalers[name],
                    eval_config=eval_config,
                    exclude_special_tokens=exclude_special_tokens,
                    model_kwargs=dict(self.model_kwargs),
                )
                # Drop metrics that are already logged during training (matches
                # the single-SAE LLMSaeEvaluator).
                for k in (
                    "metrics/explained_variance",
                    "metrics/explained_variance_std",
                    "metrics/l0",
                    "metrics/l1",
                    "metrics/mse",
                    "metrics/total_tokens_evaluated",
                ):
                    metrics.pop(k, None)
                for k, v in metrics.items():
                    out[f"{name}/{k}"] = v

                if self.user_evaluator is not None:
                    user_view = _SingleHookDataProviderView(
                        data_provider, hook_names[name]
                    )
                    user_metrics = self.user_evaluator(
                        sae, user_view, activation_scalers[name]
                    )
                    for k, v in user_metrics.items():
                        out[f"{name}/{k}"] = v
        return out


class MultiSAETrainingRunner:
    """
    Orchestrator that wires the model, multi-hook ActivationsStore, dict of
    `TrainingSAE`s, evaluator, and `MultiSAETrainer`. Public surface:
    `__init__(cfg, ...)` and `run() -> dict[str, TrainingSAE]`.
    """

    cfg: MultiSAETrainingRunnerConfig
    model: HookedRootModule
    saes: dict[str, TrainingSAE[Any]]
    activations_store: ActivationsStore

    def __init__(
        self,
        cfg: MultiSAETrainingRunnerConfig,
        override_dataset: HfDataset | None = None,
        override_model: HookedRootModule | None = None,
        override_saes: dict[str, TrainingSAE[Any]] | None = None,
    ) -> None:
        self.cfg = cfg

        if override_dataset is not None:
            logger.warning(
                f"override_dataset overrides cfg.dataset_path={cfg.dataset_path!r}; "
                "this run will not be reproducible from configuration alone."
            )
        if override_model is not None:
            logger.warning(
                f"override_model overrides cfg.model_name={cfg.model_name!r}; "
                "this run will not be reproducible from configuration alone."
            )

        llm_device = cfg.llm_device
        assert llm_device is not None  # set in __post_init__

        self.model = (
            override_model
            if override_model is not None
            else load_model(
                cfg.model_class_name,
                cfg.model_name,
                device=llm_device,
                model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
            )
        )

        unique_hooks = list(dict.fromkeys(cfg.hook_names_per_sae.values()))
        hook_d_ins: dict[str, int] = {}
        hook_head_indices: dict[str, int | None] = {}
        for name, hook in cfg.hook_names_per_sae.items():
            hook_d_ins.setdefault(hook, cfg.saes[name].d_in)
            hook_head_indices.setdefault(hook, cfg.hook_head_indices_per_sae[name])

        # All SAEs share normalize_activations? No — each SAE's normalize is per-SAE,
        # but the store doesn't apply normalization itself (the trainer's per-SAE
        # ActivationScaler does). Pass "none" to the store to keep its normalize
        # field a no-op; per-SAE scaling is handled inside the trainer.
        self.activations_store = ActivationsStore.from_config_multi_hook(
            model=self.model,
            dataset=override_dataset
            if override_dataset is not None
            else cfg.dataset_path,
            hook_names=unique_hooks,
            hook_d_ins=hook_d_ins,
            hook_head_indices=hook_head_indices,
            streaming=cfg.streaming,
            context_size=cfg.context_size,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size_prompts=cfg.store_batch_size_prompts,
            train_batch_size_tokens=cfg.train_batch_size_tokens,
            prepend_bos=cfg.prepend_bos,
            normalize_activations="none",
            device=torch.device(cfg.act_store_device),  # type: ignore[arg-type]
            dtype=cfg.dtype,
            model_kwargs=cfg.model_kwargs,
            autocast_lm=cfg.autocast_lm,
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
            seqpos_slice=cfg.seqpos_slice,
            exclude_special_tokens=_resolve_exclude_special_tokens(
                cfg.exclude_special_tokens,
                self.model,
                torch.device(cfg.act_store_device),  # type: ignore[arg-type]
            ),
            disable_concat_sequences=cfg.disable_concat_sequences,
            sequence_separator_token=cfg.sequence_separator_token,
            activations_mixing_fraction=cfg.activations_mixing_fraction,
            use_chat_formatting=cfg.use_chat_formatting,
        )

        if override_saes is None:
            saes: dict[str, TrainingSAE[Any]] = {}
            for name, sae_cfg in cfg.saes.items():
                sae_class, _ = get_sae_training_class(sae_cfg.architecture())
                saes[name] = sae_class(sae_cfg)
            self.saes = saes
        else:
            extra = set(override_saes) - set(cfg.saes)
            missing = set(cfg.saes) - set(override_saes)
            if extra or missing:
                raise ValueError(
                    f"override_saes keys must match cfg.saes; extra: {sorted(extra)}; missing: {sorted(missing)}"
                )
            self.saes = override_saes

        for sae in self.saes.values():
            sae.to(cfg.device)

    def run(self) -> dict[str, TrainingSAE[Any]]:
        self._set_sae_metadata()
        if self.cfg.logger.log_to_wandb:
            wandb.init(
                project=self.cfg.logger.wandb_project,
                entity=self.cfg.logger.wandb_entity,
                config=self.cfg.to_dict(),
                name=self.cfg.logger.run_name,
                id=self.cfg.logger.wandb_id,
            )

        evaluator = MultiSAEEvaluator(
            model=self.model,
            activations_store=self.activations_store,
            eval_batch_size_prompts=self.cfg.eval_batch_size_prompts,
            n_eval_batches=self.cfg.n_eval_batches,
            model_kwargs=self.cfg.model_kwargs,
            user_evaluator=self.cfg.evaluator,
        )

        data_provider: MultiHookDataProvider = (
            self.activations_store.get_multi_hook_data_loader()
        )
        if self.cfg.prefetch_llm_batches:
            prefetch_size = (
                1
                if isinstance(self.cfg.prefetch_llm_batches, bool)
                else self.cfg.prefetch_llm_batches
            )
            data_provider = PrefetchingIterator(data_provider, prefetch=prefetch_size)

        if self.cfg.compile_llm:
            self.model = torch.compile(self.model, mode=self.cfg.llm_compilation_mode)  # type: ignore[assignment]

        trainer = MultiSAETrainer(
            cfg=self.cfg.to_sae_trainer_config(),
            saes=self.saes,
            hook_names=self.cfg.hook_names_per_sae,
            data_provider=data_provider,
            evaluator=evaluator,
            save_checkpoint_fn=self._save_runner_checkpoint_files,
        )

        if self.cfg.resume_from_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {self.cfg.resume_from_checkpoint}")
            trainer.load_checkpoint(self.cfg.resume_from_checkpoint)
            self.activations_store.load_from_checkpoint(self.cfg.resume_from_checkpoint)

        saes = self._run_with_interruption_handling(trainer)

        if self.cfg.output_path is not None:
            self._save_final_outputs(saes, trainer)

        if self.cfg.logger.log_to_wandb:
            wandb.finish()

        return saes

    def _run_with_interruption_handling(
        self, trainer: MultiSAETrainer
    ) -> dict[str, TrainingSAE[Any]]:
        try:
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)
            return trainer.fit()
        except (KeyboardInterrupt, InterruptedException):
            if self.cfg.checkpoint_path is not None:
                logger.warning("interrupted, saving progress")
                trainer.save_checkpoint(checkpoint_name=str(trainer.n_training_samples))
                logger.info("done saving")
            raise

    def _set_sae_metadata(self) -> None:
        for name, sae in self.saes.items():
            sae.cfg.metadata.dataset_path = self.cfg.dataset_path
            sae.cfg.metadata.hook_name = self.cfg.hook_names_per_sae[name]
            sae.cfg.metadata.model_name = self.cfg.model_name
            sae.cfg.metadata.model_class_name = self.cfg.model_class_name
            sae.cfg.metadata.hook_head_index = self.cfg.hook_head_indices_per_sae[name]
            sae.cfg.metadata.context_size = self.cfg.context_size
            sae.cfg.metadata.seqpos_slice = self.cfg.seqpos_slice
            sae.cfg.metadata.model_from_pretrained_kwargs = (
                self.cfg.model_from_pretrained_kwargs
            )
            sae.cfg.metadata.prepend_bos = self.cfg.prepend_bos
            sae.cfg.metadata.exclude_special_tokens = self.cfg.exclude_special_tokens
            sae.cfg.metadata.sequence_separator_token = (
                self.cfg.sequence_separator_token
            )
            sae.cfg.metadata.disable_concat_sequences = (
                self.cfg.disable_concat_sequences
            )

    def _save_runner_checkpoint_files(self, checkpoint_path: Path | None) -> None:
        """Hook called by `MultiSAETrainer.save_checkpoint` to drop runner-level files."""
        if checkpoint_path is None:
            return
        self.activations_store.save_to_checkpoint(checkpoint_path)
        with open(checkpoint_path / RUNNER_CFG_FILENAME, "w") as f:
            json.dump(self.cfg.to_dict(), f)

    def _save_final_outputs(
        self, saes: dict[str, TrainingSAE[Any]], trainer: MultiSAETrainer
    ) -> None:
        assert self.cfg.output_path is not None
        base = Path(self.cfg.output_path)
        base.mkdir(exist_ok=True, parents=True)

        for name, sae in saes.items():
            per_sae = base / name
            per_sae.mkdir(exist_ok=True, parents=True)
            weights_path, cfg_path = sae.save_inference_model(str(per_sae))

            sparsity_path = per_sae / SPARSITY_FILENAME
            save_file(
                {"sparsity": trainer.trainers[name].log_feature_sparsity},
                sparsity_path,
            )

            if self.cfg.logger.log_to_wandb:
                # Trainer's logger.log expects a `trainer` object with `sae` and `cfg` —
                # we pass a tiny shim so wandb artifact naming and metadata work.
                shim = _ArtifactLogShim(sae=sae, cfg=self.cfg.to_sae_trainer_config())
                self.cfg.logger.log(
                    shim,
                    weights_path,
                    cfg_path,
                    sparsity_path=sparsity_path,
                    wandb_aliases=["final_model"],
                )

        with open(base / RUNNER_CFG_FILENAME, "w") as f:
            json.dump(self.cfg.to_dict(), f)


@dataclass
class _ArtifactLogShim:
    sae: TrainingSAE[Any]
    cfg: SAETrainerConfig


def _resolve_exclude_special_tokens(
    raw: bool | list[int],
    model: HookedRootModule,
    device: torch.device,
) -> torch.Tensor | None:
    if raw is False:
        return None
    ids = (
        list(get_special_token_ids(model.tokenizer))  # type: ignore[arg-type]
        if raw is True
        else list(raw)
    )
    return torch.tensor(ids, dtype=torch.long, device=device)
