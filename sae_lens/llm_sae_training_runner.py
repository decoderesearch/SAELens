import json
import signal
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic

import torch
import wandb
from safetensors.torch import save_file
from simple_parsing import ArgumentParser
from transformer_lens.hook_points import HookedRootModule
from typing_extensions import deprecated

from sae_lens import logger
from sae_lens.config import HfDataset, LanguageModelSAERunnerConfig
from sae_lens.constants import (
    RUNNER_CFG_FILENAME,
    SPARSITY_FILENAME,
)
from sae_lens.evals import EvalConfig, run_evals
from sae_lens.load_model import load_model
from sae_lens.registry import SAE_TRAINING_CLASS_REGISTRY
from sae_lens.saes.sae import (
    T_TRAINING_SAE,
    T_TRAINING_SAE_CONFIG,
    TrainingSAE,
    TrainingSAEConfig,
)
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.types import DataProvider


class InterruptedException(Exception):
    pass


def interrupt_callback(sig_num: Any, stack_frame: Any):  # noqa: ARG001
    raise InterruptedException()


@dataclass
class LLMSaeEvaluator(Generic[T_TRAINING_SAE]):
    model: HookedRootModule
    activations_store: ActivationsStore
    eval_batch_size_prompts: int | None
    n_eval_batches: int
    model_kwargs: dict[str, Any]

    def __call__(
        self,
        sae: T_TRAINING_SAE,
        data_provider: DataProvider,
        activation_scaler: ActivationScaler,
    ) -> dict[str, Any]:
        exclude_special_tokens = False
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

        eval_metrics, _ = run_evals(
            sae=sae,
            activation_store=self.activations_store,
            model=self.model,
            activation_scaler=activation_scaler,
            eval_config=eval_config,
            exclude_special_tokens=exclude_special_tokens,
            model_kwargs=self.model_kwargs,
        )  # not calculating featurwise metrics here.

        # Remove eval metrics that are already logged during training
        eval_metrics.pop("metrics/explained_variance", None)
        eval_metrics.pop("metrics/explained_variance_std", None)
        eval_metrics.pop("metrics/l0", None)
        eval_metrics.pop("metrics/l1", None)
        eval_metrics.pop("metrics/mse", None)

        # Remove metrics that are not useful for wandb logging
        eval_metrics.pop("metrics/total_tokens_evaluated", None)

        return eval_metrics


class LanguageModelSAETrainingRunner:
    """
    Class to run the training of a Sparse Autoencoder (SAE) on a TransformerLens model.
    """

    cfg: LanguageModelSAERunnerConfig[Any]
    model: HookedRootModule
    sae: TrainingSAE[Any]
    activations_store: ActivationsStore

    def __init__(
        self,
        cfg: LanguageModelSAERunnerConfig[T_TRAINING_SAE_CONFIG],
        override_dataset: HfDataset | None = None,
        override_model: HookedRootModule | None = None,
        override_sae: TrainingSAE[Any] | None = None,
        resume_from_checkpoint: Path | str | None = None,
    ):
        if override_dataset is not None:
            logger.warning(
                f"You just passed in a dataset which will override the one specified in your configuration: {cfg.dataset_path}. As a consequence this run will not be reproducible via configuration alone."
            )
        if override_model is not None:
            logger.warning(
                f"You just passed in a model which will override the one specified in your configuration: {cfg.model_name}. As a consequence this run will not be reproducible via configuration alone."
            )

        self.cfg = cfg
        # set in cfg.__post_init__; locally bound so type checkers see a str
        llm_device = self.cfg.llm_device
        assert llm_device is not None

        if override_model is None:
            self.model = load_model(
                self.cfg.model_class_name,
                self.cfg.model_name,
                device=llm_device,
                model_from_pretrained_kwargs=self.cfg.model_from_pretrained_kwargs,
            )
        else:
            self.model = override_model

        self.activations_store = ActivationsStore.from_config(
            self.model,
            self.cfg,
            override_dataset=override_dataset,
        )

        if override_sae is None:
            if self.cfg.from_pretrained_path is not None:
                self.sae = TrainingSAE.load_from_disk(
                    self.cfg.from_pretrained_path, self.cfg.device
                )
            else:
                self.sae = TrainingSAE.from_dict(
                    TrainingSAEConfig.from_dict(
                        self.cfg.get_training_sae_cfg_dict(),
                    ).to_dict()
                )
        else:
            self.sae = override_sae

        self.sae.to(self.cfg.device)

    def run(self):
        """
        Run the training of the SAE.
        """
        self._set_sae_metadata()
        if self.cfg.logger.log_to_wandb:
            wandb.init(
                project=self.cfg.logger.wandb_project,
                entity=self.cfg.logger.wandb_entity,
                config=self.cfg.to_dict(),
                name=self.cfg.logger.run_name,
                id=self.cfg.logger.wandb_id,
            )

        evaluator = LLMSaeEvaluator(
            model=self.model,
            activations_store=self.activations_store,
            eval_batch_size_prompts=self.cfg.eval_batch_size_prompts,
            n_eval_batches=self.cfg.n_eval_batches,
            model_kwargs=self.cfg.model_kwargs,
        )

        trainer = SAETrainer(
            sae=self.sae,
            data_provider=self.activations_store,
            evaluator=evaluator,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg.to_sae_trainer_config(),
        )

        if self.cfg.resume_from_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {self.cfg.resume_from_checkpoint}")
            trainer.load_trainer_state(self.cfg.resume_from_checkpoint)
            self.sae.load_weights_from_checkpoint(self.cfg.resume_from_checkpoint)
            self.activations_store.load_from_checkpoint(self.cfg.resume_from_checkpoint)

        self._compile_if_needed()
        sae = self.run_trainer_with_interruption_handling(trainer)

        if self.cfg.output_path is not None:
            self.save_final_sae(
                sae=sae,
                output_path=self.cfg.output_path,
                log_feature_sparsity=trainer.log_feature_sparsity,
            )

        if self.cfg.logger.log_to_wandb:
            wandb.finish()

        return sae

    def save_final_sae(
        self,
        sae: TrainingSAE[Any],
        output_path: str,
        log_feature_sparsity: torch.Tensor | None = None,
    ):
        base_output_path = Path(output_path)
        base_output_path.mkdir(exist_ok=True, parents=True)

        weights_path, cfg_path = sae.save_inference_model(str(base_output_path))

        sparsity_path = None
        if log_feature_sparsity is not None:
            sparsity_path = base_output_path / SPARSITY_FILENAME
            save_file({"sparsity": log_feature_sparsity}, sparsity_path)

        runner_config = self.cfg.to_dict()
        with open(base_output_path / RUNNER_CFG_FILENAME, "w") as f:
            json.dump(runner_config, f)

        if self.cfg.logger.log_to_wandb:
            self.cfg.logger.log(
                self,
                weights_path,
                cfg_path,
                sparsity_path=sparsity_path,
                wandb_aliases=["final_model"],
            )

    def _set_sae_metadata(self):
        self.sae.cfg.metadata.dataset_path = self.cfg.dataset_path
        self.sae.cfg.metadata.hook_name = self.cfg.hook_name
        self.sae.cfg.metadata.model_name = self.cfg.model_name
        self.sae.cfg.metadata.model_class_name = self.cfg.model_class_name
        self.sae.cfg.metadata.hook_head_index = self.cfg.hook_head_index
        self.sae.cfg.metadata.context_size = self.cfg.context_size
        self.sae.cfg.metadata.seqpos_slice = self.cfg.seqpos_slice
        self.sae.cfg.metadata.model_from_pretrained_kwargs = (
            self.cfg.model_from_pretrained_kwargs
        )
        self.sae.cfg.metadata.prepend_bos = self.cfg.prepend_bos
        self.sae.cfg.metadata.exclude_special_tokens = self.cfg.exclude_special_tokens
        self.sae.cfg.metadata.sequence_separator_token = (
            self.cfg.sequence_separator_token
        )
        self.sae.cfg.metadata.disable_concat_sequences = (
            self.cfg.disable_concat_sequences
        )

    def _compile_if_needed(self):
        # Compile model and SAE
        #  torch.compile can provide significant speedups (10-20% in testing)
        # using max-autotune gives the best speedups but:
        # (a) increases VRAM usage,
        # (b) can't be used on both SAE and LM (some issue with cudagraphs), and
        # (c) takes some time to compile
        # optimal settings seem to be:
        # use max-autotune on SAE and max-autotune-no-cudagraphs on LM
        # (also pylance seems to really hate this)
        if self.cfg.compile_llm:
            self.model = torch.compile(
                self.model,
                mode=self.cfg.llm_compilation_mode,
            )  # type: ignore

        if self.cfg.compile_sae:
            backend = "aot_eager" if self.cfg.device == "mps" else "inductor"

            self.sae.training_forward_pass = torch.compile(  # type: ignore
                self.sae.training_forward_pass,
                mode=self.cfg.sae_compilation_mode,
                backend=backend,
            )  # type: ignore

    def run_trainer_with_interruption_handling(
        self, trainer: SAETrainer[TrainingSAE[TrainingSAEConfig], TrainingSAEConfig]
    ):
        try:
            # signal handlers (if preempted)
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)

            # train SAE
            sae = trainer.fit()

        except (KeyboardInterrupt, InterruptedException):
            if self.cfg.checkpoint_path is not None:
                logger.warning("interrupted, saving progress")
                checkpoint_path = Path(self.cfg.checkpoint_path) / str(
                    trainer.n_training_samples
                )
                self.save_checkpoint(checkpoint_path)
                logger.info("done saving")
            raise

        return sae

    def save_checkpoint(
        self,
        checkpoint_path: Path | None,
    ) -> None:
        if checkpoint_path is None:
            return

        self.activations_store.save_to_checkpoint(checkpoint_path)

        runner_config = self.cfg.to_dict()
        with open(checkpoint_path / RUNNER_CFG_FILENAME, "w") as f:
            json.dump(runner_config, f)


def _parse_cfg_args(
    args: Sequence[str],
) -> LanguageModelSAERunnerConfig[TrainingSAEConfig]:
    """
    Parse command line arguments into a LanguageModelSAERunnerConfig.

    This function first parses the architecture argument to determine which
    concrete SAE config class to use, then parses the full configuration
    with that concrete type.
    """
    if len(args) == 0:
        args = ["--help"]

    # First, parse only the architecture to determine which concrete class to use
    architecture_parser = ArgumentParser(
        description="Parse architecture to determine SAE config class",
        exit_on_error=False,
        add_help=False,  # Don't add help to avoid conflicts
    )
    architecture_parser.add_argument(
        "--architecture",
        type=str,
        choices=["standard", "gated", "jumprelu", "topk", "batchtopk"],
        default="standard",
        help="SAE architecture to use",
    )

    # Parse known args to extract architecture, ignore unknown args for now
    arch_args, remaining_args = architecture_parser.parse_known_args(args)
    architecture = arch_args.architecture

    # Remove architecture from remaining args if it exists
    filtered_args = []
    skip_next = False
    for arg in remaining_args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--architecture":
            skip_next = True  # Skip the next argument (the architecture value)
            continue
        filtered_args.append(arg)

    # Create a custom wrapper class that simple_parsing can handle
    def create_config_class(
        sae_config_type: type[TrainingSAEConfig],
    ) -> type[LanguageModelSAERunnerConfig[TrainingSAEConfig]]:
        """Create a concrete config class for the given SAE config type."""

        # Create the base config without the sae field
        from dataclasses import field as dataclass_field
        from dataclasses import fields, make_dataclass

        # Get all fields from LanguageModelSAERunnerConfig except the generic sae field
        base_fields = []
        for field_obj in fields(LanguageModelSAERunnerConfig):
            if field_obj.name != "sae":
                base_fields.append((field_obj.name, field_obj.type, field_obj))

        # Add the concrete sae field
        base_fields.append(
            (
                "sae",
                sae_config_type,
                dataclass_field(
                    default_factory=lambda: sae_config_type(d_in=512, d_sae=1024)
                ),
            )
        )

        # Create the concrete class
        return make_dataclass(
            f"{sae_config_type.__name__}RunnerConfig",
            base_fields,
            bases=(LanguageModelSAERunnerConfig,),
        )

    # Map architecture to concrete config class
    sae_config_map: dict[str, type[TrainingSAEConfig]] = {
        name: cfg for name, (_, cfg) in SAE_TRAINING_CLASS_REGISTRY.items()
    }

    sae_config_type = sae_config_map[architecture]
    concrete_config_class = create_config_class(sae_config_type)

    # Now parse the full configuration with the concrete type
    parser = ArgumentParser(exit_on_error=False)
    parser.add_arguments(concrete_config_class, dest="cfg")

    # Parse the filtered arguments (without --architecture)
    parsed_args = parser.parse_args(filtered_args)

    # Return the parsed configuration
    return parsed_args.cfg


# moved into its own function to make it easier to test
def _run_cli(args: Sequence[str]):
    cfg = _parse_cfg_args(args)
    LanguageModelSAETrainingRunner(cfg=cfg).run()


if __name__ == "__main__":
    _run_cli(args=sys.argv[1:])


@deprecated("Use LanguageModelSAETrainingRunner instead")
class SAETrainingRunner(LanguageModelSAETrainingRunner):
    pass
