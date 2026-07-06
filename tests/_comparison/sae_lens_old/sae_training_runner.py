import json
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
import wandb
from simple_parsing import ArgumentParser
from transformer_lens.HookedTransformer import HookedRootModule

from tests._comparison.sae_lens_old import logger
from tests._comparison.sae_lens_old.config import (
    HfDataset,
    LanguageModelSAERunnerConfig,
)
from tests._comparison.sae_lens_old.load_model import load_model
from tests._comparison.sae_lens_old.training.activations_store import ActivationsStore
from tests._comparison.sae_lens_old.training.geometric_median import (
    compute_geometric_median,
)
from tests._comparison.sae_lens_old.training.sae_trainer import SAETrainer
from tests._comparison.sae_lens_old.training.training_sae import (
    TrainingSAE,
    TrainingSAEConfig,
)


class InterruptedException(Exception):
    pass


def interrupt_callback(sig_num: Any, stack_frame: Any):  # noqa: ARG001
    raise InterruptedException()


class SAETrainingRunner:
    """
    Class to run the training of a Sparse Autoencoder (SAE) on a TransformerLens model.
    """

    cfg: LanguageModelSAERunnerConfig
    model: HookedRootModule
    sae: TrainingSAE
    activations_store: ActivationsStore

    def __init__(
        self,
        cfg: LanguageModelSAERunnerConfig,
        override_dataset: HfDataset | None = None,
        override_model: HookedRootModule | None = None,
        override_sae: TrainingSAE | None = None,
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

        if override_model is None:
            self.model = load_model(
                self.cfg.model_class_name,
                self.cfg.model_name,
                device=self.cfg.device,
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
                self.sae = TrainingSAE.load_from_pretrained(
                    self.cfg.from_pretrained_path, self.cfg.device
                )
            else:
                self.sae = TrainingSAE(
                    TrainingSAEConfig.from_dict(
                        self.cfg.get_training_sae_cfg_dict(),
                    )
                )
                self._init_sae_group_b_decs()
        else:
            self.sae = override_sae

    def run(self):
        """
        Run the training of the SAE.
        """

        if self.cfg.log_to_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                config=cast(Any, self.cfg),
                name=self.cfg.run_name,
                id=self.cfg.wandb_id,
            )

        trainer = SAETrainer(
            model=self.model,
            sae=self.sae,
            activation_store=self.activations_store,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg,
        )

        self._compile_if_needed()
        sae = self.run_trainer_with_interruption_handling(trainer)

        if self.cfg.log_to_wandb:
            wandb.finish()

        return sae

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

    def run_trainer_with_interruption_handling(self, trainer: SAETrainer):
        try:
            # signal handlers (if preempted)
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)

            # train SAE
            sae = trainer.fit()

        except (KeyboardInterrupt, InterruptedException):
            logger.warning("interrupted, saving progress")
            checkpoint_name = str(trainer.n_training_tokens)
            self.save_checkpoint(trainer, checkpoint_name=checkpoint_name)
            logger.info("done saving")
            raise

        return sae

    # TODO: move this into the SAE trainer or Training SAE class
    def _init_sae_group_b_decs(
        self,
    ) -> None:
        """
        extract all activations at a certain layer and use for sae b_dec initialization
        """

        if self.cfg.b_dec_init_method == "geometric_median":
            self.activations_store.set_norm_scaling_factor_if_needed()
            layer_acts = self.activations_store.storage_buffer.detach()[:, 0, :]
            # get geometric median of the activations if we're using those.
            median = compute_geometric_median(
                layer_acts,
                maxiter=100,
            ).median
            self.sae.initialize_b_dec_with_precalculated(median)  # type: ignore
        elif self.cfg.b_dec_init_method == "mean":
            self.activations_store.set_norm_scaling_factor_if_needed()
            layer_acts = self.activations_store.storage_buffer.detach().cpu()[:, 0, :]
            self.sae.initialize_b_dec_with_mean(layer_acts)  # type: ignore

    @staticmethod
    def save_checkpoint(
        trainer: SAETrainer,
        checkpoint_name: str,
        wandb_aliases: list[str] | None = None,
    ) -> None:
        base_path = Path(trainer.cfg.checkpoint_path) / checkpoint_name
        base_path.mkdir(exist_ok=True, parents=True)

        trainer.activations_store.save(
            str(base_path / "activations_store_state.safetensors")
        )

        if trainer.sae.cfg.normalize_sae_decoder:
            trainer.sae.set_decoder_norm_to_unit_norm()

        weights_path, cfg_path, sparsity_path = trainer.sae.save_model(
            str(base_path),
            trainer.log_feature_sparsity,
        )

        # let's over write the cfg file with the trainer cfg, which is a super set of the original cfg.
        # and should not cause issues but give us more info about SAEs we trained in SAE Lens.
        config = trainer.cfg.to_dict()
        with open(cfg_path, "w") as f:
            json.dump(config, f)

        if trainer.cfg.log_to_wandb:
            # Avoid wandb saving errors such as:
            #   ValueError: Artifact name may only contain alphanumeric characters, dashes, underscores, and dots. Invalid name: sae_google/gemma-2b_etc
            sae_name = trainer.sae.get_name().replace("/", "__")

            # save model weights and cfg
            model_artifact = wandb.Artifact(
                sae_name,
                type="model",
                metadata=dict(trainer.cfg.__dict__),
            )
            model_artifact.add_file(str(weights_path))
            model_artifact.add_file(str(cfg_path))
            wandb.log_artifact(model_artifact, aliases=wandb_aliases)

            # save log feature sparsity
            sparsity_artifact = wandb.Artifact(
                f"{sae_name}_log_feature_sparsity",
                type="log_feature_sparsity",
                metadata=dict(trainer.cfg.__dict__),
            )
            sparsity_artifact.add_file(str(sparsity_path))
            wandb.log_artifact(sparsity_artifact)


def _parse_cfg_args(args: Sequence[str]) -> LanguageModelSAERunnerConfig:
    if len(args) == 0:
        args = ["--help"]
    parser = ArgumentParser(exit_on_error=False)
    parser.add_arguments(LanguageModelSAERunnerConfig, dest="cfg")
    return parser.parse_args(args).cfg


# moved into its own function to make it easier to test
def _run_cli(args: Sequence[str]):
    cfg = _parse_cfg_args(args)
    SAETrainingRunner(cfg=cfg).run()


if __name__ == "__main__":
    _run_cli(args=sys.argv[1:])
