import warnings
from dataclasses import dataclass, field
from typing import Generator

import torch
from typing_extensions import override

from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput
from sae_lens.saes.topk_sae import (
    _sparse_matmul_nd,
    act_times_W_dec,
    calculate_topk_aux_acts,
)


@dataclass
class MatryoshkaBatchTopKTrainingSAEConfig(BatchTopKTrainingSAEConfig):
    """
    Configuration class for training a MatryoshkaBatchTopKTrainingSAE.

    [Matryoshka SAEs](https://arxiv.org/pdf/2503.17547) use a series of nested reconstruction
    losses of different widths during training to avoid feature absorption. This also has a
    nice side-effect of encouraging higher-frequency features to be learned in earlier levels.
    However, this SAE has more hyperparameters to tune than standard BatchTopK SAEs, and takes
    longer to train due to requiring multiple forward passes per training step.

    After training, MatryoshkaBatchTopK SAEs are saved as JumpReLU SAEs.

    Args:
        matryoshka_widths (list[int]): The widths of the matryoshka levels. Defaults to an empty list.
        k (float): The number of features to keep active. Inherited from BatchTopKTrainingSAEConfig.
            Defaults to 100.
        topk_threshold_lr (float): Learning rate for updating the global topk threshold.
            The threshold is updated using an exponential moving average of the minimum
            positive activation value. Defaults to 0.01.
        use_matryoshka_aux_loss (bool): Whether to encourage dead latents to reconstruct the error
            of just their own level rather than the error of the entire SAE. This should result in
            better feature revival, but is slower to train. Defaults to False.
        aux_loss_coefficient (float): Coefficient for the auxiliary loss that encourages
            dead neurons to learn useful features. Inherited from TopKTrainingSAEConfig.
            Defaults to 1.0.
        rescale_acts_by_decoder_norm (bool): Treat the decoder as if it was already normalized.
            Inherited from TopKTrainingSAEConfig. Defaults to True.
        decoder_init_norm (float | None): Norm to initialize decoder weights to.
            Inherited from TrainingSAEConfig. Defaults to 0.1.
        d_in (int): Input dimension (dimensionality of the activations being encoded).
            Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
            Defaults to "float32".
        device (str): Device to place the SAE on. Inherited from SAEConfig.
            Defaults to "cpu".
    """

    matryoshka_widths: list[int] = field(default_factory=list)
    use_matryoshka_aux_loss: bool = False

    @override
    @classmethod
    def architecture(cls) -> str:
        return "matryoshka_batchtopk"


class MatryoshkaBatchTopKTrainingSAE(BatchTopKTrainingSAE):
    """
    Global Batch TopK Training SAE

    This SAE will maintain the k on average across the batch, rather than enforcing the k per-sample as in standard TopK.

    BatchTopK SAEs are saved as JumpReLU SAEs after training.
    """

    cfg: MatryoshkaBatchTopKTrainingSAEConfig  # type: ignore[assignment]

    def __init__(
        self, cfg: MatryoshkaBatchTopKTrainingSAEConfig, use_error_term: bool = False
    ):
        super().__init__(cfg, use_error_term)
        _validate_matryoshka_config(cfg)

    def _iterable_decode(
        self, feature_acts: torch.Tensor, include_outer_loss: bool = False
    ) -> Generator[tuple[int, torch.Tensor], None, None]:
        if self.cfg.rescale_acts_by_decoder_norm:
            # need to multiply by the inverse of the norm because division is illegal with sparse tensors
            inv_W_dec_norm = 1 / self.W_dec.norm(dim=-1)
            feature_acts = feature_acts * inv_W_dec_norm
        widths = self.cfg.matryoshka_widths
        prev_width = 0
        if not include_outer_loss:
            widths = widths[:-1]
        decoded = self.b_dec
        for width in widths:
            inner_feature_acts = feature_acts[:, prev_width:width]
            if inner_feature_acts.is_sparse:
                decoded = (
                    _sparse_matmul_nd(inner_feature_acts, self.W_dec[prev_width:width])
                    + decoded
                )
            else:
                decoded = inner_feature_acts @ self.W_dec[prev_width:width] + decoded
            prev_width = width
            yield width, self.run_time_activation_norm_fn_out(decoded)

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        base_output = super().training_forward_pass(step_input)
        # the outer matryoshka level is the base SAE, so we don't need to add an extra loss for it
        for width, inner_reconstruction in self._iterable_decode(
            base_output.feature_acts, include_outer_loss=False
        ):
            inner_mse_loss = (
                self.mse_loss_fn(inner_reconstruction, step_input.sae_in)
                .sum(dim=-1)
                .mean()
            )
            base_output.losses[f"inner_mse_loss_{width}"] = inner_mse_loss
            base_output.loss = base_output.loss + inner_mse_loss
        return base_output

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Calculate the auxiliary loss for dead neurons
        if self.cfg.use_matryoshka_aux_loss:
            aux_loss = self.calculate_matryoshka_aux_loss(
                sae_in=step_input.sae_in,
                sae_out=sae_out,
                feature_acts=feature_acts,
                hidden_pre=hidden_pre,
                dead_neuron_mask=step_input.dead_neuron_mask,
            )
        else:
            aux_loss = self.calculate_topk_aux_loss(
                sae_in=step_input.sae_in,
                sae_out=sae_out,
                hidden_pre=hidden_pre,
                dead_neuron_mask=step_input.dead_neuron_mask,
            )
        return {"auxiliary_reconstruction_loss": aux_loss}

    def calculate_matryoshka_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # calculate a separate aux loss for each new matryoshka portion of the SAE
        if dead_neuron_mask is not None and int(dead_neuron_mask.sum()) > 0:
            k_aux = sae_in.shape[-1] // 2
            prev_width = 0
            aux_losses = []

            # Normalize decoder weights once to avoid repeated computation across levels
            scaled_W_dec = (
                self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
                if self.cfg.rescale_acts_by_decoder_norm
                else self.W_dec
            )
            # TODO: find a way to implement this without needing to recalculate the SAE output for each level
            # may need to wait for a refactor in the next release of sae_lens for a clean way to do this
            for width, partial_sae_out in self._iterable_decode(
                feature_acts, include_outer_loss=True
            ):
                partial_dead_neuron_mask = dead_neuron_mask[prev_width:width]
                partial_num_dead = int(partial_dead_neuron_mask.sum())
                if partial_num_dead == 0:
                    prev_width = width
                    continue

                # Reduce the scale of the loss if there are a small number of dead latents
                scale = min(partial_num_dead / k_aux, 1.0)
                partial_k_aux = min(k_aux, partial_num_dead)
                partial_hidden_pre = hidden_pre[:, prev_width:width]
                residual = (sae_in - partial_sae_out).detach()
                auxk_acts = calculate_topk_aux_acts(
                    k_aux=partial_k_aux,
                    hidden_pre=partial_hidden_pre,
                    dead_neuron_mask=partial_dead_neuron_mask,
                )

                # Encourage the top ~50% of dead latents to predict the residual of the
                # top k living latents
                recons = act_times_W_dec(
                    auxk_acts,
                    scaled_W_dec[prev_width:width],
                    rescale_acts_by_decoder_norm=False,
                )
                auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()
                aux_losses.append(scale * auxk_loss)
                prev_width = width
            stacked_losses = torch.stack(aux_losses)
            return self.cfg.aux_loss_coefficient * stacked_losses.sum()
        return sae_out.new_tensor(0.0)


def _validate_matryoshka_config(cfg: MatryoshkaBatchTopKTrainingSAEConfig) -> None:
    if cfg.matryoshka_widths[-1] != cfg.d_sae:
        # warn the users that we will add a final matryoshka level
        warnings.warn(
            "WARNING: The final matryoshka level width is not set to cfg.d_sae. "
            "A final matryoshka level of width=cfg.d_sae will be added."
        )
        cfg.matryoshka_widths.append(cfg.d_sae)

    for prev_width, curr_width in zip(
        cfg.matryoshka_widths[:-1], cfg.matryoshka_widths[1:]
    ):
        if prev_width >= curr_width:
            raise ValueError("cfg.matryoshka_widths must be strictly increasing.")
    if len(cfg.matryoshka_widths) == 1:
        warnings.warn(
            "WARNING: You have only set one matryoshka level. This is equivalent to using a standard BatchTopK SAE and is likely not what you want."
        )
