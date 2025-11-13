import warnings
from dataclasses import dataclass, field

import torch
from typing_extensions import override

from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput
from sae_lens.saes.topk_sae import _sparse_matmul_nd


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

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        base_output = super().training_forward_pass(step_input)
        inv_W_dec_norm = 1 / self.W_dec.norm(dim=-1)
        # the outer matryoshka level is the base SAE, so we don't need to add an extra loss for it
        for width in self.cfg.matryoshka_widths[:-1]:
            inner_reconstruction = self._decode_matryoshka_level(
                base_output.feature_acts, width, inv_W_dec_norm
            )
            inner_mse_loss = (
                self.mse_loss_fn(inner_reconstruction, step_input.sae_in)
                .sum(dim=-1)
                .mean()
            )
            base_output.losses[f"inner_mse_loss_{width}"] = inner_mse_loss
            base_output.loss = base_output.loss + inner_mse_loss
        return base_output

    def _decode_matryoshka_level(
        self,
        feature_acts: torch.Tensor,
        width: int,
        inv_W_dec_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes feature activations back into input space for a matryoshka level
        """
        inner_feature_acts = feature_acts[:, :width]
        # Handle sparse tensors using efficient sparse matrix multiplication
        if self.cfg.rescale_acts_by_decoder_norm:
            # need to multiply by the inverse of the norm because division is illegal with sparse tensors
            inner_feature_acts = inner_feature_acts * inv_W_dec_norm[:width]
        if inner_feature_acts.is_sparse:
            sae_out_pre = (
                _sparse_matmul_nd(inner_feature_acts, self.W_dec[:width]) + self.b_dec
            )
        else:
            sae_out_pre = inner_feature_acts @ self.W_dec[:width] + self.b_dec
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)


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
