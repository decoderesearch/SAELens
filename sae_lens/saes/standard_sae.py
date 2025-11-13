from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from typing_extensions import override

from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    TrainCoefficientConfig,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
)


@dataclass
class StandardSAEConfig(SAEConfig):
    """
    Configuration class for a StandardSAE.
    """

    @override
    @classmethod
    def architecture(cls) -> str:
        return "standard"


class StandardSAE(SAE[StandardSAEConfig]):
    """
    StandardSAE is an inference-only implementation of a Sparse Autoencoder (SAE)
    using a simple linear encoder and decoder.

    It implements the required abstract methods from BaseSAE:
      - initialize_weights: sets up simple parameter initializations for W_enc, b_enc, W_dec, and b_dec.
      - encode: computes the feature activations from an input.
      - decode: reconstructs the input from the feature activations.

    The BaseSAE.forward() method automatically calls encode and decode,
    including any error-term processing if configured.
    """

    b_enc: nn.Parameter

    def __init__(self, cfg: StandardSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def initialize_weights(self) -> None:
        # Initialize encoder weights and bias.
        super().initialize_weights()
        _init_weights_standard(self)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor into the feature space.
        """
        # Preprocess the SAE input (casting type, applying hooks, normalization)
        sae_in = self.process_sae_in(x)
        # Compute the pre-activation values
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        # Apply the activation function (e.g., ReLU, depending on config)
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode the feature activations back to the input space.
        Now, if hook_z reshaping is turned on, we reverse the flattening.
        """
        # 1) linear transform
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        # 2) hook reconstruction
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        # 4) optional out-normalization (e.g. constant_norm_rescale)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        # 5) if hook_z is enabled, rearrange back to (..., n_heads, d_head).
        return self.reshape_fn_out(sae_out_pre, self.d_head)


@dataclass
class StandardTrainingSAEConfig(TrainingSAEConfig):
    """
    Configuration class for training a StandardTrainingSAE.
    """

    l1_coefficient: float = 1.0
    lp_norm: float = 1.0
    l1_warm_up_steps: int = 0

    @override
    @classmethod
    def architecture(cls) -> str:
        return "standard"


class StandardTrainingSAE(TrainingSAE[StandardTrainingSAEConfig]):
    """
    StandardTrainingSAE is a concrete implementation of BaseTrainingSAE using the "standard" SAE architecture.
    It implements:
      - initialize_weights: basic weight initialization for encoder/decoder.
      - encode: inference encoding (invokes encode_with_hidden_pre).
      - decode: a simple linear decoder.
      - encode_with_hidden_pre: computes activations and pre-activations.
      - calculate_aux_loss: computes a sparsity penalty based on the (optionally scaled) p-norm of feature activations.
    """

    b_enc: nn.Parameter

    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_standard(self)

    @override
    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        return {
            "l1": TrainCoefficientConfig(
                value=self.cfg.l1_coefficient,
                warm_up_steps=self.cfg.l1_warm_up_steps,
            ),
        }

    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Process the input (including dtype conversion, hook call, and any activation normalization)
        sae_in = self.process_sae_in(x)
        # Compute the pre-activation (and allow for a hook if desired)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)  # type: ignore
        # Apply the activation function (and any post-activation hook)
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return feature_acts, hidden_pre

    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # The "standard" auxiliary loss is a sparsity penalty on the feature activations
        weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)

        # Compute the p-norm (set by cfg.lp_norm) over the feature dimension
        sparsity = weighted_feature_acts.norm(p=self.cfg.lp_norm, dim=-1)
        l1_loss = (step_input.coefficients["l1"] * sparsity).mean()

        return {"l1_loss": l1_loss}

    def log_histograms(self) -> dict[str, NDArray[np.generic]]:
        """Log histograms of the weights and biases."""
        b_e_dist = self.b_enc.detach().float().cpu().numpy()
        return {
            **super().log_histograms(),
            "weights/b_e": b_e_dist,
        }


def _init_weights_standard(
    sae: SAE[StandardSAEConfig] | TrainingSAE[StandardTrainingSAEConfig],
) -> None:
    sae.b_enc = nn.Parameter(
        torch.zeros(sae.cfg.d_sae, dtype=sae.dtype, device=sae.device)
    )
