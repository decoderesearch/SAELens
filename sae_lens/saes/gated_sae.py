from dataclasses import dataclass
from typing import Any

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
class GatedSAEConfig(SAEConfig):
    """
    Configuration class for a GatedSAE.
    """

    @override
    @classmethod
    def architecture(cls) -> str:
        return "gated"


class GatedSAE(SAE[GatedSAEConfig]):
    """
    GatedSAE is an inference-only implementation of a Sparse Autoencoder (SAE)
    using a gated linear encoder and a standard linear decoder.
    """

    b_gate: nn.Parameter
    b_mag: nn.Parameter
    r_mag: nn.Parameter

    def __init__(self, cfg: GatedSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        # Ensure b_enc does not exist for the gated architecture
        self.b_enc = None

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_gated(self)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor into the feature space using a gated encoder.
        This must match the original encode_gated implementation from SAE class.
        """
        # Preprocess the SAE input (casting type, applying hooks, normalization)
        sae_in = self.process_sae_in(x)

        # Gating path exactly as in original SAE.encode_gated
        gating_pre_activation = sae_in @ self.W_enc + self.b_gate
        active_features = (gating_pre_activation > 0).to(self.dtype)

        # Magnitude path (weight sharing with gated encoder)
        magnitude_pre_activation = self.hook_sae_acts_pre(
            sae_in @ (self.W_enc * self.r_mag.exp()) + self.b_mag
        )
        feature_magnitudes = self.activation_fn(magnitude_pre_activation)

        # Combine gating and magnitudes
        return self.hook_sae_acts_post(active_features * feature_magnitudes)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode the feature activations back into the input space:
          1) Apply optional finetuning scaling.
          2) Linear transform plus bias.
          3) Run any reconstruction hooks and out-normalization if configured.
          4) If the SAE was reshaping hook_z activations, reshape back.
        """
        # 1) optional finetuning scaling
        # 2) linear transform
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        # 3) hooking and normalization
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        # 4) reshape if needed (hook_z)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @torch.no_grad()
    def fold_W_dec_norm(self):
        """Override to handle gated-specific parameters."""
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T

        # Gated-specific parameters need special handling
        # r_mag doesn't need scaling since W_enc scaling is sufficient for magnitude path
        self.b_gate.data = self.b_gate.data * W_dec_norms.squeeze()
        self.b_mag.data = self.b_mag.data * W_dec_norms.squeeze()


@dataclass
class GatedTrainingSAEConfig(TrainingSAEConfig):
    """
    Configuration class for training a GatedTrainingSAE.
    """

    l1_coefficient: float = 1.0
    l1_warm_up_steps: int = 0

    @override
    @classmethod
    def architecture(cls) -> str:
        return "gated"


class GatedTrainingSAE(TrainingSAE[GatedTrainingSAEConfig]):
    """
    GatedTrainingSAE is a concrete implementation of BaseTrainingSAE for the "gated" SAE architecture.
    It implements:
      - initialize_weights: sets up gating parameters (as in GatedSAE) plus optional training-specific init.
      - encode: calls encode_with_hidden_pre (standard training approach).
      - decode: linear transformation + hooking, same as GatedSAE or StandardTrainingSAE.
      - encode_with_hidden_pre: gating logic.
      - calculate_aux_loss: includes an auxiliary reconstruction path and gating-based sparsity penalty.
      - training_forward_pass: calls encode_with_hidden_pre, decode, and sums up MSE + gating losses.
    """

    b_gate: nn.Parameter  # type: ignore
    b_mag: nn.Parameter  # type: ignore
    r_mag: nn.Parameter  # type: ignore

    def __init__(self, cfg: GatedTrainingSAEConfig, use_error_term: bool = False):
        if use_error_term:
            raise ValueError(
                "GatedSAE does not support `use_error_term`. Please set `use_error_term=False`."
            )
        super().__init__(cfg, use_error_term)

    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_gated(self)

    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gated forward pass with pre-activation (for training).
        """
        sae_in = self.process_sae_in(x)

        # Gating path
        gating_pre_activation = sae_in @ self.W_enc + self.b_gate
        active_features = (gating_pre_activation > 0).to(self.dtype)

        # Magnitude path
        magnitude_pre_activation = sae_in @ (self.W_enc * self.r_mag.exp()) + self.b_mag
        magnitude_pre_activation = self.hook_sae_acts_pre(magnitude_pre_activation)

        feature_magnitudes = self.activation_fn(magnitude_pre_activation)

        # Combine gating path and magnitude path
        feature_acts = self.hook_sae_acts_post(active_features * feature_magnitudes)

        # Return both the final feature activations and the pre-activation (for logging or penalty)
        return feature_acts, magnitude_pre_activation

    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Re-center the input if apply_b_dec_to_input is set
        sae_in_centered = step_input.sae_in - (
            self.b_dec * self.cfg.apply_b_dec_to_input
        )

        # The gating pre-activation (pi_gate) for the auxiliary path
        pi_gate = sae_in_centered @ self.W_enc + self.b_gate
        pi_gate_act = torch.relu(pi_gate)

        # L1-like penalty scaled by W_dec norms
        l1_loss = (
            step_input.coefficients["l1"]
            * torch.sum(pi_gate_act * self.W_dec.norm(dim=1), dim=-1).mean()
        )

        # Aux reconstruction: reconstruct x purely from gating path
        via_gate_reconstruction = pi_gate_act @ self.W_dec + self.b_dec
        aux_recon_loss = (
            (via_gate_reconstruction - step_input.sae_in).pow(2).sum(dim=-1).mean()
        )

        # Return both losses separately
        return {"l1_loss": l1_loss, "auxiliary_reconstruction_loss": aux_recon_loss}

    def log_histograms(self) -> dict[str, NDArray[Any]]:
        """Log histograms of the weights and biases."""
        b_gate_dist = self.b_gate.detach().float().cpu().numpy()
        b_mag_dist = self.b_mag.detach().float().cpu().numpy()
        return {
            **super().log_histograms(),
            "weights/b_gate": b_gate_dist,
            "weights/b_mag": b_mag_dist,
        }

    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        return {
            "l1": TrainCoefficientConfig(
                value=self.cfg.l1_coefficient,
                warm_up_steps=self.cfg.l1_warm_up_steps,
            ),
        }

    @torch.no_grad()
    def fold_W_dec_norm(self):
        """Override to handle gated-specific parameters."""
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T

        # Gated-specific parameters need special handling
        # r_mag doesn't need scaling since W_enc scaling is sufficient for magnitude path
        self.b_gate.data = self.b_gate.data * W_dec_norms.squeeze()
        self.b_mag.data = self.b_mag.data * W_dec_norms.squeeze()


def _init_weights_gated(
    sae: SAE[GatedSAEConfig] | TrainingSAE[GatedTrainingSAEConfig],
) -> None:
    sae.b_gate = nn.Parameter(
        torch.zeros(sae.cfg.d_sae, dtype=sae.dtype, device=sae.device)
    )
    # Ensure r_mag is initialized to zero as in original
    sae.r_mag = nn.Parameter(
        torch.zeros(sae.cfg.d_sae, dtype=sae.dtype, device=sae.device)
    )
    sae.b_mag = nn.Parameter(
        torch.zeros(sae.cfg.d_sae, dtype=sae.dtype, device=sae.device)
    )
