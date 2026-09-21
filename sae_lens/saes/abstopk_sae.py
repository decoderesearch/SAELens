"""AbsTopK SAE variant: per-sample top-k selection on pre-activation magnitude,
with the sign of each selected feature preserved so features can fire in either
direction (bidirectional features).

Ported from "AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features"
(ICLR 2026, https://arxiv.org/abs/2510.00404). The selection rule (top-k on
``hidden_pre.abs()``, then scatter the signed values with no ReLU) follows the
paper's MIT-licensed reference implementation at
https://github.com/GoXzascc/AbsTopK-SAE (Copyright 2025 GoXzascc), reimplemented
here against SAELens's SAE base classes. SAELens is MIT-licensed
(Copyright 2023 Joseph Bloom).
"""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from typing_extensions import override

from sae_lens.saes.sae import TrainStepInput
from sae_lens.saes.topk_sae import (
    TopKSAE,
    TopKSAEConfig,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
    act_times_W_dec,
)


class AbsTopK(nn.Module):
    """
    A TopK activation that zeroes out all but the top K elements along the last
    dimension, where the ranking is by magnitude (absolute value) and the signed
    values of the selected elements are kept (no ReLU), so the surviving features
    may be negative.
    """

    def __init__(
        self,
        k: int,
    ):
        super().__init__()
        self.k = k

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        1) Select top K elements along the last dimension by magnitude.
        2) Keep their signed values.
        3) Zero out all other entries.
        """
        topk_indices = torch.topk(x.abs(), k=self.k, dim=-1, sorted=False).indices
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, x.gather(-1, topk_indices))
        return result


@dataclass
class AbsTopKSAEConfig(TopKSAEConfig):
    """
    Configuration class for AbsTopKSAE inference.

    Args:
        k (int): Number of top features to keep active during inference. Only the
            top k features with the largest pre-activation magnitudes will be
            non-zero (with their sign preserved).
        rescale_acts_by_decoder_norm (bool): Whether to treat the decoder as if it
            was already normalized. This affects the topk selection by rescaling
            pre-activations by decoder norms. Requires that the SAE was trained
            this way. Inherited from TopKSAEConfig.
        d_in (int): Input dimension (dimensionality of the activations being
            encoded). Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
        device (str): Device to place the SAE on. Inherited from SAEConfig.
        apply_b_dec_to_input (bool): Whether to apply decoder bias to the input
            before encoding. Inherited from SAEConfig.
        normalize_activations (Literal["none", "expected_average_only_in", "constant_norm_rescale", "layer_norm"]):
            Normalization strategy for input activations. Inherited from SAEConfig.
        reshape_activations (Literal["none", "hook_z"]): How to reshape activations
            (useful for attention head outputs). Inherited from SAEConfig.
        metadata (SAEMetadata): Metadata about the SAE (model name, hook name,
            etc.). Inherited from SAEConfig.
    """

    @override
    @classmethod
    def architecture(cls) -> str:
        return "abstopk"


class AbsTopKSAE(TopKSAE):
    """
    An inference-only sparse autoencoder using an "abstopk" activation function.
    It uses linear encoder and decoder layers, applying the AbsTopK activation
    (per-sample top-k by magnitude, sign preserved) to the hidden pre-activation
    in its encode step, so features can fire in either direction.
    """

    cfg: AbsTopKSAEConfig  # type: ignore[assignment]

    def __init__(self, cfg: AbsTopKSAEConfig, use_error_term: bool = False):
        """
        Args:
            cfg: SAEConfig defining model size and behavior.
            use_error_term: Whether to apply the error-term approach in the
                forward pass.
        """
        super().__init__(cfg, use_error_term)

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return AbsTopK(self.cfg.k)


@dataclass
class AbsTopKTrainingSAEConfig(TopKTrainingSAEConfig):
    """
    Configuration class for training an AbsTopKTrainingSAE.

    AbsTopK SAEs enforce exactly k active features per sample like TopK SAEs, but
    select them by pre-activation magnitude and keep their signs, so features are
    bidirectional.

    Args:
        k (int): Number of top features to keep active. Only the top k features
            with the largest pre-activation magnitudes will be non-zero (with
            their sign preserved). Inherited from TopKTrainingSAEConfig.
        use_sparse_activations (bool): Ignored by the AbsTopK architecture; the
            AbsTopK activation always produces dense activations (kept for
            compatibility with TopKTrainingSAEConfig).
        aux_loss_coefficient (float): Coefficient for the auxiliary loss that
            encourages dead neurons to learn useful features. Dead latents are
            selected by pre-activation magnitude. Inherited from
            TopKTrainingSAEConfig.
        rescale_acts_by_decoder_norm (bool): Treat the decoder as if it was
            already normalized. This is a good idea since decoder norm can
            randomly drift during training, and this affects what the topk
            activations will be. Inherited from TopKTrainingSAEConfig.
        decoder_init_norm (float | None): Norm to initialize decoder weights to.
            0.1 corresponds to the "heuristic" initialization from Anthropic's
            April update. Use None to disable. Inherited from
            TrainingSAEConfig.
        d_in (int): Input dimension (dimensionality of the activations being
            encoded). Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
        device (str): Device to place the SAE on. Inherited from SAEConfig.
        apply_b_dec_to_input (bool): Whether to apply decoder bias to the input
            before encoding. Inherited from SAEConfig.
        normalize_activations (Literal["none", "expected_average_only_in", "constant_norm_rescale", "layer_norm"]):
            Normalization strategy for input activations. Inherited from
            SAEConfig.
        reshape_activations (Literal["none", "hook_z"]): How to reshape activations
            (useful for attention head outputs). Inherited from SAEConfig.
        metadata (SAEMetadata): Metadata about the SAE training (model name, hook
            name, etc.). Inherited from SAEConfig.
    """

    @override
    @classmethod
    def architecture(cls) -> str:
        return "abstopk"


class AbsTopKTrainingSAE(TopKTrainingSAE):
    """
    AbsTopK variant with training functionality. Calculates a topk-related
    auxiliary loss (with dead latents selected by magnitude), etc.
    """

    cfg: AbsTopKTrainingSAEConfig  # type: ignore[assignment]

    def __init__(self, cfg: AbsTopKTrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return AbsTopK(self.cfg.k)

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Calculate the auxiliary loss for dead neurons
        abstopk_loss = self.calculate_abstopk_aux_loss(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            hidden_pre=hidden_pre,
            dead_neuron_mask=step_input.dead_neuron_mask,
        )
        return {"auxiliary_reconstruction_loss": abstopk_loss}

    def calculate_abstopk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Calculate the AbsTopK auxiliary loss.

        Same as the TopK auxiliary loss (dead neurons reconstruct the residual
        error from the live neurons to prevent neuron death), except that the
        dead latents are selected by the magnitude of their pre-activations and
        keep their signed values, so strongly negative dead latents participate
        too.
        """
        # Same as the TopK aux loss, except for the magnitude-based selection in
        # calculate_abstopk_aux_acts.
        # NOTE: checking the number of dead neurons will force a GPU sync, so
        # performance can likely be improved here
        if dead_neuron_mask is None or (num_dead := int(dead_neuron_mask.sum())) == 0:
            return sae_out.new_tensor(0.0)

        if self.cfg.normalize_activations in ("constant_norm_rescale", "layer_norm"):
            raise ValueError(
                "AbsTopK auxiliary loss does not support activation normalization "
                f"(normalize_activations={self.cfg.normalize_activations!r}). "
                "The aux loss reconstruction would be in normalized space while the "
                "residual is in the original space, producing incorrect gradients."
            )

        residual = (sae_in - sae_out).detach()

        # Heuristic from Appendix B.1 in the TopK paper
        k_aux = sae_in.shape[-1] // 2

        # Reduce the scale of the loss if there are a small number of dead latents
        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)

        auxk_acts = calculate_abstopk_aux_acts(
            k_aux=k_aux,
            hidden_pre=hidden_pre,
            dead_neuron_mask=dead_neuron_mask,
        )

        # Encourage the top ~50% of dead latents to predict the residual of the
        # top k living latents. Per the TopK paper (Appendix A.2), the
        # reconstruction is ê = W_dec @ z (no bias), since b_dec is already in
        # the residual.
        recons = act_times_W_dec(
            auxk_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
        )
        # Apply the same reshaping as decode() so recons matches the residual's
        # shape
        recons = self.reshape_fn_out(recons, self.d_head)
        auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()
        return self.cfg.aux_loss_coefficient * scale * auxk_loss


def calculate_abstopk_aux_acts(
    k_aux: int,
    hidden_pre: torch.Tensor,
    dead_neuron_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Helper method to calculate activations for the auxiliary loss.

    Args:
        k_aux: Number of top dead neurons to select
        hidden_pre: Pre-activation values from encoder
        dead_neuron_mask: Boolean mask indicating which neurons are dead

    Returns:
        Tensor with signed activations for only the top-k dead neurons by
        magnitude, zeros elsewhere
    """

    # Don't include living latents in this loss
    auxk_latents = torch.where(dead_neuron_mask[None], hidden_pre.abs(), -torch.inf)
    # Top-k dead latents by magnitude
    auxk_topk = auxk_latents.topk(k_aux, sorted=False)
    # Gather the signed pre-activations and set the activations to zero for all
    # but the top k_aux dead latents
    auxk_acts = torch.zeros_like(hidden_pre)
    auxk_acts.scatter_(-1, auxk_topk.indices, hidden_pre.gather(-1, auxk_topk.indices))
    # Set activations to zero for all but top k_aux dead latents
    return auxk_acts
