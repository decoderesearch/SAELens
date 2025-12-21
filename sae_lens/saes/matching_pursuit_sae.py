"""Inference-only TopKSAE variant, similar in spirit to StandardSAE but using a TopK-based activation."""

from dataclasses import dataclass
from typing import Any

import torch
from typing_extensions import override

from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    TrainCoefficientConfig,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
    TrainStepOutput,
)
from sae_lens.saes.topk_sae import calculate_topk_aux_acts

# --- inference ---


@dataclass
class MatchingPursuitSAEConfig(SAEConfig):
    """
    Configuration class for MatchingPursuitSAE inference.

    Args:
        residual_threshold (float): residual error at which to stop selecting latents. Default 1e-2.
        max_iterations (int | None): Maximum iterations (default: d_in if set to None).
            Defaults to None.
        d_in (int): Input dimension (dimensionality of the activations being encoded).
            Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
            Defaults to "float32".
        device (str): Device to place the SAE on. Inherited from SAEConfig.
            Defaults to "cpu".
        apply_b_dec_to_input (bool): Whether to apply decoder bias to the input
            before encoding. Inherited from SAEConfig. Defaults to True.
        normalize_activations (Literal["none", "expected_average_only_in", "constant_norm_rescale", "layer_norm"]):
            Normalization strategy for input activations. Inherited from SAEConfig.
            Defaults to "none".
        reshape_activations (Literal["none", "hook_z"]): How to reshape activations
            (useful for attention head outputs). Inherited from SAEConfig.
            Defaults to "none".
        metadata (SAEMetadata): Metadata about the SAE (model name, hook name, etc.).
            Inherited from SAEConfig.
    """

    residual_threshold: float = 1e-2
    max_iterations: int | None = None

    @override
    @classmethod
    def architecture(cls) -> str:
        return "matching_pursuit"


class MatchingPursuitSAE(SAE[MatchingPursuitSAEConfig]):
    """
    An inference-only sparse autoencoder using a "matching pursuit" activation function.
    """

    # Matching pursuit is a tied SAE, so we use W_enc as the decoder transposed
    @property
    def W_enc(self) -> torch.Tensor:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.W_dec.T

    # hacky way to get around the base class having W_enc.
    # TODO: harmonize with the base class in next major release
    @override
    def __setattr__(self, name: str, value: Any):
        if name == "W_enc":
            return
        super().__setattr__(name, value)

    @override
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts input x into feature activations.
        """
        sae_in = self.process_sae_in(x)
        return _encode_matching_pursuit(sae_in, self.W_dec, self.cfg.residual_threshold)

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        raise NotImplementedError(
            "Folding W_dec_norm is not safe for MatchingPursuit SAEs, as this may change the resulting activations"
        )

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode the feature activations back to the input space.
        Now, if hook_z reshaping is turned on, we reverse the flattening.
        """
        sae_out_pre = feature_acts @ self.W_dec
        # since this is a tied SAE, we need to make sure b_dec is only applied if applied at input
        if self.cfg.apply_b_dec_to_input:
            sae_out_pre = sae_out_pre + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)


# --- training ---


@dataclass
class MatchingPursuitTrainingSAEConfig(TrainingSAEConfig):
    """
    Configuration class for training a MatchingPursitTrainingSAE.

    Args:
        residual_threshold (float): residual error at which to stop selecting latents. Default 1e-2.
        aux_loss_coefficient (float): Coefficient for the auxiliary loss that revives dead latents.
            Defaults to 1.0.
        use_gram (bool): Whether to use the Gram matrix for incremental correlation updates.
            This is faster per-iteration but uses O(d_sae^2) memory. Recommended for small d_sae.
            Defaults to False.
        max_iterations (int | None): Maximum iterations (default: d_in if set to None).
            Defaults to None.
        decoder_init_norm (float | None): Norm to initialize decoder weights to.
            0.1 corresponds to the "heuristic" initialization from Anthropic's April update.
            Use None to disable. Inherited from TrainingSAEConfig. Defaults to 0.1.
        d_in (int): Input dimension (dimensionality of the activations being encoded).
            Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
            Defaults to "float32".
        device (str): Device to place the SAE on. Inherited from SAEConfig.
            Defaults to "cpu".
        apply_b_dec_to_input (bool): Whether to apply decoder bias to the input
            before encoding. Inherited from SAEConfig. Defaults to True.
        normalize_activations (Literal["none", "expected_average_only_in", "constant_norm_rescale", "layer_norm"]):
            Normalization strategy for input activations. Inherited from SAEConfig.
            Defaults to "none".
        reshape_activations (Literal["none", "hook_z"]): How to reshape activations
            (useful for attention head outputs). Inherited from SAEConfig.
            Defaults to "none".
        metadata (SAEMetadata): Metadata about the SAE training (model name, hook name, etc.).
            Inherited from SAEConfig.
    """

    residual_threshold: float = 1e-2
    aux_loss_coefficient: float = 1.0
    use_gram: bool = False
    max_iterations: int | None = None

    @override
    @classmethod
    def architecture(cls) -> str:
        return "matching_pursuit"


class MatchingPursuitTrainingSAE(TrainingSAE[MatchingPursuitTrainingSAEConfig]):
    # Matching pursuit is a tied SAE, so we use W_enc as the decoder transposed
    @property
    def W_enc(self) -> torch.Tensor:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.W_dec.T

    # hacky way to get around the base class having W_enc.
    # TODO: harmonize with the base class in next major release
    @override
    def __setattr__(self, name: str, value: Any):
        if name == "W_enc":
            return
        super().__setattr__(name, value)

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_pre doesn't make sense for matching pursuit, since there is not a single pre-activation.
        We just return zeros for the hidden_pre.
        """

        sae_in = self.process_sae_in(x)
        acts = _encode_matching_pursuit(
            sae_in,
            self.W_dec,
            self.cfg.residual_threshold,
            use_gram=self.cfg.use_gram,
            max_iterations=self.cfg.max_iterations,
        )
        return acts, torch.zeros_like(acts)

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        raise NotImplementedError(
            "Folding W_dec_norm is not safe for MatchingPursuit SAEs, as this may change the resulting activations"
        )

    @override
    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        return {}

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Calculate the auxiliary loss for dead neurons
        aux_loss = self.calculate_topk_aux_loss(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            hidden_pre=hidden_pre,
            dead_neuron_mask=step_input.dead_neuron_mask,
        )
        return {"auxiliary_reconstruction_loss": aux_loss}

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)
        l0 = output.feature_acts.bool().float().sum(-1).to_dense()
        output.metrics["max_l0"] = l0.max()
        output.metrics["min_l0"] = l0.min()

        return output

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode the feature activations back to the input space.
        Now, if hook_z reshaping is turned on, we reverse the flattening.
        """
        sae_out_pre = feature_acts @ self.W_dec
        # since this is a tied SAE, we need to make sure b_dec is only applied if applied at input
        if self.cfg.apply_b_dec_to_input:
            sae_out_pre = sae_out_pre + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    def calculate_topk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Calculate auxiliary loss, based aux loss for topk SAEs.
        """
        # Mostly taken from https://github.com/EleutherAI/sae/blob/main/sae/sae.py, except without variance normalization
        # NOTE: checking the number of dead neurons will force a GPU sync, so performance can likely be improved here
        if dead_neuron_mask is None or (num_dead := int(dead_neuron_mask.sum())) == 0:
            return sae_out.new_tensor(0.0)
        residual = (sae_in - sae_out).detach()

        # Heuristic from Appendix B.1 in the paper
        k_aux = sae_in.shape[-1] // 2

        # Reduce the scale of the loss if there are a small number of dead latents
        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)

        auxk_acts = calculate_topk_aux_acts(
            k_aux=k_aux,
            hidden_pre=hidden_pre,
            dead_neuron_mask=dead_neuron_mask,
        )

        # Encourage the top ~50% of dead latents to predict the residual of the
        # top k living latents
        recons = self.decode(auxk_acts)
        auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()
        return self.cfg.aux_loss_coefficient * scale * auxk_loss


# --- shared ---


def _encode_matching_pursuit(
    sae_in_centered: torch.Tensor,
    W_dec: torch.Tensor,
    residual_threshold: float,
    max_iterations: int | None = None,
    use_gram: bool = False,
) -> torch.Tensor:
    """
    Matching pursuit encoding.

    Args:
        sae_in_centered: Input activations, centered by b_dec. Shape [..., d_in].
        W_dec: Decoder weight matrix. Shape [d_sae, d_in].
        residual_threshold: Stop when residual norm falls below this.
        max_iterations: Maximum iterations (default: d_in). Prevents infinite loops.
        use_gram: If True, compute the Gram matrix (W_dec @ W_dec.T) and use incremental
            correlation updates. This is faster per-iteration but uses O(d_sae^2) memory.
            Only recommended for small d_sae (e.g., during training with small SAEs).
    """
    residual = sae_in_centered.clone()

    # Handle multi-dimensional inputs by flattening all but the last dimension
    original_shape = residual.shape
    if residual.ndim > 2:
        residual = residual.reshape(-1, residual.shape[-1])

    batch_size = residual.shape[0]
    d_sae, d_in = W_dec.shape

    if max_iterations is None:
        max_iterations = d_in  # Sensible upper bound

    acts = torch.zeros(batch_size, d_sae, device=W_dec.device, dtype=residual.dtype)
    prev_support = torch.zeros(batch_size, d_sae, dtype=torch.bool, device=W_dec.device)
    done = torch.zeros(batch_size, dtype=torch.bool, device=W_dec.device)

    # Optionally use Gram matrix for incremental correlation updates
    W_dec_gram: torch.Tensor | None = None
    correlations: torch.Tensor | None = None
    if use_gram:
        W_dec_gram = W_dec @ W_dec.T
        correlations = residual @ W_dec.T

    for _ in range(max_iterations):
        if use_gram and correlations is not None:
            with torch.no_grad():
                indices = correlations.relu().max(dim=1, keepdim=True).indices
                indices_flat = indices.squeeze(1)
        else:
            # Find indices without gradients - the full [batch, d_sae] matmul result
            # doesn't need to be saved for backward since max indices don't need gradients
            with torch.no_grad():
                indices = (residual @ W_dec.T).relu().max(dim=1, keepdim=True).indices
                indices_flat = indices.squeeze(1)  # [batch_size]

        # Compute values with gradients using only the selected decoder rows.
        # This stores [batch, d_in] for backward instead of [batch, d_sae].
        selected_dec = W_dec[indices_flat]  # [batch_size, d_in]
        values = (residual * selected_dec).sum(dim=-1, keepdim=True).relu()

        # Mask values for samples that are already done
        active_mask = (~done).unsqueeze(1)
        masked_values = (values * active_mask.to(values.dtype)).to(acts.dtype)

        acts.scatter_add_(1, indices, masked_values)

        # Update residual
        residual = residual - masked_values * selected_dec

        # Incremental correlation update using Gram matrix
        if use_gram and W_dec_gram is not None and correlations is not None:
            with torch.no_grad():
                gram_rows = W_dec_gram[indices_flat]
            correlations = correlations - masked_values * gram_rows

        with torch.no_grad():
            support = acts != 0

            # A sample is considered converged if:
            # (1) the support set hasn't changed from the previous iteration (stability), or
            # (2) the residual norm is below a given threshold (good enough reconstruction)
            converged = (support == prev_support).all(dim=1) | (
                residual.norm(dim=-1) < residual_threshold
            )
            done = done | converged
            prev_support = support

            if done.all():
                break

    # Reshape acts back to original shape (replacing last dimension with d_sae)
    if len(original_shape) > 2:
        acts = acts.reshape(*original_shape[:-1], acts.shape[-1])

    return acts
