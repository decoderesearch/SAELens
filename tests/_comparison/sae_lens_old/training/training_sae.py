"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""

from dataclasses import dataclass, fields
from typing import Any

import einops
import numpy as np
import torch
from torch import nn
from typing_extensions import deprecated

from tests._comparison.sae_lens_old import logger
from tests._comparison.sae_lens_old.config import LanguageModelSAERunnerConfig
from tests._comparison.sae_lens_old.sae import SAE, SAEConfig
from tests._comparison.sae_lens_old.toolkit.pretrained_sae_loaders import (
    PretrainedSaeDiskLoader,
    handle_config_defaulting,
    sae_lens_disk_loader,
)


def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)


class Step(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,  # noqa: ARG004
    ) -> torch.Tensor:
        return (x > threshold).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        threshold_grad = torch.sum(
            -(1.0 / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output,
            dim=0,
        )
        return None, threshold_grad, None


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,  # noqa: ARG004
    ) -> torch.Tensor:
        return (x * (x > threshold)).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold) * grad_output  # We don't apply STE to x input
        threshold_grad = torch.sum(
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None


@dataclass
class TrainStepOutput:
    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    hidden_pre: torch.Tensor
    loss: torch.Tensor  # we need to call backwards on this
    losses: dict[str, float | torch.Tensor]


@dataclass(kw_only=True)
class TrainingSAEConfig(SAEConfig):
    # Sparsity Loss Calculations
    l1_coefficient: float
    lp_norm: float
    use_ghost_grads: bool
    normalize_sae_decoder: bool
    noise_scale: float
    decoder_orthogonal_init: bool
    mse_loss_normalization: str | None
    jumprelu_init_threshold: float
    jumprelu_bandwidth: float
    decoder_heuristic_init: bool
    decoder_heuristic_init_norm: float
    init_encoder_as_decoder_transpose: bool
    scale_sparsity_penalty_by_decoder_norm: bool

    @classmethod
    def from_sae_runner_config(
        cls, cfg: LanguageModelSAERunnerConfig
    ) -> "TrainingSAEConfig":
        return cls(
            # base config
            architecture=cfg.architecture,
            d_in=cfg.d_in,
            d_sae=cfg.d_sae,  # type: ignore
            dtype=cfg.dtype,
            device=cfg.device,
            model_name=cfg.model_name,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            hook_head_index=cfg.hook_head_index,
            activation_fn_str=cfg.activation_fn,
            activation_fn_kwargs=cfg.activation_fn_kwargs,
            apply_b_dec_to_input=cfg.apply_b_dec_to_input,
            finetuning_scaling_factor=cfg.finetuning_method is not None,
            sae_lens_training_version=cfg.sae_lens_training_version,
            context_size=cfg.context_size,
            dataset_path=cfg.dataset_path,
            prepend_bos=cfg.prepend_bos,
            seqpos_slice=cfg.seqpos_slice,
            # Training cfg
            l1_coefficient=cfg.l1_coefficient,
            lp_norm=cfg.lp_norm,
            use_ghost_grads=cfg.use_ghost_grads,
            normalize_sae_decoder=cfg.normalize_sae_decoder,
            noise_scale=cfg.noise_scale,
            decoder_orthogonal_init=cfg.decoder_orthogonal_init,
            mse_loss_normalization=cfg.mse_loss_normalization,
            decoder_heuristic_init=cfg.decoder_heuristic_init,
            decoder_heuristic_init_norm=cfg.decoder_heuristic_init_norm,
            init_encoder_as_decoder_transpose=cfg.init_encoder_as_decoder_transpose,
            scale_sparsity_penalty_by_decoder_norm=cfg.scale_sparsity_penalty_by_decoder_norm,
            normalize_activations=cfg.normalize_activations,
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
            model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs or {},
            jumprelu_init_threshold=cfg.jumprelu_init_threshold,
            jumprelu_bandwidth=cfg.jumprelu_bandwidth,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSAEConfig":
        # remove any keys that are not in the dataclass
        # since we sometimes enhance the config with the whole LM runner config
        valid_field_names = {field.name for field in fields(cls)}
        valid_config_dict = {
            key: val for key, val in config_dict.items() if key in valid_field_names
        }

        # ensure seqpos slice is tuple
        # ensure that seqpos slices is a tuple
        # Ensure seqpos_slice is a tuple
        if "seqpos_slice" in valid_config_dict:
            if isinstance(valid_config_dict["seqpos_slice"], list):
                valid_config_dict["seqpos_slice"] = tuple(
                    valid_config_dict["seqpos_slice"]
                )
            elif not isinstance(valid_config_dict["seqpos_slice"], tuple):
                valid_config_dict["seqpos_slice"] = (valid_config_dict["seqpos_slice"],)

        return TrainingSAEConfig(**valid_config_dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "l1_coefficient": self.l1_coefficient,
            "lp_norm": self.lp_norm,
            "use_ghost_grads": self.use_ghost_grads,
            "normalize_sae_decoder": self.normalize_sae_decoder,
            "noise_scale": self.noise_scale,
            "decoder_orthogonal_init": self.decoder_orthogonal_init,
            "init_encoder_as_decoder_transpose": self.init_encoder_as_decoder_transpose,
            "mse_loss_normalization": self.mse_loss_normalization,
            "decoder_heuristic_init": self.decoder_heuristic_init,
            "decoder_heuristic_init_norm": self.decoder_heuristic_init_norm,
            "scale_sparsity_penalty_by_decoder_norm": self.scale_sparsity_penalty_by_decoder_norm,
            "normalize_activations": self.normalize_activations,
            "jumprelu_init_threshold": self.jumprelu_init_threshold,
            "jumprelu_bandwidth": self.jumprelu_bandwidth,
        }

    # this needs to exist so we can initialize the parent sae cfg without the training specific
    # parameters. Maybe there's a cleaner way to do this
    def get_base_sae_cfg_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "activation_fn_str": self.activation_fn_str,
            "activation_fn_kwargs": self.activation_fn_kwargs,
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "dtype": self.dtype,
            "model_name": self.model_name,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "hook_head_index": self.hook_head_index,
            "device": self.device,
            "context_size": self.context_size,
            "prepend_bos": self.prepend_bos,
            "finetuning_scaling_factor": self.finetuning_scaling_factor,
            "normalize_activations": self.normalize_activations,
            "dataset_path": self.dataset_path,
            "dataset_trust_remote_code": self.dataset_trust_remote_code,
            "sae_lens_training_version": self.sae_lens_training_version,
        }


class TrainingSAE(SAE):
    """
    A SAE used for training. This class provides a `training_forward_pass` method which calculates
    losses used for training.
    """

    cfg: TrainingSAEConfig
    use_error_term: bool
    dtype: torch.dtype
    device: torch.device

    def __init__(self, cfg: TrainingSAEConfig, use_error_term: bool = False):
        base_sae_cfg = SAEConfig.from_dict(cfg.get_base_sae_cfg_dict())
        super().__init__(base_sae_cfg)
        self.cfg = cfg  # type: ignore

        if cfg.architecture == "standard" or cfg.architecture == "topk":
            self.encode_with_hidden_pre_fn = self.encode_with_hidden_pre
        elif cfg.architecture == "gated":
            self.encode_with_hidden_pre_fn = self.encode_with_hidden_pre_gated
        elif cfg.architecture == "jumprelu":
            self.encode_with_hidden_pre_fn = self.encode_with_hidden_pre_jumprelu
            self.bandwidth = cfg.jumprelu_bandwidth
            self.log_threshold.data = torch.ones(
                self.cfg.d_sae, dtype=self.dtype, device=self.device
            ) * np.log(cfg.jumprelu_init_threshold)

        else:
            raise ValueError(f"Unknown architecture: {cfg.architecture}")

        self.check_cfg_compatibility()

        self.use_error_term = use_error_term

        self.initialize_weights_complex()

        # The training SAE will assume that the activation store handles
        # reshaping.
        self.turn_off_forward_pass_hook_z_reshaping()

        self.mse_loss_fn = self._get_mse_loss_fn()

    def initialize_weights_jumprelu(self):
        # same as the superclass, except we use a log_threshold parameter instead of threshold
        self.log_threshold = nn.Parameter(
            torch.empty(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        self.initialize_weights_basic()

    @property
    def threshold(self) -> torch.Tensor:
        if self.cfg.architecture != "jumprelu":
            raise ValueError("Threshold is only defined for Jumprelu SAEs")
        return torch.exp(self.log_threshold)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSAE":
        return cls(TrainingSAEConfig.from_dict(config_dict))

    def check_cfg_compatibility(self):
        if self.cfg.architecture != "standard" and self.cfg.use_ghost_grads:
            raise ValueError(f"{self.cfg.architecture} SAEs do not support ghost grads")
        if self.cfg.architecture == "gated" and self.use_error_term:
            raise ValueError("Gated SAEs do not support error terms")

    def encode_standard(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcuate SAE features from inputs
        """
        feature_acts, _ = self.encode_with_hidden_pre_fn(x)
        return feature_acts

    def encode_with_hidden_pre_jumprelu(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        hidden_pre = sae_in @ self.W_enc + self.b_enc

        if self.training:
            hidden_pre = (
                hidden_pre + torch.randn_like(hidden_pre) * self.cfg.noise_scale
            )

        threshold = torch.exp(self.log_threshold)

        feature_acts = JumpReLU.apply(hidden_pre, threshold, self.bandwidth)

        return feature_acts, hidden_pre  # type: ignore

    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        hidden_pre_noised = hidden_pre + (
            torch.randn_like(hidden_pre) * self.cfg.noise_scale * self.training
        )
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre_noised))

        return feature_acts, hidden_pre_noised

    def encode_with_hidden_pre_gated(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        # Gating path with Heaviside step function
        gating_pre_activation = sae_in @ self.W_enc + self.b_gate
        active_features = (gating_pre_activation > 0).to(self.dtype)

        # Magnitude path with weight sharing
        magnitude_pre_activation = sae_in @ (self.W_enc * self.r_mag.exp()) + self.b_mag
        # magnitude_pre_activation_noised = magnitude_pre_activation + (
        #     torch.randn_like(magnitude_pre_activation) * self.cfg.noise_scale * self.training
        # )
        feature_magnitudes = self.activation_fn(
            magnitude_pre_activation
        )  # magnitude_pre_activation_noised)

        # Return both the gated feature activations and the magnitude pre-activations
        return (
            active_features * feature_magnitudes,
            magnitude_pre_activation,
        )  # magnitude_pre_activation_noised

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        feature_acts, _ = self.encode_with_hidden_pre_fn(x)
        return self.decode(feature_acts)

    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: torch.Tensor | None = None,
    ) -> TrainStepOutput:
        # do a forward pass to get SAE out, but we also need the
        # hidden pre.
        feature_acts, hidden_pre = self.encode_with_hidden_pre_fn(sae_in)
        sae_out = self.decode(feature_acts)

        # MSE LOSS
        per_item_mse_loss = self.mse_loss_fn(sae_out, sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        losses: dict[str, float | torch.Tensor] = {}

        if self.cfg.architecture == "gated":
            # Gated SAE Loss Calculation

            # Shared variables
            sae_in_centered = (
                self.reshape_fn_in(sae_in) - self.b_dec * self.cfg.apply_b_dec_to_input
            )
            pi_gate = sae_in_centered @ self.W_enc + self.b_gate
            pi_gate_act = torch.relu(pi_gate)

            # SFN sparsity loss - summed over the feature dimension and averaged over the batch
            l1_loss = (
                current_l1_coefficient
                * torch.sum(pi_gate_act * self.W_dec.norm(dim=1), dim=-1).mean()
            )

            # Auxiliary reconstruction loss - summed over the feature dimension and averaged over the batch
            via_gate_reconstruction = pi_gate_act @ self.W_dec + self.b_dec
            aux_reconstruction_loss = torch.sum(
                (via_gate_reconstruction - sae_in) ** 2, dim=-1
            ).mean()
            loss = mse_loss + l1_loss + aux_reconstruction_loss
            losses["auxiliary_reconstruction_loss"] = aux_reconstruction_loss
            losses["l1_loss"] = l1_loss
        elif self.cfg.architecture == "jumprelu":
            threshold = torch.exp(self.log_threshold)
            l0 = torch.sum(Step.apply(hidden_pre, threshold, self.bandwidth), dim=-1)  # type: ignore
            l0_loss = (current_l1_coefficient * l0).mean()
            loss = mse_loss + l0_loss
            losses["l0_loss"] = l0_loss
        elif self.cfg.architecture == "topk":
            topk_loss = self.calculate_topk_aux_loss(
                sae_in=sae_in,
                sae_out=sae_out,
                hidden_pre=hidden_pre,
                dead_neuron_mask=dead_neuron_mask,
            )
            losses["auxiliary_reconstruction_loss"] = topk_loss
            loss = mse_loss + topk_loss
        else:
            # default SAE sparsity loss
            weighted_feature_acts = feature_acts
            if self.cfg.scale_sparsity_penalty_by_decoder_norm:
                weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)
            sparsity = weighted_feature_acts.norm(
                p=self.cfg.lp_norm, dim=-1
            )  # sum over the feature dimension

            l1_loss = (current_l1_coefficient * sparsity).mean()
            loss = mse_loss + l1_loss
            if (
                self.cfg.use_ghost_grads
                and self.training
                and dead_neuron_mask is not None
            ):
                ghost_grad_loss = self.calculate_ghost_grad_loss(
                    x=sae_in,
                    sae_out=sae_out,
                    per_item_mse_loss=per_item_mse_loss,
                    hidden_pre=hidden_pre,
                    dead_neuron_mask=dead_neuron_mask,
                )
                losses["ghost_grad_loss"] = ghost_grad_loss
                loss = loss + ghost_grad_loss
            losses["l1_loss"] = l1_loss

        losses["mse_loss"] = mse_loss

        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=loss,
            losses=losses,
        )

    def calculate_topk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor | None,
    ) -> torch.Tensor:
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

        auxk_acts = _calculate_topk_aux_acts(
            k_aux=k_aux,
            hidden_pre=hidden_pre,
            dead_neuron_mask=dead_neuron_mask,
        )

        # Encourage the top ~50% of dead latents to predict the residual of the
        # top k living latents
        recons = self.decode(auxk_acts)
        auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()
        return scale * auxk_loss

    def calculate_ghost_grad_loss(
        self,
        x: torch.Tensor,
        sae_out: torch.Tensor,
        per_item_mse_loss: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 1.
        residual = x - sae_out
        l2_norm_residual = torch.norm(residual, dim=-1)

        # 2.
        # ghost grads use an exponentional activation function, ignoring whatever
        # the activation function is in the SAE. The forward pass uses the dead neurons only.
        feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_neuron_mask])
        ghost_out = feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask, :]
        l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
        norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
        ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

        # 3. There is some fairly complex rescaling here to make sure that the loss
        # is comparable to the original loss. This is because the ghost grads are
        # only calculated for the dead neurons, so we need to rescale the loss to
        # make sure that the loss is comparable to the original loss.
        # There have been methodological improvements that are not implemented here yet
        # see here: https://www.lesswrong.com/posts/C5KAZQib3bzzpeyrg/full-post-progress-update-1-from-the-gdm-mech-interp-team#Improving_ghost_grads
        per_item_mse_loss_ghost_resid = self.mse_loss_fn(ghost_out, residual.detach())
        mse_rescaling_factor = (
            per_item_mse_loss / (per_item_mse_loss_ghost_resid + 1e-6)
        ).detach()
        per_item_mse_loss_ghost_resid = (
            mse_rescaling_factor * per_item_mse_loss_ghost_resid
        )

        return per_item_mse_loss_ghost_resid.mean()

    @torch.no_grad()
    def _get_mse_loss_fn(self) -> Any:
        def standard_mse_loss_fn(
            preds: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            return torch.nn.functional.mse_loss(preds, target, reduction="none")

        def batch_norm_mse_loss_fn(
            preds: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            target_centered = target - target.mean(dim=0, keepdim=True)
            normalization = target_centered.norm(dim=-1, keepdim=True)
            return torch.nn.functional.mse_loss(preds, target, reduction="none") / (
                normalization + 1e-6
            )

        if self.cfg.mse_loss_normalization == "dense_batch":
            return batch_norm_mse_loss_fn
        return standard_mse_loss_fn

    def process_state_dict_for_saving(self, state_dict: dict[str, Any]) -> None:
        if self.cfg.architecture == "jumprelu" and "log_threshold" in state_dict:
            threshold = torch.exp(state_dict["log_threshold"]).detach().contiguous()
            del state_dict["log_threshold"]
            state_dict["threshold"] = threshold

    def process_state_dict_for_loading(self, state_dict: dict[str, Any]) -> None:
        if self.cfg.architecture == "jumprelu" and "threshold" in state_dict:
            threshold = state_dict["threshold"]
            del state_dict["threshold"]
            state_dict["log_threshold"] = torch.log(threshold).detach().contiguous()

    @classmethod
    @deprecated("Use load_from_disk instead")
    def load_from_pretrained(
        cls, path: str, device: str = "cpu", dtype: str | None = None
    ) -> "TrainingSAE":
        return cls.load_from_disk(path, device, dtype)

    @classmethod
    def load_from_disk(
        cls,
        path: str,
        device: str = "cpu",
        dtype: str | None = None,
        converter: PretrainedSaeDiskLoader = sae_lens_disk_loader,
    ) -> "TrainingSAE":
        overrides = {"dtype": dtype} if dtype is not None else None
        cfg_dict, state_dict = converter(path, device, cfg_overrides=overrides)
        cfg_dict = handle_config_defaulting(cfg_dict)
        sae_cfg = TrainingSAEConfig.from_dict(cfg_dict)
        sae = cls(sae_cfg)
        sae.process_state_dict_for_loading(state_dict)
        sae.load_state_dict(state_dict)
        return sae

    def initialize_weights_complex(self):
        """ """

        if self.cfg.decoder_orthogonal_init:
            self.W_dec.data = nn.init.orthogonal_(self.W_dec.data.T).T

        elif self.cfg.decoder_heuristic_init:
            self.W_dec = nn.Parameter(
                torch.randn(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
            self.initialize_decoder_norm_constant_norm(
                self.cfg.decoder_heuristic_init_norm
            )

        # Then we initialize the encoder weights (either as the transpose of decoder or not)
        if self.cfg.init_encoder_as_decoder_transpose:
            self.W_enc.data = self.W_dec.data.T.clone().contiguous()
        else:
            self.W_enc = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(
                        self.cfg.d_in,
                        self.cfg.d_sae,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
            )

        if self.cfg.normalize_sae_decoder:
            with torch.no_grad():
                # Anthropic normalize this to have unit columns
                self.set_decoder_norm_to_unit_norm()

    @torch.no_grad()
    def fold_W_dec_norm(self):
        # need to deal with the jumprelu having a log_threshold in training
        if self.cfg.architecture == "jumprelu":
            cur_threshold = self.threshold.clone()
            W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
            super().fold_W_dec_norm()
            self.log_threshold.data = torch.log(cur_threshold * W_dec_norms.squeeze())
        else:
            super().fold_W_dec_norm()

    ## Initialization Methods
    @torch.no_grad()
    def initialize_b_dec_with_precalculated(self, origin: torch.Tensor):
        out = torch.tensor(origin, dtype=self.dtype, device=self.device)
        self.b_dec.data = out

    @torch.no_grad()
    def initialize_b_dec_with_mean(self, all_activations: torch.Tensor):
        previous_b_dec = self.b_dec.clone().cpu()
        out = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        logger.info("Reinitializing b_dec with mean of activations")
        logger.debug(
            f"Previous distances: {previous_distances.median(0).values.mean().item()}"
        )
        logger.debug(f"New distances: {distances.median(0).values.mean().item()}")

        self.b_dec.data = out.to(self.dtype).to(self.device)

    ## Training Utils
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def initialize_decoder_norm_constant_norm(self, norm: float = 0.1):
        """
        A heuristic proceedure inspired by:
        https://transformer-circuits.pub/2024/april-update/index.html#training-saes
        """
        # TODO: Parameterise this as a function of m and n

        # ensure W_dec norms at unit norm
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data *= norm  # will break tests but do this for now.

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )


def _calculate_topk_aux_acts(
    k_aux: int,
    hidden_pre: torch.Tensor,
    dead_neuron_mask: torch.Tensor,
) -> torch.Tensor:
    # Don't include living latents in this loss
    auxk_latents = torch.where(dead_neuron_mask[None], hidden_pre, -torch.inf)
    # Top-k dead latents
    auxk_topk = auxk_latents.topk(k_aux, sorted=False)
    # Set the activations to zero for all but the top k_aux dead latents
    auxk_acts = torch.zeros_like(hidden_pre)
    auxk_acts.scatter_(-1, auxk_topk.indices, auxk_topk.values)
    # Set activations to zero for all but top k_aux dead latents
    return auxk_acts
