"""Phase-Multiplexed Sparse Autoencoder (PhaseSAE).

A time-division multiplexed dictionary learning architecture that tiles the tensor
space across phase-clocked temporal micro-ticks, trading latency for memory bandwidth.

Instead of materializing a monolithic d_in x d_sae matrix multiply in a single pass,
PhaseSAE partitions the feature space into P phase tiles {W^(0), ..., W^(P-1)}.
Each micro-tick operates on a 1/P parameter slice against the residual error of
prior phases, enforcing progressive orthogonalization while keeping peak streaming
memory bandwidth bounded to 1/P.
"""

from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from typing_extensions import override

from sae_lens.registry import register_sae_class, register_sae_training_class
from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
    TrainStepOutput,
    _disable_hooks,
)
from sae_lens.saes.topk_sae import SparseHookPoint, TopK, act_times_W_dec


@dataclass
class PhaseMultiplexedSAEConfig(SAEConfig):
    """Configuration class for PhaseMultiplexedSAE inference.

    Args:
        num_phases (int): Number of sequential phase tiles P. Defaults to 4.
        k_per_phase (int): Sparsity budget K/P per phase tile. Defaults to 8.
        phase_mode (Literal["cascaded_ticks", "sequence_tdm", "resonant_clock"]):
            TDM execution mode. Defaults to "cascaded_ticks".
        rescale_acts_by_decoder_norm (bool): Whether to rescale pre-acts by decoder norm.
        use_sparse_activations (bool): Whether to produce sparse COO tensors.
        d_in (int): Input dimension. Inherited from SAEConfig.
        d_sae (int): Total SAE latent dimension (must be divisible by num_phases).
        dtype (str): Data type. Inherited from SAEConfig.
        device (str): Device. Inherited from SAEConfig.
        apply_b_dec_to_input (bool): Centering with decoder bias.
        normalize_activations (str): Normalization strategy.
        reshape_activations (str): Reshaping strategy.
        metadata (SAEMetadata): Metadata.
    """

    num_phases: int = 4
    k_per_phase: int = 8
    phase_mode: Literal["cascaded_ticks", "sequence_tdm", "resonant_clock"] = (
        "cascaded_ticks"
    )
    rescale_acts_by_decoder_norm: bool = False
    use_sparse_activations: bool = False
    exit_threshold: float | None = None

    @override
    @classmethod
    def architecture(cls) -> str:
        return "phase_multiplexed"

    @property
    def d_sae_per_phase(self) -> int:
        return self.d_sae // self.num_phases

    @property
    def total_k(self) -> int:
        return self.k_per_phase * self.num_phases

    def __post_init__(self):
        super().__post_init__()
        if self.num_phases < 1:
            raise ValueError(f"num_phases must be >= 1, got {self.num_phases}")
        if self.d_sae % self.num_phases != 0:
            raise ValueError(
                f"d_sae ({self.d_sae}) must be divisible by num_phases ({self.num_phases})"
            )
        if self.k_per_phase > self.d_sae_per_phase:
            raise ValueError(
                f"k_per_phase ({self.k_per_phase}) cannot exceed d_sae_per_phase ({self.d_sae_per_phase})"
            )


class PhaseMultiplexedSAE(SAE[PhaseMultiplexedSAEConfig]):
    """An inference-only phase-multiplexed sparse autoencoder.

    Evaluates P sequential micro-ticks over residuals, progressively reconstructing
    the input vector while constraining peak active tensor memory to 1/P of the full dictionary.
    """

    b_enc: nn.Parameter

    def __init__(self, cfg: PhaseMultiplexedSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        if self.cfg.use_sparse_activations:
            self.hook_sae_acts_post = SparseHookPoint(self.cfg.d_sae)
            self.setup()

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return TopK(
            self.cfg.k_per_phase,
            use_sparse_activations=False,
        )

    @override
    @override
    def encode(
        self, x: torch.Tensor, exit_threshold: float | None = None
    ) -> torch.Tensor:
        """Converts input x into feature activations via cascaded phase micro-ticks."""
        sae_in = self.process_sae_in(x)
        cur_residual = sae_in
        accumulated_recon = torch.zeros_like(sae_in)
        feature_acts = torch.zeros(
            *x.shape[:-1], self.cfg.d_sae, dtype=self.dtype, device=x.device
        )

        threshold = (
            exit_threshold if exit_threshold is not None else self.cfg.exit_threshold
        )
        m = self.cfg.d_sae_per_phase
        for p in range(self.cfg.num_phases):
            start_idx = p * m
            end_idx = (p + 1) * m

            w_enc_p = self.W_enc[:, start_idx:end_idx]
            b_enc_p = self.b_enc[start_idx:end_idx]
            w_dec_p = self.W_dec[start_idx:end_idx, :]

            # Project current residual onto phase tile
            pre_p = cur_residual @ w_enc_p + b_enc_p
            if self.cfg.rescale_acts_by_decoder_norm:
                pre_p = pre_p * w_dec_p.norm(dim=-1)

            # Sparsity activation for this phase
            acts_p = self.activation_fn(pre_p)
            feature_acts[..., start_idx:end_idx] = acts_p

            # Partial decode & residual update for next micro-tick
            recon_p = act_times_W_dec(
                acts_p, w_dec_p, self.cfg.rescale_acts_by_decoder_norm
            )
            accumulated_recon = accumulated_recon + recon_p
            cur_residual = sae_in - accumulated_recon

            if threshold is not None:
                residual_ratio = cur_residual.norm(dim=-1) / (
                    sae_in.norm(dim=-1) + 1e-8
                )
                if (residual_ratio < threshold).all():
                    break

        return self.hook_sae_acts_post(feature_acts)

    def stream_phase_ticks(
        self, x: torch.Tensor, exit_threshold: float | None = None
    ) -> Generator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """Time-Division Multiplexing generator.

        Yields (phase_idx, phase_acts, partial_recon, cur_residual) tick by tick,
        allowing streaming consumption with 1/P memory bandwidth.
        """
        sae_in = self.process_sae_in(x)
        cur_residual = sae_in
        accumulated_recon = torch.zeros_like(sae_in)
        m = self.cfg.d_sae_per_phase
        threshold = (
            exit_threshold if exit_threshold is not None else self.cfg.exit_threshold
        )

        for p in range(self.cfg.num_phases):
            start_idx = p * m
            end_idx = (p + 1) * m

            w_enc_p = self.W_enc[:, start_idx:end_idx]
            b_enc_p = self.b_enc[start_idx:end_idx]
            w_dec_p = self.W_dec[start_idx:end_idx, :]

            pre_p = cur_residual @ w_enc_p + b_enc_p
            if self.cfg.rescale_acts_by_decoder_norm:
                pre_p = pre_p * w_dec_p.norm(dim=-1)

            acts_p = self.activation_fn(pre_p)
            recon_p = act_times_W_dec(
                acts_p, w_dec_p, self.cfg.rescale_acts_by_decoder_norm
            )
            accumulated_recon = accumulated_recon + recon_p
            cur_residual = sae_in - accumulated_recon

            yield (p, acts_p, recon_p, cur_residual)

            if threshold is not None:
                residual_ratio = cur_residual.norm(dim=-1) / (
                    sae_in.norm(dim=-1) + 1e-8
                )
                if (residual_ratio < threshold).all():
                    break

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode feature activations back to the input space."""
        sae_out_pre = act_times_W_dec(
            feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
        )
        if self.cfg.apply_b_dec_to_input:
            sae_out_pre = sae_out_pre + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @override
    def forward(
        self, x: torch.Tensor, exit_threshold: float | None = None
    ) -> torch.Tensor:
        feature_acts = self.encode(x, exit_threshold=exit_threshold)
        sae_out = self.decode(feature_acts)

        if self.use_error_term:
            with torch.no_grad():
                with _disable_hooks(self):
                    feature_acts_clean = self.encode(x, exit_threshold=exit_threshold)
                    x_reconstruct_clean = self.decode(feature_acts_clean)
                sae_error = self.hook_sae_error(x - x_reconstruct_clean)
            sae_out = sae_out + sae_error

        return self.hook_sae_output(sae_out)

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        if not self.cfg.rescale_acts_by_decoder_norm:
            raise NotImplementedError(
                "Folding W_dec_norm is not safe for PhaseMultiplexedSAE when rescale_acts_by_decoder_norm is False"
            )
        super().fold_W_dec_norm()


# --- Training ---


@dataclass(kw_only=True)
class PhaseMultiplexedTrainingSAEConfig(TrainingSAEConfig):
    """Configuration class for training a PhaseMultiplexedTrainingSAE."""

    num_phases: int = 4
    k_per_phase: int = 8
    phase_mode: Literal["cascaded_ticks", "sequence_tdm", "resonant_clock"] = (
        "cascaded_ticks"
    )
    rescale_acts_by_decoder_norm: bool = False
    use_sparse_activations: bool = False
    aux_loss_coefficient: float = 1.0
    exit_threshold: float | None = None

    @override
    @classmethod
    def architecture(cls) -> str:
        return "phase_multiplexed"

    @property
    def d_sae_per_phase(self) -> int:
        return self.d_sae // self.num_phases

    @property
    def total_k(self) -> int:
        return self.k_per_phase * self.num_phases

    def __post_init__(self):
        super().__post_init__()
        if self.num_phases < 1:
            raise ValueError(f"num_phases must be >= 1, got {self.num_phases}")
        if self.d_sae % self.num_phases != 0:
            raise ValueError(
                f"d_sae ({self.d_sae}) must be divisible by num_phases ({self.num_phases})"
            )
        if self.k_per_phase > self.d_sae_per_phase:
            raise ValueError(
                f"k_per_phase ({self.k_per_phase}) cannot exceed d_sae_per_phase ({self.d_sae_per_phase})"
            )


class PhaseMultiplexedTrainingSAE(TrainingSAE[PhaseMultiplexedTrainingSAEConfig]):
    """Phase-Multiplexed SAE with training step logic, per-phase metrics, and backprop."""

    b_enc: nn.Parameter

    def __init__(
        self, cfg: PhaseMultiplexedTrainingSAEConfig, use_error_term: bool = False
    ):
        super().__init__(cfg, use_error_term)
        if self.cfg.use_sparse_activations:
            self.hook_sae_acts_post = SparseHookPoint(self.cfg.d_sae)
            self.setup()

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return TopK(
            self.cfg.k_per_phase,
            use_sparse_activations=False,
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor, exit_threshold: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        cur_residual = sae_in
        accumulated_recon = torch.zeros_like(sae_in)
        feature_acts = torch.zeros(
            *x.shape[:-1], self.cfg.d_sae, dtype=self.dtype, device=x.device
        )
        hidden_pre = torch.zeros(
            *x.shape[:-1], self.cfg.d_sae, dtype=self.dtype, device=x.device
        )

        threshold = (
            exit_threshold if exit_threshold is not None else self.cfg.exit_threshold
        )
        m = self.cfg.d_sae_per_phase
        for p in range(self.cfg.num_phases):
            start_idx = p * m
            end_idx = (p + 1) * m

            w_enc_p = self.W_enc[:, start_idx:end_idx]
            b_enc_p = self.b_enc[start_idx:end_idx]
            w_dec_p = self.W_dec[start_idx:end_idx, :]

            pre_p = cur_residual @ w_enc_p + b_enc_p
            if self.cfg.rescale_acts_by_decoder_norm:
                pre_p = pre_p * w_dec_p.norm(dim=-1)
            hidden_pre[..., start_idx:end_idx] = pre_p

            acts_p = self.activation_fn(pre_p)
            feature_acts[..., start_idx:end_idx] = acts_p

            recon_p = act_times_W_dec(
                acts_p, w_dec_p, self.cfg.rescale_acts_by_decoder_norm
            )
            accumulated_recon = accumulated_recon + recon_p
            cur_residual = sae_in - accumulated_recon

            if threshold is not None:
                residual_ratio = cur_residual.norm(dim=-1) / (
                    sae_in.norm(dim=-1) + 1e-8
                )
                if (residual_ratio < threshold).all():
                    break

        return (
            self.hook_sae_acts_post(feature_acts),
            self.hook_sae_acts_pre(hidden_pre),
        )

    @override
    def encode(
        self, x: torch.Tensor, exit_threshold: float | None = None
    ) -> torch.Tensor:
        acts, _ = self.encode_with_hidden_pre(x, exit_threshold=exit_threshold)
        return acts

    def stream_phase_ticks(
        self, x: torch.Tensor, exit_threshold: float | None = None
    ) -> Generator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """Time-Division Multiplexing generator for training SAE."""
        sae_in = self.process_sae_in(x)
        cur_residual = sae_in
        accumulated_recon = torch.zeros_like(sae_in)
        m = self.cfg.d_sae_per_phase
        threshold = (
            exit_threshold if exit_threshold is not None else self.cfg.exit_threshold
        )

        for p in range(self.cfg.num_phases):
            start_idx = p * m
            end_idx = (p + 1) * m

            w_enc_p = self.W_enc[:, start_idx:end_idx]
            b_enc_p = self.b_enc[start_idx:end_idx]
            w_dec_p = self.W_dec[start_idx:end_idx, :]

            pre_p = cur_residual @ w_enc_p + b_enc_p
            if self.cfg.rescale_acts_by_decoder_norm:
                pre_p = pre_p * w_dec_p.norm(dim=-1)

            acts_p = self.activation_fn(pre_p)
            recon_p = act_times_W_dec(
                acts_p, w_dec_p, self.cfg.rescale_acts_by_decoder_norm
            )
            accumulated_recon = accumulated_recon + recon_p
            cur_residual = sae_in - accumulated_recon

            yield (p, acts_p, recon_p, cur_residual)

            if threshold is not None:
                residual_ratio = cur_residual.norm(dim=-1) / (
                    sae_in.norm(dim=-1) + 1e-8
                )
                if (residual_ratio < threshold).all():
                    break

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        sae_out_pre = act_times_W_dec(
            feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
        )
        if self.cfg.apply_b_dec_to_input:
            sae_out_pre = sae_out_pre + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @override
    def get_coefficients(self) -> dict[str, Any]:
        return {}

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {}

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)
        if step_input.is_logging_step:
            l0 = output.feature_acts.bool().float().sum(-1)
            residual_norm = (step_input.sae_in - output.sae_out).norm(dim=-1)
            output.metrics["max_l0"] = l0.max()
            output.metrics["min_l0"] = l0.min()
            output.metrics["mean_l0"] = l0.mean()
            output.metrics["residual_norm"] = residual_norm.mean()

            # Per-phase L0 diagnostics
            m = self.cfg.d_sae_per_phase
            for p in range(self.cfg.num_phases):
                phase_l0 = (
                    output.feature_acts[..., p * m : (p + 1) * m]
                    .bool()
                    .float()
                    .sum(-1)
                    .mean()
                )
                output.metrics[f"phase_{p}_l0"] = phase_l0

        return output

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        if not self.cfg.rescale_acts_by_decoder_norm:
            raise NotImplementedError(
                "Folding W_dec_norm is not safe for PhaseMultiplexedTrainingSAE when rescale_acts_by_decoder_norm is False"
            )
        super().fold_W_dec_norm()


# Register with SAELens global registries
register_sae_class(
    PhaseMultiplexedSAEConfig.architecture(),
    PhaseMultiplexedSAE,
    PhaseMultiplexedSAEConfig,
)
register_sae_training_class(
    PhaseMultiplexedTrainingSAEConfig.architecture(),
    PhaseMultiplexedTrainingSAE,
    PhaseMultiplexedTrainingSAEConfig,
)
