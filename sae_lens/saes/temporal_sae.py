"""TemporalSAE: A Sparse Autoencoder with temporal attention mechanism.

TemporalSAE decomposes activations into:
1. Predicted codes (from attention over context)
2. Novel codes (sparse features of the residual)

See: https://arxiv.org/abs/2410.04185
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

from sae_lens import logger
from sae_lens.saes.sae import SAE, SAEConfig


def get_attention(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Compute causal attention weights."""
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_bias.to(query.dtype)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    return torch.softmax(attn_weight, dim=-1)


class ManualAttention(nn.Module):
    """Manual attention implementation for TemporalSAE."""

    def __init__(
        self,
        dimin: int,
        n_heads: int = 4,
        bottleneck_factor: int = 64,
        bias_k: bool = True,
        bias_q: bool = True,
        bias_v: bool = True,
        bias_o: bool = True,
    ):
        super().__init__()
        assert dimin % (bottleneck_factor * n_heads) == 0

        self.n_heads = n_heads
        self.n_embds = dimin // bottleneck_factor
        self.dimin = dimin

        # Key, query, value projections
        self.k_ctx = nn.Linear(dimin, self.n_embds, bias=bias_k)
        self.q_target = nn.Linear(dimin, self.n_embds, bias=bias_q)
        self.v_ctx = nn.Linear(dimin, dimin, bias=bias_v)
        self.c_proj = nn.Linear(dimin, dimin, bias=bias_o)

        # Normalize to match scale with representations
        with torch.no_grad():
            scaling = 1 / math.sqrt(self.n_embds // self.n_heads)
            self.k_ctx.weight.copy_(
                scaling
                * self.k_ctx.weight
                / (1e-6 + torch.linalg.norm(self.k_ctx.weight, dim=1, keepdim=True))
            )
            self.q_target.weight.copy_(
                scaling
                * self.q_target.weight
                / (1e-6 + torch.linalg.norm(self.q_target.weight, dim=1, keepdim=True))
            )

            scaling = 1 / math.sqrt(self.dimin // self.n_heads)
            self.v_ctx.weight.copy_(
                scaling
                * self.v_ctx.weight
                / (1e-6 + torch.linalg.norm(self.v_ctx.weight, dim=1, keepdim=True))
            )

            scaling = 1 / math.sqrt(self.dimin)
            self.c_proj.weight.copy_(
                scaling
                * self.c_proj.weight
                / (1e-6 + torch.linalg.norm(self.c_proj.weight, dim=1, keepdim=True))
            )

    def forward(
        self, x_ctx: torch.Tensor, x_target: torch.Tensor, get_attn_map: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute projective attention output."""
        k = self.k_ctx(x_ctx)
        v = self.v_ctx(x_ctx)
        q = self.q_target(x_target)

        # Split into heads
        B, T, _ = x_ctx.size()
        k = k.view(B, T, self.n_heads, self.n_embds // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.n_embds // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.dimin // self.n_heads).transpose(1, 2)

        # Attention map (optional)
        attn_map = None
        if get_attn_map:
            attn_map = get_attention(query=q, key=k)

        # Scaled dot-product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0, is_causal=True
        )

        # Reshape and project
        d_target = self.c_proj(
            attn_output.transpose(1, 2).contiguous().view(B, T, self.dimin)
        )

        return d_target, attn_map


@dataclass
class TemporalSAEConfig(SAEConfig):
    """Configuration for TemporalSAE inference.

    Args:
        d_in: Input dimension (dimensionality of the activations being encoded)
        d_sae: SAE latent dimension (number of features)
        n_heads: Number of attention heads in temporal attention
        n_attn_layers: Number of attention layers
        bottleneck_factor: Bottleneck factor for attention dimension
        sae_diff_type: Type of SAE for novel codes ('relu' or 'topk')
        kval_topk: K value for top-k sparsity (if sae_diff_type='topk')
        tied_weights: Whether to tie encoder and decoder weights
        activation_normalization_factor: Scalar factor for rescaling activations (used with normalize_activations='constant_scalar_rescale')
    """

    n_heads: int = 8
    n_attn_layers: int = 1
    bottleneck_factor: int = 64
    sae_diff_type: Literal["relu", "topk"] = "topk"
    kval_topk: int | None = None
    tied_weights: bool = True
    activation_normalization_factor: float = 1.0

    def __post_init__(self):
        # Call parent's __post_init__ first, but allow constant_scalar_rescale
        if self.normalize_activations not in [
            "none",
            "expected_average_only_in",
            "constant_norm_rescale",
            "constant_scalar_rescale",  # Temporal SAEs support this
            "layer_norm",
        ]:
            raise ValueError(
                f"normalize_activations must be none, expected_average_only_in, layer_norm, constant_norm_rescale, or constant_scalar_rescale. Got {self.normalize_activations}"
            )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "temporal"


class TemporalSAE(SAE[TemporalSAEConfig]):
    """TemporalSAE: Sparse Autoencoder with temporal attention.

    This SAE decomposes each activation x_t into:
    - x_pred: Information aggregated from context {x_0, ..., x_{t-1}}
    - x_novel: Novel information at position t (encoded sparsely)

    The forward pass:
    1. Uses attention layers to predict x_t from context
    2. Encodes the residual (novel part) with a sparse SAE
    3. Combines both for reconstruction
    """

    # Custom parameters (in addition to W_enc, W_dec, b_dec from base)
    attn_layers: nn.ModuleList  # Attention layers
    eps: float
    lam: float

    def __init__(self, cfg: TemporalSAEConfig, use_error_term: bool = False):
        # Call parent init first
        super().__init__(cfg, use_error_term)

        # Initialize attention layers after parent init and move to correct device
        self.attn_layers = nn.ModuleList(
            [
                ManualAttention(
                    dimin=cfg.d_sae,
                    n_heads=cfg.n_heads,
                    bottleneck_factor=cfg.bottleneck_factor,
                    bias_k=True,
                    bias_q=True,
                    bias_v=True,
                    bias_o=True,
                ).to(device=self.device, dtype=self.dtype)
                for _ in range(cfg.n_attn_layers)
            ]
        )

        self.eps = 1e-6
        self.lam = 1 / (4 * self.cfg.d_in)

    @override
    def _setup_activation_normalization(self):
        """Set up activation normalization functions for TemporalSAE.

        Overrides the base implementation to handle constant_scalar_rescale
        using the temporal-specific activation_normalization_factor.
        """
        if self.cfg.normalize_activations == "constant_scalar_rescale":
            # Handle constant scalar rescaling for temporal SAEs
            def run_time_activation_norm_fn_in(x: torch.Tensor) -> torch.Tensor:
                return x * self.cfg.activation_normalization_factor

            def run_time_activation_norm_fn_out(x: torch.Tensor) -> torch.Tensor:
                return x / self.cfg.activation_normalization_factor

            self.run_time_activation_norm_fn_in = run_time_activation_norm_fn_in
            self.run_time_activation_norm_fn_out = run_time_activation_norm_fn_out
        else:
            # Delegate to parent for all other normalization types
            super()._setup_activation_normalization()

    @override
    def initialize_weights(self) -> None:
        """Initialize TemporalSAE weights."""
        # Initialize D (decoder) and b (bias)
        self.W_dec = nn.Parameter(
            torch.randn(
                (self.cfg.d_sae, self.cfg.d_in), dtype=self.dtype, device=self.device
            )
        )
        self.b_dec = nn.Parameter(
            torch.zeros((self.cfg.d_in), dtype=self.dtype, device=self.device)
        )

        # Initialize E (encoder) if not tied
        if not self.cfg.tied_weights:
            self.W_enc = nn.Parameter(
                torch.randn(
                    (self.cfg.d_in, self.cfg.d_sae),
                    dtype=self.dtype,
                    device=self.device,
                )
            )

    def encode_with_predictions(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to novel codes only.

        Returns only the sparse novel codes (not predicted codes).
        This is the main feature representation for TemporalSAE.
        """
        # Process input through SAELens preprocessing
        x = self.process_sae_in(x)

        B, L, _ = x.shape

        if self.cfg.tied_weights:  # noqa: SIM108
            W_enc = self.W_dec.T
        else:
            W_enc = self.W_enc

        # Compute predicted codes using attention
        x_residual = x
        z_pred = torch.zeros((B, L, self.cfg.d_sae), device=x.device, dtype=x.dtype)

        for attn_layer in self.attn_layers:
            # Encode input to latent space
            z_input = F.relu(torch.matmul(x_residual * self.lam, W_enc))

            # Shift context (causal masking)
            z_ctx = torch.cat(
                (torch.zeros_like(z_input[:, :1, :]), z_input[:, :-1, :].clone()), dim=1
            )

            # Apply attention to get predicted codes
            z_pred_, _ = attn_layer(z_ctx, z_input, get_attn_map=False)
            z_pred_ = F.relu(z_pred_)

            # Project predicted codes back to input space
            Dz_pred_ = torch.matmul(z_pred_, self.W_dec)
            Dz_norm_ = Dz_pred_.norm(dim=-1, keepdim=True) + self.eps

            # Compute projection scale
            proj_scale = (Dz_pred_ * x_residual).sum(
                dim=-1, keepdim=True
            ) / Dz_norm_.pow(2)

            # Accumulate predicted codes
            z_pred = z_pred + (z_pred_ * proj_scale)

            # Remove prediction from residual
            x_residual = x_residual - proj_scale * Dz_pred_

        # Encode residual (novel part) with sparse SAE
        z_novel = F.relu(torch.matmul(x_residual * self.lam, W_enc))
        if self.cfg.sae_diff_type == "topk":
            kval = self.cfg.kval_topk
            if kval is not None:
                _, topk_indices = torch.topk(z_novel, kval, dim=-1)
                mask = torch.zeros_like(z_novel)
                mask.scatter_(-1, topk_indices, 1)
                z_novel = z_novel * mask

        # Return only novel codes (these are the interpretable features)
        return z_novel, z_pred

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_with_predictions(x)[0]

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode novel codes to reconstruction.

        Note: This only decodes the novel codes. For full reconstruction,
        use forward() which includes predicted codes.
        """
        # Decode novel codes
        sae_out = torch.matmul(feature_acts, self.W_dec)
        sae_out = sae_out + self.b_dec

        # Apply hook
        sae_out = self.hook_sae_recons(sae_out)

        # Apply output activation normalization (reverses input normalization)
        sae_out = self.run_time_activation_norm_fn_out(sae_out)

        # Add bias (already removed in process_sae_in)
        logger.warning(
            "NOTE this only decodes x_novel. The x_pred is missing, so we're not reconstructing the full x."
        )
        return sae_out

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass through TemporalSAE.

        Returns complete reconstruction (predicted + novel).
        """
        # Encode
        z_novel, z_pred = self.encode_with_predictions(x)

        # Decode the sum of predicted and novel codes.
        x_recons = torch.matmul(z_novel + z_pred, self.W_dec) + self.b_dec

        # Apply output activation normalization (reverses input normalization)
        x_recons = self.run_time_activation_norm_fn_out(x_recons)

        return self.hook_sae_output(x_recons)

    @override
    def fold_W_dec_norm(self) -> None:
        raise NotImplementedError("Folding W_dec_norm is not supported for TemporalSAE")

    @override
    @torch.no_grad()
    def fold_activation_norm_scaling_factor(self, scaling_factor: float) -> None:
        raise NotImplementedError(
            "Folding activation norm scaling factor is not supported for TemporalSAE"
        )
