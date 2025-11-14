import os

import pytest
import torch

from sae_lens.constants import DTYPE_MAP
from sae_lens.saes.sae import SAE
from sae_lens.saes.temporal_sae import ManualAttention, TemporalSAE
from tests.helpers import build_temporal_sae_cfg


def test_TemporalSAE_initialization():
    cfg = build_temporal_sae_cfg()
    sae = TemporalSAE.from_dict(cfg.to_dict())

    assert isinstance(sae.W_dec, torch.nn.Parameter)
    assert isinstance(sae.b_dec, torch.nn.Parameter)
    assert sae.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert sae.b_dec.shape == (cfg.d_in,)

    if not sae.cfg.tied_weights:
        assert isinstance(sae.W_enc, torch.nn.Parameter)
        assert sae.W_enc.shape == (cfg.d_in, cfg.d_sae)

    assert len(sae.attn_layers) == cfg.n_attn_layers
    for attn_layer in sae.attn_layers:
        assert attn_layer.n_heads == cfg.n_heads
        assert attn_layer.dimin == cfg.d_sae


@pytest.mark.parametrize("tied_weights", [True, False])
def test_TemporalSAE_forward(tied_weights: bool):
    cfg = build_temporal_sae_cfg(tied_weights=tied_weights)
    sae = TemporalSAE.from_dict(cfg.to_dict())

    batch_size = 4
    seq_len = 16
    x = torch.randn(batch_size, seq_len, cfg.d_in)

    reconstruction = sae.forward(x)

    assert reconstruction.shape == x.shape


@pytest.mark.parametrize("tied_weights", [True, False])
def test_TemporalSAE_encode(tied_weights: bool):
    cfg = build_temporal_sae_cfg(
        tied_weights=tied_weights, sae_diff_type="topk", kval_topk=32
    )
    sae = TemporalSAE.from_dict(cfg.to_dict())

    batch_size = 4
    seq_len = 16
    x = torch.randn(batch_size, seq_len, cfg.d_in)

    novel_codes = sae.encode(x)

    assert novel_codes.shape == (batch_size, seq_len, cfg.d_sae)
    assert (novel_codes >= 0).all()
    assert cfg.kval_topk is not None
    l0 = (novel_codes != 0).sum(dim=-1).float().mean().item()
    assert l0 <= cfg.kval_topk


def test_TemporalSAE_decode():
    cfg = build_temporal_sae_cfg()
    sae = TemporalSAE.from_dict(cfg.to_dict())

    batch_size = 4
    seq_len = 16
    novel_codes = torch.randn(
        batch_size, seq_len, cfg.d_sae, dtype=DTYPE_MAP[sae.cfg.dtype]
    ).relu()

    reconstruction = sae.decode(novel_codes)

    assert reconstruction.shape == (batch_size, seq_len, cfg.d_in)


@pytest.mark.parametrize("sae_diff_type", ["relu", "topk"])
def test_TemporalSAE_sae_diff_type(sae_diff_type: str):
    cfg = build_temporal_sae_cfg(
        sae_diff_type=sae_diff_type,
        kval_topk=32 if sae_diff_type == "topk" else None,
    )
    sae = TemporalSAE.from_dict(cfg.to_dict())

    batch_size = 4
    seq_len = 16
    x = torch.randn(batch_size, seq_len, cfg.d_in)

    novel_codes = sae.encode(x)

    assert novel_codes.shape == (batch_size, seq_len, cfg.d_sae)
    assert (novel_codes >= 0).all()

    if sae_diff_type == "topk":
        assert cfg.kval_topk is not None
        l0 = (novel_codes != 0).sum(dim=-1).float()
        assert (l0 <= cfg.kval_topk).all()


@pytest.mark.parametrize("n_attn_layers", [1, 2])
def test_TemporalSAE_n_attn_layers(n_attn_layers: int):
    cfg = build_temporal_sae_cfg(n_attn_layers=n_attn_layers)
    sae = TemporalSAE.from_dict(cfg.to_dict())

    assert len(sae.attn_layers) == n_attn_layers

    batch_size = 4
    seq_len = 16
    x = torch.randn(batch_size, seq_len, cfg.d_in)

    reconstruction = sae.forward(x)
    assert reconstruction.shape == x.shape


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Uses too much disk space on CI",
)
def test_TemporalSAE_load_from_pretrained_and_forward():
    """Test loading a TemporalSAE from HuggingFace and performing a forward pass."""
    # Load the TemporalSAE from HuggingFace
    sae = SAE.from_pretrained(
        release="temporal-sae-gemma-2-2b",
        sae_id="blocks.12.hook_resid_post",
        device="cpu",
    )

    # Create random input data with the correct shape
    # TemporalSAE expects input of shape (batch, sequence, d_in)
    batch_size = 2
    seq_len = 8
    d_in = 2304

    x = torch.randn(batch_size, seq_len, d_in)

    # Perform forward pass
    x_recons = sae(x)

    # Check output shape matches input shape
    assert x_recons.shape == x.shape

    # Test encode method
    z_novel = sae.encode(x)
    assert z_novel.shape == (batch_size, seq_len, sae.cfg.d_sae)


def test_TemporalSAE_matches_original_implementation():
    """Test that SAELens TemporalSAE produces identical outputs to the original implementation."""
    from tests._comparison.temporal_sae_original import (
        TemporalSAE as OriginalTemporalSAE,
    )

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create configuration
    d_in = 64
    d_sae = 256
    n_heads = 8
    bottleneck_factor = 2
    kval_topk = 32
    tied_weights = True
    sae_diff_type = "topk"
    n_attn_layers = 1
    activation_scaling_factor = 0.3

    # Create SAELens implementation
    cfg = build_temporal_sae_cfg(
        d_in=d_in,
        d_sae=d_sae,
        n_heads=n_heads,
        n_attn_layers=n_attn_layers,
        bottleneck_factor=bottleneck_factor,
        sae_diff_type=sae_diff_type,
        kval_topk=kval_topk,
        tied_weights=tied_weights,
        activation_normalization_factor=activation_scaling_factor,
    )
    sae_sl = TemporalSAE.from_dict(cfg.to_dict())

    # Create original implementation with matching parameters
    sae_original = OriginalTemporalSAE(
        dimin=d_in,
        width=d_sae,
        n_heads=n_heads,
        sae_diff_type=sae_diff_type,
        kval_topk=kval_topk,
        tied_weights=tied_weights,
        n_attn_layers=n_attn_layers,
        bottleneck_factor=bottleneck_factor,
        activation_scaling_factor=activation_scaling_factor,
    )

    # Copy weights from SAELens to original implementation to ensure exact match
    # Note: Original uses D (decoder) and E (encoder), SAELens uses W_dec and W_enc
    # Copy decoder weights (D in original = W_dec in SAELens)
    sae_original.D.data = sae_sl.W_dec.data

    # Copy bias (b in original = b_dec in SAELens, but original has shape (1, d_in))
    sae_original.b.data = sae_sl.b_dec.unsqueeze(0).data

    # Copy attention layer weights
    for i in range(n_attn_layers):
        # Copy attention weights for each layer
        attn_orig: ManualAttention = sae_original.attn_layers[i]  # type: ignore[assignment]
        attn_sl: ManualAttention = sae_sl.attn_layers[i]  # type: ignore[assignment]

        attn_orig.k_ctx.weight.data = attn_sl.k_ctx.weight.data
        attn_orig.k_ctx.bias.data = attn_sl.k_ctx.bias.data
        attn_orig.q_target.weight.data = attn_sl.q_target.weight.data
        attn_orig.q_target.bias.data = attn_sl.q_target.bias.data
        attn_orig.v_ctx.weight.data = attn_sl.v_ctx.weight.data
        attn_orig.v_ctx.bias.data = attn_sl.v_ctx.bias.data
        attn_orig.c_proj.weight.data = attn_sl.c_proj.weight.data
        attn_orig.c_proj.bias.data = attn_sl.c_proj.bias.data

    # Set both to eval mode
    sae_sl.eval()
    sae_original.eval()

    # Create random input
    batch_size = 4
    seq_len = 16
    torch.manual_seed(123)
    x = torch.randn(batch_size, seq_len, d_in).to(DTYPE_MAP[sae_sl.cfg.dtype])

    # Run both implementations
    with torch.no_grad():
        # SAELens implementation
        x_recons_lens = sae_sl(x)
        z_novel_lens = sae_sl.encode(x)

        # Original implementation
        x_recons_orig, results_dict_orig = sae_original(x, return_graph=False)
        z_novel_orig = results_dict_orig["novel_codes"]

    # Compare outputs
    # Check reconstruction
    assert torch.allclose(
        x_recons_lens, x_recons_orig, rtol=1e-5, atol=1e-6
    ), f"Reconstructions differ: max diff = {(x_recons_lens - x_recons_orig).abs().max()}"

    # Check novel codes
    assert torch.allclose(
        z_novel_lens, z_novel_orig, rtol=1e-5, atol=1e-6
    ), f"Novel codes differ: max diff = {(z_novel_lens - z_novel_orig).abs().max()}"
