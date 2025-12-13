from transformer_lens.hook_points import torch

from sae_lens.saes.matching_pursuit_sae import (
    MatchingPursuitSAE,
    MatchingPursuitTrainingSAE,
)
from tests.helpers import (
    build_matching_pursuit_sae_cfg,
    build_matching_pursuit_sae_training_cfg,
)


def test_MatchingPursuitSAE_selects_correct_latents_with_orthognal_dictionary():
    sae = MatchingPursuitSAE(
        build_matching_pursuit_sae_cfg(d_in=10, d_sae=10, residual_threshold=1e-8)
    )
    batch_size = 32
    torch.nn.init.orthogonal(sae.W_dec)
    sae.b_dec.data = torch.randn_like(sae.b_dec)

    true_feats = (torch.randn(batch_size, 10) - 0.25).relu()
    sae_in = torch.einsum("fi,bf->bi", sae.W_dec, true_feats) + sae.b_dec
    feats = sae.encode(sae_in)
    assert torch.allclose(feats, true_feats, rtol=1e-4, atol=1e-6)
    assert torch.allclose(sae.decode(feats), sae_in, rtol=1e-4, atol=1e-6)


def test_MatchingPursuitTrainingSAE_selects_correct_latents_with_orthognal_dictionary():
    sae = MatchingPursuitTrainingSAE(
        build_matching_pursuit_sae_training_cfg(
            d_in=10, d_sae=10, residual_threshold=1e-8
        )
    )
    batch_size = 32
    torch.nn.init.orthogonal(sae.W_dec)
    sae.b_dec.data = torch.randn_like(sae.b_dec)

    true_feats = (torch.randn(batch_size, 10) - 0.25).relu()
    sae_in = torch.einsum("fi,bf->bi", sae.W_dec, true_feats) + sae.b_dec
    feats = sae.encode(sae_in)
    assert torch.allclose(feats, true_feats, rtol=1e-4, atol=1e-6)
    assert torch.allclose(sae.decode(feats), sae_in, rtol=1e-4, atol=1e-6)
