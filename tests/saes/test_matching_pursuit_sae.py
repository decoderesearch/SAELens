import pytest
from transformer_lens.hook_points import torch

from sae_lens.saes.matching_pursuit_sae import (
    MatchingPursuitSAE,
    MatchingPursuitTrainingSAE,
    MatchingPursuitTrainingSAEConfig,
    _encode_matching_pursuit,
)
from tests.helpers import (
    assert_close,
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


def test_MatchingPursuitTrainingSAE_handles_3d_inputs():
    sae = MatchingPursuitTrainingSAE(
        build_matching_pursuit_sae_training_cfg(
            d_in=10, d_sae=10, residual_threshold=1e-8
        )
    )
    sae.b_dec.data = torch.randn_like(sae.b_dec)

    sae_in_3d = torch.randn(32, 10, 10)
    sae_in_2d = sae_in_3d.view(320, 10)

    feats_3d = sae.encode(sae_in_3d)
    feats_2d = sae.encode(sae_in_2d)
    assert feats_3d.shape == (32, 10, 10)
    assert feats_2d.shape == (320, 10)
    assert_close(feats_3d.view(320, 10), feats_2d)


def test_MatchingPursuitTrainingSAEConfig_raises_warning_if_decoder_init_norm_is_not_1_0():
    with pytest.warns(UserWarning):
        cfg = MatchingPursuitTrainingSAEConfig(
            d_in=10, d_sae=10, residual_threshold=1e-8, decoder_init_norm=0.3
        )
        assert cfg.decoder_init_norm == 1.0


def _encode_matching_pursuit_reference_implementation(
    sae_in_centered: torch.Tensor,
    W_dec: torch.Tensor,
    residual_threshold: float,
) -> torch.Tensor:
    residual = sae_in_centered.clone()
    batch_size = sae_in_centered.shape[0]

    z = torch.zeros(batch_size, W_dec.shape[0], device=W_dec.device)
    prev_support = torch.zeros_like(z).bool()
    done = torch.zeros(batch_size, dtype=torch.bool, device=W_dec.device)

    while not done.all():
        WTr = torch.relu(residual @ W_dec.T)

        values, indices = torch.max(torch.relu(WTr), dim=1, keepdim=True)

        z_ = torch.zeros_like(z)
        z_.scatter_(1, indices, values)
        z = torch.where(done.unsqueeze(1), z, z + z_)

        update = torch.matmul(z_, W_dec)
        residual = torch.where(done.unsqueeze(1), residual, residual - update)

        support = z != 0

        # A sample is considered converged if:
        # (1) the support set hasn't changed from the previous iteration (stability), or
        # (2) the residual norm is below a given threshold (good enough reconstruction)
        converged = (support == prev_support).all(dim=1) | (
            residual.norm(dim=1) < residual_threshold
        )
        done = done | converged
        prev_support = support

    return z


def test_encode_matching_pursuit_matches_reference_implementation():
    sae_in_centered = torch.randn(10, 10)
    W_dec = torch.randn(10, 10, requires_grad=True)
    W_dec_ref = W_dec.clone().detach().requires_grad_(True)
    residual_threshold = 1e-2
    z_ref = _encode_matching_pursuit_reference_implementation(
        sae_in_centered, W_dec_ref, residual_threshold
    )
    z = _encode_matching_pursuit(sae_in_centered, W_dec, residual_threshold)

    (sae_in_centered - z_ref).norm(dim=1).mean().backward()
    (sae_in_centered - z).norm(dim=1).mean().backward()

    assert_close(z, z_ref)
    assert W_dec.grad is not None
    assert W_dec_ref.grad is not None
    # Use looser tolerances for gradients since the optimized implementation
    # computes values differently (via indexing + dot product instead of full matmul),
    # leading to small numerical differences in floating-point accumulation
    assert_close(W_dec.grad, W_dec_ref.grad, atol=1e-4, rtol=1e-4)
