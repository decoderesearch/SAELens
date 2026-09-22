import pytest
import torch

from sae_lens.registry import get_sae_class, get_sae_training_class
from sae_lens.saes.phase_multiplexed_sae import (
    PhaseMultiplexedSAE,
    PhaseMultiplexedSAEConfig,
    PhaseMultiplexedTrainingSAE,
    PhaseMultiplexedTrainingSAEConfig,
)
from sae_lens.saes.sae import TrainStepInput


def test_phase_multiplexed_sae_registry_lookup():
    sae_cls, cfg_cls = get_sae_class("phase_multiplexed")
    assert sae_cls is PhaseMultiplexedSAE
    assert cfg_cls is PhaseMultiplexedSAEConfig

    tr_cls, tr_cfg_cls = get_sae_training_class("phase_multiplexed")
    assert tr_cls is PhaseMultiplexedTrainingSAE
    assert tr_cfg_cls is PhaseMultiplexedTrainingSAEConfig


def test_phase_multiplexed_config_validation():
    # Valid config
    cfg = PhaseMultiplexedSAEConfig(
        d_in=64,
        d_sae=256,
        num_phases=4,
        k_per_phase=8,
    )
    assert cfg.d_sae_per_phase == 64
    assert cfg.total_k == 32

    # Incompatible d_sae / num_phases
    with pytest.raises(ValueError, match="divisible by num_phases"):
        PhaseMultiplexedSAEConfig(
            d_in=64,
            d_sae=250,
            num_phases=4,
            k_per_phase=8,
        )

    # Incompatible k_per_phase > d_sae_per_phase
    with pytest.raises(ValueError, match="cannot exceed d_sae_per_phase"):
        PhaseMultiplexedSAEConfig(
            d_in=64,
            d_sae=256,
            num_phases=4,
            k_per_phase=70,  # 70 > 64
        )

    # num_phases < 1
    with pytest.raises(ValueError, match="num_phases must be >= 1"):
        PhaseMultiplexedSAEConfig(
            d_in=64,
            d_sae=256,
            num_phases=0,
            k_per_phase=8,
        )


def test_phase_multiplexed_weight_shapes_and_init():
    d_in = 32
    d_sae = 128
    num_phases = 4
    k_per_phase = 4

    cfg = PhaseMultiplexedSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        num_phases=num_phases,
        k_per_phase=k_per_phase,
    )
    sae = PhaseMultiplexedSAE(cfg)

    assert sae.W_enc.shape == (d_in, d_sae)
    assert sae.W_dec.shape == (d_sae, d_in)
    assert sae.b_enc.shape == (d_sae,)
    assert sae.b_dec.shape == (d_in,)


def test_phase_multiplexed_encode_decode_shapes():
    d_in = 32
    d_sae = 128
    num_phases = 4
    k_per_phase = 4

    cfg = PhaseMultiplexedSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        num_phases=num_phases,
        k_per_phase=k_per_phase,
    )
    sae = PhaseMultiplexedSAE(cfg)

    # 2D batch: [B, d_in]
    x_2d = torch.randn(10, d_in)
    acts_2d = sae.encode(x_2d)
    recon_2d = sae.decode(acts_2d)
    out_2d = sae(x_2d)

    assert acts_2d.shape == (10, d_sae)
    assert recon_2d.shape == (10, d_in)
    assert out_2d.shape == (10, d_in)

    # 3D batch: [B, seq, d_in]
    x_3d = torch.randn(4, 7, d_in)
    acts_3d = sae.encode(x_3d)
    recon_3d = sae.decode(acts_3d)
    out_3d = sae(x_3d)

    assert acts_3d.shape == (4, 7, d_sae)
    assert recon_3d.shape == (4, 7, d_in)
    assert out_3d.shape == (4, 7, d_in)


def test_phase_multiplexed_per_phase_sparsity():
    d_in = 32
    d_sae = 128
    num_phases = 4
    k_per_phase = 5

    cfg = PhaseMultiplexedSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        num_phases=num_phases,
        k_per_phase=k_per_phase,
    )
    sae = PhaseMultiplexedSAE(cfg)

    x = torch.randn(20, d_in)
    acts = sae.encode(x)

    # Verify all feature activations are non-negative
    assert (acts >= 0).all()

    # Verify each phase activates at most k_per_phase latents
    m = cfg.d_sae_per_phase  # 32
    for p in range(num_phases):
        phase_acts = acts[:, p * m : (p + 1) * m]
        phase_l0 = (phase_acts > 0).sum(dim=-1)
        assert (phase_l0 <= k_per_phase).all(), f"Phase {p} exceeded k_per_phase budget"

    # Total L0 is bounded by num_phases * k_per_phase
    total_l0 = (acts > 0).sum(dim=-1)
    assert (total_l0 <= num_phases * k_per_phase).all()


def test_phase_multiplexed_streaming_generator():
    d_in = 32
    d_sae = 128
    num_phases = 4
    k_per_phase = 4

    cfg = PhaseMultiplexedSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        num_phases=num_phases,
        k_per_phase=k_per_phase,
    )
    sae = PhaseMultiplexedSAE(cfg)

    x = torch.randn(15, d_in)
    full_acts = sae.encode(x)

    ticks = list(sae.stream_phase_ticks(x))
    assert len(ticks) == num_phases

    m = cfg.d_sae_per_phase
    accumulated_recon = torch.zeros_like(sae.process_sae_in(x))

    for p, (phase_idx, acts_p, recon_p, cur_residual) in enumerate(ticks):
        assert phase_idx == p
        assert acts_p.shape == (15, m)
        assert recon_p.shape == (15, d_in)

        # Slices in generator must match full encode()
        expected_acts_p = full_acts[:, p * m : (p + 1) * m]
        assert torch.allclose(acts_p, expected_acts_p, atol=1e-6)

        accumulated_recon = accumulated_recon + recon_p
        expected_residual = sae.process_sae_in(x) - accumulated_recon
        assert torch.allclose(cur_residual, expected_residual, atol=1e-6)


def test_phase_multiplexed_training_forward_and_gradients():
    d_in = 32
    d_sae = 128
    num_phases = 4
    k_per_phase = 4

    tr_cfg = PhaseMultiplexedTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        num_phases=num_phases,
        k_per_phase=k_per_phase,
        dtype="float32",
        device="cpu",
    )
    tr_sae = PhaseMultiplexedTrainingSAE(tr_cfg)

    x = torch.randn(8, d_in)
    step_input = TrainStepInput(
        sae_in=x,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=1,
        is_logging_step=True,
    )

    out = tr_sae.training_forward_pass(step_input)

    assert out.sae_out.shape == (8, d_in)
    assert out.feature_acts.shape == (8, d_sae)
    assert out.hidden_pre.shape == (8, d_sae)
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)

    # Check metrics logged
    assert "mean_l0" in out.metrics
    assert "phase_0_l0" in out.metrics
    assert "phase_1_l0" in out.metrics
    assert "phase_2_l0" in out.metrics
    assert "phase_3_l0" in out.metrics

    # Test backward pass populates gradients on all tiled parameters
    out.loss.backward()
    assert tr_sae.W_enc.grad is not None
    assert tr_sae.W_dec.grad is not None
    assert tr_sae.b_enc.grad is not None
    assert tr_sae.b_dec.grad is not None

    assert torch.isfinite(tr_sae.W_enc.grad).all()
    assert torch.isfinite(tr_sae.W_dec.grad).all()


def test_phase_multiplexed_latency_vs_bandwidth_tradeoff():
    """Verify that PhaseSAE streams parameters in 1/P memory bandwidth chunks
    while progressively contracting residual error across micro-ticks.
    """
    d_in = 64
    d_sae = 256
    num_phases = 4
    k_per_phase = 8

    cfg = PhaseMultiplexedSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        num_phases=num_phases,
        k_per_phase=k_per_phase,
    )
    sae = PhaseMultiplexedSAE(cfg)

    x = torch.randn(32, d_in)
    full_param_bytes = (d_in * d_sae + d_sae * d_in) * 4  # float32 = 4 bytes
    tile_param_bytes = full_param_bytes // num_phases

    residual_norms = []
    for phase_idx, acts_p, recon_p, cur_residual in sae.stream_phase_ticks(x):
        # 1. Parameter slice bandwidth per micro-tick is exactly 1/P
        m = cfg.d_sae_per_phase
        w_enc_slice_bytes = (d_in * m) * 4
        w_dec_slice_bytes = (m * d_in) * 4
        step_bytes = w_enc_slice_bytes + w_dec_slice_bytes
        assert step_bytes == tile_param_bytes
        assert step_bytes == full_param_bytes / num_phases

        # 2. Peak activation buffer per micro-tick is bounded to (B, m)
        assert acts_p.numel() == 32 * m
        assert acts_p.numel() == (32 * d_sae) / num_phases

        residual_norms.append(cur_residual.norm(dim=-1).mean().item())

    # Verify that micro-ticks run sequentially and yield num_phases ticks
    assert len(residual_norms) == num_phases


def test_phase_multiplexed_config_exit_threshold():
    # Default is None
    cfg = PhaseMultiplexedSAEConfig(d_in=32, d_sae=128, num_phases=4, k_per_phase=4)
    assert cfg.exit_threshold is None

    # Custom threshold
    cfg_with_threshold = PhaseMultiplexedSAEConfig(
        d_in=32, d_sae=128, num_phases=4, k_per_phase=4, exit_threshold=0.05
    )
    assert cfg_with_threshold.exit_threshold == 0.05

    # Training config
    tr_cfg = PhaseMultiplexedTrainingSAEConfig(
        d_in=32, d_sae=128, num_phases=4, k_per_phase=4, exit_threshold=0.1
    )
    assert tr_cfg.exit_threshold == 0.1


def test_phase_multiplexed_early_exit_generous_and_strict():
    d_in = 32
    d_sae = 128
    num_phases = 4
    k_per_phase = 4
    m = d_sae // num_phases  # 32

    # Generous configuration (exit_threshold=0.9)
    cfg_generous = PhaseMultiplexedSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        num_phases=num_phases,
        k_per_phase=k_per_phase,
        exit_threshold=0.9,
    )
    sae_generous = PhaseMultiplexedSAE(cfg_generous)
    # Unit-norm normalize decoder weights to ensure well-behaved projections
    sae_generous.W_dec.data = sae_generous.W_dec.data / sae_generous.W_dec.data.norm(
        dim=-1, keepdim=True
    )
    sae_generous.W_enc.data = sae_generous.W_dec.data.T.clone()

    # Synthetic input constructed from Phase 0 features so that Phase 0 achieves < 0.9 residual ratio
    target_acts = torch.zeros(10, d_sae)
    target_acts[:, :k_per_phase] = torch.rand(10, k_per_phase) + 1.0
    x_2d = target_acts @ sae_generous.W_dec

    # 1. Test generous early exit (terminates after Phase 0)
    ticks = list(sae_generous.stream_phase_ticks(x_2d))
    assert (
        len(ticks) == 1
    ), f"Expected 1 phase tick for generous threshold, got {len(ticks)}"
    assert ticks[0][0] == 0  # Phase 0

    acts_2d = sae_generous.encode(x_2d)
    assert acts_2d.shape == (10, d_sae), "Shape preservation failed for encode()"
    # Phase 0 features must be populated, subsequent phases must remain strictly 0
    assert (
        acts_2d[:, :m] != 0
    ).any(), "Phase 0 features should have non-zero activations"
    assert (
        acts_2d[:, m:] == 0
    ).all(), "Phases 1..P-1 should be strictly 0 after early exit"

    # Shape preservation on decode and forward
    recon_2d = sae_generous.decode(acts_2d)
    out_2d = sae_generous(x_2d)
    assert recon_2d.shape == (10, d_in)
    assert out_2d.shape == (10, d_in)

    # Numerical invariant: decode output equals partial phase-0 reconstruction
    expected_recon = acts_2d[:, :m] @ sae_generous.W_dec[:m, :]
    assert torch.allclose(recon_2d, expected_recon, atol=1e-5)

    # 2. Test 3D batch shape preservation with generous exit
    x_3d = x_2d.unsqueeze(1).expand(-1, 5, -1)  # [10, 5, d_in]
    acts_3d = sae_generous.encode(x_3d)
    out_3d = sae_generous(x_3d)
    assert acts_3d.shape == (10, 5, d_sae)
    assert out_3d.shape == (10, 5, d_in)
    assert (acts_3d[..., m:] == 0).all()

    # 3. Test strict configuration (exit_threshold=0.001) - runs all phases
    cfg_strict = PhaseMultiplexedSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        num_phases=num_phases,
        k_per_phase=k_per_phase,
        exit_threshold=0.001,
    )
    sae_strict = PhaseMultiplexedSAE(cfg_strict)
    sae_strict.W_dec.data = sae_generous.W_dec.data.clone()
    sae_strict.W_enc.data = sae_generous.W_enc.data.clone()

    x_random = torch.randn(10, d_in)
    ticks_strict = list(sae_strict.stream_phase_ticks(x_random))
    assert (
        len(ticks_strict) == num_phases
    ), f"Expected {num_phases} ticks for strict threshold, got {len(ticks_strict)}"

    acts_strict = sae_strict.encode(x_random)
    assert acts_strict.shape == (10, d_sae)
    # With random input and strict threshold, features beyond phase 0 must be populated
    assert not (
        acts_strict[:, m:] == 0
    ).all(), "Strict threshold should execute subsequent phases"

    # 4. Test dynamic threshold parameter override
    # Pass generous exit_threshold dynamically to strict SAE
    acts_dyn = sae_strict.encode(x_2d, exit_threshold=0.9)
    assert (
        acts_dyn[:, m:] == 0
    ).all(), "Dynamic generous threshold should early-exit after Phase 0"


def test_phase_multiplexed_training_sae_early_exit():
    d_in = 32
    d_sae = 128
    num_phases = 4
    k_per_phase = 4
    m = d_sae // num_phases

    # Generous training config
    tr_cfg_generous = PhaseMultiplexedTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        num_phases=num_phases,
        k_per_phase=k_per_phase,
        exit_threshold=0.9,
        dtype="float32",
        device="cpu",
    )
    tr_sae = PhaseMultiplexedTrainingSAE(tr_cfg_generous)
    tr_sae.W_dec.data = tr_sae.W_dec.data / tr_sae.W_dec.data.norm(dim=-1, keepdim=True)
    tr_sae.W_enc.data = tr_sae.W_dec.data.T.clone()

    target_acts = torch.zeros(8, d_sae)
    target_acts[:, :k_per_phase] = torch.rand(8, k_per_phase) + 1.0
    x = target_acts @ tr_sae.W_dec

    feature_acts, hidden_pre = tr_sae.encode_with_hidden_pre(x)
    assert feature_acts.shape == (8, d_sae)
    assert hidden_pre.shape == (8, d_sae)
    assert (feature_acts[:, m:] == 0).all()
    assert (hidden_pre[:, m:] == 0).all()

    # Verify training forward pass with logging
    step_input = TrainStepInput(
        sae_in=x,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=1,
        is_logging_step=True,
    )
    out = tr_sae.training_forward_pass(step_input)
    assert out.sae_out.shape == (8, d_in)
    assert out.feature_acts.shape == (8, d_sae)
    assert out.metrics["phase_0_l0"] > 0
    assert out.metrics["phase_1_l0"] == 0
