import os
from pathlib import Path

import pytest
import torch
from torch import nn

from sae_lens.saes.gated_sae import GatedSAE, GatedTrainingSAE
from sae_lens.saes.sae import SAE, TrainStepInput
from tests.helpers import (
    assert_close,
    assert_not_close,
    build_gated_sae_cfg,
    build_gated_sae_training_cfg,
)


@pytest.mark.parametrize("use_error_term", [True, False])
def test_GatedSAE_forward(use_error_term: bool):
    sae = GatedSAE(build_gated_sae_cfg(d_in=2, d_sae=3))
    sae.use_error_term = use_error_term
    sae.W_enc.data = torch.ones_like(sae.W_enc.data)
    sae.W_dec.data = torch.ones_like(sae.W_dec.data)
    sae.b_dec.data = torch.zeros_like(sae.b_dec.data)
    sae.b_gate.data = torch.tensor([-2.0, 0.0, 1.0])
    sae.r_mag.data = torch.tensor([1.0, 2.0, 3.0])
    sae.b_mag.data = torch.tensor([1.0, 1.0, 1.0])

    sae_in = torch.tensor([[0.3, 0.3]])

    # expected gating pre acts: [0.6 - 2 = -1.4, 0.6, 0.6 + 1 = 1.6]
    # so the first gate should be off
    # mags should be [0.6 * exp(1), 0.6 * exp(2), 0.6 * exp(3)] + b_mag => [2.6310,  5.4334, 13.0513]

    expected_recons = torch.tensor([[18.4848, 18.4848]])
    # if we use error term, we should always get the same output as what we put in
    expected_output = sae_in if use_error_term else expected_recons
    out, cache = sae.run_with_cache(sae_in)

    assert_close(out, expected_output, atol=1e-3)
    assert_close(cache["hook_sae_input"], sae_in, atol=1e-3)
    assert_close(cache["hook_sae_output"], out, atol=1e-3)
    assert_close(cache["hook_sae_recons"], expected_recons, atol=1e-3)
    assert_close(
        cache["hook_sae_acts_pre"],
        torch.tensor([[2.6310, 5.4334, 13.0513]]),
        atol=1e-3,
    )
    # the threshold of 1.0 should block the first latent from firing
    assert_close(
        cache["hook_sae_acts_post"],
        torch.tensor([[0.0, 5.4334, 13.0513]]),
        atol=1e-3,
    )
    if use_error_term:
        assert_close(
            cache["hook_sae_error"],
            expected_output - expected_recons,
        )


def test_GatedTrainingSAE_encoding():
    cfg = build_gated_sae_training_cfg()
    sae = GatedTrainingSAE(cfg)

    batch_size = 32
    d_in = sae.cfg.d_in
    d_sae = sae.cfg.d_sae

    x = torch.randn(batch_size, d_in)
    feature_acts, hidden_pre = sae.encode_with_hidden_pre(x)

    assert feature_acts.shape == (batch_size, d_sae)
    assert hidden_pre.shape == (batch_size, d_sae)

    # Check the gating mechanism
    gating_pre_activation = x @ sae.W_enc + sae.b_gate
    active_features = (gating_pre_activation > 0).float()
    magnitude_pre_activation = x @ (sae.W_enc * sae.r_mag.exp()) + sae.b_mag
    feature_magnitudes = sae.activation_fn(magnitude_pre_activation)

    expected_feature_acts = active_features * feature_magnitudes
    assert_close(feature_acts, expected_feature_acts, atol=1e-6)


def test_GatedTrainingSAE_loss():
    cfg = build_gated_sae_training_cfg(
        decoder_init_norm=1.0,  # TODO: why is this needed??
    )
    sae = GatedTrainingSAE(cfg)

    batch_size = 32
    d_in = sae.cfg.d_in
    x = torch.randn(batch_size, d_in)

    train_step_output = sae.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=x,
            coefficients={"l1": sae.cfg.l1_coefficient},
            dead_neuron_mask=None,
            n_training_steps=0,
            is_logging_step=False,
        ),
    )

    assert train_step_output.sae_out.shape == (batch_size, d_in)
    assert train_step_output.feature_acts.shape == (batch_size, sae.cfg.d_sae)

    sae_in_centered = x - sae.b_dec
    via_gate_feature_magnitudes = torch.relu(sae_in_centered @ sae.W_enc + sae.b_gate)
    preactivation_l1_loss = (
        sae.cfg.l1_coefficient * torch.sum(via_gate_feature_magnitudes, dim=-1).mean()
    )

    via_gate_reconstruction = (
        via_gate_feature_magnitudes @ sae.W_dec.detach() + sae.b_dec.detach()
    )
    aux_reconstruction_loss = torch.sum(
        (via_gate_reconstruction - x) ** 2, dim=-1
    ).mean()

    expected_loss = (
        train_step_output.losses["mse_loss"]
        + preactivation_l1_loss
        + aux_reconstruction_loss
    )
    assert (
        pytest.approx(train_step_output.loss.item(), rel=1e-3) == expected_loss.item()
    )


def test_GatedSAE_save_and_load_from_pretrained(tmp_path: Path) -> None:
    cfg = build_gated_sae_cfg()
    model_path = str(tmp_path)
    sae = GatedSAE(cfg)
    sae_state_dict = sae.state_dict()
    sae.save_model(model_path)

    assert os.path.exists(model_path)

    sae_loaded = SAE.load_from_pretrained(model_path, device="cpu")

    assert isinstance(sae_loaded, GatedSAE)
    sae_loaded_state_dict = sae_loaded.state_dict()

    # check state_dict matches the original
    for key in sae.state_dict():
        assert_close(
            sae_state_dict[key],
            sae_loaded_state_dict[key],
        )

    sae_in = torch.randn(10, cfg.d_in, device=cfg.device)
    sae_out_1 = sae(sae_in)
    sae_out_2 = sae_loaded(sae_in)
    assert_close(sae_out_1, sae_out_2)


def test_GatedTrainingSAE_forward_pass():
    cfg = build_gated_sae_training_cfg()
    sae = GatedTrainingSAE(cfg)

    batch_size = 32
    d_in = sae.cfg.d_in

    x = torch.randn(batch_size, d_in)
    train_step_output = sae.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=x,
            coefficients={"l1": sae.cfg.l1_coefficient},
            dead_neuron_mask=None,
            n_training_steps=0,
            is_logging_step=False,
        ),
    )

    assert train_step_output.sae_out.shape == (batch_size, d_in)
    assert train_step_output.feature_acts.shape == (batch_size, sae.cfg.d_sae)

    # Detach the loss tensor and convert to numpy for comparison
    detached_loss = train_step_output.loss.detach().cpu().numpy()
    expected_loss = (
        (
            train_step_output.losses["mse_loss"]
            + train_step_output.losses["l1_loss"]
            + train_step_output.losses["auxiliary_reconstruction_loss"]
        )
        .detach()  # type: ignore
        .cpu()
        .numpy()
    )

    assert pytest.approx(detached_loss, rel=1e-3) == expected_loss


def test_GatedSAE_initialization():
    cfg = build_gated_sae_cfg()
    sae = GatedSAE.from_dict(cfg.to_dict())
    assert isinstance(sae.W_enc, nn.Parameter)
    assert isinstance(sae.W_dec, nn.Parameter)
    assert isinstance(sae.b_dec, nn.Parameter)
    assert isinstance(sae.b_gate, nn.Parameter)
    assert isinstance(sae.r_mag, nn.Parameter)
    assert isinstance(sae.b_mag, nn.Parameter)

    assert sae.W_enc.shape == (cfg.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert sae.b_dec.shape == (cfg.d_in,)
    assert sae.b_gate.shape == (cfg.d_sae,)
    assert sae.r_mag.shape == (cfg.d_sae,)
    assert sae.b_mag.shape == (cfg.d_sae,)

    assert_not_close(
        sae.W_enc,
        torch.zeros_like(sae.W_enc),
    )
    assert_not_close(
        sae.W_dec,
        torch.zeros_like(sae.W_dec),
    )
    assert_close(
        sae.b_dec,
        torch.zeros_like(sae.b_dec),
    )
    assert_close(
        sae.b_gate,
        torch.zeros_like(sae.b_gate),
    )
    assert_close(
        sae.r_mag,
        torch.zeros_like(sae.r_mag),
    )
    assert_close(
        sae.b_mag,
        torch.zeros_like(sae.b_mag),
    )


def test_GatedTrainingSAE_initialization():
    cfg = build_gated_sae_training_cfg()
    sae = GatedTrainingSAE.from_dict(cfg.to_dict())

    assert sae.W_enc.shape == (cfg.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert sae.b_mag.shape == (cfg.d_sae,)
    assert sae.b_gate.shape == (cfg.d_sae,)
    assert sae.r_mag.shape == (cfg.d_sae,)
    assert sae.b_dec.shape == (cfg.d_in,)
    assert isinstance(sae.activation_fn, torch.nn.ReLU)
    assert sae.device == torch.device("cpu")
    assert sae.dtype == torch.float32

    # biases
    assert_close(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert_close(sae.b_mag, torch.zeros_like(sae.b_mag), atol=1e-6)
    assert_close(sae.b_gate, torch.zeros_like(sae.b_gate), atol=1e-6)

    # check if the decoder weight norm is 0.1 by default
    assert_close(
        sae.W_dec.norm(dim=1),
        0.1 * torch.ones_like(sae.W_dec.norm(dim=1)),
        atol=1e-6,
    )

    #  Default currently should be tranpose initialization
    assert_close(sae.W_enc, sae.W_dec.T, atol=1e-6)


def test_GatedTrainingSAE_save_and_load_inference_sae(tmp_path: Path) -> None:
    # Create a training SAE with specific parameter values
    cfg = build_gated_sae_training_cfg(device="cpu")
    training_sae = GatedTrainingSAE(cfg)

    # Set some known values for testing
    training_sae.W_enc.data = torch.randn_like(training_sae.W_enc.data)
    training_sae.W_dec.data = torch.randn_like(training_sae.W_dec.data)
    training_sae.b_dec.data = torch.randn_like(training_sae.b_dec.data)
    training_sae.b_gate.data = torch.randn_like(training_sae.b_gate.data)
    training_sae.b_mag.data = torch.randn_like(training_sae.b_mag.data)
    training_sae.r_mag.data = torch.randn_like(training_sae.r_mag.data)

    # Save original state for comparison
    original_W_enc = training_sae.W_enc.data.clone()
    original_W_dec = training_sae.W_dec.data.clone()
    original_b_dec = training_sae.b_dec.data.clone()
    original_b_gate = training_sae.b_gate.data.clone()
    original_b_mag = training_sae.b_mag.data.clone()
    original_r_mag = training_sae.r_mag.data.clone()

    # Save as inference model
    model_path = str(tmp_path)
    training_sae.save_inference_model(model_path)

    assert os.path.exists(model_path)

    # Load as inference SAE
    inference_sae = SAE.load_from_disk(model_path, device="cpu")

    # Should be loaded as GatedSAE
    assert isinstance(inference_sae, GatedSAE)

    # Check that all parameters match
    assert_close(inference_sae.W_enc, original_W_enc)
    assert_close(inference_sae.W_dec, original_W_dec)
    assert_close(inference_sae.b_dec, original_b_dec)
    assert_close(inference_sae.b_gate, original_b_gate)
    assert_close(inference_sae.b_mag, original_b_mag)
    assert_close(inference_sae.r_mag, original_r_mag)

    # Verify forward pass gives same results
    sae_in = torch.randn(10, cfg.d_in, device="cpu")

    # Get output from training SAE
    training_feature_acts, _ = training_sae.encode_with_hidden_pre(sae_in)
    training_sae_out = training_sae.decode(training_feature_acts)

    # Get output from inference SAE
    inference_feature_acts = inference_sae.encode(sae_in)
    inference_sae_out = inference_sae.decode(inference_feature_acts)

    # Should produce identical outputs
    assert_close(training_feature_acts, inference_feature_acts)
    assert_close(training_sae_out, inference_sae_out)

    # Test the full forward pass
    training_full_out = training_sae(sae_in)
    inference_full_out = inference_sae(sae_in)
    assert_close(training_full_out, inference_full_out)
