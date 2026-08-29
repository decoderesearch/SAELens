import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from sae_lens.constants import SAE_WEIGHTS_FILENAME
from sae_lens.saes.jumprelu_sae import (
    JumpReLU,
    JumpReLUSAE,
    JumpReLUTrainingSAE,
    calculate_pre_act_loss,
)
from sae_lens.saes.sae import SAE, TrainingSAE, TrainStepInput
from tests.helpers import (
    assert_close,
    assert_not_close,
    build_jumprelu_sae_cfg,
    build_jumprelu_sae_training_cfg,
    run_training_forward_pass_with_cache,
)


def test_JumpReLUTrainingSAE_encoding():
    sae = JumpReLUTrainingSAE(build_jumprelu_sae_training_cfg())

    batch_size = 32
    d_in = sae.cfg.d_in
    d_sae = sae.cfg.d_sae

    x = torch.randn(batch_size, d_in)
    feature_acts, hidden_pre = sae.encode_with_hidden_pre(x)

    assert feature_acts.shape == (batch_size, d_sae)
    assert hidden_pre.shape == (batch_size, d_sae)

    # Check the JumpReLU thresholding
    sae_in = sae.process_sae_in(x)
    expected_hidden_pre = sae_in @ sae.W_enc + sae.b_enc
    expected_feature_acts = JumpReLU.apply(
        expected_hidden_pre, sae.threshold, sae.bandwidth, False
    )

    assert_close(feature_acts, expected_feature_acts, atol=1e-6)  # type: ignore


def _jumprelu_step_input(sae: JumpReLUTrainingSAE, batch_size: int) -> TrainStepInput:
    return TrainStepInput(
        sae_in=torch.randn(batch_size, sae.cfg.d_in),
        coefficients={"l0": sae.cfg.l0_coefficient},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )


def test_JumpReLUTrainingSAE_training_forward_pass_projects_negative_threshold_to_zero():
    sae = JumpReLUTrainingSAE(build_jumprelu_sae_training_cfg())
    sae.threshold.data = torch.full_like(sae.threshold.data, -0.5)

    output = sae.training_forward_pass(step_input=_jumprelu_step_input(sae, 512))

    # JumpReLU is x * (x > threshold), so a negative threshold would let
    # pre-activations in (-0.5, 0] through as negative feature activations.
    assert_close(sae.threshold.detach(), torch.zeros_like(sae.threshold))
    assert (output.feature_acts < 0).sum() == 0


def test_JumpReLUTrainingSAE_threshold_at_zero_still_receives_gradient():
    sae = JumpReLUTrainingSAE(build_jumprelu_sae_training_cfg())
    sae.threshold.data = torch.zeros_like(sae.threshold.data)

    output = sae.training_forward_pass(step_input=_jumprelu_step_input(sae, 512))
    output.loss.backward()

    # The projection happens on .data, outside the autograd graph. Clamping
    # inside the graph instead would zero this gradient, pinning any latent that
    # reaches zero there for the rest of training.
    assert sae.threshold.grad is not None
    assert sae.threshold.grad.abs().sum() > 0


def test_JumpReLUTrainingSAE_training_forward_pass():
    sae = JumpReLUTrainingSAE(build_jumprelu_sae_training_cfg())

    batch_size = 32
    d_in = sae.cfg.d_in

    x = torch.randn(batch_size, d_in)
    step_input = TrainStepInput(
        sae_in=x,
        coefficients={"l0": sae.cfg.l0_coefficient},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )
    train_step_output, train_cache = run_training_forward_pass_with_cache(
        sae, step_input
    )

    assert train_step_output.sae_out.shape == (batch_size, d_in)
    assert train_step_output.feature_acts.shape == (batch_size, sae.cfg.d_sae)
    assert (
        pytest.approx(train_step_output.loss.detach(), rel=1e-3)
        == (
            train_step_output.losses["mse_loss"] + train_step_output.losses["l0_loss"]
        ).item()  # type: ignore
    )

    expected_mse_loss = (
        (torch.pow((train_step_output.sae_out - x.float()), 2))
        .sum(dim=-1)
        .mean()
        .detach()
        .float()
    )

    assert (
        pytest.approx(train_step_output.losses["mse_loss"].item()) == expected_mse_loss  # type: ignore
    )

    assert train_cache["hook_sae_input"].equal(x)
    assert train_cache["hook_sae_acts_pre"].equal(train_step_output.hidden_pre)
    assert train_cache["hook_sae_acts_post"].equal(train_step_output.feature_acts)
    assert train_cache["hook_sae_recons"].equal(train_step_output.sae_out)

    # Verify training output matches a regular run_with_cache forward pass
    _, cache = sae.run_with_cache(x)
    assert train_cache["hook_sae_input"].equal(cache["hook_sae_input"])
    assert train_cache["hook_sae_acts_pre"].equal(cache["hook_sae_acts_pre"])
    assert train_cache["hook_sae_acts_post"].equal(cache["hook_sae_acts_post"])
    assert train_cache["hook_sae_recons"].equal(cache["hook_sae_recons"])
    assert train_cache["hook_sae_recons"].equal(cache["hook_sae_output"])


def test_JumpReLUSAE_initialization():
    cfg = build_jumprelu_sae_cfg(device="cpu")
    sae = JumpReLUSAE.from_dict(cfg.to_dict())
    assert isinstance(sae.W_enc, nn.Parameter)
    assert isinstance(sae.W_dec, nn.Parameter)
    assert isinstance(sae.b_enc, nn.Parameter)
    assert isinstance(sae.b_dec, nn.Parameter)
    assert isinstance(sae.threshold, nn.Parameter)

    assert sae.W_enc.shape == (cfg.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert sae.b_enc.shape == (cfg.d_sae,)
    assert sae.b_dec.shape == (cfg.d_in,)
    assert sae.threshold.shape == (cfg.d_sae,)

    # encoder/decoder should be initialized, everything else should be 0s
    assert_not_close(sae.W_enc, torch.zeros_like(sae.W_enc))
    assert_not_close(sae.W_dec, torch.zeros_like(sae.W_dec))
    assert_close(sae.b_dec, torch.zeros_like(sae.b_dec))
    assert_close(sae.b_enc, torch.zeros_like(sae.b_enc))
    assert_close(sae.threshold, torch.zeros_like(sae.threshold))


@pytest.mark.parametrize("use_error_term", [True, False])
def test_JumpReLUSAE_forward(use_error_term: bool):
    cfg = build_jumprelu_sae_cfg(d_in=2, d_sae=3)
    sae = JumpReLUSAE.from_dict(cfg.to_dict())
    sae.use_error_term = use_error_term
    sae.threshold.data = torch.tensor([1.0, 0.5, 0.25])
    sae.W_enc.data = torch.ones_like(sae.W_enc.data)
    sae.W_dec.data = torch.ones_like(sae.W_dec.data)
    sae.b_enc.data = torch.zeros_like(sae.b_enc.data)
    sae.b_dec.data = torch.zeros_like(sae.b_dec.data)

    sae_in = 0.3 * torch.ones(1, 2)
    expected_recons = torch.tensor([[1.2, 1.2]])
    # if we use error term, we should always get the same output as what we put in
    expected_output = sae_in if use_error_term else expected_recons
    out, cache = sae.run_with_cache(sae_in)
    assert_close(out, expected_output)
    assert_close(cache["hook_sae_input"], sae_in)
    assert_close(cache["hook_sae_output"], out)
    assert_close(cache["hook_sae_recons"], expected_recons)
    if use_error_term:
        assert_close(cache["hook_sae_error"], expected_output - expected_recons)

    assert_close(cache["hook_sae_acts_pre"], torch.tensor([[0.6, 0.6, 0.6]]))
    # the threshold of 1.0 should block the first latent from firing
    assert_close(cache["hook_sae_acts_post"], torch.tensor([[0.0, 0.6, 0.6]]))


def test_JumpReLUTrainingSAE_initialization():
    cfg = build_jumprelu_sae_training_cfg()
    sae = JumpReLUTrainingSAE.from_dict(cfg.to_dict())

    assert sae.W_enc.shape == (cfg.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert isinstance(sae.threshold, torch.nn.Parameter)
    assert sae.threshold.shape == (cfg.d_sae,)
    assert sae.b_enc.shape == (cfg.d_sae,)
    assert sae.b_dec.shape == (cfg.d_in,)
    assert isinstance(sae.activation_fn, torch.nn.ReLU)
    assert sae.device == torch.device("cpu")
    assert sae.dtype == torch.float32

    # biases
    assert_close(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert_close(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)

    # check if the decoder weight norm is 0.1 by default
    assert_close(
        sae.W_dec.norm(dim=1),
        0.1 * torch.ones_like(sae.W_dec.norm(dim=1)),
        atol=1e-6,
    )

    #  Default currently should be tranpose initialization
    assert_close(sae.W_enc, sae.W_dec.T, atol=1e-6)


def test_JumpReLUTrainingSAE_save_and_load_inference_sae(tmp_path: Path) -> None:
    # Create a training SAE with specific parameter values
    cfg = build_jumprelu_sae_training_cfg(device="cpu")
    training_sae = JumpReLUTrainingSAE(cfg)

    # Set some known values for testing
    training_sae.W_enc.data = torch.randn_like(training_sae.W_enc.data)
    training_sae.W_dec.data = torch.randn_like(training_sae.W_dec.data)
    training_sae.b_enc.data = torch.randn_like(training_sae.b_enc.data)
    training_sae.b_dec.data = torch.randn_like(training_sae.b_dec.data)
    training_sae.threshold.data = torch.rand_like(training_sae.threshold.data)

    # Save original state for comparison
    original_W_enc = training_sae.W_enc.data.clone()
    original_W_dec = training_sae.W_dec.data.clone()
    original_b_enc = training_sae.b_enc.data.clone()
    original_b_dec = training_sae.b_dec.data.clone()
    original_threshold = training_sae.threshold.data.clone()

    # Save as inference model
    model_path = str(tmp_path)
    training_sae.save_inference_model(model_path)

    assert os.path.exists(model_path)

    # Load as inference SAE
    inference_sae = SAE.load_from_disk(model_path, device="cpu")

    # Should be loaded as JumpReLUSAE
    assert isinstance(inference_sae, JumpReLUSAE)

    # Check that all parameters match
    assert_close(inference_sae.W_enc, original_W_enc)
    assert_close(inference_sae.W_dec, original_W_dec)
    assert_close(inference_sae.b_enc, original_b_enc)
    assert_close(inference_sae.b_dec, original_b_dec)

    # Most importantly, check that the threshold roundtrips to inference
    assert_close(inference_sae.threshold, original_threshold)

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


def test_calculate_pre_act_loss_dead_neuron_mask_none():
    """Test that pre-activation loss returns 0.0 when dead_neuron_mask is None."""
    d_sae = 8
    pre_act_loss_coefficient = 0.5
    threshold = torch.ones(d_sae) * 0.5
    hidden_pre = torch.randn(4, d_sae)
    dead_neuron_mask = None
    W_dec_norm = torch.ones(d_sae)

    loss = calculate_pre_act_loss(
        pre_act_loss_coefficient, threshold, hidden_pre, dead_neuron_mask, W_dec_norm
    )

    expected_loss = torch.tensor(0.0)
    assert_close(loss, expected_loss)


def test_calculate_pre_act_loss_normal_case():
    """Test pre-activation loss calculation with some dead neurons."""
    pre_act_loss_coefficient = 1.0
    threshold = torch.tensor([1.0, 2.0, 1.5, 0.5])
    # Create hidden_pre where some values are below threshold
    hidden_pre = torch.tensor(
        [
            [0.5, 1.5, 1.0, 0.8],  # batch 1: neurons 0,2 below threshold
            [0.8, 2.5, 1.2, 0.2],  # batch 2: neurons 0,3 below threshold
        ]
    )
    # Mark neurons 0 and 2 as dead
    dead_neuron_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    W_dec_norm = torch.tensor([2.0, 1.0, 1.5, 3.0])

    loss = calculate_pre_act_loss(
        pre_act_loss_coefficient, threshold, hidden_pre, dead_neuron_mask, W_dec_norm
    )

    # Calculate expected loss manually:
    # For each item in batch, calculate (threshold - hidden_pre).relu() * dead_neuron_mask * W_dec_norm
    # Batch 1:
    #   neuron 0: (1.0 - 0.5) * 1.0 * 2.0 = 1.0
    #   neuron 1: (2.0 - 1.5) * 0.0 * 1.0 = 0.0 (not dead)
    #   neuron 2: (1.5 - 1.0) * 1.0 * 1.5 = 0.75
    #   neuron 3: (0.5 - 0.8) * 0.0 * 3.0 = 0.0 (not dead, also negative so relu=0)
    #   Sum for batch 1: 1.0 + 0.0 + 0.75 + 0.0 = 1.75

    # Batch 2:
    #   neuron 0: (1.0 - 0.8) * 1.0 * 2.0 = 0.4
    #   neuron 1: (2.0 - 2.5) * 0.0 * 1.0 = 0.0 (not dead, also negative so relu=0)
    #   neuron 2: (1.5 - 1.2) * 1.0 * 1.5 = 0.45
    #   neuron 3: (0.5 - 0.2) * 0.0 * 3.0 = 0.0 (not dead)
    #   Sum for batch 2: 0.4 + 0.0 + 0.45 + 0.0 = 0.85

    # Mean across batch: (1.75 + 0.85) / 2 = 1.3
    # Final loss: pre_act_loss_coefficient * mean = 1.0 * 1.3 = 1.3
    expected_loss = torch.tensor(1.3)
    assert_close(loss, expected_loss, atol=1e-6)


def test_JumpReLUTrainingSAE_forward_tanh_sparsity_with_pre_act_loss():
    """Test training forward pass with tanh sparsity mode and pre-activation loss."""
    cfg = build_jumprelu_sae_training_cfg(
        jumprelu_sparsity_loss_mode="tanh",
        pre_act_loss_coefficient=0.1,
        l0_coefficient=0.5,
    )
    sae = JumpReLUTrainingSAE(cfg)

    batch_size = 4
    d_in = sae.cfg.d_in
    x = torch.randn(batch_size, d_in)

    # Create a dead neuron mask with some dead neurons
    dead_neuron_mask = torch.zeros(sae.cfg.d_sae)
    dead_neuron_mask[0] = 1.0  # Mark first neuron as dead
    dead_neuron_mask[2] = 1.0  # Mark third neuron as dead

    train_step_output = sae.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=x,
            coefficients={"l0": sae.cfg.l0_coefficient},
            dead_neuron_mask=dead_neuron_mask,
            n_training_steps=0,
            is_logging_step=False,
        ),
    )

    # Check that outputs have correct shapes
    assert train_step_output.sae_out.shape == (batch_size, d_in)
    assert train_step_output.feature_acts.shape == (batch_size, sae.cfg.d_sae)

    # Check that we have the expected loss components
    assert "mse_loss" in train_step_output.losses
    assert "l0_loss" in train_step_output.losses
    assert "pre_act_loss" in train_step_output.losses

    # Verify total loss is sum of components
    expected_total_loss = (
        train_step_output.losses["mse_loss"]
        + train_step_output.losses["l0_loss"]
        + train_step_output.losses["pre_act_loss"]
    )
    assert_close(train_step_output.loss, expected_total_loss, atol=1e-6)

    # Verify pre_act_loss is positive (since we have dead neurons)
    assert train_step_output.losses["pre_act_loss"] >= 0.0


def test_JumpReLUTrainingSAE_tanh_scale_increases_l0_loss():
    """Test that increasing jumprelu_tanh_scale increases l0_loss for tanh sparsity mode."""
    batch_size = 4

    # Create SAE with smaller tanh scale
    cfg_small = build_jumprelu_sae_training_cfg(
        jumprelu_sparsity_loss_mode="tanh",
        jumprelu_tanh_scale=2.0,
        l0_coefficient=1.0,
    )
    sae_small = JumpReLUTrainingSAE(cfg_small)

    # Create SAE with larger tanh scale (same architecture, different scale)
    cfg_large = build_jumprelu_sae_training_cfg(
        jumprelu_sparsity_loss_mode="tanh",
        jumprelu_tanh_scale=8.0,  # 4x larger than small
        l0_coefficient=1.0,
    )
    sae_large = JumpReLUTrainingSAE(cfg_large)

    # Use same weights for both SAEs to ensure fair comparison
    sae_large.W_enc.data = sae_small.W_enc.data.clone()
    sae_large.W_dec.data = sae_small.W_dec.data.clone()
    sae_large.b_enc.data = sae_small.b_enc.data.clone()
    sae_large.b_dec.data = sae_small.b_dec.data.clone()
    sae_large.threshold.data = sae_small.threshold.data.clone()

    # Use same input for both
    x = torch.randn(batch_size, cfg_small.d_in)

    # Forward pass with small tanh scale
    train_step_output_small = sae_small.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=x,
            coefficients={"l0": 1.0},
            dead_neuron_mask=None,
            n_training_steps=0,
            is_logging_step=False,
        ),
    )

    # Forward pass with large tanh scale
    train_step_output_large = sae_large.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=x,
            coefficients={"l0": 1.0},
            dead_neuron_mask=None,
            n_training_steps=0,
            is_logging_step=False,
        ),
    )

    # L0 loss should be larger with higher tanh scale
    l0_loss_small = train_step_output_small.losses["l0_loss"]
    l0_loss_large = train_step_output_large.losses["l0_loss"]

    assert (
        l0_loss_large > l0_loss_small
    ), f"Expected l0_loss_large ({l0_loss_large}) > l0_loss_small ({l0_loss_small})"

    # Verify the feature activations are the same (since weights are identical)
    assert_close(
        train_step_output_small.feature_acts, train_step_output_large.feature_acts
    )


def test_JumpReLUTrainingSAE_threshold_travels_at_the_adam_step_size():
    init_threshold = 0.01
    lr = 1e-3
    steps = 50
    cfg = build_jumprelu_sae_training_cfg(
        jumprelu_init_threshold=init_threshold, l0_coefficient=1.0
    )
    sae = JumpReLUTrainingSAE(cfg)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    for _ in range(steps):
        train_step_output = sae.training_forward_pass(
            step_input=TrainStepInput(
                sae_in=torch.randn(512, cfg.d_in),
                coefficients={"l0": 1.0},
                dead_neuron_mask=None,
                n_training_steps=0,
                is_logging_step=False,
            ),
        )
        optimizer.zero_grad()
        train_step_output.loss.backward()
        optimizer.step()

    movement = sae.threshold.detach().mean().item() - init_threshold
    # Adam normalizes by the gradient magnitude, so a step moves a parameter by
    # at most ~lr. Parameterizing the threshold as exp(log_threshold) therefore
    # caps its travel at ~init_threshold * lr * steps, which freezes it and
    # leaves the l0 penalty unable to reduce l0 at all (#494).
    assert movement > 20 * init_threshold * lr * steps
    assert movement < lr * steps


def test_JumpReLUTrainingSAE_loads_legacy_log_threshold_checkpoint(
    tmp_path: Path,
) -> None:
    cfg = build_jumprelu_sae_training_cfg(device="cpu")
    sae = JumpReLUTrainingSAE(cfg)
    sae.threshold.data = torch.rand_like(sae.threshold.data) + 0.1
    expected_threshold = sae.threshold.data.clone()

    legacy_state_dict = {k: v.clone() for k, v in sae.state_dict().items()}
    legacy_state_dict["log_threshold"] = torch.log(legacy_state_dict.pop("threshold"))
    save_file(legacy_state_dict, tmp_path / SAE_WEIGHTS_FILENAME)

    loaded_sae = JumpReLUTrainingSAE(cfg)
    loaded_sae.load_weights_from_checkpoint(tmp_path)

    assert_close(loaded_sae.threshold, expected_threshold, atol=1e-6)


def test_JumpReLUTrainingSAE_errors_on_invalid_sparsity_loss_mode():
    # Create SAE with smaller tanh scale
    cfg = build_jumprelu_sae_training_cfg(
        jumprelu_sparsity_loss_mode="nonsense",
        jumprelu_tanh_scale=2.0,
        l0_coefficient=1.0,
    )
    sae = JumpReLUTrainingSAE(cfg)

    x = torch.randn(64, cfg.d_in)

    with pytest.raises(ValueError):
        sae.training_forward_pass(
            step_input=TrainStepInput(
                sae_in=x,
                coefficients={"l0": 1.0},
                dead_neuron_mask=None,
                n_training_steps=0,
                is_logging_step=False,
            ),
        )


@pytest.mark.parametrize("ste_to_input", [True, False])
def test_JumpReLU_ste_to_input_matches_the_analytic_gradient(ste_to_input: bool):
    bandwidth = 0.5
    threshold = torch.full((4,), 1.0, requires_grad=True)
    # one pre-activation per case: below the window, inside it, above the threshold
    x = torch.tensor([[0.0, 0.9, 1.05, 3.0]], requires_grad=True)

    out = JumpReLU.apply(x, threshold, bandwidth, ste_to_input)
    out.sum().backward()  # type: ignore
    assert x.grad is not None and threshold.grad is not None

    in_window = ((x - threshold).abs() < bandwidth / 2).float()
    ste = threshold / bandwidth * in_window
    expected_x_grad = (x > threshold).float() + (ste if ste_to_input else 0.0)

    assert_close(x.grad, expected_x_grad, atol=1e-6)
    # the threshold gradient is unaffected by where the estimator is routed
    assert_close(threshold.grad, -ste.squeeze(0), atol=1e-6)


def test_JumpReLUTrainingSAE_ste_to_input_reaches_the_encoder_for_gated_off_latents():
    x = torch.ones(1, 2)

    def encoder_grad(ste_to_input: bool) -> torch.Tensor:
        sae = JumpReLUTrainingSAE(
            build_jumprelu_sae_training_cfg(
                d_in=2,
                d_sae=2,
                jumprelu_bandwidth=1.0,
                jumprelu_init_threshold=1.0,
                jumprelu_ste_to_input=ste_to_input,
            )
        )
        # pre-activations of 0.8 sit below the threshold of 1.0 but inside the
        # estimator's window of (0.5, 1.5), so nothing fires and any encoder
        # gradient can only have arrived through the estimator
        sae.W_enc.data = 0.8 * torch.eye(2)
        sae.b_enc.data = torch.zeros(2)
        feature_acts, _ = sae.encode_with_hidden_pre(x)
        assert (feature_acts == 0).all()
        feature_acts.sum().backward()
        assert sae.W_enc.grad is not None
        return sae.W_enc.grad

    # estimator value is threshold / bandwidth = 1.0, and x is all ones
    assert_close(encoder_grad(ste_to_input=True), torch.ones(2, 2), atol=1e-6)
    assert_close(encoder_grad(ste_to_input=False), torch.zeros(2, 2), atol=1e-6)


def test_JumpReLUTrainingSAE_keeps_threshold_in_float32_for_bfloat16_saes():
    lr, steps, init = 1e-3, 100, 1.1
    sae = JumpReLUTrainingSAE(
        build_jumprelu_sae_training_cfg(
            d_in=16,
            d_sae=32,
            dtype="bfloat16",
            jumprelu_init_threshold=init,
            jumprelu_bandwidth=2.0,
            l0_coefficient=1.0,
        )
    )
    assert sae.W_enc.dtype == torch.bfloat16
    assert sae.threshold.dtype == torch.float32

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    for _ in range(steps):
        output = sae.training_forward_pass(
            step_input=TrainStepInput(
                sae_in=torch.randn(64, sae.cfg.d_in, dtype=torch.bfloat16),
                coefficients={"l0": 1.0},
                dead_neuron_mask=None,
                n_training_steps=0,
                is_logging_step=False,
            )
        )
        # the threshold is cast at the use site, so the big tensors stay bfloat16
        assert output.feature_acts.dtype == torch.bfloat16
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()

    # bfloat16 resolves ~0.004 at 1.1, so a bfloat16 threshold would round away
    # most of the lr-sized steps and barely move
    moved = abs(sae.threshold.detach().float().mean().item() - init)
    assert moved > 20 * lr


def test_JumpReLUTrainingSAE_keeps_threshold_in_float32_across_a_disk_roundtrip(
    tmp_path: Path,
) -> None:
    sae = JumpReLUTrainingSAE(
        build_jumprelu_sae_training_cfg(dtype="bfloat16", jumprelu_init_threshold=1.1)
    )
    sae.save_model(str(tmp_path))

    loaded = TrainingSAE.load_from_disk(tmp_path, device="cpu")

    # load_from_disk ends in sae.to(dtype=cfg.dtype), which would otherwise
    # downcast the threshold and silently undo the guard
    assert loaded.W_enc.dtype == torch.bfloat16
    assert loaded.threshold.dtype == torch.float32
