import os
from pathlib import Path

import pytest
import torch
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
    MatryoshkaBatchTopKTrainer,
)

from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE
from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from sae_lens.saes.matryoshka_batchtopk_sae import (
    MatryoshkaBatchTopKTrainingSAE,
    _validate_matryoshka_config,
)
from sae_lens.saes.sae import SAE, TrainStepInput
from tests.helpers import (
    assert_close,
    assert_not_close,
    build_matryoshka_batchtopk_sae_training_cfg,
    random_params,
)


def test_MatryoshkaBatchTopKTrainingSAEConfig_initialization():
    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=10,
        d_sae=20,
        matryoshka_widths=[5, 10, 20],
    )
    assert cfg.matryoshka_widths == [5, 10, 20]
    assert cfg.d_in == 10
    assert cfg.d_sae == 20


def test_validate_matryoshka_config_appends_d_sae_if_missing():
    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_sae=20,
        k=5,
        matryoshka_widths=[5, 10],
    )
    _validate_matryoshka_config(cfg)
    assert cfg.matryoshka_widths == [5, 10, 20]


def test_validate_matryoshka_config_does_not_append_d_sae_if_already_present():
    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_sae=20,
        k=5,
        matryoshka_widths=[5, 10, 20],
    )
    _validate_matryoshka_config(cfg)
    assert cfg.matryoshka_widths == [5, 10, 20]


def test_validate_matryoshka_config_raises_if_not_strictly_increasing():
    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_sae=20,
        matryoshka_widths=[5, 10, 10, 20],
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate_matryoshka_config(cfg)


def test_validate_matryoshka_config_raises_if_decreasing():
    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_sae=20,
        matryoshka_widths=[10, 5, 20],
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate_matryoshka_config(cfg)


def test_MatryoshkaBatchTopKTrainingSAE_initialization():
    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=10,
        d_sae=20,
        k=5,
        matryoshka_widths=[5, 10, 20],
        device="cpu",
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)

    assert sae.W_enc.shape == (10, 20)
    assert sae.W_dec.shape == (20, 10)
    assert sae.b_enc.shape == (20,)
    assert sae.b_dec.shape == (10,)
    assert cfg.matryoshka_widths == [5, 10, 20]


def test_MatryoshkaBatchTopKTrainingSAE_training_forward_pass_computes_inner_losses():
    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=8,
        d_sae=16,
        k=4,
        matryoshka_widths=[4, 8, 16],
        device="cpu",
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    random_params(sae)

    sae_in = torch.randn(10, 8)
    train_step_input = TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
    )

    output = sae.training_forward_pass(train_step_input)

    assert "inner_mse_loss_4" in output.losses
    assert "inner_mse_loss_8" in output.losses
    assert "inner_mse_loss_16" not in output.losses

    assert output.losses["inner_mse_loss_4"].item() >= 0
    assert output.losses["inner_mse_loss_8"].item() >= 0


def test_MatryoshkaBatchTopKTrainingSAE_training_forward_pass_adds_inner_losses_to_total_loss():
    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=8,
        d_sae=16,
        k=4,
        matryoshka_widths=[4, 8, 16],
        device="cpu",
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    random_params(sae)

    sae_in = torch.randn(10, 8)
    train_step_input = TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
    )

    output = sae.training_forward_pass(train_step_input)

    expected_loss = (
        output.losses["mse_loss"]
        + output.losses["inner_mse_loss_4"]
        + output.losses["inner_mse_loss_8"]
    )
    assert_close(output.loss, expected_loss)


def test_MatryoshkaBatchTopKTrainingSAE_with_single_matryoshka_level_matches_batchtopk():
    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=8,
        d_sae=16,
        k=4,
        matryoshka_widths=[16],
        device="cpu",
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    random_params(sae)

    btk_sae = BatchTopKTrainingSAE(cfg)
    btk_sae.load_state_dict(sae.state_dict())

    sae_in = torch.randn(10, 8)
    train_step_input = TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
    )

    output = sae.training_forward_pass(train_step_input)
    btk_output = btk_sae.training_forward_pass(train_step_input)

    assert len([k for k in output.losses if k.startswith("inner_mse_loss")]) == 0
    assert_close(output.loss, btk_output.loss)
    assert_close(output.hidden_pre, btk_output.hidden_pre)
    assert_close(output.sae_out, btk_output.sae_out)
    assert_close(output.feature_acts, btk_output.feature_acts)


def test_MatryoshkaBatchTopKTrainingSAE_with_two_matryoshka_levels():
    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=8,
        d_sae=16,
        k=4,
        matryoshka_widths=[8, 16],
        device="cpu",
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    random_params(sae)

    sae_in = torch.randn(10, 8)
    train_step_input = TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
    )

    output = sae.training_forward_pass(train_step_input)

    assert "inner_mse_loss_8" in output.losses


@pytest.mark.parametrize("rescale_acts_by_decoder_norm", [True, False])
def test_MatryoshkaBatchTopKTrainingSAE_decode_matryoshka_level_matches_standard_decode_at_full_width(
    rescale_acts_by_decoder_norm: bool,
):
    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=8,
        d_sae=16,
        k=4,
        matryoshka_widths=[4, 16],
        rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
        device="cpu",
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    random_params(sae)

    feature_acts = torch.randn(10, 16)
    inv_W_dec_norm = 1 / sae.W_dec.norm(dim=-1)
    output_mat = sae._decode_matryoshka_level(feature_acts, 16, inv_W_dec_norm)
    output_base = sae.decode(feature_acts)
    assert_close(output_mat, output_base)
    assert output_mat.shape == (10, 8)


@pytest.mark.parametrize("rescale_acts_by_decoder_norm", [True, False])
def test_MatryoshkaBatchTopKTrainingSAE_save_and_load_inference_sae(
    tmp_path: Path,
    rescale_acts_by_decoder_norm: bool,
):
    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=8,
        d_sae=16,
        k=4,
        matryoshka_widths=[4, 8, 16],
        rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
        device="cpu",
    )
    training_sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    random_params(training_sae)

    sae_in = torch.randn(30, training_sae.cfg.d_in)
    train_step_input = TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
    )

    # run some test data through to learn the correct threshold
    for _ in range(500):
        training_sae.training_forward_pass(train_step_input)

    # Save original state for comparison
    original_W_enc = training_sae.W_enc.data.clone()
    original_W_dec = training_sae.W_dec.data.clone()
    original_b_enc = training_sae.b_enc.data.clone()
    original_b_dec = training_sae.b_dec.data.clone()
    original_threshold = training_sae.topk_threshold.item()

    # Save as inference model
    model_path = str(tmp_path)
    training_sae.save_inference_model(model_path)

    assert os.path.exists(model_path)

    # Load as inference SAE
    inference_sae = SAE.load_from_disk(model_path, device="cpu")

    # Should be loaded as JumpReLUSAE
    assert isinstance(inference_sae, JumpReLUSAE)

    # Check that all parameters match
    if rescale_acts_by_decoder_norm:
        assert_not_close(inference_sae.W_dec, original_W_dec)
        assert_close(
            inference_sae.W_dec.norm(dim=-1),
            torch.ones_like(inference_sae.b_enc),
        )
        assert_not_close(inference_sae.W_enc, original_W_enc)
        assert_not_close(inference_sae.b_enc, original_b_enc)
    else:
        assert_close(inference_sae.W_dec, original_W_dec)
        assert_close(inference_sae.W_enc, original_W_enc)
        assert_close(inference_sae.b_enc, original_b_enc)
    assert_close(inference_sae.b_dec, original_b_dec)

    # Check that topk_threshold was converted to threshold
    assert_close(
        inference_sae.threshold,
        original_threshold * torch.ones_like(inference_sae.b_enc),
    )

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


def test_MatryoshkaBatchTopKTrainingSAE_matches_dictionary_learning() -> None:
    """
    Test that our MatryoshkaBatchTopKTrainingSAE implementation produces the same
    outputs as the dictionary_learning MatryoshkaBatchTopKSAE implementation.
    """
    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=5,
        d_sae=10,
        k=2,
        matryoshka_widths=[2, 3, 4, 10],
        device="cpu",
        apply_b_dec_to_input=True,  # dictionary_learning subtracts b_dec from input
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    random_params(sae)
    sae.fold_W_dec_norm()

    # Create comparison SAE from dictionary_learning
    # group_sizes=[2, 1, 1, 6] corresponds to widths [2, 3, 4, 10]
    comparison_sae = MatryoshkaBatchTopKSAE(
        activation_dim=5,
        dict_size=10,
        k=2,
        group_sizes=[2, 1, 1, 6],
    )

    # Copy parameters from our SAE to the comparison SAE
    comparison_sae.W_enc.data = sae.W_enc.data
    comparison_sae.W_dec.data = sae.W_dec.data
    comparison_sae.b_dec.data = sae.b_dec.data
    comparison_sae.b_enc.data = sae.b_enc.data
    comparison_sae.threshold.data = sae.topk_threshold.data  # type: ignore

    x = torch.randn(4, 5)

    with torch.no_grad():
        output = sae.training_forward_pass(TrainStepInput(x, {}, None, 0))

        comp_feats = comparison_sae.encode(x, use_threshold=False)
        comp_sae_out = comparison_sae.decode(comp_feats)  # type: ignore

    assert_close(output.feature_acts, comp_feats)  # type: ignore
    assert_close(output.sae_out, comp_sae_out)


def test_MatryoshkaBatchTopKTrainingSAE_matches_dictionary_learning_losses() -> None:
    """
    Test that our MatryoshkaBatchTopKTrainingSAE losses match the dictionary_learning
    MatryoshkaBatchTopKTrainer losses.
    """
    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=5,
        d_sae=10,
        k=2,
        matryoshka_widths=[2, 3, 4, 10],
        device="cpu",
        apply_b_dec_to_input=True,  # dictionary_learning subtracts b_dec from input
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    random_params(sae)
    sae.fold_W_dec_norm()

    # Create comparison trainer from dictionary_learning
    # group_fractions=[0.2, 0.1, 0.1, 0.6] corresponds to widths [2, 3, 4, 10]
    comparison_trainer = MatryoshkaBatchTopKTrainer(
        steps=100,
        activation_dim=5,
        dict_size=10,
        k=2,
        layer=0,
        warmup_steps=0,
        lm_name="gpt2",
        group_fractions=[0.2, 0.1, 0.1, 0.6],
        group_weights=[1.0, 1.0, 1.0, 1.0],
        device="cpu",
    )

    # Copy parameters from our SAE to the comparison trainer
    comparison_trainer.ae.W_enc.data = sae.W_enc.data
    comparison_trainer.ae.W_dec.data = sae.W_dec.data
    comparison_trainer.ae.b_dec.data = sae.b_dec.data
    comparison_trainer.ae.b_enc.data = sae.b_enc.data
    comparison_trainer.ae.threshold = sae.topk_threshold

    x = torch.randn(4, 5)

    with torch.no_grad():
        output = sae.training_forward_pass(TrainStepInput(x, {}, None, 0))
        comp_losses = comparison_trainer.loss(x, logging=True, step=0).losses  # type: ignore

    # Compare losses
    combined_loss = (
        output.losses["mse_loss"]
        + output.losses["inner_mse_loss_2"]
        + output.losses["inner_mse_loss_3"]
        + output.losses["inner_mse_loss_4"]
    )
    num_levels = len(cfg.matryoshka_widths)

    # dictionary_learning loss is divided by the number of levels
    assert combined_loss.item() / num_levels == pytest.approx(comp_losses["loss"])
    assert output.loss.item() / num_levels == pytest.approx(comp_losses["loss"])
