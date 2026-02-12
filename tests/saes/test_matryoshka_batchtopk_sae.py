import copy
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
from sae_lens.saes.topk_sae import calculate_topk_aux_acts
from tests.helpers import (
    assert_close,
    assert_not_close,
    build_batchtopk_sae_training_cfg,
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
    output_mat = list(sae._iterable_decode(feature_acts, True))[-1][1]
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


def test_matryoshka_aux_loss_manual_computation():
    d_in = 4
    d_sae = 6
    k = 2
    widths = [3, 6]
    batch_size = 2

    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        matryoshka_widths=widths,
        use_matryoshka_aux_loss=True,
        rescale_acts_by_decoder_norm=False,
        device="cpu",
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    random_params(sae)

    sae_in = torch.randn(batch_size, d_in)
    hidden_pre = sae_in @ sae.W_enc + sae.b_enc

    # Only 1 dead feature in level 1 to exercise scale < 1.0
    # k_aux = d_in // 2 = 2, but only 1 dead feature, so scale = 1/2 = 0.5
    dead_neuron_mask = torch.zeros(d_sae, dtype=torch.bool)
    dead_neuron_mask[1] = True  # 1 dead in level 1
    dead_neuron_mask[3:6] = True  # all 3 dead in level 2

    acts = sae.activation_fn(hidden_pre)
    sae_out = acts @ sae.W_dec + sae.b_dec

    k_aux = d_in // 2  # = 2

    # Level 1: features 0:3, 1 dead feature
    partial_sae_out_1 = acts[:, 0:3] @ sae.W_dec.data[0:3] + sae.b_dec.data
    residual_1 = sae_in - partial_sae_out_1

    partial_hidden_pre_1 = hidden_pre[:, 0:3]
    partial_dead_mask_1 = dead_neuron_mask[0:3]
    num_dead_1 = int(partial_dead_mask_1.sum())  # = 1
    scale_1 = min(num_dead_1 / k_aux, 1.0)  # = 0.5
    partial_k_aux_1 = min(k_aux, num_dead_1)  # = 1
    auxk_acts_1 = calculate_topk_aux_acts(
        partial_k_aux_1, partial_hidden_pre_1, partial_dead_mask_1
    )
    recons_1 = auxk_acts_1 @ sae.W_dec.data[0:3]
    loss_1 = (recons_1 - residual_1).pow(2).sum(dim=-1).mean()

    # Level 2: features 3:6, 3 dead features
    partial_sae_out_2 = (
        acts[:, 0:3] @ sae.W_dec.data[0:3]
        + acts[:, 3:6] @ sae.W_dec.data[3:6]
        + sae.b_dec.data
    )
    residual_2 = sae_in - partial_sae_out_2

    partial_hidden_pre_2 = hidden_pre[:, 3:6]
    partial_dead_mask_2 = dead_neuron_mask[3:6]
    num_dead_2 = int(partial_dead_mask_2.sum())  # = 3
    scale_2 = min(num_dead_2 / k_aux, 1.0)  # = 1.0
    partial_k_aux_2 = min(k_aux, num_dead_2)  # = 2
    auxk_acts_2 = calculate_topk_aux_acts(
        partial_k_aux_2, partial_hidden_pre_2, partial_dead_mask_2
    )
    recons_2 = auxk_acts_2 @ sae.W_dec.data[3:6]
    loss_2 = (recons_2 - residual_2).pow(2).sum(dim=-1).mean()

    expected_loss = cfg.aux_loss_coefficient * (scale_1 * loss_1 + scale_2 * loss_2)

    actual_loss = sae.calculate_matryoshka_aux_loss(
        sae_in, sae_out, acts, hidden_pre, dead_neuron_mask
    )

    assert actual_loss.item() == pytest.approx(expected_loss.item())


def test_matryoshka_aux_loss_uses_level_residual_not_full_residual():
    d_in = 4
    d_sae = 6
    k = 2
    widths = [3, 6]
    batch_size = 4

    cfg_matryoshka = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        matryoshka_widths=widths,
        use_matryoshka_aux_loss=True,
        rescale_acts_by_decoder_norm=False,
        device="cpu",
    )
    sae_matryoshka = MatryoshkaBatchTopKTrainingSAE(cfg_matryoshka)

    cfg_base = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        matryoshka_widths=widths,
        use_matryoshka_aux_loss=False,
        rescale_acts_by_decoder_norm=False,
        device="cpu",
    )
    sae_base = MatryoshkaBatchTopKTrainingSAE(cfg_base)

    # Construct weights so level 1 reconstructs poorly but level 2 compensates.
    # Level 1 (features 0:3): set W_dec to zeros so it contributes nothing.
    # Level 2 (features 3:6): set W_dec to reconstruct well.
    for sae in [sae_matryoshka, sae_base]:
        random_params(sae)
        sae.W_dec.data[0:3] = 0.0  # Level 1 decoder does nothing
        sae.b_dec.data.zero_()

    # Ensure both SAEs have identical weights
    sae_base.load_state_dict(sae_matryoshka.state_dict())

    sae_in = torch.randn(batch_size, d_in)
    hidden_pre = sae_in @ sae_matryoshka.W_enc + sae_matryoshka.b_enc

    acts = sae_matryoshka.activation_fn(hidden_pre)
    sae_out = acts @ sae_matryoshka.W_dec + sae_matryoshka.b_dec

    # Make all features dead so both paths compute aux loss over all features
    dead_neuron_mask = torch.ones(d_sae, dtype=torch.bool)

    matryoshka_loss = sae_matryoshka.calculate_matryoshka_aux_loss(
        sae_in, sae_out, acts, hidden_pre, dead_neuron_mask
    )
    base_loss = sae_base.calculate_topk_aux_loss(
        sae_in, sae_out, hidden_pre, dead_neuron_mask
    )

    # The matryoshka loss should be larger because level 1's residual
    # is large (W_dec[0:3]=0 means level 1 reconstruction is just b_dec=0, so
    # residual_1 = sae_in). The base loss targets the full SAE residual which is smaller.
    assert matryoshka_loss.item() > base_loss.item()
    assert_not_close(matryoshka_loss, base_loss)


def test_matryoshka_aux_loss_gradients_flow_through_own_level_weights():
    d_in = 4
    d_sae = 6
    k = 2
    widths = [3, 6]
    batch_size = 4

    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        matryoshka_widths=widths,
        use_matryoshka_aux_loss=True,
        rescale_acts_by_decoder_norm=False,
        device="cpu",
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    random_params(sae)

    sae_in = torch.randn(batch_size, d_in)
    hidden_pre = sae_in @ sae.W_enc + sae.b_enc
    acts = sae.activation_fn(hidden_pre)
    sae_out = acts @ sae.W_dec + sae.b_dec

    # Only level 2 features (3:6) are dead
    dead_neuron_mask = torch.zeros(d_sae, dtype=torch.bool)
    dead_neuron_mask[3:6] = True

    sae.zero_grad()
    loss = sae.calculate_matryoshka_aux_loss(
        sae_in, sae_out, acts, hidden_pre, dead_neuron_mask
    )
    loss.backward()

    # Level 2 dead features should produce gradients on W_dec[3:6] only
    assert sae.W_dec.grad is not None
    assert sae.W_dec.grad[3:6].abs().sum() > 0, "Level 2 W_dec should have gradients"
    assert sae.W_dec.grad[0:3].abs().sum() == pytest.approx(
        0.0, abs=1e-7
    ), "Level 1 W_dec should have no gradients from level 2 dead features"

    # W_enc gradients should only be on the level 2 columns (3:6)
    # because hidden_pre[:, 3:6] only depends on W_enc[:, 3:6]
    assert sae.W_enc.grad is not None
    assert (
        sae.W_enc.grad[:, 3:6].abs().sum() > 0
    ), "Level 2 W_enc columns should have gradients"
    assert sae.W_enc.grad[:, 0:3].abs().sum() == pytest.approx(
        0.0, abs=1e-7
    ), "Level 1 W_enc columns should have no gradients from level 2 dead features"


def test_matryoshka_aux_loss_flag_off_matches_base_class():
    d_in = 4
    d_sae = 6
    k = 2
    widths = [3, 6]
    batch_size = 4

    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        matryoshka_widths=widths,
        use_matryoshka_aux_loss=False,
        rescale_acts_by_decoder_norm=False,
        device="cpu",
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)

    base_cfg = build_batchtopk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        rescale_acts_by_decoder_norm=False,
        device="cpu",
    )
    base_sae = BatchTopKTrainingSAE(base_cfg)

    # Copy weights
    base_sae.load_state_dict(sae.state_dict())

    sae_in = torch.randn(batch_size, d_in)
    hidden_pre = sae_in @ sae.W_enc + sae.b_enc
    acts = sae.activation_fn(hidden_pre)
    sae_out = acts @ sae.W_dec + sae.b_dec

    dead_neuron_mask = torch.ones(d_sae, dtype=torch.bool)

    matryoshka_loss = sae.calculate_topk_aux_loss(
        sae_in, sae_out, hidden_pre, dead_neuron_mask
    )
    base_loss = base_sae.calculate_topk_aux_loss(
        sae_in, sae_out, hidden_pre, dead_neuron_mask
    )

    assert_close(matryoshka_loss, base_loss)


def test_matryoshka_aux_loss_only_levels_with_dead_features_contribute():
    d_in = 4
    d_sae = 9
    k = 2
    widths = [3, 6, 9]
    batch_size = 4

    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        matryoshka_widths=widths,
        use_matryoshka_aux_loss=True,
        rescale_acts_by_decoder_norm=False,
        device="cpu",
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    random_params(sae)

    sae_in = torch.randn(batch_size, d_in)
    hidden_pre = sae_in @ sae.W_enc + sae.b_enc
    acts = sae.activation_fn(hidden_pre)
    sae_out = acts @ sae.W_dec + sae.b_dec

    # Only level 1 (features 0:3) has dead features
    dead_mask_level1_only = torch.zeros(d_sae, dtype=torch.bool)
    dead_mask_level1_only[0:3] = True

    loss_level1_only = sae.calculate_matryoshka_aux_loss(
        sae_in, sae_out, acts, hidden_pre, dead_mask_level1_only
    )

    # Only level 3 (features 6:9) has dead features
    dead_mask_level3_only = torch.zeros(d_sae, dtype=torch.bool)
    dead_mask_level3_only[6:9] = True

    loss_level3_only = sae.calculate_matryoshka_aux_loss(
        sae_in, sae_out, acts, hidden_pre, dead_mask_level3_only
    )

    assert loss_level1_only.item() > 0
    assert loss_level3_only.item() > 0

    # When no features are dead, loss should be zero
    dead_mask_none = torch.zeros(d_sae, dtype=torch.bool)
    loss_none = sae.calculate_matryoshka_aux_loss(
        sae_in, sae_out, acts, hidden_pre, dead_mask_none
    )
    assert loss_none.item() == pytest.approx(0.0)

    # Levels 1 and 3 have dead features, but level 2 (the middle) does not.
    # This exercises the prev_width tracking when a middle level is skipped.
    dead_mask_skip_middle = torch.zeros(d_sae, dtype=torch.bool)
    dead_mask_skip_middle[0:3] = True
    dead_mask_skip_middle[6:9] = True

    loss_skip_middle = sae.calculate_matryoshka_aux_loss(
        sae_in, sae_out, acts, hidden_pre, dead_mask_skip_middle
    )

    assert_close(loss_skip_middle, loss_level1_only + loss_level3_only)


@pytest.mark.parametrize("use_matryoshka_aux_loss", [True, False])
def test_aux_loss_rescale_acts_matches_fold_W_dec_norm(use_matryoshka_aux_loss: bool):
    d_in = 8
    d_sae = 16

    cfg = build_matryoshka_batchtopk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_sae,
        k=4,
        matryoshka_widths=[4, 8, 16],
        use_matryoshka_aux_loss=use_matryoshka_aux_loss,
        rescale_acts_by_decoder_norm=True,
        device="cpu",
    )
    sae_rescale = MatryoshkaBatchTopKTrainingSAE(cfg)
    random_params(sae_rescale)

    sae_folded = copy.deepcopy(sae_rescale)
    sae_folded.fold_W_dec_norm()
    sae_folded.cfg.rescale_acts_by_decoder_norm = False

    # Dead features in every level so all levels contribute to the aux loss
    dead_neuron_mask = torch.zeros(d_sae, dtype=torch.bool)
    dead_neuron_mask[1] = True  # 1 dead in level 1 (0:4)
    dead_neuron_mask[5:7] = True  # 2 dead in level 2 (4:8)
    dead_neuron_mask[12:16] = True  # 4 dead in level 3 (8:16)

    sae_in = torch.randn(10, d_in)
    train_step_input = TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=dead_neuron_mask,
        n_training_steps=0,
    )

    with torch.no_grad():
        output_rescale = sae_rescale.training_forward_pass(train_step_input)
        output_folded = sae_folded.training_forward_pass(train_step_input)

    assert output_rescale.losses["auxiliary_reconstruction_loss"].item() > 0
    assert_close(output_rescale.sae_out, output_folded.sae_out)
    assert_close(output_rescale.feature_acts, output_folded.feature_acts)
    assert_close(output_rescale.loss, output_folded.loss)
    for key in output_rescale.losses:
        assert_close(output_rescale.losses[key], output_folded.losses[key])
