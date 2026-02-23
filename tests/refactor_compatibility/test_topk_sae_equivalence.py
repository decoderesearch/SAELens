from typing import Union

import pytest
import torch

# New modules
from sae_lens.saes.sae import TrainStepInput
from sae_lens.saes.topk_sae import (
    TopKSAE,
    TopKSAEConfig,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
)

# Old modules
from tests._comparison.sae_lens_old.sae import (
    SAE as OldSAE,
)
from tests._comparison.sae_lens_old.sae import (
    SAEConfig as OldSAEConfig,
)
from tests._comparison.sae_lens_old.training.training_sae import (
    TrainingSAE as OldTrainingSAE,
)
from tests._comparison.sae_lens_old.training.training_sae import (
    TrainingSAEConfig as OldTrainingSAEConfig,
)
from tests.helpers import assert_close


@pytest.fixture
def seed_everything():
    """
    Ensure deterministic tests by setting a fixed random seed.
    """
    torch.manual_seed(42)
    yield
    torch.manual_seed(0)


def compare_params(
    old_model: Union[OldSAE, OldTrainingSAE],
    new_model: Union[TopKSAE, TopKTrainingSAE],
):
    """
    Compare parameter names and shapes between old and new SAEs.
    """
    old_params = dict(old_model.named_parameters())
    new_params = dict(new_model.named_parameters())
    assert sorted(old_params.keys()) == sorted(new_params.keys()), (
        f"Parameter names differ.\n"
        f"Old: {sorted(old_params.keys())}\nNew: {sorted(new_params.keys())}"
    )
    for k in old_params:
        assert old_params[k].shape == new_params[k].shape, (
            f"Param {k} shape mismatch: "
            f"old {old_params[k].shape} vs new {new_params[k].shape}"
        )


def make_old_topk_sae(
    d_in: int = 16, d_sae: int = 8, use_error_term: bool = False
) -> OldSAE:
    """
    Instantiate the old (original) topk SAE.
    """
    cfg = OldSAEConfig(
        architecture="topk",
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name="blocks.0.hook_resid_pre",  # avoid hook_z for simpler shape
        hook_layer=0,
        hook_head_index=None,
        activation_fn_str="topk",
        activation_fn_kwargs={"k": 4},  # example k
        apply_b_dec_to_input=False,
        finetuning_scaling_factor=False,
        normalize_activations="none",
        context_size=128,
        dataset_path="fake/path",
        dataset_trust_remote_code=False,
        sae_lens_training_version="test_version",
        model_from_pretrained_kwargs={},
        seqpos_slice=(None,),
        prepend_bos=False,
    )
    old_sae = OldSAE(cfg)
    old_sae.use_error_term = use_error_term
    return old_sae


def make_new_topk_sae(
    d_in: int = 16, d_sae: int = 8, use_error_term: bool = False
) -> TopKSAE:
    """
    Instantiate the new TopKSAE re-implementation.
    """
    new_cfg = TopKSAEConfig(
        k=4,
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        apply_b_dec_to_input=False,
        normalize_activations="none",
    )
    return TopKSAE(new_cfg, use_error_term=use_error_term)


def test_topk_sae_inference_equivalence():
    """
    Compare old vs new topk SAE on:
      - parameter shape
      - forward pass shape
      - optional error_term
    """
    old_sae = make_old_topk_sae(d_in=16, d_sae=8, use_error_term=False)
    new_sae = make_new_topk_sae(d_in=16, d_sae=8, use_error_term=False)

    compare_params(old_sae, new_sae)

    # Ensure parameters are identical before comparing outputs
    with torch.no_grad():
        old_params = dict(old_sae.named_parameters())
        new_params = dict(new_sae.named_parameters())
        for k in sorted(old_params.keys()):
            new_params[k].copy_(old_params[k])

    # Provide example input
    x = torch.randn(2, 4, 16, dtype=torch.float32)

    with torch.no_grad():
        old_out = old_sae(x)
        new_out = new_sae(x)

    assert old_out.shape == new_out.shape, "Output shape mismatch."
    assert torch.isfinite(old_out).all(), "Old SAE produced NaNs or inf"
    assert torch.isfinite(new_out).all(), "New SAE produced NaNs or inf"
    # Check for numerical equality
    assert_close(
        old_out,
        new_out,
        atol=1e-5,
        msg="Outputs differ between old and new implementations.",
    )

    # Now test with error_term
    old_sae_err = make_old_topk_sae(d_in=16, d_sae=8, use_error_term=True)
    new_sae_err = make_new_topk_sae(d_in=16, d_sae=8, use_error_term=True)

    # Align error term model parameters
    with torch.no_grad():
        old_params_err = dict(old_sae_err.named_parameters())
        new_params_err = dict(new_sae_err.named_parameters())
        for k in sorted(old_params_err.keys()):
            new_params_err[k].copy_(old_params_err[k])

    with torch.no_grad():
        old_err_out = old_sae_err(x)
        new_err_out = new_sae_err(x)

    # Check shape, finiteness, and numerical equality for error term outputs
    assert old_err_out.shape == new_err_out.shape
    assert torch.isfinite(old_err_out).all(), "Old error-term output has NaNs/inf"
    assert torch.isfinite(new_err_out).all(), "New error-term output has NaNs/inf"
    assert_close(
        old_err_out,
        new_err_out,
        atol=1e-5,
        msg="Error term outputs differ between old and new implementations.",
    )


def test_topk_sae_run_with_cache_equivalence():  # type: ignore
    """
    Compare hooking behavior for TopKSAE. We'll check that hooking triggers
    the same number of calls and that the actual values passed to hooks match.
    """

    old_sae = make_old_topk_sae()
    new_sae = make_new_topk_sae()

    # Ensure parameters are identical before comparing outputs
    with torch.no_grad():
        old_params = dict(old_sae.named_parameters())
        new_params = dict(new_sae.named_parameters())
        for k in sorted(old_params.keys()):
            new_params[k].copy_(old_params[k])

    x = torch.randn(2, 4, 16, dtype=torch.float32)
    with torch.no_grad():
        old_out, old_cache = old_sae.run_with_cache(x)
        new_out, new_cache = new_sae.run_with_cache(x)

    assert old_out.shape == new_out.shape, "Output shape mismatch."
    assert_close(
        old_out,
        new_out,
        atol=1e-5,
        msg="Output values differ.",
    )

    assert len(old_cache) == len(new_cache), "Cache length mismatch."

    for old_key, new_key in zip(sorted(old_cache.keys()), sorted(new_cache.keys())):
        assert old_key == new_key, f"Cache keys differ: {old_key} vs {new_key}"
        assert_close(
            old_cache[old_key],
            new_cache[new_key],
            atol=1e-5,
            msg=f"Cache values differ for key: {old_key}",
        )


def test_topk_sae_training_equivalence():
    """
    Compare old vs new topk training SEAs on:
      - parameter shape
      - training forward pass shape
      - partial correctness of losses
    """
    # Build old vs new training configs
    old_training_cfg = OldTrainingSAEConfig(
        architecture="topk",
        d_in=16,
        d_sae=8,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name="blocks.0.hook_resid_pre",
        hook_layer=0,
        hook_head_index=None,
        activation_fn_str="topk",
        activation_fn_kwargs={"k": 4},
        apply_b_dec_to_input=False,
        finetuning_scaling_factor=False,
        normalize_activations="none",
        context_size=128,
        dataset_path="fake/path",
        dataset_trust_remote_code=False,
        sae_lens_training_version="test_version",
        model_from_pretrained_kwargs={},
        seqpos_slice=(None,),
        prepend_bos=False,
        # Training-specific:
        l1_coefficient=0.01,
        lp_norm=1.0,
        use_ghost_grads=False,
        normalize_sae_decoder=False,
        noise_scale=0.0,
        decoder_orthogonal_init=False,
        mse_loss_normalization=None,
        jumprelu_init_threshold=0.0,
        jumprelu_bandwidth=1.0,
        decoder_heuristic_init=False,
        init_encoder_as_decoder_transpose=False,
        scale_sparsity_penalty_by_decoder_norm=False,
        decoder_heuristic_init_norm=0.1,
    )

    old_training_sae = OldTrainingSAE(old_training_cfg)
    # Convert old config dict for new config, applying defaults
    new_training_cfg = TopKTrainingSAEConfig(
        k=4,
        d_in=16,
        d_sae=8,
        rescale_acts_by_decoder_norm=False,
    )
    new_training_sae = TopKTrainingSAE(new_training_cfg)

    # Compare param shapes using updated compare_params
    compare_params(old_training_sae, new_training_sae)

    # Align parameters for numerical comparison
    with torch.no_grad():
        old_params = dict(old_training_sae.named_parameters())
        new_params = dict(new_training_sae.named_parameters())
        for k in sorted(old_params.keys()):
            new_params[k].copy_(old_params[k])

    # Provide random input
    x = torch.randn(2, 3, 16, dtype=torch.float32)

    old_training_sae.train()
    new_training_sae.train()

    old_out = old_training_sae.training_forward_pass(
        sae_in=x,
        current_l1_coefficient=old_training_cfg.l1_coefficient,
        dead_neuron_mask=None,
    )
    new_out = new_training_sae.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=x,
            coefficients={},  # topk SAEs don't care about L1 coefficient
            dead_neuron_mask=None,
            n_training_steps=0,
            is_logging_step=False,
        )
    )

    assert_close(
        old_out.sae_out,
        new_out.sae_out,
        atol=1e-5,
        msg="Training sae_out differs between old and new implementations.",
    )
    assert_close(
        old_out.loss,
        new_out.loss,
        atol=1e-5,
        msg="Training loss differs between old and new implementations.",
    )
    assert_close(
        old_out.feature_acts,
        new_out.feature_acts.to_dense(),
        atol=1e-5,
        msg="Training feature_acts differ between old and new implementations.",
    )
    assert_close(
        old_out.hidden_pre,
        new_out.hidden_pre,
        atol=1e-5,
        msg="Training hidden_pre differ between old and new implementations.",
    )
