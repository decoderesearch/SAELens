import pytest
import torch

from sae_lens.saes.gated_sae import (
    GatedSAE,
    GatedSAEConfig,
    GatedTrainingSAE,
    GatedTrainingSAEConfig,
)
from sae_lens.saes.sae import (
    TrainStepInput,
)
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
    Ensure deterministic behavior for tests by setting a fixed random seed.
    """
    torch.manual_seed(42)
    yield
    torch.manual_seed(0)


def make_old_gated_sae(
    d_in: int = 16, d_sae: int = 8, use_error_term: bool = False
) -> OldSAE:
    """
    Helper to instantiate an old Gated SAE instance for testing.
    This creates an old SAE with architecture='gated'.
    """
    old_cfg = OldSAEConfig(
        architecture="gated",
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name="blocks.0.hook_resid_pre",
        hook_layer=0,
        hook_head_index=None,
        activation_fn_str="relu",
        activation_fn_kwargs={},
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
    old_sae = OldSAE(old_cfg)
    # Note: Gated SAE doesn't support error term, so this should be False
    old_sae.use_error_term = use_error_term
    return old_sae


def make_new_gated_sae(
    d_in: int = 16, d_sae: int = 8, use_error_term: bool = False
) -> GatedSAE:
    """
    Helper to instantiate a new GatedSAE instance for testing (inference only).
    """
    new_cfg = GatedSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        apply_b_dec_to_input=False,
        normalize_activations="none",
    )
    return GatedSAE(new_cfg, use_error_term=use_error_term)


def compare_params(
    old_sae: OldSAE | OldTrainingSAE, new_sae: GatedSAE | GatedTrainingSAE
):
    """
    Compare parameter names and shapes between the old Gated SAE and the new GatedSAE.
    """
    old_params = dict(old_sae.named_parameters())
    new_params = dict(new_sae.named_parameters())
    old_keys = sorted(old_params.keys())
    new_keys = sorted(new_params.keys())

    assert (
        old_keys == new_keys
    ), f"Parameter names differ.\nOld: {old_keys}\nNew: {new_keys}"

    for key in old_keys:
        v_old = old_params[key]
        v_new = new_params[key]
        assert (
            v_old.shape == v_new.shape
        ), f"Param {key} shape mismatch: old {v_old.shape}, new {v_new.shape}"


@pytest.mark.parametrize("use_error_term", [False])
def test_gated_inference_equivalence(use_error_term):  # type: ignore
    """
    Test that the old vs new Gated SAEs match in parameter shape and forward pass outputs.
    Note: Gated SAE doesn't support error_term=True.
    """
    old_sae = make_old_gated_sae(d_in=16, d_sae=8, use_error_term=use_error_term)
    new_sae = make_new_gated_sae(d_in=16, d_sae=8, use_error_term=use_error_term)

    # Ensure parameters are identical before comparing outputs
    with torch.no_grad():
        old_params = dict(old_sae.named_parameters())
        new_params = dict(new_sae.named_parameters())
        for k in sorted(old_params.keys()):
            new_params[k].copy_(old_params[k])

    # Compare parameter shapes
    compare_params(old_sae, new_sae)

    # Provide a random input
    x = torch.randn(2, 4, 16, dtype=torch.float32)

    with torch.no_grad():
        old_out = old_sae(x)
        new_out = new_sae(x)

    assert old_out.shape == new_out.shape, "Output shape mismatch."
    assert torch.isfinite(old_out).all(), "Old output contains non-finite values."
    assert torch.isfinite(new_out).all(), "New output contains non-finite values."

    # Now they really should match, since we forcibly aligned params
    assert_close(
        old_out,
        new_out,
        atol=1e-5,
        msg="Outputs differ between old and new implementations.",
    )


@pytest.mark.parametrize(
    "fold_fn",
    [
        # "fold_W_dec_norm", # this has changed in the refactor, it seems like old behavior was wrong.
        "fold_activation_norm_scaling_factor",
    ],
)
def test_gated_fold_equivalence(fold_fn):  # type: ignore
    """
    Test that folding functions (fold_W_dec_norm or fold_activation_norm_scaling_factor)
    on old vs new Gated SAE yields consistent results on forward passes.

    This is especially important for fold_W_dec_norm, which has gated-specific logic.
    """
    old_sae = make_old_gated_sae(use_error_term=False)
    new_sae = make_new_gated_sae(use_error_term=False)
    compare_params(old_sae, new_sae)

    # We'll line up parameters by name so that p_old and p_new actually match in shape.
    old_params = dict(old_sae.named_parameters())
    new_params = dict(new_sae.named_parameters())

    # fill random data so that norms differ
    with torch.no_grad():
        for k in sorted(old_params.keys()):
            rand = torch.rand_like(old_params[k])
            old_params[k].copy_(rand)
            new_params[k].copy_(rand)

    # Now call the fold function
    if fold_fn == "fold_W_dec_norm":
        old_sae.fold_W_dec_norm()
        new_sae.fold_W_dec_norm()
    elif fold_fn == "fold_activation_norm_scaling_factor":
        scale_factor = 2.0
        old_sae.fold_activation_norm_scaling_factor(scale_factor)
        new_sae.fold_activation_norm_scaling_factor(scale_factor)

    # Provide input, compare outputs
    x = torch.randn(2, 3, 16, dtype=torch.float32)
    old_out = old_sae(x)
    new_out = new_sae(x)

    assert old_out.shape == new_out.shape, f"Output shape mismatch after {fold_fn}"

    # Compare the actual values - they should match closely after folding
    assert_close(
        old_out,
        new_out,
        atol=1e-5,
        msg=f"{fold_fn} produces different results between old and new implementations",
    )

    # Also check the folded parameters directly
    for k in sorted(old_params.keys()):
        assert_close(
            old_params[k],
            new_params[k],
            atol=1e-5,
            msg=f"Parameter {k} differs after {fold_fn}",
        )


def test_gated_run_with_cache_equivalence():  # type: ignore
    """
    Compare hooking behavior for GatedSAE. We'll check that hooking triggers
    the same number of calls and that the actual values passed to hooks match.
    """

    old_sae = make_old_gated_sae()
    new_sae = make_new_gated_sae()

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

    for old_key, new_key in zip(old_cache.keys(), new_cache.keys()):
        assert old_key == new_key, "Cache keys differ."
        assert_close(
            old_cache[old_key],
            new_cache[new_key],
            atol=1e-5,
            msg="Cache values differ.",
        )


#####################################
# Training Equivalence
#####################################


def make_old_gated_training_sae(d_in: int = 16, d_sae: int = 8) -> OldTrainingSAE:
    """
    Helper to instantiate an old TrainingSAE configured as Gated for testing.
    """
    old_training_cfg = OldTrainingSAEConfig(
        activation_fn_str="relu",
        architecture="gated",
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name="blocks.0.hook_resid_pre",
        hook_layer=0,
        hook_head_index=None,
        activation_fn_kwargs={},
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
        # training fields
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
        decoder_heuristic_init_norm=0.1,
        init_encoder_as_decoder_transpose=False,
        scale_sparsity_penalty_by_decoder_norm=False,
    )
    return OldTrainingSAE(old_training_cfg)


def make_new_gated_training_sae(d_in: int = 16, d_sae: int = 8) -> GatedTrainingSAE:
    """
    Helper to instantiate a new GatedTrainingSAE instance.
    """
    new_training_cfg = GatedTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        apply_b_dec_to_input=False,
        normalize_activations="none",
        l1_coefficient=0.01,
    )
    return GatedTrainingSAE(new_training_cfg)


def test_gated_training_equivalence():  # type: ignore
    """
    Test that old vs new Gated SAEs match in training behavior.
    We'll check the outputs, losses, and ensure numeric equivalence.
    """
    old_sae = make_old_gated_training_sae()
    new_sae = make_new_gated_training_sae()

    # Ensure parameters are identical before comparing outputs
    with torch.no_grad():
        old_params = dict(old_sae.named_parameters())
        new_params = dict(new_sae.named_parameters())
        for k in sorted(old_params.keys()):
            new_params[k].copy_(old_params[k])

    # Put in training mode
    old_sae.train()
    new_sae.train()

    # Compare parameters
    compare_params(old_sae, new_sae)

    # Create consistent random data
    batch_size, seq_len, d_in = 2, 4, 16
    x = torch.randn(batch_size, seq_len, d_in, dtype=torch.float32)

    # Get training outputs
    old_out = old_sae.training_forward_pass(
        sae_in=x,
        current_l1_coefficient=old_sae.cfg.l1_coefficient,
        dead_neuron_mask=None,
    )
    new_out = new_sae.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=x,
            coefficients={"l1": new_sae.cfg.l1_coefficient},
            dead_neuron_mask=None,
            n_training_steps=0,
            is_logging_step=False,
        )
    )

    # Check output shapes
    assert (
        old_out.sae_out.shape == new_out.sae_out.shape
    ), "Gated training output shape mismatch."

    # Check all values are finite
    assert torch.isfinite(
        old_out.sae_out
    ).all(), "Old Gated training out is not finite."
    assert torch.isfinite(
        new_out.sae_out
    ).all(), "New Gated training out is not finite."
    assert torch.isfinite(old_out.loss).all(), "Old Gated training loss is not finite."
    assert torch.isfinite(new_out.loss).all(), "New Gated training loss is not finite."

    # Check for losses present in both
    assert "mse_loss" in old_out.losses, "Missing MSE loss in old implementation"
    assert "mse_loss" in new_out.losses, "Missing MSE loss in new implementation"

    # Gated SAE has specific losses
    assert (
        "l1_loss" in old_out.losses or "auxiliary_reconstruction_loss" in old_out.losses
    ), "Old Gated training missing expected loss terms."
    assert (
        "l1_loss" in new_out.losses or "gating_aux_loss" in new_out.losses
    ), "New Gated training missing expected loss terms."

    # Check if training forward pass is equivalent
    assert_close(
        old_out.sae_out,
        new_out.sae_out,
        atol=1e-5,
        msg="Output differs between old and new Gated implementation",
    )
    assert_close(
        old_out.loss,
        new_out.loss,
        atol=1e-5,
        msg="Loss differs between old and new Gated implementation",
    )
