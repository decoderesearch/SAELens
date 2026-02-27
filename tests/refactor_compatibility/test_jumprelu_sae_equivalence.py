import pytest
import torch

from sae_lens.saes.jumprelu_sae import (
    JumpReLUSAE,
    JumpReLUSAEConfig,
    JumpReLUTrainingSAE,
    JumpReLUTrainingSAEConfig,
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


def make_old_jumprelu_sae(
    d_in: int = 16, d_sae: int = 8, use_error_term: bool = False
) -> OldSAE:  # Added types
    """
    Helper to instantiate an old JumpReLU SAE instance for testing.
    This creates an old SAE with architecture='jumprelu'.
    """
    # We replicate the logic from test_standard_sae_equivalence.make_old_sae,
    # but specify architecture="jumprelu".
    old_cfg = OldSAEConfig(  # Use OldSAEConfig
        architecture="jumprelu",
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
        context_size=128,  # Added default value
        dataset_path="fake/path",  # Added default value
        dataset_trust_remote_code=False,
        sae_lens_training_version="test_version",
        model_from_pretrained_kwargs={},
        seqpos_slice=(None,),  # Added default value (tuple as in old config)
        prepend_bos=False,
    )
    old_sae = OldSAE(old_cfg)  # Use OldSAE
    old_sae.use_error_term = use_error_term
    return old_sae


def make_new_jumprelu_sae(
    d_in: int = 16, d_sae: int = 8, use_error_term: bool = False
) -> JumpReLUSAE:  # Added types
    """
    Helper to instantiate a new JumpReLUSAE instance for testing (inference only).
    """
    new_cfg = JumpReLUSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        apply_b_dec_to_input=False,
        normalize_activations="none",
    )
    return JumpReLUSAE(new_cfg, use_error_term=use_error_term)


def compare_params(
    old_sae: OldSAE | OldTrainingSAE, new_sae: JumpReLUSAE | JumpReLUTrainingSAE
):  # Updated types
    """
    Compare parameter names and shapes between the old JumpReLU SAE and the new JumpReLUSAE.
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


@pytest.mark.parametrize("use_error_term", [False, True])
def test_jumprelu_inference_equivalence(use_error_term: bool):  # Added type
    """
    Test that the old vs new JumpReLU SAEs match in parameter shape and forward pass outputs,
    and can optionally test the error_term usage.
    """
    old_sae = make_old_jumprelu_sae(d_in=16, d_sae=8, use_error_term=use_error_term)
    new_sae = make_new_jumprelu_sae(d_in=16, d_sae=8, use_error_term=use_error_term)

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
    assert torch.isfinite(old_out).all()
    assert torch.isfinite(new_out).all()

    # Check for numerical equivalence
    assert_close(
        old_out,
        new_out,
        atol=1e-5,
        msg="Outputs differ between old and new implementations.",
    )

    # If error_term is True, they might diverge for non-standard architectures,
    # but let's allow a tolerance for closeness or just check shape
    # It's okay if they differ numerically, but let's see if "jumprelu" lines up
    # No need for specific check if use_error_term is True, allclose covers it.
    # if use_error_term:
    #     # We might not expect exact equality, but we can still check shape
    #     assert old_out.shape == new_out.shape


@pytest.mark.parametrize(
    "fold_fn", ["fold_W_dec_norm", "fold_activation_norm_scaling_factor"]
)
def test_jumprelu_fold_equivalence(fold_fn):  # type: ignore
    """
    Test that folding functions (fold_W_dec_norm or fold_activation_norm_scaling_factor)
    on old vs new JumpReLU yields consistent results on forward passes.
    """
    old_sae = make_old_jumprelu_sae(use_error_term=False)
    new_sae = make_new_jumprelu_sae(use_error_term=False)
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

    assert old_out.shape == new_out.shape
    # The numeric results might not match exactly if JumpReLU is stricter with thresholds,
    # but let's see if they match within a tolerance. If they do for your code, match them strictly.
    assert_close(
        old_out,
        new_out,
        atol=1e-5,
        msg=f"{fold_fn} mismatch between old and new",
    )

    # Also check the folded parameters directly
    for k in sorted(old_params.keys()):
        assert_close(
            old_params[k],
            new_params[k],
            atol=1e-5,
            msg=f"Parameter {k} differs after {fold_fn}",
        )


def test_jumprelu_run_with_cache_equivalence():  # type: ignore
    """
    Compare run_with_cache behavior for JumpReLUSAE.
    Checks outputs and cache contents for numerical equivalence.
    """
    old_sae = make_old_jumprelu_sae()
    new_sae = make_new_jumprelu_sae()

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

    assert len(old_cache) == len(
        new_cache
    ), f"Cache length mismatch. Old: {len(old_cache)}, New: {len(new_cache)}"

    # Sort keys to ensure consistent comparison order
    old_keys = sorted(old_cache.keys())
    new_keys = sorted(new_cache.keys())

    assert old_keys == new_keys, f"Cache keys differ.\nOld: {old_keys}\nNew: {new_keys}"

    for key in old_keys:
        old_val = old_cache[key]
        new_val = new_cache[key]
        assert_close(
            old_val,
            new_val,
            atol=1e-5,
            msg=f"Cache values for key '{key}' differ.",
        )


#####################################
# Training Equivalence
#####################################


def make_old_jumprelu_training_sae(
    d_in: int = 16, d_sae: int = 8
) -> OldTrainingSAE:  # Added types
    """
    Helper to instantiate an old TrainingSAE configured as JumpReLU for testing.
    """
    old_training_cfg = OldTrainingSAEConfig(  # Use OldTrainingSAEConfig
        architecture="jumprelu",
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
        context_size=128,  # Added default value
        dataset_path="fake/path",  # Added default value
        dataset_trust_remote_code=False,
        sae_lens_training_version="test_version",
        model_from_pretrained_kwargs={},
        seqpos_slice=(None,),  # Added default value (tuple as in old config)
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
        init_encoder_as_decoder_transpose=True,
        scale_sparsity_penalty_by_decoder_norm=True,
        decoder_heuristic_init_norm=0.1,
    )
    return OldTrainingSAE(old_training_cfg)  # Use OldTrainingSAE


def make_new_jumprelu_training_sae(
    d_in: int = 16, d_sae: int = 8
) -> JumpReLUTrainingSAE:  # Added types
    """
    Helper to instantiate a new JumpReLUTrainingSAE instance.
    """
    new_training_cfg = JumpReLUTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        apply_b_dec_to_input=False,
        normalize_activations="none",
        jumprelu_init_threshold=0.0,
        jumprelu_bandwidth=1.0,
        l0_coefficient=0.01,
        l0_warm_up_steps=0,
    )
    return JumpReLUTrainingSAE(new_training_cfg)


def test_jumprelu_training_equivalence():  # type: ignore # Kept ignore as return type is complex
    """
    Test that old vs new JumpReLU SAEs match shapes in outputs and remain finite.
    We won't require exact numeric equivalence, as the old code might differ in how it
    handles threshold or error terms. But we'll check overall shape and finiteness.
    """
    old_sae = make_old_jumprelu_training_sae()
    new_sae = make_new_jumprelu_training_sae()

    # Ensure parameters are identical before comparing outputs
    with torch.no_grad():
        old_params = dict(old_sae.named_parameters())
        new_params = dict(new_sae.named_parameters())
        for k in sorted(old_params.keys()):
            new_params[k].copy_(old_params[k])

    old_sae.train()
    new_sae.train()

    # Compare parameters post-alignment
    compare_params(old_sae, new_sae)

    batch_size, seq_len, d_in = 2, 4, 16
    x = torch.randn(batch_size, seq_len, d_in, dtype=torch.float32)

    old_out = old_sae.training_forward_pass(
        sae_in=x,
        current_l1_coefficient=old_sae.cfg.l1_coefficient,
        dead_neuron_mask=None,
    )
    new_out = new_sae.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=x,
            coefficients={"l0": new_sae.cfg.l0_coefficient},
            dead_neuron_mask=None,
            n_training_steps=0,
            is_logging_step=False,
        )
    )

    assert (
        old_out.sae_out.shape == new_out.sae_out.shape
    ), "JumpReLU training output shape mismatch."
    assert torch.isfinite(
        old_out.sae_out
    ).all(), "Old JumpReLU training out is not finite."
    assert torch.isfinite(
        new_out.sae_out
    ).all(), "New JumpReLU training out is not finite."

    # Check if training forward pass is equivalent
    assert_close(old_out.sae_out, new_out.sae_out, atol=1e-5)

    # Check that we do have MSE and L0 losses
    assert "mse_loss" in old_out.losses
    assert "mse_loss" in new_out.losses
    old_l0_loss = old_out.losses["l0_loss"]
    new_l0_loss = new_out.losses["l0_loss"]

    assert isinstance(old_l0_loss, torch.Tensor)
    assert torch.isfinite(old_l0_loss).all()  # Check tensor and use .all()
    assert torch.isfinite(new_l0_loss).all()  # Check tensor and use .all()

    # Compare individual loss components numerically
    assert_close(
        old_out.losses["mse_loss"],  # type: ignore
        new_out.losses["mse_loss"],
        atol=1e-5,
    )
    assert_close(old_l0_loss, new_l0_loss, atol=1e-5)
    assert_close(old_out.loss, new_out.loss, atol=1e-5)
