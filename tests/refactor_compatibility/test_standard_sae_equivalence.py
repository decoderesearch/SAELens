import einops
import pytest
import torch

# New modules
from sae_lens.saes.sae import SAEMetadata, TrainStepInput
from sae_lens.saes.standard_sae import (
    StandardSAE,
    StandardSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
)

# Old modules
from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name
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


def make_old_sae(
    d_in: int = 16,
    d_sae: int = 8,
    use_error_term: bool = False,
    hook_name: str = "blocks.0.hook_resid_pre",
    apply_b_dec_to_input: bool = False,
) -> OldSAE:
    """
    Helper to instantiate an old SAE instance for testing.
    Note: use a hook_name that does NOT end with '_z', so that we avoid
    the old code's auto-flattening logic that changes shapes unexpectedly.
    """
    old_cfg = OldSAEConfig(
        architecture="standard",
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name=hook_name,
        hook_layer=0,
        hook_head_index=None,
        activation_fn_str="relu",  # Use activation_fn_str
        activation_fn_kwargs={},
        apply_b_dec_to_input=apply_b_dec_to_input,
        finetuning_scaling_factor=False,
        normalize_activations="none",
        context_size=128,  # Add default
        dataset_path="fake/path",  # Add default
        dataset_trust_remote_code=False,
        sae_lens_training_version="test_version",
        model_from_pretrained_kwargs={},
        seqpos_slice=(None,),  # Use (None,)
        prepend_bos=False,
    )
    old_sae = OldSAE(old_cfg)
    old_sae.use_error_term = use_error_term
    return old_sae


def make_new_sae(
    d_in: int = 16,
    d_sae: int = 8,
    use_error_term: bool = False,
    hook_name: str = "blocks.0.hook_resid_pre",
    apply_b_dec_to_input: bool = False,
) -> StandardSAE:
    """
    Helper to instantiate a new StandardSAE instance for testing.
    Mirror the same hook_name that does NOT end with '_z'.
    """
    new_cfg = StandardSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        apply_b_dec_to_input=apply_b_dec_to_input,
        normalize_activations="none",
        metadata=SAEMetadata(
            model_name="test_model",
            hook_name=hook_name,
        ),
    )
    return StandardSAE(new_cfg, use_error_term=use_error_term)


def compare_params(
    old_sae: OldSAE | OldTrainingSAE, new_sae: StandardSAE | StandardTrainingSAE
):
    """
    Compare parameter names and shapes between the old SAE and the new StandardSAE.
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


def test_standard_sae_inference_equivalence():
    """
    Extended test of old vs new SAE inference equivalence across multiple architectures:
      - compare param shape
      - compare forward pass output shapes & ensure finite
      - optionally test error_term usage
    """

    old_sae = make_old_sae(d_in=16, d_sae=8, use_error_term=False)
    new_sae = make_new_sae(d_in=16, d_sae=8, use_error_term=False)
    compare_params(old_sae, new_sae)

    # Provide a random input
    x = torch.randn(2, 4, 16, dtype=torch.float32)

    with torch.no_grad():
        old_out = old_sae(x)
        new_out = new_sae(x)

    assert old_out.shape == new_out.shape, "Output shape mismatch."
    assert torch.isfinite(old_out).all()
    assert torch.isfinite(new_out).all()

    # Now test error_term usage
    old_sae_error = make_old_sae(d_in=16, d_sae=8, use_error_term=True)
    new_sae_error = make_new_sae(d_in=16, d_sae=8, use_error_term=True)

    with torch.no_grad():
        old_err_out = old_sae_error(x)
        new_err_out = new_sae_error(x)

    assert old_err_out.shape == new_err_out.shape
    # standard architecture can match exactly
    assert_close(
        old_err_out,
        new_err_out,
        atol=1e-5,
        msg="Mismatch in old/new output with error term (standard arch)",
    )


@pytest.mark.parametrize(
    "fold_fn", ["fold_W_dec_norm", "fold_activation_norm_scaling_factor"]
)
def test_standard_sae_fold_equivalence(fold_fn: str):
    """
    Test that calling fold functions (like fold_W_dec_norm or fold_activation_norm_scaling_factor)
    on old vs new yields consistent results on forward passes.
    """
    old_sae = make_old_sae(use_error_term=False)
    new_sae = make_new_sae(use_error_term=False)
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
        # pick a random scaling factor
        scale_factor = 2.0
        old_sae.fold_activation_norm_scaling_factor(scale_factor)
        new_sae.fold_activation_norm_scaling_factor(scale_factor)

    # Compare parameters post-folding
    for k in sorted(old_params.keys()):
        assert_close(
            old_params[k],
            new_params[k],
            atol=1e-5,
            msg=f"Parameter {k} differs after {fold_fn}",
        )

    # Provide input, compare outputs
    x = torch.randn(2, 3, 16, dtype=torch.float32)
    old_out = old_sae(x)
    new_out = new_sae(x)
    assert old_out.shape == new_out.shape
    assert_close(
        old_out,
        new_out,
        atol=1e-5,
        msg=f"{fold_fn} mismatch between old and new",
    )


def test_standard_sae_run_hooks_equivalence():
    """
    Compare hooking behavior (similar to test_hooked_sae).
    We'll check that hooking triggers the same number of calls in old_sae vs new_sae
    and that output shapes match.
    """
    old_sae = make_old_sae()
    new_sae = make_new_sae()

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


@pytest.mark.parametrize(
    "hook_name",
    [
        "blocks.0.hook_resid_pre",  # standard case
        "blocks.0.attn.hook_z",  # hook_z case
    ],
)
def test_standard_sae_hook_z_equivalence(hook_name: str):
    """
    Test that both old and new SAEs handle hook_z cases correctly.
    For hook_z, inputs should be reshaped from (..., n_heads, d_head) to (..., n_heads*d_head).
    """
    # For hook_z case, d_in should be n_heads * d_head
    n_heads = 8
    d_head = 8
    d_in = n_heads * d_head  # Always use flattened dimension for SAE config

    def make_old_sae_with_hook(hook_name: str, d_in: int) -> OldSAE:
        cfg = OldSAEConfig(
            architecture="standard",  # Explicitly set Literal
            d_in=d_in,  # Always use flattened dimension
            d_sae=32,
            dtype="float32",
            device="cpu",
            model_name="test_model",
            hook_name=hook_name,
            hook_layer=0,
            hook_head_index=None,
            activation_fn_str="relu",
            activation_fn_kwargs={},
            apply_b_dec_to_input=False,  # Important: set to False to avoid shape issues
            finetuning_scaling_factor=False,
            normalize_activations="none",
            context_size=128,  # Add default
            dataset_path="fake/path",  # Add default
            dataset_trust_remote_code=False,
            sae_lens_training_version="test_version",
            model_from_pretrained_kwargs={},
            seqpos_slice=(None,),  # Use None
            prepend_bos=False,
        )
        return OldSAE(cfg)

    def make_new_sae_with_hook(hook_name: str, d_in: int) -> StandardSAE:
        cfg = StandardSAEConfig(
            d_in=d_in,  # Always use flattened dimension
            d_sae=32,
            dtype="float32",
            device="cpu",
            apply_b_dec_to_input=False,  # Important: set to False to avoid shape issues
            normalize_activations="none",
            metadata=SAEMetadata(
                model_name="test_model",
                hook_name=hook_name,
            ),
        )
        return StandardSAE(cfg)

    old_sae = make_old_sae_with_hook(hook_name, d_in)
    new_sae = make_new_sae_with_hook(hook_name, d_in)

    # Create input with appropriate shape
    batch_size = 2
    seq_len = 4
    if hook_name.endswith("_z"):
        # Input shape for attention hook: (batch, seq, heads, d_head)
        x = torch.randn(batch_size, seq_len, n_heads, d_head)

        # New SAE needs explicit hook_z handling turned on
        new_sae.turn_on_forward_pass_hook_z_reshaping()
    else:
        # Standard shape for other hooks: (batch, seq, d_in)
        x = torch.randn(batch_size, seq_len, d_in)

    # Test forward pass shapes
    with torch.no_grad():
        old_out = old_sae(x)
        new_out = new_sae(x)

    assert (
        old_out.shape == new_out.shape == x.shape
    ), f"Output shape mismatch for {hook_name}. Old: {old_out.shape}, New: {new_out.shape}, Input: {x.shape}"

    # For hook_z case, verify the internal reshape happened
    if hook_name.endswith("_z"):
        # Run with caching to check internal shapes
        old_out, old_cache = old_sae.run_with_cache(x)
        new_out, new_cache = new_sae.run_with_cache(x)

        # Both should have flattened internal activations
        assert (
            old_cache["hook_sae_input"].shape[-1] == n_heads * d_head
        ), "Old SAE didn't flatten hook_z input correctly"
        assert (
            new_cache["hook_sae_input"].shape[-1] == n_heads * d_head
        ), "New SAE didn't flatten hook_z input correctly"

        # Verify encoder output shape is correct
        assert (
            old_cache["hook_sae_acts_pre"].shape[-1] == old_sae.cfg.d_sae
        ), "Old SAE encoder output shape incorrect"
        assert (
            new_cache["hook_sae_acts_pre"].shape[-1] == new_sae.cfg.d_sae
        ), "New SAE encoder output shape incorrect"

    # Clean up hook_z mode if it was enabled
    if hook_name.endswith("_z"):
        new_sae.turn_off_forward_pass_hook_z_reshaping()


@pytest.mark.parametrize(
    "hook_name",
    [
        "blocks.0.hook_resid_pre",  # standard case
        "blocks.0.attn.hook_z",  # hook_z case
    ],
)
def test_standard_sae_training_hook_z_equivalence(hook_name: str):
    """
    Test that training works correctly with hook_z reshaping.
    """
    # For hook_z case, d_in should be n_heads * d_head
    n_heads = 8
    d_head = 8
    d_in = n_heads * d_head  # Always use flattened dimension

    old_training_cfg = OldTrainingSAEConfig(
        architecture="standard",  # Explicitly set Literal
        d_in=d_in,  # Always use flattened dimension
        d_sae=32,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name=hook_name,
        hook_layer=extract_stop_at_layer_from_tlens_hook_name(hook_name) or 0,
        hook_head_index=None,
        activation_fn_str="relu",
        activation_fn_kwargs={},
        apply_b_dec_to_input=False,  # Important: set to False
        finetuning_scaling_factor=False,
        normalize_activations="none",
        context_size=128,  # Add default
        dataset_path="fake/path",  # Add default
        dataset_trust_remote_code=False,
        sae_lens_training_version="test_version",
        model_from_pretrained_kwargs={},
        seqpos_slice=(None,),  # Use None
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
        init_encoder_as_decoder_transpose=True,
        scale_sparsity_penalty_by_decoder_norm=True,
        decoder_heuristic_init_norm=0.1,
    )

    old_training_sae = OldTrainingSAE(old_training_cfg)
    # Remove old config specific args if any
    new_training_cfg = StandardTrainingSAEConfig(
        d_in=d_in,
        d_sae=32,
        l1_coefficient=0.01,
    )
    new_training_sae = StandardTrainingSAE(new_training_cfg)

    # Create appropriate input shape and prepare data for each implementation
    batch_size = 2
    seq_len = 4
    if hook_name.endswith("_z"):
        # For new SAE: Keep original shape, it will handle reshaping internally
        new_input_data = torch.randn(batch_size, seq_len, n_heads, d_head)
        # For old SAE: Pre-flatten the input as it expects
        old_input_data = einops.rearrange(
            new_input_data.clone(), "... n_heads d_head -> ... (n_heads d_head)"
        )

        # Turn on hook_z reshaping for new SAE only (old one expects pre-flattened input)
        new_training_sae.turn_on_forward_pass_hook_z_reshaping()
    else:
        # For non-hook_z case, both use the same shape
        new_input_data = torch.randn(batch_size, seq_len, d_in)
        old_input_data = new_input_data.clone()

    old_training_sae.train()
    new_training_sae.train()

    # Forward pass with appropriate input for each
    old_out = old_training_sae.training_forward_pass(
        sae_in=old_input_data,
        current_l1_coefficient=old_training_cfg.l1_coefficient,
        dead_neuron_mask=None,
    )
    new_out = new_training_sae.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=new_input_data,
            coefficients={"l1": old_training_cfg.l1_coefficient},
            dead_neuron_mask=None,
            n_training_steps=0,
            is_logging_step=False,
        )
    )

    # Check shapes match input
    if hook_name.endswith("_z"):
        assert old_out.sae_out.shape == old_input_data.shape
        assert new_out.sae_out.shape == new_input_data.shape
    else:
        assert old_out.sae_out.shape == new_out.sae_out.shape == new_input_data.shape

    # Check losses exist and are finite
    assert "mse_loss" in old_out.losses and "mse_loss" in new_out.losses
    assert torch.isfinite(old_out.losses["mse_loss"]).all()  # type: ignore
    assert torch.isfinite(new_out.losses["mse_loss"]).all()

    # Check sparsity loss (name differs between implementations)
    old_sparsity_loss = old_out.losses.get("aux_loss", old_out.losses.get("l1_loss"))
    new_sparsity_loss = new_out.losses.get("aux_loss", new_out.losses.get("l1_loss"))
    assert old_sparsity_loss is not None
    assert new_sparsity_loss is not None
    assert torch.isfinite(old_sparsity_loss).all()  # type: ignore
    assert torch.isfinite(new_sparsity_loss).all()

    # For hook_z case, verify the internal shapes
    if hook_name.endswith("_z"):
        # Run with caching to check internal shapes
        old_out, old_cache = old_training_sae.run_with_cache(old_input_data)
        new_out, new_cache = new_training_sae.run_with_cache(new_input_data)

        # Check input shapes
        assert (
            old_cache["hook_sae_input"].shape[-1] == n_heads * d_head
        ), "Old SAE input shape incorrect"
        assert (
            new_cache["hook_sae_input"].shape[-1] == n_heads * d_head
        ), "New SAE input shape incorrect"

        # Verify encoder output shape is correct
        assert (
            old_cache["hook_sae_acts_pre"].shape[-1] == old_training_sae.cfg.d_sae
        ), "Old SAE encoder output shape incorrect"
        assert (
            new_cache["hook_sae_acts_pre"].shape[-1] == new_training_sae.cfg.d_sae
        ), "New SAE encoder output shape incorrect"

    if hook_name.endswith("_z"):
        new_training_sae.turn_off_forward_pass_hook_z_reshaping()


def test_standard_sae_forward_equivalence():
    """
    Test standard forward pass equivalence (non-hook_z).
    """
    d_in = 16
    d_sae = 32
    hook_name = "blocks.0.hook_resid_pre"

    old_sae = make_old_sae(
        d_in=d_in, d_sae=d_sae, hook_name=hook_name, apply_b_dec_to_input=False
    )
    new_sae = make_new_sae(
        d_in=d_in, d_sae=d_sae, hook_name=hook_name, apply_b_dec_to_input=False
    )

    # Align parameters
    with torch.no_grad():
        old_params = dict(old_sae.named_parameters())
        new_params = dict(new_sae.named_parameters())
        for k in sorted(old_params.keys()):
            new_params[k].copy_(old_params[k])
    compare_params(old_sae, new_sae)

    # Standard input shape
    batch_size = 2
    seq_len = 4
    x = torch.randn(batch_size, seq_len, d_in)

    # Test forward pass
    with torch.no_grad():
        old_out = old_sae(x)
        new_out = new_sae(x)

    assert (
        old_out.shape == x.shape
    ), f"Old output shape mismatch. Got {old_out.shape}, expected {x.shape}"
    assert (
        new_out.shape == x.shape
    ), f"New output shape mismatch. Got {new_out.shape}, expected {x.shape}"
    assert_close(
        old_out,
        new_out,
        atol=1e-5,
        msg="Standard forward outputs differ numerically.",
    )


def test_sae_hook_z_forward_equivalence():
    """
    Test forward pass equivalence for the hook_z case, including internal reshaping checks.
    """
    n_heads = 8
    d_head = 8
    d_in = n_heads * d_head  # Flattened dimension for SAE config
    d_sae = 32
    hook_name = "blocks.0.attn.hook_z"

    old_sae = make_old_sae(
        d_in=d_in, d_sae=d_sae, hook_name=hook_name, apply_b_dec_to_input=False
    )
    new_sae = make_new_sae(
        d_in=d_in, d_sae=d_sae, hook_name=hook_name, apply_b_dec_to_input=False
    )

    # Align parameters
    with torch.no_grad():
        old_params = dict(old_sae.named_parameters())
        new_params = dict(new_sae.named_parameters())
        for k in sorted(old_params.keys()):
            new_params[k].copy_(old_params[k])
    compare_params(old_sae, new_sae)

    # Input shape for attention hook: (batch, seq, heads, d_head)
    batch_size = 2
    seq_len = 4
    x = torch.randn(batch_size, seq_len, n_heads, d_head)

    # Turn on hook_z reshaping for the new SAE
    new_sae.turn_on_forward_pass_hook_z_reshaping()

    try:
        # Test forward pass
        with torch.no_grad():
            old_out = old_sae(
                x
            )  # Old SAE expects flattened input handled internally by HookedTransformer
            new_out = new_sae(
                x
            )  # New SAE handles reshaping internally when hook_z is on

        assert (
            old_out.shape == x.shape
        ), f"Old hook_z output shape mismatch. Got {old_out.shape}, expected {x.shape}"
        assert (
            new_out.shape == x.shape
        ), f"New hook_z output shape mismatch. Got {new_out.shape}, expected {x.shape}"
        assert_close(
            old_out,
            new_out,
            atol=1e-5,
            msg="Hook_z forward outputs differ numerically.",
        )

        # Verify internal shapes using run_with_cache
        with torch.no_grad():
            _, old_cache = old_sae.run_with_cache(x)
            _, new_cache = new_sae.run_with_cache(x)

        # Both should have flattened internal activations input to the SAE linear layers
        assert (
            old_cache["hook_sae_input"].shape[-1] == n_heads * d_head
        ), "Old SAE didn't flatten hook_z input correctly"
        assert (
            new_cache["hook_sae_input"].shape[-1] == n_heads * d_head
        ), "New SAE didn't flatten hook_z input correctly"
        # Verify encoder output shape is correct
        assert (
            old_cache["hook_sae_acts_pre"].shape[-1] == old_sae.cfg.d_sae
        ), "Old SAE encoder output shape incorrect"
        assert (
            new_cache["hook_sae_acts_pre"].shape[-1] == new_sae.cfg.d_sae
        ), "New SAE encoder output shape incorrect"
        # Check decoder input has correct shape
        assert (
            old_cache["hook_sae_acts_post"].shape[-1] == old_sae.cfg.d_sae
        ), "Old SAE decoder input shape incorrect"
        assert (
            new_cache["hook_sae_acts_post"].shape[-1] == new_sae.cfg.d_sae
        ), "New SAE decoder input shape incorrect"

    finally:
        # Clean up hook_z mode
        new_sae.turn_off_forward_pass_hook_z_reshaping()


#####################################
# Training Equivalence Helpers
#####################################


def make_old_training_sae(
    d_in: int = 16,
    d_sae: int = 32,
    hook_name: str = "blocks.0.hook_resid_pre",
    l1_coefficient: float = 0.01,
    apply_b_dec_to_input: bool = False,
) -> OldTrainingSAE:
    """Helper to instantiate an old TrainingSAE instance."""
    old_training_cfg = OldTrainingSAEConfig(
        architecture="standard",
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name=hook_name,
        hook_layer=extract_stop_at_layer_from_tlens_hook_name(hook_name) or 0,
        hook_head_index=None,
        activation_fn_str="relu",
        activation_fn_kwargs={},
        apply_b_dec_to_input=apply_b_dec_to_input,
        finetuning_scaling_factor=False,
        normalize_activations="none",
        context_size=128,
        dataset_path="fake/path",
        dataset_trust_remote_code=False,
        sae_lens_training_version="test_version",
        model_from_pretrained_kwargs={},
        seqpos_slice=(None,),
        prepend_bos=False,
        l1_coefficient=l1_coefficient,
        lp_norm=1.0,
        use_ghost_grads=False,
        normalize_sae_decoder=False,
        noise_scale=0.0,
        decoder_orthogonal_init=False,
        mse_loss_normalization=None,
        jumprelu_init_threshold=0.0,  # Not used but part of config
        jumprelu_bandwidth=1.0,  # Not used but part of config
        decoder_heuristic_init=False,
        init_encoder_as_decoder_transpose=False,
        scale_sparsity_penalty_by_decoder_norm=True,
        decoder_heuristic_init_norm=0.1,
    )
    return OldTrainingSAE(old_training_cfg)


def make_new_training_sae(
    d_in: int = 16,
    d_sae: int = 32,
    hook_name: str = "blocks.0.hook_resid_pre",
    l1_coefficient: float = 0.01,
    apply_b_dec_to_input: bool = False,
) -> StandardTrainingSAE:
    """Helper to instantiate a new StandardTrainingSAE instance."""
    new_training_cfg = StandardTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        l1_coefficient=l1_coefficient,
        apply_b_dec_to_input=apply_b_dec_to_input,
        reshape_activations="hook_z" if "hook_z" in hook_name else "none",
        metadata=SAEMetadata(
            model_name="test_model",
            hook_name=hook_name,
            hook_head_index=None,
        ),
    )
    return StandardTrainingSAE(new_training_cfg)


#####################################
# Training Equivalence Tests
#####################################


def test_standard_sae_training_equivalence():
    """
    Test standard training pass equivalence (non-hook_z).
    """
    d_in = 16
    d_sae = 32
    hook_name = "blocks.0.hook_resid_pre"
    l1_coefficient = 0.01

    old_sae = make_old_training_sae(
        d_in=d_in,
        d_sae=d_sae,
        hook_name=hook_name,
        l1_coefficient=l1_coefficient,
        apply_b_dec_to_input=False,
    )
    new_sae = make_new_training_sae(
        d_in=d_in,
        d_sae=d_sae,
        hook_name=hook_name,
        l1_coefficient=l1_coefficient,
        apply_b_dec_to_input=False,
    )

    # Align parameters
    with torch.no_grad():
        old_params = dict(old_sae.named_parameters())
        new_params = dict(new_sae.named_parameters())
        for k in sorted(old_params.keys()):
            new_params[k].copy_(old_params[k])
    compare_params(old_sae, new_sae)

    # Standard input shape
    batch_size = 2
    seq_len = 4
    x = torch.randn(batch_size, seq_len, d_in)

    old_sae.train()
    new_sae.train()

    # Forward pass
    old_out = old_sae.training_forward_pass(
        sae_in=x,
        current_l1_coefficient=l1_coefficient,
        dead_neuron_mask=None,
    )
    new_out = new_sae.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=x,
            coefficients={"l1": l1_coefficient},
            dead_neuron_mask=None,
            n_training_steps=0,
            is_logging_step=False,
        )
    )

    # Check shapes
    assert old_out.sae_out.shape == x.shape, "Old output shape mismatch"
    assert new_out.sae_out.shape == x.shape, "New output shape mismatch"

    # Check numerical equivalence
    assert_close(
        old_out.sae_out,
        new_out.sae_out,
        atol=1e-5,
        msg="SAE output differs",
    )
    assert_close(
        old_out.loss,
        new_out.loss,
        atol=1e-5,
        msg="Total loss differs",
    )

    # Check loss components
    old_mse = old_out.losses["mse_loss"]
    new_mse = new_out.losses["mse_loss"]
    old_sparsity = old_out.losses.get("aux_loss", old_out.losses.get("l1_loss"))
    new_sparsity = new_out.losses.get("aux_loss", new_out.losses.get("l1_loss"))

    assert old_sparsity is not None and new_sparsity is not None
    assert isinstance(old_mse, torch.Tensor) and torch.isfinite(old_mse).all()
    assert isinstance(new_mse, torch.Tensor) and torch.isfinite(new_mse).all()
    assert isinstance(old_sparsity, torch.Tensor) and torch.isfinite(old_sparsity).all()
    assert isinstance(new_sparsity, torch.Tensor) and torch.isfinite(new_sparsity).all()

    assert_close(
        old_mse,
        new_mse,
        atol=1e-5,
        msg="MSE loss differs",
    )
    assert_close(
        old_sparsity,
        new_sparsity,
        atol=1e-5,
        msg="Sparsity loss differs",
    )


def test_sae_hook_z_training_equivalence():
    """
    Test training pass equivalence for the hook_z case.
    """
    n_heads = 8
    d_head = 8
    d_in = n_heads * d_head
    d_sae = 32
    hook_name = "blocks.0.attn.hook_z"
    l1_coefficient = 0.01

    old_sae = make_old_training_sae(
        d_in=d_in,
        d_sae=d_sae,
        hook_name=hook_name,
        l1_coefficient=l1_coefficient,
        apply_b_dec_to_input=False,
    )
    new_sae = make_new_training_sae(
        d_in=d_in,
        d_sae=d_sae,
        hook_name=hook_name,
        l1_coefficient=l1_coefficient,
        apply_b_dec_to_input=False,
    )

    # Align parameters
    with torch.no_grad():
        old_params = dict(old_sae.named_parameters())
        new_params = dict(new_sae.named_parameters())
        for k in sorted(old_params.keys()):
            new_params[k].copy_(old_params[k])
    compare_params(old_sae, new_sae)

    # Input shapes
    batch_size = 2
    seq_len = 4
    # New SAE takes original shape, old SAE takes flattened
    x_raw = torch.randn(batch_size, seq_len, n_heads, d_head)
    x_reshaped = einops.rearrange(x_raw.clone(), "... h d -> ... (h d)")

    old_sae.train()
    new_sae.train()

    # Forward pass
    old_out = old_sae.training_forward_pass(
        sae_in=x_reshaped,
        current_l1_coefficient=l1_coefficient,
        dead_neuron_mask=None,
    )
    new_out = new_sae.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=x_reshaped,
            coefficients={"l1": l1_coefficient},
            dead_neuron_mask=None,
            n_training_steps=0,
            is_logging_step=False,
        )
    )

    # Check shapes (output matches input shape for each)
    assert old_out.sae_out.shape == x_reshaped.shape, "Old hook_z output shape mismatch"
    assert new_out.sae_out.shape == x_reshaped.shape, "New hook_z output shape mismatch"

    # Check numerical equivalence (reshape old output to match new)
    assert_close(
        old_out.sae_out,
        new_out.sae_out,
        atol=1e-5,
        msg="Hook_z SAE output differs",
    )
    assert_close(
        old_out.loss,
        new_out.loss,
        atol=1e-5,
        msg="Hook_z total loss differs",
    )

    # Check loss components
    old_mse = old_out.losses["mse_loss"]
    new_mse = new_out.losses["mse_loss"]
    old_sparsity = old_out.losses.get("aux_loss", old_out.losses.get("l1_loss"))
    new_sparsity = new_out.losses.get("aux_loss", new_out.losses.get("l1_loss"))

    assert old_sparsity is not None and new_sparsity is not None
    assert isinstance(old_mse, torch.Tensor) and torch.isfinite(old_mse).all()
    assert isinstance(new_mse, torch.Tensor) and torch.isfinite(new_mse).all()
    assert isinstance(old_sparsity, torch.Tensor) and torch.isfinite(old_sparsity).all()
    assert isinstance(new_sparsity, torch.Tensor) and torch.isfinite(new_sparsity).all()

    assert_close(
        old_mse,
        new_mse,
        atol=1e-5,
        msg="Hook_z Sparsity loss differs",
    )
    assert_close(
        old_sparsity,
        new_sparsity,
        atol=1e-5,
        msg="Hook_z Sparsity loss differs",
    )

    with torch.no_grad():
        _, old_cache = old_sae.run_with_cache(x_reshaped)
        _, new_cache = new_sae.run_with_cache(x_reshaped)

    assert (
        old_cache["hook_sae_input"].shape[-1] == n_heads * d_head
    ), "Old SAE cache input shape incorrect"
    assert (
        new_cache["hook_sae_input"].shape[-1] == n_heads * d_head
    ), "New SAE cache input shape incorrect"
    assert (
        old_cache["hook_sae_acts_pre"].shape[-1] == old_sae.cfg.d_sae
    ), "Old SAE cache acts_pre shape incorrect"
    assert (
        new_cache["hook_sae_acts_pre"].shape[-1] == new_sae.cfg.d_sae
    ), "New SAE cache acts_pre shape incorrect"
