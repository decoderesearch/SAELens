# type: ignore
"""Comparison tests to verify new HookedSAETransformer matches old implementation.

These tests compare the current implementation with the old implementation in
tests/_comparison/sae_lens_old to ensure backwards compatibility.
"""

import pytest
from transformer_lens import HookedTransformer

from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from sae_lens.saes.sae import SAEMetadata
from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
from tests._comparison.sae_lens_old.analysis.hooked_sae_transformer import (
    HookedSAETransformer as OldHookedSAETransformer,
)
from tests._comparison.sae_lens_old.sae import SAE as OldSAE
from tests.helpers import TINYSTORIES_MODEL, assert_close

MODEL = TINYSTORIES_MODEL
PROMPT = "Hello World!"


def make_sae(model: HookedTransformer, act_name: str) -> StandardSAE:
    site_to_size = {
        "hook_z": model.cfg.d_head * model.cfg.n_heads,
        "hook_mlp_out": model.cfg.d_model,
        "hook_resid_pre": model.cfg.d_model,
        "hook_post": model.cfg.d_mlp,
    }
    site = act_name.split(".")[-1]
    d_in = site_to_size[site]

    sae_cfg = StandardSAEConfig(
        d_in=d_in,
        d_sae=d_in * 2,
        dtype="float32",
        device="cpu",
        reshape_activations="hook_z" if act_name.endswith("hook_z") else "none",
        metadata=SAEMetadata(
            model_name=MODEL,
            hook_name=act_name,
            hook_head_index=None,
            prepend_bos=True,
        ),
    )
    return StandardSAE(sae_cfg)


def make_old_sae(model: HookedTransformer, act_name: str) -> OldSAE:
    site_to_size = {
        "hook_z": model.cfg.d_head * model.cfg.n_heads,
        "hook_mlp_out": model.cfg.d_model,
        "hook_resid_pre": model.cfg.d_model,
        "hook_post": model.cfg.d_mlp,
    }
    site = act_name.split(".")[-1]
    d_in = site_to_size[site]

    sae_cfg = {
        "d_in": d_in,
        "d_sae": d_in * 2,
        "dtype": "float32",
        "device": "cpu",
        "hook_name": act_name,
        "hook_layer": 0,
        "hook_head_index": None,
        "model_name": MODEL,
        "prepend_bos": True,
        "activation_fn_str": "relu",
        "apply_b_dec_to_input": True,
        "normalize_activations": "none",
        "context_size": 128,
        "finetuning_scaling_factor": False,
        "architecture": "standard",
        "dataset_path": "test",
        "dataset_trust_remote_code": True,
        "sae_lens_training_version": "test",
    }
    return OldSAE.from_dict(sae_cfg)


def sync_weights(new_sae: StandardSAE, old_sae: OldSAE):
    """Copy weights from new SAE to old SAE to ensure identical behavior."""
    old_sae.W_enc.data = new_sae.W_enc.data.clone()
    old_sae.W_dec.data = new_sae.W_dec.data.clone()
    old_sae.b_enc.data = new_sae.b_enc.data.clone()
    old_sae.b_dec.data = new_sae.b_dec.data.clone()


@pytest.fixture(scope="module")
def new_model():
    model = HookedSAETransformer.from_pretrained(MODEL, device="cpu")
    yield model
    model.reset_saes()


@pytest.fixture(scope="module")
def old_model():
    model = OldHookedSAETransformer.from_pretrained(MODEL, device="cpu")
    yield model
    model.reset_saes()


def test_sae_behavior_matches_old_no_error_term(
    new_model: HookedSAETransformer, old_model: OldHookedSAETransformer
):
    act_name = "blocks.0.hook_mlp_out"
    new_sae = make_sae(new_model, act_name)
    old_sae = make_old_sae(old_model, act_name)
    sync_weights(new_sae, old_sae)

    new_model.add_sae(new_sae, use_error_term=False)
    old_model.add_sae(old_sae, use_error_term=False)

    new_output = new_model(PROMPT)
    old_output = old_model(PROMPT)

    assert_close(new_output, old_output, atol=1e-5)

    new_model.reset_saes()
    old_model.reset_saes()


def test_sae_behavior_matches_old_with_error_term(
    new_model: HookedSAETransformer, old_model: OldHookedSAETransformer
):
    act_name = "blocks.0.hook_mlp_out"
    new_sae = make_sae(new_model, act_name)
    old_sae = make_old_sae(old_model, act_name)
    sync_weights(new_sae, old_sae)

    new_model.add_sae(new_sae, use_error_term=True)
    old_model.add_sae(old_sae, use_error_term=True)

    new_output = new_model(PROMPT)
    old_output = old_model(PROMPT)

    assert_close(new_output, old_output, atol=1e-4)

    new_model.reset_saes()
    old_model.reset_saes()


def test_gradients_match_old_no_error_term(
    new_model: HookedSAETransformer, old_model: OldHookedSAETransformer
):
    act_name = "blocks.0.hook_mlp_out"
    new_sae = make_sae(new_model, act_name)
    old_sae = make_old_sae(old_model, act_name)
    sync_weights(new_sae, old_sae)

    # Forward + backward with new model
    new_model.add_sae(new_sae, use_error_term=False)
    new_output = new_model(PROMPT)
    new_output.sum().backward()
    new_grads = {
        name: p.grad.clone()
        for name, p in new_sae.named_parameters()
        if p.grad is not None
    }
    new_sae.zero_grad()
    new_model.reset_saes()

    # Forward + backward with old model
    old_model.add_sae(old_sae, use_error_term=False)
    old_output = old_model(PROMPT)
    old_output.sum().backward()
    old_grads = {
        name: p.grad.clone()
        for name, p in old_sae.named_parameters()
        if p.grad is not None
    }
    old_sae.zero_grad()
    old_model.reset_saes()

    # Compare outputs
    assert_close(new_output, old_output, atol=1e-5)

    # Compare gradients
    for name in new_grads:
        if name in old_grads:
            assert_close(new_grads[name], old_grads[name], atol=1e-5)


def test_gradients_match_old_with_error_term(
    new_model: HookedSAETransformer, old_model: OldHookedSAETransformer
):
    act_name = "blocks.0.hook_mlp_out"
    new_sae = make_sae(new_model, act_name)
    old_sae = make_old_sae(old_model, act_name)
    sync_weights(new_sae, old_sae)

    # Forward + backward with new model
    new_model.add_sae(new_sae, use_error_term=True)
    new_output = new_model(PROMPT)
    new_output.sum().backward()
    new_grads = {
        name: p.grad.clone()
        for name, p in new_sae.named_parameters()
        if p.grad is not None
    }
    new_sae.zero_grad()
    new_model.reset_saes()

    # Forward + backward with old model
    old_model.add_sae(old_sae, use_error_term=True)
    old_output = old_model(PROMPT)
    old_output.sum().backward()
    old_grads = {
        name: p.grad.clone()
        for name, p in old_sae.named_parameters()
        if p.grad is not None
    }
    old_sae.zero_grad()
    old_model.reset_saes()

    # Compare outputs
    assert_close(new_output, old_output, atol=1e-4)

    # Compare gradients
    for name in new_grads:
        if name in old_grads:
            assert_close(new_grads[name], old_grads[name], atol=1e-4)


@pytest.mark.parametrize("use_error_term", [False, True])
def test_run_with_cache_matches_old(
    new_model: HookedSAETransformer,
    old_model: OldHookedSAETransformer,
    use_error_term: bool,
):
    act_name = "blocks.0.hook_mlp_out"
    new_sae = make_sae(new_model, act_name)
    old_sae = make_old_sae(old_model, act_name)
    sync_weights(new_sae, old_sae)

    new_output, new_cache = new_model.run_with_cache_with_saes(
        PROMPT, saes=[new_sae], use_error_term=use_error_term
    )
    old_output, old_cache = old_model.run_with_cache_with_saes(
        PROMPT, saes=[old_sae], use_error_term=use_error_term
    )

    assert_close(new_output, old_output, atol=1e-5)

    # Compare SAE activations in cache
    new_acts = new_cache[act_name + ".hook_sae_acts_post"]
    old_acts = old_cache[act_name + ".hook_sae_acts_post"]
    assert_close(new_acts, old_acts, atol=1e-5)


def test_run_with_saes_matches_old(
    new_model: HookedSAETransformer, old_model: OldHookedSAETransformer
):
    act_name = "blocks.0.hook_mlp_out"
    new_sae = make_sae(new_model, act_name)
    old_sae = make_old_sae(old_model, act_name)
    sync_weights(new_sae, old_sae)

    new_output = new_model.run_with_saes(PROMPT, saes=[new_sae], use_error_term=False)
    old_output = old_model.run_with_saes(PROMPT, saes=[old_sae], use_error_term=False)

    assert_close(new_output, old_output, atol=1e-5)


def test_context_manager_matches_old(
    new_model: HookedSAETransformer, old_model: OldHookedSAETransformer
):
    act_name = "blocks.0.hook_mlp_out"
    new_sae = make_sae(new_model, act_name)
    old_sae = make_old_sae(old_model, act_name)
    sync_weights(new_sae, old_sae)

    with new_model.saes(saes=[new_sae], use_error_term=False):
        new_output = new_model(PROMPT)

    with old_model.saes(saes=[old_sae], use_error_term=False):
        old_output = old_model(PROMPT)

    assert_close(new_output, old_output, atol=1e-5)


def test_multiple_hook_points_match_old(
    new_model: HookedSAETransformer, old_model: OldHookedSAETransformer
):
    act_names = ["blocks.0.hook_mlp_out", "blocks.0.hook_resid_pre"]

    new_saes = [make_sae(new_model, act_name) for act_name in act_names]
    old_saes = [make_old_sae(old_model, act_name) for act_name in act_names]

    for new_sae, old_sae in zip(new_saes, old_saes):
        sync_weights(new_sae, old_sae)

    for new_sae in new_saes:
        new_model.add_sae(new_sae, use_error_term=False)
    for old_sae in old_saes:
        old_model.add_sae(old_sae, use_error_term=False)

    new_output = new_model(PROMPT)
    old_output = old_model(PROMPT)

    assert_close(new_output, old_output, atol=1e-5)

    new_model.reset_saes()
    old_model.reset_saes()


def test_sae_with_use_error_term_preset_matches_old(
    new_model: HookedSAETransformer, old_model: OldHookedSAETransformer
):
    """Verify that when SAE has use_error_term=True set, add_sae() respects it."""
    act_name = "blocks.0.hook_mlp_out"
    new_sae = make_sae(new_model, act_name)
    old_sae = make_old_sae(old_model, act_name)
    sync_weights(new_sae, old_sae)

    # Set use_error_term=True on SAEs before adding (without explicit arg to add_sae)
    new_sae._use_error_term = True  # Set directly to avoid deprecation warning
    old_sae.use_error_term = True

    # Add without specifying use_error_term - should respect SAE's setting
    new_model.add_sae(new_sae)
    old_model.add_sae(old_sae)

    new_output = new_model(PROMPT)
    old_output = old_model(PROMPT)

    assert_close(new_output, old_output, atol=1e-4)

    new_model.reset_saes()
    old_model.reset_saes()


def test_sae_with_use_error_term_preset_gradients_match_old(
    new_model: HookedSAETransformer, old_model: OldHookedSAETransformer
):
    """Verify gradients match when SAE has use_error_term=True set before add_sae()."""
    act_name = "blocks.0.hook_mlp_out"
    new_sae = make_sae(new_model, act_name)
    old_sae = make_old_sae(old_model, act_name)
    sync_weights(new_sae, old_sae)

    # Set use_error_term=True on SAEs before adding
    new_sae._use_error_term = True
    old_sae.use_error_term = True

    # Forward + backward with new model
    new_model.add_sae(new_sae)
    new_output = new_model(PROMPT)
    new_output.sum().backward()
    new_grads = {
        name: p.grad.clone()
        for name, p in new_sae.named_parameters()
        if p.grad is not None
    }
    new_sae.zero_grad()
    new_model.reset_saes()

    # Forward + backward with old model
    old_model.add_sae(old_sae)
    old_output = old_model(PROMPT)
    old_output.sum().backward()
    old_grads = {
        name: p.grad.clone()
        for name, p in old_sae.named_parameters()
        if p.grad is not None
    }
    old_sae.zero_grad()
    old_model.reset_saes()

    # Compare outputs
    assert_close(new_output, old_output, atol=1e-4)

    # Compare gradients
    for name in new_grads:
        if name in old_grads:
            assert_close(new_grads[name], old_grads[name], atol=1e-4)
