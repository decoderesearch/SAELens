"""Tests for SAETransformerBridge.

These tests verify that SAETransformerBridge behaves identically to HookedSAETransformer
for the functionality it supports.
"""

import pytest
import torch
from transformer_lens.hook_points import HookPoint

from sae_lens.analysis.hooked_sae_transformer import (
    HookedSAETransformer,
    _SAEWrapper,
    get_deep_attr,
)
from sae_lens.analysis.sae_transformer_bridge import SAETransformerBridge
from sae_lens.saes.sae import SAEMetadata
from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
from sae_lens.saes.transcoder import Transcoder, TranscoderConfig
from tests.helpers import TINYSTORIES_MODEL, assert_close, assert_not_close

MODEL = TINYSTORIES_MODEL
PROMPT = "Hello World!"


def make_sae(d_model: int, act_name: str) -> StandardSAE:
    sae_cfg = StandardSAEConfig(
        d_in=d_model,
        d_sae=d_model * 2,
        dtype="float32",
        device="cpu",
        metadata=SAEMetadata(
            model_name=MODEL,
            hook_name=act_name,
            hook_head_index=None,
            prepend_bos=True,
        ),
    )
    sae = StandardSAE(sae_cfg)
    sae.initialize_weights()  # Ensure weights are properly initialized
    return sae


@pytest.fixture(scope="module")
def hooked_model():
    # Use from_pretrained_no_processing to match TransformerBridge behavior
    model = HookedSAETransformer.from_pretrained_no_processing(MODEL, device="cpu")
    yield model
    model.reset_saes()


@pytest.fixture(scope="module")
def bridge_model():
    model = SAETransformerBridge.boot_transformers(MODEL, device="cpu")
    yield model
    model.reset_saes()


# =============================================================================
# Comparison tests: verify HookedSAETransformer and SAETransformerBridge match
# =============================================================================


def test_both_models_produce_same_logits_without_saes(
    hooked_model: HookedSAETransformer, bridge_model: SAETransformerBridge
) -> None:
    hooked_logits = hooked_model(PROMPT)
    bridge_logits = bridge_model(PROMPT)
    assert_close(hooked_logits, bridge_logits, atol=1e-4)


def test_both_models_produce_same_logits_with_sae(
    hooked_model: HookedSAETransformer, bridge_model: SAETransformerBridge
) -> None:
    sae = make_sae(hooked_model.cfg.d_model, "blocks.0.hook_mlp_out")

    hooked_model.add_sae(sae)
    hooked_logits = hooked_model(PROMPT)
    hooked_model.reset_saes()

    bridge_model.add_sae(sae)
    bridge_logits = bridge_model(PROMPT)
    bridge_model.reset_saes()

    assert_close(hooked_logits, bridge_logits, atol=1e-4)


def test_both_models_produce_same_logits_with_multiple_saes(
    hooked_model: HookedSAETransformer, bridge_model: SAETransformerBridge
) -> None:
    d_model = hooked_model.cfg.d_model
    saes = [
        make_sae(d_model, "blocks.0.hook_mlp_out"),
        make_sae(d_model, "blocks.0.hook_resid_pre"),
    ]

    for sae in saes:
        hooked_model.add_sae(sae)
    hooked_logits = hooked_model(PROMPT)
    hooked_model.reset_saes()

    for sae in saes:
        bridge_model.add_sae(sae)
    bridge_logits = bridge_model(PROMPT)
    bridge_model.reset_saes()

    assert_close(hooked_logits, bridge_logits, atol=1e-4)


def test_use_error_term_preserves_original_output(
    hooked_model: HookedSAETransformer, bridge_model: SAETransformerBridge
) -> None:
    sae = make_sae(hooked_model.cfg.d_model, "blocks.0.hook_mlp_out")

    hooked_original = hooked_model(PROMPT)
    bridge_original = bridge_model(PROMPT)

    hooked_model.add_sae(sae, use_error_term=True)
    hooked_with_sae = hooked_model(PROMPT)
    hooked_model.reset_saes()

    bridge_model.add_sae(sae, use_error_term=True)
    bridge_with_sae = bridge_model(PROMPT)
    bridge_model.reset_saes()

    # With use_error_term=True, output should match original
    assert_close(hooked_with_sae, hooked_original, atol=1e-4)
    assert_close(bridge_with_sae, bridge_original, atol=1e-4)
    assert_close(hooked_with_sae, bridge_with_sae, atol=1e-4)


def test_run_with_saes_matches(
    hooked_model: HookedSAETransformer, bridge_model: SAETransformerBridge
) -> None:
    sae = make_sae(hooked_model.cfg.d_model, "blocks.0.hook_mlp_out")

    hooked_logits = hooked_model.run_with_saes(PROMPT, saes=[sae])
    bridge_logits = bridge_model.run_with_saes(PROMPT, saes=[sae])

    assert isinstance(hooked_logits, torch.Tensor)
    assert isinstance(bridge_logits, torch.Tensor)
    assert_close(hooked_logits, bridge_logits, atol=1e-4)
    assert len(hooked_model._acts_to_saes) == 0
    assert len(bridge_model._acts_to_saes) == 0


def test_saes_context_manager_matches(
    hooked_model: HookedSAETransformer, bridge_model: SAETransformerBridge
) -> None:
    sae = make_sae(hooked_model.cfg.d_model, "blocks.0.hook_mlp_out")

    with hooked_model.saes(saes=[sae]):
        hooked_logits = hooked_model(PROMPT)

    with bridge_model.saes(saes=[sae]):
        bridge_logits = bridge_model(PROMPT)

    assert_close(hooked_logits, bridge_logits, atol=1e-4)
    assert len(hooked_model._acts_to_saes) == 0
    assert len(bridge_model._acts_to_saes) == 0


def test_run_with_cache_captures_sae_activations(
    hooked_model: HookedSAETransformer, bridge_model: SAETransformerBridge
) -> None:
    sae = make_sae(hooked_model.cfg.d_model, "blocks.0.hook_mlp_out")

    hooked_model.add_sae(sae)
    hooked_logits, hooked_cache = hooked_model.run_with_cache(PROMPT)
    hooked_model.reset_saes()

    bridge_model.add_sae(sae)
    bridge_logits, bridge_cache = bridge_model.run_with_cache(PROMPT)
    bridge_model.reset_saes()

    assert isinstance(hooked_logits, torch.Tensor)
    assert isinstance(bridge_logits, torch.Tensor)
    assert_close(hooked_logits, bridge_logits, atol=1e-4)

    # Check SAE activations are in cache (hook names differ due to alias resolution)
    assert "blocks.0.hook_mlp_out.hook_sae_acts_post" in hooked_cache
    assert "blocks.0.mlp.hook_out.hook_sae_acts_post" in bridge_cache

    hooked_sae_acts = hooked_cache["blocks.0.hook_mlp_out.hook_sae_acts_post"]
    bridge_sae_acts = bridge_cache["blocks.0.mlp.hook_out.hook_sae_acts_post"]
    assert_close(hooked_sae_acts, bridge_sae_acts, atol=1e-4)


# =============================================================================
# SAETransformerBridge-specific tests
# =============================================================================


def test_boot_transformers_loads_model(bridge_model: SAETransformerBridge) -> None:
    assert isinstance(bridge_model, SAETransformerBridge)
    assert hasattr(bridge_model, "_acts_to_saes")
    assert len(bridge_model._acts_to_saes) == 0


def test_add_sae_changes_output(bridge_model: SAETransformerBridge) -> None:
    original_logits = bridge_model(PROMPT)

    sae = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")
    bridge_model.add_sae(sae)
    logits_with_sae = bridge_model(PROMPT)
    bridge_model.reset_saes()

    assert_not_close(original_logits, logits_with_sae)


def test_add_sae_updates__acts_to_saes(bridge_model: SAETransformerBridge) -> None:
    sae = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")
    act_name = sae.cfg.metadata.hook_name
    assert act_name is not None

    bridge_model.add_sae(sae)
    assert len(bridge_model._acts_to_saes) == 1
    assert bridge_model._acts_to_saes[act_name].sae == sae
    bridge_model.reset_saes()


def test_add_sae_overwrites_previous_sae(bridge_model: SAETransformerBridge) -> None:
    sae1 = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")
    sae2 = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")
    act_name = sae1.cfg.metadata.hook_name
    assert act_name is not None

    bridge_model.add_sae(sae1)
    assert bridge_model._acts_to_saes[act_name].sae == sae1

    bridge_model.add_sae(sae2)
    assert len(bridge_model._acts_to_saes) == 1
    assert bridge_model._acts_to_saes[act_name].sae == sae2
    bridge_model.reset_saes()


def test_reset_sae_removes_sae(bridge_model: SAETransformerBridge) -> None:
    sae = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")
    act_name = sae.cfg.metadata.hook_name
    assert act_name is not None

    bridge_model.add_sae(sae)
    assert len(bridge_model._acts_to_saes) == 1

    bridge_model._reset_sae(act_name)
    assert len(bridge_model._acts_to_saes) == 0


def test_reset_saes_removes_all_saes(bridge_model: SAETransformerBridge) -> None:
    d_model = bridge_model.cfg.d_model
    saes = [
        make_sae(d_model, "blocks.0.hook_mlp_out"),
        make_sae(d_model, "blocks.0.hook_resid_pre"),
    ]

    for sae in saes:
        bridge_model.add_sae(sae)
    assert len(bridge_model._acts_to_saes) == 2

    bridge_model.reset_saes()
    assert len(bridge_model._acts_to_saes) == 0


def test_saes_context_manager_removes_saes_after(
    bridge_model: SAETransformerBridge,
) -> None:
    sae = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")

    assert len(bridge_model._acts_to_saes) == 0
    with bridge_model.saes(saes=[sae]):
        assert len(bridge_model._acts_to_saes) == 1
        bridge_model(PROMPT)
    assert len(bridge_model._acts_to_saes) == 0


def test_saes_context_manager_restores_previous_state(
    bridge_model: SAETransformerBridge,
) -> None:
    d_model = bridge_model.cfg.d_model
    sae1 = make_sae(d_model, "blocks.0.hook_mlp_out")
    sae2 = make_sae(d_model, "blocks.0.hook_mlp_out")
    act_name = sae1.cfg.metadata.hook_name
    assert act_name is not None

    bridge_model.add_sae(sae1)
    assert bridge_model._acts_to_saes[act_name].sae == sae1

    with bridge_model.saes(saes=[sae2]):
        assert bridge_model._acts_to_saes[act_name].sae == sae2

    assert bridge_model._acts_to_saes[act_name].sae == sae1
    bridge_model.reset_saes()


def test_run_with_saes_removes_saes_after(bridge_model: SAETransformerBridge) -> None:
    original_logits = bridge_model(PROMPT)
    sae = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")

    logits = bridge_model.run_with_saes(PROMPT, saes=[sae])

    assert isinstance(logits, torch.Tensor)
    assert isinstance(original_logits, torch.Tensor)
    assert_not_close(logits, original_logits)
    assert len(bridge_model._acts_to_saes) == 0


def test_add_sae_with_use_error_term_flag(bridge_model: SAETransformerBridge) -> None:
    sae = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")
    act_name = sae.cfg.metadata.hook_name
    assert act_name is not None
    # Resolve the alias to get the actual hook path
    actual_hook = bridge_model._resolve_hook_name(act_name)

    bridge_model.add_sae(sae, use_error_term=True)
    wrapper = get_deep_attr(bridge_model, actual_hook)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.use_error_term is True

    bridge_model.add_sae(sae, use_error_term=False)
    wrapper = get_deep_attr(bridge_model, actual_hook)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.use_error_term is False

    # None defaults to SAE's setting (currently False)
    bridge_model.add_sae(sae, use_error_term=None)
    wrapper = get_deep_attr(bridge_model, actual_hook)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.use_error_term is False  # SAE's use_error_term is False

    bridge_model.reset_saes()


def test_add_sae_respects_sae_use_error_term_setting(
    bridge_model: SAETransformerBridge,
) -> None:
    hook_name = "blocks.0.hook_mlp_out"
    sae = make_sae(bridge_model.cfg.d_model, hook_name)
    actual_hook = bridge_model._resolve_hook_name(hook_name)

    # When SAE has use_error_term=True and we pass None, should use SAE's setting
    sae._use_error_term = True  # Set directly to avoid deprecation warning
    bridge_model.add_sae(sae, use_error_term=None)
    wrapper = get_deep_attr(bridge_model, actual_hook)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.use_error_term is True  # Respects SAE's setting

    bridge_model.reset_saes()

    # Explicit False should override SAE's True
    bridge_model.add_sae(sae, use_error_term=False)
    wrapper = get_deep_attr(bridge_model, actual_hook)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.use_error_term is False

    bridge_model.reset_saes()
    sae._use_error_term = False  # Reset to original


def test_use_error_term_restored_after_exception(
    bridge_model: SAETransformerBridge,
) -> None:
    sae = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")
    original_use_error_term = sae.use_error_term

    try:
        with bridge_model.saes(saes=[sae], use_error_term=True):
            raise RuntimeError("Test exception")
    except RuntimeError:
        pass

    assert sae.use_error_term == original_use_error_term
    assert len(bridge_model._acts_to_saes) == 0


def test_hook_dict_includes_sae_hooks_when_attached(
    bridge_model: SAETransformerBridge,
) -> None:
    sae = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")
    bridge_model.add_sae(sae)

    hook_dict = bridge_model.hook_dict
    # The alias blocks.0.hook_mlp_out resolves to blocks.0.mlp.hook_out
    assert "blocks.0.mlp.hook_out.hook_sae_acts_post" in hook_dict
    assert "blocks.0.mlp.hook_out.hook_sae_input" in hook_dict
    assert "blocks.0.mlp.hook_out.hook_sae_output" in hook_dict

    bridge_model.reset_saes()


def test_sae_activations_same_regardless_of_use_error_term(
    bridge_model: SAETransformerBridge,
) -> None:
    sae = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")

    bridge_model.add_sae(sae, use_error_term=False)
    _, cache_without_error = bridge_model.run_with_cache(PROMPT)
    bridge_model.reset_saes()

    bridge_model.add_sae(sae, use_error_term=True)
    _, cache_with_error = bridge_model.run_with_cache(PROMPT)
    bridge_model.reset_saes()

    # SAE activations should be identical regardless of use_error_term
    acts_without = cache_without_error["blocks.0.mlp.hook_out.hook_sae_acts_post"]
    acts_with = cache_with_error["blocks.0.mlp.hook_out.hook_sae_acts_post"]
    assert_close(acts_without, acts_with)


def test_sae_activations_match_manual_calculation(
    bridge_model: SAETransformerBridge,
) -> None:
    sae = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")

    # Get activations without SAE inserted
    _, cache_no_sae = bridge_model.run_with_cache(PROMPT)
    mlp_out = cache_no_sae["blocks.0.mlp.hook_out"]

    # Get SAE activations with SAE inserted
    bridge_model.add_sae(sae)
    _, cache_with_sae = bridge_model.run_with_cache(PROMPT)
    bridge_model.reset_saes()

    # Manually compute SAE activations
    manual_sae_acts = sae.encode(mlp_out)

    # Compare with cached SAE activations
    cached_sae_acts = cache_with_sae["blocks.0.mlp.hook_out.hook_sae_acts_post"]
    assert_close(manual_sae_acts, cached_sae_acts)


def test_gradients_match_between_hooked_and_bridge(
    hooked_model: HookedSAETransformer, bridge_model: SAETransformerBridge
) -> None:
    sae = make_sae(hooked_model.cfg.d_model, "blocks.0.hook_mlp_out")

    # Run hooked model with gradients
    hooked_model.add_sae(sae)
    hooked_logits = hooked_model(PROMPT)
    hooked_loss = hooked_logits.sum()
    hooked_loss.backward()
    assert sae.W_enc.grad is not None
    hooked_grad = sae.W_enc.grad.clone()
    sae.zero_grad()
    hooked_model.reset_saes()

    # Run bridge model with gradients
    bridge_model.add_sae(sae)
    bridge_logits = bridge_model(PROMPT)
    bridge_loss = bridge_logits.sum()
    bridge_loss.backward()
    assert sae.W_enc.grad is not None
    bridge_grad = sae.W_enc.grad.clone()
    sae.zero_grad()
    bridge_model.reset_saes()

    # Use relative tolerance since gradient magnitudes can be large
    # Small numerical differences accumulate during backprop
    assert_close(hooked_grad, bridge_grad, rtol=1e-2)


def test_add_sae_warns_for_invalid_hook_name(
    bridge_model: SAETransformerBridge, caplog: pytest.LogCaptureFixture
) -> None:
    sae = make_sae(bridge_model.cfg.d_model, "blocks.999.hook_mlp_out")

    with caplog.at_level("WARNING"):
        bridge_model.add_sae(sae)

    assert "No hook found for blocks.999.hook_mlp_out" in caplog.text
    assert len(bridge_model._acts_to_saes) == 0


def test_reset_sae_warns_for_unattached_sae(
    bridge_model: SAETransformerBridge, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        bridge_model._reset_sae("blocks.0.hook_mlp_out")

    assert "No SAE is attached to blocks.0.hook_mlp_out" in caplog.text


def test_run_with_cache_with_saes(bridge_model: SAETransformerBridge) -> None:
    sae = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")

    logits, cache = bridge_model.run_with_cache_with_saes(PROMPT, saes=[sae])

    assert logits is not None
    assert "blocks.0.mlp.hook_out.hook_sae_acts_post" in cache
    assert len(bridge_model._acts_to_saes) == 0


def test_run_with_hooks_with_saes() -> None:
    # Use a fresh model to avoid state issues from module-scoped fixtures
    model = SAETransformerBridge.boot_transformers(MODEL, device="cpu")
    sae = make_sae(model.cfg.d_model, "blocks.0.hook_mlp_out")
    hook_called: list[str] = []

    def hook_fn(act: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        assert hook.name is not None
        hook_called.append(hook.name)
        return act

    # Hook on the base hook point (not SAE internal hooks, which aren't in the registry)
    logits = model.run_with_hooks_with_saes(
        PROMPT,
        saes=[sae],
        fwd_hooks=[("blocks.0.hook_in", hook_fn)],
    )

    assert logits is not None
    assert "blocks.0.hook_in" in hook_called
    assert len(model._acts_to_saes) == 0


def test_run_with_saes_accepts_single_sae(bridge_model: SAETransformerBridge) -> None:
    sae = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")

    logits = bridge_model.run_with_saes(PROMPT, saes=sae)

    assert logits is not None
    assert len(bridge_model._acts_to_saes) == 0


def test_saes_context_manager_accepts_single_sae(
    bridge_model: SAETransformerBridge,
) -> None:
    sae = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")

    with bridge_model.saes(saes=sae):
        assert len(bridge_model._acts_to_saes) == 1
        bridge_model(PROMPT)

    assert len(bridge_model._acts_to_saes) == 0


def test_reset_saes_accepts_string_act_name(bridge_model: SAETransformerBridge) -> None:
    sae = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_mlp_out")
    act_name = sae.cfg.metadata.hook_name

    bridge_model.add_sae(sae)
    assert len(bridge_model._acts_to_saes) == 1

    bridge_model.reset_saes(act_names=act_name)
    assert len(bridge_model._acts_to_saes) == 0


# =============================================================================
# Transcoder tests
# =============================================================================


def make_transcoder(d_model: int) -> Transcoder:
    """Create a transcoder: blocks.0.mlp.hook_in -> blocks.0.hook_mlp_out."""
    cfg = TranscoderConfig(
        d_in=d_model,
        d_sae=d_model * 2,
        d_out=d_model,
        apply_b_dec_to_input=False,
        metadata=SAEMetadata(
            hook_name="blocks.0.mlp.hook_in",
            hook_name_out="blocks.0.hook_mlp_out",
        ),
    )
    return Transcoder(cfg)


@pytest.fixture(scope="module")
def transcoder(bridge_model: SAETransformerBridge) -> Transcoder:
    return make_transcoder(bridge_model.cfg.d_model)


@pytest.fixture(scope="module")
def bridge_original_logits(bridge_model: SAETransformerBridge) -> torch.Tensor:
    return bridge_model(PROMPT)


def test_add_transcoder_changes_output(
    bridge_model: SAETransformerBridge,
    transcoder: Transcoder,
    bridge_original_logits: torch.Tensor,
) -> None:
    bridge_model.add_sae(transcoder)
    assert len(bridge_model._acts_to_saes) == 1
    logits_with_transcoder = bridge_model(PROMPT)
    assert_not_close(bridge_original_logits, logits_with_transcoder)
    bridge_model.reset_saes()


def test_transcoder_with_error_term_preserves_output(
    bridge_model: SAETransformerBridge,
    transcoder: Transcoder,
    bridge_original_logits: torch.Tensor,
) -> None:
    bridge_model.add_sae(transcoder, use_error_term=True)
    assert len(bridge_model._acts_to_saes) == 1
    logits_with_transcoder = bridge_model(PROMPT)
    bridge_model.reset_saes()
    assert_close(bridge_original_logits, logits_with_transcoder, atol=1e-4)


def test_transcoder_reset_removes_transcoder(
    bridge_model: SAETransformerBridge,
    transcoder: Transcoder,
) -> None:
    act_name = transcoder.cfg.metadata.hook_name
    bridge_model.add_sae(transcoder)
    assert len(bridge_model._acts_to_saes) == 1
    assert act_name in bridge_model._acts_to_saes
    bridge_model.reset_saes()
    assert len(bridge_model._acts_to_saes) == 0
    assert isinstance(bridge_model._hook_registry[act_name], HookPoint)


def test_transcoder_context_manager(
    bridge_model: SAETransformerBridge,
    transcoder: Transcoder,
    bridge_original_logits: torch.Tensor,
) -> None:
    act_name = transcoder.cfg.metadata.hook_name

    assert len(bridge_model._acts_to_saes) == 0
    with bridge_model.saes(saes=[transcoder]):
        assert len(bridge_model._acts_to_saes) == 1
        assert act_name in bridge_model._acts_to_saes
        logits_with_transcoder = bridge_model(PROMPT)
        assert_not_close(bridge_original_logits, logits_with_transcoder)

    assert len(bridge_model._acts_to_saes) == 0
    assert isinstance(bridge_model._hook_registry[act_name], HookPoint)


def test_transcoder_context_manager_with_error_term(
    bridge_model: SAETransformerBridge,
    transcoder: Transcoder,
    bridge_original_logits: torch.Tensor,
) -> None:
    with bridge_model.saes(saes=[transcoder], use_error_term=True):
        logits_with_transcoder = bridge_model(PROMPT)
        assert_close(bridge_original_logits, logits_with_transcoder, atol=1e-4)

    assert len(bridge_model._acts_to_saes) == 0


def test_transcoder_run_with_cache(
    bridge_model: SAETransformerBridge,
    transcoder: Transcoder,
    bridge_original_logits: torch.Tensor,
) -> None:
    # For transcoders, the wrapper is placed at output_hook, so cache keys use that
    output_hook = transcoder.cfg.metadata.hook_name_out
    assert output_hook is not None

    logits_with_transcoder, cache = bridge_model.run_with_cache_with_saes(
        PROMPT, saes=[transcoder]
    )
    assert logits_with_transcoder is not None
    assert isinstance(logits_with_transcoder, torch.Tensor)
    assert_not_close(bridge_original_logits, logits_with_transcoder)
    # TransformerBridge uses different hook naming convention
    assert "blocks.0.mlp.hook_out.hook_sae_acts_post" in cache
    assert len(bridge_model._acts_to_saes) == 0


def test_transcoder_activations_match_manual_calculation(
    bridge_model: SAETransformerBridge,
    transcoder: Transcoder,
) -> None:
    input_hook = transcoder.cfg.metadata.hook_name
    output_hook = transcoder.cfg.metadata.hook_name_out
    assert input_hook is not None
    assert output_hook is not None

    # Resolve aliases to actual hook names for cache lookup
    input_hook_actual = bridge_model._resolve_hook_name(input_hook)
    output_hook_actual = bridge_model._resolve_hook_name(output_hook)

    # Get activations without transcoder inserted
    _, cache_no_transcoder = bridge_model.run_with_cache(PROMPT)
    mlp_input = cache_no_transcoder[input_hook_actual]

    # Get transcoder activations with transcoder inserted
    bridge_model.add_sae(transcoder)
    _, cache_with_transcoder = bridge_model.run_with_cache(PROMPT)
    bridge_model.reset_saes()

    # Manually compute transcoder encode/decode
    manual_transcoder_acts = transcoder.encode(mlp_input)
    manual_transcoder_output = transcoder.decode(manual_transcoder_acts)

    # Compare cached activations with manual calculation
    cached_transcoder_acts = cache_with_transcoder[
        output_hook_actual + ".hook_sae_acts_post"
    ]
    assert_close(manual_transcoder_acts, cached_transcoder_acts)

    # Compare cached output with manual calculation
    # This verifies the output hook location is correctly overridden
    cached_transcoder_output = cache_with_transcoder[
        output_hook_actual + ".hook_sae_recons"
    ]
    assert_close(manual_transcoder_output, cached_transcoder_output)


def test_mixed_sae_and_transcoder(
    bridge_model: SAETransformerBridge,
    transcoder: Transcoder,
    bridge_original_logits: torch.Tensor,
) -> None:
    sae = make_sae(bridge_model.cfg.d_model, "blocks.0.hook_resid_pre")
    transcoder_act_name = transcoder.cfg.metadata.hook_name
    sae_act_name = sae.cfg.metadata.hook_name

    bridge_model.add_sae(sae)
    bridge_model.add_sae(transcoder)
    assert len(bridge_model._acts_to_saes) == 2
    assert sae_act_name in bridge_model._acts_to_saes
    assert transcoder_act_name in bridge_model._acts_to_saes

    logits = bridge_model(PROMPT)
    assert_not_close(bridge_original_logits, logits)
    bridge_model.reset_saes()
    assert len(bridge_model._acts_to_saes) == 0


def test_run_with_saes_with_transcoder(
    bridge_model: SAETransformerBridge,
    transcoder: Transcoder,
    bridge_original_logits: torch.Tensor,
) -> None:
    logits = bridge_model.run_with_saes(PROMPT, saes=[transcoder])
    assert logits is not None
    assert isinstance(logits, torch.Tensor)
    assert_not_close(bridge_original_logits, logits)
    assert len(bridge_model._acts_to_saes) == 0


def test_transcoder_gradients_flow(
    bridge_model: SAETransformerBridge, transcoder: Transcoder
) -> None:
    bridge_model.add_sae(transcoder, use_error_term=False)
    output = bridge_model(PROMPT)
    loss = output.sum()
    loss.backward()

    assert transcoder.W_enc.grad is not None
    assert transcoder.W_dec.grad is not None
    assert transcoder.W_enc.grad.abs().sum() > 0
    assert transcoder.W_dec.grad.abs().sum() > 0

    transcoder.zero_grad()
    bridge_model.reset_saes()


def test_transcoder_with_error_term_gradients_flow(
    bridge_model: SAETransformerBridge, transcoder: Transcoder
) -> None:
    bridge_model.add_sae(transcoder, use_error_term=True)
    output = bridge_model(PROMPT)
    loss = output.sum()
    loss.backward()

    assert transcoder.W_enc.grad is not None
    assert transcoder.W_dec.grad is not None
    assert transcoder.W_enc.grad.abs().sum() > 0
    assert transcoder.W_dec.grad.abs().sum() > 0

    transcoder.zero_grad()
    bridge_model.reset_saes()


def test_transcoder_behavior_matches_hooked_transformer(
    hooked_model: HookedSAETransformer,
    bridge_model: SAETransformerBridge,
) -> None:
    transcoder = make_transcoder(hooked_model.cfg.d_model)

    hooked_model.add_sae(transcoder)
    hooked_logits = hooked_model(PROMPT)
    hooked_model.reset_saes()

    bridge_model.add_sae(transcoder)
    bridge_logits = bridge_model(PROMPT)
    bridge_model.reset_saes()

    assert_close(hooked_logits, bridge_logits, atol=1e-4)


def test_transcoder_with_error_term_matches_hooked_transformer(
    hooked_model: HookedSAETransformer,
    bridge_model: SAETransformerBridge,
) -> None:
    transcoder = make_transcoder(hooked_model.cfg.d_model)

    hooked_model.add_sae(transcoder, use_error_term=True)
    hooked_logits = hooked_model(PROMPT)
    hooked_model.reset_saes()

    bridge_model.add_sae(transcoder, use_error_term=True)
    bridge_logits = bridge_model(PROMPT)
    bridge_model.reset_saes()

    assert_close(hooked_logits, bridge_logits, atol=1e-4)
