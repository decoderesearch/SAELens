# type: ignore

import pytest
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint  # Hooking utilities
from transformer_lens.HookedTransformer import Loss

from sae_lens.analysis.hooked_sae_transformer import (
    HookedSAETransformer,
    _SAEWrapper,
    get_deep_attr,
)
from sae_lens.saes.sae import SAE, SAEMetadata
from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
from sae_lens.saes.transcoder import Transcoder, TranscoderConfig
from tests.helpers import TINYSTORIES_MODEL, assert_close, assert_not_close

MODEL = TINYSTORIES_MODEL
prompt = "Hello World!"


Output = torch.Tensor | tuple[torch.Tensor, Loss] | None


def get_logits(output: Output) -> torch.Tensor:
    if output is None:
        raise ValueError("Model output is None")
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and len(output) == 2:
        return output[0]
    raise ValueError(f"Unexpected output type: {type(output)}")


class Counter:
    def __init__(self):
        self.count = 0

    def inc(self, *args, **kwargs):  # type: ignore
        self.count += 1


@pytest.fixture(scope="module")
def model():
    model = HookedSAETransformer.from_pretrained(MODEL, device="cpu")
    yield model
    model.reset_saes()  # type: ignore


@pytest.fixture(scope="module")
def original_logits(model: HookedTransformer):
    return model(prompt)


def get_hooked_sae(model: HookedTransformer, act_name: str) -> SAE:
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


@pytest.fixture(
    scope="module",
    params=[
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
    ids=[
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def hooked_sae(
    model: HookedTransformer,
    request: pytest.FixtureRequest,
) -> SAE:
    return get_hooked_sae(model, request.param)


@pytest.fixture(scope="module")
def list_of_hooked_saes(
    model: HookedTransformer,
):
    act_names = [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ]

    return [get_hooked_sae(model, act_name) for act_name in act_names]


def test_model_with_no_saes_matches_original_model(
    model: HookedTransformer, original_logits: torch.Tensor
):
    """Verifies that HookedSAETransformer behaves like a normal HookedTransformer model when no SAEs are attached."""
    assert len(model._acts_to_saes) == 0  # type: ignore
    logits = model(prompt)
    assert_close(original_logits, logits)


def test_model_with_saes_does_not_match_original_model(
    model: HookedTransformer,
    hooked_sae: SAE,
    original_logits: torch.Tensor,
):
    """Verifies that the attached (and turned on) SAEs actually affect the models output logits"""
    assert len(model._acts_to_saes) == 0  # type: ignore
    model.add_sae(hooked_sae)  # type: ignore
    assert len(model._acts_to_saes) == 1  # type: ignore
    logits_with_saes = model(prompt)
    assert_not_close(original_logits, logits_with_saes)
    model.reset_saes()


def test_add_sae(model: HookedTransformer, hooked_sae: SAE):
    """Verifies that add_sae correctly updates the model's _acts_to_saes dictionary and replaces the HookPoint."""
    act_name = hooked_sae.cfg.metadata.hook_name
    model.add_sae(hooked_sae)  # type: ignore
    assert len(model._acts_to_saes) == 1  # type: ignore
    assert model._acts_to_saes[act_name].sae == hooked_sae
    wrapper = get_deep_attr(model, act_name)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.sae == hooked_sae
    model.reset_saes()


def test_acts_to_saes_property_returns_unwrapped_saes(
    model: HookedSAETransformer, hooked_sae: SAE
):
    """Verifies that acts_to_saes property returns unwrapped SAEs, not wrappers."""
    act_name = hooked_sae.cfg.metadata.hook_name
    model.add_sae(hooked_sae)

    # The property should return the underlying SAE, not the wrapper
    assert act_name in model.acts_to_saes
    assert model.acts_to_saes[act_name] == hooked_sae
    assert not isinstance(model.acts_to_saes[act_name], _SAEWrapper)

    # The internal dict should have wrappers
    assert isinstance(model._acts_to_saes[act_name], _SAEWrapper)

    model.reset_saes()


def test_add_sae_warns_for_invalid_hook(
    model: HookedSAETransformer, caplog: pytest.LogCaptureFixture
):
    """Verifies that add_sae logs a warning when given an invalid hook name."""
    cfg = StandardSAEConfig(
        d_in=model.cfg.d_model,
        d_sae=model.cfg.d_model * 2,
        metadata=SAEMetadata(hook_name="invalid.hook.name"),
    )
    sae = StandardSAE(cfg)

    with caplog.at_level("WARNING"):
        model.add_sae(sae)

    assert "No hook found for invalid.hook.name" in caplog.text
    assert len(model._acts_to_saes) == 0


def test_reset_sae_warns_for_unattached_sae(
    model: HookedSAETransformer, caplog: pytest.LogCaptureFixture
):
    """Verifies that reset_sae logs a warning when no SAE is attached."""
    with caplog.at_level("WARNING"):
        model._reset_sae("blocks.0.hook_mlp_out")

    assert "No SAE is attached to blocks.0.hook_mlp_out" in caplog.text


def test_add_sae_overwrites_prev_sae(model: HookedTransformer, hooked_sae: SAE):
    """Verifies that add_sae correctly updates the model's _acts_to_saes dictionary and replaces the HookPoint."""

    act_name = hooked_sae.cfg.metadata.hook_name
    model.add_sae(hooked_sae)

    assert len(model._acts_to_saes) == 1
    assert model._acts_to_saes[act_name].sae == hooked_sae
    wrapper = get_deep_attr(model, act_name)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.sae == hooked_sae

    second_hooked_sae = SAE.from_dict(hooked_sae.cfg.to_dict())  # type: ignore
    model.add_sae(second_hooked_sae)
    assert len(model._acts_to_saes) == 1
    assert model._acts_to_saes[act_name].sae == second_hooked_sae
    wrapper = get_deep_attr(model, act_name)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.sae == second_hooked_sae
    model.reset_saes()


def test_reset_sae_removes_sae_by_default(model: HookedTransformer, hooked_sae: SAE):
    """Verifies that reset_sae correctly removes the SAE from the model's _acts_to_saes dictionary and replaces the wrapper with a HookPoint."""

    act_name = hooked_sae.cfg.metadata.hook_name
    model.add_sae(hooked_sae)
    assert len(model._acts_to_saes) == 1
    assert model._acts_to_saes[act_name].sae == hooked_sae
    wrapper = get_deep_attr(model, act_name)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.sae == hooked_sae
    model._reset_sae(act_name)
    assert len(model._acts_to_saes) == 0
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()


def test_reset_sae_replaces_sae(model: HookedTransformer, hooked_sae: SAE):
    """Verifies that reset_sae correctly removes the SAE from the model's _acts_to_saes dictionary and replaces the wrapper with a new wrapper."""

    act_name = hooked_sae.cfg.metadata.hook_name
    second_hooked_sae = SAE.from_dict(hooked_sae.cfg.to_dict())  # type: ignore

    model.add_sae(hooked_sae)
    assert len(model._acts_to_saes) == 1
    assert model._acts_to_saes[act_name].sae == hooked_sae
    wrapper = get_deep_attr(model, act_name)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.sae == hooked_sae
    model._reset_sae(act_name, _SAEWrapper(second_hooked_sae))
    assert len(model._acts_to_saes) == 1
    wrapper = get_deep_attr(model, act_name)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.sae == second_hooked_sae
    model.reset_saes()


def test_reset_saes_removes_all_saes_by_default(
    model: HookedTransformer, list_of_hooked_saes: list[SAE]
):
    """Verifies that reset_saes correctly removes all SAEs from the model's _acts_to_saes dictionary and replaces the HookedSAEs with HookPoints."""

    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_hooked_saes
    ]
    for hooked_sae in list_of_hooked_saes:
        model.add_sae(hooked_sae)
    assert len(model._acts_to_saes) == len(act_names)
    for act_name, hooked_sae in zip(act_names, list_of_hooked_saes):
        assert model._acts_to_saes[act_name].sae == hooked_sae
        wrapper = get_deep_attr(model, act_name)
        assert isinstance(wrapper, _SAEWrapper)
        assert wrapper.sae == hooked_sae
    model.reset_saes()
    assert len(model._acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()


def test_reset_saes_with_specific_act_names(
    model: HookedTransformer, list_of_hooked_saes: list[SAE]
):
    """Verifies that reset_saes can reset specific SAEs by act_name."""

    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_hooked_saes
    ]

    for hooked_sae in list_of_hooked_saes:
        model.add_sae(hooked_sae)

    assert len(model._acts_to_saes) == len(act_names)

    # Reset only the first SAE
    model.reset_saes(act_names[0])
    assert len(model._acts_to_saes) == len(act_names) - 1
    assert act_names[0] not in model._acts_to_saes
    assert isinstance(get_deep_attr(model, act_names[0]), HookPoint)

    # Remaining SAEs should still be attached
    for act_name in act_names[1:]:
        assert act_name in model._acts_to_saes

    model.reset_saes()


def test_saes_context_manager_removes_saes_after(
    model: HookedTransformer, list_of_hooked_saes: list[SAE]
):
    """Verifies that the model.saes context manager successfully adds the SAEs for the specified activation name in the context manager and resets off after the context manager exits."""

    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_hooked_saes
    ]

    assert len(model._acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    with model.saes(saes=list_of_hooked_saes):
        for act_name, hooked_sae in zip(act_names, list_of_hooked_saes):
            assert model._acts_to_saes[act_name].sae == hooked_sae
            wrapper = get_deep_attr(model, act_name)
            assert isinstance(wrapper, _SAEWrapper)
            assert wrapper.sae == hooked_sae
        model.forward(prompt)  # type: ignore
    assert len(model._acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()


def test_saes_context_manager_restores_previous_sae_state(
    model: HookedTransformer, list_of_hooked_saes: list[SAE]
):
    """Verifies that the model.saes context manager successfully adds the SAEs for the specified activation name in the context manager and resets off after the context manager exits."""

    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_hooked_saes
    ]

    # First add SAEs statefully
    prev_hooked_saes = list_of_hooked_saes
    for act_name, prev_hooked_sae in zip(act_names, prev_hooked_saes):
        model.add_sae(prev_hooked_sae)
        wrapper = get_deep_attr(model, act_name)
        assert isinstance(wrapper, _SAEWrapper)
        assert wrapper.sae == prev_hooked_sae
    assert len(model._acts_to_saes) == len(prev_hooked_saes)

    # Now temporarily run with new SAEs
    hooked_saes = [get_hooked_sae(model, act_name) for act_name in act_names]
    with model.saes(saes=hooked_saes):
        for act_name, hooked_sae in zip(act_names, hooked_saes):
            assert model._acts_to_saes[act_name].sae == hooked_sae
            wrapper = get_deep_attr(model, act_name)
            assert isinstance(wrapper, _SAEWrapper)
            assert wrapper.sae == hooked_sae
        model.forward(prompt)  # type: ignore

    # Check that the previously attached SAEs have been restored
    assert len(model._acts_to_saes) == len(prev_hooked_saes)
    for act_name, prev_hooked_sae in zip(act_names, prev_hooked_saes):
        wrapper = get_deep_attr(model, act_name)
        assert isinstance(wrapper, _SAEWrapper)
        assert wrapper.sae == prev_hooked_sae
    model.reset_saes()


def test_saes_context_manager_run_with_cache(
    model: HookedTransformer, list_of_hooked_saes: list[SAE]
):
    """Verifies that the model.run_with_cache method works correctly in the context manager."""

    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_hooked_saes
    ]
    assert len(model._acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    with model.saes(saes=list_of_hooked_saes):
        for act_name, hooked_sae in zip(act_names, list_of_hooked_saes):
            assert model._acts_to_saes[act_name].sae == hooked_sae
            wrapper = get_deep_attr(model, act_name)
            assert isinstance(wrapper, _SAEWrapper)
            assert wrapper.sae == hooked_sae
        model.run_with_cache(prompt)
    assert len(model._acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()


def test_run_with_saes(
    model: HookedTransformer,
    list_of_hooked_saes: list[SAE],
    original_logits: torch.Tensor,
):
    """Verifies that the model.run_with_saes method works correctly. The logits with SAEs should be different from the original logits, but the SAE should be removed immediately after the forward pass."""

    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_hooked_saes
    ]
    assert len(model._acts_to_saes) == 0
    logits_with_saes = model.run_with_saes(prompt, saes=list_of_hooked_saes)
    assert_not_close(logits_with_saes, original_logits)
    assert len(model._acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()


def test_run_with_cache(
    model: HookedTransformer,
    list_of_hooked_saes: list[SAE],
    original_logits: torch.Tensor,
):
    """Verifies that the model.run_with_cache method works correctly. The logits with SAEs should be different from the original logits and the cache should contain SAE activations for the attached SAE."""
    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_hooked_saes
    ]
    for hooked_sae in list_of_hooked_saes:
        model.add_sae(hooked_sae)
    assert len(model._acts_to_saes) == len(list_of_hooked_saes)
    logits_with_saes, cache = model.run_with_cache(prompt)
    assert_not_close(logits_with_saes, original_logits)
    assert isinstance(cache, ActivationCache)
    for act_name, hooked_sae in zip(act_names, list_of_hooked_saes):
        assert act_name + ".hook_sae_acts_post" in cache
        wrapper = get_deep_attr(model, act_name)
        assert isinstance(wrapper, _SAEWrapper)
        assert wrapper.sae == hooked_sae
    model.reset_saes()


def test_run_with_cache_with_saes(
    model: HookedTransformer,
    list_of_hooked_saes: list[SAE],
    original_logits: torch.Tensor,
):
    """Verifies that the model.run_with_cache_with_saes method works correctly. The logits with SAEs should be different from the original logits and the cache should contain SAE activations for the attached SAE."""

    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_hooked_saes
    ]
    logits_with_saes, cache = model.run_with_cache_with_saes(
        prompt, saes=list_of_hooked_saes
    )
    assert_not_close(logits_with_saes, original_logits)
    assert isinstance(cache, ActivationCache)

    assert len(model._acts_to_saes) == 0
    for act_name, _ in zip(act_names, list_of_hooked_saes):
        assert act_name + ".hook_sae_acts_post" in cache
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()


def test_run_with_hooks(
    model: HookedTransformer,
    list_of_hooked_saes: list[SAE],
    original_logits: torch.Tensor,
):
    """Verifies that the model.run_with_hooks method works correctly when SAEs are attached. The count should be incremented by 1 when the hooked SAE is called, and the SAE should stay attached after the forward pass"""

    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_hooked_saes
    ]
    c = Counter()

    for hooked_sae in list_of_hooked_saes:
        model.add_sae(hooked_sae)

    logits_with_saes = model.run_with_hooks(
        prompt,
        fwd_hooks=[(act_name + ".hook_sae_acts_post", c.inc) for act_name in act_names],
    )
    assert_not_close(logits_with_saes, original_logits)

    for act_name, hooked_sae in zip(act_names, list_of_hooked_saes):
        wrapper = get_deep_attr(model, act_name)
        assert isinstance(wrapper, _SAEWrapper)
        assert wrapper.sae == hooked_sae
    assert c.count == len(act_names)
    model.reset_saes()
    model.remove_all_hook_fns(including_permanent=True)


def test_run_with_hooks_with_saes(
    model: HookedTransformer,
    list_of_hooked_saes: list[SAE],
    original_logits: torch.Tensor,
):
    """Verifies that the model.run_with_hooks_with_saes method works correctly when SAEs are attached. The count should be incremented by 1 when the hooked SAE is called, but the SAE should be removed immediately after the forward pass."""

    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_hooked_saes
    ]

    c = Counter()

    logits_with_saes = model.run_with_hooks_with_saes(
        prompt,
        saes=list_of_hooked_saes,
        fwd_hooks=[(act_name + ".hook_sae_acts_post", c.inc) for act_name in act_names],
    )
    assert_not_close(logits_with_saes, original_logits)
    assert c.count == len(act_names)

    assert len(model._acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()
    model.remove_all_hook_fns(including_permanent=True)


def test_model_with_use_error_term_saes_matches_original_model(
    model: HookedTransformer,
    hooked_sae: SAE,
    original_logits: torch.Tensor,
):
    """Verifies that the attached (and turned on) SAEs actually affect the models output logits"""
    assert len(model._acts_to_saes) == 0
    model.add_sae(hooked_sae, use_error_term=True)
    assert len(model._acts_to_saes) == 1
    logits_with_saes = model(prompt)
    model.reset_saes()
    assert_close(original_logits, logits_with_saes, atol=1e-4)


def test_add_sae_with_use_error_term(model: HookedSAETransformer, hooked_sae: SAE):
    """Verifies that add_sae correctly sets the use_error_term on the wrapper."""
    act_name = hooked_sae.cfg.metadata.hook_name

    model.add_sae(hooked_sae, use_error_term=True)
    wrapper = get_deep_attr(model, act_name)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.use_error_term is True

    model.add_sae(hooked_sae, use_error_term=False)
    wrapper = get_deep_attr(model, act_name)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.use_error_term is False

    # None defaults to SAE's setting (currently False)
    model.add_sae(hooked_sae, use_error_term=None)
    wrapper = get_deep_attr(model, act_name)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.use_error_term is False  # SAE's use_error_term is False

    model.reset_saes()


def test_add_sae_respects_sae_use_error_term_setting(
    model: HookedSAETransformer, hooked_sae: SAE
):
    """Verifies that add_sae respects the SAE's use_error_term when None is passed."""
    act_name = hooked_sae.cfg.metadata.hook_name

    # When SAE has use_error_term=True and we pass None, should use SAE's setting
    hooked_sae._use_error_term = True  # Set directly to avoid deprecation warning
    model.add_sae(hooked_sae, use_error_term=None)
    wrapper = get_deep_attr(model, act_name)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.use_error_term is True  # Respects SAE's setting

    model.reset_saes()

    # Explicit False should override SAE's True
    model.add_sae(hooked_sae, use_error_term=False)
    wrapper = get_deep_attr(model, act_name)
    assert isinstance(wrapper, _SAEWrapper)
    assert wrapper.use_error_term is False

    model.reset_saes()
    hooked_sae._use_error_term = False  # Reset to original


def test_saes_context_manager_with_use_error_term(
    model: HookedSAETransformer, hooked_sae: SAE
):
    """Verifies that the saes context manager correctly handles use_error_term."""
    act_name = hooked_sae.cfg.metadata.hook_name
    original_use_error_term = hooked_sae.use_error_term

    with model.saes(saes=[hooked_sae], use_error_term=True):
        wrapper = get_deep_attr(model, act_name)
        assert isinstance(wrapper, _SAEWrapper)
        assert wrapper.use_error_term is True

    assert hooked_sae.use_error_term == original_use_error_term
    assert len(model._acts_to_saes) == 0


def test_run_with_saes_with_use_error_term(
    model: HookedSAETransformer,
    hooked_sae: SAE,
):
    """Verifies that run_with_saes correctly handles use_error_term."""
    original_use_error_term = hooked_sae.use_error_term

    model.run_with_saes(prompt, saes=[hooked_sae], use_error_term=True)
    assert hooked_sae.use_error_term == original_use_error_term
    assert len(model._acts_to_saes) == 0


def test_run_with_cache_with_saes_with_use_error_term(
    model: HookedSAETransformer,
    hooked_sae: SAE,
):
    """Verifies that run_with_cache_with_saes correctly handles use_error_term."""
    act_name = hooked_sae.cfg.metadata.hook_name
    original_use_error_term = hooked_sae.use_error_term

    _, cache = model.run_with_cache_with_saes(
        prompt, saes=[hooked_sae], use_error_term=True
    )
    assert hooked_sae.use_error_term == original_use_error_term
    assert len(model._acts_to_saes) == 0
    assert act_name + ".hook_sae_acts_post" in cache


def test_use_error_term_restoration_after_exception(
    model: HookedSAETransformer,
    hooked_sae: SAE,
):
    """Verifies that use_error_term is restored even if an exception occurs."""
    original_use_error_term = hooked_sae.use_error_term

    try:
        with model.saes(saes=[hooked_sae], use_error_term=True):
            raise Exception("Test exception")
    except Exception:
        pass

    assert hooked_sae.use_error_term == original_use_error_term
    assert len(model._acts_to_saes) == 0


def test_add_sae_with_use_error_term_true(
    model: HookedSAETransformer,
    hooked_sae: SAE,
):
    """Verifies that add_sae with use_error_term=True doesn't change the model output."""
    # Get output without SAE
    output_without_sae = get_logits(model(prompt))

    # Add SAE with use_error_term=True
    model.add_sae(hooked_sae, use_error_term=True)
    output_with_sae = get_logits(model(prompt))

    # Compare outputs
    assert_close(output_without_sae, output_with_sae, atol=1e-4)

    # Clean up
    model.reset_saes()


def test_run_with_saes_use_error_term_true(
    model: HookedSAETransformer,
    hooked_sae: SAE,
):
    """Verifies that run_with_saes with use_error_term=True doesn't change the model output."""
    # Get output without SAE
    output_without_sae = get_logits(model(prompt))

    # Run with SAE and use_error_term=True
    output_with_sae = get_logits(
        model.run_with_saes(prompt, saes=[hooked_sae], use_error_term=True)
    )

    # Compare outputs
    assert_close(output_without_sae, output_with_sae, atol=1e-4)


def test_run_with_cache_with_saes_use_error_term_true(
    model: HookedSAETransformer,
    hooked_sae: SAE,
):
    """Verifies that run_with_cache_with_saes with use_error_term=True doesn't change the model output."""
    # Get output without SAE
    output_without_sae, _ = model.run_with_cache(prompt)
    output_without_sae = get_logits(output_without_sae)

    # Run with SAE and use_error_term=True
    output_with_sae, cache_with_sae = model.run_with_cache_with_saes(
        prompt, saes=[hooked_sae], use_error_term=True
    )
    output_with_sae = get_logits(output_with_sae)

    # Compare outputs - with use_error_term=True, final outputs should match
    assert_close(output_without_sae, output_with_sae, atol=1e-4)

    # Verify that the cache contains the SAE activations
    assert hooked_sae.cfg.metadata.hook_name + ".hook_sae_acts_post" in cache_with_sae


def test_add_sae_with_use_error_term_false(
    model: HookedSAETransformer,
    hooked_sae: SAE,
):
    """Verifies that add_sae with use_error_term=False changes the model output."""
    # Get output without SAE
    output_without_sae = get_logits(model(prompt))

    # Add SAE with use_error_term=False
    model.add_sae(hooked_sae, use_error_term=False)
    output_with_sae = get_logits(model(prompt))

    # Compare outputs - they should be different
    assert_not_close(output_without_sae, output_with_sae, atol=1e-5)

    # Clean up
    model.reset_saes()


def test_run_with_saes_use_error_term_false(
    model: HookedSAETransformer,
    hooked_sae: SAE,
):
    """Verifies that run_with_saes with use_error_term=False changes the model output."""
    # Get output without SAE
    output_without_sae = get_logits(model(prompt))

    # Run with SAE and use_error_term=False
    output_with_sae = get_logits(
        model.run_with_saes(prompt, saes=[hooked_sae], use_error_term=False)
    )

    # Compare outputs - they should be different
    assert_not_close(output_without_sae, output_with_sae, atol=1e-4)


def test_run_with_cache_with_saes_use_error_term_false(
    model: HookedSAETransformer,
    hooked_sae: SAE,
):
    """Verifies that run_with_cache_with_saes with use_error_term=False changes the model output."""
    # Get output without SAE
    output_without_sae, cache_without_sae = model.run_with_cache(prompt)
    output_without_sae = get_logits(output_without_sae)

    # Run with SAE and use_error_term=False
    output_with_sae, cache_with_sae = model.run_with_cache_with_saes(
        prompt, saes=[hooked_sae], use_error_term=False
    )
    output_with_sae = get_logits(output_with_sae)

    # Compare outputs - they should be different
    assert_not_close(output_without_sae, output_with_sae, atol=1e-4)

    # Verify that the cache contains the SAE activations
    assert hooked_sae.cfg.metadata.hook_name + ".hook_sae_acts_post" in cache_with_sae

    # Verify that the activations at the SAE hook point are different in both caches
    assert_not_close(
        cache_without_sae[hooked_sae.cfg.metadata.hook_name],
        cache_with_sae[hooked_sae.cfg.metadata.hook_name + ".hook_sae_output"],
        atol=1e-5,
    )


def test_HookedSAETransformer_works_with_hook_z_saes():
    sae = SAE.from_pretrained("gpt2-small-hook-z-kk", "blocks.2.hook_z", device="cpu")
    model = HookedSAETransformer.from_pretrained("gpt2", device="cpu")
    logits_sans_sae = model(prompt)
    logits, cache = model.run_with_cache_with_saes(
        prompt, saes=[sae], use_error_term=False
    )
    assert_not_close(logits, logits_sans_sae, atol=1e-4)
    assert cache[sae.cfg.metadata.hook_name + ".hook_sae_output"] is not None
    expected_shape = (1, 4, 12, 64)  # due to hook_z reshaping
    assert (
        cache[sae.cfg.metadata.hook_name + ".hook_sae_output"].shape == expected_shape
    )


def test_HookedSAETransformer_adds_hook_in_to_mlp():
    model = HookedSAETransformer.from_pretrained("gpt2", device="cpu")
    _, cache = model.run_with_cache(prompt)
    for n in range(model.cfg.n_layers):
        assert f"blocks.{n}.mlp.hook_in" in cache
        assert cache[f"blocks.{n}.mlp.hook_in"].shape == (1, 4, 768)


# ============================================================================
# Transcoder Tests
# ============================================================================


def get_transcoder(model: HookedTransformer) -> Transcoder:
    """Create a transcoder: blocks.0.mlp.hook_in -> blocks.0.hook_mlp_out."""
    cfg = TranscoderConfig(
        d_in=model.cfg.d_model,
        d_sae=model.cfg.d_model * 2,
        d_out=model.cfg.d_model,
        apply_b_dec_to_input=False,
        metadata=SAEMetadata(
            hook_name="blocks.0.mlp.hook_in",
            hook_name_out="blocks.0.hook_mlp_out",
        ),
    )
    return Transcoder(cfg)


@pytest.fixture(scope="module")
def transcoder(model: HookedTransformer) -> Transcoder:
    return get_transcoder(model)


def test_add_transcoder_changes_output(
    model: HookedSAETransformer,
    transcoder: Transcoder,
    original_logits: torch.Tensor,
):
    model.add_sae(transcoder)
    assert len(model._acts_to_saes) == 1
    logits_with_transcoder = model(prompt)
    assert_not_close(original_logits, logits_with_transcoder)
    model.reset_saes()


def test_transcoder_with_error_term_preserves_output(
    model: HookedSAETransformer,
    transcoder: Transcoder,
    original_logits: torch.Tensor,
):
    model.add_sae(transcoder, use_error_term=True)
    assert len(model._acts_to_saes) == 1
    logits_with_transcoder = model(prompt)
    model.reset_saes()
    assert_close(original_logits, logits_with_transcoder, atol=1e-4)


def test_transcoder_reset_removes_transcoder(
    model: HookedSAETransformer,
    transcoder: Transcoder,
):
    act_name = transcoder.cfg.metadata.hook_name
    model.add_sae(transcoder)
    assert len(model._acts_to_saes) == 1
    assert act_name in model._acts_to_saes
    model.reset_saes()
    assert len(model._acts_to_saes) == 0
    assert isinstance(get_deep_attr(model, act_name), HookPoint)


def test_transcoder_context_manager(
    model: HookedSAETransformer,
    transcoder: Transcoder,
    original_logits: torch.Tensor,
):
    act_name = transcoder.cfg.metadata.hook_name

    assert len(model._acts_to_saes) == 0
    with model.saes(saes=[transcoder]):
        assert len(model._acts_to_saes) == 1
        assert act_name in model._acts_to_saes
        logits_with_transcoder = model(prompt)
        assert_not_close(original_logits, logits_with_transcoder)

    assert len(model._acts_to_saes) == 0
    assert isinstance(get_deep_attr(model, act_name), HookPoint)


def test_transcoder_context_manager_with_error_term(
    model: HookedSAETransformer,
    transcoder: Transcoder,
    original_logits: torch.Tensor,
):
    with model.saes(saes=[transcoder], use_error_term=True):
        logits_with_transcoder = model(prompt)
        assert_close(original_logits, logits_with_transcoder, atol=1e-4)

    assert len(model._acts_to_saes) == 0


def test_transcoder_run_with_cache(
    model: HookedSAETransformer,
    transcoder: Transcoder,
    original_logits: torch.Tensor,
):
    # For transcoders, the wrapper is placed at output_hook with SAE as submodule
    output_hook = transcoder.cfg.metadata.hook_name_out

    logits_with_transcoder, cache = model.run_with_cache_with_saes(
        prompt, saes=[transcoder]
    )
    assert_not_close(original_logits, logits_with_transcoder)
    assert isinstance(cache, ActivationCache)
    # Hooks are copied directly to wrapper, so cache key doesn't include .sae. prefix
    assert output_hook + ".hook_sae_acts_post" in cache
    assert len(model._acts_to_saes) == 0


def test_mixed_sae_and_transcoder(
    model: HookedSAETransformer,
    transcoder: Transcoder,
    original_logits: torch.Tensor,
):
    sae = get_hooked_sae(model, "blocks.0.hook_resid_pre")
    transcoder_act_name = transcoder.cfg.metadata.hook_name
    sae_act_name = sae.cfg.metadata.hook_name

    model.add_sae(sae)
    model.add_sae(transcoder)
    assert len(model._acts_to_saes) == 2
    assert sae_act_name in model._acts_to_saes
    assert transcoder_act_name in model._acts_to_saes

    logits = model(prompt)
    assert_not_close(original_logits, logits)
    model.reset_saes()
    assert len(model._acts_to_saes) == 0


def test_run_with_saes_with_transcoder(
    model: HookedSAETransformer,
    transcoder: Transcoder,
    original_logits: torch.Tensor,
):
    logits = model.run_with_saes(prompt, saes=[transcoder])
    assert_not_close(original_logits, logits)
    assert len(model._acts_to_saes) == 0


def test_transcoder_gradients_flow(model: HookedSAETransformer, transcoder: Transcoder):
    model.add_sae(transcoder, use_error_term=False)
    output = model(prompt)
    loss = output.sum()
    loss.backward()

    assert transcoder.W_enc.grad is not None
    assert transcoder.W_dec.grad is not None
    assert transcoder.W_enc.grad.abs().sum() > 0
    assert transcoder.W_dec.grad.abs().sum() > 0

    transcoder.zero_grad()
    model.reset_saes()


def test_transcoder_with_error_term_gradients_flow(
    model: HookedSAETransformer, transcoder: Transcoder
):
    model.add_sae(transcoder, use_error_term=True)
    output = model(prompt)
    loss = output.sum()
    loss.backward()

    assert transcoder.W_enc.grad is not None
    assert transcoder.W_dec.grad is not None
    assert transcoder.W_enc.grad.abs().sum() > 0
    assert transcoder.W_dec.grad.abs().sum() > 0

    transcoder.zero_grad()
    model.reset_saes()


def test_multiple_chained_transcoders_get_gradients():
    """Test that multiple transcoders in sequence all receive gradients correctly."""
    model = HookedSAETransformer.from_pretrained(MODEL, device="cpu")

    # Create transcoders for layers 0, 1, and 2
    transcoders = []
    for layer in range(3):
        cfg = TranscoderConfig(
            d_in=model.cfg.d_model,
            d_sae=model.cfg.d_model * 2,
            d_out=model.cfg.d_model,
            apply_b_dec_to_input=False,
            metadata=SAEMetadata(
                hook_name=f"blocks.{layer}.mlp.hook_in",
                hook_name_out=f"blocks.{layer}.hook_mlp_out",
            ),
        )
        transcoders.append(Transcoder(cfg))

    # Attach all transcoders
    for tc in transcoders:
        model.add_sae(tc)

    assert len(model._acts_to_saes) == 3

    # Forward and backward pass
    output = model(prompt)
    loss = output.sum()
    loss.backward()

    # Verify all transcoders got gradients
    for i, tc in enumerate(transcoders):
        assert tc.W_enc.grad is not None, f"Transcoder {i} W_enc has no gradient"
        assert tc.W_dec.grad is not None, f"Transcoder {i} W_dec has no gradient"
        assert tc.W_enc.grad.abs().sum() > 0, f"Transcoder {i} W_enc gradient is zero"
        assert tc.W_dec.grad.abs().sum() > 0, f"Transcoder {i} W_dec gradient is zero"
        tc.zero_grad()

    model.reset_saes()


def test_multiple_saes_get_gradients():
    """Test that multiple SAEs attached simultaneously all receive gradients correctly."""
    model = HookedSAETransformer.from_pretrained(MODEL, device="cpu")

    # Create SAEs for different hook points across layers
    hook_names = [
        "blocks.0.hook_mlp_out",
        "blocks.1.hook_mlp_out",
        "blocks.2.hook_resid_pre",
    ]
    saes = []
    for hook_name in hook_names:
        cfg = StandardSAEConfig(
            d_in=model.cfg.d_model,
            d_sae=model.cfg.d_model * 2,
            metadata=SAEMetadata(hook_name=hook_name),
        )
        saes.append(StandardSAE(cfg))

    # Attach all SAEs
    for sae in saes:
        model.add_sae(sae)

    assert len(model._acts_to_saes) == 3

    # Forward and backward pass
    output = model(prompt)
    loss = output.sum()
    loss.backward()

    # Verify all SAEs got gradients
    for i, sae in enumerate(saes):
        assert sae.W_enc.grad is not None, f"SAE {i} W_enc has no gradient"
        assert sae.W_dec.grad is not None, f"SAE {i} W_dec has no gradient"
        assert sae.W_enc.grad.abs().sum() > 0, f"SAE {i} W_enc gradient is zero"
        assert sae.W_dec.grad.abs().sum() > 0, f"SAE {i} W_dec gradient is zero"
        sae.zero_grad()

    model.reset_saes()
