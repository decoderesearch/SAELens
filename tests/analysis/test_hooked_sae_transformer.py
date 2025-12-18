# type: ignore

import pytest
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint  # Hooking utilities
from transformer_lens.HookedTransformer import Loss

from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer, get_deep_attr
from sae_lens.saes.sae import SAE, SAEMetadata
from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
from tests.helpers import assert_close, assert_not_close

MODEL = "solu-1l"
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
    assert len(model.acts_to_saes) == 0  # type: ignore
    logits = model(prompt)
    assert_close(original_logits, logits)


def test_model_with_saes_does_not_match_original_model(
    model: HookedTransformer,
    hooked_sae: SAE,
    original_logits: torch.Tensor,
):
    """Verifies that the attached (and turned on) SAEs actually affect the models output logits"""
    assert len(model.acts_to_saes) == 0  # type: ignore
    model.add_sae(hooked_sae)  # type: ignore
    assert len(model.acts_to_saes) == 1  # type: ignore
    logits_with_saes = model(prompt)
    assert_not_close(original_logits, logits_with_saes)
    model.reset_saes()


def test_add_sae(model: HookedTransformer, hooked_sae: SAE):
    """Verifies that add_sae correctly updates the model's acts_to_saes dictionary and replaces the HookPoint."""
    act_name = hooked_sae.cfg.metadata.hook_name
    model.add_sae(hooked_sae)  # type: ignore
    assert len(model.acts_to_saes) == 1  # type: ignore
    assert model.acts_to_saes[act_name] == hooked_sae
    assert get_deep_attr(model, act_name) == hooked_sae
    model.reset_saes()


def test_add_sae_overwrites_prev_sae(model: HookedTransformer, hooked_sae: SAE):
    """Verifies that add_sae correctly updates the model's acts_to_saes dictionary and replaces the HookPoint."""

    act_name = hooked_sae.cfg.metadata.hook_name
    model.add_sae(hooked_sae)

    assert len(model.acts_to_saes) == 1
    assert model.acts_to_saes[act_name] == hooked_sae
    assert get_deep_attr(model, act_name) == hooked_sae

    second_hooked_sae = SAE.from_dict(hooked_sae.cfg.to_dict())  # type: ignore
    model.add_sae(second_hooked_sae)
    assert len(model.acts_to_saes) == 1
    assert model.acts_to_saes[act_name] == second_hooked_sae
    assert get_deep_attr(model, act_name) == second_hooked_sae
    model.reset_saes()


def test_reset_sae_removes_sae_by_default(model: HookedTransformer, hooked_sae: SAE):
    """Verifies that reset_sae correctly removes the SAE from the model's acts_to_saes dictionary and replaces the HookedSAE with a HookPoint."""

    act_name = hooked_sae.cfg.metadata.hook_name
    model.add_sae(hooked_sae)
    assert len(model.acts_to_saes) == 1
    assert model.acts_to_saes[act_name] == hooked_sae
    assert get_deep_attr(model, act_name) == hooked_sae
    model._reset_sae(act_name)
    assert len(model.acts_to_saes) == 0
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()


def test_reset_sae_replaces_sae(model: HookedTransformer, hooked_sae: SAE):
    """Verifies that reset_sae correctly removes the SAE from the model's acts_to_saes dictionary and replaces the HookedSAE with a HookPoint."""

    act_name = hooked_sae.cfg.metadata.hook_name
    second_hooked_sae = SAE.from_dict(hooked_sae.cfg.to_dict())  # type: ignore

    model.add_sae(hooked_sae)
    assert len(model.acts_to_saes) == 1
    assert model.acts_to_saes[act_name] == hooked_sae
    assert get_deep_attr(model, act_name) == hooked_sae
    model._reset_sae(act_name, second_hooked_sae)
    assert len(model.acts_to_saes) == 1
    assert get_deep_attr(model, act_name) == second_hooked_sae
    model.reset_saes()


def test_reset_saes_removes_all_saes_by_default(
    model: HookedTransformer, list_of_hooked_saes: list[SAE]
):
    """Verifies that reset_saes correctly removes all SAEs from the model's acts_to_saes dictionary and replaces the HookedSAEs with HookPoints."""

    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_hooked_saes
    ]
    for hooked_sae in list_of_hooked_saes:
        model.add_sae(hooked_sae)
    assert len(model.acts_to_saes) == len(act_names)
    for act_name, hooked_sae in zip(act_names, list_of_hooked_saes):
        assert model.acts_to_saes[act_name] == hooked_sae
        assert get_deep_attr(model, act_name) == hooked_sae
    model.reset_saes()
    assert len(model.acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()


def test_reset_saes_replaces_saes(
    model: HookedTransformer, list_of_hooked_saes: list[SAE]
):
    """Verifies that reset_saes correctly removes all SAEs from the model's acts_to_saes dictionary and replaces the HookedSAEs with HookPoints."""

    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_hooked_saes
    ]

    for hooked_sae in list_of_hooked_saes:
        model.add_sae(hooked_sae)

    prev_hooked_saes = [get_hooked_sae(model, act_name) for act_name in act_names]

    assert len(model.acts_to_saes) == len(act_names)
    for act_name, hooked_sae in zip(act_names, list_of_hooked_saes):
        assert model.acts_to_saes[act_name] == hooked_sae
        assert get_deep_attr(model, act_name) == hooked_sae
    model.reset_saes(act_names, prev_hooked_saes)
    assert len(model.acts_to_saes) == len(prev_hooked_saes)
    for act_name, prev_hooked_sae in zip(act_names, prev_hooked_saes):
        assert get_deep_attr(model, act_name) == prev_hooked_sae
    model.reset_saes()


def test_saes_context_manager_removes_saes_after(
    model: HookedTransformer, list_of_hooked_saes: list[SAE]
):
    """Verifies that the model.saes context manager successfully adds the SAEs for the specified activation name in the context manager and resets off after the context manager exits."""

    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_hooked_saes
    ]

    assert len(model.acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    with model.saes(saes=list_of_hooked_saes):
        for act_name, hooked_sae in zip(act_names, list_of_hooked_saes):
            assert model.acts_to_saes[act_name] == hooked_sae
            assert isinstance(get_deep_attr(model, act_name), SAE)
            assert get_deep_attr(model, act_name) == hooked_sae
        model.forward(prompt)  # type: ignore
    assert len(model.acts_to_saes) == 0
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
        assert get_deep_attr(model, act_name) == prev_hooked_sae
    assert len(model.acts_to_saes) == len(prev_hooked_saes)

    # Now temporarily run with new SAEs
    hooked_saes = [get_hooked_sae(model, act_name) for act_name in act_names]
    with model.saes(saes=hooked_saes):
        for act_name, hooked_sae in zip(act_names, hooked_saes):
            assert model.acts_to_saes[act_name] == hooked_sae
            assert isinstance(get_deep_attr(model, act_name), SAE)
            assert get_deep_attr(model, act_name) == hooked_sae
        model.forward(prompt)  # type: ignore

    # Check that the previously attached SAEs have been restored
    assert len(model.acts_to_saes) == len(prev_hooked_saes)
    for act_name, prev_hooked_sae in zip(act_names, prev_hooked_saes):
        assert isinstance(get_deep_attr(model, act_name), SAE)
        assert get_deep_attr(model, act_name) == prev_hooked_sae
    model.reset_saes()


def test_saes_context_manager_run_with_cache(
    model: HookedTransformer, list_of_hooked_saes: list[SAE]
):
    """Verifies that the model.run_with_cache method works correctly in the context manager."""

    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_hooked_saes
    ]
    assert len(model.acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    with model.saes(saes=list_of_hooked_saes):
        for act_name, hooked_sae in zip(act_names, list_of_hooked_saes):
            assert model.acts_to_saes[act_name] == hooked_sae
            assert isinstance(get_deep_attr(model, act_name), SAE)
            assert get_deep_attr(model, act_name) == hooked_sae
        model.run_with_cache(prompt)
    assert len(model.acts_to_saes) == 0
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
    assert len(model.acts_to_saes) == 0
    logits_with_saes = model.run_with_saes(prompt, saes=list_of_hooked_saes)
    assert_not_close(logits_with_saes, original_logits)
    assert len(model.acts_to_saes) == 0
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
    assert len(model.acts_to_saes) == len(list_of_hooked_saes)
    logits_with_saes, cache = model.run_with_cache(prompt)
    assert_not_close(logits_with_saes, original_logits)
    assert isinstance(cache, ActivationCache)
    for act_name, hooked_sae in zip(act_names, list_of_hooked_saes):
        assert act_name + ".hook_sae_acts_post" in cache
        assert isinstance(get_deep_attr(model, act_name), SAE)
        assert get_deep_attr(model, act_name) == hooked_sae
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

    assert len(model.acts_to_saes) == 0
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
        assert isinstance(get_deep_attr(model, act_name), SAE)
        assert get_deep_attr(model, act_name) == hooked_sae
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

    assert len(model.acts_to_saes) == 0
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
    assert len(model.acts_to_saes) == 0
    model.add_sae(hooked_sae, use_error_term=True)
    assert len(model.acts_to_saes) == 1
    logits_with_saes = model(prompt)
    model.reset_saes()
    assert_close(original_logits, logits_with_saes, atol=1e-4)


def test_add_sae_with_use_error_term(model: HookedSAETransformer, hooked_sae: SAE):
    """Verifies that add_sae correctly sets the use_error_term when specified."""
    act_name = hooked_sae.cfg.metadata.hook_name
    original_use_error_term = hooked_sae.use_error_term

    model.add_sae(hooked_sae, use_error_term=True)
    assert model.acts_to_saes[act_name].use_error_term is True

    model.add_sae(hooked_sae, use_error_term=False)
    assert model.acts_to_saes[act_name].use_error_term is False

    model.add_sae(hooked_sae, use_error_term=None)
    assert model.acts_to_saes[act_name].use_error_term == original_use_error_term

    model.reset_saes()


def test_saes_context_manager_with_use_error_term(
    model: HookedSAETransformer, hooked_sae: SAE
):
    """Verifies that the saes context manager correctly handles use_error_term."""
    act_name = hooked_sae.cfg.metadata.hook_name
    original_use_error_term = hooked_sae.use_error_term

    with model.saes(saes=[hooked_sae], use_error_term=True):
        assert model.acts_to_saes[act_name].use_error_term is True

    assert hooked_sae.use_error_term == original_use_error_term
    assert len(model.acts_to_saes) == 0


def test_run_with_saes_with_use_error_term(
    model: HookedSAETransformer,
    hooked_sae: SAE,
):
    """Verifies that run_with_saes correctly handles use_error_term."""
    original_use_error_term = hooked_sae.use_error_term

    model.run_with_saes(prompt, saes=[hooked_sae], use_error_term=True)
    assert hooked_sae.use_error_term == original_use_error_term
    assert len(model.acts_to_saes) == 0


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
    assert len(model.acts_to_saes) == 0
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
    assert len(model.acts_to_saes) == 0


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
    output_without_sae, cache_without_sae = model.run_with_cache(prompt)
    output_without_sae = get_logits(output_without_sae)

    # Run with SAE and use_error_term=True
    output_with_sae, cache_with_sae = model.run_with_cache_with_saes(
        prompt, saes=[hooked_sae], use_error_term=True
    )
    output_with_sae = get_logits(output_with_sae)

    # Compare outputs
    assert_close(output_without_sae, output_with_sae, atol=1e-4)

    # Verify that the cache contains the SAE activations
    assert hooked_sae.cfg.metadata.hook_name + ".hook_sae_acts_post" in cache_with_sae

    # Verify that the activations at the SAE hook point are the same in both caches
    assert_close(
        cache_without_sae[hooked_sae.cfg.metadata.hook_name],
        cache_with_sae[hooked_sae.cfg.metadata.hook_name + ".hook_sae_output"],
        atol=1e-5,
    )


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
