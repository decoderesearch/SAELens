# type: ignore

import pytest
import torch
from transformer_lens.ActivationCache import ActivationCache

from sae_lens.analysis.compat import (
    get_transformer_lens_version,
    has_transformer_bridge,
)
from sae_lens.saes.sae import SAE, SAEMetadata
from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
from tests.helpers import assert_close, assert_not_close

# Skip all tests in this module if transformer-lens v3+ is not installed
pytestmark = pytest.mark.skipif(
    not has_transformer_bridge(),
    reason="TransformerBridge requires transformer-lens v3+",
)

MODEL = "gpt2"
prompt = "Hello World!"


class Counter:
    def __init__(self):
        self.count = 0

    def inc(self, *args, **kwargs):
        self.count += 1


@pytest.fixture(scope="module")
def bridge_model():
    from sae_lens.analysis.sae_transformer_bridge import SAETransformerBridge

    model = SAETransformerBridge.boot_transformers(MODEL, device="cpu")
    yield model
    model.reset_saes()


@pytest.fixture(scope="module")
def bridge_original_logits(bridge_model):
    return bridge_model(prompt)


def get_hooked_sae(d_model: int, act_name: str) -> SAE:
    site_to_size = {
        "hook_mlp_out": d_model,
        "hook_resid_pre": d_model,
        "mlp.hook_out": d_model,
        "hook_out": d_model,
    }
    site = act_name.split(".")[-1]
    # Also check for two-part names like mlp.hook_out
    if site not in site_to_size and len(act_name.split(".")) >= 2:
        site = ".".join(act_name.split(".")[-2:])
    d_in = site_to_size.get(site, d_model)

    sae_cfg = StandardSAEConfig(
        d_in=d_in,
        d_sae=d_in * 2,
        dtype="float32",
        device="cpu",
        metadata=SAEMetadata(
            model_name=MODEL,
            hook_name=act_name,
            hook_head_index=None,
            prepend_bos=True,
        ),
    )
    return StandardSAE(sae_cfg)


@pytest.fixture(scope="module")
def bridge_hooked_sae(bridge_model):
    d_model = bridge_model.cfg.d_model
    return get_hooked_sae(d_model, "blocks.0.hook_mlp_out")


@pytest.fixture(scope="module")
def bridge_hooked_sae_resid(bridge_model):
    d_model = bridge_model.cfg.d_model
    return get_hooked_sae(d_model, "blocks.0.hook_resid_pre")


@pytest.fixture(scope="module")
def list_of_bridge_hooked_saes(bridge_model):
    d_model = bridge_model.cfg.d_model
    act_names = [
        "blocks.0.hook_mlp_out",
        "blocks.0.hook_resid_pre",
    ]
    return [get_hooked_sae(d_model, act_name) for act_name in act_names]


def test_get_transformer_lens_version_parses_correctly():
    major, minor, patch = get_transformer_lens_version()
    assert isinstance(major, int)
    assert isinstance(minor, int)
    assert isinstance(patch, int)
    assert major >= 2


def test_has_transformer_bridge_returns_bool():
    result = has_transformer_bridge()
    assert isinstance(result, bool)


def test_boot_transformers_loads_model(bridge_model):
    from sae_lens.analysis.sae_transformer_bridge import SAETransformerBridge

    assert isinstance(bridge_model, SAETransformerBridge)
    assert hasattr(bridge_model, "acts_to_saes")
    assert len(bridge_model.acts_to_saes) == 0


def test_bridge_model_produces_output(bridge_model):
    output = bridge_model(prompt)
    assert output is not None
    assert isinstance(output, torch.Tensor)


def test_model_with_no_saes_matches_original_model(
    bridge_model, bridge_original_logits
):
    assert len(bridge_model.acts_to_saes) == 0
    logits = bridge_model(prompt)
    assert_close(bridge_original_logits, logits)


def test_model_with_saes_does_not_match_original_model(
    bridge_model,
    bridge_hooked_sae,
    bridge_original_logits,
):
    assert len(bridge_model.acts_to_saes) == 0
    bridge_model.add_sae(bridge_hooked_sae)
    assert len(bridge_model.acts_to_saes) == 1
    logits_with_saes = bridge_model(prompt)
    assert_not_close(bridge_original_logits, logits_with_saes)
    bridge_model.reset_saes()


def test_add_sae(bridge_model, bridge_hooked_sae):
    act_name = bridge_hooked_sae.cfg.metadata.hook_name
    bridge_model.add_sae(bridge_hooked_sae)
    assert len(bridge_model.acts_to_saes) == 1
    assert bridge_model.acts_to_saes[act_name] == bridge_hooked_sae
    bridge_model.reset_saes()


def test_add_sae_overwrites_prev_sae(bridge_model, bridge_hooked_sae):
    act_name = bridge_hooked_sae.cfg.metadata.hook_name
    bridge_model.add_sae(bridge_hooked_sae)

    assert len(bridge_model.acts_to_saes) == 1
    assert bridge_model.acts_to_saes[act_name] == bridge_hooked_sae

    second_hooked_sae = SAE.from_dict(bridge_hooked_sae.cfg.to_dict())
    bridge_model.add_sae(second_hooked_sae)
    assert len(bridge_model.acts_to_saes) == 1
    assert bridge_model.acts_to_saes[act_name] == second_hooked_sae
    bridge_model.reset_saes()


def test_reset_sae_removes_sae_by_default(bridge_model, bridge_hooked_sae):
    act_name = bridge_hooked_sae.cfg.metadata.hook_name
    bridge_model.add_sae(bridge_hooked_sae)
    assert len(bridge_model.acts_to_saes) == 1
    assert bridge_model.acts_to_saes[act_name] == bridge_hooked_sae
    bridge_model._reset_sae(act_name)
    assert len(bridge_model.acts_to_saes) == 0
    bridge_model.reset_saes()


def test_reset_sae_replaces_sae(bridge_model, bridge_hooked_sae):
    act_name = bridge_hooked_sae.cfg.metadata.hook_name
    second_hooked_sae = SAE.from_dict(bridge_hooked_sae.cfg.to_dict())

    bridge_model.add_sae(bridge_hooked_sae)
    assert len(bridge_model.acts_to_saes) == 1
    assert bridge_model.acts_to_saes[act_name] == bridge_hooked_sae
    bridge_model._reset_sae(act_name, second_hooked_sae)
    assert len(bridge_model.acts_to_saes) == 1
    assert bridge_model.acts_to_saes[act_name] == second_hooked_sae
    bridge_model.reset_saes()


def test_reset_saes_removes_all_saes_by_default(
    bridge_model, list_of_bridge_hooked_saes
):
    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_bridge_hooked_saes
    ]
    for hooked_sae in list_of_bridge_hooked_saes:
        bridge_model.add_sae(hooked_sae)
    assert len(bridge_model.acts_to_saes) == len(act_names)
    for act_name, hooked_sae in zip(act_names, list_of_bridge_hooked_saes):
        assert bridge_model.acts_to_saes[act_name] == hooked_sae
    bridge_model.reset_saes()
    assert len(bridge_model.acts_to_saes) == 0
    bridge_model.reset_saes()


def test_saes_context_manager_removes_saes_after(
    bridge_model, list_of_bridge_hooked_saes
):
    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_bridge_hooked_saes
    ]

    assert len(bridge_model.acts_to_saes) == 0
    with bridge_model.saes(saes=list_of_bridge_hooked_saes):
        for act_name, hooked_sae in zip(act_names, list_of_bridge_hooked_saes):
            assert bridge_model.acts_to_saes[act_name] == hooked_sae
        bridge_model.forward(prompt)
    assert len(bridge_model.acts_to_saes) == 0
    bridge_model.reset_saes()


def test_saes_context_manager_restores_previous_sae_state(
    bridge_model, list_of_bridge_hooked_saes
):
    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_bridge_hooked_saes
    ]

    # First add SAEs statefully
    prev_hooked_saes = list_of_bridge_hooked_saes
    for act_name, prev_hooked_sae in zip(act_names, prev_hooked_saes):
        bridge_model.add_sae(prev_hooked_sae)
        assert bridge_model.acts_to_saes[act_name] == prev_hooked_sae
    assert len(bridge_model.acts_to_saes) == len(prev_hooked_saes)

    # Now temporarily run with new SAEs
    d_model = bridge_model.cfg.d_model
    hooked_saes = [get_hooked_sae(d_model, act_name) for act_name in act_names]
    with bridge_model.saes(saes=hooked_saes):
        for act_name, hooked_sae in zip(act_names, hooked_saes):
            assert bridge_model.acts_to_saes[act_name] == hooked_sae
            assert isinstance(bridge_model.acts_to_saes[act_name], SAE)
        bridge_model.forward(prompt)

    # Check that the previously attached SAEs have been restored
    assert len(bridge_model.acts_to_saes) == len(prev_hooked_saes)
    for act_name, prev_hooked_sae in zip(act_names, prev_hooked_saes):
        assert isinstance(bridge_model.acts_to_saes[act_name], SAE)
        assert bridge_model.acts_to_saes[act_name] == prev_hooked_sae
    bridge_model.reset_saes()


def test_run_with_saes(
    bridge_model,
    list_of_bridge_hooked_saes,
    bridge_original_logits,
):
    assert len(bridge_model.acts_to_saes) == 0
    logits_with_saes = bridge_model.run_with_saes(
        prompt, saes=list_of_bridge_hooked_saes
    )
    assert_not_close(logits_with_saes, bridge_original_logits)
    assert len(bridge_model.acts_to_saes) == 0
    bridge_model.reset_saes()


@pytest.mark.xfail(
    reason="TransformerBridge.run_with_cache assumes all hooks are HookPoints, not SAEs"
)
def test_run_with_cache(
    bridge_model,
    list_of_bridge_hooked_saes,
    bridge_original_logits,
):
    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_bridge_hooked_saes
    ]
    for hooked_sae in list_of_bridge_hooked_saes:
        bridge_model.add_sae(hooked_sae)
    assert len(bridge_model.acts_to_saes) == len(list_of_bridge_hooked_saes)
    logits_with_saes, cache = bridge_model.run_with_cache(prompt)
    assert_not_close(logits_with_saes, bridge_original_logits)
    assert isinstance(cache, ActivationCache)
    for act_name, hooked_sae in zip(act_names, list_of_bridge_hooked_saes):
        assert act_name + ".hook_sae_acts_post" in cache
    bridge_model.reset_saes()


@pytest.mark.xfail(
    reason="TransformerBridge.run_with_cache assumes all hooks are HookPoints, not SAEs"
)
def test_run_with_cache_with_saes(
    bridge_model,
    list_of_bridge_hooked_saes,
    bridge_original_logits,
):
    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_bridge_hooked_saes
    ]
    logits_with_saes, cache = bridge_model.run_with_cache_with_saes(
        prompt, saes=list_of_bridge_hooked_saes
    )
    assert_not_close(logits_with_saes, bridge_original_logits)
    assert isinstance(cache, ActivationCache)

    assert len(bridge_model.acts_to_saes) == 0
    for act_name, _ in zip(act_names, list_of_bridge_hooked_saes):
        assert act_name + ".hook_sae_acts_post" in cache
    bridge_model.reset_saes()


@pytest.mark.xfail(
    reason="TransformerBridge.run_with_hooks assumes all hooks are HookPoints, not SAEs"
)
def test_run_with_hooks(
    bridge_model,
    list_of_bridge_hooked_saes,
    bridge_original_logits,
):
    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_bridge_hooked_saes
    ]
    c = Counter()

    for hooked_sae in list_of_bridge_hooked_saes:
        bridge_model.add_sae(hooked_sae)

    logits_with_saes = bridge_model.run_with_hooks(
        prompt,
        fwd_hooks=[(act_name + ".hook_sae_acts_post", c.inc) for act_name in act_names],
    )
    assert_not_close(logits_with_saes, bridge_original_logits)
    assert c.count == len(act_names)
    bridge_model.reset_saes()
    bridge_model.remove_all_hook_fns(including_permanent=True)


@pytest.mark.xfail(
    reason="TransformerBridge.run_with_hooks assumes all hooks are HookPoints, not SAEs"
)
def test_run_with_hooks_with_saes(
    bridge_model,
    list_of_bridge_hooked_saes,
    bridge_original_logits,
):
    act_names = [
        hooked_sae.cfg.metadata.hook_name for hooked_sae in list_of_bridge_hooked_saes
    ]

    c = Counter()

    logits_with_saes = bridge_model.run_with_hooks_with_saes(
        prompt,
        saes=list_of_bridge_hooked_saes,
        fwd_hooks=[(act_name + ".hook_sae_acts_post", c.inc) for act_name in act_names],
    )
    assert_not_close(logits_with_saes, bridge_original_logits)
    assert c.count == len(act_names)

    assert len(bridge_model.acts_to_saes) == 0
    bridge_model.reset_saes()
    bridge_model.remove_all_hook_fns(including_permanent=True)


def test_model_with_use_error_term_saes_matches_original_model(
    bridge_model,
    bridge_hooked_sae,
    bridge_original_logits,
):
    # Clean up any SAEs left by previous tests (especially xfail tests)
    bridge_model.reset_saes()
    assert len(bridge_model.acts_to_saes) == 0
    bridge_model.add_sae(bridge_hooked_sae, use_error_term=True)
    assert len(bridge_model.acts_to_saes) == 1
    logits_with_saes = bridge_model(prompt)
    bridge_model.reset_saes()
    assert_close(bridge_original_logits, logits_with_saes, atol=1e-4)


def test_add_sae_with_use_error_term(bridge_model, bridge_hooked_sae):
    act_name = bridge_hooked_sae.cfg.metadata.hook_name
    original_use_error_term = bridge_hooked_sae.use_error_term

    bridge_model.add_sae(bridge_hooked_sae, use_error_term=True)
    assert bridge_model.acts_to_saes[act_name].use_error_term is True

    bridge_model.add_sae(bridge_hooked_sae, use_error_term=False)
    assert bridge_model.acts_to_saes[act_name].use_error_term is False

    bridge_model.add_sae(bridge_hooked_sae, use_error_term=None)
    assert bridge_model.acts_to_saes[act_name].use_error_term == original_use_error_term

    bridge_model.reset_saes()


def test_saes_context_manager_with_use_error_term(bridge_model, bridge_hooked_sae):
    act_name = bridge_hooked_sae.cfg.metadata.hook_name
    original_use_error_term = bridge_hooked_sae.use_error_term

    with bridge_model.saes(saes=[bridge_hooked_sae], use_error_term=True):
        assert bridge_model.acts_to_saes[act_name].use_error_term is True

    assert bridge_hooked_sae.use_error_term == original_use_error_term
    assert len(bridge_model.acts_to_saes) == 0


def test_run_with_saes_with_use_error_term(
    bridge_model,
    bridge_hooked_sae,
):
    original_use_error_term = bridge_hooked_sae.use_error_term

    bridge_model.run_with_saes(prompt, saes=[bridge_hooked_sae], use_error_term=True)
    assert bridge_hooked_sae.use_error_term == original_use_error_term
    assert len(bridge_model.acts_to_saes) == 0


@pytest.mark.xfail(
    reason="TransformerBridge.run_with_cache assumes all hooks are HookPoints, not SAEs"
)
def test_run_with_cache_with_saes_with_use_error_term(
    bridge_model,
    bridge_hooked_sae,
):
    act_name = bridge_hooked_sae.cfg.metadata.hook_name
    original_use_error_term = bridge_hooked_sae.use_error_term

    _, cache = bridge_model.run_with_cache_with_saes(
        prompt, saes=[bridge_hooked_sae], use_error_term=True
    )
    assert bridge_hooked_sae.use_error_term == original_use_error_term
    assert len(bridge_model.acts_to_saes) == 0
    assert act_name + ".hook_sae_acts_post" in cache


def test_use_error_term_restoration_after_exception(
    bridge_model,
    bridge_hooked_sae,
):
    original_use_error_term = bridge_hooked_sae.use_error_term

    try:
        with bridge_model.saes(saes=[bridge_hooked_sae], use_error_term=True):
            raise Exception("Test exception")
    except Exception:
        pass

    assert bridge_hooked_sae.use_error_term == original_use_error_term
    assert len(bridge_model.acts_to_saes) == 0


def test_hook_dict_property_returns_registry(bridge_model):
    hook_dict = bridge_model.hook_dict
    assert isinstance(hook_dict, dict)
    assert len(hook_dict) > 0


def test_hook_aliases_work(bridge_model, bridge_original_logits):
    d_model = bridge_model.cfg.d_model
    # blocks.0.hook_mlp_out is an alias that should work
    sae = get_hooked_sae(d_model, "blocks.0.hook_mlp_out")

    # Get output before adding SAE (should match original)
    before_sae = bridge_model(prompt)
    assert_close(before_sae, bridge_original_logits)

    # Add SAE - output should change
    bridge_model.add_sae(sae)
    assert "blocks.0.hook_mlp_out" in bridge_model.acts_to_saes
    with_sae = bridge_model(prompt)
    assert_not_close(with_sae, bridge_original_logits)

    # Reset - output should match original again
    bridge_model.reset_saes()
    after_reset = bridge_model(prompt)
    assert_close(after_reset, bridge_original_logits)
