# type: ignore
import einops
import pytest
import torch
from transformer_lens.hook_points import HookPoint

from sae_lens.analysis.compat import has_hooked_transformer
from sae_lens.saes.sae import SAE, SAEMetadata
from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
from tests.helpers import assert_close

if not has_hooked_transformer():
    pytest.skip(
        "HookedTransformer was removed in transformer-lens 4.0", allow_module_level=True
    )

from transformer_lens import HookedTransformer

from sae_lens import HookedSAETransformer

MODEL = "solu-1l"
prompt = "Hello World!"


class Counter:
    def __init__(self):
        self.count = 0

    def inc(self, *args, **kwargs):  # type: ignore
        self.count += 1


@pytest.fixture(scope="module")
def model():
    model = HookedSAETransformer.from_pretrained_no_processing(MODEL, device="cpu")
    yield model
    model.reset_saes()


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
        apply_b_dec_to_input=False,
        reshape_activations="hook_z" if act_name.endswith("hook_z") else "none",
        metadata=SAEMetadata(
            hook_name=act_name,
            model_name=MODEL,
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


def test_forward_reconstructs_input(model: HookedTransformer, hooked_sae: SAE):
    """Verfiy that the HookedSAE returns an output with the same shape as the input activations."""

    act_name = hooked_sae.cfg.metadata.hook_name
    _, cache = model.run_with_cache(prompt, names_filter=act_name)
    x = cache[act_name]

    sae_output = hooked_sae(x)
    assert sae_output.shape == x.shape


def test_run_with_cache(model: HookedTransformer, hooked_sae: SAE):
    """Verifies that run_with_cache caches SAE activations"""

    act_name = hooked_sae.cfg.metadata.hook_name
    _, cache = model.run_with_cache(prompt, names_filter=act_name)
    x = cache[act_name]

    sae_output, cache = hooked_sae.run_with_cache(x)
    assert sae_output.shape == x.shape

    assert "hook_sae_input" in cache
    assert "hook_sae_acts_pre" in cache
    assert "hook_sae_acts_post" in cache
    assert "hook_sae_recons" in cache
    assert "hook_sae_output" in cache


def test_run_with_hooks(model: HookedTransformer, hooked_sae: SAE):
    """Verifies that run_with_hooks works with SAE activations"""

    c = Counter()
    act_name = hooked_sae.cfg.metadata.hook_name

    _, cache = model.run_with_cache(prompt, names_filter=act_name)
    x = cache[act_name]

    sae_hooks = [
        "hook_sae_input",
        "hook_sae_acts_pre",
        "hook_sae_acts_post",
        "hook_sae_recons",
        "hook_sae_output",
    ]

    sae_output = hooked_sae.run_with_hooks(
        x, fwd_hooks=[(sae_hook_name, c.inc) for sae_hook_name in sae_hooks]
    )
    assert sae_output.shape == x.shape

    assert c.count == len(sae_hooks)


def test_error_term(model: HookedTransformer, hooked_sae: SAE):
    """Verifies that that if we use error_terms, HookedSAE returns an output that is equal tdef test_feature_grads_with_error_term(model: HookedTransformer, hooked_sae: SparseAutoencoderBase):
    o the input activations."""

    act_name = hooked_sae.cfg.metadata.hook_name
    hooked_sae.use_error_term = True

    _, cache = model.run_with_cache(prompt, names_filter=act_name)
    x = cache[act_name]

    sae_output = hooked_sae(x)
    assert sae_output.shape == x.shape
    assert_close(sae_output, x, atol=1e-6)

    """Verifies that pytorch backward computes the correct feature gradients when using error_terms. Motivated by the need to compute feature gradients for attribution patching."""

    act_name = hooked_sae.cfg.metadata.hook_name
    hooked_sae.use_error_term = True

    # Get input activations
    _, cache = model.run_with_cache(prompt, names_filter=act_name)
    x = cache[act_name]

    # Cache gradients with respect to feature acts
    hooked_sae.reset_hooks()
    grad_cache = {}

    def backward_cache_hook(act: torch.Tensor, hook: HookPoint):
        grad_cache[hook.name] = act.detach()

    hooked_sae.add_hook("hook_sae_acts_post", backward_cache_hook, "bwd")  # type: ignore
    hooked_sae.add_hook("hook_sae_output", backward_cache_hook, "bwd")  # type: ignore

    sae_output = hooked_sae(x)
    assert_close(sae_output, x, atol=1e-6)
    value = sae_output.sum()
    value.backward()
    hooked_sae.reset_hooks()

    # Compute gradient analytically
    if act_name.endswith("hook_z"):
        reshaped_output_grad = einops.rearrange(
            grad_cache["hook_sae_output"], "... n_heads d_head -> ... (n_heads d_head)"
        )
        analytic_grad = reshaped_output_grad @ hooked_sae.W_dec.T
    else:
        analytic_grad = grad_cache["hook_sae_output"] @ hooked_sae.W_dec.T

    # Compare analytic gradient with pytorch computed gradient
    assert_close(grad_cache["hook_sae_acts_post"], analytic_grad, atol=1e-6)
