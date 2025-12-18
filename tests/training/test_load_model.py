import sys

import pytest
from mamba_lens import HookedMamba
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_lens.load_model import HookedProxyLM, _extract_logits_from_output, load_model
from tests.helpers import assert_close


@pytest.fixture
def gpt2_proxy_model():
    return load_model(
        model_class_name="AutoModelForCausalLM",
        model_name="gpt2",
        device="cpu",
    )


def test_load_model_works_with_mamba():
    model = load_model(
        model_class_name="HookedMamba",
        model_name="state-spaces/mamba-130m",
        device="cpu",
    )
    assert isinstance(model, HookedMamba)


def test_load_model_works_without_model_kwargs():
    model = load_model(
        model_class_name="HookedTransformer",
        model_name="pythia-14m",
        device="cpu",
    )
    assert isinstance(model, HookedTransformer)
    assert model.cfg.checkpoint_index is None


# TODO: debug why this is suddenly failing on CI. It may resolve itself in the future.
@pytest.mark.skip(
    reason="This is failing on CI but not locally due to huggingface headers."
)
def test_load_model_works_with_model_kwargs():
    model = load_model(
        model_class_name="HookedTransformer",
        model_name="pythia-14m",
        device="cpu",
        model_from_pretrained_kwargs={"checkpoint_index": 0},
    )
    assert isinstance(model, HookedTransformer)
    assert model.cfg.checkpoint_index == 0


def test_load_model_with_generic_huggingface_lm():
    model = load_model(
        model_class_name="AutoModelForCausalLM",
        model_name="gpt2",
        device="cpu",
    )
    assert isinstance(model, HookedProxyLM)


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Test crashes Python interpreter on macOS"
)
def test_HookedProxyLM_gives_same_cached_states_as_original_implementation():
    hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    hooked_model = HookedProxyLM(hf_model, tokenizer)
    input_ids = tokenizer.encode("hi", return_tensors="pt")
    proxy_logits, cache = hooked_model.run_with_cache(input_ids)

    hf_output = hf_model(input_ids, output_hidden_states=True)

    assert_close(proxy_logits, hf_output.logits)
    for i in range(len(hf_output.hidden_states) - 2):
        assert_close(cache[f"transformer.h.{i}"], hf_output.hidden_states[i + 1])


def test_HookedProxyLM_gives_same_cached_states_as_tlens_implementation(
    gpt2_proxy_model: HookedProxyLM,
):
    tlens_model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")

    input_ids = tlens_model.to_tokens("hi")
    hf_cache = gpt2_proxy_model.run_with_cache(input_ids)[1]
    tlens_cache = tlens_model.run_with_cache(input_ids)[1]
    for i in range(12):
        assert_close(
            hf_cache[f"transformer.h.{i}"],
            tlens_cache[f"blocks.{i}.hook_resid_post"],
            atol=1e-3,
        )


def test_HookedProxyLM_forward_gives_same_output_as_tlens(
    gpt2_proxy_model: HookedProxyLM,
):
    tlens_model = HookedTransformer.from_pretrained("gpt2", device="cpu")

    batch_tokens = tlens_model.to_tokens("hi there")
    tlens_output = tlens_model(batch_tokens, return_type="both", loss_per_token=True)
    hf_output = gpt2_proxy_model(batch_tokens, return_type="both", loss_per_token=True)

    # Seems like tlens removes the means before softmaxing
    hf_logits_normed = hf_output[0] - hf_output[0].mean(dim=-1, keepdim=True)

    assert_close(tlens_output[0], hf_logits_normed, atol=1e-3)
    assert_close(tlens_output[1], hf_output[1], atol=1e-3)


def test_extract_logits_from_output_works_with_multiple_return_types():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode("hi there", return_tensors="pt")
    out_dict = model(tokens, return_dict=True)
    out_tuple = model(tokens, return_dict=False)

    logits_dict = _extract_logits_from_output(out_dict)
    logits_tuple = _extract_logits_from_output(out_tuple)

    assert_close(logits_dict, logits_tuple)


def test_HookedProxyLM_to_tokens_gives_same_output_as_tlens(
    gpt2_proxy_model: HookedProxyLM,
):
    tlens_model = HookedTransformer.from_pretrained("gpt2", device="cpu")

    tl_tokens = tlens_model.to_tokens(
        "hi there", prepend_bos=False, truncate=False, move_to_device=False
    )
    hf_tokens = gpt2_proxy_model.to_tokens(
        "hi there", prepend_bos=False, truncate=False, move_to_device=False
    )

    assert_close(tl_tokens, hf_tokens)


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Test crashes Python interpreter on macOS"
)
def test_HookedProxyLM_gives_same_hidden_states_when_stop_at_layer_and_names_filter_are_set(
    gpt2_proxy_model: HookedProxyLM,
):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode("hi", return_tensors="pt")
    layers = ["transformer.h.0", "transformer.h.1"]

    # Get initial hook counts for the modules we're interested in
    initial_hook_counts = {}
    for layer in layers:
        module = gpt2_proxy_model.named_modules_dict[layer]
        initial_hook_counts[layer] = len(module._forward_hooks)

    res_with_stop, cache_with_stop = gpt2_proxy_model.run_with_cache(
        input_ids,
        stop_at_layer=3,
        names_filter=layers,
    )

    # Verify hooks are removed after first run
    for layer in layers:
        module = gpt2_proxy_model.named_modules_dict[layer]
        assert (
            len(module._forward_hooks) == initial_hook_counts[layer]
        ), f"Stop hooks not removed from {layer}"

    res_no_stop, cache_no_stop = gpt2_proxy_model.run_with_cache(
        input_ids, names_filter=layers
    )

    # Verify hooks are still clean after second run
    for layer in layers:
        module = gpt2_proxy_model.named_modules_dict[layer]
        assert (
            len(module._forward_hooks) == initial_hook_counts[layer]
        ), f"Stop hooks not removed from {layer}"

    assert res_with_stop is None
    assert res_no_stop is not None
    for layer in layers:
        assert_close(cache_with_stop[layer], cache_no_stop[layer])


def test_HookedProxyLM_to_tokens_raises_error_on_invalid_prepend_bos(
    gpt2_proxy_model: HookedProxyLM,
):
    with pytest.raises(ValueError, match="Only works with prepend_bos=False"):
        gpt2_proxy_model.to_tokens("hi", prepend_bos=True)

    with pytest.raises(ValueError, match="Only works with prepend_bos=False"):
        gpt2_proxy_model.to_tokens("hi", prepend_bos=None)


def test_HookedProxyLM_to_tokens_raises_error_on_invalid_padding_side(
    gpt2_proxy_model: HookedProxyLM,
):
    with pytest.raises(ValueError, match="Only works with padding_side=None"):
        gpt2_proxy_model.to_tokens("hi", prepend_bos=False, padding_side="left")

    with pytest.raises(ValueError, match="Only works with padding_side=None"):
        gpt2_proxy_model.to_tokens("hi", prepend_bos=False, padding_side="right")


def test_HookedProxyLM_to_tokens_raises_error_on_invalid_truncate(
    gpt2_proxy_model: HookedProxyLM,
):
    with pytest.raises(ValueError, match="Only works with truncate=False"):
        gpt2_proxy_model.to_tokens("hi", prepend_bos=False, truncate=True)


def test_HookedProxyLM_to_tokens_raises_error_on_invalid_move_to_device(
    gpt2_proxy_model: HookedProxyLM,
):
    with pytest.raises(ValueError, match="Only works with move_to_device=False"):
        gpt2_proxy_model.to_tokens(
            "hi", prepend_bos=False, truncate=False, move_to_device=True
        )


def test_HookedProxyLM_forward_raises_error_on_invalid_return_type(
    gpt2_proxy_model: HookedProxyLM,
):
    tokens = gpt2_proxy_model.to_tokens(
        "hi", prepend_bos=False, move_to_device=False, truncate=False
    )

    with pytest.raises(NotImplementedError, match="Only return_type supported is"):
        gpt2_proxy_model.forward(tokens, return_type="loss")  # type: ignore

    with pytest.raises(NotImplementedError, match="Only return_type supported is"):
        gpt2_proxy_model.forward(tokens, return_type="activations")  # type: ignore


def test_HookedProxyLM_forward_raises_error_on_stop_at_layer_with_return_both(
    gpt2_proxy_model: HookedProxyLM,
):
    tokens = gpt2_proxy_model.to_tokens(
        "hi", prepend_bos=False, move_to_device=False, truncate=False
    )

    with pytest.raises(
        NotImplementedError,
        match="stop_at_layer is not supported for return_type='both'",
    ):
        gpt2_proxy_model.forward(
            tokens,
            return_type="both",
            stop_at_layer=3,
            _names_filter=["transformer.h.0"],
        )
