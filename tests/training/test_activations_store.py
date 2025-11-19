import os
import tempfile
from collections.abc import Iterable
from math import ceil
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from sae_lens.config import LanguageModelSAERunnerConfig, PretokenizeRunnerConfig
from sae_lens.load_model import load_model
from sae_lens.pretokenize_runner import pretokenize_dataset
from sae_lens.saes.sae import SAE
from sae_lens.saes.standard_sae import StandardSAEConfig, StandardTrainingSAEConfig
from sae_lens.training.activations_store import (
    ActivationsStore,
    _filter_buffer_acts,
    permute_together,
    validate_pretokenized_dataset_tokenizer,
)
from tests.helpers import (
    NEEL_NANDA_C4_10K_DATASET,
    assert_close,
    assert_not_close,
    build_runner_cfg,
    load_model_cached,
)


def tokenize_with_bos(model: HookedTransformer, text: str) -> list[int]:
    assert model.tokenizer is not None
    assert model.tokenizer.bos_token_id is not None
    return [model.tokenizer.bos_token_id] + model.tokenizer.encode(text)  # type: ignore


# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": NEEL_NANDA_C4_10K_DATASET,
            "hook_name": "blocks.1.hook_resid_pre",
            "d_in": 64,
            "normalize_activations": "expected_average_only_in",
            "streaming": False,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": NEEL_NANDA_C4_10K_DATASET,
            "hook_name": "blocks.1.attn.hook_z",
            "d_in": 64,
            "streaming": False,
        },
        {
            "model_name": "gpt2",
            "dataset_path": "apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
            "hook_name": "blocks.1.hook_resid_pre",
            "d_in": 768,
            "context_size": 1024,
            "streaming": True,
        },
        {
            "model_name": "gpt2",
            "dataset_path": NEEL_NANDA_C4_10K_DATASET,
            "hook_name": "blocks.1.hook_resid_pre",
            "d_in": 768,
            "exclude_special_tokens": True,
            "streaming": False,
        },
    ],
    ids=[
        "c4-10k-resid-pre",
        "c4-10k-attn-out",
        "gpt2-tokenized",
        "gpt2",
    ],
)
def cfg(
    request: pytest.FixtureRequest,
) -> LanguageModelSAERunnerConfig[StandardTrainingSAEConfig]:
    # This function will be called with each parameter set
    params = request.param
    return build_runner_cfg(**params)


@pytest.fixture
def model(cfg: LanguageModelSAERunnerConfig[StandardTrainingSAEConfig]):
    return load_model_cached(cfg.model_name)


# tests involving loading real models / real datasets are very slow
# so do lots of stuff in this one test to make each load of model / data count
# poetry run py.test tests/training/test_activations_store.py -k 'test_activations_store__shapes_look_correct_with_real_models_and_datasets' --profile-svg -s
def test_activations_store__shapes_look_correct_with_real_models_and_datasets(
    cfg: LanguageModelSAERunnerConfig[StandardTrainingSAEConfig],
    model: HookedTransformer,
):
    # --- first, test initialisation ---

    # config if you want to benchmark this:
    #
    # cfg.context_size = 1024
    # cfg.n_batches_in_buffer = 64
    # cfg.store_batch_size_prompts = 16

    store = ActivationsStore.from_config(model, cfg)

    assert store.model == model

    assert isinstance(store.iterable_sequences, Iterable)

    # --- Next, get batch tokens and assert they look correct ---

    batch = store.get_batch_tokens()

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (
        store.store_batch_size_prompts,
        store.context_size,
    )
    assert batch.device == store.device

    # --- Next, get activations and assert they look correct ---

    activations = store.get_activations(batch)

    assert isinstance(activations, torch.Tensor)
    assert activations.shape == (
        store.store_batch_size_prompts,
        store.context_size,
        store.d_in,
    )
    assert activations.device == store.device

    # --- Next, get buffer and assert it looks correct ---

    n_batches_in_buffer = 3
    act_buffer, tok_buffer = store.get_raw_buffer(n_batches_in_buffer)

    assert isinstance(act_buffer, torch.Tensor)
    assert isinstance(tok_buffer, torch.Tensor)
    buffer_size_expected = (
        store.store_batch_size_prompts * store.context_size * n_batches_in_buffer
    )

    assert act_buffer.shape == (buffer_size_expected, store.d_in)
    assert tok_buffer.shape == (buffer_size_expected,)
    assert act_buffer.device == store.device
    assert tok_buffer.device == store.device


def test_activations_store__get_activations_head_hook(ts_model: HookedTransformer):
    cfg = build_runner_cfg(
        hook_name="blocks.0.attn.hook_q",
        hook_head_index=2,
        d_in=4,
    )
    activation_store_head_hook = ActivationsStore.from_config(ts_model, cfg)
    batch = activation_store_head_hook.get_batch_tokens()
    activations = activation_store_head_hook.get_activations(batch)

    assert isinstance(activations, torch.Tensor)
    assert activations.shape == (
        activation_store_head_hook.store_batch_size_prompts,
        activation_store_head_hook.context_size,
        activation_store_head_hook.d_in,
    )
    assert activations.device == activation_store_head_hook.device


def test_activations_store__get_activations__gives_same_results_with_hf_model_and_tlens_model():
    hf_model = load_model(
        model_class_name="AutoModelForCausalLM",
        model_name="gpt2",
        device="cpu",
    )
    tlens_model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")
    dataset = Dataset.from_list(
        [
            {"text": "hello world"},
        ]
        * 100
    )

    cfg = build_runner_cfg(hook_name="blocks.4.hook_resid_post", d_in=768)
    store_tlens = ActivationsStore.from_config(
        tlens_model, cfg, override_dataset=dataset
    )
    batch_tlens = store_tlens.get_batch_tokens()
    activations_tlens = store_tlens.get_activations(batch_tlens)

    cfg = build_runner_cfg(hook_name="transformer.h.4", d_in=768)
    store_hf = ActivationsStore.from_config(hf_model, cfg, override_dataset=dataset)
    batch_hf = store_hf.get_batch_tokens()
    activations_hf = store_hf.get_activations(batch_hf)

    assert_close(activations_hf, activations_tlens, atol=1e-3)


# 12 is divisible by the length of "hello world", 11 and 13 are not
@pytest.mark.parametrize("context_size", [11, 12, 13])
def test_activations_store__get_batch_tokens__fills_the_context_separated_by_bos(
    ts_model: HookedTransformer, context_size: int
):
    assert ts_model.tokenizer is not None
    dataset = Dataset.from_list(
        [
            {"text": "hello world"},
        ]
        * 100
    )
    cfg = build_runner_cfg(
        store_batch_size_prompts=2,
        context_size=context_size,
    )

    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    encoded_text = tokenize_with_bos(ts_model, "hello world")
    tokens = activation_store.get_batch_tokens()
    assert tokens.shape == (2, context_size)  # batch_size x context_size
    all_expected_tokens = (encoded_text * ceil(2 * context_size / len(encoded_text)))[
        : 2 * context_size
    ]
    expected_tokens1 = all_expected_tokens[:context_size]
    expected_tokens2 = all_expected_tokens[context_size:]
    if expected_tokens2[0] != ts_model.tokenizer.bos_token_id:
        expected_tokens2 = [ts_model.tokenizer.bos_token_id] + expected_tokens2[:-1]
    assert tokens[0].tolist() == expected_tokens1
    assert tokens[1].tolist() == expected_tokens2


def test_activations_store__iterate_raw_dataset_tokens__tokenizes_each_example_in_order(
    ts_model: HookedTransformer,
):
    tokenizer = ts_model.tokenizer
    assert tokenizer is not None
    cfg = build_runner_cfg()
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
    )
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    iterator = activation_store._iterate_raw_dataset_tokens()

    assert next(iterator).tolist() == tokenizer.encode("hello world1")
    assert next(iterator).tolist() == tokenizer.encode("hello world2")
    assert next(iterator).tolist() == tokenizer.encode("hello world3")


def test_activations_store__iterate_raw_dataset_tokens__can_handle_long_examples(
    ts_model: HookedTransformer,
):
    cfg = build_runner_cfg()
    dataset = Dataset.from_list(
        [
            {"text": " France" * 3000},
        ]
    )
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    iterator = activation_store._iterate_raw_dataset_tokens()

    assert len(next(iterator).tolist()) == 3000


def test_activations_store_goes_to_cpu(ts_model: HookedTransformer):
    cfg = build_runner_cfg(act_store_device="cpu")
    activation_store = ActivationsStore.from_config(ts_model, cfg)
    activations = activation_store.next_batch()
    assert activations[0].device == torch.device("cpu")
    assert activations[1] is not None
    assert activations[1].device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU to test on.")
def test_activations_store_with_model_on_gpu(ts_model: HookedTransformer):
    cfg = build_runner_cfg(act_store_device="cpu", device="cuda:0")
    activation_store = ActivationsStore.from_config(ts_model.to("cuda:0"), cfg)  # type: ignore
    activations = activation_store.next_batch()
    assert activations[0].device == torch.device("cpu")
    assert activations[1] is not None
    assert activations[1].device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU to test on.")
def test_activations_store_moves_with_model(ts_model: HookedTransformer):
    # "with_model" resets to default so the second post_init in build_sae_cfg works
    cfg = build_runner_cfg(act_store_device="with_model", device="cuda:0")
    activation_store = ActivationsStore.from_config(ts_model.to("cuda:0"), cfg)  # type: ignore
    activations = activation_store.next_batch()
    assert activations[0].device == torch.device("cpu")
    assert activations[1] is not None
    assert activations[1].device == torch.device("cpu")


def test_activations_store___iterate_tokenized_sequences__yields_concat_and_batched_sequences(
    ts_model: HookedTransformer,
):
    tokenizer = ts_model.tokenizer
    assert tokenizer is not None
    cfg = build_runner_cfg(prepend_bos=True, context_size=5)
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
    )
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    iterator = activation_store._iterate_tokenized_sequences()

    expected = [
        tokenizer.bos_token_id,
        *tokenizer.encode("hello world1"),
        tokenizer.bos_token_id,
        *tokenizer.encode("hello world2"),
        tokenizer.bos_token_id,
        *tokenizer.encode("hello world3"),
    ]
    assert next(iterator).tolist() == expected[:5]


def test_activations_store___iterate_tokenized_sequences__yields_sequences_of_context_size(
    ts_model: HookedTransformer,
):
    tokenizer = ts_model.tokenizer
    assert tokenizer is not None
    cfg = build_runner_cfg(prepend_bos=True, context_size=5)
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 20
    )
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    for toks in activation_store._iterate_tokenized_sequences():
        assert toks.shape == (5,)


def test_activations_store___iterate_tokenized_sequences__works_with_huggingface_models():
    hf_model = load_model(
        model_class_name="AutoModelForCausalLM",
        model_name="gpt2",
        device="cpu",
    )
    cfg = build_runner_cfg(prepend_bos=True, context_size=5)
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 20
    )
    activation_store = ActivationsStore.from_config(
        hf_model, cfg, override_dataset=dataset
    )
    for toks in activation_store._iterate_tokenized_sequences():
        assert toks.shape == (5,)


# We expect the code to work for context_size being less than or equal to the
# length of the dataset
@pytest.mark.parametrize(
    "context_size, expected_error",
    [(5, RuntimeWarning), (10, None), (15, ValueError)],
)
def test_activations_store__errors_on_context_size_mismatch(
    ts_model: HookedTransformer, context_size: int, expected_error: ValueError | None
):
    tokenizer = ts_model.tokenizer
    assert tokenizer is not None
    cfg = build_runner_cfg(prepend_bos=True, context_size=context_size)
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 20
    )
    pretokenize_cfg = PretokenizeRunnerConfig(context_size=10)
    tokenized_dataset = pretokenize_dataset(dataset, tokenizer, cfg=pretokenize_cfg)

    # This context_size should raise an error or a warning if it mismatches the dataset size
    if expected_error is ValueError:
        with pytest.raises(expected_error):
            ActivationsStore.from_config(
                ts_model, cfg, override_dataset=tokenized_dataset
            )
    elif expected_error is RuntimeWarning:
        # If the context_size is smaller than the dataset size we should output a RuntimeWarning
        with pytest.warns(expected_error):
            ActivationsStore.from_config(
                ts_model, cfg, override_dataset=tokenized_dataset
            )
    else:
        # If the context_size is equal to the dataset size the function should pass
        ActivationsStore.from_config(ts_model, cfg, override_dataset=tokenized_dataset)


def test_activations_store__errors_on_negative_context_size():
    with pytest.raises(ValueError):
        # We should raise an error when the context_size is negative
        build_runner_cfg(prepend_bos=True, context_size=-1)


def test_activations_store___iterate_tokenized_sequences__yields_identical_results_with_and_without_pretokenizing(
    ts_model: HookedTransformer,
):
    tokenizer = ts_model.tokenizer
    assert tokenizer is not None
    cfg = build_runner_cfg(prepend_bos=True, context_size=5)
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 20
    )
    pretokenize_cfg = PretokenizeRunnerConfig(
        context_size=5,
        num_proc=1,
        shuffle=False,
        begin_batch_token="bos",
        sequence_separator_token="bos",
    )
    tokenized_dataset = pretokenize_dataset(dataset, tokenizer, cfg=pretokenize_cfg)
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    tokenized_activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=tokenized_dataset
    )
    seqs = [seq.tolist() for seq in activation_store._iterate_tokenized_sequences()]
    pretok_seqs = [
        seq.tolist()
        for seq in tokenized_activation_store._iterate_tokenized_sequences()
    ]
    assert seqs == pretok_seqs


def test_activation_store__errors_if_neither_dataset_nor_dataset_path(
    ts_model: HookedTransformer,
):
    cfg = build_runner_cfg(dataset_path="")

    example_ds = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 20
    )

    ActivationsStore.from_config(ts_model, cfg, override_dataset=example_ds)

    with pytest.raises(ValueError):
        ActivationsStore.from_config(ts_model, cfg, override_dataset=None)


def test_validate_pretokenized_dataset_tokenizer_errors_if_the_tokenizer_doesnt_match_the_model():
    ds_path = "chanind/openwebtext-gpt2"
    model_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    assert model_tokenizer is not None
    with pytest.raises(ValueError):
        validate_pretokenized_dataset_tokenizer(ds_path, model_tokenizer)


def test_validate_pretokenized_dataset_tokenizer_runs_successfully_if_tokenizers_match(
    ts_model: HookedTransformer,
):
    ds_path = "chanind/openwebtext-gpt2"
    model_tokenizer = ts_model.tokenizer
    assert model_tokenizer is not None
    validate_pretokenized_dataset_tokenizer(ds_path, model_tokenizer)


def test_validate_pretokenized_dataset_tokenizer_does_nothing_if_the_dataset_is_not_created_by_sae_lens(
    ts_model: HookedTransformer,
):
    ds_path = "apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2"
    model_tokenizer = ts_model.tokenizer
    assert model_tokenizer is not None
    validate_pretokenized_dataset_tokenizer(ds_path, model_tokenizer)


def test_validate_pretokenized_dataset_tokenizer_does_nothing_if_the_dataset_path_doesnt_exist(
    ts_model: HookedTransformer,
):
    ds_path = "blah/nonsense-1234"
    model_tokenizer = ts_model.tokenizer
    assert model_tokenizer is not None
    validate_pretokenized_dataset_tokenizer(ds_path, model_tokenizer)


def test_activations_store_respects_position_offsets(ts_model: HookedTransformer):
    cfg = build_runner_cfg(
        context_size=10,
        seqpos_slice=(2, 8),  # Only consider positions 2 to 7 (inclusive)
    )
    dataset = Dataset.from_list(
        [
            {"text": "This is a test sentence for slicing."},
        ]
        * 100
    )

    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )

    batch = activation_store.get_batch_tokens(1)
    activations = activation_store.get_activations(batch)

    assert batch.shape == (1, 10)  # Full context size
    assert activations.shape == (1, 6, cfg.sae.d_in)  # Only 6 positions (2 to 7)


def test_activations_store_save(ts_model: HookedTransformer):
    cfg = build_runner_cfg()
    activation_store = ActivationsStore.from_config(ts_model, cfg)
    with tempfile.NamedTemporaryFile() as temp_file:
        activation_store.save(temp_file.name)
        assert os.path.exists(temp_file.name)
        state_dict = load_file(temp_file.name)
        assert isinstance(state_dict, dict)
        assert "n_dataset_processed" in state_dict


def test_activations_store_save_and_restore_from_checkpoint(
    ts_model: HookedTransformer,
    tmp_path: Path,
):
    cfg = build_runner_cfg(disable_concat_sequences=True)
    activation_store = ActivationsStore.from_config(ts_model, cfg)

    for _ in range(300):
        activation_store.next_batch()
    assert activation_store.n_dataset_processed > 0
    activation_store.save_to_checkpoint(tmp_path)

    new_activation_store = ActivationsStore.from_config(ts_model, cfg)
    new_activation_store.load_from_checkpoint(tmp_path)
    assert (
        new_activation_store.n_dataset_processed == activation_store.n_dataset_processed
    )
    next_tokens = next(activation_store.iterable_sequences)
    next_tokens_new = next(new_activation_store.iterable_sequences)
    assert torch.allclose(next_tokens, next_tokens_new)


def test_activations_store_buffer_contains_token_ids(ts_model: HookedTransformer):
    """Test that the buffer contains both activations and token IDs."""
    cfg = build_runner_cfg(context_size=3, store_batch_size_prompts=5)
    dataset = Dataset.from_list([{"text": "hello world"}] * 100)

    store = ActivationsStore.from_config(ts_model, cfg, override_dataset=dataset)
    acts, token_ids = store.get_raw_buffer(n_batches_in_buffer=2)

    assert acts.shape == (30, 64)  # (batch_size x context_size x n_batches, d_in)
    assert token_ids is not None
    assert token_ids.shape == (30,)  # (batch_size x context_size x n_batches,)

    expected_tokens = set(ts_model.to_tokens("hello world").squeeze().tolist())  # type: ignore
    assert set(token_ids.tolist()) == expected_tokens


def test_activations_store_buffer_shuffling(ts_model: HookedTransformer):
    """Test that buffer shuffling maintains alignment between acts and token_ids."""
    cfg = build_runner_cfg()
    dataset = Dataset.from_list([{"text": "hello world"}] * 100)

    # Get unshuffled buffer
    store = ActivationsStore.from_config(ts_model, cfg, override_dataset=dataset)
    acts_unshuffled_1, token_ids_unshuffled_1 = store.get_raw_buffer(
        n_batches_in_buffer=2, shuffle=False
    )

    store = ActivationsStore.from_config(ts_model, cfg, override_dataset=dataset)
    acts_unshuffled_2, token_ids_unshuffled_2 = store.get_raw_buffer(
        n_batches_in_buffer=2, shuffle=False
    )

    # Get shuffled buffer
    store = ActivationsStore.from_config(ts_model, cfg, override_dataset=dataset)
    acts_shuffled, token_ids_shuffled = store.get_raw_buffer(
        n_batches_in_buffer=2, shuffle=True
    )

    assert token_ids_unshuffled_1 is not None
    assert token_ids_unshuffled_2 is not None
    assert token_ids_shuffled is not None

    assert_close(acts_unshuffled_1, acts_unshuffled_2)
    assert_close(token_ids_unshuffled_1, token_ids_unshuffled_2)
    assert_not_close(acts_unshuffled_1, acts_shuffled)
    assert_not_close(token_ids_unshuffled_1, token_ids_shuffled)

    assert set(token_ids_shuffled.tolist()) == set(token_ids_unshuffled_1.tolist())


@torch.no_grad()
def test_activations_store_storage_buffer_excludes_special_tokens(
    ts_model: HookedTransformer,
):
    hook_name = "blocks.0.hook_resid_post"
    base_cfg = build_runner_cfg(
        exclude_special_tokens=False,
        context_size=5,
        store_batch_size_prompts=2,
        hook_name=hook_name,
    )
    cfg = build_runner_cfg(
        exclude_special_tokens=True,
        context_size=5,
        store_batch_size_prompts=2,
        hook_name=hook_name,
    )
    dataset = Dataset.from_list([{"text": "hello world"}] * 100)
    _, cache = ts_model.run_with_cache(dataset[0]["text"])
    bos_act = cache[hook_name][0, 0]
    store_base = ActivationsStore.from_config(
        ts_model, base_cfg, override_dataset=dataset
    )
    store_exclude_special_tokens = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    store_base_it = store_base._iterate_filtered_activations()
    store_exclude_special_tokens_it = (
        store_exclude_special_tokens._iterate_filtered_activations()
    )

    assert next(store_base_it).shape[0] == 10
    assert next(store_exclude_special_tokens_it).shape[0] < 10

    # bos act should be in the base buffer, but not in the exclude special tokens buffer
    assert (next(store_base_it).squeeze() - bos_act).abs().sum(
        dim=-1
    ).min().item() == pytest.approx(0.0, abs=1e-5)
    assert (next(store_exclude_special_tokens_it).squeeze() - bos_act).abs().sum(
        dim=-1
    ).min().item() != pytest.approx(0.0, abs=1e-5)


@torch.no_grad()
def test_activations_next_batch_excludes_special_tokens(
    ts_model: HookedTransformer,
):
    hook_name = "blocks.0.hook_resid_post"
    base_cfg = build_runner_cfg(
        exclude_special_tokens=False,
        context_size=5,
        store_batch_size_prompts=2,
        hook_name=hook_name,
        train_batch_size_tokens=5,
    )
    cfg = build_runner_cfg(
        exclude_special_tokens=True,
        context_size=5,
        store_batch_size_prompts=2,
        hook_name=hook_name,
        train_batch_size_tokens=5,
    )
    dataset = Dataset.from_list([{"text": "hello world"}] * 100)
    _, cache = ts_model.run_with_cache(dataset[0]["text"])
    bos_act = cache[hook_name][0, 0]
    store_base = ActivationsStore.from_config(
        ts_model, base_cfg, override_dataset=dataset
    )
    store_exclude_special_tokens = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    batch_base = store_base.next_batch()
    batch_exclude_special_tokens = store_exclude_special_tokens.next_batch()
    assert batch_base.shape[0] == 5
    assert batch_exclude_special_tokens.shape[0] == 5

    # bos act should be in the base batch, but not in the exclude special tokens batch
    assert (batch_base.squeeze() - bos_act).abs().sum(
        dim=-1
    ).min().item() == pytest.approx(0.0, abs=1e-5)
    assert (batch_exclude_special_tokens.squeeze() - bos_act).abs().sum(
        dim=-1
    ).min().item() != pytest.approx(0.0, abs=1e-5)


def test_permute_together():
    """Test that permute_together correctly permutes tensors together."""
    # Create test tensors
    t1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    t2 = torch.tensor([10, 20, 30])
    t3 = torch.tensor([[100], [200], [300]])

    # Permute them together
    p1, p2, p3 = permute_together([t1, t2, t3])

    # Verify shapes are preserved
    assert p1.shape == t1.shape
    assert p2.shape == t2.shape
    assert p3.shape == t3.shape

    # Find the permutation that was applied by looking at t2
    perm = torch.zeros_like(t2, dtype=torch.long)
    for i in range(len(t2)):
        perm[i] = torch.where(p2 == t2[i])[0]

    # Verify all tensors used the same permutation
    for i in range(len(t2)):
        assert_close(p1[i], t1[perm[i]])
        assert_close(p2[i], t2[perm[i]])
        assert_close(p3[i], t3[perm[i]])


def test_permute_together_different_sizes_raises():
    """Test that permute_together raises an error if tensors have different first dimensions."""
    t1 = torch.tensor([[1, 2], [3, 4], [5, 6]])  # Shape (3, 2)
    t2 = torch.tensor([10, 20])  # Shape (2,)

    with pytest.raises(IndexError):
        permute_together([t1, t2])


def test_filter_buffer_acts_no_filtering():
    """Test that _filter_buffer_acts returns original activations when no filtering needed."""
    activations = torch.randn(10, 5)  # 10 tokens, 5 features
    tokens = None
    exclude_tokens = None

    filtered = _filter_buffer_acts((activations, tokens), exclude_tokens)

    assert_close(filtered, activations)


def test_filter_buffer_acts_with_filtering():
    """Test that _filter_buffer_acts correctly filters out specified tokens."""
    activations = torch.tensor(
        [
            [1.0, 2.0],  # token 0
            [3.0, 4.0],  # token 1
            [5.0, 6.0],  # token 2
            [7.0, 8.0],  # token 3
        ]
    )
    tokens = torch.tensor([0, 1, 0, 2])
    exclude_tokens = torch.tensor([0, 2])  # Filter out tokens 0 and 2

    filtered = _filter_buffer_acts((activations, tokens), exclude_tokens)

    expected = torch.tensor([[3.0, 4.0]])  # Only token 1 remains
    assert_close(filtered, expected)


def test_filter_buffer_acts_no_matches():
    """Test that _filter_buffer_acts handles case where no tokens match exclusion list."""
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tokens = torch.tensor([0, 1])
    exclude_tokens = torch.tensor([2, 3])  # No matches

    filtered = _filter_buffer_acts((activations, tokens), exclude_tokens)

    assert_close(filtered, activations)  # All tokens kept


def test_filter_buffer_acts_all_filtered():
    """Test that _filter_buffer_acts handles case where all tokens are filtered."""
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tokens = torch.tensor([0, 0])
    exclude_tokens = torch.tensor([0])  # All tokens filtered

    filtered = _filter_buffer_acts((activations, tokens), exclude_tokens)

    assert filtered.shape[0] == 0  # Empty tensor returned
    assert filtered.shape[1] == activations.shape[1]  # Feature dimension preserved


def test_activations_store_get_batch_tokens_disable_concat_sequences(
    ts_model: HookedTransformer,
):
    cfg = build_runner_cfg(
        context_size=5,
        disable_concat_sequences=True,
        store_batch_size_prompts=2,
        n_batches_in_buffer=2,
    )

    dataset = Dataset.from_list(
        [
            {"text": "short"},  # this gets ignored
            {"text": "hello world this is long enough"},
            {"text": "another longer sequence for testing"},
        ]
    )

    activation_store = ActivationsStore.from_config(
        ts_model, cfg=cfg, override_dataset=dataset
    )

    batch_tokens = activation_store.get_batch_tokens()
    assert batch_tokens.shape == (2, cfg.context_size)

    tokenizer = ts_model.tokenizer
    # get pyright checks to pass
    assert tokenizer is not None

    # Since prepend_bos=True (the default), we expect BOS token at the start
    bos_token_id = tokenizer.bos_token_id
    assert bos_token_id is not None

    expected_tokens_1 = [bos_token_id] + tokenizer.encode(
        "hello world this is long enough"
    )[: cfg.context_size - 1]
    expected_tokens_2 = [bos_token_id] + tokenizer.encode(
        "another longer sequence for testing"
    )[: cfg.context_size - 1]

    assert batch_tokens[0].tolist() == expected_tokens_1
    assert batch_tokens[1].tolist() == expected_tokens_2


def test_activations_store_get_batch_tokens_disable_concat_sequences_no_bos(
    ts_model: HookedTransformer,
):
    """Test disable_concat_sequences with prepend_bos=False"""
    cfg = build_runner_cfg(
        context_size=5,
        disable_concat_sequences=True,
        prepend_bos=False,  # Explicitly disable BOS
        store_batch_size_prompts=2,
        n_batches_in_buffer=2,
    )

    dataset = Dataset.from_list(
        [
            {"text": "short"},  # this gets ignored
            {"text": "hello world this is long enough"},
            {"text": "another longer sequence for testing"},
        ]
    )

    activation_store = ActivationsStore.from_config(
        ts_model, cfg=cfg, override_dataset=dataset
    )

    batch_tokens = activation_store.get_batch_tokens()
    assert batch_tokens.shape == (2, cfg.context_size)

    tokenizer = ts_model.tokenizer
    assert tokenizer is not None

    # With prepend_bos=False, should NOT have BOS token
    expected_tokens_1 = tokenizer.encode("hello world this is long enough")[
        : cfg.context_size
    ]
    expected_tokens_2 = tokenizer.encode("another longer sequence for testing")[
        : cfg.context_size
    ]

    assert batch_tokens[0].tolist() == expected_tokens_1
    assert batch_tokens[1].tolist() == expected_tokens_2


def test_activations_store_get_batch_tokens_no_sequence_separator_token(
    ts_model: HookedTransformer,
):
    cfg = build_runner_cfg(
        sequence_separator_token=None,
        context_size=8,
        store_batch_size_prompts=1,
    )

    dataset = Dataset.from_list([{"text": "hi"}, {"text": "bye"}] * 10)

    activations_store = ActivationsStore.from_config(
        ts_model, cfg=cfg, override_dataset=dataset
    )
    batch_tokens = activations_store.get_batch_tokens()

    tokenizer = ts_model.tokenizer
    # get pyright checks to pass
    assert tokenizer is not None

    encoded_text = tokenizer.encode("hi")

    # there's no BOS between sequences, where it would usually be
    assert batch_tokens[0, 1 + len(encoded_text)] != tokenizer.bos_token_id


def test_activations_store_from_sae_defaults_to_context_size_from_sae_config(
    ts_model: HookedTransformer, gpt2_res_jb_l4_sae: SAE[StandardSAEConfig]
):
    gpt2_res_jb_l4_sae.cfg.metadata.context_size = 1234
    store = ActivationsStore.from_sae(
        model=ts_model,
        sae=gpt2_res_jb_l4_sae,
        dataset="NeelNanda/c4-10k",
    )
    assert store.context_size == 1234


def test_activations_store_from_sae_allows_null_context_size_with_override(
    ts_model: HookedTransformer, gpt2_res_jb_l4_sae: SAE[StandardSAEConfig]
):
    gpt2_res_jb_l4_sae.cfg.metadata.context_size = None
    custom_context_size = 256

    store = ActivationsStore.from_sae(
        model=ts_model,
        sae=gpt2_res_jb_l4_sae,
        dataset="NeelNanda/c4-10k",
        context_size=custom_context_size,
    )

    assert store.context_size == custom_context_size
