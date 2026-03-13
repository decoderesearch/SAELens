import json
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from datasets import Dataset, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sae_lens import __version__
from sae_lens.config import PretokenizeRunnerConfig
from sae_lens.pretokenize_runner import (
    PretokenizeRunner,
    _tokenize_example,
    get_special_token_from_cfg,
    pretokenize_dataset,
)


@pytest.fixture
def ts_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")


def test_get_special_token_from_cfg(ts_tokenizer: PreTrainedTokenizerBase):
    assert get_special_token_from_cfg("bos", ts_tokenizer) == ts_tokenizer.bos_token_id
    assert get_special_token_from_cfg("eos", ts_tokenizer) == ts_tokenizer.eos_token_id
    assert get_special_token_from_cfg("sep", ts_tokenizer) == ts_tokenizer.sep_token_id
    assert get_special_token_from_cfg(123, ts_tokenizer) == 123
    with pytest.raises(ValueError, match="Invalid token type: invalid"):
        get_special_token_from_cfg("invalid", ts_tokenizer)  # type: ignore


def test_pretokenize_dataset_concatenates_text_until_context_size(
    ts_tokenizer: PreTrainedTokenizerBase,
):
    dataset = Dataset.from_list([{"text": "hello world"}] * 30)
    cfg = PretokenizeRunnerConfig(
        context_size=10,
        num_proc=1,
        shuffle=False,
        begin_batch_token=None,
        sequence_separator_token=None,
        begin_sequence_token=None,
    )

    tokenized_dataset = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    assert np.array(tokenized_dataset["input_ids"]).shape[1] == cfg.context_size
    assert (
        ts_tokenizer.decode(tokenized_dataset["input_ids"][0])
        == "hello worldhello worldhello worldhello worldhello world"
    )


def test_pretokenize_dataset_can_add_bos_tokens_to_the_start_of_each_batch(
    ts_tokenizer: PreTrainedTokenizerBase,
):
    dataset = Dataset.from_list([{"text": "hello world"}] * 30)
    cfg = PretokenizeRunnerConfig(
        context_size=10,
        num_proc=1,
        shuffle=False,
        begin_batch_token="bos",
        sequence_separator_token=None,
        begin_sequence_token=None,
    )

    tokenized_dataset = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    assert np.array(tokenized_dataset["input_ids"]).shape[1] == cfg.context_size
    assert (
        ts_tokenizer.decode(tokenized_dataset["input_ids"][0])
        == "<|endoftext|>hello worldhello worldhello worldhello worldhello"
    )
    for batch in tokenized_dataset["input_ids"]:
        assert ts_tokenizer.decode(batch[0]) == "<|endoftext|>"


def test_pretokenize_dataset_can_separate_sequences_with_bos(
    ts_tokenizer: PreTrainedTokenizerBase,
):
    dataset = Dataset.from_list([{"text": "hello world"}] * 30)
    cfg = PretokenizeRunnerConfig(
        context_size=10,
        num_proc=1,
        shuffle=False,
        begin_batch_token=None,
        sequence_separator_token="bos",
        begin_sequence_token=None,
    )

    tokenized_dataset = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    assert np.array(tokenized_dataset["input_ids"]).shape[1] == cfg.context_size
    assert (
        ts_tokenizer.decode(tokenized_dataset["input_ids"][0])
        == "hello world<|endoftext|>hello world<|endoftext|>hello world<|endoftext|>hello"
    )


def test_pretokenize_dataset_can_begin_sequences_with_bos(
    ts_tokenizer: PreTrainedTokenizerBase,
):
    dataset = Dataset.from_list([{"text": "hello world"}] * 30)
    cfg = PretokenizeRunnerConfig(
        context_size=10,
        num_proc=1,
        shuffle=False,
        begin_batch_token=None,
        sequence_separator_token=None,
        begin_sequence_token="bos",
    )

    tokenized_dataset = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    assert np.array(tokenized_dataset["input_ids"]).shape[1] == cfg.context_size
    assert (
        ts_tokenizer.decode(tokenized_dataset["input_ids"][0])
        == "<|endoftext|>hello world<|endoftext|>hello world<|endoftext|>hello world<|endoftext|>"
    )


def test_pretokenize_dataset_dedupes_bos(
    ts_tokenizer: PreTrainedTokenizerBase,
):
    dataset = Dataset.from_list([{"text": "hello world"}] * 30)
    cfg = PretokenizeRunnerConfig(
        context_size=10,
        num_proc=1,
        shuffle=False,
        begin_batch_token="bos",
        sequence_separator_token="bos",
        begin_sequence_token="bos",
    )

    tokenized_dataset = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    assert np.array(tokenized_dataset["input_ids"]).shape[1] == cfg.context_size
    assert (
        ts_tokenizer.decode(tokenized_dataset["input_ids"][0])
        == "<|endoftext|>hello world<|endoftext|>hello world<|endoftext|>hello world<|endoftext|>"
    )


def test_pretokenize_dataset_can_shuffle(ts_tokenizer: PreTrainedTokenizerBase):
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 5000
    )
    cfg = PretokenizeRunnerConfig(context_size=10, num_proc=1, shuffle=True)

    # assert ts_model.tokenizer is not None
    tokenized_dataset1 = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    tokenized_dataset2 = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    assert len(tokenized_dataset1) == len(tokenized_dataset2)
    assert (
        np.array(tokenized_dataset1["input_ids"]).tolist()
        != np.array(tokenized_dataset2["input_ids"]).tolist()
    )


def test_pretokenize_runner_save_dataset_locally(tmp_path: Path):
    save_path = tmp_path / "ds"
    cfg = PretokenizeRunnerConfig(
        tokenizer_name="gpt2",
        context_size=10,
        num_proc=2,
        shuffle=True,
        save_path=str(save_path),
        dataset_path="NeelNanda/c4-10k",
        split="train[:20]",
        begin_batch_token="bos",
        sequence_separator_token="eos",
    )
    dataset = PretokenizeRunner(cfg).run()
    assert save_path.exists()
    loaded_dataset = Dataset.load_from_disk(str(save_path))
    assert len(dataset) == len(loaded_dataset)
    assert (
        np.array(dataset["input_ids"]).tolist()
        == np.array(loaded_dataset["input_ids"]).tolist()
    )
    with open(save_path / "sae_lens.json") as f:
        metadata_dict = json.load(f)
    assert metadata_dict["original_dataset"] == "NeelNanda/c4-10k"
    assert metadata_dict["original_split"] == "train[:20]"
    assert metadata_dict["original_column_name"] == "text"
    assert metadata_dict["context_size"] == 10
    assert metadata_dict["shuffled"] is True
    assert metadata_dict["begin_batch_token"] == "bos"
    assert metadata_dict["begin_sequence_token"] is None
    assert metadata_dict["sequence_separator_token"] == "eos"
    assert metadata_dict["sae_lens_version"] == __version__


def test_pretokenize_runner_with_dataset_name(tmp_path: Path):
    save_path = tmp_path / "ds_with_dataset_name"
    cfg = PretokenizeRunnerConfig(
        tokenizer_name="gpt2",
        context_size=10,
        num_proc=2,
        shuffle=True,
        save_path=str(save_path),
        dataset_path="nyu-mll/glue",
        dataset_name="ax",
        split="test[:20]",
        column_name="premise",
        begin_batch_token="bos",
        sequence_separator_token="eos",
    )
    dataset = cast(Any, PretokenizeRunner(cfg).run())
    assert save_path.exists()
    loaded_dataset = Dataset.load_from_disk(str(save_path))
    assert len(dataset) == len(loaded_dataset)
    assert (
        np.array(dataset["input_ids"]).tolist()
        == np.array(loaded_dataset["input_ids"]).tolist()
    )
    with open(save_path / "sae_lens.json") as f:
        metadata_dict = json.load(f)
    assert metadata_dict["original_dataset"] == "nyu-mll/glue"
    assert metadata_dict["original_dataset_name"] == "ax"
    assert metadata_dict["original_split"] == "test[:20]"
    assert metadata_dict["original_column_name"] == "premise"
    assert metadata_dict["context_size"] == 10
    assert metadata_dict["shuffled"] is True
    assert metadata_dict["begin_batch_token"] == "bos"
    assert metadata_dict["begin_sequence_token"] is None
    assert metadata_dict["sequence_separator_token"] == "eos"
    assert metadata_dict["sae_lens_version"] == __version__


def test_pretokenize_runner_streaming_dataset():
    cfg = PretokenizeRunnerConfig(
        tokenizer_name="gpt2",
        context_size=10,
        num_proc=1,
        dataset_path="NeelNanda/c4-10k",
        split="train",
        streaming=True,
    )
    dataset = PretokenizeRunner(cfg).run()
    assert isinstance(dataset, IterableDataset)

    cfg = PretokenizeRunnerConfig(
        tokenizer_name="gpt2",
        context_size=10,
        num_proc=2,
        dataset_path="NeelNanda/c4-10k",
        split="train",
        streaming=False,
    )
    dataset = PretokenizeRunner(cfg).run()
    assert not isinstance(dataset, IterableDataset)


def test_pretokenize_runner_raises_error_when_num_proc_is_greater_than_1_and_streaming_is_true():
    cfg = PretokenizeRunnerConfig(
        tokenizer_name="gpt2",
        context_size=10,
        num_proc=2,
        dataset_path="NeelNanda/c4-10k",
        split="train",
        streaming=True,
    )
    with pytest.raises(ValueError):
        PretokenizeRunner(cfg).run()


def test_pretokenize_dataset_with_chat_formatting(
    ts_tokenizer: PreTrainedTokenizerBase,
):
    ts_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}"
        "{% endfor %}"
    )

    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    dataset = Dataset.from_list([{"conversation": conversation}] * 30)
    cfg = PretokenizeRunnerConfig(
        context_size=10,
        num_proc=1,
        shuffle=False,
        column_name="conversation",
        use_chat_formatting=True,
        begin_batch_token=None,
        sequence_separator_token=None,
        begin_sequence_token=None,
    )

    tokenized_dataset = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    assert np.array(tokenized_dataset["input_ids"]).shape[1] == cfg.context_size

    decoded = ts_tokenizer.decode(tokenized_dataset["input_ids"][0])
    assert "<|user|>" in decoded
    assert "Hello" in decoded


def test_pretokenize_dataset_with_chat_formatting_wraps_text_as_user_message(
    ts_tokenizer: PreTrainedTokenizerBase,
):
    ts_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}"
        "{% endfor %}"
    )

    dataset = Dataset.from_list([{"text": "hello world"}] * 30)
    cfg = PretokenizeRunnerConfig(
        context_size=10,
        num_proc=1,
        shuffle=False,
        column_name="text",
        use_chat_formatting=True,
        begin_batch_token=None,
        sequence_separator_token=None,
        begin_sequence_token=None,
    )

    tokenized_dataset = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    assert np.array(tokenized_dataset["input_ids"]).shape[1] == cfg.context_size

    decoded = ts_tokenizer.decode(tokenized_dataset["input_ids"][0])
    assert "<|user|>" in decoded
    assert "hello world" in decoded


def test_tokenize_example_with_conversation_does_not_warn(
    ts_tokenizer: PreTrainedTokenizerBase,
):
    ts_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}"
        "{% endfor %}"
    )

    conversation = [{"role": "user", "content": "Hello"}]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tokens = _tokenize_example(conversation, ts_tokenizer, use_chat_formatting=True)

    assert tokens.dtype == torch.long
    decoded = ts_tokenizer.decode(tokens)
    assert "<|user|>" in decoded
    assert "Hello" in decoded


def test_tokenize_example_with_string_warns(
    ts_tokenizer: PreTrainedTokenizerBase,
):
    ts_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}"
        "{% endfor %}"
    )

    with pytest.warns(
        UserWarning,
        match="use_chat_formatting is True but column contains strings",
    ):
        tokens = _tokenize_example(
            "hello world", ts_tokenizer, use_chat_formatting=True
        )

    decoded = ts_tokenizer.decode(tokens)
    assert "<|user|>" in decoded
    assert "hello world" in decoded
