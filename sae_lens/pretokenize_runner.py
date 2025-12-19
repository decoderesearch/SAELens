import io
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing_extensions import deprecated

from sae_lens import __version__
from sae_lens.config import PretokenizeRunnerConfig
from sae_lens.tokenization_and_batching import concat_and_batch_sequences


@dataclass
class PretokenizedDatasetMetadata:
    """
    This metadata will be saved along with the pretokenized dataset as a JSON file.
    """

    sae_lens_version: str
    tokenizer_name: str
    original_dataset: str
    original_dataset_name: str | None
    original_split: str | None
    original_data_files: list[str] | None
    original_column_name: str | None
    context_size: int
    shuffled: bool
    seed: int | None
    begin_batch_token: int | Literal["bos", "eos", "sep"] | None
    begin_sequence_token: int | Literal["bos", "eos", "sep"] | None
    sequence_separator_token: int | Literal["bos", "eos", "sep"] | None
    disable_concat_sequences: bool


def metadata_from_config(cfg: PretokenizeRunnerConfig) -> PretokenizedDatasetMetadata:
    return PretokenizedDatasetMetadata(
        sae_lens_version=__version__,
        tokenizer_name=cfg.tokenizer_name,
        original_dataset=cfg.dataset_path,
        original_dataset_name=cfg.dataset_name,
        original_split=cfg.split,
        original_data_files=cfg.data_files,
        original_column_name=cfg.column_name,
        context_size=cfg.context_size,
        shuffled=cfg.shuffle,
        seed=cfg.seed,
        begin_batch_token=cfg.begin_batch_token,
        begin_sequence_token=cfg.begin_sequence_token,
        sequence_separator_token=cfg.sequence_separator_token,
        disable_concat_sequences=cfg.disable_concat_sequences,
    )


def get_special_token_from_cfg(
    cfg_token: int | Literal["bos", "eos", "sep"] | None,
    tokenizer: PreTrainedTokenizerBase,
) -> int | None:
    if cfg_token is None:
        return None
    if isinstance(cfg_token, int):
        return cfg_token
    if cfg_token == "bos":
        return tokenizer.bos_token_id  # type: ignore
    if cfg_token == "eos":
        return tokenizer.eos_token_id  # type: ignore
    if cfg_token == "sep":
        return tokenizer.sep_token_id  # type: ignore
    raise ValueError(f"Invalid token type: {cfg_token}")


def pretokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PretokenizeRunnerConfig,
):
    def process_examples(examples: dict[str, list[str]]):
        tokens_iterator = cast(
            Iterator[torch.Tensor],
            (
                tokenizer.encode(text, return_tensors="pt")[0]
                for text in examples[cfg.column_name]
            ),
        )
        return {
            "input_ids": list(
                concat_and_batch_sequences(
                    tokens_iterator=tokens_iterator,
                    context_size=cfg.context_size,
                    begin_batch_token_id=get_special_token_from_cfg(
                        cfg.begin_batch_token, tokenizer
                    ),
                    begin_sequence_token_id=get_special_token_from_cfg(
                        cfg.begin_sequence_token, tokenizer
                    ),
                    sequence_separator_token_id=get_special_token_from_cfg(
                        cfg.sequence_separator_token, tokenizer
                    ),
                    disable_concat_sequences=cfg.disable_concat_sequences,
                )
            )
        }

    if cfg.streaming:
        if cfg.num_proc > 1:
            raise ValueError("num_proc must be 1 when streaming is True")
        tokenized_dataset = dataset.map(
            process_examples,
            batched=True,
            batch_size=cfg.pretokenize_batch_size,
            remove_columns=dataset.column_names,
        )
    else:
        tokenized_dataset = dataset.map(
            process_examples,
            batched=True,
            batch_size=cfg.pretokenize_batch_size,
            num_proc=cfg.num_proc,
            remove_columns=dataset.column_names,
        )

    if cfg.shuffle:
        tokenized_dataset = tokenized_dataset.shuffle(seed=cfg.seed)

    if cfg.streaming:
        tokenized_dataset = tokenized_dataset.with_format(type="torch")
    else:
        tokenized_dataset.set_format(type="torch", columns=["input_ids"])

    return tokenized_dataset


def push_to_hugging_face_hub(
    dataset: Dataset,
    cfg: PretokenizeRunnerConfig,
):
    assert cfg.hf_repo_id is not None
    dataset.push_to_hub(
        repo_id=cfg.hf_repo_id,
        num_shards=cfg.hf_num_shards,
        private=cfg.hf_is_private_repo,
        revision=cfg.hf_revision,
    )
    # also upload metadata file
    metadata = metadata_from_config(cfg)
    meta_io = io.BytesIO()
    meta_contents = json.dumps(metadata.__dict__, indent=2, ensure_ascii=False).encode(
        "utf-8"
    )
    meta_io.write(meta_contents)
    meta_io.seek(0)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=meta_io,
        path_in_repo="sae_lens.json",
        repo_id=cfg.hf_repo_id,
        repo_type="dataset",
        commit_message="Add sae_lens metadata",
    )


@deprecated("Use PretokenizeRunner instead")
def pretokenize_runner(
    cfg: PretokenizeRunnerConfig,
):
    runner = PretokenizeRunner(cfg)
    return runner.run()


class PretokenizeRunner:
    """
    Runner to pretokenize a dataset using a given tokenizer, and optionally upload to Huggingface.
    """

    def __init__(self, cfg: PretokenizeRunnerConfig):
        self.cfg = cfg

    def run(self):
        """
        Load the dataset, tokenize it, and save it to disk and/or upload to Huggingface.
        """
        dataset = load_dataset(  # type: ignore
            self.cfg.dataset_path,
            name=self.cfg.dataset_name,
            data_dir=self.cfg.data_dir,
            data_files=self.cfg.data_files,
            split=self.cfg.split,  # type: ignore
            streaming=self.cfg.streaming,  # type: ignore
        )
        if isinstance(dataset, DatasetDict):
            raise ValueError(
                "Dataset has multiple splits. Must provide a 'split' param."
            )
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
        tokenizer.model_max_length = sys.maxsize
        tokenized_dataset = pretokenize_dataset(
            cast(Dataset, dataset), tokenizer, self.cfg
        )

        if self.cfg.save_path is not None:
            tokenized_dataset.save_to_disk(self.cfg.save_path)
            metadata = metadata_from_config(self.cfg)
            metadata_path = Path(self.cfg.save_path) / "sae_lens.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata.__dict__, f, indent=2, ensure_ascii=False)

        if self.cfg.hf_repo_id is not None:
            push_to_hugging_face_hub(tokenized_dataset, self.cfg)

        return tokenized_dataset
