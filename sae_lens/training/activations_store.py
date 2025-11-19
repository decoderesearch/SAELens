from __future__ import annotations

import json
import os
import warnings
from collections.abc import Generator, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import datasets
import torch
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from requests import HTTPError
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
from transformer_lens.hook_points import HookedRootModule
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sae_lens import logger
from sae_lens.config import (
    CacheActivationsRunnerConfig,
    HfDataset,
    LanguageModelSAERunnerConfig,
)
from sae_lens.constants import ACTIVATIONS_STORE_STATE_FILENAME, DTYPE_MAP
from sae_lens.pretokenize_runner import get_special_token_from_cfg
from sae_lens.saes.sae import SAE, T_SAE_CONFIG, T_TRAINING_SAE_CONFIG
from sae_lens.tokenization_and_batching import concat_and_batch_sequences
from sae_lens.training.mixing_buffer import mixing_buffer
from sae_lens.util import (
    extract_stop_at_layer_from_tlens_hook_name,
    get_special_token_ids,
)


# TODO: Make an activation store config class to be consistent with the rest of the code.
class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    model: HookedRootModule
    dataset: HfDataset
    cached_activations_path: str | None
    cached_activation_dataset: Dataset | None = None
    tokens_column: Literal["tokens", "input_ids", "text", "problem"]
    hook_name: str
    hook_head_index: int | None
    _dataloader: Iterator[Any] | None = None
    exclude_special_tokens: torch.Tensor | None = None
    device: torch.device

    @classmethod
    def from_cache_activations(
        cls,
        model: HookedRootModule,
        cfg: CacheActivationsRunnerConfig,
    ) -> ActivationsStore:
        """
        Public api to create an ActivationsStore from a cached activations dataset.
        """
        return cls(
            cached_activations_path=cfg.new_cached_activations_path,
            dtype=cfg.dtype,
            hook_name=cfg.hook_name,
            context_size=cfg.context_size,
            d_in=cfg.d_in,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size_prompts=cfg.model_batch_size,  # get_buffer
            train_batch_size_tokens=cfg.model_batch_size,  # dataloader
            seqpos_slice=(None,),
            device=torch.device(cfg.device),  # since we're sending these to SAE
            # NOOP
            prepend_bos=False,
            hook_head_index=None,
            dataset=cfg.dataset_path,
            streaming=False,
            model=model,
            normalize_activations="none",
            model_kwargs=None,
            autocast_lm=False,
            dataset_trust_remote_code=None,
            exclude_special_tokens=None,
        )

    @classmethod
    def from_config(
        cls,
        model: HookedRootModule,
        cfg: LanguageModelSAERunnerConfig[T_TRAINING_SAE_CONFIG]
        | CacheActivationsRunnerConfig,
        override_dataset: HfDataset | None = None,
    ) -> ActivationsStore:
        if isinstance(cfg, CacheActivationsRunnerConfig):
            return cls.from_cache_activations(model, cfg)

        cached_activations_path = cfg.cached_activations_path
        # set cached_activations_path to None if we're not using cached activations
        if (
            isinstance(cfg, LanguageModelSAERunnerConfig)
            and not cfg.use_cached_activations
        ):
            cached_activations_path = None

        if override_dataset is None and cfg.dataset_path == "":
            raise ValueError(
                "You must either pass in a dataset or specify a dataset_path in your configutation."
            )

        device = torch.device(cfg.act_store_device)
        exclude_special_tokens = cfg.exclude_special_tokens
        if exclude_special_tokens is False:
            exclude_special_tokens = None
        if exclude_special_tokens is True:
            exclude_special_tokens = get_special_token_ids(model.tokenizer)  # type: ignore
        if exclude_special_tokens is not None:
            exclude_special_tokens = torch.tensor(
                exclude_special_tokens, dtype=torch.long, device=device
            )
        return cls(
            model=model,
            dataset=override_dataset or cfg.dataset_path,
            streaming=cfg.streaming,
            hook_name=cfg.hook_name,
            hook_head_index=cfg.hook_head_index,
            context_size=cfg.context_size,
            d_in=cfg.d_in
            if isinstance(cfg, CacheActivationsRunnerConfig)
            else cfg.sae.d_in,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size_prompts=cfg.store_batch_size_prompts,
            train_batch_size_tokens=cfg.train_batch_size_tokens,
            prepend_bos=cfg.prepend_bos,
            normalize_activations=cfg.sae.normalize_activations,
            device=device,
            dtype=cfg.dtype,
            cached_activations_path=cached_activations_path,
            model_kwargs=cfg.model_kwargs,
            autocast_lm=cfg.autocast_lm,
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
            seqpos_slice=cfg.seqpos_slice,
            exclude_special_tokens=exclude_special_tokens,
            disable_concat_sequences=cfg.disable_concat_sequences,
            sequence_separator_token=cfg.sequence_separator_token,
        )

    @classmethod
    def from_sae(
        cls,
        model: HookedRootModule,
        sae: SAE[T_SAE_CONFIG],
        dataset: HfDataset | str,
        dataset_trust_remote_code: bool = False,
        context_size: int | None = None,
        streaming: bool = True,
        store_batch_size_prompts: int = 8,
        n_batches_in_buffer: int = 8,
        train_batch_size_tokens: int = 4096,
        total_tokens: int = 10**9,
        device: str = "cpu",
        disable_concat_sequences: bool = False,
        sequence_separator_token: int | Literal["bos", "eos", "sep"] | None = "bos",
    ) -> ActivationsStore:
        if context_size is None:
            context_size = sae.cfg.metadata.context_size
        if sae.cfg.metadata.hook_name is None:
            raise ValueError("hook_name is required")
        if context_size is None:
            raise ValueError("context_size is required")
        if sae.cfg.metadata.prepend_bos is None:
            raise ValueError("prepend_bos is required")
        return cls(
            model=model,
            dataset=dataset,
            d_in=sae.cfg.d_in,
            hook_name=sae.cfg.metadata.hook_name,
            hook_head_index=sae.cfg.metadata.hook_head_index,
            context_size=context_size,
            prepend_bos=sae.cfg.metadata.prepend_bos,
            streaming=streaming,
            store_batch_size_prompts=store_batch_size_prompts,
            train_batch_size_tokens=train_batch_size_tokens,
            n_batches_in_buffer=n_batches_in_buffer,
            total_training_tokens=total_tokens,
            normalize_activations=sae.cfg.normalize_activations,
            dataset_trust_remote_code=dataset_trust_remote_code,
            dtype=sae.cfg.dtype,
            device=torch.device(device),
            seqpos_slice=sae.cfg.metadata.seqpos_slice or (None,),
            disable_concat_sequences=disable_concat_sequences,
            sequence_separator_token=sequence_separator_token,
        )

    def __init__(
        self,
        model: HookedRootModule,
        dataset: HfDataset | str,
        streaming: bool,
        hook_name: str,
        hook_head_index: int | None,
        context_size: int,
        d_in: int,
        n_batches_in_buffer: int,
        total_training_tokens: int,
        store_batch_size_prompts: int,
        train_batch_size_tokens: int,
        prepend_bos: bool,
        normalize_activations: str,
        device: torch.device,
        dtype: str,
        cached_activations_path: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        autocast_lm: bool = False,
        dataset_trust_remote_code: bool | None = None,
        seqpos_slice: tuple[int | None, ...] = (None,),
        exclude_special_tokens: torch.Tensor | None = None,
        disable_concat_sequences: bool = False,
        sequence_separator_token: int | Literal["bos", "eos", "sep"] | None = "bos",
    ):
        self.model = model
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs
        self.dataset = (
            load_dataset(
                dataset,
                split="train",
                streaming=streaming,
                trust_remote_code=dataset_trust_remote_code,  # type: ignore
            )
            if isinstance(dataset, str)
            else dataset
        )

        if isinstance(dataset, (Dataset, DatasetDict)):
            self.dataset = cast(Dataset | DatasetDict, self.dataset)
            n_samples = len(self.dataset)

            if n_samples < total_training_tokens:
                warnings.warn(
                    f"The training dataset contains fewer samples ({n_samples}) than the number of samples required by your training configuration ({total_training_tokens}). This will result in multiple training epochs and some samples being used more than once."
                )

        self.hook_name = hook_name
        self.hook_head_index = hook_head_index
        self.context_size = context_size
        self.d_in = d_in
        self.n_batches_in_buffer = n_batches_in_buffer
        self.half_buffer_size = n_batches_in_buffer // 2
        self.total_training_tokens = total_training_tokens
        self.store_batch_size_prompts = store_batch_size_prompts
        self.train_batch_size_tokens = train_batch_size_tokens
        self.prepend_bos = prepend_bos
        self.normalize_activations = normalize_activations
        self.device = torch.device(device)
        self.dtype = DTYPE_MAP[dtype]
        self.cached_activations_path = cached_activations_path
        self.autocast_lm = autocast_lm
        self.seqpos_slice = seqpos_slice
        self.training_context_size = len(range(context_size)[slice(*seqpos_slice)])
        self.exclude_special_tokens = exclude_special_tokens
        self.disable_concat_sequences = disable_concat_sequences
        self.sequence_separator_token: int | Literal["bos", "eos", "sep"] | None = (
            sequence_separator_token
        )

        self.n_dataset_processed = 0

        # Check if dataset is tokenized
        dataset_sample = next(iter(self.dataset))

        # check if it's tokenized
        if "tokens" in dataset_sample:
            self.is_dataset_tokenized = True
            self.tokens_column = "tokens"
        elif "input_ids" in dataset_sample:
            self.is_dataset_tokenized = True
            self.tokens_column = "input_ids"
        elif "text" in dataset_sample:
            self.is_dataset_tokenized = False
            self.tokens_column = "text"
        elif "problem" in dataset_sample:
            self.is_dataset_tokenized = False
            self.tokens_column = "problem"
        else:
            raise ValueError(
                "Dataset must have a 'tokens', 'input_ids', 'text', or 'problem' column."
            )
        if self.is_dataset_tokenized:
            ds_context_size = len(dataset_sample[self.tokens_column])  # type: ignore
            if ds_context_size < self.context_size:
                raise ValueError(
                    f"""pretokenized dataset has context_size {ds_context_size}, but the provided context_size is {self.context_size}.
                    The context_size {ds_context_size} is expected to be larger than or equal to the provided context size {self.context_size}."""
                )
            if self.context_size != ds_context_size:
                warnings.warn(
                    f"""pretokenized dataset has context_size {ds_context_size}, but the provided context_size is {self.context_size}. Some data will be discarded in this case.""",
                    RuntimeWarning,
                )
            # TODO: investigate if this can work for iterable datasets, or if this is even worthwhile as a perf improvement
            if hasattr(self.dataset, "set_format"):
                self.dataset.set_format(type="torch", columns=[self.tokens_column])  # type: ignore

            if (
                isinstance(dataset, str)
                and hasattr(model, "tokenizer")
                and model.tokenizer is not None
            ):
                validate_pretokenized_dataset_tokenizer(
                    dataset_path=dataset,
                    model_tokenizer=model.tokenizer,  # type: ignore
                )
        else:
            warnings.warn(
                "Dataset is not tokenized. Pre-tokenizing will improve performance and allows for more control over special tokens. See https://decoderesearch.github.io/SAELens/training_saes/#pretokenizing-datasets for more info."
            )

        self.iterable_sequences = self._iterate_tokenized_sequences()

        self.cached_activation_dataset = self.load_cached_activation_dataset()

        # TODO add support for "mixed loading" (ie use cache until you run out, then switch over to streaming from HF)

    def _iterate_raw_dataset(
        self,
    ) -> Generator[torch.Tensor | list[int] | str, None, None]:
        """
        Helper to iterate over the dataset while incrementing n_dataset_processed
        """
        for row in self.dataset:
            # typing datasets is difficult
            yield row[self.tokens_column]  # type: ignore
            self.n_dataset_processed += 1

    def _iterate_raw_dataset_tokens(self) -> Generator[torch.Tensor, None, None]:
        """
        Helper to create an iterator which tokenizes raw text from the dataset on the fly
        """
        for row in self._iterate_raw_dataset():
            tokens = (
                self.model.to_tokens(
                    row,
                    truncate=False,
                    move_to_device=False,  # we move to device below
                    prepend_bos=False,
                )  # type: ignore
                .squeeze(0)
                .to(self.device)
            )
            if len(tokens.shape) != 1:
                raise ValueError(f"tokens.shape should be 1D but was {tokens.shape}")
            yield tokens

    def _iterate_tokenized_sequences(self) -> Generator[torch.Tensor, None, None]:
        """
        Generator which iterates over full sequence of context_size tokens
        """
        # If the datset is pretokenized, we will slice the dataset to the length of the context window if needed. Otherwise, no further processing is needed.
        # We assume that all necessary BOS/EOS/SEP tokens have been added during pretokenization.
        if self.is_dataset_tokenized:
            for row in self._iterate_raw_dataset():
                yield torch.tensor(
                    row[
                        : self.context_size
                    ],  # If self.context_size = None, this line simply returns the whole row
                    dtype=torch.long,
                    device=self.device,
                    requires_grad=False,
                )
        # If the dataset isn't tokenized, we'll tokenize, concat, and batch on the fly
        else:
            tokenizer = getattr(self.model, "tokenizer", None)
            bos_token_id = None if tokenizer is None else tokenizer.bos_token_id

            yield from concat_and_batch_sequences(
                tokens_iterator=self._iterate_raw_dataset_tokens(),
                context_size=self.context_size,
                begin_batch_token_id=(bos_token_id if self.prepend_bos else None),
                begin_sequence_token_id=None,
                sequence_separator_token_id=get_special_token_from_cfg(
                    self.sequence_separator_token, tokenizer
                )
                if tokenizer is not None
                else None,
                disable_concat_sequences=self.disable_concat_sequences,
            )

    def load_cached_activation_dataset(self) -> Dataset | None:
        """
        Load the cached activation dataset from disk.

        - If cached_activations_path is set, returns Huggingface Dataset else None
        - Checks that the loaded dataset has current has activations for hooks in config and that shapes match.
        """
        if self.cached_activations_path is None:
            return None

        assert self.cached_activations_path is not None  # keep pyright happy
        # Sanity check: does the cache directory exist?
        if not os.path.exists(self.cached_activations_path):
            raise FileNotFoundError(
                f"Cache directory {self.cached_activations_path} does not exist. "
                "Consider double-checking your dataset, model, and hook names."
            )

        # ---
        # Actual code
        activations_dataset = datasets.load_from_disk(self.cached_activations_path)
        columns = [self.hook_name]
        if "token_ids" in activations_dataset.column_names:
            columns.append("token_ids")
        activations_dataset.set_format(
            type="torch", columns=columns, device=self.device, dtype=self.dtype
        )
        self.current_row_idx = 0  # idx to load next batch from
        # ---

        assert isinstance(activations_dataset, Dataset)

        # multiple in hooks future
        if not set([self.hook_name]).issubset(activations_dataset.column_names):
            raise ValueError(
                f"loaded dataset does not include hook activations, got {activations_dataset.column_names}"
            )

        if activations_dataset.features[self.hook_name].shape != (
            self.context_size,
            self.d_in,
        ):
            raise ValueError(
                f"Given dataset of shape {activations_dataset.features[self.hook_name].shape} does not match context_size ({self.context_size}) and d_in ({self.d_in})"
            )

        return activations_dataset

    def shuffle_input_dataset(self, seed: int, buffer_size: int = 1):
        """
        This applies a shuffle to the huggingface dataset that is the input to the activations store. This
        also shuffles the shards of the dataset, which is especially useful for evaluating on different
        sections of very large streaming datasets. Buffer size is only relevant for streaming datasets.
        The default buffer_size of 1 means that only the shard will be shuffled; larger buffer sizes will
        additionally shuffle individual elements within the shard.
        """
        if isinstance(self.dataset, IterableDataset):
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)
        else:
            self.dataset = self.dataset.shuffle(seed=seed)
        self.iterable_dataset = iter(self.dataset)

    def reset_input_dataset(self):
        """
        Resets the input dataset iterator to the beginning.
        """
        self.iterable_dataset = iter(self.dataset)

    def get_batch_tokens(
        self, batch_size: int | None = None, raise_at_epoch_end: bool = False
    ):
        """
        Streams a batch of tokens from a dataset.

        If raise_at_epoch_end is true we will reset the dataset at the end of each epoch and raise a StopIteration. Otherwise we will reset silently.
        """
        if not batch_size:
            batch_size = self.store_batch_size_prompts
        sequences = []
        # the sequences iterator yields fully formed tokens of size context_size, so we just need to cat these into a batch
        for _ in range(batch_size):
            try:
                sequences.append(next(self.iterable_sequences))
            except StopIteration:
                self.iterable_sequences = self._iterate_tokenized_sequences()
                if raise_at_epoch_end:
                    raise StopIteration(
                        f"Ran out of tokens in dataset after {self.n_dataset_processed} samples, beginning the next epoch."
                    )
                sequences.append(next(self.iterable_sequences))

        return torch.stack(sequences, dim=0).to(_get_model_device(self.model))

    @torch.no_grad()
    def get_activations(self, batch_tokens: torch.Tensor):
        """
        Returns activations of shape (batches, context, num_layers, d_in)

        d_in may result from a concatenated head dimension.
        """
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.autocast_lm,
        ):
            layerwise_activations_cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_name],
                stop_at_layer=extract_stop_at_layer_from_tlens_hook_name(
                    self.hook_name
                ),
                prepend_bos=False,
                **self.model_kwargs,
            )[1]

        layerwise_activations = layerwise_activations_cache[self.hook_name][
            :, slice(*self.seqpos_slice)
        ]

        n_batches, n_context = layerwise_activations.shape[:2]

        stacked_activations = torch.zeros((n_batches, n_context, self.d_in))

        if self.hook_head_index is not None:
            stacked_activations[:, :] = layerwise_activations[
                :, :, self.hook_head_index
            ]
        elif layerwise_activations.ndim > 3:  # if we have a head dimension
            try:
                stacked_activations[:, :] = layerwise_activations.view(
                    n_batches, n_context, -1
                )
            except RuntimeError as e:
                logger.error(f"Error during view operation: {e}")
                logger.info("Attempting to use reshape instead...")
                stacked_activations[:, :] = layerwise_activations.reshape(
                    n_batches, n_context, -1
                )
        else:
            stacked_activations[:, :] = layerwise_activations

        return stacked_activations

    def _load_buffer_from_cached(
        self,
        total_size: int,
        context_size: int,
        d_in: int,
        raise_on_epoch_end: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """
        Loads `total_size` activations from `cached_activation_dataset`

        The dataset has columns for each hook_name,
        each containing activations of shape (context_size, d_in).

        raises StopIteration
        """
        assert self.cached_activation_dataset is not None
        # In future, could be a list of multiple hook names
        if self.hook_name not in self.cached_activation_dataset.column_names:
            raise ValueError(
                f"Missing columns in dataset. Expected {self.hook_name}, "
                f"got {self.cached_activation_dataset.column_names}."
            )

        if self.current_row_idx > len(self.cached_activation_dataset) - total_size:
            self.current_row_idx = 0
            if raise_on_epoch_end:
                raise StopIteration

        new_buffer = []
        ds_slice = self.cached_activation_dataset[
            self.current_row_idx : self.current_row_idx + total_size
        ]
        # Load activations for each hook.
        # Usually faster to first slice dataset then pick column
        new_buffer = ds_slice[self.hook_name]
        if new_buffer.shape != (total_size, context_size, d_in):
            raise ValueError(
                f"new_buffer has shape {new_buffer.shape}, "
                f"but expected ({total_size}, {context_size}, {d_in})."
            )

        self.current_row_idx += total_size
        acts_buffer = new_buffer.reshape(total_size * context_size, d_in)

        if "token_ids" not in self.cached_activation_dataset.column_names:
            return acts_buffer, None

        token_ids_buffer = ds_slice["token_ids"]
        if token_ids_buffer.shape != (total_size, context_size):
            raise ValueError(
                f"token_ids_buffer has shape {token_ids_buffer.shape}, "
                f"but expected ({total_size}, {context_size})."
            )
        token_ids_buffer = token_ids_buffer.reshape(total_size * context_size)
        return acts_buffer, token_ids_buffer

    @torch.no_grad()
    def get_raw_buffer(
        self,
        n_batches_in_buffer: int,
        raise_on_epoch_end: bool = False,
        shuffle: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Loads the next n_batches_in_buffer batches of activations into a tensor and returns it.

        The primary purpose here is maintaining a shuffling buffer.

        If raise_on_epoch_end is True, when the dataset it exhausted it will automatically refill the dataset and then raise a StopIteration so that the caller has a chance to react.
        """
        context_size = self.context_size
        batch_size = self.store_batch_size_prompts
        d_in = self.d_in
        total_size = batch_size * n_batches_in_buffer

        if self.cached_activation_dataset is not None:
            return self._load_buffer_from_cached(
                total_size, context_size, d_in, raise_on_epoch_end
            )

        refill_iterator = range(0, total_size, batch_size)
        # Initialize empty tensor buffer of the maximum required size with an additional dimension for layers
        new_buffer_activations = torch.zeros(
            (total_size, self.training_context_size, d_in),
            dtype=self.dtype,  # type: ignore
            device=self.device,
        )
        new_buffer_token_ids = torch.zeros(
            (total_size, self.training_context_size),
            dtype=torch.long,
            device=self.device,
        )

        for refill_batch_idx_start in tqdm(
            refill_iterator, leave=False, desc="Refilling buffer"
        ):
            # move batch toks to gpu for model
            refill_batch_tokens = self.get_batch_tokens(
                raise_at_epoch_end=raise_on_epoch_end
            ).to(_get_model_device(self.model))
            refill_activations = self.get_activations(refill_batch_tokens)
            # move acts back to cpu
            refill_activations.to(self.device)
            new_buffer_activations[
                refill_batch_idx_start : refill_batch_idx_start + batch_size, ...
            ] = refill_activations

            # handle seqpos_slice, this is done for activations in get_activations
            refill_batch_tokens = refill_batch_tokens[:, slice(*self.seqpos_slice)]
            new_buffer_token_ids[
                refill_batch_idx_start : refill_batch_idx_start + batch_size, ...
            ] = refill_batch_tokens

        new_buffer_activations = new_buffer_activations.reshape(-1, d_in)
        new_buffer_token_ids = new_buffer_token_ids.reshape(-1)
        if shuffle:
            new_buffer_activations, new_buffer_token_ids = permute_together(
                [new_buffer_activations, new_buffer_token_ids]
            )

        return (
            new_buffer_activations,
            new_buffer_token_ids,
        )

    def get_filtered_buffer(
        self,
        n_batches_in_buffer: int,
        raise_on_epoch_end: bool = False,
        shuffle: bool = True,
    ) -> torch.Tensor:
        return _filter_buffer_acts(
            self.get_raw_buffer(
                n_batches_in_buffer=n_batches_in_buffer,
                raise_on_epoch_end=raise_on_epoch_end,
                shuffle=shuffle,
            ),
            self.exclude_special_tokens,
        )

    def _iterate_filtered_activations(self) -> Generator[torch.Tensor, None, None]:
        """
        Iterate over the filtered tokens in the buffer.
        """
        while True:
            try:
                yield self.get_filtered_buffer(
                    self.half_buffer_size, raise_on_epoch_end=True
                )
            except StopIteration:
                warnings.warn(
                    "All samples in the training dataset have been exhausted, beginning new epoch."
                )
                try:
                    yield self.get_filtered_buffer(self.half_buffer_size)
                except StopIteration:
                    raise ValueError(
                        "Unable to fill buffer after starting new epoch. Dataset may be too small."
                    )

    def get_data_loader(
        self,
    ) -> Iterator[Any]:
        """
        Return an auto-refilling stream of filtered and mixed activations.
        """
        return mixing_buffer(
            buffer_size=self.n_batches_in_buffer * self.training_context_size,
            batch_size=self.train_batch_size_tokens,
            activations_loader=self._iterate_filtered_activations(),
        )

    def next_batch(self) -> torch.Tensor:
        """Get next batch, updating buffer if needed."""
        return self.__next__()

    # ActivationsStore should be an iterator
    def __next__(self) -> torch.Tensor:
        if self._dataloader is None:
            self._dataloader = self.get_data_loader()
        return next(self._dataloader)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"n_dataset_processed": torch.tensor(self.n_dataset_processed)}

    def save(self, file_path: str):
        """save the state dict to a file in safetensors format"""
        save_file(self.state_dict(), file_path)

    def save_to_checkpoint(self, checkpoint_path: str | Path):
        """Save the state dict to a checkpoint path"""
        self.save(str(Path(checkpoint_path) / ACTIVATIONS_STORE_STATE_FILENAME))

    def load_from_checkpoint(self, checkpoint_path: str | Path):
        """Load the state dict from a checkpoint path"""
        self.load(str(Path(checkpoint_path) / ACTIVATIONS_STORE_STATE_FILENAME))

    def load(self, file_path: str):
        """Load the state dict from a file in safetensors format"""

        state_dict = load_file(file_path)

        if "n_dataset_processed" in state_dict:
            target_n_dataset_processed = state_dict["n_dataset_processed"].item()

            # Only fast-forward if needed

            if target_n_dataset_processed > self.n_dataset_processed:
                logger.info(
                    "Fast-forwarding through dataset samples to match checkpoint position"
                )
                samples_to_skip = target_n_dataset_processed - self.n_dataset_processed

                pbar = tqdm(
                    total=samples_to_skip,
                    desc="Fast-forwarding through dataset",
                    leave=False,
                )
                while target_n_dataset_processed > self.n_dataset_processed:
                    start = self.n_dataset_processed
                    try:
                        # Just consume and ignore the values to fast-forward
                        next(self.iterable_sequences)
                    except StopIteration:
                        logger.warning(
                            "Dataset exhausted during fast-forward. Resetting dataset."
                        )
                        self.iterable_sequences = self._iterate_tokenized_sequences()
                    pbar.update(self.n_dataset_processed - start)
                pbar.close()


def validate_pretokenized_dataset_tokenizer(
    dataset_path: str, model_tokenizer: PreTrainedTokenizerBase
) -> None:
    """
    Helper to validate that the tokenizer used to pretokenize the dataset matches the model tokenizer.
    """
    try:
        tokenization_cfg_path = hf_hub_download(
            dataset_path, "sae_lens.json", repo_type="dataset"
        )
    except HfHubHTTPError:
        return
    if tokenization_cfg_path is None:
        return
    with open(tokenization_cfg_path) as f:
        tokenization_cfg = json.load(f)
    tokenizer_name = tokenization_cfg["tokenizer_name"]
    try:
        ds_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # if we can't download the specified tokenizer to verify, just continue
    except HTTPError:
        return
    if ds_tokenizer.get_vocab() != model_tokenizer.get_vocab():
        raise ValueError(
            f"Dataset tokenizer {tokenizer_name} does not match model tokenizer {model_tokenizer}."
        )


def _get_model_device(model: HookedRootModule) -> torch.device:
    if hasattr(model, "W_E"):
        return model.W_E.device  # type: ignore
    if hasattr(model, "cfg") and hasattr(model.cfg, "device"):
        return model.cfg.device  # type: ignore
    return next(model.parameters()).device  # type: ignore


def _filter_buffer_acts(
    buffer: tuple[torch.Tensor, torch.Tensor | None],
    exclude_tokens: torch.Tensor | None,
) -> torch.Tensor:
    """
    Filter out activations for tokens that are in exclude_tokens.
    """

    activations, tokens = buffer
    if tokens is None or exclude_tokens is None:
        return activations

    mask = torch.isin(tokens, exclude_tokens)
    return activations[~mask]


def permute_together(tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Permute tensors together."""
    permutation = torch.randperm(tensors[0].shape[0])
    return tuple(t[permutation] for t in tensors)
