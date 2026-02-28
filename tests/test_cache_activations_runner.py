import dataclasses
import math
import os
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pytest
import torch
from datasets import Dataset, load_dataset
from tqdm import trange
from transformer_lens import HookedTransformer

from sae_lens.cache_activations_runner import CacheActivationsRunner
from sae_lens.config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
)
from sae_lens.constants import DTYPE_MAP
from sae_lens.load_model import load_model
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig
from sae_lens.training.activations_store import ActivationsStore
from tests.helpers import assert_close


def _default_cfg(
    tmp_path: Path,
    batch_size: int = 16,
    context_size: int = 8,
    dataset_num_rows: int = 128,
    n_buffers: int = 4,
    shuffle: bool = False,
    **kwargs: Any,
) -> CacheActivationsRunnerConfig:
    d_in = 512
    dtype = "float32"
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    sliced_context_size = kwargs.get("seqpos_slice")
    if sliced_context_size is not None:
        sliced_context_size = len(range(context_size)[slice(*sliced_context_size)])
    else:
        sliced_context_size = context_size

    # Calculate buffer_size_gb to achieve desired n_buffers
    bytes_per_token = d_in * DTYPE_MAP[dtype].itemsize
    tokens_per_buffer = math.ceil(dataset_num_rows * sliced_context_size / n_buffers)
    buffer_size_gb = (tokens_per_buffer * bytes_per_token) / 1_000_000_000
    total_training_tokens = dataset_num_rows * sliced_context_size

    cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=str(tmp_path),
        dataset_path="chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests",
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        ### Parameters
        training_tokens=total_training_tokens,
        model_batch_size=batch_size,
        buffer_size_gb=buffer_size_gb,
        context_size=context_size,
        ###
        d_in=d_in,
        shuffle=shuffle,
        prepend_bos=False,
        device=device,
        seed=42,
        dtype=dtype,
        **kwargs,
    )
    assert cfg.n_buffers == n_buffers
    assert cfg.n_seq_in_dataset == dataset_num_rows
    assert (
        cfg.n_tokens_in_buffer
        == cfg.n_batches_in_buffer * batch_size * sliced_context_size
    )
    return cfg


# The way to run this with this command:
# poetry run py.test tests/test_cache_activations_runner.py --profile-svg -s
def test_cache_activations_runner(tmp_path: Path):
    cfg = _default_cfg(tmp_path)
    runner = CacheActivationsRunner(cfg)
    dataset = runner.run()

    assert len(dataset) == cfg.n_buffers * (cfg.n_tokens_in_buffer // cfg.context_size)
    assert cfg.n_seq_in_dataset == len(dataset)
    assert dataset.column_names == [cfg.hook_name, "token_ids"]

    features = dataset.features
    assert isinstance(features[cfg.hook_name], datasets.Array2D)
    assert features[cfg.hook_name].shape == (cfg.context_size, cfg.d_in)
    assert isinstance(features["token_ids"], datasets.Sequence)
    assert features["token_ids"].length == cfg.context_size


def test_load_cached_activations(tmp_path: Path):
    cfg = _default_cfg(tmp_path)
    runner = CacheActivationsRunner(cfg)
    runner.run()

    model = HookedTransformer.from_pretrained(cfg.model_name)

    activations_store = ActivationsStore.from_config(model, cfg)

    # get_raw_llm_batch returns one batch at a time
    # Total batches = n_buffers * n_batches_in_buffer
    for _ in range(cfg.n_buffers * cfg.n_batches_in_buffer):
        buffer = activations_store.get_raw_llm_batch()
        assert buffer[0].shape == (
            activations_store.store_batch_size_prompts * cfg.context_size,
            cfg.d_in,
        )
        assert buffer[1] is not None
        assert buffer[1].shape == (
            activations_store.store_batch_size_prompts * cfg.context_size,
        )


def test_cache_activations_runner_to_string():
    cfg = _default_cfg(Path("tmp_path"))
    runner = CacheActivationsRunner(cfg)
    result = str(runner)

    # Check that the string contains the expected summary format
    assert "Activation Cache Runner:" in result
    assert "Total training tokens: 1024" in result
    assert "Number of buffers: 4" in result
    assert "Tokens per buffer: 256" in result
    assert "Disk space required: 0.00 GB" in result
    assert "Configuration:" in result

    # Check that the config contains expected fields
    assert "dataset_path='chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests'" in result
    assert "model_name='gelu-1l'" in result
    assert "hook_name='blocks.0.hook_mlp_out'" in result
    assert "training_tokens=1024" in result
    assert "context_size=8" in result
    assert "d_in=512" in result


def test_activations_store_refreshes_dataset_when_it_runs_out(tmp_path: Path):
    context_size = 8
    n_batches_in_buffer = 4
    store_batch_size = 1
    total_training_steps = 4
    batch_size = 4
    total_training_tokens = total_training_steps * batch_size

    cache_cfg = _default_cfg(tmp_path)
    runner = CacheActivationsRunner(cache_cfg)
    runner.run()

    cfg = LanguageModelSAERunnerConfig(
        sae=StandardTrainingSAEConfig(d_in=512, d_sae=768),
        cached_activations_path=str(tmp_path),
        use_cached_activations=True,
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        dataset_path="",
        context_size=context_size,
        is_dataset_tokenized=True,
        prepend_bos=True,
        training_tokens=total_training_tokens // 2,
        train_batch_size_tokens=8,
        n_batches_in_buffer=n_batches_in_buffer,
        store_batch_size_prompts=store_batch_size,
        device="cpu",
        seed=42,
        dtype="float16",
    )

    class MockModel:
        def to_tokens(self, *args: tuple[Any, ...], **kwargs: Any) -> torch.Tensor:
            return torch.ones(context_size)

        @property
        def W_E(self) -> torch.Tensor:
            return torch.ones(16, 16)

        @property
        def cfg(self) -> LanguageModelSAERunnerConfig[StandardTrainingSAEConfig]:
            return cfg

    dataset = Dataset.from_list([{"text": "hello world1"}] * 64)

    model = MockModel()
    activations_store = ActivationsStore.from_config(
        model,  # type: ignore
        cfg,
        override_dataset=dataset,
    )
    for _ in range(16):
        _ = activations_store.get_batch_tokens(batch_size, raise_at_epoch_end=True)

    # assert a stop iteration is raised when we do one more get_batch_tokens

    pytest.raises(
        StopIteration,
        activations_store.get_batch_tokens,
        batch_size,
        raise_at_epoch_end=True,
    )

    # no errors are ever raised if we do not ask for raise_at_epoch_end
    for _ in range(32):
        _ = activations_store.get_batch_tokens(batch_size, raise_at_epoch_end=False)


def test_compare_cached_activations_end_to_end_with_ground_truth(tmp_path: Path):
    """
    Creates activations using CacheActivationsRunner and compares them with ground truth
    model.run_with_cache
    """

    torch.manual_seed(42)
    cfg = _default_cfg(tmp_path)
    runner = CacheActivationsRunner(cfg)
    activation_dataset = runner.run()
    activation_dataset.set_format("torch")
    dataset_acts: torch.Tensor = activation_dataset[cfg.hook_name]  # type: ignore

    model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)
    token_dataset: Dataset = load_dataset(
        cfg.dataset_path, split=f"train[:{cfg.n_seq_in_dataset}]"
    )  # type: ignore
    token_dataset.set_format("torch", device=cfg.device)

    ground_truth_acts = []
    for i in trange(0, cfg.n_seq_in_dataset, cfg.model_batch_size):
        tokens = token_dataset[i : i + cfg.model_batch_size]["input_ids"][
            :, : cfg.context_size
        ]
        _, layerwise_activations = model.run_with_cache(
            tokens,
            names_filter=[cfg.hook_name],
        )
        acts = layerwise_activations[cfg.hook_name]
        ground_truth_acts.append(acts)

    ground_truth_acts = torch.cat(ground_truth_acts, dim=0).cpu()

    dataset_acts_tensor = torch.tensor(np.array(dataset_acts))
    assert_close(ground_truth_acts, dataset_acts_tensor, rtol=1e-3, atol=5e-2)


def test_load_activations_store_with_nonexistent_dataset(tmp_path: Path):
    cfg = _default_cfg(tmp_path)

    model = load_model(
        model_class_name=cfg.model_class_name,
        model_name=cfg.model_name,
        device=cfg.device,
        model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
    )

    # Attempt to load from a non-existent dataset
    with pytest.raises(
        FileNotFoundError,
        match="is neither a `Dataset` directory nor a `DatasetDict` directory.",
    ):
        ActivationsStore.from_config(model, cfg)


def test_cache_activations_runner_with_nonempty_directory(tmp_path: Path):
    # Create a file to make the directory non-empty
    with open(tmp_path / "some_file.txt", "w") as f:
        f.write("test")

    with pytest.raises(
        Exception, match="is not empty. Please delete it or specify a different path."
    ):
        cfg = _default_cfg(tmp_path)
        runner = CacheActivationsRunner(cfg)
        runner.run()


def test_cache_activations_runner_with_incorrect_d_in(tmp_path: Path):
    correct_cfg = _default_cfg(tmp_path)

    # d_in different from hook
    wrong_d_in_cfg = CacheActivationsRunnerConfig(
        **dataclasses.asdict(correct_cfg),
    )
    wrong_d_in_cfg.d_in = 513

    runner = CacheActivationsRunner(wrong_d_in_cfg)
    with pytest.raises(
        RuntimeError,
        match=r"The expanded size of the tensor \(513\) must match the existing size \(512\) at non-singleton dimension 2.",
    ):
        runner.run()


def test_cache_activations_runner_load_dataset_with_incorrect_config(tmp_path: Path):
    correct_cfg = _default_cfg(tmp_path, context_size=16)
    runner = CacheActivationsRunner(correct_cfg)
    runner.run()
    model = runner.model

    # Context size different from dataset
    wrong_context_size_cfg = CacheActivationsRunnerConfig(
        **dataclasses.asdict(correct_cfg),
    )
    wrong_context_size_cfg.context_size = 13

    with pytest.raises(
        ValueError,
        match=r"Given dataset of shape \(16, 512\) does not match context_size \(13\) and d_in \(512\)",
    ):
        ActivationsStore.from_config(model, wrong_context_size_cfg)

    # d_in different from dataset
    wrong_d_in_cfg = CacheActivationsRunnerConfig(
        **dataclasses.asdict(correct_cfg),
    )
    wrong_d_in_cfg.d_in = 513

    with pytest.raises(
        ValueError,
        match=r"Given dataset of shape \(16, 512\) does not match context_size \(16\) and d_in \(513\)",
    ):
        ActivationsStore.from_config(model, wrong_d_in_cfg)

    # Incorrect hook_name
    wrong_hook_cfg = CacheActivationsRunnerConfig(
        **dataclasses.asdict(correct_cfg),
    )
    wrong_hook_cfg.hook_name = "blocks.1.hook_mlp_out"

    with pytest.raises(
        ValueError,
        match=r"Columns \['blocks.1.hook_mlp_out'\] not in the dataset. Current columns in the dataset: \['blocks.0.hook_mlp_out'\, 'token_ids'\]",
    ):
        ActivationsStore.from_config(model, wrong_hook_cfg)


def test_cache_activations_runner_with_valid_seqpos(tmp_path: Path):
    cfg = _default_cfg(
        tmp_path,
        batch_size=1,
        context_size=16,
        n_buffers=3,
        dataset_num_rows=12,
        seqpos_slice=(3, -3),
    )
    runner = CacheActivationsRunner(cfg)

    activation_dataset = runner.run()
    activation_dataset.set_format("torch", device=cfg.device)
    dataset_acts: torch.Tensor = activation_dataset[cfg.hook_name]  # type: ignore

    assert os.path.exists(tmp_path)

    # assert that there are n_buffer files in the directory.
    buffer_files = [
        f
        for f in os.listdir(tmp_path)
        if f.startswith("data-") and f.endswith(".arrow")
    ]
    assert len(buffer_files) == cfg.n_buffers

    for act in dataset_acts:
        # should be 16 - 3 - 3 = 10
        assert act.shape == (10, cfg.d_in)


def test_cache_activations_runner_stores_token_ids(tmp_path: Path):
    cfg = _default_cfg(tmp_path)
    runner = CacheActivationsRunner(cfg)
    dataset = runner.run()
    dataset.set_format("torch")

    assert "token_ids" in dataset.features
    token_ids_array = np.array(dataset["token_ids"])
    mlp_out_array = np.array(dataset["blocks.0.hook_mlp_out"])
    assert token_ids_array.shape[1] == cfg.context_size
    assert mlp_out_array.shape[:2] == token_ids_array.shape


def test_cache_activations_runner_shuffling(tmp_path: Path):
    """Test that when shuffle=True, activations and token IDs remain aligned after shuffling."""
    # Create test dataset with arbitrary unique tokens
    tokenizer = HookedTransformer.from_pretrained("gelu-1l").tokenizer
    text = "".join(
        [
            " " + word[1:]
            for word in tokenizer.vocab  # type: ignore
            if word[0] == "Ġ" and word[1:].isascii() and word.isalnum()
        ]
    )
    dataset = Dataset.from_list([{"text": text}])

    # Create configs for unshuffled and shuffled versions
    base_cfg = _default_cfg(
        tmp_path / "base",
        context_size=3,
        batch_size=2,
        dataset_num_rows=8,
        shuffle=False,
    )
    shuffle_cfg = _default_cfg(
        tmp_path / "shuffled",
        context_size=3,
        batch_size=2,
        dataset_num_rows=8,
        shuffle=True,
    )

    # Get unshuffled dataset
    unshuffled_runner = CacheActivationsRunner(base_cfg, override_dataset=dataset)
    unshuffled_ds = unshuffled_runner.run()
    unshuffled_ds.set_format("torch")

    # Get shuffled dataset
    shuffled_runner = CacheActivationsRunner(shuffle_cfg, override_dataset=dataset)
    shuffled_ds = shuffled_runner.run()
    shuffled_ds.set_format("torch")

    # Get activations and tokens
    hook_name = base_cfg.hook_name
    unshuffled_acts: torch.Tensor = unshuffled_ds[hook_name]  # type: ignore
    unshuffled_tokens: torch.Tensor = unshuffled_ds["token_ids"]  # type: ignore
    shuffled_acts: torch.Tensor = shuffled_ds[hook_name]  # type: ignore
    shuffled_tokens: torch.Tensor = shuffled_ds["token_ids"]  # type: ignore

    # Verify shapes are preserved
    unshuffled_acts_array = np.array(unshuffled_acts)
    shuffled_acts_array = np.array(shuffled_acts)
    unshuffled_tokens_array = np.array(unshuffled_tokens)
    shuffled_tokens_array = np.array(shuffled_tokens)
    assert unshuffled_acts_array.shape == shuffled_acts_array.shape
    assert unshuffled_tokens_array.shape == shuffled_tokens_array.shape

    # Verify data is actually shuffled
    assert not np.array_equal(unshuffled_acts_array, shuffled_acts_array)
    assert not np.array_equal(unshuffled_tokens_array, shuffled_tokens_array)

    # For each token in unshuffled, find its position in shuffled
    # and verify the activations were moved together
    for i in range(len(unshuffled_tokens_array)):
        token = unshuffled_tokens_array[i]
        # Find where this token went in shuffled version
        shuffled_idx = np.where(shuffled_tokens_array == token)[0][0]
        # Verify activations moved with it
        assert_close(
            torch.from_numpy(unshuffled_acts_array[i]),
            torch.from_numpy(shuffled_acts_array[shuffled_idx]),
        )


def test_cache_activations_runner_shuffled_saved_to_disk(tmp_path: Path):
    """Test that when shuffle=True, the shuffled dataset is saved to disk (not just returned)."""
    # Create test dataset with arbitrary unique tokens
    tokenizer = HookedTransformer.from_pretrained("gelu-1l").tokenizer
    text = "".join(
        [
            " " + word[1:]
            for word in tokenizer.vocab  # type: ignore
            if word[0] == "Ġ" and word[1:].isascii() and word.isalnum()
        ]
    )
    dataset = Dataset.from_list([{"text": text}])

    # Create configs for unshuffled and shuffled versions
    unshuffled_path = tmp_path / "unshuffled"
    shuffled_path = tmp_path / "shuffled"

    unshuffled_cfg = _default_cfg(
        unshuffled_path,
        context_size=3,
        batch_size=2,
        dataset_num_rows=8,
        shuffle=False,
    )
    shuffled_cfg = _default_cfg(
        shuffled_path,
        context_size=3,
        batch_size=2,
        dataset_num_rows=8,
        shuffle=True,
    )

    # Run both
    unshuffled_runner = CacheActivationsRunner(unshuffled_cfg, override_dataset=dataset)
    unshuffled_runner.run()

    shuffled_runner = CacheActivationsRunner(shuffled_cfg, override_dataset=dataset)
    returned_shuffled_ds = shuffled_runner.run()
    returned_shuffled_ds.set_format("torch")

    # Load datasets from disk
    unshuffled_from_disk = datasets.load_from_disk(str(unshuffled_path))
    shuffled_from_disk = datasets.load_from_disk(str(shuffled_path))
    unshuffled_from_disk.set_format("torch")
    shuffled_from_disk.set_format("torch")

    hook_name = unshuffled_cfg.hook_name

    # Verify the shuffled dataset on disk is different from the unshuffled one
    unshuffled_tokens = np.array(unshuffled_from_disk["token_ids"])
    shuffled_tokens_on_disk = np.array(shuffled_from_disk["token_ids"])
    assert not np.array_equal(
        unshuffled_tokens, shuffled_tokens_on_disk
    ), "Shuffled dataset on disk should be different from unshuffled"

    # Verify the shuffled dataset on disk matches what was returned
    returned_tokens = np.array(returned_shuffled_ds["token_ids"])
    assert np.array_equal(
        shuffled_tokens_on_disk, returned_tokens
    ), "Dataset on disk should match returned dataset"

    # Also verify activations match
    shuffled_acts_on_disk = np.array(shuffled_from_disk[hook_name])
    returned_acts = np.array(returned_shuffled_ds[hook_name])
    assert np.array_equal(
        shuffled_acts_on_disk, returned_acts
    ), "Activations on disk should match returned activations"


def test_cache_activations_runner_shuffle_across_sequences(tmp_path: Path):
    """Test that shuffle_across_sequences shuffles individual activations across all sequence positions."""
    # Create test dataset with unique tokens
    tokenizer = HookedTransformer.from_pretrained("gelu-1l").tokenizer
    text = "".join(
        [
            " " + word[1:]
            for word in tokenizer.vocab  # type: ignore
            if word[0] == "Ġ" and word[1:].isascii() and word.isalnum()
        ]
    )
    dataset = Dataset.from_list([{"text": text}])

    # Create configs for unshuffled and shuffle_across_sequences versions
    base_cfg = _default_cfg(
        tmp_path / "base",
        context_size=4,
        batch_size=2,
        dataset_num_rows=8,
        shuffle=False,
        shuffle_across_sequences=False,
    )
    shuffle_across_cfg = _default_cfg(
        tmp_path / "shuffle_across",
        context_size=4,
        batch_size=2,
        dataset_num_rows=8,
        shuffle=True,  # Required when shuffle_across_sequences=True
        shuffle_across_sequences=True,
    )

    # Get unshuffled dataset
    unshuffled_runner = CacheActivationsRunner(base_cfg, override_dataset=dataset)
    unshuffled_ds = unshuffled_runner.run()
    unshuffled_ds.set_format("torch")

    # Get shuffle_across_sequences dataset
    shuffled_runner = CacheActivationsRunner(
        shuffle_across_cfg, override_dataset=dataset
    )
    shuffled_ds = shuffled_runner.run()
    shuffled_ds.set_format("torch")

    # Get activations and tokens
    hook_name = base_cfg.hook_name
    unshuffled_acts: torch.Tensor = unshuffled_ds[hook_name]  # type: ignore
    unshuffled_tokens: torch.Tensor = unshuffled_ds["token_ids"]  # type: ignore
    shuffled_acts: torch.Tensor = shuffled_ds[hook_name]  # type: ignore
    shuffled_tokens: torch.Tensor = shuffled_ds["token_ids"]  # type: ignore

    # Convert to numpy for easier manipulation
    unshuffled_acts_np = np.array(unshuffled_acts)
    unshuffled_tokens_np = np.array(unshuffled_tokens)
    shuffled_acts_np = np.array(shuffled_acts)
    shuffled_tokens_np = np.array(shuffled_tokens)

    # Verify shapes are preserved
    assert unshuffled_acts_np.shape == shuffled_acts_np.shape
    assert unshuffled_tokens_np.shape == shuffled_tokens_np.shape

    # Flatten to compare individual activations
    unshuffled_acts_flat = unshuffled_acts_np.reshape(-1, unshuffled_acts_np.shape[-1])
    unshuffled_tokens_flat = unshuffled_tokens_np.reshape(-1)
    shuffled_acts_flat = shuffled_acts_np.reshape(-1, shuffled_acts_np.shape[-1])
    shuffled_tokens_flat = shuffled_tokens_np.reshape(-1)

    # Verify data is actually shuffled (activations should be in different positions)
    assert not np.array_equal(unshuffled_acts_flat, shuffled_acts_flat)
    assert not np.array_equal(unshuffled_tokens_flat, shuffled_tokens_flat)

    # Verify token-activation pairs remain aligned after shuffling
    # For each unique token in unshuffled, find its activation and verify
    # the same token has the same activation in shuffled
    for i in range(len(unshuffled_tokens_flat)):
        token = unshuffled_tokens_flat[i]
        act = unshuffled_acts_flat[i]
        # Find where this token is in the shuffled version
        shuffled_indices = np.where(shuffled_tokens_flat == token)[0]
        # At least one of these positions should have the matching activation
        found_match = False
        for idx in shuffled_indices:
            if np.allclose(act, shuffled_acts_flat[idx], rtol=1e-5, atol=1e-5):
                found_match = True
                break
        assert found_match, f"Token {token} at position {i} lost its paired activation"


def test_cache_activations_runner_shuffle_across_sequences_reproducible(tmp_path: Path):
    """Test that shuffle_across_sequences is reproducible with the same seed."""
    tokenizer = HookedTransformer.from_pretrained("gelu-1l").tokenizer
    text = "".join(
        [
            " " + word[1:]
            for word in tokenizer.vocab  # type: ignore
            if word[0] == "Ġ" and word[1:].isascii() and word.isalnum()
        ]
    )
    dataset = Dataset.from_list([{"text": text}])

    # Create two configs with the same seed (default is 42)
    cfg1 = _default_cfg(
        tmp_path / "run1",
        context_size=4,
        batch_size=2,
        dataset_num_rows=8,
        shuffle=True,  # Required when shuffle_across_sequences=True
        shuffle_across_sequences=True,
    )
    cfg2 = _default_cfg(
        tmp_path / "run2",
        context_size=4,
        batch_size=2,
        dataset_num_rows=8,
        shuffle=True,  # Required when shuffle_across_sequences=True
        shuffle_across_sequences=True,
    )

    # Run both
    runner1 = CacheActivationsRunner(cfg1, override_dataset=dataset)
    ds1 = runner1.run()
    ds1.set_format("torch")

    runner2 = CacheActivationsRunner(cfg2, override_dataset=dataset)
    ds2 = runner2.run()
    ds2.set_format("torch")

    # Results should be identical
    hook_name = cfg1.hook_name
    acts1 = np.array(ds1[hook_name])
    acts2 = np.array(ds2[hook_name])
    tokens1 = np.array(ds1["token_ids"])
    tokens2 = np.array(ds2["token_ids"])

    assert np.array_equal(
        acts1, acts2
    ), "Same seed should produce identical activations"
    assert np.array_equal(tokens1, tokens2), "Same seed should produce identical tokens"


def test_cache_activations_runner_shuffle_across_sequences_requires_shuffle(
    tmp_path: Path,
):
    """Test that shuffle_across_sequences=True requires shuffle=True."""
    with pytest.raises(
        ValueError,
        match="shuffle_across_sequences=True requires shuffle=True",
    ):
        _default_cfg(
            tmp_path,
            shuffle=False,
            shuffle_across_sequences=True,
        )
