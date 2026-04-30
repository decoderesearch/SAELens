from typing import Any

import pytest
import torch
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_lens.training.activations_store import ActivationsStore
from tests.helpers import TINYSTORIES_MODEL, assert_close, load_model_cached


def _common_kwargs(dataset: Dataset) -> dict[str, Any]:
    return dict(
        dataset=dataset,
        streaming=False,
        context_size=8,
        n_batches_in_buffer=2,
        total_training_tokens=10_000,
        store_batch_size_prompts=4,
        train_batch_size_tokens=4,
        prepend_bos=True,
        normalize_activations="none",
        device=torch.device("cpu"),
        dtype="float32",
        seqpos_slice=(None,),
        activations_mixing_fraction=0.0,
    )


@pytest.fixture
def ts_model() -> HookedTransformer:
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture
def dataset() -> Dataset:
    # Each row is distinct so different positions have different activations
    # — catches misaligned splits across hooks.
    return Dataset.from_list(
        [{"text": f"document number {i} the quick brown fox jumps"} for i in range(200)]
    )


def test_multi_hook_get_multi_hook_activations_matches_per_hook_get_activations(
    ts_model: HookedTransformer,
    dataset: Dataset,
):
    hook_a = "blocks.0.hook_resid_pre"
    hook_b = "blocks.0.hook_mlp_out"
    d_in = ts_model.cfg.d_model

    common = _common_kwargs(dataset)

    store_single_a = ActivationsStore(
        model=ts_model, hook_name=hook_a, hook_head_index=None, d_in=d_in, **common
    )
    store_single_b = ActivationsStore(
        model=ts_model, hook_name=hook_b, hook_head_index=None, d_in=d_in, **common
    )
    store_multi = ActivationsStore.from_config_multi_hook(
        model=ts_model,
        hook_names=[hook_a, hook_b],
        hook_d_ins={hook_a: d_in, hook_b: d_in},
        **common,
    )

    batch_tokens = store_single_a.get_batch_tokens()
    # all stores see same dataset so independent fetches yield same tokens
    same_tokens = store_single_b.get_batch_tokens()
    assert torch.equal(batch_tokens, same_tokens)
    multi_tokens = store_multi.get_batch_tokens()
    assert torch.equal(batch_tokens, multi_tokens)

    single_a = store_single_a.get_activations(batch_tokens)
    single_b = store_single_b.get_activations(batch_tokens)
    multi = store_multi.get_multi_hook_activations(batch_tokens)

    assert set(multi.keys()) == {hook_a, hook_b}
    assert_close(multi[hook_a], single_a)
    assert_close(multi[hook_b], single_b)


def test_multi_hook_data_loader_matches_single_hook_per_hook(
    ts_model: HookedTransformer,
    dataset: Dataset,
):
    hook_a = "blocks.0.hook_resid_pre"
    hook_b = "blocks.0.hook_mlp_out"
    d_in = ts_model.cfg.d_model

    common = _common_kwargs(dataset)

    store_single_a = ActivationsStore(
        model=ts_model, hook_name=hook_a, hook_head_index=None, d_in=d_in, **common
    )
    store_single_b = ActivationsStore(
        model=ts_model, hook_name=hook_b, hook_head_index=None, d_in=d_in, **common
    )
    store_multi = ActivationsStore.from_config_multi_hook(
        model=ts_model,
        hook_names=[hook_a, hook_b],
        hook_d_ins={hook_a: d_in, hook_b: d_in},
        **common,
    )

    multi_iter = store_multi.get_multi_hook_data_loader()
    for _ in range(5):
        multi_batch = next(multi_iter)
        single_a_batch = next(store_single_a)
        single_b_batch = next(store_single_b)
        assert set(multi_batch.keys()) == {hook_a, hook_b}
        assert multi_batch[hook_a].shape == single_a_batch.shape
        assert_close(multi_batch[hook_a], single_a_batch)
        assert_close(multi_batch[hook_b], single_b_batch)


def test_multi_hook_supports_different_d_ins_via_attn_hook(
    ts_model: HookedTransformer,
    dataset: Dataset,
):
    # blocks.0.attn.hook_z has shape (B, C, n_heads, d_head); flattened d_in = n_heads * d_head.
    # For tiny-stories-1M (d_model=64, n_heads varying), this differs from hook_resid_pre's d_model.
    hook_resid = "blocks.0.hook_resid_pre"
    hook_z = "blocks.0.attn.hook_z"
    d_resid = ts_model.cfg.d_model
    d_z = ts_model.cfg.n_heads * ts_model.cfg.d_head

    common = _common_kwargs(dataset)

    store_single_z = ActivationsStore(
        model=ts_model, hook_name=hook_z, hook_head_index=None, d_in=d_z, **common
    )
    store_single_resid = ActivationsStore(
        model=ts_model,
        hook_name=hook_resid,
        hook_head_index=None,
        d_in=d_resid,
        **common,
    )
    store_multi = ActivationsStore.from_config_multi_hook(
        model=ts_model,
        hook_names=[hook_resid, hook_z],
        hook_d_ins={hook_resid: d_resid, hook_z: d_z},
        **common,
    )

    multi_iter = store_multi.get_multi_hook_data_loader()
    for _ in range(3):
        multi_batch = next(multi_iter)
        single_z_batch = next(store_single_z)
        single_resid_batch = next(store_single_resid)
        # different feature dims per hook — verifies concat-split with heterogeneous shapes
        assert multi_batch[hook_resid].shape[-1] == d_resid
        assert multi_batch[hook_z].shape[-1] == d_z
        assert_close(multi_batch[hook_z], single_z_batch)
        assert_close(multi_batch[hook_resid], single_resid_batch)


def test_multi_hook_factory_validates_inputs(
    ts_model: HookedTransformer, dataset: Dataset
):
    common = _common_kwargs(dataset)
    common.pop("dataset")
    with pytest.raises(ValueError, match="hook_names must be non-empty"):
        ActivationsStore.from_config_multi_hook(
            model=ts_model,
            dataset=dataset,
            hook_names=[],
            hook_d_ins={},
            **common,
        )
    with pytest.raises(ValueError, match="hook_d_ins missing entries"):
        ActivationsStore.from_config_multi_hook(
            model=ts_model,
            dataset=dataset,
            hook_names=["blocks.0.hook_mlp_out"],
            hook_d_ins={},
            **common,
        )


def test_get_multi_hook_data_loader_errors_on_single_hook_store(
    ts_model: HookedTransformer, dataset: Dataset
):
    store = ActivationsStore(
        model=ts_model,
        hook_name="blocks.0.hook_mlp_out",
        hook_head_index=None,
        d_in=ts_model.cfg.d_model,
        **_common_kwargs(dataset),
    )
    with pytest.raises(RuntimeError, match="from_config_multi_hook"):
        store.get_multi_hook_data_loader()
    with pytest.raises(RuntimeError, match="from_config_multi_hook"):
        store.get_multi_hook_activations(torch.zeros((2, 8), dtype=torch.long))
