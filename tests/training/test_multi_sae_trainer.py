import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.standard_sae import StandardTrainingSAE, StandardTrainingSAEConfig
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from sae_lens.training.sae_trainer import SAETrainer
from tests.helpers import (
    TINYSTORIES_MODEL,
    assert_close,
    load_model_cached,
    random_params,
)


@pytest.fixture
def ts_model() -> HookedTransformer:
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture
def dataset() -> Dataset:
    return Dataset.from_list(
        [{"text": f"row {i} the quick brown fox jumps"} for i in range(200)]
    )


def _common_store_kwargs(dataset: Dataset) -> dict[str, Any]:
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


def _make_sae(d_in: int) -> StandardTrainingSAE:
    cfg = StandardTrainingSAEConfig(
        d_in=d_in,
        d_sae=32,
        l1_coefficient=1e-3,
        decoder_init_norm=0.1,
        normalize_activations="none",
        dtype="float32",
        device="cpu",
    )
    sae = StandardTrainingSAE(cfg)
    random_params(sae)
    return sae


def _trainer_cfg(total_samples: int) -> SAETrainerConfig:
    return SAETrainerConfig(
        total_training_samples=total_samples,
        train_batch_size_samples=4,
        lr=1e-3,
        lr_end=1e-4,
        device="cpu",
        n_checkpoints=0,
        save_final_checkpoint=False,
        logger=LoggingConfig(log_to_wandb=False),
        n_batches_for_norm_estimate=2,
    )


def test_multi_sae_trainer_matches_two_independent_single_trainers(
    ts_model: HookedTransformer, dataset: Dataset
):
    hook_a = "blocks.0.hook_resid_pre"
    hook_b = "blocks.0.hook_mlp_out"
    d_in = ts_model.cfg.d_model
    n_steps = 5
    common = _common_store_kwargs(dataset)

    sae_a_proto = _make_sae(d_in)
    sae_b_proto = _make_sae(d_in)
    sae_a_single = copy.deepcopy(sae_a_proto)
    sae_b_single = copy.deepcopy(sae_b_proto)
    sae_a_multi = copy.deepcopy(sae_a_proto)
    sae_b_multi = copy.deepcopy(sae_b_proto)

    cfg = _trainer_cfg(total_samples=n_steps * 4)

    store_a = ActivationsStore(
        model=ts_model, hook_name=hook_a, hook_head_index=None, d_in=d_in, **common
    )
    store_b = ActivationsStore(
        model=ts_model, hook_name=hook_b, hook_head_index=None, d_in=d_in, **common
    )
    trainer_a = SAETrainer(cfg=cfg, sae=sae_a_single, data_provider=store_a)
    trainer_b = SAETrainer(cfg=cfg, sae=sae_b_single, data_provider=store_b)
    trainer_a.fit()
    trainer_b.fit()

    multi_store = ActivationsStore.from_config_multi_hook(
        model=ts_model,
        hook_names=[hook_a, hook_b],
        hook_d_ins={hook_a: d_in, hook_b: d_in},
        **common,
    )
    multi_trainer = MultiSAETrainer(
        cfg=cfg,
        saes={"a": sae_a_multi, "b": sae_b_multi},
        hook_names={"a": hook_a, "b": hook_b},
        data_provider=multi_store.get_multi_hook_data_loader(),
    )
    multi_trainer.fit()

    for p_single, p_multi in zip(
        sae_a_single.parameters(), sae_a_multi.parameters(), strict=True
    ):
        assert_close(p_single, p_multi)
    for p_single, p_multi in zip(
        sae_b_single.parameters(), sae_b_multi.parameters(), strict=True
    ):
        assert_close(p_single, p_multi)

    # Optimizer state should also match.
    opt_single = trainer_a.optimizer.state_dict()["state"]
    opt_multi = multi_trainer.trainers["a"].optimizer.state_dict()["state"]
    for pid in opt_single:
        for key, val in opt_single[pid].items():
            mv = opt_multi[pid][key]
            if torch.is_tensor(val):
                assert_close(val, mv)
            else:
                assert val == mv

    # Counters mirror correctly.
    assert multi_trainer.n_training_samples == trainer_a.n_training_samples
    assert multi_trainer.n_training_steps == trainer_a.n_training_steps


def test_multi_sae_trainer_two_saes_at_same_hook_train_independently(
    ts_model: HookedTransformer, dataset: Dataset
):
    hook = "blocks.0.hook_mlp_out"
    d_in = ts_model.cfg.d_model
    n_steps = 4
    common = _common_store_kwargs(dataset)

    sae_a = _make_sae(d_in)
    sae_b = _make_sae(d_in)

    multi_store = ActivationsStore.from_config_multi_hook(
        model=ts_model,
        hook_names=[hook],
        hook_d_ins={hook: d_in},
        **common,
    )
    multi_trainer = MultiSAETrainer(
        cfg=_trainer_cfg(total_samples=n_steps * 4),
        saes={"a": sae_a, "b": sae_b},
        hook_names={"a": hook, "b": hook},
        data_provider=multi_store.get_multi_hook_data_loader(),
    )
    multi_trainer.fit()

    # SAEs have independent params. They received the same activations but
    # had different starting weights, so should still differ.
    a_params = list(sae_a.parameters())
    b_params = list(sae_b.parameters())
    differs = any(
        not torch.allclose(a, b) for a, b in zip(a_params, b_params, strict=True)
    )
    assert differs, "two SAEs at same hook collapsed to identical weights"


def test_multi_sae_trainer_runs_with_normalize_activations(
    ts_model: HookedTransformer, dataset: Dataset
):
    hook = "blocks.0.hook_mlp_out"
    d_in = ts_model.cfg.d_model
    common = _common_store_kwargs(dataset)

    cfg_a = StandardTrainingSAEConfig(
        d_in=d_in,
        d_sae=32,
        l1_coefficient=1e-3,
        decoder_init_norm=0.1,
        normalize_activations="expected_average_only_in",
        dtype="float32",
        device="cpu",
    )
    sae = StandardTrainingSAE(cfg_a)

    multi_store = ActivationsStore.from_config_multi_hook(
        model=ts_model,
        hook_names=[hook],
        hook_d_ins={hook: d_in},
        **common,
    )
    multi_trainer = MultiSAETrainer(
        cfg=_trainer_cfg(total_samples=4 * 4),
        saes={"a": sae},
        hook_names={"a": hook},
        data_provider=multi_store.get_multi_hook_data_loader(),
    )
    multi_trainer.fit()

    # After training, the scaling factor should be folded back to None.
    assert multi_trainer.trainers["a"].activation_scaler.scaling_factor is None


def test_multi_sae_trainer_save_and_load_round_trip(
    ts_model: HookedTransformer, dataset: Dataset, tmp_path: Path
):
    hook_a = "blocks.0.hook_resid_pre"
    hook_b = "blocks.0.hook_mlp_out"
    d_in = ts_model.cfg.d_model
    common = _common_store_kwargs(dataset)

    sae_a = _make_sae(d_in)
    sae_b = _make_sae(d_in)

    multi_store = ActivationsStore.from_config_multi_hook(
        model=ts_model,
        hook_names=[hook_a, hook_b],
        hook_d_ins={hook_a: d_in, hook_b: d_in},
        **common,
    )

    cfg = _trainer_cfg(total_samples=4 * 4)
    cfg.checkpoint_path = str(tmp_path / "ckpt")

    trainer = MultiSAETrainer(
        cfg=cfg,
        saes={"a": sae_a, "b": sae_b},
        hook_names={"a": hook_a, "b": hook_b},
        data_provider=multi_store.get_multi_hook_data_loader(),
    )
    # Run a few real steps so optimizer/sparsity state is non-trivial.
    trainer.fit()
    saved_a_w_enc = sae_a.W_enc.clone()
    saved_b_w_enc = sae_b.W_enc.clone()
    saved_n_samples = trainer.n_training_samples

    trainer.save_checkpoint(checkpoint_name="ckpt0")
    checkpoint_dir = tmp_path / "ckpt" / "ckpt0"
    assert (checkpoint_dir / "a" / "sae_weights.safetensors").exists()
    assert (checkpoint_dir / "b" / "sae_weights.safetensors").exists()
    assert (checkpoint_dir / "a" / "trainer_state.pt").exists()
    assert (checkpoint_dir / "b" / "trainer_state.pt").exists()

    # Build a fresh trainer with new SAEs, load checkpoint, verify state restored.
    fresh_a = _make_sae(d_in)
    fresh_b = _make_sae(d_in)
    fresh_trainer = MultiSAETrainer(
        cfg=cfg,
        saes={"a": fresh_a, "b": fresh_b},
        hook_names={"a": hook_a, "b": hook_b},
        data_provider=multi_store.get_multi_hook_data_loader(),
    )
    fresh_trainer.load_checkpoint(checkpoint_dir)

    assert_close(fresh_a.W_enc, saved_a_w_enc)
    assert_close(fresh_b.W_enc, saved_b_w_enc)
    assert fresh_trainer.n_training_samples == saved_n_samples


def test_multi_sae_trainer_load_checkpoint_errors_on_missing_sae_subdir(
    ts_model: HookedTransformer, dataset: Dataset, tmp_path: Path
):
    hook = "blocks.0.hook_mlp_out"
    d_in = ts_model.cfg.d_model
    multi_store = ActivationsStore.from_config_multi_hook(
        model=ts_model,
        hook_names=[hook],
        hook_d_ins={hook: d_in},
        **_common_store_kwargs(dataset),
    )
    trainer = MultiSAETrainer(
        cfg=_trainer_cfg(total_samples=4),
        saes={"a": _make_sae(d_in), "b": _make_sae(d_in)},
        hook_names={"a": hook, "b": hook},
        data_provider=multi_store.get_multi_hook_data_loader(),
    )
    # the directory exists but has no per-SAE subdirectories
    (tmp_path / "empty_ckpt").mkdir()
    with pytest.raises(FileNotFoundError, match="missing per-SAE subdirectories"):
        trainer.load_checkpoint(tmp_path / "empty_ckpt")


def test_multi_sae_trainer_update_pbar_refreshes_on_interval(
    ts_model: HookedTransformer, dataset: Dataset
):
    hook = "blocks.0.hook_mlp_out"
    d_in = ts_model.cfg.d_model
    multi_store = ActivationsStore.from_config_multi_hook(
        model=ts_model,
        hook_names=[hook],
        hook_d_ins={hook: d_in},
        **_common_store_kwargs(dataset),
    )
    trainer = MultiSAETrainer(
        cfg=_trainer_cfg(total_samples=4),
        saes={"a": _make_sae(d_in)},
        hook_names={"a": hook},
        data_provider=multi_store.get_multi_hook_data_loader(),
    )
    multi_batch = next(trainer.data_provider)
    step_outputs = {"a": trainer.trainers["a"].step(multi_batch[hook])}

    pbar = tqdm(total=1000)
    trainer.n_training_steps = 100  # a multiple of the 100-step refresh interval
    trainer._update_pbar(step_outputs, pbar)  # type: ignore[reportPrivateUsage]
    assert pbar.desc.startswith("100|")
    pbar.close()


def test_multi_sae_trainer_validates_hook_keys(ts_model: HookedTransformer):
    cfg = _trainer_cfg(total_samples=4)
    sae = _make_sae(ts_model.cfg.d_model)
    with pytest.raises(ValueError, match="identical keys"):
        MultiSAETrainer(
            cfg=cfg,
            saes={"a": sae},
            hook_names={"b": "blocks.0.hook_mlp_out"},
            data_provider=iter(()),  # type: ignore[arg-type]
        )


def test_multi_sae_trainer_per_sae_loss_decreases_over_training(
    ts_model: HookedTransformer, dataset: Dataset
):
    hook = "blocks.0.hook_mlp_out"
    d_in = ts_model.cfg.d_model
    common = _common_store_kwargs(dataset)

    sae_a = _make_sae(d_in)
    sae_b = _make_sae(d_in)
    multi_store = ActivationsStore.from_config_multi_hook(
        model=ts_model,
        hook_names=[hook],
        hook_d_ins={hook: d_in},
        **common,
    )
    trainer = MultiSAETrainer(
        cfg=_trainer_cfg(total_samples=4),  # 1 step
        saes={"a": sae_a, "b": sae_b},
        hook_names={"a": hook, "b": hook},
        data_provider=multi_store.get_multi_hook_data_loader(),
    )
    # Pull one batch and step both SAEs on the same activations 5 times.
    multi_batch = next(trainer.data_provider)
    same_input = multi_batch[hook]
    losses_a: list[float] = []
    losses_b: list[float] = []
    for _ in range(5):
        out_a = trainer.trainers["a"].step(same_input.clone())
        out_b = trainer.trainers["b"].step(same_input.clone())
        losses_a.append(out_a.loss.item())
        losses_b.append(out_b.loss.item())
    for prev, nxt in zip(losses_a[:-1], losses_a[1:]):
        assert nxt < prev
    for prev, nxt in zip(losses_b[:-1], losses_b[1:]):
        assert nxt < prev


def test_multi_sae_trainer_writes_n_checkpoints_during_fit(
    ts_model: HookedTransformer, dataset: Dataset, tmp_path: Path
):
    hook = "blocks.0.hook_mlp_out"
    d_in = ts_model.cfg.d_model
    common = _common_store_kwargs(dataset)
    sae_a = _make_sae(d_in)
    sae_b = _make_sae(d_in)

    multi_store = ActivationsStore.from_config_multi_hook(
        model=ts_model,
        hook_names=[hook],
        hook_d_ins={hook: d_in},
        **common,
    )
    cfg = _trainer_cfg(total_samples=12 * 4)  # 12 steps
    cfg.checkpoint_path = str(tmp_path / "ckpts")
    cfg.n_checkpoints = 2
    cfg.save_final_checkpoint = True

    trainer = MultiSAETrainer(
        cfg=cfg,
        saes={"a": sae_a, "b": sae_b},
        hook_names={"a": hook, "b": hook},
        data_provider=multi_store.get_multi_hook_data_loader(),
    )
    trainer.fit()

    base = Path(cfg.checkpoint_path)
    sae_weight_files = list(base.glob("**/sae_weights.safetensors"))
    # 2 intermediate + 1 final = 3 checkpoint dirs, each with 2 SAEs
    assert (
        len(sae_weight_files) == 3 * 2
    ), f"expected 6 sae_weights files (3 ckpts × 2 SAEs), got {len(sae_weight_files)}"


def test_multi_sae_trainer_skips_checkpoints_when_path_none_and_no_wandb(
    ts_model: HookedTransformer, dataset: Dataset
):
    hook = "blocks.0.hook_mlp_out"
    d_in = ts_model.cfg.d_model
    common = _common_store_kwargs(dataset)
    sae_a = _make_sae(d_in)

    multi_store = ActivationsStore.from_config_multi_hook(
        model=ts_model,
        hook_names=[hook],
        hook_d_ins={hook: d_in},
        **common,
    )
    cfg = _trainer_cfg(total_samples=4 * 4)
    cfg.checkpoint_path = None
    cfg.n_checkpoints = 2
    cfg.save_final_checkpoint = True

    received_paths: list[Any] = []

    def save_fn(checkpoint_path: Any) -> None:
        received_paths.append(checkpoint_path)

    trainer = MultiSAETrainer(
        cfg=cfg,
        saes={"a": sae_a},
        hook_names={"a": hook},
        data_provider=multi_store.get_multi_hook_data_loader(),
        save_checkpoint_fn=save_fn,
    )
    trainer.fit()
    # save_checkpoint should have been invoked but each call passes path=None
    assert all(p is None for p in received_paths)
    assert len(received_paths) >= 1  # at least the final-checkpoint call


def test_multi_sae_trainer_advances_counters_in_lockstep(
    ts_model: HookedTransformer, dataset: Dataset
):
    hook = "blocks.0.hook_mlp_out"
    d_in = ts_model.cfg.d_model
    common = _common_store_kwargs(dataset)
    sae_a = _make_sae(d_in)
    sae_b = _make_sae(d_in)

    multi_store = ActivationsStore.from_config_multi_hook(
        model=ts_model,
        hook_names=[hook],
        hook_d_ins={hook: d_in},
        **common,
    )
    trainer = MultiSAETrainer(
        cfg=_trainer_cfg(total_samples=8 * 4),
        saes={"a": sae_a, "b": sae_b},
        hook_names={"a": hook, "b": hook},
        data_provider=multi_store.get_multi_hook_data_loader(),
    )
    trainer.fit()
    assert trainer.trainers["a"].n_training_steps == trainer.n_training_steps
    assert trainer.trainers["b"].n_training_steps == trainer.n_training_steps
    assert trainer.trainers["a"].n_training_samples == trainer.n_training_samples
    assert trainer.trainers["b"].n_training_samples == trainer.n_training_samples
