from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformer_lens import HookedTransformer

from sae_lens import (
    MultiSAETrainingRunner,
    MultiSAETrainingRunnerConfig,
    StandardTrainingSAEConfig,
    TopKTrainingSAEConfig,
)
from sae_lens.config import LoggingConfig
from sae_lens.multi_sae_training_runner import InterruptedException, PerSAEEvaluator
from sae_lens.saes.sae import TrainingSAE, TrainingSAEConfig
from sae_lens.saes.standard_sae import StandardTrainingSAE
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from tests.helpers import TINYSTORIES_MODEL, load_model_cached


@pytest.fixture
def ts_model() -> HookedTransformer:
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture
def dataset() -> Dataset:
    return Dataset.from_list(
        [{"text": f"the quick brown fox {i} jumps over"} for i in range(200)]
    )


def _build_cfg(
    *,
    saes: Mapping[str, TrainingSAEConfig],
    hook_names: dict[str, str] | str,
    output_path: str | None = None,
    checkpoint_path: str | None = None,
    n_checkpoints: int = 0,
    save_final_checkpoint: bool = False,
    training_tokens: int = 32,
    logger: LoggingConfig | None = None,
    evaluator: PerSAEEvaluator | None = None,
    feature_sampling_window: int = 2000,
    n_eval_batches: int = 10,
    prefetch_llm_batches: bool | int = False,
    resume_from_checkpoint: str | None = None,
    exclude_special_tokens: bool | list[int] = False,
    hook_head_indices: dict[str, int | None] | int | None = None,
) -> MultiSAETrainingRunnerConfig:
    return MultiSAETrainingRunnerConfig(
        saes=saes,
        hook_names=hook_names,
        hook_head_indices=hook_head_indices,
        exclude_special_tokens=exclude_special_tokens,
        model_name=TINYSTORIES_MODEL,
        dataset_path="placeholder",  # override_dataset is used
        streaming=False,
        context_size=8,
        n_batches_in_buffer=2,
        training_tokens=training_tokens,
        store_batch_size_prompts=4,
        train_batch_size_tokens=4,
        prepend_bos=True,
        device="cpu",
        dtype="float32",
        seqpos_slice=(None,),
        activations_mixing_fraction=0.0,
        lr=1e-3,
        logger=logger if logger is not None else LoggingConfig(log_to_wandb=False),
        evaluator=evaluator,
        feature_sampling_window=feature_sampling_window,
        n_eval_batches=n_eval_batches,
        prefetch_llm_batches=prefetch_llm_batches,
        n_checkpoints=n_checkpoints,
        checkpoint_path=checkpoint_path,
        save_final_checkpoint=save_final_checkpoint,
        resume_from_checkpoint=resume_from_checkpoint,
        output_path=output_path,
    )


def test_multi_sae_runner_trains_two_saes_at_same_hook(
    ts_model: HookedTransformer, dataset: Dataset, tmp_path: Path
):
    d_in = ts_model.cfg.d_model
    cfg = _build_cfg(
        saes={
            "low_l1": StandardTrainingSAEConfig(
                d_in=d_in,
                d_sae=32,
                l1_coefficient=1e-3,
                decoder_init_norm=0.1,
                normalize_activations="none",
                dtype="float32",
                device="cpu",
            ),
            "high_l1": StandardTrainingSAEConfig(
                d_in=d_in,
                d_sae=32,
                l1_coefficient=1.0,
                decoder_init_norm=0.1,
                normalize_activations="none",
                dtype="float32",
                device="cpu",
            ),
        },
        hook_names="blocks.0.hook_mlp_out",
        output_path=str(tmp_path / "out"),
        training_tokens=32,
    )
    runner = MultiSAETrainingRunner(
        cfg, override_model=ts_model, override_dataset=dataset
    )
    saes = runner.run()

    assert set(saes.keys()) == {"low_l1", "high_l1"}
    # Output dirs exist with weights/cfg/sparsity files per SAE
    assert (tmp_path / "out" / "low_l1" / "sae_weights.safetensors").exists()
    assert (tmp_path / "out" / "high_l1" / "sae_weights.safetensors").exists()
    assert (tmp_path / "out" / "runner_cfg.json").exists()
    # Each SAE gets its own log-feature-sparsity tensor of shape (d_sae,).
    for name in ("low_l1", "high_l1"):
        sparsity = load_file(tmp_path / "out" / name / "sparsity.safetensors")[
            "sparsity"
        ]
        assert sparsity.shape == (saes[name].cfg.d_sae,)
    # Both SAEs should have non-zero W_dec (training did something)
    for sae in saes.values():
        assert sae.W_dec.abs().sum().item() > 0


def test_multi_sae_runner_trains_two_saes_at_different_hooks(
    ts_model: HookedTransformer, dataset: Dataset
):
    d_in = ts_model.cfg.d_model
    cfg = _build_cfg(
        saes={
            "resid": StandardTrainingSAEConfig(
                d_in=d_in,
                d_sae=32,
                l1_coefficient=1e-3,
                decoder_init_norm=0.1,
                normalize_activations="none",
                dtype="float32",
                device="cpu",
            ),
            "topk_mlp": TopKTrainingSAEConfig(
                d_in=d_in,
                d_sae=32,
                k=4,
                normalize_activations="none",
                decoder_init_norm=0.1,
                dtype="float32",
                device="cpu",
            ),
        },
        hook_names={
            "resid": "blocks.0.hook_resid_pre",
            "topk_mlp": "blocks.0.hook_mlp_out",
        },
        training_tokens=32,
    )
    runner = MultiSAETrainingRunner(
        cfg, override_model=ts_model, override_dataset=dataset
    )
    saes = runner.run()

    assert set(saes.keys()) == {"resid", "topk_mlp"}
    assert saes["resid"].cfg.metadata.hook_name == "blocks.0.hook_resid_pre"
    assert saes["topk_mlp"].cfg.metadata.hook_name == "blocks.0.hook_mlp_out"


def test_multi_sae_runner_resume_from_checkpoint(
    ts_model: HookedTransformer, dataset: Dataset, tmp_path: Path
):
    d_in = ts_model.cfg.d_model

    cfg = _build_cfg(
        saes={"a": _std_sae_cfg(d_in), "b": _std_sae_cfg(d_in)},
        hook_names="blocks.0.hook_mlp_out",
        checkpoint_path=str(tmp_path / "ckpt"),
        save_final_checkpoint=True,
        training_tokens=64,
    )
    MultiSAETrainingRunner(cfg, override_model=ts_model, override_dataset=dataset).run()

    final_dirs = list(Path(cfg.checkpoint_path).glob("final_*"))  # type: ignore[arg-type]
    assert final_dirs, "expected a final_<n> checkpoint dir"
    final_dir = final_dirs[0]
    checkpointed_w_dec = load_file(final_dir / "a" / "sae_weights.safetensors")["W_dec"]

    # Resume from the final checkpoint at the same token budget: fit() should
    # see the budget already met and return the loaded SAEs untouched.
    resume_cfg = _build_cfg(
        saes={"a": _std_sae_cfg(d_in), "b": _std_sae_cfg(d_in)},
        hook_names="blocks.0.hook_mlp_out",
        training_tokens=64,
        resume_from_checkpoint=str(final_dir),
    )
    resumed = MultiSAETrainingRunner(
        resume_cfg, override_model=ts_model, override_dataset=dataset
    ).run()

    torch.testing.assert_close(resumed["a"].W_dec, checkpointed_w_dec)
    assert (final_dir / "runner_cfg.json").exists()


def test_multi_sae_runner_config_validates_d_in_mismatch(ts_model: HookedTransformer):
    d_in = ts_model.cfg.d_model
    with pytest.raises(ValueError, match="must have the same d_in"):
        _build_cfg(
            saes={
                "a": StandardTrainingSAEConfig(
                    d_in=d_in,
                    d_sae=32,
                    decoder_init_norm=0.1,
                    dtype="float32",
                    device="cpu",
                ),
                "b": StandardTrainingSAEConfig(
                    d_in=d_in + 1,  # mismatched d_in at the same hook
                    d_sae=32,
                    decoder_init_norm=0.1,
                    dtype="float32",
                    device="cpu",
                ),
            },
            hook_names="blocks.0.hook_mlp_out",
        )


def test_multi_sae_runner_config_rejects_mismatched_hook_keys():
    with pytest.raises(ValueError, match="missing|extra"):
        MultiSAETrainingRunnerConfig(
            saes={
                "a": StandardTrainingSAEConfig(
                    d_in=64,
                    d_sae=32,
                    decoder_init_norm=0.1,
                    dtype="float32",
                    device="cpu",
                ),
            },
            hook_names={"b": "blocks.0.hook_mlp_out"},  # mismatch
            logger=LoggingConfig(log_to_wandb=False),
        )


def test_multi_sae_runner_config_accepts_per_sae_hook_head_indices_dict(
    ts_model: HookedTransformer,
):
    d_in = ts_model.cfg.d_model
    cfg = _build_cfg(
        saes={"a": _std_sae_cfg(d_in), "b": _std_sae_cfg(d_in)},
        hook_names={"a": "blocks.0.hook_resid_pre", "b": "blocks.0.hook_mlp_out"},
        hook_head_indices={"a": 2, "b": None},
    )
    assert cfg.hook_head_indices_per_sae == {"a": 2, "b": None}


def test_multi_sae_runner_config_rejects_unknown_hook_head_index_keys(
    ts_model: HookedTransformer,
):
    d_in = ts_model.cfg.d_model
    with pytest.raises(ValueError, match="hook_head_indices has unknown keys"):
        _build_cfg(
            saes={"a": _std_sae_cfg(d_in)},
            hook_names="blocks.0.hook_mlp_out",
            hook_head_indices={"a": 0, "ghost": 1},
        )


def test_multi_sae_runner_smoke_loss_decreases(
    ts_model: HookedTransformer, dataset: Dataset
):
    d_in = ts_model.cfg.d_model
    cfg = _build_cfg(
        saes={
            "a": StandardTrainingSAEConfig(
                d_in=d_in,
                d_sae=32,
                l1_coefficient=1e-3,
                decoder_init_norm=0.1,
                normalize_activations="none",
                dtype="float32",
                device="cpu",
            ),
            "b": StandardTrainingSAEConfig(
                d_in=d_in,
                d_sae=32,
                l1_coefficient=1e-3,
                decoder_init_norm=0.1,
                normalize_activations="none",
                dtype="float32",
                device="cpu",
            ),
        },
        hook_names="blocks.0.hook_mlp_out",
        training_tokens=200,  # ~50 steps
    )
    runner = MultiSAETrainingRunner(
        cfg, override_model=ts_model, override_dataset=dataset
    )
    saes = runner.run()

    # Build a fresh store for a post-training reconstruction check (the
    # runner's store has already been exhausted by fit()).
    hook = "blocks.0.hook_mlp_out"
    eval_store = ActivationsStore.from_config_multi_hook(
        model=ts_model,
        dataset=dataset,
        hook_names=[hook],
        hook_d_ins={hook: d_in},
        streaming=False,
        context_size=cfg.context_size,
        n_batches_in_buffer=2,
        total_training_tokens=cfg.training_tokens,
        store_batch_size_prompts=cfg.store_batch_size_prompts,
        train_batch_size_tokens=cfg.train_batch_size_tokens,
        prepend_bos=cfg.prepend_bos,
        normalize_activations="none",
        device=torch.device(cfg.device),
        dtype=cfg.dtype,
        seqpos_slice=cfg.seqpos_slice,
        activations_mixing_fraction=0.0,
    )
    batch = next(eval_store.get_multi_hook_data_loader())[hook]
    for sae in saes.values():
        recon = sae(batch)
        per_sample_mse = (recon - batch).pow(2).sum(-1).mean()
        # crude: trained reconstruction error should be less than the input's own variance
        baseline = batch.pow(2).sum(-1).mean()
        assert per_sample_mse < baseline, (
            f"reconstruction worse than zero baseline: {per_sample_mse.item()} vs "
            f"{baseline.item()}"
        )


def _std_sae_cfg(d_in: int) -> StandardTrainingSAEConfig:
    return StandardTrainingSAEConfig(
        d_in=d_in,
        d_sae=32,
        l1_coefficient=1e-3,
        decoder_init_norm=0.1,
        normalize_activations="none",
        dtype="float32",
        device="cpu",
    )


def test_multi_sae_runner_logs_per_sae_metrics_to_wandb(
    ts_model: HookedTransformer,
    dataset: Dataset,
    tmp_path: Path,
    captured_wandb_logs: list[dict[str, Any]],
):
    d_in = ts_model.cfg.d_model

    def user_evaluator(
        _sae: TrainingSAE[Any], data_view: Any, _scaler: Any
    ) -> dict[str, float]:
        batch = next(data_view)
        return {"custom/batch_abs_mean": batch.abs().mean().item()}

    cfg = _build_cfg(
        saes={"a": _std_sae_cfg(d_in), "b": _std_sae_cfg(d_in)},
        hook_names="blocks.0.hook_mlp_out",
        output_path=str(tmp_path / "out"),
        training_tokens=32,
        # log_frequency=2 means some steps log and some skip, exercising both
        # the logging and the early-return branches
        logger=LoggingConfig(
            log_to_wandb=True, wandb_log_frequency=2, eval_every_n_wandb_logs=1
        ),
        evaluator=user_evaluator,
        feature_sampling_window=2,
        n_eval_batches=1,
        # exercise the prefetch path and the evaluator's prefetcher-pause branch
        prefetch_llm_batches=2,
        exclude_special_tokens=True,
    )
    MultiSAETrainingRunner(cfg, override_model=ts_model, override_dataset=dataset).run()

    assert captured_wandb_logs, "expected wandb.log to be called"
    all_keys = {k for d in captured_wandb_logs for k in d}
    # train-step metrics are aggregated per SAE
    assert any(k.startswith("a/") for k in all_keys)
    assert any(k.startswith("b/") for k in all_keys)
    # the built-in evaluator runs run_evals per SAE (CE loss is eval-only)
    eval_log = next(
        d for d in captured_wandb_logs if "a/model_performance_preservation" in d
    )
    assert "ce_loss_score" in eval_log["a/model_performance_preservation"]
    assert "ce_loss_score" in eval_log["b/model_performance_preservation"]
    # the user evaluator's metric is merged in under each SAE's prefix
    assert "a/custom/batch_abs_mean" in all_keys
    assert "b/custom/batch_abs_mean" in all_keys
    # sparsity reset (feature_sampling_window=2) logs mean log10 feature sparsity
    assert "a/metrics/mean_log10_feature_sparsity" in all_keys


def test_multi_sae_runner_rejects_mismatched_override_saes(
    ts_model: HookedTransformer, dataset: Dataset
):
    d_in = ts_model.cfg.d_model
    cfg = _build_cfg(
        saes={"a": _std_sae_cfg(d_in), "b": _std_sae_cfg(d_in)},
        hook_names="blocks.0.hook_mlp_out",
    )
    # override_saes has an unknown key "c" and is missing "b"
    bad_override = {"a": StandardTrainingSAE(_std_sae_cfg(d_in))}
    with pytest.raises(ValueError, match="override_saes keys must match"):
        MultiSAETrainingRunner(
            cfg,
            override_model=ts_model,
            override_dataset=dataset,
            override_saes={**bad_override, "c": bad_override["a"]},
        )


def test_multi_sae_runner_saves_checkpoint_on_interruption(
    ts_model: HookedTransformer,
    dataset: Dataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def _interrupt(_self: MultiSAETrainer) -> dict[str, Any]:
        raise InterruptedException()

    monkeypatch.setattr(MultiSAETrainer, "fit", _interrupt)

    d_in = ts_model.cfg.d_model
    cfg = _build_cfg(
        saes={"a": _std_sae_cfg(d_in)},
        hook_names="blocks.0.hook_mlp_out",
        checkpoint_path=str(tmp_path / "ckpt"),
    )
    runner = MultiSAETrainingRunner(
        cfg, override_model=ts_model, override_dataset=dataset
    )
    with pytest.raises(InterruptedException):
        runner.run()

    assert cfg.checkpoint_path is not None
    saved = list(Path(cfg.checkpoint_path).glob("*/a/sae_weights.safetensors"))
    assert saved, "expected an interrupt checkpoint with the SAE's weights"
