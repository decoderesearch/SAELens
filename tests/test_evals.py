import argparse
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.evals import (
    EvalConfig,
    _kl,
    all_loadable_saes,
    get_downstream_reconstruction_metrics,
    get_eval_everything_config,
    get_saes_from_regex,
    get_sparsity_and_variance_metrics,
    process_args,
    process_results,
    run_evals,
    run_evals_cli,
    run_evaluations,
)
from sae_lens.load_model import load_model
from sae_lens.loading.pretrained_saes_directory import PretrainedSAELookup
from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
)
from sae_lens.saes.sae import SAE, TrainingSAE
from sae_lens.saes.standard_sae import StandardSAE, StandardTrainingSAE
from sae_lens.saes.topk_sae import TopKTrainingSAE
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.activations_store import ActivationsStore
from tests.helpers import (
    NEEL_NANDA_C4_10K_DATASET,
    TINYSTORIES_MODEL,
    build_batchtopk_runner_cfg,
    build_runner_cfg,
    build_topk_runner_cfg,
    load_model_cached,
    random_params,
)

TRAINER_EVAL_CONFIG = EvalConfig(
    n_eval_reconstruction_batches=10,
    compute_ce_loss=True,
    n_eval_sparsity_variance_batches=1,
    compute_l2_norms=True,
)


@pytest.fixture
def example_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 20
    )


# not sure why we have NaNs in the feature metrics, but this is a quick fix for tests
def _replace_nan(list: list[float]) -> list[float]:
    return [0 if math.isnan(x) else x for x in list]


@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "NeelNanda/c4-10k",
            "hook_name": "blocks.1.hook_resid_pre",
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
            "hook_name": "blocks.1.hook_resid_pre",
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "NeelNanda/c4-10k",
            "hook_name": "blocks.1.attn.hook_z",
            "d_in": 16 * 4,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "NeelNanda/c4-10k",
            "hook_name": "blocks.1.attn.hook_q",
            "d_in": 16 * 4,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "NeelNanda/c4-10k",
            "hook_name": "blocks.1.attn.hook_q",
            "d_in": 4,
            "hook_head_index": 2,
        },
    ],
    ids=[
        "tiny-stories-1M-resid-pre",
        "tiny-stories-1M-resid-pre-pretokenized",
        "tiny-stories-1M-hook-z",
        "tiny-stories-1M-hook-q",
        "tiny-stories-1M-hook-q-head-index-2",
    ],
)
def cfg(request: pytest.FixtureRequest):
    """
    Pytest fixture to create a mock instance of LanguageModelSAERunnerConfig.
    """
    params = request.param
    return build_runner_cfg(**params)


@pytest.fixture
def model():
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture
def activation_store(model: HookedTransformer, cfg: LanguageModelSAERunnerConfig[Any]):
    return ActivationsStore.from_config(
        model, cfg, override_dataset=Dataset.from_list([{"text": "hello world"}] * 2000)
    )


@pytest.fixture
def base_sae(training_sae: TrainingSAE[Any]):  # type: ignore
    return SAE.from_dict(training_sae.cfg.get_inference_sae_cfg_dict())


@pytest.fixture
def training_sae(cfg: LanguageModelSAERunnerConfig[Any]):  # type: ignore
    return TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())


all_possible_keys = [
    "model_behavior_preservation",
    "model_performance_preservation",
    "reconstruction_quality",
    "shrinkage",
    "sparsity",
    "token_stats",
]

all_featurewise_keys_expected = [
    "feature_density",
    "consistent_activation_heuristic",
    "encoder_bias",
    "encoder_decoder_cosine_sim",
    "encoder_norm",
]


def test_run_evals_base_sae(
    base_sae: SAE[Any],
    activation_store: ActivationsStore,
    model: HookedTransformer,
):
    eval_metrics, _ = run_evals(
        sae=base_sae,
        activation_store=activation_store,
        activation_scaler=ActivationScaler(),
        model=model,
        eval_config=get_eval_everything_config(),
    )

    assert set(eval_metrics.keys()).issubset(set(all_possible_keys))
    assert len(eval_metrics) > 0


@pytest.mark.parametrize("use_sparse_activations", [True, False])
def test_run_evals_sparse_topk_sae(
    model: HookedTransformer,
    use_sparse_activations: bool,
):
    cfg = build_topk_runner_cfg(
        use_sparse_activations=use_sparse_activations,
        model_name="tiny-stories-1M",
        dataset_path="roneneldan/TinyStories",
        hook_name="blocks.1.hook_resid_pre",
        d_in=64,
    )
    sae = TopKTrainingSAE(cfg.sae)
    activation_store = ActivationsStore.from_config(
        model, cfg, override_dataset=Dataset.from_list([{"text": "hello world"}] * 2000)
    )
    eval_metrics, _ = run_evals(
        sae=sae,
        activation_store=activation_store,
        activation_scaler=ActivationScaler(),
        model=model,
        eval_config=get_eval_everything_config(),
    )

    assert set(eval_metrics.keys()).issubset(set(all_possible_keys))
    assert len(eval_metrics) > 0


def test_run_evals_training_sae(
    training_sae: TrainingSAE[Any],
    activation_store: ActivationsStore,
    model: HookedTransformer,
):
    eval_metrics, feature_metrics = run_evals(
        sae=training_sae,
        activation_store=activation_store,
        activation_scaler=ActivationScaler(),
        model=model,
        eval_config=get_eval_everything_config(),
    )

    assert set(eval_metrics.keys()).issubset(set(all_possible_keys))
    assert len(eval_metrics) > 0
    assert set(feature_metrics.keys()).issubset(set(all_featurewise_keys_expected))


def test_run_evals_training_sae_ignore_bos(
    training_sae: TrainingSAE[Any],
    activation_store: ActivationsStore,
    model: HookedTransformer,
):
    eval_metrics, _ = run_evals(
        sae=training_sae,
        activation_store=activation_store,
        activation_scaler=ActivationScaler(),
        model=model,
        eval_config=get_eval_everything_config(),
        exclude_special_tokens={
            model.tokenizer.bos_token_id,  # type: ignore
            model.tokenizer.eos_token_id,  # type: ignore
            model.tokenizer.pad_token_id,  # type: ignore
        },
    )

    assert set(eval_metrics.keys()).issubset(set(all_possible_keys))
    assert len(eval_metrics) > 0


def test_training_eval_config(
    base_sae: SAE[Any],
    activation_store: ActivationsStore,
    model: HookedTransformer,
):
    expected_keys = [
        "model_performance_preservation",
        "shrinkage",
        "token_stats",
    ]
    eval_config = TRAINER_EVAL_CONFIG
    eval_metrics, _ = run_evals(
        sae=base_sae,
        activation_store=activation_store,
        activation_scaler=ActivationScaler(),
        model=model,
        eval_config=eval_config,
    )
    assert set(eval_metrics.keys()) == set(expected_keys)


def test_training_eval_config_ignore_control_tokens(
    base_sae: SAE[Any],
    activation_store: ActivationsStore,
    model: HookedTransformer,
):
    expected_keys = [
        "model_performance_preservation",
        "shrinkage",
        "token_stats",
    ]
    eval_config = TRAINER_EVAL_CONFIG
    eval_metrics, _ = run_evals(
        sae=base_sae,
        activation_store=activation_store,
        activation_scaler=ActivationScaler(),
        model=model,
        eval_config=eval_config,
        exclude_special_tokens={
            model.tokenizer.pad_token_id,  # type: ignore
            model.tokenizer.eos_token_id,  # type: ignore
            model.tokenizer.bos_token_id,  # type: ignore
        },
    )
    assert set(eval_metrics.keys()) == set(expected_keys)


def test_run_empty_evals(
    base_sae: SAE[Any],
    activation_store: ActivationsStore,
    model: HookedTransformer,
):
    empty_config = EvalConfig(
        n_eval_reconstruction_batches=0,
        n_eval_sparsity_variance_batches=0,
        compute_ce_loss=False,
        compute_kl=False,
        compute_l2_norms=False,
        compute_sparsity_metrics=False,
        compute_variance_metrics=False,
        compute_featurewise_density_statistics=False,
    )
    eval_metrics, feature_metrics = run_evals(
        sae=base_sae,
        activation_store=activation_store,
        activation_scaler=ActivationScaler(),
        model=model,
        eval_config=empty_config,
    )

    assert len(eval_metrics) == 1, "Expected only token_stats in eval_metrics"
    assert "token_stats" in eval_metrics, "Expected token_stats in eval_metrics"
    assert len(feature_metrics) == 0, "Expected empty feature_metrics"


@pytest.fixture
def mock_args():
    args = argparse.Namespace()
    args.sae_regex_pattern = "test_pattern"
    args.sae_block_pattern = "test_block"
    args.num_eval_batches = 2
    args.batch_size_prompts = 4
    args.eval_batch_size_prompts = 4
    args.n_eval_reconstruction_batches = 1
    args.n_eval_sparsity_variance_batches = 1
    args.datasets = ["test_dataset"]
    args.ctx_lens = [64]
    args.output_dir = "test_output"
    args.dataset_trust_remote_code = False
    args.verbose = False
    return args


@patch("sae_lens.evals.get_saes_from_regex")
@patch("sae_lens.evals.multiple_evals")
def test_run_evaluations(
    mock_multiple_evals: MagicMock,
    mock_get_saes_from_regex: MagicMock,
    mock_args: argparse.Namespace,
):
    mock_get_saes_from_regex.return_value = [
        ("release1", "sae1", 0.8, 10),
        ("release2", "sae2", 0.7, 8),
    ]
    mock_multiple_evals.return_value = [{"test": "result"}]

    result = run_evaluations(mock_args)

    mock_get_saes_from_regex.assert_called_once_with(
        mock_args.sae_regex_pattern, mock_args.sae_block_pattern
    )
    mock_multiple_evals.assert_called_once_with(
        sae_regex_pattern=mock_args.sae_regex_pattern,
        sae_block_pattern=mock_args.sae_block_pattern,
        eval_batch_size_prompts=mock_args.eval_batch_size_prompts,
        n_eval_reconstruction_batches=mock_args.n_eval_reconstruction_batches,
        n_eval_sparsity_variance_batches=mock_args.n_eval_sparsity_variance_batches,
        datasets=mock_args.datasets,
        ctx_lens=mock_args.ctx_lens,
        output_dir=mock_args.output_dir,
        verbose=mock_args.verbose,
        dataset_trust_remote_code=mock_args.dataset_trust_remote_code,
    )
    assert result == [{"test": "result"}]


def test_process_results(tmp_path: Path):
    eval_results = [
        {
            "unique_id": "test-sae",
            "eval_cfg": {"context_size": 64, "dataset": "test/dataset"},
            "metrics": {"metric1": 0.5, "metric2": 0.7},
            "sae_cfg": {"config1": "value1"},
        }
    ]
    output_dir = tmp_path / "test_output"

    process_results(eval_results, str(output_dir))  # type: ignore

    # Check if individual JSON file is created
    individual_json_path = output_dir / "test-sae_64_test_dataset.json"
    assert individual_json_path.exists()
    with open(individual_json_path) as f:
        assert json.load(f) == eval_results[0]

    # Check if combined JSON file is created
    combined_json_path = output_dir / "all_eval_results.json"
    assert combined_json_path.exists()
    with open(combined_json_path) as f:
        assert json.load(f) == eval_results

    # Check if CSV file is created
    csv_path = output_dir / "all_eval_results.csv"
    assert csv_path.exists()


def test_get_downstream_reconstruction_metrics_with_hf_model_gives_same_results_as_tlens_model(
    gpt2_res_jb_l4_sae: SAE[Any], example_dataset: Dataset
):
    hf_model = load_model(
        model_class_name="AutoModelForCausalLM",
        model_name="gpt2",
        device="cpu",
    )
    tlens_model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")

    cfg = build_runner_cfg(hook_name="transformer.h.3")
    gpt2_res_jb_l4_sae.cfg.metadata.hook_name = "transformer.h.3"
    hf_store = ActivationsStore.from_config(
        hf_model, cfg, override_dataset=example_dataset
    )
    hf_metrics = get_downstream_reconstruction_metrics(
        sae=gpt2_res_jb_l4_sae,
        model=hf_model,
        activation_store=hf_store,
        activation_scaler=ActivationScaler(),
        compute_kl=True,
        compute_ce_loss=True,
        n_batches=1,
        eval_batch_size_prompts=4,
    )

    cfg = build_runner_cfg(hook_name="blocks.4.hook_resid_pre")
    gpt2_res_jb_l4_sae.cfg.metadata.hook_name = "blocks.4.hook_resid_pre"
    tlens_store = ActivationsStore.from_config(
        tlens_model, cfg, override_dataset=example_dataset
    )
    tlens_metrics = get_downstream_reconstruction_metrics(
        sae=gpt2_res_jb_l4_sae,
        model=tlens_model,
        activation_store=tlens_store,
        activation_scaler=ActivationScaler(),
        compute_kl=True,
        compute_ce_loss=True,
        n_batches=1,
        eval_batch_size_prompts=4,
    )

    for key in hf_metrics:
        assert hf_metrics[key] == pytest.approx(tlens_metrics[key], abs=1e-3)


def test_get_sparsity_and_variance_metrics_with_hf_model_gives_same_results_as_tlens_model(
    gpt2_res_jb_l4_sae: StandardSAE,
    example_dataset: Dataset,
):
    hf_model = load_model(
        model_class_name="AutoModelForCausalLM",
        model_name="gpt2",
        device="cpu",
    )
    tlens_model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")

    cfg = build_runner_cfg(hook_name="transformer.h.3")
    gpt2_res_jb_l4_sae.cfg.metadata.hook_name = "transformer.h.3"
    hf_store = ActivationsStore.from_config(
        hf_model, cfg, override_dataset=example_dataset
    )
    hf_metrics, hf_feat_metrics = get_sparsity_and_variance_metrics(
        sae=gpt2_res_jb_l4_sae,
        model=hf_model,
        activation_store=hf_store,
        activation_scaler=ActivationScaler(),
        n_batches=1,
        compute_l2_norms=True,
        compute_sparsity_metrics=True,
        compute_variance_metrics=True,
        compute_featurewise_density_statistics=True,
        eval_batch_size_prompts=4,
        model_kwargs={},
    )

    cfg = build_runner_cfg(hook_name="blocks.4.hook_resid_pre")
    gpt2_res_jb_l4_sae.cfg.metadata.hook_name = "blocks.4.hook_resid_pre"
    tlens_store = ActivationsStore.from_config(
        tlens_model, cfg, override_dataset=example_dataset
    )
    tlens_metrics, tlens_feat_metrics = get_sparsity_and_variance_metrics(
        sae=gpt2_res_jb_l4_sae,
        model=tlens_model,
        activation_store=tlens_store,
        activation_scaler=ActivationScaler(),
        n_batches=1,
        compute_l2_norms=True,
        compute_sparsity_metrics=True,
        compute_variance_metrics=True,
        compute_featurewise_density_statistics=True,
        eval_batch_size_prompts=4,
        model_kwargs={},
    )

    for key in hf_metrics:
        assert hf_metrics[key] == pytest.approx(tlens_metrics[key], rel=1e-4)
    for key in hf_feat_metrics:
        assert _replace_nan(hf_feat_metrics[key]) == pytest.approx(
            _replace_nan(tlens_feat_metrics[key]), rel=1e-4
        )


@patch("sae_lens.evals.get_pretrained_saes_directory")
def test_all_loadable_saes(mock_get_pretrained_saes_directory: MagicMock):
    mock_get_pretrained_saes_directory.return_value = {
        "release1": PretrainedSAELookup(
            release="release1",
            repo_id="repo1",
            model="model1",
            conversion_func=None,
            saes_map={"sae1": "path1", "sae2": "path2"},
            expected_var_explained={"sae1": 0.9, "sae2": 0.85},
            expected_l0={"sae1": 0.1, "sae2": 0.15},
            neuronpedia_id={},
            norm_scaling_factor={},
            config_overrides=None,
        ),
        "release2": PretrainedSAELookup(
            release="release2",
            repo_id="repo2",
            model="model2",
            conversion_func=None,
            saes_map={"sae3": "path3"},
            expected_var_explained={"sae3": 0.8},
            expected_l0={"sae3": 0.2},
            neuronpedia_id={},
            norm_scaling_factor={},
            config_overrides=None,
        ),
    }

    result = all_loadable_saes()

    expected = [
        ("release1", "sae1", 0.9, 0.1),
        ("release1", "sae2", 0.85, 0.15),
        ("release2", "sae3", 0.8, 0.2),
    ]
    assert result == expected


mock_all_saes = [
    ("release1", "sae1", 0.9, 0.1),
    ("release1", "sae2", 0.85, 0.15),
    ("release2", "sae3", 0.8, 0.2),
    ("release2", "block1", 0.95, 0.05),
]


@patch("sae_lens.evals.all_loadable_saes")
def test_get_saes_from_regex_no_match(mock_all_loadable_saes: MagicMock):
    mock_all_loadable_saes.return_value = mock_all_saes

    result = get_saes_from_regex("release1", "sae3")

    assert not result


@patch("sae_lens.evals.all_loadable_saes")
def test_get_saes_from_regex_single_match(mock_all_loadable_saes: MagicMock):
    mock_all_loadable_saes.return_value = mock_all_saes

    result = get_saes_from_regex("release1", "sae1")

    expected = [("release1", "sae1", 0.9, 0.1)]
    assert result == expected


@patch("sae_lens.evals.all_loadable_saes")
def test_get_saes_from_regex_multiple_matches(mock_all_loadable_saes: MagicMock):
    mock_all_loadable_saes.return_value = mock_all_saes

    result = get_saes_from_regex("release.*", "sae.*")

    expected = [
        ("release1", "sae1", 0.9, 0.1),
        ("release1", "sae2", 0.85, 0.15),
        ("release2", "sae3", 0.8, 0.2),
    ]
    assert result == expected


@pytest.mark.parametrize("scaling_factor", [None, 3.0])
def test_get_sparsity_and_variance_metrics_identity_sae_perfect_reconstruction(
    model: HookedTransformer,
    example_dataset: Dataset,
    scaling_factor: float | None,
):
    """Test that an identity SAE (d_in = d_sae, W_enc = W_dec = Identity, zero biases) gets perfect variance explained."""
    # Create a special configuration for an identity SAE
    d_in = 64  # Choose a small dimension for test efficiency
    identity_cfg = build_runner_cfg(
        d_in=d_in,
        d_sae=2 * d_in,  # 2 x d_in, to do both pos and neg identity matrix
        hook_name="blocks.1.hook_resid_pre",
    )

    # Create an SAE and manually set weights to identity matrices
    training_sae = StandardTrainingSAE.from_dict(
        identity_cfg.get_training_sae_cfg_dict()
    )
    identity_sae = StandardSAE.from_dict(training_sae.cfg.get_inference_sae_cfg_dict())
    with torch.no_grad():
        # Set encoder and decoder weights to identity matrices
        identity_sae.W_dec.data = torch.cat([torch.eye(d_in), -1 * torch.eye(d_in)])
        identity_sae.W_enc.data = identity_sae.W_dec.T.clone()
        # Set biases to zero
        identity_sae.b_enc.data = torch.zeros_like(identity_sae.b_enc)
        identity_sae.b_dec.data = torch.zeros_like(identity_sae.b_dec)

    # Create an activation store
    activation_store = ActivationsStore.from_config(
        model, identity_cfg, override_dataset=example_dataset
    )

    # Get metrics
    metrics, _ = get_sparsity_and_variance_metrics(
        sae=identity_sae,
        model=model,
        activation_store=activation_store,
        activation_scaler=ActivationScaler(scaling_factor),
        n_batches=3,
        compute_l2_norms=True,
        compute_sparsity_metrics=True,
        compute_variance_metrics=True,
        compute_featurewise_density_statistics=True,
        eval_batch_size_prompts=4,
        model_kwargs={},
    )

    # An identity SAE should perfectly reconstruct the input,
    # so variance explained should be 1.0 (or very close to it)
    assert metrics["explained_variance"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["explained_variance_legacy"] == pytest.approx(1.0, abs=1e-5)

    # Also check that L0 is exactly d_in (all features active)
    assert metrics["l0"] == pytest.approx(d_in, abs=1e-5)

    # MSE loss should be very close to 0
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-5)


def test_process_args():
    args = [
        "gpt2-small-res_scefr-ajt",
        "blocks.10.*",
        "--batch_size_prompts",
        "16",
        "--n_eval_sparsity_variance_batches",
        "200",
        "--n_eval_reconstruction_batches",
        "20",
        "--output_dir",
        "demo_eval_results",
        "--verbose",
    ]
    opts = process_args(args)
    assert opts.sae_regex_pattern == "gpt2-small-res_scefr-ajt"
    assert opts.sae_block_pattern == "blocks.10.*"
    assert opts.batch_size_prompts == 16
    assert opts.n_eval_sparsity_variance_batches == 200
    assert opts.n_eval_reconstruction_batches == 20
    assert opts.output_dir == "demo_eval_results"
    assert opts.verbose is True


def test_run_evals_cli(tmp_path: Path):
    args = [
        "gpt2-small-res-jb",
        "blocks.10.*",
        "--batch_size_prompts",
        "1",
        "--n_eval_sparsity_variance_batches",
        "2",
        "--output_dir",
        str(tmp_path),
        "--datasets",
        NEEL_NANDA_C4_10K_DATASET,
    ]
    run_evals_cli(args)

    assert (tmp_path / "all_eval_results.json").exists()
    assert (tmp_path / "all_eval_results.csv").exists()
    assert (
        tmp_path
        / "gpt2-small-res-jb-blocks.10.hook_resid_pre_128_NeelNanda_c4-10k.json"
    ).exists()

    with open(tmp_path / "all_eval_results.json") as f:
        eval_results = json.load(f)
    assert len(eval_results) == 1
    assert eval_results[0]["unique_id"] == "gpt2-small-res-jb-blocks.10.hook_resid_pre"
    assert eval_results[0]["eval_cfg"]["context_size"] == 128
    assert eval_results[0]["eval_cfg"]["dataset"] == NEEL_NANDA_C4_10K_DATASET
    for metric in [
        "ce_loss_score",
        "ce_loss_with_ablation",
        "ce_loss_with_sae",
        "ce_loss_without_sae",
    ]:
        assert (
            eval_results[0]["metrics"]["model_performance_preservation"][metric] > 0.1
        )


def _original_kl(original_logits: torch.Tensor, new_logits: torch.Tensor):
    original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
    log_original_probs = torch.log(original_probs)
    new_probs = torch.nn.functional.softmax(new_logits, dim=-1)
    log_new_probs = torch.log(new_probs)
    kl_div = original_probs * (log_original_probs - log_new_probs)
    return kl_div.sum(dim=-1)


def test_kl_matches_old_implementation():
    test_original_logits = torch.randn(2, 10, 30)
    test_new_logits = torch.randn(2, 10, 30)
    assert _original_kl(test_original_logits, test_new_logits) == pytest.approx(
        _kl(test_original_logits, test_new_logits)
    )


def test_get_sparsity_and_variance_metrics_works_with_batchtopk_saes(
    ts_model: HookedTransformer,
):
    example_dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 20
    )
    runner_cfg = build_batchtopk_runner_cfg(
        k=2,
        d_in=64,
        d_sae=10,
        rescale_acts_by_decoder_norm=True,
    )
    sae = BatchTopKTrainingSAE(runner_cfg.sae)
    random_params(sae)
    sae.b_enc.data = torch.randn(10) + 10.0

    store = ActivationsStore.from_config(
        ts_model, runner_cfg, override_dataset=example_dataset
    )

    # Get metrics
    sparsity, _ = get_sparsity_and_variance_metrics(
        sae=sae,
        model=ts_model,
        activation_store=store,
        activation_scaler=ActivationScaler(),
        n_batches=2,
        compute_l2_norms=False,
        compute_sparsity_metrics=True,
        compute_variance_metrics=False,
        compute_featurewise_density_statistics=False,
        eval_batch_size_prompts=2,
        model_kwargs={"device": "cpu"},
    )

    # Check that l0 is close to k
    assert sparsity["l0"] == pytest.approx(2.0)
