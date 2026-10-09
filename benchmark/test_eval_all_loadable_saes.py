# import pandas as pd
# import plotly.express as px
# import numpy as np
import argparse
import json
from pathlib import Path

import pytest
import torch

from sae_lens import SAE, ActivationsStore
from sae_lens.analysis.neuronpedia_integration import open_neuronpedia_feature_dashboard
from sae_lens.evals import (
    all_loadable_saes,
    get_eval_everything_config,
    process_results,
    run_evals,
    run_evaluations,
)
from sae_lens.loading.pretrained_sae_loaders import load_sae_config_from_huggingface
from tests.helpers import load_model_cached

# from sae_lens.evals import run_evals


# from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

example_text = """
Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say
that they were perfectly normal, thank you very much. They were the last
people you'd expect to be involved in anything strange or mysterious,
because they just didn't hold with such nonsense.
"""


def test_load_sae_config_from_huggingface():
    repo_id = "jbloom/GPT2-Small-SAEs-Reformatted"
    cfg = load_sae_config_from_huggingface(
        release=repo_id,
        sae_id="blocks.0.hook_resid_pre",
    )
    assert cfg is not None


@pytest.mark.parametrize(
    "release, sae_name, expected_var_explained, expected_l0", all_loadable_saes()
)
def test_loading_pretrained_saes(
    release: str,
    sae_name: str,
    expected_var_explained: float,  # noqa: ARG001
    expected_l0: float,  # noqa: ARG001
):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    sae = SAE.from_pretrained(release, sae_name, device=device)
    assert isinstance(sae, SAE)


@pytest.mark.parametrize(
    "release, sae_name, expected_var_explained, expected_l0", all_loadable_saes()
)
def test_loading_pretrained_saes_open_neuronpedia(
    release: str,
    sae_name: str,
    expected_var_explained: float,  # noqa: ARG001
    expected_l0: float,  # noqa: ARG001
):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    sae = SAE.from_pretrained(release, sae_name, device=device)
    assert isinstance(sae, SAE)

    open_neuronpedia_feature_dashboard(sae, 0)


@pytest.mark.parametrize(
    "release, sae_name, expected_var_explained, expected_l0", all_loadable_saes()
)
def test_loading_pretrained_saes_do_forward_pass(
    release: str,
    sae_name: str,
    expected_var_explained: float,  # noqa: ARG001
    expected_l0: float,  # noqa: ARG001
):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    sae = SAE.from_pretrained(release, sae_name, device=device)
    assert isinstance(sae, SAE)

    # from transformer_lens import HookedTransformer
    # model = HookedTransformer.from_pretrained("gemma-2-9b")
    # sae_in = model.run_with_cache("test test test")[1][sae.cfg.metadata.hook_name]

    if "hook_z" in sae.cfg.metadata.hook_name:
        # check that reshaping works as intended
        from transformer_lens.loading_from_pretrained import get_pretrained_model_config

        model_cfg = get_pretrained_model_config(sae.cfg.metadata.model_name)
        sae_in = torch.randn(1, 4, model_cfg.n_heads, model_cfg.d_head).to(device)
        sae_out = sae(sae_in)
        assert sae_out.shape == sae_in.shape

    sae.turn_off_forward_pass_hook_z_reshaping()  # just in case
    sae_in = torch.randn(1, sae.cfg.d_in).to(device)
    sae_out = sae(sae_in)
    assert sae_out.shape == sae_in.shape

    assert True  # If we get here, we're good


@pytest.mark.parametrize(
    "release, sae_name, expected_var_explained, expected_l0", all_loadable_saes()
)
def test_eval_all_loadable_saes(
    release: str, sae_name: str, expected_var_explained: float, expected_l0: float
):
    """This test is currently only passing for a subset of SAEs because we need to
    have the normalization factors on hand to normalize the activations. We should
    really fold these into SAEs so this test is easy to do."""

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    sae = SAE.from_pretrained(release, sae_name, device=device)
    sae.fold_W_dec_norm()

    model = load_model_cached(sae.cfg.metadata.model_name)
    model.to(device)

    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        dataset=sae.cfg.metadata.dataset_path,
        streaming=True,
        # fairly conservative parameters here so can use same for larger
        # models without running out of memory.
        store_batch_size_prompts=8,
        train_batch_size_tokens=4096,
        n_sequences_in_buffer=4,
        device=device,
    )

    eval_config = get_eval_everything_config(
        batch_size_prompts=8,
        n_eval_reconstruction_batches=3,
        n_eval_sparsity_variance_batches=100,
    )

    metrics, _ = run_evals(
        sae=sae,
        activation_store=activation_store,
        model=model,
        eval_config=eval_config,
        exclude_special_tokens={
            model.tokenizer.pad_token_id,  # type: ignore
            model.tokenizer.eos_token_id,  # type: ignore
            model.tokenizer.bos_token_id,  # type: ignore
        },
    )

    assert pytest.approx(metrics["l0"], abs=5) == expected_l0
    assert (
        pytest.approx(metrics["explained_variance"], abs=0.1) == expected_var_explained
    )


@pytest.fixture
def mock_evals_simple_args(tmp_path: Path):
    class Args:
        sae_regex_pattern = "gpt2-small-res-jb"
        sae_block_pattern = "blocks.0.hook_resid_pre"
        num_eval_batches = 1
        n_eval_reconstruction_batches = 1
        n_eval_sparsity_variance_batches = 1

        eval_batch_size_prompts = 2
        datasets = ["Skylion007/openwebtext"]
        ctx_lens = [128]
        output_dir = str(tmp_path)
        verbose = False

    return Args()


def test_run_evaluations_process_results(mock_evals_simple_args: argparse.Namespace):
    """
    This test is more like an acceptance test for the evals code than a benchmark.
    """
    eval_results = run_evaluations(mock_evals_simple_args)
    output_files = process_results(eval_results, mock_evals_simple_args.output_dir)

    print("Evaluation complete. Output files:")
    print(f"Individual JSONs: {len(output_files['individual_jsons'])}")  # type: ignore
    print(f"Combined JSON: {output_files['combined_json']}")
    print(f"CSV: {output_files['csv']}")

    # open and validate the files
    combined_json_path = output_files["combined_json"]
    assert isinstance(combined_json_path, Path)
    assert combined_json_path.exists()
    with open(combined_json_path) as f:
        data = json.load(f)[0]
        assert "metrics" in data
        assert "feature_metrics" in data
