import pandas as pd
import pytest

from sae_lens.loading.pretrained_saes_directory import (
    PretrainedSAELookup,
    get_pretrained_saes_directory,
    get_releases_for_repo_id,
    get_repo_id_and_folder_name,
)


def test_get_pretrained_saes_directory():
    sae_directory = get_pretrained_saes_directory()
    assert isinstance(sae_directory, dict)
    expected_result = PretrainedSAELookup(
        release="gpt2-small-res-jb",
        repo_id="jbloom/GPT2-Small-SAEs-Reformatted",
        model="gpt2-small",
        conversion_func=None,
        saes_map={
            "blocks.0.hook_resid_pre": "blocks.0.hook_resid_pre",
            "blocks.1.hook_resid_pre": "blocks.1.hook_resid_pre",
            "blocks.2.hook_resid_pre": "blocks.2.hook_resid_pre",
            "blocks.3.hook_resid_pre": "blocks.3.hook_resid_pre",
            "blocks.4.hook_resid_pre": "blocks.4.hook_resid_pre",
            "blocks.5.hook_resid_pre": "blocks.5.hook_resid_pre",
            "blocks.6.hook_resid_pre": "blocks.6.hook_resid_pre",
            "blocks.7.hook_resid_pre": "blocks.7.hook_resid_pre",
            "blocks.8.hook_resid_pre": "blocks.8.hook_resid_pre",
            "blocks.9.hook_resid_pre": "blocks.9.hook_resid_pre",
            "blocks.10.hook_resid_pre": "blocks.10.hook_resid_pre",
            "blocks.11.hook_resid_pre": "blocks.11.hook_resid_pre",
            "blocks.11.hook_resid_post": "blocks.11.hook_resid_post",
        },
        expected_var_explained={
            "blocks.0.hook_resid_pre": 0.999,
            "blocks.1.hook_resid_pre": 0.999,
            "blocks.2.hook_resid_pre": 0.999,
            "blocks.3.hook_resid_pre": 0.999,
            "blocks.4.hook_resid_pre": 0.9,
            "blocks.5.hook_resid_pre": 0.9,
            "blocks.6.hook_resid_pre": 0.9,
            "blocks.7.hook_resid_pre": 0.9,
            "blocks.8.hook_resid_pre": 0.9,
            "blocks.9.hook_resid_pre": 0.77,
            "blocks.10.hook_resid_pre": 0.77,
            "blocks.11.hook_resid_pre": 0.77,
            "blocks.11.hook_resid_post": 0.77,
        },
        expected_l0={
            "blocks.0.hook_resid_pre": 10.0,
            "blocks.1.hook_resid_pre": 10.0,
            "blocks.2.hook_resid_pre": 18.0,
            "blocks.3.hook_resid_pre": 23.0,
            "blocks.4.hook_resid_pre": 31.0,
            "blocks.5.hook_resid_pre": 41.0,
            "blocks.6.hook_resid_pre": 51.0,
            "blocks.7.hook_resid_pre": 54.0,
            "blocks.8.hook_resid_pre": 60.0,
            "blocks.9.hook_resid_pre": 70.0,
            "blocks.10.hook_resid_pre": 52.0,
            "blocks.11.hook_resid_pre": 56.0,
            "blocks.11.hook_resid_post": 70.0,
        },
        config_overrides={
            "model_from_pretrained_kwargs": {
                "center_writing_weights": True,
            }
        },
        neuronpedia_id={
            "blocks.0.hook_resid_pre": "gpt2-small/0-res-jb",
            "blocks.1.hook_resid_pre": "gpt2-small/1-res-jb",
            "blocks.2.hook_resid_pre": "gpt2-small/2-res-jb",
            "blocks.3.hook_resid_pre": "gpt2-small/3-res-jb",
            "blocks.4.hook_resid_pre": "gpt2-small/4-res-jb",
            "blocks.5.hook_resid_pre": "gpt2-small/5-res-jb",
            "blocks.6.hook_resid_pre": "gpt2-small/6-res-jb",
            "blocks.7.hook_resid_pre": "gpt2-small/7-res-jb",
            "blocks.8.hook_resid_pre": "gpt2-small/8-res-jb",
            "blocks.9.hook_resid_pre": "gpt2-small/9-res-jb",
            "blocks.10.hook_resid_pre": "gpt2-small/10-res-jb",
            "blocks.11.hook_resid_pre": "gpt2-small/11-res-jb",
            "blocks.11.hook_resid_post": "gpt2-small/12-res-jb",
        },
    )

    assert sae_directory["gpt2-small-res-jb"] == expected_result


def test_get_pretrained_saes_directory_unique_np_ids():
    # ideally this code should be elsewhere but as a stop-gap we'll leave it here.
    df = pd.DataFrame.from_records(
        {k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}
    ).T
    df.drop(
        columns=[
            "repo_id",
            "saes_map",
            "expected_var_explained",
            "expected_l0",
            "config_overrides",
            "conversion_func",
        ],
        inplace=True,
    )
    df["neuronpedia_id_list"] = df["neuronpedia_id"].apply(lambda x: list(x.items()))
    df_exploded = df.explode("neuronpedia_id_list")
    df_exploded[["sae_lens_id", "neuronpedia_id"]] = pd.DataFrame(
        df_exploded["neuronpedia_id_list"].tolist(), index=df_exploded.index
    )
    df_exploded = df_exploded.drop(columns=["neuronpedia_id_list"])
    df_exploded = df_exploded.reset_index(drop=True)
    df_exploded["neuronpedia_set"] = df_exploded["neuronpedia_id"].apply(
        lambda x: "-".join(x.split("/")[-1].split("-")[1:]) if x is not None else None
    )

    duplicate_ids = df_exploded.groupby("neuronpedia_id").sae_lens_id.apply(
        lambda x: len(x)
    )
    assert (
        duplicate_ids.max() == 1
    ), f"Duplicate IDs found: {duplicate_ids[duplicate_ids > 1]}"


def test_get_repo_id_and_folder_name_release_found():
    repo_id, folder_name = get_repo_id_and_folder_name(
        "gpt2-small-res-jb", sae_id="blocks.0.hook_resid_pre"
    )
    assert repo_id == "jbloom/GPT2-Small-SAEs-Reformatted"
    assert folder_name == "blocks.0.hook_resid_pre"


def test_get_repo_id_and_folder_name_release_not_found():
    repo_id, folder_name = get_repo_id_and_folder_name("release1", "sae1")
    assert repo_id == "release1"
    assert folder_name == "sae1"


def test_get_repo_id_and_folder_name_raises_error_if_sae_id_not_found():
    with pytest.raises(ValueError):
        get_repo_id_and_folder_name("gpt2-small-res-jb", sae_id="sae1")


def test_get_releases_for_repo_id_found():
    releases = get_releases_for_repo_id("jbloom/GPT2-Small-SAEs-Reformatted")
    assert "gpt2-small-res-jb" in releases


def test_get_releases_for_repo_id_not_found():
    releases = get_releases_for_repo_id("nonexistent/repo")
    assert releases == []
