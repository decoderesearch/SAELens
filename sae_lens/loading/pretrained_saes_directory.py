from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any

import yaml


@dataclass
class PretrainedSAELookup:
    release: str
    repo_id: str
    model: str
    conversion_func: str | None
    saes_map: dict[str, str]  # id -> path
    expected_var_explained: dict[str, float]
    expected_l0: dict[str, float]
    neuronpedia_id: dict[str, str]
    norm_scaling_factor: dict[str, float]
    config_overrides: dict[str, str] | dict[str, dict[str, str | bool | int]] | None


@cache
def get_pretrained_saes_directory() -> dict[str, PretrainedSAELookup]:
    package = "sae_lens"
    # Access the file within the package using importlib.resources
    directory: dict[str, PretrainedSAELookup] = {}
    yaml_file = files(package).joinpath("pretrained_saes.yaml")
    with yaml_file.open("r") as file:
        # Load the YAML file content
        data = yaml.safe_load(file)
        for release, value in data.items():
            saes_map: dict[str, str] = {}
            var_explained_map: dict[str, float] = {}
            l0_map: dict[str, float] = {}
            neuronpedia_id_map: dict[str, str] = {}
            norm_scaling_factor_map: dict[str, float] = {}

            if "saes" not in value:
                raise KeyError(f"Missing 'saes' key in {release}")
            for hook_info in value["saes"]:
                sae_id = hook_info["id"]
                saes_map[sae_id] = hook_info["path"]
                var_explained_map[sae_id] = hook_info.get("variance_explained", 1.00)
                l0_map[sae_id] = hook_info.get("l0", 0.00)
                neuronpedia_id_map[sae_id] = hook_info.get("neuronpedia")
                if "norm_scaling_factor" in hook_info:
                    norm_scaling_factor_map[sae_id] = hook_info["norm_scaling_factor"]
            directory[release] = PretrainedSAELookup(
                release=release,
                repo_id=value["repo_id"],
                model=value["model"],
                conversion_func=value.get("conversion_func"),
                saes_map=saes_map,
                expected_var_explained=var_explained_map,
                expected_l0=l0_map,
                neuronpedia_id=neuronpedia_id_map,
                norm_scaling_factor=norm_scaling_factor_map,
                config_overrides=value.get("config_overrides"),
            )
    return directory


def get_norm_scaling_factor(release: str, sae_id: str) -> float | None:
    saes_directory = get_pretrained_saes_directory()
    sae_info = saes_directory.get(release)
    if sae_info is None:
        return None
    return sae_info.norm_scaling_factor.get(sae_id)


def get_repo_id_and_folder_name(release: str, sae_id: str) -> tuple[str, str]:
    saes_directory = get_pretrained_saes_directory()
    sae_info = saes_directory.get(release, None)

    if sae_info is None:
        return release, sae_id

    if sae_id not in sae_info.saes_map:
        raise ValueError(f"SAE ID '{sae_id}' not found in release '{release}'")

    repo_id = sae_info.repo_id
    folder_name = sae_info.saes_map[sae_id]
    return repo_id, folder_name


def get_config_overrides(release: str, sae_id: str) -> dict[str, Any]:
    saes_directory = get_pretrained_saes_directory()
    sae_info = saes_directory.get(release, None)
    config_overrides = {}
    if sae_info is not None:
        config_overrides = {**(sae_info.config_overrides or {})}
        if sae_info.neuronpedia_id is not None and sae_id in sae_info.neuronpedia_id:
            config_overrides["neuronpedia_id"] = sae_info.neuronpedia_id[sae_id]
    return config_overrides


def get_releases_for_repo_id(repo_id: str) -> list[str]:
    """
    Find all release names that use the given HuggingFace repo_id.

    Args:
        repo_id: The HuggingFace repository ID to search for.

    Returns:
        A list of release names that use this repo_id.
    """
    saes_directory = get_pretrained_saes_directory()
    return [
        release
        for release, lookup in saes_directory.items()
        if lookup.repo_id == repo_id
    ]
