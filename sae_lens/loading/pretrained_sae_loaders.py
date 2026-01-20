import json
import re
import warnings
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import requests
import torch
import yaml
from huggingface_hub import hf_hub_download, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError, build_hf_headers
from packaging.version import Version
from safetensors import safe_open

from sae_lens import logger
from sae_lens.constants import (
    SAE_CFG_FILENAME,
    SAE_WEIGHTS_FILENAME,
    SPARSIFY_WEIGHTS_FILENAME,
    SPARSITY_FILENAME,
)
from sae_lens.loading.pretrained_saes_directory import (
    get_config_overrides,
    get_pretrained_saes_directory,
    get_repo_id_and_folder_name,
)
from sae_lens.registry import get_sae_class
from sae_lens.util import filter_valid_dataclass_fields, str_to_dtype

LLM_METADATA_KEYS = {
    "model_name",
    "hook_name",
    "model_class_name",
    "hook_head_index",
    "model_from_pretrained_kwargs",
    "prepend_bos",
    "exclude_special_tokens",
    "neuronpedia_id",
    "context_size",
    "seqpos_slice",
    "dataset_path",
    "sae_lens_version",
    "sae_lens_training_version",
    "hook_name_out",
    "hook_head_index_out",
    "hf_hook_name",
    "hf_hook_name_out",
}


def load_safetensors_weights(
    path: str | Path, device: str = "cpu", dtype: torch.dtype | str | None = None
) -> dict[str, torch.Tensor]:
    """Load safetensors weights and optionally convert to a different dtype"""
    loaded_weights = {}
    dtype = str_to_dtype(dtype) if isinstance(dtype, str) else dtype
    with safe_open(path, framework="pt", device=device) as f:
        for k in f.keys():  # noqa: SIM118
            weight = f.get_tensor(k)
            if dtype is not None:
                weight = weight.to(dtype=dtype)
            loaded_weights[k] = weight
    return loaded_weights


# loaders take in a release, sae_id, device, and whether to force download, and returns a tuple of config, state_dict, and log sparsity
class PretrainedSaeHuggingfaceLoader(Protocol):
    def __call__(
        self,
        repo_id: str,
        folder_name: str,
        device: str,
        force_download: bool,
        cfg_overrides: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]: ...


class PretrainedSaeConfigHuggingfaceLoader(Protocol):
    def __call__(
        self,
        repo_id: str,
        folder_name: str,
        device: str,
        force_download: bool,
        cfg_overrides: dict[str, Any] | None,
    ) -> dict[str, Any]: ...


class PretrainedSaeDiskLoader(Protocol):
    def __call__(
        self,
        path: str | Path,
        device: str,
        cfg_overrides: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]: ...


class PretrainedSaeConfigDiskLoader(Protocol):
    def __call__(
        self,
        path: str | Path,
        device: str | None,
        cfg_overrides: dict[str, Any] | None,
    ) -> dict[str, Any]: ...


def sae_lens_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """Loads SAEs from Hugging Face"""
    cfg_dict = get_sae_lens_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    weights_filename = f"{folder_name}/{SAE_WEIGHTS_FILENAME}"
    sae_path = hf_hub_download(
        repo_id=repo_id, filename=weights_filename, force_download=force_download
    )

    try:
        sparsity_filename = f"{folder_name}/{SPARSITY_FILENAME}"
        log_sparsity_path = hf_hub_download(
            repo_id=repo_id, filename=sparsity_filename, force_download=force_download
        )
    except EntryNotFoundError:
        log_sparsity_path = None  # no sparsity file

    cfg_dict, state_dict = read_sae_components_from_disk(
        cfg_dict=cfg_dict,
        weight_path=sae_path,
        device=device,
    )

    # get sparsity tensor if it exists
    if log_sparsity_path is not None:
        with safe_open(log_sparsity_path, framework="pt", device=device) as f:  # type: ignore
            log_sparsity = f.get_tensor("sparsity")
    else:
        log_sparsity = None

    return cfg_dict, state_dict, log_sparsity


def sae_lens_disk_loader(
    path: str | Path,
    device: str = "cpu",
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Loads SAEs from disk"""

    weights_path = Path(path) / SAE_WEIGHTS_FILENAME
    cfg_dict = get_sae_lens_config_from_disk(path, device, cfg_overrides)
    cfg_dict, state_dict = read_sae_components_from_disk(
        cfg_dict=cfg_dict,
        weight_path=weights_path,
        device=device,
    )
    return cfg_dict, state_dict


def get_sae_lens_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Retrieve the configuration for a Sparse Autoencoder (SAE) from a Hugging Face repository.

    Args:
        repo_id (str): The repository ID on Hugging Face.
        folder_name (str): The folder name within the repository containing the config file.
        force_download (bool, optional): Whether to force download the config file. Defaults to False.
        cfg_overrides (dict[str, Any] | None, optional): Overrides for the config. Defaults to None.

    Returns:
        dict[str, Any]: The configuration dictionary for the SAE.
    """
    cfg_filename = f"{folder_name}/{SAE_CFG_FILENAME}"
    cfg_path = hf_hub_download(
        repo_id=repo_id, filename=cfg_filename, force_download=force_download
    )
    sae_path = Path(cfg_path).parent
    return get_sae_lens_config_from_disk(sae_path, device, cfg_overrides)


def get_sae_lens_config_from_disk(
    path: str | Path,
    device: str | None = None,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg_filename = Path(path) / SAE_CFG_FILENAME

    with open(cfg_filename) as f:
        cfg_dict: dict[str, Any] = json.load(f)

    if device is not None:
        cfg_dict["device"] = device

    if cfg_overrides is not None:
        cfg_dict.update(cfg_overrides)

    return cfg_dict


def handle_config_defaulting(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    sae_lens_version = cfg_dict.get("sae_lens_version")
    if not sae_lens_version and "metadata" in cfg_dict:
        sae_lens_version = cfg_dict["metadata"].get("sae_lens_version")

    if not sae_lens_version or Version(sae_lens_version) < Version("6.0.0-rc.0"):
        cfg_dict = handle_pre_6_0_config(cfg_dict)
    return cfg_dict


def handle_pre_6_0_config(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Format a config dictionary for a Sparse Autoencoder (SAE) to be compatible with the new 6.0 format.
    """

    rename_keys_map = {
        "hook_point": "hook_name",
        "hook_point_head_index": "hook_head_index",
        "activation_fn_str": "activation_fn",
    }
    new_cfg = {rename_keys_map.get(k, k): v for k, v in cfg_dict.items()}

    # Set default values for backwards compatibility
    new_cfg.setdefault("prepend_bos", True)
    new_cfg.setdefault("dataset_trust_remote_code", True)
    new_cfg.setdefault("apply_b_dec_to_input", True)
    new_cfg.setdefault("finetuning_scaling_factor", False)
    new_cfg.setdefault("sae_lens_training_version", None)
    new_cfg.setdefault("activation_fn", new_cfg.get("activation_fn", "relu"))
    new_cfg.setdefault("architecture", "standard")
    new_cfg.setdefault("neuronpedia_id", None)
    new_cfg.setdefault(
        "reshape_activations",
        "hook_z" if "hook_z" in new_cfg.get("hook_name", "") else "none",
    )
    if (
        new_cfg.get("activation_fn") == "topk"
        and new_cfg.get("activation_fn_kwargs", {}).get("k") is not None
    ):
        new_cfg["architecture"] = "topk"
        new_cfg["k"] = new_cfg["activation_fn_kwargs"]["k"]

    if "normalize_activations" in new_cfg and isinstance(
        new_cfg["normalize_activations"], bool
    ):
        # backwards compatibility
        new_cfg["normalize_activations"] = (
            "none"
            if not new_cfg["normalize_activations"]
            else "expected_average_only_in"
        )

    if new_cfg.get("normalize_activations") is None:
        new_cfg["normalize_activations"] = "none"

    new_cfg.setdefault("device", "cpu")

    architecture = new_cfg.get("architecture", "standard")

    config_class = get_sae_class(architecture)[1]

    sae_cfg_dict = filter_valid_dataclass_fields(new_cfg, config_class)
    if architecture == "topk" and "activation_fn_kwargs" in new_cfg:
        sae_cfg_dict["k"] = new_cfg["activation_fn_kwargs"]["k"]

    sae_cfg_dict["metadata"] = {
        k: v for k, v in new_cfg.items() if k in LLM_METADATA_KEYS
    }
    sae_cfg_dict["architecture"] = architecture
    return sae_cfg_dict


def get_connor_rob_hook_z_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = folder_name.split(".pt")[0] + "_cfg.json"
    config_path = hf_hub_download(repo_id, config_path, force_download=force_download)

    with open(config_path) as config_file:
        old_cfg_dict = json.load(config_file)

    return {
        "architecture": "standard",
        "d_in": old_cfg_dict["act_size"],
        "d_sae": old_cfg_dict["dict_size"],
        "dtype": "float32",
        "device": device if device is not None else "cpu",
        "model_name": "gpt2-small",
        "hook_name": old_cfg_dict["act_name"],
        "hook_head_index": None,
        "activation_fn": "relu",
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "Skylion007/openwebtext",
        "context_size": 128,
        "normalize_activations": "none",
        "dataset_trust_remote_code": True,
        "reshape_activations": "hook_z",
        **(cfg_overrides or {}),
    }


def connor_rob_hook_z_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], None]:
    cfg_dict = get_connor_rob_hook_z_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    file_path = hf_hub_download(
        repo_id=repo_id, filename=folder_name, force_download=force_download
    )
    weights = torch.load(file_path, map_location=device)

    return cfg_dict, weights, None


def read_sae_components_from_disk(
    cfg_dict: dict[str, Any],
    weight_path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """
    Given a loaded dictionary and a path to a weight file, load the weights and return the state_dict.
    """
    if dtype is None:
        dtype = str_to_dtype(cfg_dict["dtype"])

    state_dict = {}
    with safe_open(weight_path, framework="pt", device=device) as f:  # type: ignore
        for k in f.keys():  # noqa: SIM118
            state_dict[k] = f.get_tensor(k).to(dtype=dtype)

    # if bool and True, then it's the April update method of normalizing activations and hasn't been folded in.
    if "scaling_factor" in state_dict:
        # we were adding it anyway for a period of time but are no longer doing so.
        # so we should delete it if
        if torch.allclose(
            state_dict["scaling_factor"],
            torch.ones_like(state_dict["scaling_factor"]),
        ):
            del state_dict["scaling_factor"]
            cfg_dict["finetuning_scaling_factor"] = False
        else:
            if not cfg_dict["finetuning_scaling_factor"]:
                raise ValueError(
                    "Scaling factor is present but finetuning_scaling_factor is False."
                )
            state_dict["finetuning_scaling_factor"] = state_dict["scaling_factor"]
            del state_dict["scaling_factor"]
    else:
        # it's there and it's not all 1's, we should use it.
        cfg_dict["finetuning_scaling_factor"] = False

    return cfg_dict, state_dict


def get_gemma_2_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str,
    force_download: bool = False,  # noqa: ARG001
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Detect width from folder_name
    width_map = {
        "width_4k": 4096,
        "width_16k": 16384,
        "width_32k": 32768,
        "width_65k": 65536,
        "width_131k": 131072,
        "width_262k": 262144,
        "width_524k": 524288,
        "width_1m": 1048576,
    }
    d_sae = next(
        (width for key, width in width_map.items() if key in folder_name), None
    )

    # Detect layer from folder_name
    match = re.search(r"layer_(\d+)", folder_name)
    layer = int(match.group(1)) if match else None
    if layer is None:
        if "embedding" in folder_name:
            layer = 0
        else:
            raise ValueError("Layer not found in folder_name and no override provided.")

    # Model specific parameters
    model_params = {
        "2b-it": {"name": "gemma-2-2b-it", "d_in": 2304},
        "9b-it": {"name": "gemma-2-9b-it", "d_in": 3584},
        "27b-it": {"name": "gemma-2-27b-it", "d_in": 4608},
        "2b": {"name": "gemma-2-2b", "d_in": 2304},
        "9b": {"name": "gemma-2-9b", "d_in": 3584},
        "27b": {"name": "gemma-2-27b", "d_in": 4608},
    }
    model_info = next(
        (info for key, info in model_params.items() if key in repo_id), None
    )
    if not model_info:
        raise ValueError("Model name not found in repo_id.")

    model_name, d_in = model_info["name"], model_info["d_in"]

    # Hook specific parameters
    if "res" in repo_id and "embedding" not in folder_name:
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif "res" in repo_id and "embedding" in folder_name:
        hook_name = "hook_embed"
    elif "mlp" in repo_id:
        hook_name = f"blocks.{layer}.hook_mlp_out"
    elif "att" in repo_id:
        hook_name = f"blocks.{layer}.attn.hook_z"
        d_in = {"2b": 2048, "9b": 4096, "27b": 4608}.get(
            next(key for key in model_params if key in repo_id), d_in
        )
    else:
        raise ValueError("Hook name not found in folder_name.")

    cfg = {
        "architecture": "jumprelu",
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "model_name": model_name,
        "hook_name": hook_name,
        "hook_head_index": None,
        "activation_fn": "relu",
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "dataset_trust_remote_code": True,
        "apply_b_dec_to_input": False,
        "normalize_activations": None,
    }
    if device is not None:
        cfg["device"] = device

    if cfg_overrides is not None:
        cfg.update(cfg_overrides)

    return cfg


def gemma_2_sae_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Custom loader for Gemma 2 SAEs.
    """
    cfg_dict = get_gemma_2_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Download the SAE weights
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename="params.npz",
        subfolder=folder_name,
        force_download=force_download,
    )

    # Load and convert the weights
    state_dict = {}
    with np.load(sae_path) as data:
        for key in data:
            state_dict_key = "W_" + key[2:] if key.startswith("w_") else key
            state_dict[state_dict_key] = (
                torch.tensor(data[key]).to(dtype=torch.float32).to(device)
            )

    # Handle scaling factor
    if "scaling_factor" in state_dict:
        if torch.allclose(
            state_dict["scaling_factor"], torch.ones_like(state_dict["scaling_factor"])
        ):
            del state_dict["scaling_factor"]
            cfg_dict["finetuning_scaling_factor"] = False
        else:
            if not cfg_dict["finetuning_scaling_factor"]:
                raise ValueError(
                    "Scaling factor is present but finetuning_scaling_factor is False."
                )
            state_dict["finetuning_scaling_factor"] = state_dict.pop("scaling_factor")
    else:
        cfg_dict["finetuning_scaling_factor"] = False

    # No sparsity tensor for Gemma 2 SAEs
    log_sparsity = None

    # if it is an embedding SAE, then we need to adjust for the scale of d_model because of how they trained it
    if "embedding" in folder_name:
        logger.debug("Adjusting for d_model in embedding SAE")
        state_dict["W_enc"].data = state_dict["W_enc"].data / np.sqrt(cfg_dict["d_in"])
        state_dict["W_dec"].data = state_dict["W_dec"].data * np.sqrt(cfg_dict["d_in"])

    return cfg_dict, state_dict, log_sparsity


def _infer_gemma_3_raw_cfg_dict(repo_id: str, folder_name: str) -> dict[str, Any]:
    """
    Infer the raw config dict for Gemma 3 SAEs from the repo_id and folder_name.
    This is used when config.json doesn't exist in the repo.
    """
    # Extract layer number from folder name
    layer_match = re.search(r"layer_(\d+)", folder_name)
    if layer_match is None:
        raise ValueError(
            f"Could not extract layer number from folder_name: {folder_name}"
        )
    layer = int(layer_match.group(1))

    # Convert repo_id to model_name: google/gemma-scope-2-{size}-{suffix} -> google/gemma-3-{size}-{suffix}
    model_name = repo_id.replace("gemma-scope-2", "gemma-3")

    # Determine hook type and HF hook points based on folder_name
    if "transcoder" in folder_name or "clt" in folder_name:
        hf_hook_point_in = f"model.layers.{layer}.pre_feedforward_layernorm.output"
        hf_hook_point_out = f"model.layers.{layer}.post_feedforward_layernorm.output"
    elif "resid_post" in folder_name:
        hf_hook_point_in = f"model.layers.{layer}.output"
        hf_hook_point_out = None
    elif "attn_out" in folder_name:
        hf_hook_point_in = f"model.layers.{layer}.self_attn.o_proj.input"
        hf_hook_point_out = None
    elif "mlp_out" in folder_name:
        hf_hook_point_in = f"model.layers.{layer}.post_feedforward_layernorm.output"
        hf_hook_point_out = None
    else:
        raise ValueError(f"Could not infer hook type from folder_name: {folder_name}")

    cfg: dict[str, Any] = {
        "architecture": "jump_relu",
        "model_name": model_name,
        "hf_hook_point_in": hf_hook_point_in,
    }
    if "transcoder" in folder_name or "clt" in folder_name:
        cfg["affine_connection"] = "affine" in folder_name
    if hf_hook_point_out is not None:
        cfg["hf_hook_point_out"] = hf_hook_point_out

    return cfg


def get_gemma_3_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Try to load config.json from the repo, fall back to inferring if it doesn't exist
    try:
        config_path = hf_hub_download(
            repo_id, f"{folder_name}/config.json", force_download=force_download
        )
        with open(config_path) as config_file:
            raw_cfg_dict = json.load(config_file)
    except EntryNotFoundError:
        raw_cfg_dict = _infer_gemma_3_raw_cfg_dict(repo_id, folder_name)

    if raw_cfg_dict.get("architecture") != "jump_relu":
        raise ValueError(
            f"Unexpected architecture in Gemma 3 config: {raw_cfg_dict.get('architecture')}"
        )

    layer_match = re.search(r"layer_(\d+)", folder_name)
    if layer_match is None:
        raise ValueError(
            f"Could not extract layer number from folder_name: {folder_name}"
        )
    layer = int(layer_match.group(1))
    hook_name_out = None
    d_out = None
    if "resid_post" in folder_name:
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif "attn_out" in folder_name:
        hook_name = f"blocks.{layer}.attn.hook_z"
    elif "mlp_out" in folder_name:
        hook_name = f"blocks.{layer}.hook_mlp_out"
    elif "transcoder" in folder_name or "clt" in folder_name:
        hook_name = f"blocks.{layer}.hook_mlp_in"
        hook_name_out = f"blocks.{layer}.hook_mlp_out"
    else:
        raise ValueError("Hook name not found in folder_name.")

    # hackily deal with clt file names
    params_file_part = "/params.safetensors"
    if "clt" in folder_name:
        params_file_part = ".safetensors"

    shapes_dict = get_safetensors_tensor_shapes(
        repo_id, f"{folder_name}{params_file_part}"
    )
    d_in, d_sae = shapes_dict["w_enc"]
    # TODO: update this for real model info
    model_name = raw_cfg_dict["model_name"]
    if "google" not in model_name:
        model_name = "google/" + model_name
    model_name = model_name.replace("-v3", "-3")
    if "270m" in model_name:
        # for some reason the 270m model on huggingface doesn't have the -pt suffix
        model_name = model_name.replace("-pt", "")

    architecture = "jumprelu"
    if "transcoder" in folder_name or "clt" in folder_name:
        architecture = (
            "jumprelu_skip_transcoder"
            if raw_cfg_dict.get("affine_connection", False)
            else "jumprelu_transcoder"
        )
        d_out = shapes_dict["w_dec"][-1]

    cfg = {
        "architecture": architecture,
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "model_name": model_name,
        "hook_name": hook_name,
        "hook_head_index": None,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "apply_b_dec_to_input": False,
        "normalize_activations": None,
        "hf_hook_name": raw_cfg_dict.get("hf_hook_point_in"),
    }
    if hook_name_out is not None:
        cfg["hook_name_out"] = hook_name_out
        cfg["hf_hook_name_out"] = raw_cfg_dict.get("hf_hook_point_out")
    if d_out is not None:
        cfg["d_out"] = d_out
    if device is not None:
        cfg["device"] = device

    if cfg_overrides is not None:
        cfg.update(cfg_overrides)

    return cfg


def gemma_3_sae_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Custom loader for Gemma 3 SAEs.
    """
    cfg_dict = get_gemma_3_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    params_file = "params.safetensors"
    if "clt" in folder_name:
        params_file = folder_name.split("/")[-1] + ".safetensors"
        folder_name = "/".join(folder_name.split("/")[:-1])

    # Download the SAE weights
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename=params_file,
        subfolder=folder_name,
        force_download=force_download,
    )

    raw_state_dict = load_safetensors_weights(
        sae_path, device=device, dtype=cfg_dict.get("dtype")
    )

    with torch.no_grad():
        w_dec = raw_state_dict["w_dec"]
        if "clt" in folder_name:
            w_dec = w_dec.sum(dim=1).contiguous()

    state_dict = {
        "W_enc": raw_state_dict["w_enc"],
        "W_dec": w_dec,
        "b_enc": raw_state_dict["b_enc"],
        "b_dec": raw_state_dict["b_dec"],
        "threshold": raw_state_dict["threshold"],
    }

    if "affine_skip_connection" in raw_state_dict:
        state_dict["W_skip"] = raw_state_dict["affine_skip_connection"]

    return cfg_dict, state_dict, None


def get_goodfire_config_from_hf(
    repo_id: str,
    folder_name: str,  # noqa: ARG001
    device: str,
    force_download: bool = False,  # noqa: ARG001
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg_dict = None
    if repo_id == "Goodfire/Llama-3.3-70B-Instruct-SAE-l50":
        if folder_name != "Llama-3.3-70B-Instruct-SAE-l50.pt":
            raise ValueError(f"Unsupported Goodfire SAE: {repo_id}/{folder_name}")
        cfg_dict = {
            "architecture": "standard",
            "d_in": 8192,
            "d_sae": 65536,
            "model_name": "meta-llama/Llama-3.3-70B-Instruct",
            "hook_name": "blocks.50.hook_resid_post",
            "hook_head_index": None,
            "dataset_path": "lmsys/lmsys-chat-1m",
            "apply_b_dec_to_input": False,
        }
    elif repo_id == "Goodfire/Llama-3.1-8B-Instruct-SAE-l19":
        if folder_name != "Llama-3.1-8B-Instruct-SAE-l19.pth":
            raise ValueError(f"Unsupported Goodfire SAE: {repo_id}/{folder_name}")
        cfg_dict = {
            "architecture": "standard",
            "d_in": 4096,
            "d_sae": 65536,
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "hook_name": "blocks.19.hook_resid_post",
            "hook_head_index": None,
            "dataset_path": "lmsys/lmsys-chat-1m",
            "apply_b_dec_to_input": False,
        }
    if cfg_dict is None:
        raise ValueError(f"Unsupported Goodfire SAE: {repo_id}/{folder_name}")
    if device is not None:
        cfg_dict["device"] = device
    if cfg_overrides is not None:
        cfg_dict.update(cfg_overrides)
    return cfg_dict


def get_goodfire_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    cfg_dict = get_goodfire_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Download the SAE weights
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename=folder_name,
        force_download=force_download,
    )
    raw_state_dict = torch.load(sae_path, map_location=device)

    target_dtype = str_to_dtype(cfg_dict.get("dtype", "float32"))

    state_dict = {
        "W_enc": raw_state_dict["encoder_linear.weight"].T.to(dtype=target_dtype),
        "W_dec": raw_state_dict["decoder_linear.weight"].T.to(dtype=target_dtype),
        "b_enc": raw_state_dict["encoder_linear.bias"].to(dtype=target_dtype),
        "b_dec": raw_state_dict["decoder_linear.bias"].to(dtype=target_dtype),
    }

    return cfg_dict, state_dict, None


def get_llama_scope_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Llama Scope SAEs
    # repo_id: fnlp/Llama3_1-8B-Base-LX{sublayer}-{exp_factor}x
    # folder_name: Llama3_1-8B-Base-L{layer}{sublayer}-{exp_factor}x
    config_path = folder_name + "/hyperparams.json"
    config_path = hf_hub_download(repo_id, config_path, force_download=force_download)

    with open(config_path) as f:
        old_cfg_dict = json.load(f)

    # Model specific parameters
    model_name, d_in = "meta-llama/Llama-3.1-8B", old_cfg_dict["d_model"]

    # Get norm scaling factor to rescale jumprelu threshold.
    # We need this because sae.fold_activation_norm_scaling_factor folds scaling norm into W_enc.
    # This requires jumprelu threshold to be scaled in the same way
    norm_scaling_factor = (
        d_in**0.5 / old_cfg_dict["dataset_average_activation_norm"]["in"]
    )

    cfg_dict = {
        "architecture": "jumprelu",
        "jump_relu_threshold": old_cfg_dict["jump_relu_threshold"]
        * norm_scaling_factor,
        # We use a scalar jump_relu_threshold for all features
        # This is different from Gemma Scope JumpReLU SAEs.
        # Scaled with norm_scaling_factor to match sae.fold_activation_norm_scaling_factor
        "d_in": d_in,
        "d_sae": old_cfg_dict["d_sae"],
        "dtype": "bfloat16",
        "model_name": model_name,
        "hook_name": old_cfg_dict["hook_point_in"],
        "hook_head_index": None,
        "activation_fn": "relu",
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "cerebras/SlimPajama-627B",
        "context_size": 1024,
        "dataset_trust_remote_code": True,
        "apply_b_dec_to_input": False,
        "normalize_activations": "expected_average_only_in",
    }

    if device is not None:
        cfg_dict["device"] = device

    if cfg_overrides is not None:
        cfg_dict.update(cfg_overrides)

    return cfg_dict


def llama_scope_sae_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Custom loader for Llama Scope SAEs.

    Args:
        release: Release identifier
        sae_id: SAE identifier
        device: Device to load tensors to
        force_download: Whether to force download even if files exist
        cfg_overrides: Configuration overrides (optional)
        d_sae_override: Override for SAE dimension (optional)
        layer_override: Override for layer number (optional)

    Returns:
        tuple of (config dict, state dict, log sparsity tensor)
    """
    cfg_dict = get_llama_scope_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Download the SAE weights
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename="final.safetensors",
        subfolder=folder_name + "/checkpoints",
        force_download=force_download,
    )

    state_dict_loaded = load_safetensors_weights(
        sae_path, device=device, dtype=cfg_dict.get("dtype")
    )

    # Convert and organize the weights
    state_dict = {
        "W_enc": state_dict_loaded["encoder.weight"].T,
        "W_dec": state_dict_loaded["decoder.weight"].T,
        "b_enc": state_dict_loaded["encoder.bias"],
        "b_dec": state_dict_loaded["decoder.bias"],
        "threshold": torch.ones(
            cfg_dict["d_sae"],
            dtype=str_to_dtype(cfg_dict["dtype"]),
            device=cfg_dict["device"],
        )
        * cfg_dict["jump_relu_threshold"],
    }

    # No sparsity tensor for Llama Scope SAEs
    log_sparsity = None

    return cfg_dict, state_dict, log_sparsity


def get_dictionary_learning_config_1_from_hf(
    repo_id: str,
    folder_name: str,
    device: str,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Suitable for SAEs from https://huggingface.co/canrager/lm_sae.
    """
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{folder_name}/config.json",
        force_download=force_download,
    )
    with open(config_path) as f:
        config = json.load(f)

    trainer = config["trainer"]
    buffer = config["buffer"]

    hook_point_name = f"blocks.{trainer['layer']}.hook_resid_post"

    activation_fn = "topk" if trainer["dict_class"] == "AutoEncoderTopK" else "relu"
    activation_fn_kwargs = {"k": trainer["k"]} if activation_fn == "topk" else {}

    architecture = "standard"
    if trainer["dict_class"] == "GatedAutoEncoder":
        architecture = "gated"
    elif trainer["dict_class"] in ["MatryoshkaBatchTopKSAE", "BatchTopKSAE"]:
        architecture = "jumprelu"

    return {
        "architecture": architecture,
        "d_in": trainer["activation_dim"],
        "d_sae": trainer["dict_size"],
        "dtype": "float32",
        "device": device,
        "model_name": trainer["lm_name"].split("/")[-1],
        "hook_name": hook_point_name,
        "hook_head_index": None,
        "activation_fn": activation_fn,
        "activation_fn_kwargs": activation_fn_kwargs,
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": buffer["ctx_len"],
        "normalize_activations": "none",
        "neuronpedia_id": None,
        "dataset_trust_remote_code": True,
        **(cfg_overrides or {}),
    }


def get_deepseek_r1_config_from_hf(
    repo_id: str,  # noqa: ARG001
    folder_name: str,
    device: str,
    force_download: bool = False,  # noqa: ARG001
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get config for DeepSeek R1 SAEs."""

    match = re.search(r"l(\d+)", folder_name)
    if match is None:
        raise ValueError(f"Could not find layer number in filename: {folder_name}")
    layer = int(match.group(1))

    return {
        "architecture": "standard",
        "d_in": 4096,  # LLaMA 8B hidden size
        "d_sae": 4096 * 16,  # Expansion factor 16
        "dtype": "bfloat16",
        "context_size": 1024,
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "hook_name": f"blocks.{layer}.hook_resid_post",
        "hook_head_index": None,
        "prepend_bos": True,
        "dataset_path": "lmsys/lmsys-chat-1m",
        "dataset_trust_remote_code": True,
        "sae_lens_training_version": None,
        "activation_fn": "relu",
        "normalize_activations": "none",
        "device": device,
        "apply_b_dec_to_input": False,
        "finetuning_scaling_factor": False,
        **(cfg_overrides or {}),
    }


def deepseek_r1_sae_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """Load a DeepSeek R1 SAE."""
    # Download weights
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename=folder_name,
        force_download=force_download,
    )

    # Load state dict
    state_dict_loaded = torch.load(sae_path, map_location=device)

    # Create config
    cfg_dict = get_deepseek_r1_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Convert weights
    state_dict = {
        "W_enc": state_dict_loaded["encoder.weight"].T,
        "W_dec": state_dict_loaded["decoder.weight"].T,
        "b_enc": state_dict_loaded["encoder.bias"],
        "b_dec": state_dict_loaded["decoder.bias"],
    }

    return cfg_dict, state_dict, None


def get_conversion_loader_name(release: str) -> str:
    saes_directory = get_pretrained_saes_directory()
    sae_info = saes_directory.get(release, None)
    conversion_loader_name = "sae_lens"
    if sae_info is not None and sae_info.conversion_func is not None:
        conversion_loader_name = sae_info.conversion_func
    if conversion_loader_name not in NAMED_PRETRAINED_SAE_LOADERS:
        raise ValueError(
            f"Conversion func '{conversion_loader_name}' not found in NAMED_PRETRAINED_SAE_LOADERS."
        )
    return conversion_loader_name


def load_sae_config_from_huggingface(
    release: str,
    sae_id: str,
    device: str = "cpu",
    force_download: bool = False,
) -> dict[str, Any]:
    cfg_overrides = get_config_overrides(release, sae_id)
    conversion_loader_name = get_conversion_loader_name(release)
    config_getter = NAMED_PRETRAINED_SAE_CONFIG_GETTERS[conversion_loader_name]
    repo_id, folder_name = get_repo_id_and_folder_name(release, sae_id=sae_id)
    cfg = {
        **config_getter(repo_id, folder_name, device, force_download, cfg_overrides),
    }
    return handle_config_defaulting(cfg)


def dictionary_learning_sae_huggingface_loader_1(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Suitable for SAEs from https://huggingface.co/canrager/lm_sae.
    """
    cfg_dict = get_dictionary_learning_config_1_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    encoder_path = hf_hub_download(
        repo_id=repo_id, filename=f"{folder_name}/ae.pt", force_download=force_download
    )
    encoder = torch.load(encoder_path, map_location="cpu")

    W_enc = encoder["W_enc"] if "W_enc" in encoder else encoder["encoder.weight"].T
    W_dec = encoder["W_dec"] if "W_dec" in encoder else encoder["decoder.weight"].T

    state_dict = {
        "W_enc": W_enc,
        "W_dec": W_dec,
        "b_dec": encoder.get(
            "b_dec", encoder.get("bias", encoder.get("decoder_bias", None))
        ),
    }

    if "encoder.bias" in encoder:
        state_dict["b_enc"] = encoder["encoder.bias"]
    if "b_enc" in encoder:
        state_dict["b_enc"] = encoder["b_enc"]

    if "mag_bias" in encoder:
        state_dict["b_mag"] = encoder["mag_bias"]
    if "gate_bias" in encoder:
        state_dict["b_gate"] = encoder["gate_bias"]
    if "r_mag" in encoder:
        state_dict["r_mag"] = encoder["r_mag"]

    if "threshold" in encoder:
        threshold = encoder["threshold"]
        if threshold.ndim == 0:
            threshold = torch.full((W_enc.size(1),), threshold)
        state_dict["threshold"] = threshold

    return cfg_dict, state_dict, None


def get_llama_scope_r1_distill_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Future Llama Scope series SAE by OpenMoss group use this config.
    # repo_id: [
    #   fnlp/Llama-Scope-R1-Distill
    # ]
    # folder_name: [
    #   800M-Slimpajama-0-OpenR1-Math-220k/L{layer}R,
    #   400M-Slimpajama-400M-OpenR1-Math-220k/L{layer}R,
    #   0-Slimpajama-800M-OpenR1-Math-220k/L{layer}R,
    # ]
    config_path = folder_name + "/config.json"
    config_path = hf_hub_download(repo_id, config_path, force_download=force_download)

    with open(config_path) as f:
        huggingface_cfg_dict = json.load(f)

    # Model specific parameters
    model_name, d_in = "meta-llama/Llama-3.1-8B", huggingface_cfg_dict["d_model"]

    return {
        "architecture": "jumprelu",
        "d_in": d_in,
        "d_sae": d_in * huggingface_cfg_dict["expansion_factor"],
        "dtype": "float32",
        "device": device,
        "model_name": model_name,
        "hook_name": huggingface_cfg_dict["hook_point_in"],
        "hook_head_index": None,
        "activation_fn": "relu",
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "cerebras/SlimPajama-627B",
        "context_size": 1024,
        "dataset_trust_remote_code": True,
        "apply_b_dec_to_input": False,
        "normalize_activations": "expected_average_only_in",
        **(cfg_overrides or {}),
    }


def llama_scope_r1_distill_sae_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Custom loader for Llama Scope SAEs.

    Args:
        release: Release identifier
        sae_id: SAE identifier
        device: Device to load tensors to
        force_download: Whether to force download even if files exist
        cfg_overrides: Configuration overrides (optional)
        d_sae_override: Override for SAE dimension (optional)
        layer_override: Override for layer number (optional)

    Returns:
        tuple of (config dict, state dict, log sparsity tensor)
    """
    cfg_dict = get_llama_scope_r1_distill_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Download the SAE weights
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename=SAE_WEIGHTS_FILENAME,
        subfolder=folder_name,
        force_download=force_download,
    )

    state_dict_loaded = load_safetensors_weights(
        sae_path, device=device, dtype=cfg_dict.get("dtype")
    )

    # Convert and organize the weights
    state_dict = {
        "W_enc": state_dict_loaded["encoder.weight"].T,
        "W_dec": state_dict_loaded["decoder.weight"].T,
        "b_enc": state_dict_loaded["encoder.bias"],
        "b_dec": state_dict_loaded["decoder.bias"],
        "threshold": state_dict_loaded["log_jumprelu_threshold"].exp(),
    }

    # No sparsity tensor for Llama Scope SAEs
    log_sparsity = None

    return cfg_dict, state_dict, log_sparsity


def get_sparsify_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg_filename = f"{folder_name}/{SAE_CFG_FILENAME}"
    cfg_path = hf_hub_download(
        repo_id,
        filename=cfg_filename,
        force_download=force_download,
    )
    sae_path = Path(cfg_path).parent
    return get_sparsify_config_from_disk(
        sae_path, device=device, cfg_overrides=cfg_overrides
    )


def get_sparsify_config_from_disk(
    path: str | Path,
    device: str | None = None,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(path)

    with open(path / SAE_CFG_FILENAME) as f:
        old_cfg_dict = json.load(f)

    config_path = path.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
    else:
        config_dict = {}

    folder_name = path.name
    if folder_name == "embed_tokens":
        hook_name, layer = "hook_embed", 0
    else:
        match = re.search(r"layers[._](\d+)", folder_name)
        if match is None:
            raise ValueError(f"Unrecognized Sparsify folder: {folder_name}")
        layer = int(match.group(1))
        hook_name = f"blocks.{layer}.hook_resid_post"

    d_sae = old_cfg_dict.get("num_latents")
    if d_sae is None:
        d_sae = old_cfg_dict["d_in"] * old_cfg_dict["expansion_factor"]

    cfg_dict: dict[str, Any] = {
        "architecture": "standard",
        "d_in": old_cfg_dict["d_in"],
        "d_sae": d_sae,
        "dtype": "bfloat16",
        "device": device or "cpu",
        "model_name": config_dict.get("model", path.parts[-2]),
        "hook_name": hook_name,
        "hook_layer": layer,
        "hook_head_index": None,
        "activation_fn_str": "topk",
        "activation_fn_kwargs": {
            "k": old_cfg_dict["k"],
            "signed": old_cfg_dict.get("signed", False),
        },
        "apply_b_dec_to_input": not old_cfg_dict.get("normalize_decoder", False),
        "dataset_path": config_dict.get(
            "dataset", "togethercomputer/RedPajama-Data-1T-Sample"
        ),
        "context_size": config_dict.get("ctx_len", 2048),
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_trust_remote_code": True,
        "normalize_activations": "none",
        "neuronpedia_id": None,
    }

    if cfg_overrides:
        cfg_dict.update(cfg_overrides)

    return cfg_dict


def sparsify_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], None]:
    weights_filename = f"{folder_name}/{SPARSIFY_WEIGHTS_FILENAME}"
    sae_path = hf_hub_download(
        repo_id,
        filename=weights_filename,
        force_download=force_download,
    )
    cfg_dict, state_dict = sparsify_disk_loader(
        Path(sae_path).parent, device=device, cfg_overrides=cfg_overrides
    )
    return cfg_dict, state_dict, None


def sparsify_disk_loader(
    path: str | Path,
    device: str = "cpu",
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cfg_dict = get_sparsify_config_from_disk(path, device, cfg_overrides)

    weight_path = Path(path) / SPARSIFY_WEIGHTS_FILENAME
    state_dict_loaded = load_safetensors_weights(
        weight_path, device=device, dtype=cfg_dict.get("dtype")
    )

    W_enc = (
        state_dict_loaded["W_enc"]
        if "W_enc" in state_dict_loaded
        else state_dict_loaded["encoder.weight"].T
    )

    if "W_dec" in state_dict_loaded:
        W_dec = state_dict_loaded["W_dec"].T
    else:
        W_dec = state_dict_loaded["decoder.weight"].T

    if "b_enc" in state_dict_loaded:
        b_enc = state_dict_loaded["b_enc"]
    elif "encoder.bias" in state_dict_loaded:
        b_enc = state_dict_loaded["encoder.bias"]
    else:
        b_enc = torch.zeros(cfg_dict["d_sae"], dtype=W_dec.dtype, device=device)

    if "b_dec" in state_dict_loaded:
        b_dec = state_dict_loaded["b_dec"]
    elif "decoder.bias" in state_dict_loaded:
        b_dec = state_dict_loaded["decoder.bias"]
    else:
        b_dec = torch.zeros(cfg_dict["d_in"], dtype=W_dec.dtype, device=device)

    state_dict = {"W_enc": W_enc, "b_enc": b_enc, "W_dec": W_dec, "b_dec": b_dec}
    return cfg_dict, state_dict


def get_gemma_2_transcoder_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,  # noqa: ARG001
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get config for Gemma-2 transcoders"""
    width_map = {
        "width_4k": 4096,
        "width_16k": 16384,
        "width_65k": 65536,
        "width_262k": 262144,
        "width_524k": 524288,
        "width_1m": 1048576,
    }

    # Extract width from folder name
    d_sae = None
    for width_key, width_value in width_map.items():
        if width_key in folder_name:
            d_sae = width_value
            break

    if d_sae is None:
        # Try to extract from pattern like "width_16k"
        match = re.search(r"width_(\d+)k", folder_name)
        if match:
            d_sae = int(match.group(1)) * 1024
        else:
            raise ValueError(
                f"Could not extract dictionary size from folder name: {folder_name}"
            )

    # Extract layer
    layer_match = re.search(r"layer_(\d+)", folder_name)
    if layer_match:
        layer = int(layer_match.group(1))
    else:
        layer_match = re.search(r"layer_(\d+)", repo_id)
        if layer_match:
            layer = int(layer_match.group(1))
        else:
            raise ValueError("Could not extract layer index")

    # Determine model and dimensions from repo_id
    model_configs = {
        "2b-it": ("gemma-2-2b-it", 2304),
        "2b": ("gemma-2-2b", 2304),
        "9b-it": ("gemma-2-9b-it", 3584),
        "9b": ("gemma-2-9b", 3584),
        "27b-it": ("gemma-2-27b-it", 4608),
        "27b": ("gemma-2-27b", 4608),
    }

    model_name = None
    d_model = None
    for model_key, (name, dim) in model_configs.items():
        if model_key in repo_id:
            model_name = name
            d_model = dim
            break

    if model_name is None:
        raise ValueError(f"Could not determine model from repo_id: {repo_id}")

    return {
        "architecture": "jumprelu_transcoder",
        "d_in": d_model,
        "d_out": d_model,
        "d_sae": d_sae,
        "dtype": "float32",
        "device": device if device is not None else "cpu",
        "activation_fn": "relu",
        "normalize_activations": "none",
        "model_name": model_name,
        "hook_name": f"blocks.{layer}.ln2.hook_normalized",
        "hook_name_out": f"blocks.{layer}.hook_mlp_out",
        "hook_head_index": None,
        "hook_head_index_out": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "apply_b_dec_to_input": False,
        **(cfg_overrides or {}),
    }


def gemma_2_transcoder_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """Load Gemma-2 transcoders from HuggingFace"""
    cfg_dict = get_gemma_2_transcoder_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Download the npz file
    revision = cfg_overrides.get("revision", None) if cfg_overrides else None
    params_filename = f"{folder_name}/params.npz"

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=params_filename,
        force_download=force_download,
        revision=revision,
    )

    # Load weights from npz file
    params = np.load(file_path)

    # Convert to state dict with proper naming
    state_dict = {}
    for key in params.files:
        tensor = torch.tensor(params[key], dtype=torch.float32, device=device)
        # Handle various naming conventions
        key_lower = key.lower()
        if key_lower in ["w_enc", "wenc", "w_e"]:
            state_dict["W_enc"] = tensor
        elif key_lower in ["w_dec", "wdec", "w_d"]:
            state_dict["W_dec"] = tensor
        elif key_lower in ["b_enc", "benc", "b_e"]:
            state_dict["b_enc"] = tensor
        elif key_lower in ["b_dec", "bdec", "b_d"]:
            state_dict["b_dec"] = tensor
        if key_lower in ["threshold"]:
            state_dict["threshold"] = tensor

    return cfg_dict, state_dict, None


def get_mwhanna_transcoder_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,  # noqa: ARG001
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get config for mwhanna transcoders"""

    # Extract layer from folder name
    layer = int(folder_name.replace(".safetensors", "").split("_")[-1])

    try:
        # mwhanna transcoders sometimes have a typo in the config file name, so check for both
        wandb_config_path = hf_hub_download(
            repo_id, "wandb-config.yaml", force_download=force_download
        )
    except EntryNotFoundError:
        wandb_config_path = hf_hub_download(
            repo_id, "wanb-config.yaml", force_download=force_download
        )
    try:
        base_config_path = hf_hub_download(
            repo_id, "config.yaml", force_download=force_download
        )
        with open(base_config_path) as f:
            base_cfg_info: dict[str, Any] = yaml.safe_load(f)
    except EntryNotFoundError:
        # the 14b transcoders don't have a config file for some reason, so just pull the model name from the repo_id
        qwen_3_size_match = re.search(r"qwen3-(\d+(?:\.\d+)?)b", repo_id)
        if not qwen_3_size_match:
            raise ValueError(f"Could not extract Qwen3 size from repo_id: {repo_id}")
        qwen_3_size = qwen_3_size_match.group(1)
        base_cfg_info = {
            "model_name": f"Qwen/Qwen3-{qwen_3_size}B",
        }

    with open(wandb_config_path) as f:
        wandb_cfg_info: dict[str, Any] = yaml.safe_load(f)

    return {
        "architecture": "transcoder",
        "d_in": wandb_cfg_info["d_model"]["value"],
        "d_out": wandb_cfg_info["d_model"]["value"],
        "d_sae": wandb_cfg_info["d_feature"]["value"],
        "dtype": "float32",
        "device": device if device is not None else "cpu",
        "activation_fn": "relu",
        "normalize_activations": "none",
        "model_name": base_cfg_info["model_name"],
        "hook_name": f"blocks.{layer}.mlp.hook_in",
        "hook_name_out": f"blocks.{layer}.hook_mlp_out",
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": wandb_cfg_info["batch_size"]["value"],
        "apply_b_dec_to_input": False,
        "model_from_pretrained_kwargs": {"fold_ln": False},
        **(cfg_overrides or {}),
    }


def mwhanna_transcoder_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """Load mwhanna transcoders from HuggingFace"""
    cfg_dict = get_mwhanna_transcoder_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Download the safetensors file
    revision = cfg_overrides.get("revision", None) if cfg_overrides else None

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=folder_name,
        force_download=force_download,
        revision=revision,
    )

    # Load weights from safetensors
    state_dict = load_safetensors_weights(
        file_path, device=device, dtype=cfg_dict.get("dtype")
    )
    state_dict["W_enc"] = state_dict["W_enc"].T

    return cfg_dict, state_dict, None


def get_safetensors_tensor_shapes(repo_id: str, filename: str) -> dict[str, list[int]]:
    """
    Get tensor shapes from a safetensors file on HuggingFace Hub
    without downloading the entire file.

    Uses HTTP range requests to fetch only the metadata header.

    Args:
        repo_id: HuggingFace repo ID (e.g., "gg-gs/gemma-scope-2-1b-pt")
        filename: Path to the safetensors file within the repo

    Returns:
        Dictionary mapping tensor names to their shapes
    """
    url = hf_hub_url(repo_id, filename)

    # Get HuggingFace headers (includes auth token if available)
    hf_headers = build_hf_headers()

    # Fetch first 8 bytes to get metadata size
    headers = {**hf_headers, "Range": "bytes=0-7"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    meta_size = int.from_bytes(response.content, byteorder="little")

    # Fetch the metadata header
    headers = {**hf_headers, "Range": f"bytes=8-{8 + meta_size - 1}"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    metadata_json = response.content.decode("utf-8").strip()
    metadata = json.loads(metadata_json)

    # Extract tensor shapes, excluding the __metadata__ key
    return {
        name: info["shape"] for name, info in metadata.items() if name != "__metadata__"
    }


def mntss_clt_layer_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,  # noqa: ARG001
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Load a MNTSS CLT layer as a single layer transcoder.
    The assumption is that the `folder_name` is the layer to load as an int
    """

    # warn that this sums the decoders together, so should only be used to find feature activations, not for reconstruction
    warnings.warn(
        "This loads the CLT layer as a single layer transcoder by summing all decoders together. This should only be used to find feature activations, not for reconstruction",
        UserWarning,
    )

    cfg_dict = get_mntss_clt_layer_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # We need to actually load the weights, since the config is missing most information
    encoder_path = hf_hub_download(
        repo_id,
        f"W_enc_{folder_name}.safetensors",
        force_download=force_download,
    )
    decoder_path = hf_hub_download(
        repo_id,
        f"W_dec_{folder_name}.safetensors",
        force_download=force_download,
    )

    encoder_state_dict = load_safetensors_weights(
        encoder_path, device=device, dtype=cfg_dict.get("dtype")
    )
    decoder_state_dict = load_safetensors_weights(
        decoder_path, device=device, dtype=cfg_dict.get("dtype")
    )

    with torch.no_grad():
        state_dict = {
            "W_enc": encoder_state_dict[f"W_enc_{folder_name}"].T,  # type: ignore
            "b_enc": encoder_state_dict[f"b_enc_{folder_name}"],  # type: ignore
            "b_dec": encoder_state_dict[f"b_dec_{folder_name}"],  # type: ignore
            "W_dec": decoder_state_dict[f"W_dec_{folder_name}"].sum(dim=1),  # type: ignore
        }

    return cfg_dict, state_dict, None


def get_mntss_clt_layer_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str,
    force_download: bool = False,  # noqa: ARG001
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load a MNTSS CLT layer as a single layer transcoder.
    The assumption is that the `folder_name` is the layer to load as an int
    """
    base_config_path = hf_hub_download(
        repo_id, "config.yaml", force_download=force_download
    )
    with open(base_config_path) as f:
        cfg_info: dict[str, Any] = yaml.safe_load(f)

    # Get tensor shapes without downloading full files
    encoder_shapes = get_safetensors_tensor_shapes(
        repo_id, f"W_enc_{folder_name}.safetensors"
    )

    # Extract shapes for the required tensors
    b_dec_shape = encoder_shapes[f"b_dec_{folder_name}"]
    b_enc_shape = encoder_shapes[f"b_enc_{folder_name}"]

    return {
        "architecture": "transcoder",
        "d_in": b_dec_shape[0],
        "d_out": b_dec_shape[0],
        "d_sae": b_enc_shape[0],
        "dtype": "float32",
        "device": device if device is not None else "cpu",
        "activation_fn": "relu",
        "normalize_activations": "none",
        "model_name": cfg_info["model_name"],
        "hook_name": f"blocks.{folder_name}.{cfg_info['feature_input_hook']}",
        "hook_name_out": f"blocks.{folder_name}.{cfg_info['feature_output_hook']}",
        "apply_b_dec_to_input": False,
        "model_from_pretrained_kwargs": {"fold_ln": False},
        **(cfg_overrides or {}),
    }


def get_temporal_sae_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get TemporalSAE config without loading weights."""
    # Download config file
    conf_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{folder_name}/conf.yaml",
        force_download=force_download,
    )

    # Load and parse config
    with open(conf_path) as f:
        yaml_config = yaml.safe_load(f)

    # Extract parameters
    d_in = yaml_config["llm"]["dimin"]
    exp_factor = yaml_config["sae"]["exp_factor"]
    d_sae = int(d_in * exp_factor)

    # extract layer from folder_name eg : "layer_12/temporal"
    layer = re.search(r"layer_(\d+)", folder_name)
    if layer is None:
        raise ValueError(f"Could not find layer in folder_name: {folder_name}")
    layer = int(layer.group(1))

    # Build config dict
    cfg_dict = {
        "architecture": "temporal",
        "hook_name": f"blocks.{layer}.hook_resid_post",
        "d_in": d_in,
        "d_sae": d_sae,
        "n_heads": yaml_config["sae"]["n_heads"],
        "n_attn_layers": yaml_config["sae"]["n_attn_layers"],
        "bottleneck_factor": yaml_config["sae"]["bottleneck_factor"],
        "sae_diff_type": yaml_config["sae"]["sae_diff_type"],
        "kval_topk": yaml_config["sae"]["kval_topk"],
        "tied_weights": yaml_config["sae"]["tied_weights"],
        "dtype": yaml_config["data"]["dtype"],
        "device": device,
        "normalize_activations": "constant_scalar_rescale",
        "activation_normalization_factor": yaml_config["sae"]["scaling_factor"],
        "apply_b_dec_to_input": True,
    }

    if cfg_overrides:
        cfg_dict.update(cfg_overrides)

    return cfg_dict


def temporal_sae_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Load TemporalSAE from canrager/temporalSAEs format (safetensors version).

    Expects folder_name to contain:

    - conf.yaml (configuration)
    - latest_ckpt.safetensors (model weights)
    """

    cfg_dict = get_temporal_sae_config_from_hf(
        repo_id=repo_id,
        folder_name=folder_name,
        device=device,
        force_download=force_download,
        cfg_overrides=cfg_overrides,
    )

    # Download checkpoint (safetensors format)
    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{folder_name}/latest_ckpt.safetensors",
        force_download=force_download,
    )

    # Load checkpoint from safetensors
    state_dict_raw = load_safetensors_weights(
        ckpt_path, device=device, dtype=cfg_dict.get("dtype")
    )

    # Convert to SAELens naming convention
    # TemporalSAE uses: D (decoder), E (encoder), b (bias), attn_layers.*
    state_dict = {}

    # Copy attention layers as-is
    for key, value in state_dict_raw.items():
        if key.startswith("attn_layers."):
            state_dict[key] = value.to(device)

    # Main parameters
    state_dict["W_dec"] = state_dict_raw["D"].to(device)
    state_dict["b_dec"] = state_dict_raw["b"].to(device)

    # Handle tied/untied weights
    if "E" in state_dict_raw:
        state_dict["W_enc"] = state_dict_raw["E"].to(device)

    return cfg_dict, state_dict, None


NAMED_PRETRAINED_SAE_LOADERS: dict[str, PretrainedSaeHuggingfaceLoader] = {
    "sae_lens": sae_lens_huggingface_loader,
    "connor_rob_hook_z": connor_rob_hook_z_huggingface_loader,
    "gemma_2": gemma_2_sae_huggingface_loader,
    "gemma_3": gemma_3_sae_huggingface_loader,
    "llama_scope": llama_scope_sae_huggingface_loader,
    "llama_scope_r1_distill": llama_scope_r1_distill_sae_huggingface_loader,
    "dictionary_learning_1": dictionary_learning_sae_huggingface_loader_1,
    "deepseek_r1": deepseek_r1_sae_huggingface_loader,
    "sparsify": sparsify_huggingface_loader,
    "gemma_2_transcoder": gemma_2_transcoder_huggingface_loader,
    "mwhanna_transcoder": mwhanna_transcoder_huggingface_loader,
    "mntss_clt_layer_transcoder": mntss_clt_layer_huggingface_loader,
    "temporal": temporal_sae_huggingface_loader,
    "goodfire": get_goodfire_huggingface_loader,
}


NAMED_PRETRAINED_SAE_CONFIG_GETTERS: dict[str, PretrainedSaeConfigHuggingfaceLoader] = {
    "sae_lens": get_sae_lens_config_from_hf,
    "connor_rob_hook_z": get_connor_rob_hook_z_config_from_hf,
    "gemma_2": get_gemma_2_config_from_hf,
    "gemma_3": get_gemma_3_config_from_hf,
    "llama_scope": get_llama_scope_config_from_hf,
    "llama_scope_r1_distill": get_llama_scope_r1_distill_config_from_hf,
    "dictionary_learning_1": get_dictionary_learning_config_1_from_hf,
    "deepseek_r1": get_deepseek_r1_config_from_hf,
    "sparsify": get_sparsify_config_from_hf,
    "gemma_2_transcoder": get_gemma_2_transcoder_config_from_hf,
    "mwhanna_transcoder": get_mwhanna_transcoder_config_from_hf,
    "mntss_clt_layer_transcoder": get_mntss_clt_layer_config_from_hf,
    "temporal": get_temporal_sae_config_from_hf,
    "goodfire": get_goodfire_config_from_hf,
}
