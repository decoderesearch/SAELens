from collections import defaultdict

from huggingface_hub import HfApi

from sae_lens.loading.pretrained_saes_directory import (
    PretrainedSAELookup,
    get_pretrained_saes_directory,
)


def is_model_file(filepath: str) -> bool:
    """Check if a filepath has a model file extension."""
    model_extensions = [".npz", ".safetensors", ".pt", ".json"]
    return any(filepath.endswith(ext) for ext in model_extensions)


def is_file_path(path: str) -> bool:
    """Determine if a path likely points to a file based on model file extensions or structure."""
    # Check for model file extensions
    if is_model_file(path):
        return True

    # If the path ends at a component that looks like a trainer ID (trainer_X)
    # it's likely a file path even without an extension
    if path.split("/")[-1].startswith("trainer_"):
        return True

    # Check for version tags or other special indicators
    last_component = path.split("/")[-1]
    return ":" in last_component or "." in last_component


def check_path_existence(sae_path: str, huggingface_files: list[str]) -> bool:
    """
    Check if a SAELens path exists in HuggingFace files.

    - For exact matches
    - For paths with different repo prefixes
    - For trainer directories
    """
    # Case 1: Direct match
    if sae_path in huggingface_files:
        return True

    # Case 2: Check for paths that might be subpaths or have added prefixes
    # For example, the HF path might include a main/ or other prefix
    for file in huggingface_files:
        # If the file path ends with our sae_path
        if file.endswith(sae_path):
            return True

        # If our sae_path ends with the file path
        if sae_path.endswith(file):
            return True

        # If our path is in the middle of a longer path
        if "/" + sae_path + "/" in "/" + file + "/":
            return True

    # Case 3: Special handling for trainer paths
    # If the path ends with trainer_X, check if it's part of a file path
    if sae_path.split("/")[-1].startswith("trainer_"):
        trainer_id = sae_path.split("/")[-1]
        parent_dir = "/".join(sae_path.split("/")[:-1])

        # Check if any file is in this trainer directory
        for file in huggingface_files:
            if file.startswith(sae_path + "/") or file.startswith(
                parent_dir + "/" + trainer_id + "/"
            ):
                return True

    # Case 4: Check directories
    if not is_file_path(sae_path):
        directory_prefix = sae_path + "/"
        for file in huggingface_files:
            if file.startswith(directory_prefix):
                return True

    return False


def get_hf_paths_for_sae(lookup: PretrainedSAELookup, sae_path: str) -> list[str]:
    """Return the HuggingFace weight files (or folder) an SAE needs to load."""
    # mntss CLTs keep every layer at the repo root as W_enc_{layer}.safetensors
    # and W_dec_{layer}.safetensors, and the yaml path is just the layer number
    if lookup.conversion_func == "mntss_clt_layer_transcoder":
        return [f"W_enc_{sae_path}.safetensors", f"W_dec_{sae_path}.safetensors"]
    return [sae_path]


api = HfApi()
saes_directory = get_pretrained_saes_directory()

# Group lookups by repo_id
repo_id_to_lookups = defaultdict(list)
for _, lookup in saes_directory.items():
    repo_id_to_lookups[lookup.repo_id].append(lookup)

output = []

# Process each unique repo_id
for repo_id, lookups in repo_id_to_lookups.items():
    # Get all files in the HuggingFace repo
    huggingface_files = api.list_repo_files(repo_id)

    # Check if each sae_lens path exists in HuggingFace. We only report paths
    # missing from HuggingFace: listing HF files that sae_lens doesn't reference
    # is mostly noise, and pushes the issue body past GitHub's size limit.
    missing_in_huggingface: set[str] = set()
    for lookup in lookups:
        for sae_path in lookup.saes_map.values():
            hf_paths = get_hf_paths_for_sae(lookup, sae_path)
            if not all(check_path_existence(p, huggingface_files) for p in hf_paths):
                missing_in_huggingface.add(sae_path)

    if missing_in_huggingface:
        output.append("\nPaths in sae_lens that don't exist in HuggingFace repo:")
        for sae_path in sorted(missing_in_huggingface):
            output.append(f"{repo_id}  {sae_path}")

if output:
    print("\n".join(output))
