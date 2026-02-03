"""Upload SyntheticModel to Hugging Face Hub."""

from __future__ import annotations

import io
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, create_repo, get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from sae_lens.synthetic.hierarchy import Hierarchy
from sae_lens.synthetic.synthetic_model import (
    SYNTHETIC_MODEL_CONFIG_FILENAME,
    SYNTHETIC_MODEL_HIERARCHY_FILENAME,
    SYNTHETIC_MODEL_WEIGHTS_FILENAME,
    SyntheticModel,
)


def upload_synthetic_model_to_huggingface(
    model: SyntheticModel | Path | str,
    hf_repo_id: str,
    hf_path: str | None = None,
    hf_revision: str = "main",
    add_default_readme: bool = True,
    private: bool = False,
) -> None:
    """
    Upload a SyntheticModel to the Hugging Face model hub.

    Args:
        model: The SyntheticModel instance, or path to a saved model directory
        hf_repo_id: The Hugging Face repository ID (e.g., "username/repo-name")
        hf_path: Optional subfolder within the repo. If None, uploads to repo root.
        hf_revision: The branch/revision to upload to, defaults to "main"
        add_default_readme: Whether to add a default README.md if one doesn't exist
        private: Whether to create a private repository if it doesn't exist
    """
    api = HfApi()

    if not _repo_exists(api, hf_repo_id):
        create_repo(hf_repo_id, private=private)

    with TemporaryDirectory() as tmp_dir:
        local_model_path = _build_model_path(model, tmp_dir)
        _validate_model_path(local_model_path)
        _upload_model(
            api,
            local_model_path,
            repo_id=hf_repo_id,
            hf_path=hf_path,
            revision=hf_revision,
        )

    if add_default_readme:
        if _repo_file_exists(hf_repo_id, "README.md", hf_revision):
            pass  # README already exists
        else:
            # Get the model for README generation
            synthetic_model = _get_synthetic_model(model)
            readme = _create_default_readme(hf_repo_id, hf_path, synthetic_model)
            readme_io = io.BytesIO()
            readme_io.write(readme.encode("utf-8"))
            readme_io.seek(0)
            api.upload_file(
                path_or_fileobj=readme_io,
                path_in_repo="README.md",
                repo_id=hf_repo_id,
                revision=hf_revision,
                commit_message="Add README.md",
            )


def _get_synthetic_model(model: SyntheticModel | Path | str) -> SyntheticModel:
    """Get or load a SyntheticModel instance."""
    if isinstance(model, SyntheticModel):
        return model
    path = model if isinstance(model, Path) else Path(model)
    return SyntheticModel.load_from_disk(path)


def _create_default_readme(
    repo_id: str, hf_path: str | None, model: SyntheticModel
) -> str:
    # Build load code snippet
    if hf_path:
        load_code = f'model = SyntheticModel.from_pretrained("{repo_id}", model_path="{hf_path}")'
    else:
        load_code = f'model = SyntheticModel.from_pretrained("{repo_id}")'

    # Build model info section
    model_info_lines = [
        f"- **Number of features**: {model.cfg.num_features:,}",
        f"- **Hidden dimension**: {model.cfg.hidden_dim}",
    ]

    # Hierarchy info
    if model.hierarchy is not None:
        hierarchy = model.hierarchy
        num_roots = len(hierarchy.roots)
        max_depth = _get_hierarchy_max_depth(hierarchy)
        total_nodes = len(hierarchy.feature_indices_used)
        model_info_lines.append("- **Hierarchy**: Yes")
        model_info_lines.append(f"  - Root nodes: {num_roots}")
        model_info_lines.append(f"  - Total nodes: {total_nodes}")
        model_info_lines.append(f"  - Max depth: {max_depth}")
    else:
        model_info_lines.append("- **Hierarchy**: No")

    # Correlation info
    if model.correlation_matrix is not None:
        # Get scale from config if available
        if model.cfg.correlation is not None:
            scale = model.cfg.correlation.correlation_scale
            model_info_lines.append(f"- **Feature correlation**: Yes (scale {scale})")
        else:
            model_info_lines.append("- **Feature correlation**: Yes")
    else:
        model_info_lines.append("- **Feature correlation**: No")

    model_info = "\n".join(model_info_lines)

    # Note: We don't use dedent with f-strings here because interpolated
    # multi-line variables (model_info) break dedent's common prefix detection.
    return f"""\
---
library_name: saelens
---

# Synthetic Model for SAE Training

This repository contains a SyntheticModel for use with SAELens.

## Model Info

{model_info}

## Usage

```python
from sae_lens.synthetic import SyntheticModel

{load_code}
```"""


def _get_hierarchy_max_depth(hierarchy: Hierarchy) -> int:
    """Get the maximum depth of a hierarchy."""
    from sae_lens.synthetic.hierarchy import HierarchyNode

    def node_depth(node: HierarchyNode) -> int:
        if not node.children:
            return 1
        return 1 + max(node_depth(child) for child in node.children)

    if not hierarchy.roots:
        return 0
    return max(node_depth(root) for root in hierarchy.roots)


def _repo_file_exists(repo_id: str, filename: str, revision: str) -> bool:
    try:
        url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
        get_hf_file_metadata(url)
        return True
    except EntryNotFoundError:
        return False


def _repo_exists(api: HfApi, repo_id: str) -> bool:
    try:
        api.repo_info(repo_id)
        return True
    except RepositoryNotFoundError:
        return False


def _upload_model(
    api: HfApi,
    model_path: Path,
    repo_id: str,
    hf_path: str | None,
    revision: str,
) -> None:
    path_in_repo = hf_path if hf_path else "."
    commit_message = (
        f"Upload SyntheticModel {hf_path}" if hf_path else "Upload SyntheticModel"
    )

    # Build allow_patterns based on what files exist
    allow_patterns = [
        SYNTHETIC_MODEL_CONFIG_FILENAME,
        SYNTHETIC_MODEL_WEIGHTS_FILENAME,
    ]
    if (model_path / SYNTHETIC_MODEL_HIERARCHY_FILENAME).exists():
        allow_patterns.append(SYNTHETIC_MODEL_HIERARCHY_FILENAME)

    api.upload_folder(
        folder_path=model_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        revision=revision,
        repo_type="model",
        commit_message=commit_message,
        allow_patterns=allow_patterns,
    )


def _build_model_path(model: SyntheticModel | Path | str, tmp_dir: str) -> Path:
    if isinstance(model, SyntheticModel):
        model.save(tmp_dir)
        return Path(tmp_dir)
    if isinstance(model, Path):
        return model
    return Path(model)


def _validate_model_path(model_path: Path) -> None:
    """Validate that the model files exist in the given path."""
    if not (model_path / SYNTHETIC_MODEL_CONFIG_FILENAME).exists():
        raise FileNotFoundError(
            f"SyntheticModel config file not found: "
            f"{model_path / SYNTHETIC_MODEL_CONFIG_FILENAME}"
        )
    if not (model_path / SYNTHETIC_MODEL_WEIGHTS_FILENAME).exists():
        raise FileNotFoundError(
            f"SyntheticModel weights file not found: "
            f"{model_path / SYNTHETIC_MODEL_WEIGHTS_FILENAME}"
        )
