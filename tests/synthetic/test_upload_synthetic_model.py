import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sae_lens.synthetic import (
    HierarchyConfig,
    LowRankCorrelationConfig,
    SyntheticModel,
    SyntheticModelConfig,
    upload_synthetic_model_to_huggingface,
)
from sae_lens.synthetic.upload_synthetic_model import (
    _create_default_readme,
    _get_hierarchy_max_depth,
)


def test_create_default_readme_basic() -> None:
    cfg = SyntheticModelConfig(
        num_features=1000,
        hidden_dim=256,
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)

    readme = _create_default_readme("user/repo", None, model)

    assert "# Synthetic Model for SAE Training" in readme
    assert "**Number of features**: 1,000" in readme
    assert "**Hidden dimension**: 256" in readme
    assert "**Hierarchy**: No" in readme
    assert "**Feature correlation**: No" in readme
    assert 'SyntheticModel.from_pretrained("user/repo")' in readme
    assert "model_path" not in readme


def test_create_default_readme_with_hf_path() -> None:
    cfg = SyntheticModelConfig(
        num_features=500,
        hidden_dim=128,
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)

    readme = _create_default_readme("user/repo", "my_model", model)

    assert 'from_pretrained("user/repo", model_path="my_model")' in readme


def test_create_default_readme_with_hierarchy() -> None:
    cfg = SyntheticModelConfig(
        num_features=100,
        hidden_dim=64,
        hierarchy=HierarchyConfig(
            total_root_nodes=10,
            branching_factor=3,
            max_depth=2,
            seed=42,
        ),
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)

    readme = _create_default_readme("user/repo", None, model)

    assert "**Hierarchy**: Yes" in readme
    assert "Root nodes:" in readme
    assert "Total nodes:" in readme
    assert "Max depth:" in readme


def test_create_default_readme_with_correlation() -> None:
    cfg = SyntheticModelConfig(
        num_features=100,
        hidden_dim=64,
        correlation=LowRankCorrelationConfig(rank=8, correlation_scale=0.2),
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)

    readme = _create_default_readme("user/repo", None, model)

    assert "**Feature correlation**: Yes (scale 0.2)" in readme


def test_get_hierarchy_max_depth() -> None:
    cfg = SyntheticModelConfig(
        num_features=100,
        hidden_dim=64,
        hierarchy=HierarchyConfig(
            total_root_nodes=5,
            branching_factor=2,
            max_depth=3,
            seed=42,
        ),
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)

    assert model.hierarchy is not None
    depth = _get_hierarchy_max_depth(model.hierarchy)
    assert depth >= 1
    assert depth <= 4  # max_depth + 1 for leaf nodes


def test_upload_synthetic_model_calls_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub.utils import RepositoryNotFoundError

    import sae_lens.synthetic.upload_synthetic_model as upload_module

    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)

    # Mock HfApi
    mock_api = MagicMock()
    mock_api.repo_info.side_effect = RepositoryNotFoundError("Not found")
    monkeypatch.setattr(upload_module, "HfApi", lambda: mock_api)

    # Mock create_repo
    mock_create_repo = MagicMock()
    monkeypatch.setattr(upload_module, "create_repo", mock_create_repo)

    # Mock _repo_file_exists to return False (no README)
    monkeypatch.setattr(upload_module, "_repo_file_exists", lambda *_args: False)

    upload_synthetic_model_to_huggingface(
        model=model,
        hf_repo_id="test/repo",
        hf_path=None,
        add_default_readme=True,
    )

    # Verify create_repo was called
    mock_create_repo.assert_called_once_with("test/repo", private=False)

    # Verify upload_folder was called
    mock_api.upload_folder.assert_called_once()
    call_kwargs = mock_api.upload_folder.call_args.kwargs
    assert call_kwargs["repo_id"] == "test/repo"
    assert call_kwargs["path_in_repo"] == "."

    # Verify upload_file was called for README
    mock_api.upload_file.assert_called_once()
    readme_call_kwargs = mock_api.upload_file.call_args.kwargs
    assert readme_call_kwargs["path_in_repo"] == "README.md"


def test_upload_synthetic_model_with_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import sae_lens.synthetic.upload_synthetic_model as upload_module

    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        model.save(save_path)

        # Mock HfApi
        mock_api = MagicMock()
        mock_api.repo_info.return_value = True  # Repo exists
        monkeypatch.setattr(upload_module, "HfApi", lambda: mock_api)

        # Mock _repo_file_exists to return True (README exists)
        monkeypatch.setattr(upload_module, "_repo_file_exists", lambda *_args: True)

        upload_synthetic_model_to_huggingface(
            model=save_path,
            hf_repo_id="test/repo",
            hf_path="subfolder",
            add_default_readme=True,
        )

        # Verify upload_folder was called with correct path
        mock_api.upload_folder.assert_called_once()
        call_kwargs = mock_api.upload_folder.call_args.kwargs
        assert call_kwargs["path_in_repo"] == "subfolder"

        # README should not be uploaded since it already exists
        mock_api.upload_file.assert_not_called()


def test_upload_synthetic_model_skip_readme(monkeypatch: pytest.MonkeyPatch) -> None:
    import sae_lens.synthetic.upload_synthetic_model as upload_module

    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel.from_config(cfg)

    # Mock HfApi
    mock_api = MagicMock()
    mock_api.repo_info.return_value = True  # Repo exists
    monkeypatch.setattr(upload_module, "HfApi", lambda: mock_api)

    upload_synthetic_model_to_huggingface(
        model=model,
        hf_repo_id="test/repo",
        add_default_readme=False,
    )

    # Verify upload_folder was called
    mock_api.upload_folder.assert_called_once()

    # README should not be uploaded
    mock_api.upload_file.assert_not_called()
