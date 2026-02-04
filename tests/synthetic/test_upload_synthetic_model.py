from pathlib import Path
from unittest.mock import MagicMock

import pytest
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

import sae_lens.synthetic.upload_synthetic_model as upload_module
from sae_lens.synthetic import (
    HierarchyConfig,
    LowRankCorrelationConfig,
    SyntheticModel,
    SyntheticModelConfig,
    upload_synthetic_model_to_huggingface,
)
from sae_lens.synthetic.upload_synthetic_model import (
    SYNTHETIC_MODEL_CONFIG_FILENAME,
    _create_default_readme,
    _get_hierarchy_max_depth,
    _repo_file_exists,
    _validate_model_path,
)


def test_create_default_readme_basic() -> None:
    cfg = SyntheticModelConfig(
        num_features=1000,
        hidden_dim=256,
        orthogonalization=None,
    )
    model = SyntheticModel(cfg)

    readme = _create_default_readme("user/repo", None, model)

    assert "# Synthetic Model for SAE Training" in readme
    assert "**Number of features**: 1,000" in readme
    assert "**Hidden dimension**: 256" in readme
    assert "**Hierarchy**: No" in readme
    assert "**Feature correlation**: No" in readme
    assert 'SyntheticModel.from_pretrained("user/repo")' in readme
    assert "model_path" not in readme


def test_create_default_readme_has_valid_yaml_frontmatter() -> None:
    cfg = SyntheticModelConfig(
        num_features=1000,
        hidden_dim=256,
        orthogonalization=None,
    )
    model = SyntheticModel(cfg)

    readme = _create_default_readme("user/repo", None, model)

    # Check YAML frontmatter is properly formatted (no extra indentation)
    lines = readme.split("\n")
    assert lines[0] == "---", f"First line should be '---', got: {repr(lines[0])}"
    assert (
        lines[1] == "library_name: saelens"
    ), f"Second line should be 'library_name: saelens', got: {repr(lines[1])}"
    assert lines[2] == "---", f"Third line should be '---', got: {repr(lines[2])}"

    # No line should have leading whitespace (dedent should have worked)
    for i, line in enumerate(lines):
        if line:  # skip empty lines
            assert not line.startswith(
                " "
            ), f"Line {i} has leading whitespace: {repr(line)}"


def test_create_default_readme_with_hf_path() -> None:
    cfg = SyntheticModelConfig(
        num_features=500,
        hidden_dim=128,
        orthogonalization=None,
    )
    model = SyntheticModel(cfg)

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
        ),
        orthogonalization=None,
        seed=42,
    )
    model = SyntheticModel(cfg)

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
    model = SyntheticModel(cfg)

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
        ),
        orthogonalization=None,
        seed=42,
    )
    model = SyntheticModel(cfg)

    assert model.hierarchy is not None
    depth = _get_hierarchy_max_depth(model.hierarchy)
    assert depth >= 1
    assert depth <= 4  # max_depth + 1 for leaf nodes


def test_upload_synthetic_model_calls_api(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel(cfg)

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


def test_upload_synthetic_model_with_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel(cfg)

    save_path = tmp_path / "test_model"
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
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel(cfg)

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


def test_repo_file_exists_returns_true_when_file_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mock hf_hub_url and get_hf_file_metadata to succeed
    def mock_hf_hub_url(
        *,
        repo_id: str,  # noqa: ARG001
        filename: str,  # noqa: ARG001
        revision: str,  # noqa: ARG001
    ) -> str:
        return "http://url"

    def mock_get_metadata(url: str) -> dict[str, int]:  # noqa: ARG001
        return {"size": 100}

    monkeypatch.setattr(upload_module, "hf_hub_url", mock_hf_hub_url)
    monkeypatch.setattr(upload_module, "get_hf_file_metadata", mock_get_metadata)

    result = _repo_file_exists("test/repo", "README.md", "main")
    assert result is True


def test_repo_file_exists_returns_false_when_file_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mock_hf_hub_url(
        *,
        repo_id: str,  # noqa: ARG001
        filename: str,  # noqa: ARG001
        revision: str,  # noqa: ARG001
    ) -> str:
        return "http://url"

    def raise_not_found(url: str) -> None:
        raise EntryNotFoundError(f"File not found: {url}")

    monkeypatch.setattr(upload_module, "hf_hub_url", mock_hf_hub_url)
    monkeypatch.setattr(upload_module, "get_hf_file_metadata", raise_not_found)

    result = _repo_file_exists("test/repo", "nonexistent.txt", "main")
    assert result is False


def test_upload_with_string_path_to_saved_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = SyntheticModelConfig(
        num_features=32,
        hidden_dim=16,
        orthogonalization=None,
    )
    model = SyntheticModel(cfg)

    save_path = tmp_path / "test_model"
    model.save(save_path)

    # Mock HfApi
    mock_api = MagicMock()
    mock_api.repo_info.return_value = True  # Repo exists
    monkeypatch.setattr(upload_module, "HfApi", lambda: mock_api)
    monkeypatch.setattr(upload_module, "_repo_file_exists", lambda *_args: True)

    # Upload using string path
    upload_synthetic_model_to_huggingface(
        model=str(save_path),  # String path instead of SyntheticModel
        hf_repo_id="test/repo",
        add_default_readme=False,
    )

    # Verify upload_folder was called
    mock_api.upload_folder.assert_called_once()
    call_kwargs = mock_api.upload_folder.call_args.kwargs
    assert Path(call_kwargs["folder_path"]) == save_path


def test_validate_model_path_raises_for_missing_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config file not found"):
        _validate_model_path(tmp_path)


def test_validate_model_path_raises_for_missing_weights(tmp_path: Path) -> None:
    # Create config file but not weights
    (tmp_path / SYNTHETIC_MODEL_CONFIG_FILENAME).write_text("{}")

    with pytest.raises(FileNotFoundError, match="weights file not found"):
        _validate_model_path(tmp_path)
