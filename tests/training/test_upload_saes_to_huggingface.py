from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock

import pytest
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

import sae_lens.training.upload_saes_to_huggingface as upload_module
from sae_lens.saes.sae import SAE
from sae_lens.training.upload_saes_to_huggingface import (
    _build_sae_path,
    _create_default_readme,
    _repo_exists,
    _repo_file_exists,
    _validate_sae_path,
)
from tests.helpers import build_sae_cfg


def test_create_default_readme():
    saes_dict = ["sae1", "sae2"]
    expected_readme = dedent(
        """
        ---
        library_name: saelens
        ---

        # SAEs for use with the SAELens library

        This repository contains the following SAEs:
        - sae1
        - sae2

        Load these SAEs using SAELens as below:
        ```python
        from sae_lens import SAE

        sae = SAE.from_pretrained("jimi/hendrix", "<sae_id>")
        ```
        """
    ).strip()
    assert _create_default_readme("jimi/hendrix", saes_dict) == expected_readme


def test_build_sae_path_saves_live_saes_to_tmpdir(tmp_path: Path):
    cfg = build_sae_cfg()
    sae = SAE.from_dict(cfg.to_dict())
    sae_path = _build_sae_path(sae, str(tmp_path))
    assert sae_path == tmp_path
    assert (tmp_path / "sae_weights.safetensors").exists()
    assert (tmp_path / "cfg.json").exists()


def test_build_sae_path_directly_passes_through_existing_dirs(tmp_path: Path):
    assert _build_sae_path("/sae/path", str(tmp_path)) == Path("/sae/path")
    assert _build_sae_path(Path("/sae/path"), str(tmp_path)) == Path("/sae/path")


def test_repo_exists():
    api = MagicMock()
    assert _repo_exists(api, "user/repo")
    api.repo_info.assert_called_once_with("user/repo")

    response = MagicMock()
    response.status_code = 404
    api.repo_info.side_effect = RepositoryNotFoundError("Not found", response=response)
    assert not _repo_exists(api, "fake/repo")


def test_repo_file_exists(monkeypatch: pytest.MonkeyPatch):
    checked_urls: list[str] = []

    def get_hf_file_metadata(url: str) -> MagicMock:
        checked_urls.append(url)
        if "fake_file.md" in url:
            raise EntryNotFoundError(f"File not found: {url}")
        return MagicMock()

    monkeypatch.setattr(upload_module, "get_hf_file_metadata", get_hf_file_metadata)

    assert _repo_file_exists("user/repo", "README.md", "main")
    assert not _repo_file_exists("user/repo", "fake_file.md", "dev")
    assert checked_urls == [
        "https://huggingface.co/user/repo/resolve/main/README.md",
        "https://huggingface.co/user/repo/resolve/dev/fake_file.md",
    ]


def test_validate_sae_path_errors_if_files_are_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _validate_sae_path(tmp_path)
    (tmp_path / "cfg.json").touch()
    with pytest.raises(FileNotFoundError):
        _validate_sae_path(tmp_path)
    (tmp_path / "sae_weights.safetensors").touch()
    _validate_sae_path(tmp_path)
