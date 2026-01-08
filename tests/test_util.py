from pathlib import Path

import pytest
import torch
from transformer_lens import HookedTransformer

from sae_lens.util import (
    cosine_similarities,
    dtype_to_str,
    extract_stop_at_layer_from_tlens_hook_name,
    get_special_token_ids,
    path_or_tmp_dir,
    str_to_dtype,
)


@pytest.mark.parametrize(
    "hook_name,expected_layer",
    [
        ("blocks.0.attn.hook_q", 1),
        ("blocks.12.attn.hook_k", 13),
        ("blocks.999.attn.hook_v", 1000),
        ("blocks.42.mlp.hook_pre", 43),
    ],
)
def test_extract_stop_at_layer_from_tlens_hook_name_valid(
    hook_name: str, expected_layer: int
):
    assert extract_stop_at_layer_from_tlens_hook_name(hook_name) == expected_layer


@pytest.mark.parametrize(
    "hook_name",
    [
        "blocks.attn.hook_q",  # missing layer number
        "blocks..attn.hook_q",  # empty layer number
        "hook_q",  # no layer info
        "blocks.abc.attn.hook_q",  # non-numeric layer
        "",  # empty string
    ],
)
def test_extract_stop_at_layer_from_tlens_hook_name_invalid(hook_name: str):
    assert extract_stop_at_layer_from_tlens_hook_name(hook_name) is None


def test_path_or_tmp_dir_with_none():
    with path_or_tmp_dir(None) as path:
        assert isinstance(path, Path)
        assert path.exists()
        assert path.is_dir()
        # Create a test file to verify the directory works
        test_file = path / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
    # Directory should be cleaned up after context exit
    assert not path.exists()


def test_path_or_tmp_dir_with_path(tmp_path: Path):
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    with path_or_tmp_dir(test_dir) as path:
        assert isinstance(path, Path)
        assert path == test_dir
        assert path.exists()
        assert path.is_dir()
    # Directory should still exist after context exit (not cleaned up)
    assert test_dir.exists()


def test_path_or_tmp_dir_with_string_path(tmp_path: Path):
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    with path_or_tmp_dir(str(test_dir)) as path:
        assert isinstance(path, Path)
        assert path == test_dir
        assert path.exists()
        assert path.is_dir()
    # Directory should still exist after context exit (not cleaned up)
    assert test_dir.exists()


def test_get_special_token_ids():
    # Create a mock tokenizer with some special tokens
    class MockTokenizer:
        def __init__(self):
            self.bos_token_id = 1
            self.eos_token_id = 2
            self.pad_token_id = 3
            self.unk_token_id = None  # Test handling of None values
            self.special_tokens_map = {
                "additional_special_tokens": ["<extra_0>", "<extra_1>"],
                "mask_token": "<mask>",
            }

        def convert_tokens_to_ids(self, token: str) -> int:
            token_map = {"<extra_0>": 4, "<extra_1>": 5, "<mask>": 6}
            return token_map[token]

    tokenizer = MockTokenizer()
    special_tokens = get_special_token_ids(tokenizer)  # type: ignore

    # Check that all expected token IDs are present
    assert set(special_tokens) == {1, 2, 3, 4, 5, 6}

    # Check that None values are properly handled
    assert None not in special_tokens


def test_get_special_token_ids_works_with_real_models(ts_model: HookedTransformer):
    special_tokens = get_special_token_ids(ts_model.tokenizer)  # type: ignore
    assert special_tokens == [50256]


@pytest.mark.parametrize(
    "dtype_str,expected_dtype",
    [
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("torch.float32", torch.float32),
        ("torch.float64", torch.float64),
        ("torch.float16", torch.float16),
        ("torch.bfloat16", torch.bfloat16),
    ],
)
def test_str_to_dtype_valid(dtype_str: str, expected_dtype: torch.dtype):
    assert str_to_dtype(dtype_str) == expected_dtype


@pytest.mark.parametrize(
    "invalid_dtype",
    [
        "invalid",
        "float33",
        "int32",
        "torch.int32",
        "",
        "FLOAT32",  # case-sensitive
    ],
)
def test_str_to_dtype_invalid_raises_error(invalid_dtype: str):
    with pytest.raises(ValueError, match="Invalid dtype"):
        str_to_dtype(invalid_dtype)


@pytest.mark.parametrize(
    "dtype,expected_str",
    [
        (torch.float32, "float32"),
        (torch.float64, "float64"),
        (torch.float16, "float16"),
        (torch.bfloat16, "bfloat16"),
    ],
)
def test_dtype_to_str_valid(dtype: torch.dtype, expected_str: str):
    assert dtype_to_str(dtype) == expected_str


@pytest.mark.parametrize(
    "invalid_dtype",
    [
        torch.int32,
        torch.int64,
        torch.complex64,
        torch.bool,
    ],
)
def test_dtype_to_str_invalid_raises_error(invalid_dtype: torch.dtype):
    with pytest.raises(ValueError, match="Invalid dtype"):
        dtype_to_str(invalid_dtype)


def test_cosine_similarities_identity_for_same_matrix():
    mat = torch.randn(5, 10)
    mat = mat / mat.norm(dim=1, keepdim=True)

    cos_sims = cosine_similarities(mat, mat)

    # Diagonal should be 1
    torch.testing.assert_close(torch.diag(cos_sims), torch.ones(5), atol=1e-5, rtol=0)


def test_cosine_similarities_orthogonal_vectors():
    mat1 = torch.eye(3)
    mat2 = torch.eye(3)

    cos_sims = cosine_similarities(mat1, mat2)

    expected = torch.eye(3)
    torch.testing.assert_close(cos_sims, expected, atol=1e-5, rtol=0)


def test_cosine_similarities_shape():
    mat1 = torch.randn(4, 10)
    mat2 = torch.randn(6, 10)

    cos_sims = cosine_similarities(mat1, mat2)

    assert cos_sims.shape == (4, 6)


def test_cosine_similarities_range():
    mat1 = torch.randn(5, 10)
    mat2 = torch.randn(7, 10)

    cos_sims = cosine_similarities(mat1, mat2)

    assert torch.all(cos_sims >= -1.0 - 1e-5)
    assert torch.all(cos_sims <= 1.0 + 1e-5)


def test_cosine_similarities_single_matrix_uses_self():
    """When mat2 is None, should compare mat1 to itself."""
    mat = torch.randn(5, 10)
    cos_sims = cosine_similarities(mat)
    cos_sims_explicit = cosine_similarities(mat, mat)
    torch.testing.assert_close(cos_sims, cos_sims_explicit)


def test_cosine_similarities_negated_vectors():
    """Negated vectors should have cosine similarity -1."""
    mat = torch.randn(5, 10)
    cos_sims = cosine_similarities(mat, -mat)
    diagonal = torch.diag(cos_sims)
    torch.testing.assert_close(diagonal, -torch.ones(5))


def test_cosine_similarities_scale_invariant():
    """Cosine similarity should be invariant to scaling."""
    mat1 = torch.randn(5, 10)
    mat2 = mat1 * 5.0  # Scaled version

    cos_sims = cosine_similarities(mat1, mat2)
    diagonal = torch.diag(cos_sims)
    torch.testing.assert_close(diagonal, torch.ones(5))


def test_cosine_similarities_handles_near_zero_vectors():
    """Should handle vectors with very small norms without crashing."""
    mat1 = torch.randn(3, 10)
    mat2 = torch.randn(3, 10)
    mat2[0] = 1e-10  # Near-zero vector

    cos_sims = cosine_similarities(mat1, mat2)
    assert cos_sims.shape == (3, 3)
    assert torch.all(torch.isfinite(cos_sims))


def test_cosine_similarities_handles_exact_zero_vector():
    """Should handle exact zero vectors without crashing."""
    mat1 = torch.randn(3, 10)
    mat2 = torch.randn(3, 10)
    mat2[0] = 0.0  # Exact zero vector

    cos_sims = cosine_similarities(mat1, mat2)
    assert torch.all(torch.isfinite(cos_sims))


def test_cosine_similarities_symmetric_when_same_matrix():
    """Self-similarity matrix should be symmetric."""
    mat = torch.randn(5, 10)
    cos_sims = cosine_similarities(mat)
    torch.testing.assert_close(cos_sims, cos_sims.T)
