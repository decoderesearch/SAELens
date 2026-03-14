import re
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Sequence, TypeVar

import torch
from transformers import PreTrainedTokenizerBase

from sae_lens.constants import DTYPE_MAP, DTYPE_TO_STR

K = TypeVar("K")
V = TypeVar("V")


def filter_valid_dataclass_fields(
    source: dict[str, V] | object,
    destination: object | type,
    whitelist_fields: Sequence[str] | None = None,
) -> dict[str, V]:
    """Filter a source dict or dataclass instance to only include fields that are present in the destination dataclass."""

    if not is_dataclass(destination):
        raise ValueError(f"{destination} is not a dataclass")

    if is_dataclass(source) and not isinstance(source, type):
        source_dict = asdict(source)
    elif isinstance(source, dict):
        source_dict = source
    else:
        raise ValueError(f"{source} is not a dict or dataclass")

    valid_field_names = {field.name for field in fields(destination)}
    if whitelist_fields is not None:
        valid_field_names = valid_field_names.union(whitelist_fields)
    return {key: val for key, val in source_dict.items() if key in valid_field_names}


def extract_stop_at_layer_from_tlens_hook_name(hook_name: str) -> int | None:
    """Extract the stop_at layer from a HookedTransformer hook name.

    Returns None if the hook name is not a valid HookedTransformer hook name.
    """
    layer = extract_layer_from_tlens_hook_name(hook_name)
    return None if layer is None else layer + 1


def extract_layer_from_tlens_hook_name(hook_name: str) -> int | None:
    """Extract the layer from a HookedTransformer hook name.

    Returns None if the hook name is not a valid HookedTransformer hook name.
    """
    hook_match = re.search(r"\.(\d+)\.", hook_name)
    return None if hook_match is None else int(hook_match.group(1))


@contextmanager
def temporary_seed(seed: int | None):
    """Context manager that temporarily sets the global torch random seed.

    If seed is None, this is a no-op. Otherwise the RNG state for CPU and all
    CUDA devices is saved before and restored after the block, so only code
    inside the ``with`` statement is affected by the seed.
    """
    if seed is None:
        yield
        return
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        yield


@contextmanager
def path_or_tmp_dir(path: str | Path | None):
    """Context manager that yields a concrete Path for path.

    - If path is None, creates a TemporaryDirectory and yields its Path.
      The directory is cleaned up on context exit.
    - Otherwise, yields Path(path) without creating or cleaning.
    """
    if path is None:
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)
    else:
        yield Path(path)


_SPECIAL_TOKEN_ATTRS = (
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "sep_token_id",
    "decoder_start_token_id",
)


def get_special_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Get the structural special token IDs (BOS, EOS, PAD, SEP, decoder_start).

    This only returns tokens with dedicated tokenizer attributes, not
    additional_special_tokens. Chat template tokens (e.g. <start_of_turn>,
    <end_of_turn>) are typically registered as additional special tokens and
    are intentionally excluded so they are not filtered from activations.
    """
    special_tokens = set()
    for attr in _SPECIAL_TOKEN_ATTRS:
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_tokens.add(token_id)
    return list(special_tokens)


def str_to_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Convert a string to a torch.dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in DTYPE_MAP:
        raise ValueError(
            f"Invalid dtype: {dtype}. Must be one of {list(DTYPE_MAP.keys())}"
        )
    return DTYPE_MAP[dtype]


def dtype_to_str(dtype: torch.dtype) -> str:
    """Convert a torch.dtype to a string."""
    if dtype not in DTYPE_TO_STR:
        raise ValueError(
            f"Invalid dtype: {dtype}. Must be one of {list(DTYPE_TO_STR.keys())}"
        )
    return DTYPE_TO_STR[dtype]


def cosine_similarities(
    mat1: torch.Tensor, mat2: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute cosine similarities between each row of mat1 and each row of mat2.

    Args:
        mat1: Tensor of shape [n1, d]
        mat2: Tensor of shape [n2, d]. If not provided, mat1 = mat2

    Returns:
        Tensor of shape [n1, n2] with cosine similarities
    """
    if mat2 is None:
        mat2 = mat1
    # Clamp norm to 1e-8 to prevent division by zero. This threshold is chosen
    # to be small enough to not affect normal vectors but large enough to avoid
    # numerical instability. Zero vectors will effectively map to zero similarity.
    mat1_normed = mat1 / mat1.norm(dim=1, keepdim=True).clamp(min=1e-8)
    mat2_normed = mat2 / mat2.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return mat1_normed @ mat2_normed.T
