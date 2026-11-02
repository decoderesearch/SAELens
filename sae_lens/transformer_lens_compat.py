"""Imports from transformer-lens whose location differs between supported versions."""

from transformer_lens.utilities import (
    USE_DEFAULT_VALUE,
    get_tokens_with_bos_removed,
    lm_cross_entropy_loss,
)

try:
    from transformer_lens import HookedRootModule
except ImportError:
    # Early transformer-lens 3.x releases only export it from hook_points, which
    # transformer-lens 4.0 removed it from
    from transformer_lens.hook_points import HookedRootModule  # type: ignore

__all__ = [
    "HookedRootModule",
    "USE_DEFAULT_VALUE",
    "get_tokens_with_bos_removed",
    "lm_cross_entropy_loss",
]
