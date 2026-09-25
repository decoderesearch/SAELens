"""Imports from transformer-lens whose location differs between supported versions."""

try:
    from transformer_lens import HookedRootModule
except ImportError:
    # transformer-lens 2.x and early 3.x only export it from hook_points
    from transformer_lens.hook_points import HookedRootModule  # type: ignore

try:
    from transformer_lens.utilities import (
        USE_DEFAULT_VALUE,
        get_tokens_with_bos_removed,
        lm_cross_entropy_loss,
    )
except ImportError:  # transformer-lens < 3.0 has these in transformer_lens.utils
    from transformer_lens.utils import (  # type: ignore
        USE_DEFAULT_VALUE,
        get_tokens_with_bos_removed,
        lm_cross_entropy_loss,
    )

__all__ = [
    "HookedRootModule",
    "USE_DEFAULT_VALUE",
    "get_tokens_with_bos_removed",
    "lm_cross_entropy_loss",
]
