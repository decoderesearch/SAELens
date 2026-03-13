from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from sae_lens.analysis.preflight import (
    check_sae_hook_compatibility,
    check_sae_metadata,
    check_sae_reconstruction,
    run_sae_preflight,
)

__all__ = [
    "HookedSAETransformer",
    "check_sae_hook_compatibility",
    "check_sae_metadata",
    "check_sae_reconstruction",
    "run_sae_preflight",
]

try:
    from sae_lens.analysis.compat import has_transformer_bridge

    if has_transformer_bridge():
        from sae_lens.analysis.sae_transformer_bridge import (  # noqa: F401
            SAETransformerBridge,
        )

        __all__.append("SAETransformerBridge")
except ImportError:
    pass
