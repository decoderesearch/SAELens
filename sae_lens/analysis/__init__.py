from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer

__all__ = ["HookedSAETransformer"]

try:
    from sae_lens.analysis.compat import has_transformer_bridge

    if has_transformer_bridge():
        from sae_lens.analysis.sae_transformer_bridge import (  # noqa: F401
            SAETransformerBridge,
        )

        __all__.append("SAETransformerBridge")
except ImportError:
    pass
