from sae_lens.analysis.compat import has_hooked_transformer, has_transformer_bridge

__all__: list[str] = []

if has_hooked_transformer():
    from sae_lens.analysis.hooked_sae_transformer import (  # noqa: F401
        HookedSAETransformer,
    )

    __all__.append("HookedSAETransformer")

try:
    if has_transformer_bridge():
        from sae_lens.analysis.sae_transformer_bridge import (  # noqa: F401
            SAETransformerBridge,
        )

        __all__.append("SAETransformerBridge")
except ImportError:
    pass
