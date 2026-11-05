from sae_lens.analysis.compat import has_hooked_transformer
from sae_lens.analysis.sae_transformer_bridge import SAETransformerBridge

__all__: list[str] = ["SAETransformerBridge"]

if has_hooked_transformer():
    from sae_lens.analysis.hooked_sae_transformer import (  # noqa: F401
        HookedSAETransformer,
    )

    __all__.append("HookedSAETransformer")
