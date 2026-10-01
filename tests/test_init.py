import pytest

import sae_lens


def test_accessing_HookedSAETransformer_without_HookedTransformer_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
):
    # transformer-lens >= 4.0 removed HookedTransformer, so HookedSAETransformer
    # isn't defined and module __getattr__ is used instead
    monkeypatch.delattr(sae_lens, "HookedSAETransformer", raising=False)
    assert not hasattr(sae_lens, "HookedSAETransformer")
    with pytest.raises(AttributeError, match="Use SAETransformerBridge instead"):
        getattr(sae_lens, "HookedSAETransformer")  # noqa: B009
