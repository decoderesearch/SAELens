import torch

from sae_lens.synthetic import FeatureDictionary


@torch.no_grad()
def init_sae_to_match_feature_dict(
    sae: torch.nn.Module,
    feature_dict: FeatureDictionary,
    noise_level: float = 0.0,
    feature_ordering: torch.Tensor | None = None,
) -> None:
    """
    Initialize an SAE's weights to match a feature dictionary.

    This can be useful for:

    - Starting training from a known good initialization
    - Testing SAE evaluation code with ground truth
    - Ablation studies on initialization

    Args:
        sae: The SAE to initialize. Must have W_enc and W_dec attributes.
        feature_dict: The feature dictionary to match
        noise_level: Standard deviation of Gaussian noise to add (0 = exact match)
        feature_ordering: Optional permutation of feature indices
    """
    features = feature_dict.feature_vectors  # [num_features, hidden_dim]
    min_dim = min(sae.W_enc.shape[1], features.shape[0])  # type: ignore[attr-defined]

    if feature_ordering is not None:
        features = features[feature_ordering]

    features = features[:min_dim]

    # W_enc is [hidden_dim, d_sae], feature vectors are [num_features, hidden_dim]
    sae.W_enc.data[:, :min_dim] = (  # type: ignore[index]
        features.T + torch.randn_like(features.T) * noise_level
    )
    sae.W_dec.data = sae.W_enc.data.T.clone()  # type: ignore[union-attr]
