"""
Plotting utilities for visualizing SAE training on synthetic data.

This module provides functions for:
- Plotting cosine similarities between SAE features and true features
- Automatically reordering features for better visualization
- Creating comparison plots between encoder and decoder
"""

from collections.abc import Iterable
from pathlib import Path

import torch

from sae_lens.toy_model.feature_dictionary import FeatureDictionary


def cosine_similarities(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarities between each row of mat1 and each row of mat2.

    Args:
        mat1: Tensor of shape [n1, d]
        mat2: Tensor of shape [n2, d]

    Returns:
        Tensor of shape [n1, n2] with cosine similarities
    """
    mat1_normed = mat1 / mat1.norm(dim=1, keepdim=True).clamp(min=1e-8)
    mat2_normed = mat2 / mat2.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return mat1_normed @ mat2_normed.T


def find_best_feature_ordering(
    sae_features: torch.Tensor,
    true_features: torch.Tensor,
) -> torch.Tensor:
    """
    Find the best ordering of SAE features to match true features.

    Reorders SAE features so that each SAE latent aligns with its best-matching
    true feature in order. This makes cosine similarity plots more interpretable.

    Args:
        sae_features: SAE decoder weights of shape [d_sae, hidden_dim]
        true_features: True feature vectors of shape [num_features, hidden_dim]

    Returns:
        Tensor of indices that reorders sae_features for best alignment
    """
    cos_sims = cosine_similarities(sae_features, true_features)
    best_matches = torch.argmax(torch.abs(cos_sims), dim=1)
    return torch.argsort(best_matches)


def find_best_feature_ordering_from_sae(
    sae: torch.nn.Module,
    feature_dict: FeatureDictionary,
) -> torch.Tensor:
    """
    Find the best feature ordering for an SAE given a feature dictionary.

    Args:
        sae: SAE with W_dec attribute of shape [d_sae, hidden_dim]
        feature_dict: The feature dictionary containing true features

    Returns:
        Tensor of indices that reorders SAE latents for best alignment
    """
    sae_features = sae.W_dec.detach()  # type: ignore[attr-defined]
    true_features = feature_dict.feature_vectors.detach()
    return find_best_feature_ordering(sae_features, true_features)


def find_best_feature_ordering_across_saes(
    saes: Iterable[torch.nn.Module],
    feature_dict: FeatureDictionary,
) -> torch.Tensor:
    """
    Find the best feature ordering that works across multiple SAEs.

    Useful for creating consistent orderings across training snapshots.

    Args:
        saes: Iterable of SAEs to consider
        feature_dict: The feature dictionary containing true features

    Returns:
        The best ordering tensor found across all SAEs
    """
    best_score = float("-inf")
    best_ordering: torch.Tensor | None = None

    true_features = feature_dict.feature_vectors.detach()

    for sae in saes:
        sae_features = sae.W_dec.detach()  # type: ignore[attr-defined]
        cos_sims = cosine_similarities(sae_features, true_features)
        cos_sims = torch.round(cos_sims * 100) / 100  # Reduce numerical noise

        ordering = find_best_feature_ordering(sae_features, true_features)
        score = cos_sims[ordering, torch.arange(cos_sims.shape[1])].mean().item()

        if score > best_score:
            best_score = score
            best_ordering = ordering

    if best_ordering is None:
        raise ValueError("No SAEs provided")

    return best_ordering


def plot_sae_feature_similarity(
    sae: torch.nn.Module,
    feature_dict: FeatureDictionary,
    title: str | None = None,
    reorder_features: bool | torch.Tensor = False,
    decoder_only: bool = False,
    show_values: bool = False,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot cosine similarities between SAE features and true features.

    Creates a heatmap showing how well each SAE latent aligns with each
    true feature. Useful for understanding what the SAE has learned.

    Args:
        sae: The SAE to visualize. Must have W_enc and W_dec attributes.
        feature_dict: The feature dictionary containing true features
        title: Plot title. If None, a default title is used.
        reorder_features: If True, automatically reorders features for best alignment.
            If a tensor, uses that as the ordering.
        decoder_only: If True, only plots the decoder (not encoder and decoder side-by-side)
        show_values: If True, shows numeric values on the heatmap
        figsize: Figure size as (width, height) in inches
        save_path: If provided, saves the figure to this path

    Example:
        >>> sae = TrainingSAE(cfg)
        >>> # ... train SAE ...
        >>> plot_sae_feature_similarity(sae, feature_dict, reorder_features=True)
    """
    # Import matplotlib/seaborn here to avoid requiring them for basic usage
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get cosine similarities
    true_features = feature_dict.feature_vectors.detach()
    dec_cos_sims = cosine_similarities(sae.W_dec.detach(), true_features)  # type: ignore[attr-defined]
    enc_cos_sims = cosine_similarities(sae.W_enc.T.detach(), true_features)  # type: ignore[attr-defined]

    # Round to reduce numerical noise
    dec_cos_sims = torch.round(dec_cos_sims * 100) / 100
    enc_cos_sims = torch.round(enc_cos_sims * 100) / 100

    # Apply feature reordering if requested
    if reorder_features is not False:
        if isinstance(reorder_features, bool):
            sorted_indices = find_best_feature_ordering(
                sae.W_dec.detach(),
                true_features,  # type: ignore[attr-defined]
            )
        else:
            sorted_indices = reorder_features
        dec_cos_sims = dec_cos_sims[sorted_indices]
        enc_cos_sims = enc_cos_sims[sorted_indices]

    # Create figure
    if decoder_only:
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0] / 2, figsize[1]))
        axes = [ax]
        data = [dec_cos_sims]
        subtitles = ["SAE decoder"]
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        data = [enc_cos_sims, dec_cos_sims]
        subtitles = ["SAE encoder", "SAE decoder"]

    # Plot heatmaps
    for ax, cos_sim_data, subtitle in zip(axes, data, subtitles):  # type: ignore[arg-type]
        sns.heatmap(
            cos_sim_data.cpu().numpy(),
            ax=ax,
            vmin=-1,
            vmax=1,
            cmap="RdBu",
            center=0,
            annot=show_values,
            fmt=".2f" if show_values else "",
            cbar_kws={"label": "cos sim", "ticks": [-1, 0, 1]},
        )
        ax.set_title(subtitle)
        ax.set_xlabel("True feature")
        ax.set_ylabel("SAE latent")
        ax.invert_yaxis()

    # Set main title
    if title is None:
        title = "Cosine similarity with true features"
    fig.suptitle(title)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_decoder_bias_similarity(
    sae: torch.nn.Module,
    feature_dict: FeatureDictionary,
    title: str | None = None,
    show_values: bool = False,
    figsize: tuple[float, float] = (10, 3),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot cosine similarity between SAE decoder bias and true features.

    Args:
        sae: The SAE to visualize. Must have b_dec attribute.
        feature_dict: The feature dictionary containing true features
        title: Plot title. If None, a default title is used.
        show_values: If True, shows numeric values on the heatmap
        figsize: Figure size as (width, height) in inches
        save_path: If provided, saves the figure to this path
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    true_features = feature_dict.feature_vectors.detach()
    b_dec = sae.b_dec.detach().unsqueeze(0)  # type: ignore[attr-defined]
    cos_sims = cosine_similarities(b_dec, true_features)

    _, ax = plt.subplots(1, 1, figsize=figsize)

    sns.heatmap(
        cos_sims.cpu().numpy(),
        ax=ax,
        vmin=-1,
        vmax=1,
        cmap="RdBu",
        center=0,
        annot=show_values,
        fmt=".2f" if show_values else "",
        cbar_kws={"label": "cos sim", "ticks": [-1, 0, 1]},
    )

    if title is None:
        title = "Decoder bias cosine similarity with true features"
    ax.set_title(title)
    ax.set_xlabel("True feature")
    ax.set_ylabel("b_dec")
    ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_latent_similarities(sae: torch.nn.Module, title: str | None = None) -> None:
    """
    Plot cosine similarities between SAE latents.

    Useful for checking if the SAE has learned diverse features
    or if there are redundant latents.

    Args:
        sae: The SAE to visualize. Must have W_dec attribute.
        title: Plot title. If None, a default title is used.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    latent_cos_sims = cosine_similarities(sae.W_dec.detach(), sae.W_dec.detach())  # type: ignore[attr-defined]

    _, ax = plt.subplots(1, 1, figsize=(8, 6))

    sns.heatmap(
        latent_cos_sims.cpu().numpy(),
        ax=ax,
        vmin=-1,
        vmax=1,
        cmap="RdBu",
        center=0,
        cbar_kws={"label": "cos sim", "ticks": [-1, 0, 1]},
    )

    if title is None:
        title = "SAE latent cosine similarities"
    ax.set_title(title)
    ax.set_xlabel("SAE latent")
    ax.set_ylabel("SAE latent")

    plt.tight_layout()
    plt.show()
