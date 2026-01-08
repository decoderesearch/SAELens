"""
Plotting utilities for visualizing SAE training on synthetic data.

This module provides functions for:

- Plotting cosine similarities between SAE features and true features
- Automatically reordering features for better visualization
- Creating comparison plots between encoder and decoder
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from sae_lens.saes import SAE
from sae_lens.synthetic.feature_dictionary import FeatureDictionary
from sae_lens.util import cosine_similarities


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
    sae: SAE[Any],
    feature_dict: FeatureDictionary,
    title: str | None = None,
    reorder_features: bool | torch.Tensor = False,
    decoder_only: bool = False,
    show_values: bool = False,
    height: int = 400,
    width: int = 800,
    save_path: str | Path | None = None,
    show_plot: bool = True,
    dtick: int | None = 1,
    scale: float = 1.0,
):
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
        height: Height of the figure in pixels
        width: Width of the figure in pixels
        save_path: If provided, saves the figure to this path
        show_plot: If True, displays the plot
        dtick: Tick spacing for axes
        scale: Scale factor for image resolution when saving
    """
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

    hovertemplate = "True feature: %{x}<br>SAE Latent: %{y}<br>Cosine Similarity: %{z:.3f}<extra></extra>"

    if decoder_only:
        fig = make_subplots(rows=1, cols=1)

        decoder_args: dict[str, Any] = {
            "z": dec_cos_sims.cpu().numpy(),
            "zmin": -1,
            "zmax": 1,
            "colorscale": "RdBu",
            "colorbar": dict(title="cos sim", x=1.0, dtick=1, tickvals=[-1, 0, 1]),
            "hovertemplate": hovertemplate,
        }
        if show_values:
            decoder_args["texttemplate"] = "%{z:.2f}"
            decoder_args["textfont"] = {"size": 10}

        fig.add_trace(go.Heatmap(**decoder_args), row=1, col=1)
        fig.update_xaxes(title_text="True feature", row=1, col=1, dtick=dtick)
        fig.update_yaxes(title_text="SAE Latent", row=1, col=1, dtick=dtick)
    else:
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("SAE encoder", "SAE decoder")
        )

        # Encoder heatmap
        encoder_args: dict[str, Any] = {
            "z": enc_cos_sims.cpu().numpy(),
            "zmin": -1,
            "zmax": 1,
            "colorscale": "RdBu",
            "showscale": False,
            "hovertemplate": hovertemplate,
        }
        if show_values:
            encoder_args["texttemplate"] = "%{z:.2f}"
            encoder_args["textfont"] = {"size": 10}

        fig.add_trace(go.Heatmap(**encoder_args), row=1, col=1)

        # Decoder heatmap
        decoder_args = {
            "z": dec_cos_sims.cpu().numpy(),
            "zmin": -1,
            "zmax": 1,
            "colorscale": "RdBu",
            "colorbar": dict(title="cos sim", x=1.0, dtick=1, tickvals=[-1, 0, 1]),
            "hovertemplate": hovertemplate,
        }
        if show_values:
            decoder_args["texttemplate"] = "%{z:.2f}"
            decoder_args["textfont"] = {"size": 10}

        fig.add_trace(go.Heatmap(**decoder_args), row=1, col=2)

        fig.update_xaxes(title_text="True feature", row=1, col=1, dtick=dtick)
        fig.update_xaxes(title_text="True feature", row=1, col=2, dtick=dtick)
        fig.update_yaxes(title_text="SAE Latent", row=1, col=1, dtick=dtick)
        fig.update_yaxes(title_text="SAE Latent", row=1, col=2, dtick=dtick)

    # Set main title
    if title is None:
        title = "Cosine similarity with true features"
    fig.update_layout(height=height, width=width, title_text=title)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(save_path, scale=scale)

    if show_plot:
        fig.show()
