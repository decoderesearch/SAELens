"""
Utilities for training SAEs on synthetic data.

This module provides helpers for:

- Generating training data from feature dictionaries
- Training SAEs on synthetic data
- Evaluating SAEs against known ground truth features
- Initializing SAEs to match feature dictionaries
"""

from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment

from sae_lens.synthetic.activation_generator import ActivationGenerator
from sae_lens.synthetic.feature_dictionary import FeatureDictionary


def mean_correlation_coefficient(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
) -> float:
    """
    Compute Mean Correlation Coefficient (MCC) between two sets of feature vectors.

    MCC measures how well learned features align with ground truth features by finding
    an optimal one-to-one matching using the Hungarian algorithm and computing the
    mean absolute cosine similarity of matched pairs.

    Reference: O'Neill et al. "Compute Optimal Inference and Provable Amortisation
    Gap in Sparse Autoencoders" (arXiv:2411.13117)

    Args:
        features_a: Feature vectors of shape [num_features_a, hidden_dim]
        features_b: Feature vectors of shape [num_features_b, hidden_dim]

    Returns:
        MCC score in range [0, 1], where 1 indicates perfect alignment
    """
    # Normalize to unit vectors
    a_norm = features_a / features_a.norm(dim=1, keepdim=True).clamp(min=1e-8)
    b_norm = features_b / features_b.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Compute absolute cosine similarity matrix
    cos_sim = torch.abs(a_norm @ b_norm.T)

    # Convert to cost matrix for Hungarian algorithm (which minimizes)
    cost_matrix = 1 - cos_sim.cpu().numpy()

    # Find optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Compute mean of matched similarities
    matched_similarities = cos_sim[row_ind, col_ind]
    return matched_similarities.mean().item()


@dataclass
class SyntheticDataEvalResult:
    """Results from evaluating an SAE on synthetic data."""

    true_l0: float
    """Average L0 of the true feature activations"""

    sae_l0: float
    """Average L0 of the SAE's latent activations"""

    dead_latents: int
    """Number of SAE latents that never fired"""

    shrinkage: float
    """Average ratio of SAE output norm to input norm (1.0 = no shrinkage)"""

    mcc: float
    """Mean Correlation Coefficient between SAE decoder and ground truth features"""


@torch.no_grad()
def eval_sae_on_synthetic_data(
    sae: torch.nn.Module,
    feature_dict: FeatureDictionary,
    activations_generator: ActivationGenerator,
    num_samples: int = 100_000,
) -> SyntheticDataEvalResult:
    """
    Evaluate an SAE on synthetic data with known ground truth.

    Args:
        sae: The SAE to evaluate. Must have encode() and decode() methods.
        feature_dict: The feature dictionary used to generate activations
        activations_generator: Generator that produces feature activations
        num_samples: Number of samples to use for evaluation

    Returns:
        SyntheticDataEvalResult containing evaluation metrics
    """
    sae.eval()

    # Generate samples
    feature_acts = activations_generator.sample(num_samples)
    true_l0 = (feature_acts > 0).float().sum(dim=-1).mean().item()
    hidden_acts = feature_dict(feature_acts)

    # Filter out entries where no features fire
    non_zero_mask = hidden_acts.norm(dim=-1) > 0
    hidden_acts_filtered = hidden_acts[non_zero_mask]

    # Get SAE reconstructions
    sae_latents = sae.encode(hidden_acts_filtered)  # type: ignore[attr-defined]
    sae_output = sae.decode(sae_latents)  # type: ignore[attr-defined]

    sae_l0 = (sae_latents > 0).float().sum(dim=-1).mean().item()
    dead_latents = int(
        ((sae_latents == 0).sum(dim=0) == sae_latents.shape[0]).sum().item()
    )
    if hidden_acts_filtered.shape[0] == 0:
        shrinkage = 0.0
    else:
        shrinkage = (
            (
                sae_output.norm(dim=-1)
                / hidden_acts_filtered.norm(dim=-1).clamp(min=1e-8)
            )
            .mean()
            .item()
        )

    # Compute MCC between SAE decoder and ground truth features
    sae_decoder: torch.Tensor = sae.W_dec  # type: ignore[attr-defined]
    gt_features = feature_dict.feature_vectors
    mcc = mean_correlation_coefficient(sae_decoder, gt_features)

    return SyntheticDataEvalResult(
        true_l0=true_l0,
        sae_l0=sae_l0,
        dead_latents=dead_latents,
        shrinkage=shrinkage,
        mcc=mcc,
    )
