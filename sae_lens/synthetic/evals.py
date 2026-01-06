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

from sae_lens.synthetic.activation_generator import ActivationGenerator
from sae_lens.synthetic.feature_dictionary import FeatureDictionary


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
        generate_features_fn: Function that generates feature activations
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
    shrinkage = (
        (sae_output.norm(dim=-1) / hidden_acts_filtered.norm(dim=-1)).mean().item()
    )

    return SyntheticDataEvalResult(
        true_l0=true_l0,
        sae_l0=sae_l0,
        dead_latents=dead_latents,
        shrinkage=shrinkage,
    )
