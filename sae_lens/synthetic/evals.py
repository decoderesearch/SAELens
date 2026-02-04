"""
Utilities for training SAEs on synthetic data.

This module provides helpers for:

- Generating training data from feature dictionaries
- Training SAEs on synthetic data
- Evaluating SAEs against known ground truth features
- Initializing SAEs to match feature dictionaries
"""

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment

from sae_lens.saes.sae import SAE
from sae_lens.synthetic.activation_generator import ActivationGenerator
from sae_lens.synthetic.feature_dictionary import FeatureDictionary
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.util import cosine_similarities


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
    cos_sim = cosine_similarities(features_a, features_b).abs()

    # Convert to cost matrix for Hungarian algorithm (which minimizes)
    cost_matrix = 1 - cos_sim.cpu().numpy()

    # Find optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Compute mean of matched similarities
    matched_similarities = cos_sim[row_ind, col_ind]
    return matched_similarities.mean().item()


def feature_uniqueness(
    sae_features: torch.Tensor,
    gt_features: torch.Tensor,
) -> float:
    """
    Compute uniqueness score measuring how many unique ground-truth features are tracked.

    For each SAE latent, finds the best-matching ground-truth feature. Returns the
    fraction of unique ground-truth features matched divided by number of SAE latents.

    A score of 1.0 means each SAE latent tracks a different ground-truth feature.
    A score < 1.0 indicates multiple SAE latents are converging to the same features.

    Args:
        sae_features: SAE decoder features of shape [num_sae_latents, hidden_dim]
        gt_features: Ground truth features of shape [num_gt_features, hidden_dim]

    Returns:
        Uniqueness score in range (0, 1], where 1 means all latents track unique features
    """
    if sae_features.shape[0] == 0:
        return 0.0

    cos_sim = cosine_similarities(sae_features, gt_features).abs()

    # For each SAE latent, find the best matching ground-truth feature
    best_matches = cos_sim.argmax(dim=1)

    # Count unique ground-truth features matched
    num_unique = len(best_matches.unique())

    return num_unique / sae_features.shape[0]


@dataclass
class ClassificationMetrics:
    """Classification metrics for SAE latents as feature detectors."""

    precision: float
    """Mean precision across SAE latents (TP / (TP + FP))"""

    recall: float
    """Mean recall across SAE latents (TP / (TP + FN))"""

    f1_score: float
    """Mean F1 score across SAE latents (harmonic mean of precision and recall)"""

    accuracy: float
    """Mean accuracy across SAE latents ((TP + TN) / total)"""


# =============================================================================
# Metric Calculators
# =============================================================================


class L0Calculator:
    """Calculator for L0 (sparsity) metrics over batches."""

    def __init__(self) -> None:
        self.total_l0 = 0.0
        self.num_samples = 0

    def add_batch(self, activations: torch.Tensor) -> None:
        """Add a batch of activations. Shape: (batch_size, num_features)."""
        self.total_l0 += (activations > 0).float().sum().item()
        self.num_samples += activations.shape[0]

    def compute(self) -> float:
        """Return mean L0 across all samples."""
        if self.num_samples == 0:
            return 0.0
        return self.total_l0 / self.num_samples


class DeadLatentsCalculator:
    """Calculator for counting dead (never-firing) latents over batches."""

    def __init__(self, num_latents: int) -> None:
        self.latent_ever_fired = torch.zeros(num_latents, dtype=torch.bool)

    def add_batch(self, sae_latents: torch.Tensor) -> None:
        """Add a batch of SAE latents. Shape: (batch_size, num_latents)."""
        self.latent_ever_fired |= (sae_latents > 0).any(dim=0).cpu()

    def compute(self) -> int:
        """Return count of latents that never fired."""
        return int((~self.latent_ever_fired).sum().item())


class ShrinkageCalculator:
    """Calculator for shrinkage (output/input norm ratio) over batches.

    Samples with near-zero input norm are skipped since shrinkage is undefined
    (dividing by near-zero norm would produce arbitrarily large or unstable values).

    Args:
        min_input_norm: Minimum input norm threshold. Samples with input norm below
            this value are excluded from the calculation. Default 1e-6 is chosen to
            be well above floating-point precision limits while catching effectively
            zero-norm inputs.
    """

    def __init__(self, min_input_norm: float = 1e-6) -> None:
        self.min_input_norm = min_input_norm
        self.total_shrinkage = 0.0
        self.num_samples = 0

    def add_batch(self, sae_output: torch.Tensor, hidden_acts: torch.Tensor) -> None:
        """Add a batch. Both shapes: (batch_size, hidden_dim)."""
        input_norms = hidden_acts.norm(dim=-1)
        valid_mask = input_norms > self.min_input_norm

        if valid_mask.any():
            output_norms = sae_output.norm(dim=-1)
            self.total_shrinkage += (
                (output_norms[valid_mask] / input_norms[valid_mask]).sum().item()
            )
            self.num_samples += valid_mask.sum().item()

    def compute(self) -> float:
        """Return mean shrinkage ratio across all valid samples."""
        if self.num_samples == 0:
            return 0.0
        return self.total_shrinkage / self.num_samples


class ExplainedVarianceCalculator:
    """Calculator for explained variance (R²) over batches.

    Computes: 1 - E[||x - x_hat||²] / Var(x)
    where Var(x) = E[||x||²] - ||E[x]||²
    """

    def __init__(self, hidden_dim: int, device: torch.device | str = "cpu") -> None:
        self.hidden_dim = hidden_dim
        self.device = device
        # Running sums
        self.sum_x: torch.Tensor = torch.zeros(hidden_dim, device=device)
        self.sum_squared_norm: float = 0.0
        self.sum_squared_residual: float = 0.0
        self.num_samples: int = 0

    def add_batch(self, sae_output: torch.Tensor, hidden_acts: torch.Tensor) -> None:
        """Add a batch. Both shapes: (batch_size, hidden_dim)."""
        batch_size = hidden_acts.shape[0]

        # Sum of input vectors (for computing E[x])
        self.sum_x += hidden_acts.sum(dim=0).to(self.device)

        # Sum of squared norms (for computing E[||x||²])
        self.sum_squared_norm += hidden_acts.pow(2).sum().item()

        # Sum of squared residuals (for computing E[||x - x_hat||²])
        residuals = hidden_acts - sae_output
        self.sum_squared_residual += residuals.pow(2).sum().item()

        self.num_samples += batch_size

    def compute(self) -> float:
        """Return explained variance (R²) across all samples."""
        if self.num_samples == 0:
            return 0.0

        # E[||x||²]
        mean_squared_norm = self.sum_squared_norm / self.num_samples

        # ||E[x]||²
        mean_x = self.sum_x / self.num_samples
        squared_norm_of_mean = mean_x.pow(2).sum().item()

        # Total variance = E[||x||²] - ||E[x]||²
        total_variance = mean_squared_norm - squared_norm_of_mean

        if total_variance < 1e-10:
            return 1.0 if self.sum_squared_residual < 1e-10 else 0.0

        # MSE = E[||x - x_hat||²]
        mse = self.sum_squared_residual / self.num_samples

        return 1.0 - mse / total_variance


class ClassificationMetricsCalculator:
    """
    Calculator for classification metrics over batches.

    For each SAE latent, evaluates how well it acts as a binary classifier for its
    best-matching ground-truth feature (by decoder cosine similarity).
    """

    def __init__(self, sae_decoder: torch.Tensor, gt_features: torch.Tensor):
        """
        Initialize the calculator.

        Args:
            sae_decoder: SAE decoder weights of shape (num_sae_latents, hidden_dim)
            gt_features: Ground truth feature vectors of shape (num_gt_features, hidden_dim)
        """
        self.num_sae_latents = sae_decoder.shape[0]

        if self.num_sae_latents > 0:
            # Precompute best-matching GT feature for each SAE latent
            cos_sim = cosine_similarities(sae_decoder, gt_features).abs()
            self.best_matches = cos_sim.argmax(dim=1)  # (num_sae_latents,)
        else:
            self.best_matches = torch.empty(0, dtype=torch.long)

        # Counts per latent
        self.tp = torch.zeros(self.num_sae_latents)
        self.fp = torch.zeros(self.num_sae_latents)
        self.fn = torch.zeros(self.num_sae_latents)
        self.tn = torch.zeros(self.num_sae_latents)

    def add_batch(
        self, sae_latents: torch.Tensor, gt_feature_acts: torch.Tensor
    ) -> None:
        """
        Add a batch of activations.

        Args:
            sae_latents: SAE latent activations of shape (batch_size, num_sae_latents)
            gt_feature_acts: Ground truth feature activations of shape (batch_size, num_gt_features)
        """
        if self.num_sae_latents == 0:
            return

        sae_fires = sae_latents > 0
        gt_fires = gt_feature_acts[:, self.best_matches] > 0

        self.tp += (sae_fires & gt_fires).float().sum(dim=0).cpu()
        self.fp += (sae_fires & ~gt_fires).float().sum(dim=0).cpu()
        self.fn += (~sae_fires & gt_fires).float().sum(dim=0).cpu()
        self.tn += (~sae_fires & ~gt_fires).float().sum(dim=0).cpu()

    def compute(self) -> ClassificationMetrics:
        """Compute final classification metrics from accumulated counts."""
        if self.num_sae_latents == 0:
            return ClassificationMetrics(
                precision=0.0, recall=0.0, f1_score=0.0, accuracy=0.0
            )

        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn

        # Precision: TP / (TP + FP), 0 when undefined
        precision_denom = tp + fp
        precision_per_latent = torch.where(
            precision_denom > 0,
            tp / precision_denom,
            torch.zeros_like(tp),
        )

        # Recall: TP / (TP + FN), 0 when undefined
        recall_denom = tp + fn
        recall_per_latent = torch.where(
            recall_denom > 0,
            tp / recall_denom,
            torch.zeros_like(tp),
        )

        # F1: 2 * precision * recall / (precision + recall), 0 when undefined
        f1_denom = precision_per_latent + recall_per_latent
        f1_per_latent = torch.where(
            f1_denom > 0,
            2 * precision_per_latent * recall_per_latent / f1_denom,
            torch.zeros_like(precision_per_latent),
        )

        # Accuracy: (TP + TN) / total
        total = tp + fp + fn + tn
        accuracy_per_latent = (tp + tn) / total.clamp(min=1e-8)

        return ClassificationMetrics(
            precision=precision_per_latent.mean().item(),
            recall=recall_per_latent.mean().item(),
            f1_score=f1_per_latent.mean().item(),
            accuracy=accuracy_per_latent.mean().item(),
        )


def compute_classification_metrics(
    sae_latents: torch.Tensor,
    gt_feature_acts: torch.Tensor,
    sae_decoder: torch.Tensor,
    gt_features: torch.Tensor,
) -> ClassificationMetrics:
    """
    Compute classification metrics for SAE latents as binary classifiers.

    For each SAE latent, finds its best-matching ground-truth feature (by decoder
    cosine similarity) and evaluates how well the latent firing predicts the
    ground-truth feature firing.

    Args:
        sae_latents: SAE latent activations of shape (num_samples, num_sae_latents)
        gt_feature_acts: Ground truth feature activations of shape (num_samples, num_gt_features)
        sae_decoder: SAE decoder weights of shape (num_sae_latents, hidden_dim)
        gt_features: Ground truth feature vectors of shape (num_gt_features, hidden_dim)

    Returns:
        ClassificationMetrics with mean precision, recall, F1, and accuracy
    """
    calculator = ClassificationMetricsCalculator(sae_decoder, gt_features)
    calculator.add_batch(sae_latents, gt_feature_acts)
    return calculator.compute()


@dataclass
class SyntheticDataEvalResult:
    """Results from evaluating an SAE on synthetic data."""

    true_l0: float
    """Average L0 of the synthetic model's feature activations"""

    sae_l0: float
    """Average L0 of the SAE's latent activations"""

    dead_latents: int
    """Number of SAE latents that never fired"""

    shrinkage: float
    """Average ratio of SAE output norm to input norm (1.0 = no shrinkage)"""

    explained_variance: float
    """Fraction of input variance explained by SAE reconstruction (R², 1.0 = perfect)"""

    mcc: float
    """Mean Correlation Coefficient between SAE decoder and ground truth features"""

    uniqueness: float
    """Fraction of SAE latents tracking unique ground-truth features (1.0 = all unique)"""

    classification: ClassificationMetrics
    """Classification metrics for SAE latents as feature detectors"""

    def to_log_dict(self, prefix: str = "") -> dict[str, float | int]:
        """Convert to a flat dictionary for logging (e.g., to wandb)."""
        result: dict[str, float | int] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, ClassificationMetrics):
                for sub_f in fields(value):
                    key = f"{prefix}classification/{sub_f.name}"
                    result[key] = getattr(value, sub_f.name)
            else:
                result[f"{prefix}{f.name}"] = value
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "classification": asdict(self.classification),
        }


@torch.no_grad()
def eval_sae_on_synthetic_data(
    sae: SAE[Any],
    feature_dict: FeatureDictionary,
    activations_generator: ActivationGenerator,
    num_samples: int = 100_000,
    batch_size: int | None = None,
    activation_scaler: ActivationScaler | None = None,
) -> SyntheticDataEvalResult:
    """
    Evaluate an SAE on synthetic data with known ground truth.

    Args:
        sae: The SAE to evaluate
        feature_dict: The feature dictionary used to generate activations
        activations_generator: Generator that produces feature activations
        num_samples: Number of samples to use for evaluation
        batch_size: Batch size for processing. If None, processes all samples at once.
        activation_scaler: Optional scaler to apply to activations before encoding.
            If None, no scaling is applied.

    Returns:
        SyntheticDataEvalResult containing evaluation metrics
    """
    sae.eval()

    if batch_size is None:
        batch_size = num_samples

    if activation_scaler is None:
        activation_scaler = ActivationScaler()

    # Get decoder and GT features for metrics that don't depend on samples
    sae_decoder = sae.W_dec
    gt_features = feature_dict.feature_vectors

    # Initialize calculators
    true_l0_calc = L0Calculator()
    sae_l0_calc = L0Calculator()
    dead_latents_calc = DeadLatentsCalculator(sae_decoder.shape[0])
    shrinkage_calc = ShrinkageCalculator()
    explained_variance_calc = ExplainedVarianceCalculator(
        sae_decoder.shape[-1], device=sae_decoder.device
    )
    classification_calc = ClassificationMetricsCalculator(sae_decoder, gt_features)

    # Process in batches
    num_processed = 0
    while num_processed < num_samples:
        current_batch_size = min(batch_size, num_samples - num_processed)

        # Generate samples for this batch
        feature_acts = activations_generator.sample(current_batch_size)
        true_l0_calc.add_batch(feature_acts)
        hidden_acts = feature_dict(feature_acts)

        # Scale activations before encoding (if scaler is configured)
        hidden_acts_scaled = activation_scaler.scale(hidden_acts)

        # Get SAE reconstructions
        sae_latents = sae.encode(hidden_acts_scaled)
        sae_output_scaled = sae.decode(sae_latents)

        # Unscale SAE output
        sae_output = activation_scaler.unscale(sae_output_scaled)

        # Update calculators
        sae_l0_calc.add_batch(sae_latents)
        dead_latents_calc.add_batch(sae_latents)
        shrinkage_calc.add_batch(sae_output, hidden_acts)
        explained_variance_calc.add_batch(sae_output, hidden_acts)
        classification_calc.add_batch(sae_latents, feature_acts)

        num_processed += current_batch_size

    # Compute MCC and uniqueness (only depend on decoder weights)
    mcc = mean_correlation_coefficient(sae_decoder, gt_features)
    uniqueness = feature_uniqueness(sae_decoder, gt_features)

    return SyntheticDataEvalResult(
        true_l0=true_l0_calc.compute(),
        sae_l0=sae_l0_calc.compute(),
        dead_latents=dead_latents_calc.compute(),
        shrinkage=shrinkage_calc.compute(),
        explained_variance=explained_variance_calc.compute(),
        mcc=mcc,
        uniqueness=uniqueness,
        classification=classification_calc.compute(),
    )
