"""
Feature dictionary for generating synthetic activations.

A FeatureDictionary maps feature activations (sparse coefficients) to dense hidden activations
by multiplying with a learned or constructed feature embedding matrix.
"""

from typing import Callable

import torch
from torch import nn
from tqdm import tqdm

FeatureDictionaryInitializer = Callable[["FeatureDictionary"], None]


def orthogonalize_embeddings(
    embeddings: torch.Tensor,
    target_cos_sim: float = 0,
    num_steps: int = 200,
    lr: float = 0.01,
    show_progress: bool = False,
) -> torch.Tensor:
    num_vectors = embeddings.shape[0]
    # Create a detached copy and normalize, then enable gradients
    embeddings = embeddings.detach().clone()
    embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
    embeddings = embeddings.requires_grad_(True)

    optimizer = torch.optim.Adam([embeddings], lr=lr)  # type: ignore[list-item]

    # Create a mask to zero out diagonal elements (avoid in-place operations)
    off_diagonal_mask = ~torch.eye(
        num_vectors, dtype=torch.bool, device=embeddings.device
    )

    pbar = tqdm(
        range(num_steps), desc="Orthogonalizing vectors", disable=not show_progress
    )
    for _ in pbar:
        optimizer.zero_grad()

        dot_products = embeddings @ embeddings.T
        diff = dot_products - target_cos_sim
        # Use masking instead of in-place fill_diagonal_
        off_diagonal_diff = diff * off_diagonal_mask.float()
        loss = off_diagonal_diff.pow(2).sum()
        loss = loss + num_vectors * (dot_products.diag() - 1).pow(2).sum()

        loss.backward()
        optimizer.step()
        pbar.set_description(f"loss: {loss.item():.3f}")

    with torch.no_grad():
        embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True).clamp(
            min=1e-8
        )
    return embeddings.detach().clone()


def orthogonal_initializer(
    num_steps: int = 200, lr: float = 0.01, show_progress: bool = False
) -> FeatureDictionaryInitializer:
    def initializer(feature_dict: "FeatureDictionary") -> None:
        feature_dict.feature_vectors.data = orthogonalize_embeddings(
            feature_dict.feature_vectors,
            num_steps=num_steps,
            lr=lr,
            show_progress=show_progress,
        )

    return initializer


class FeatureDictionary(nn.Module):
    """
    A feature dictionary that maps sparse feature activations to dense hidden activations.

    This class creates a set of feature vectors (the "dictionary") and provides methods
    to generate hidden activations from feature activations via a linear transformation.

    The feature vectors can be configured to have a specific pairwise cosine similarity,
    which is useful for controlling the difficulty of sparse recovery.

    Attributes:
        feature_vectors: Parameter of shape [num_features, hidden_dim] containing the
            feature embedding vectors
        bias: Parameter of shape [hidden_dim] containing the bias term (zeros if bias=False)
    """

    feature_vectors: nn.Parameter
    bias: nn.Parameter

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        bias: bool = False,
        initializer: FeatureDictionaryInitializer | None = orthogonal_initializer(),
    ):
        """
        Create a new FeatureDictionary.

        Args:
            num_features: Number of features in the dictionary
            hidden_dim: Dimensionality of the hidden space
            bias: Whether to include a bias term in the embedding
            initializer: Initializer function to use. If None, the embeddings are initialized to random unit vectors. By default will orthogonalize embeddings.
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Initialize feature vectors as unit vectors
        embeddings = torch.randn(num_features, hidden_dim)
        embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True).clamp(
            min=1e-8
        )
        self.feature_vectors = nn.Parameter(embeddings)

        # Initialize bias (zeros if not using bias, but still a parameter for consistent API)
        self.bias = nn.Parameter(torch.zeros(hidden_dim), requires_grad=bias)

        if initializer is not None:
            initializer(self)

    def forward(self, feature_activations: torch.Tensor) -> torch.Tensor:
        """
        Convert feature activations to hidden activations.

        Args:
            feature_activations: Tensor of shape [batch, num_features] containing
                sparse feature activation values

        Returns:
            Tensor of shape [batch, hidden_dim] containing dense hidden activations
        """
        return feature_activations @ self.feature_vectors + self.bias
