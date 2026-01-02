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


def orthogonalize_vectors(
    num_vectors: int,
    vector_dim: int,
    target_cos_sim: float = 0.0,
    num_steps: int = 200,
    lr: float = 0.01,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Create a set of (approximately) orthogonal unit vectors.

    Uses gradient descent to optimize random vectors toward the target pairwise
    cosine similarity while maintaining unit norms.

    Args:
        num_vectors: Number of vectors to create
        vector_dim: Dimensionality of each vector
        target_cos_sim: Target pairwise cosine similarity (0 = orthogonal)
        num_steps: Number of optimization steps
        lr: Learning rate for optimization
        show_progress: Whether to show a progress bar

    Returns:
        Tensor of shape [num_vectors, vector_dim] with unit-norm vectors

    Example:
        >>> vectors = orthogonalize_vectors(num_vectors=10, vector_dim=10)
        >>> norms = vectors.norm(dim=1)
        >>> assert torch.allclose(norms, torch.ones(10), atol=1e-5)
    """
    embeddings = torch.randn(num_vectors, vector_dim)
    return orthogonalize_embeddings(
        embeddings,
        target_cos_sim=target_cos_sim,
        num_steps=num_steps,
        lr=lr,
        show_progress=show_progress,
    )


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
    embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
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
        embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
    return embeddings.detach().clone()


def orthogonal_initializer(
    num_steps: int = 200, lr: float = 0.01, show_progress: bool = False
) -> FeatureDictionaryInitializer:
    def initializer(feature_dict: "FeatureDictionary") -> None:
        feature_dict.embed.weight.data = (
            orthogonalize_embeddings(
                feature_dict.embed.weight.T,
                num_steps=num_steps,
                lr=lr,
                show_progress=show_progress,
            )
            .T.clone()
            .contiguous()
        )

    return initializer


class FeatureDictionary(nn.Module):
    """
    A feature dictionary that maps sparse feature activations to dense hidden activations.

    This class creates a set of feature vectors (the "dictionary") and provides methods
    to generate hidden activations from feature activations via a linear transformation.

    The feature vectors can be configured to have a specific pairwise cosine similarity,
    which is useful for controlling the difficulty of sparse recovery.

    Example:
        >>> # Create a dictionary with 100 features in 64-dimensional space
        >>> feature_dict = FeatureDictionary(num_features=100, hidden_dim=64)
        >>>
        >>> # Generate some sparse feature activations (e.g., from a feature generator)
        >>> feature_activations = torch.zeros(32, 100)  # batch of 32
        >>> feature_activations[0, [5, 10, 20]] = torch.tensor([1.2, 0.8, 1.5])
        >>>
        >>> # Convert to hidden activations
        >>> hidden_activations = feature_dict(feature_activations)
        >>> hidden_activations.shape  # torch.Size([32, 64])

    Attributes:
        embed: Linear layer mapping feature space to hidden space
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        bias: bool = False,
        initializer: FeatureDictionaryInitializer | None = None,
        # Backwards-compatible parameters
        ortho_num_steps: int | None = None,
        target_cos_sim: float = 0.0,
    ):
        """
        Create a new FeatureDictionary.

        Args:
            num_features: Number of features in the dictionary
            hidden_dim: Dimensionality of the hidden space
            bias: Whether to include a bias term in the embedding
            initializer: Initializer function to use. If None, the embeddings are initialized to random unit vectors.
            ortho_num_steps: (Deprecated) Number of optimization steps for orthogonalization.
                If provided, creates an orthogonal initializer automatically.
            target_cos_sim: (Deprecated) Target pairwise cosine similarity for orthogonalization.
                Only used if ortho_num_steps is provided.
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.embed = nn.Linear(num_features, hidden_dim, bias=bias)
        with torch.no_grad():
            embeddings = torch.randn_like(self.embed.weight.data)
            embeddings /= embeddings.norm(p=2, dim=0, keepdim=True)
        self.embed.weight.data = embeddings

        # Handle backwards-compatible ortho_num_steps parameter
        if ortho_num_steps is not None and initializer is None:
            # Apply orthogonalization with target_cos_sim
            # Note: orthogonalize_embeddings uses gradient descent internally,
            # so we can't wrap it in no_grad()
            orthogonalized = orthogonalize_embeddings(
                self.embed.weight.T,
                target_cos_sim=target_cos_sim,
                num_steps=ortho_num_steps,
            )
            with torch.no_grad():
                self.embed.weight.data = orthogonalized.T.clone().contiguous()
        elif initializer is not None:
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
        return self.embed(feature_activations)

    @property
    def feature_vectors(self) -> torch.Tensor:
        """
        Get the feature vectors (dictionary columns).

        Returns:
            Tensor of shape [num_features, hidden_dim] where each row is a feature vector
        """
        return self.embed.weight.T

    @property
    def bias_vector(self) -> torch.Tensor:
        """
        Get the bias term.

        Returns:
            Tensor of shape [hidden_dim]
        """
        return self.embed.bias or self.embed.weight.new_zeros(self.hidden_dim)
