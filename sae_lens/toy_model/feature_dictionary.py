"""
Feature dictionary for generating synthetic activations.

A FeatureDictionary maps feature activations (sparse coefficients) to dense hidden activations
by multiplying with a learned or constructed feature embedding matrix.
"""

import torch
from torch import nn
from tqdm import tqdm


def orthogonalize_vectors(
    num_vectors: int,
    vector_dim: int,
    target_cos_sim: float = 0,
    num_steps: int = 1000,
    lr: float = 0.01,
) -> torch.Tensor:
    """
    Create a set of unit vectors with controlled pairwise cosine similarity.

    Uses gradient descent to find vectors that have approximately the target
    cosine similarity with each other.

    Args:
        num_vectors: Number of vectors to create
        vector_dim: Dimensionality of each vector
        target_cos_sim: Target pairwise cosine similarity (0 = orthogonal)
        num_steps: Number of optimization steps
        lr: Learning rate for Adam optimizer

    Returns:
        Tensor of shape [num_vectors, vector_dim] with unit-norm vectors
    """
    embeddings = torch.randn(num_vectors, vector_dim)
    embeddings /= embeddings.norm(p=2, dim=1, keepdim=True)
    embeddings.requires_grad_(True)

    optimizer = torch.optim.Adam([embeddings], lr=lr)  # type: ignore[list-item]

    pbar = tqdm(range(num_steps), desc="Orthogonalizing vectors")
    for _ in pbar:
        optimizer.zero_grad()

        dot_products = embeddings @ embeddings.T
        diff = dot_products - target_cos_sim
        diff.fill_diagonal_(0)
        loss = diff.pow(2).sum()
        loss += num_vectors * (dot_products.diag() - 1).pow(2).sum()

        loss.backward()
        optimizer.step()
        pbar.set_description(f"loss: {loss.item():.3f}")

    embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
    embeddings = embeddings.detach().clone()
    embeddings.requires_grad_(False)
    return embeddings


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
        target_cos_sim: float = 0,
        bias: bool = False,
        ortho_lr: float = 0.01,
        ortho_num_steps: int = 1000,
    ):
        """
        Create a new FeatureDictionary.

        Args:
            num_features: Number of features in the dictionary
            hidden_dim: Dimensionality of the hidden space
            target_cos_sim: Target pairwise cosine similarity between features.
                0 means orthogonal features (easiest for SAEs to recover).
                Higher values create more correlated features (harder).
            bias: Whether to include a bias term in the embedding
            ortho_lr: Learning rate for the orthogonalization procedure
            ortho_num_steps: Number of steps for the orthogonalization procedure
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.embed = nn.Linear(num_features, hidden_dim, bias=bias)
        embeddings = orthogonalize_vectors(
            num_features,
            hidden_dim,
            target_cos_sim=target_cos_sim,
            lr=ortho_lr,
            num_steps=ortho_num_steps,
        )
        self.embed.weight.data = embeddings.T

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
