"""
Feature dictionary for generating synthetic activations.

A FeatureDictionary maps feature activations (sparse coefficients) to dense hidden activations
by multiplying with a learned or constructed feature embedding matrix.
"""

from typing import Callable

import torch
from torch import nn
from tqdm.auto import tqdm

FeatureDictionaryInitializer = Callable[["FeatureDictionary"], None]


def orthogonalize_embeddings(
    embeddings: torch.Tensor,
    num_steps: int = 200,
    lr: float = 0.01,
    show_progress: bool = False,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Orthogonalize embeddings using gradient descent with chunked computation.

    Uses chunked computation to avoid O(n²) memory usage when computing pairwise
    dot products. Memory usage is O(chunk_size × n) instead of O(n²).

    Args:
        embeddings: Tensor of shape [num_vectors, hidden_dim]
        num_steps: Number of optimization steps
        lr: Learning rate for Adam optimizer
        show_progress: Whether to show progress bar
        chunk_size: Number of vectors to process at once. Smaller values use less
            memory but may be slower.

    Returns:
        Orthogonalized embeddings of the same shape, normalized to unit length.
    """
    num_vectors = embeddings.shape[0]
    # Create a detached copy and normalize, then enable gradients
    embeddings = embeddings.detach().clone()
    embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
    embeddings = embeddings.requires_grad_(True)

    optimizer = torch.optim.Adam([embeddings], lr=lr)  # type: ignore[list-item]

    num_chunks = (num_vectors + chunk_size - 1) // chunk_size
    total_iterations = num_steps * num_chunks

    pbar = tqdm(
        total=total_iterations,
        desc="Orthogonalizing vectors",
        disable=not show_progress,
    )
    step_loss: float | None = None
    for step in range(num_steps):
        optimizer.zero_grad()

        # Track total loss for display (detached, no gradient)
        total_loss = 0.0

        for i in range(0, num_vectors, chunk_size):
            end_i = min(i + chunk_size, num_vectors)
            chunk = embeddings[i:end_i]
            chunk_dots = chunk @ embeddings.T  # [chunk_size, num_vectors]

            # Create mask to zero out diagonal elements for this chunk
            # Diagonal of full matrix: position (i+k, i+k) → in chunk_dots: (k, i+k)
            chunk_len = end_i - i
            row_indices = torch.arange(chunk_len, device=embeddings.device)
            col_indices = i + row_indices  # column indices in full matrix

            # Boolean mask: True for off-diagonal elements we want to include
            off_diag_mask = torch.ones_like(chunk_dots, dtype=torch.bool)
            off_diag_mask[row_indices, col_indices] = False

            off_diag_loss = chunk_dots[off_diag_mask].pow(2).sum()

            # Diagonal loss: keep self-dot-products at 1
            diag_vals = chunk_dots[row_indices, col_indices]
            diag_loss = (diag_vals - 1).pow(2).sum()

            # Compute chunk loss and backward immediately to free memory
            chunk_loss = off_diag_loss + num_vectors * diag_loss
            chunk_loss.backward()
            total_loss += chunk_loss.item()

            pbar.update(1)
            desc = f"Orthogonalizing vectors: step {step + 1}/{num_steps}"
            if step_loss is not None:
                desc += f", loss {step_loss:.2e}"
            pbar.set_description(desc)

        optimizer.step()
        step_loss = total_loss

    pbar.close()

    with torch.no_grad():
        embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True).clamp(
            min=1e-8
        )
    return embeddings.detach().clone()


def orthogonal_initializer(
    num_steps: int = 200,
    lr: float = 0.01,
    show_progress: bool = False,
    chunk_size: int = 1024,
) -> FeatureDictionaryInitializer:
    def initializer(feature_dict: "FeatureDictionary") -> None:
        feature_dict.feature_vectors.data = orthogonalize_embeddings(
            feature_dict.feature_vectors,
            num_steps=num_steps,
            lr=lr,
            show_progress=show_progress,
            chunk_size=chunk_size,
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
        bias: bool | float = False,
        initializer: FeatureDictionaryInitializer | None = orthogonal_initializer(),
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ):
        """
        Create a new FeatureDictionary.

        Args:
            num_features: Number of features in the dictionary
            hidden_dim: Dimensionality of the hidden space
            bias: If False, no bias. If True, bias with norm 1.0. If float, bias with that norm.
            initializer: Initializer function to use. If None, the embeddings are initialized to random unit vectors. By default will orthogonalize embeddings.
            device: Device to use for the feature dictionary.
            seed: Random seed for reproducible initialization.
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Initialize feature vectors as unit vectors
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        embeddings = torch.randn(
            num_features, hidden_dim, device=device, generator=generator
        )
        embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True).clamp(
            min=1e-8
        )
        self.feature_vectors = nn.Parameter(embeddings)

        # Initialize bias
        if bias:
            bias_norm = 1.0 if bias is True else float(bias)
            bias_vec = torch.randn(hidden_dim, device=device, generator=generator)
            bias_vec = bias_vec / bias_vec.norm().clamp(min=1e-8) * bias_norm
            self.bias = nn.Parameter(bias_vec)
        else:
            self.bias = nn.Parameter(
                torch.zeros(hidden_dim, device=device), requires_grad=False
            )

        if initializer is not None:
            initializer(self)

    def forward(self, feature_activations: torch.Tensor) -> torch.Tensor:
        """
        Convert feature activations to hidden activations.

        Args:
            feature_activations: Tensor of shape [batch, num_features] containing
                sparse feature activation values. Can be dense or sparse COO.

        Returns:
            Tensor of shape [batch, hidden_dim] containing dense hidden activations
        """
        if feature_activations.is_sparse:
            # autocast is disabled here because sparse matmul is not supported with bfloat16
            with torch.autocast(
                device_type=feature_activations.device.type, enabled=False
            ):
                return (
                    torch.sparse.mm(feature_activations, self.feature_vectors)
                    + self.bias
                )
        return feature_activations @ self.feature_vectors + self.bias
