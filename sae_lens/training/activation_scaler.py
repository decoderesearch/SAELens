import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

from sae_lens.constants import ACTIVATION_WHITENING_FILENAME
from sae_lens.training.types import DataProvider


@dataclass
class ActivationWhitening:
    """
    Affine whitening of activations, applied as z = (x - mean) @ matrix and
    inverted as x = z @ inverse_matrix + mean.
    """

    mean: torch.Tensor  # (d_in,)
    matrix: torch.Tensor  # (d_in, d_in)
    inverse_matrix: torch.Tensor  # (d_in, d_in)

    def to(self, device: torch.device) -> "ActivationWhitening":
        if self.matrix.device == device:
            return self
        return ActivationWhitening(
            mean=self.mean.to(device),
            matrix=self.matrix.to(device),
            inverse_matrix=self.inverse_matrix.to(device),
        )


@dataclass
class ActivationScaler:
    scaling_factor: float | None = None
    whitening: ActivationWhitening | None = None

    def scale(self, acts: torch.Tensor) -> torch.Tensor:
        if self.scaling_factor is not None:
            acts = acts * self.scaling_factor
        if self.whitening is not None:
            # Move once rather than per batch: the activation store and the SAE
            # may live on different devices.
            self.whitening = whitening = self.whitening.to(acts.device)
            acts = (acts - whitening.mean.to(acts.dtype)) @ whitening.matrix.to(
                acts.dtype
            )
        return acts

    def unscale(self, acts: torch.Tensor) -> torch.Tensor:
        if self.whitening is not None:
            self.whitening = whitening = self.whitening.to(acts.device)
            acts = acts @ whitening.inverse_matrix.to(acts.dtype) + whitening.mean.to(
                acts.dtype
            )
        if self.scaling_factor is not None:
            acts = acts / self.scaling_factor
        return acts

    def __call__(self, acts: torch.Tensor) -> torch.Tensor:
        return self.scale(acts)

    @torch.no_grad()
    def _calculate_mean_norm(
        self, data_provider: DataProvider, n_batches_for_norm_estimate: int = int(1e3)
    ) -> float:
        norms_per_batch: list[float] = []
        for _ in tqdm(
            range(n_batches_for_norm_estimate),
            desc="Estimating norm scaling factor",
            leave=False,
        ):
            acts = next(data_provider)
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        return mean(norms_per_batch)

    def estimate_scaling_factor(
        self,
        d_in: int,
        data_provider: DataProvider,
        n_batches_for_norm_estimate: int = int(1e3),
    ):
        mean_norm = self._calculate_mean_norm(
            data_provider, n_batches_for_norm_estimate
        )
        self.scaling_factor = (d_in**0.5) / mean_norm

    @torch.no_grad()
    def estimate_whitening(
        self,
        d_in: int,
        data_provider: DataProvider,
        n_batches_for_norm_estimate: int = int(1e3),
        eps: float = 1e-3,
    ) -> None:
        """
        Estimate PCA whitening statistics from activations, following
        https://arxiv.org/abs/2511.13981.

        Accumulates the activation mean and covariance C = E diag(lambda) E^T over
        ``n_batches_for_norm_estimate`` batches, then sets ``self.whitening`` with
        matrix = E diag(lambda + eps)^(-1/2) and inverse_matrix = diag(lambda + eps)^(1/2) E^T,
        so that whitened activations have zero mean and identity covariance. The
        statistics are computed and kept in float64 (and cast to the activation
        dtype when applied) so that folding them into the SAE weights is exact.

        Args:
            d_in: Dimension of the activations.
            data_provider: Iterator yielding activation batches of shape (batch_size, d_in).
            n_batches_for_norm_estimate: Number of batches to accumulate statistics over.
            eps: Stabilizing constant added to the covariance eigenvalues.
        """
        count = 0
        sum_x = torch.zeros(d_in, dtype=torch.float64)
        sum_outer = torch.zeros(d_in, d_in, dtype=torch.float64)
        for _ in tqdm(
            range(n_batches_for_norm_estimate),
            desc="Estimating whitening",
            leave=False,
        ):
            acts = next(data_provider).to(torch.float64).reshape(-1, d_in)
            sum_x, sum_outer = sum_x.to(acts.device), sum_outer.to(acts.device)
            count += acts.shape[0]
            sum_x += acts.sum(dim=0)
            sum_outer += acts.T @ acts

        acts_mean = sum_x / count
        cov = (sum_outer - count * torch.outer(acts_mean, acts_mean)) / (count - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        # eigh can return tiny negative eigenvalues for a (numerically) singular covariance
        eigenvalues = eigenvalues.clamp(min=0) + eps
        self.whitening = ActivationWhitening(
            mean=acts_mean,
            matrix=(eigenvectors * eigenvalues.rsqrt()).contiguous(),
            inverse_matrix=(
                eigenvalues.sqrt().unsqueeze(1) * eigenvectors.T
            ).contiguous(),
        )

    def save(self, file_path: str):
        """save the state dict to a file in json format"""
        if not file_path.endswith(".json"):
            raise ValueError("file_path must end with .json")

        with open(file_path, "w") as f:
            json.dump({"scaling_factor": self.scaling_factor}, f)
        if self.whitening is not None:
            save_file(
                {
                    "mean": self.whitening.mean.contiguous(),
                    "matrix": self.whitening.matrix.contiguous(),
                    "inverse_matrix": self.whitening.inverse_matrix.contiguous(),
                },
                _whitening_path(file_path),
            )

    def load(self, file_path: str | Path):
        """load the state dict from a file in json format"""
        with open(file_path) as f:
            data = json.load(f)
            self.scaling_factor = data["scaling_factor"]
        whitening_path = _whitening_path(file_path)
        if whitening_path.exists():
            tensors = load_file(whitening_path)
            self.whitening = ActivationWhitening(
                mean=tensors["mean"],
                matrix=tensors["matrix"],
                inverse_matrix=tensors["inverse_matrix"],
            )
        else:
            self.whitening = None


def _whitening_path(scaler_cfg_path: str | Path) -> Path:
    # The whitening matrices are saved next to the scaler json, since a
    # (d_in, d_in) matrix does not belong in a json file.
    return Path(scaler_cfg_path).with_name(ACTIVATION_WHITENING_FILENAME)
