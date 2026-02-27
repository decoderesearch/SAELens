import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch
from tqdm.auto import tqdm

from sae_lens.training.types import DataProvider


@dataclass
class ActivationScaler:
    scaling_factor: float | None = None

    def scale(self, acts: torch.Tensor) -> torch.Tensor:
        return acts if self.scaling_factor is None else acts * self.scaling_factor

    def unscale(self, acts: torch.Tensor) -> torch.Tensor:
        return acts if self.scaling_factor is None else acts / self.scaling_factor

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

    def save(self, file_path: str):
        """save the state dict to a file in json format"""
        if not file_path.endswith(".json"):
            raise ValueError("file_path must end with .json")

        with open(file_path, "w") as f:
            json.dump({"scaling_factor": self.scaling_factor}, f)

    def load(self, file_path: str | Path):
        """load the state dict from a file in json format"""
        with open(file_path) as f:
            data = json.load(f)
            self.scaling_factor = data["scaling_factor"]
