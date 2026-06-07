from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from typing_extensions import override

from sae_lens.saes.jumprelu_sae import JumpReLUSAEConfig
from sae_lens.saes.sae import SAEConfig, TrainStepInput, TrainStepOutput
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig


class BatchTopK(nn.Module):
    """BatchTopK activation function"""

    def __init__(
        self,
        k: float,
    ):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts = x.relu()
        flat_acts = acts.flatten()
        # Calculate total number of samples across all non-feature dimensions
        num_samples = acts.shape[:-1].numel()
        acts_topk_flat = torch.topk(flat_acts, int(self.k * num_samples), dim=-1)
        return (
            torch.zeros_like(flat_acts)
            .scatter(-1, acts_topk_flat.indices, acts_topk_flat.values)
            .reshape(acts.shape)
        )


@dataclass
class BatchTopKTrainingSAEConfig(TopKTrainingSAEConfig):
    """
    Configuration class for training a BatchTopKTrainingSAE.

    BatchTopK SAEs maintain k active features on average across the entire batch,
    rather than enforcing k features per sample like standard TopK SAEs. During training,
    the SAE learns a global threshold that is updated based on the minimum positive
    activation value. After training, BatchTopK SAEs are saved as JumpReLU SAEs.

    Args:
        k (float): Average number of features to keep active across the batch. Unlike
            standard TopK SAEs where k is an integer per sample, this is a float
            representing the average number of active features across all samples in
            the batch. Defaults to 100.
        topk_threshold_lr (float): Learning rate for updating the global topk threshold.
            The threshold is updated using an exponential moving average of the minimum
            positive activation value. Defaults to 0.01.
        aux_loss_coefficient (float): Coefficient for the auxiliary loss that encourages
            dead neurons to learn useful features. Inherited from TopKTrainingSAEConfig.
            Defaults to 1.0.
        rescale_acts_by_decoder_norm (bool): Treat the decoder as if it was already normalized.
            Inherited from TopKTrainingSAEConfig. Defaults to True.
        decoder_init_norm (float | None): Norm to initialize decoder weights to.
            Inherited from TrainingSAEConfig. Defaults to 0.1.
        d_in (int): Input dimension (dimensionality of the activations being encoded).
            Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
            Defaults to "float32".
        device (str): Device to place the SAE on. Inherited from SAEConfig.
            Defaults to "cpu".
    """

    k: float = 100  # type: ignore[assignment]
    topk_threshold_lr: float = 0.01

    @override
    @classmethod
    def architecture(cls) -> str:
        return "batchtopk"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        return JumpReLUSAEConfig


class BatchTopKTrainingSAE(TopKTrainingSAE):
    """
    Global Batch TopK Training SAE

    This SAE will maintain the k on average across the batch, rather than enforcing the k per-sample as in standard TopK.

    BatchTopK SAEs are saved as JumpReLU SAEs after training.
    """

    topk_threshold: torch.Tensor
    cfg: BatchTopKTrainingSAEConfig  # type: ignore[assignment]

    def __init__(self, cfg: BatchTopKTrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

        self.register_buffer(
            "topk_threshold",
            # use double precision as otherwise we can run into numerical issues
            torch.tensor(0.0, dtype=torch.double, device=self.W_dec.device),
        )

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return BatchTopK(self.cfg.k)

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)
        self.update_topk_threshold(output.feature_acts)
        output.metrics["topk_threshold"] = self.topk_threshold
        return output

    @torch.no_grad()
    def update_topk_threshold(self, acts_topk: torch.Tensor) -> None:
        positive_mask = acts_topk > 0
        lr = self.cfg.topk_threshold_lr
        # autocast can cause numerical issues with the threshold update
        with torch.autocast(self.topk_threshold.device.type, enabled=False):
            if positive_mask.any():
                min_positive = (
                    acts_topk[positive_mask].min().to(self.topk_threshold.dtype)
                )
                self.topk_threshold = (1 - lr) * self.topk_threshold + lr * min_positive

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        super().process_state_dict_for_saving_inference(state_dict)
        # turn the topk threshold into jumprelu threshold
        topk_threshold = state_dict.pop("topk_threshold").item()
        state_dict["threshold"] = torch.ones_like(self.b_enc) * topk_threshold
