from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    SAEMetadata,
)
from sae_lens.util import filter_valid_dataclass_fields

# pyright: reportIncompatibleVariableOverride=false


@dataclass
class TranscoderConfig(SAEConfig):
    # Output dimension fields
    d_out: int = 768
    # hook_name_out: str = ""
    # hook_layer_out: int = 0
    # hook_head_index_out: int | None = None

    @classmethod
    def architecture(cls) -> str:
        """Return the architecture name for this config."""
        return "transcoder"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TranscoderConfig":
        """Create a TranscoderConfig from a dictionary."""
        # Filter to only include valid dataclass fields
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)

        # Create the config instance
        res = cls(**filtered_config_dict)

        # Handle metadata if present
        if "metadata" in config_dict:
            res.metadata = SAEMetadata(**config_dict["metadata"])

        return res

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, including parent fields."""
        # Get the base dictionary from parent
        res = super().to_dict()

        # Add transcoder-specific fields
        res.update({"d_out": self.d_out})

        return res

    def __post_init__(self):
        if self.apply_b_dec_to_input:
            raise ValueError("apply_b_dec_to_input is not supported for transcoders")
        return super().__post_init__()


class Transcoder(SAE[TranscoderConfig]):
    """
    A transcoder maps activations from one hook point to another with
    potentially different dimensions. It extends the standard SAE but with a
    decoder that maps to a different output dimension.
    """

    cfg: TranscoderConfig
    W_enc: nn.Parameter
    b_enc: nn.Parameter
    W_dec: nn.Parameter
    b_dec: nn.Parameter

    def __init__(self, cfg: TranscoderConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def initialize_weights(self):
        """Initialize transcoder weights with proper dimensions."""
        # Initialize b_dec with output dimension
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_out, dtype=self.dtype, device=self.device)
        )

        # Initialize W_dec with shape [d_sae, d_out]
        w_dec_data = torch.empty(
            self.cfg.d_sae, self.cfg.d_out, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_dec_data)
        self.W_dec = nn.Parameter(w_dec_data)

        # Initialize W_enc with shape [d_in, d_sae]
        w_enc_data = torch.empty(
            self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_enc_data)
        self.W_enc = nn.Parameter(w_enc_data)

        # Initialize b_enc
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

    def process_sae_in(self, sae_in: torch.Tensor) -> torch.Tensor:
        """
        Process input without applying decoder bias (which has wrong dimension
        for transcoder).

        Overrides the parent method to skip the bias subtraction since b_dec
        has dimension d_out which doesn't match the input dimension d_in.
        """
        # Don't apply b_dec since it has different dimension
        # Just handle dtype conversion and hooks
        sae_in = sae_in.to(self.dtype)
        sae_in = self.hook_sae_input(sae_in)
        return self.run_time_activation_norm_fn_in(sae_in)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor into the feature space.
        """
        # Preprocess the SAE input (casting type, applying hooks, normalization)
        sae_in = self.process_sae_in(x)
        # Compute the pre-activation values
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        # Apply the activation function (e.g., ReLU)
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode to output dimension."""
        # W_dec has shape [d_sae, d_out], feature_acts has shape
        # [batch, d_sae]
        sae_out = feature_acts @ self.W_dec + self.b_dec
        # Apply hooks
        # Note: We don't apply run_time_activation_norm_fn_out since the
        # output dimension is different from the input dimension
        return self.hook_sae_recons(sae_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transcoder.

        Args:
            x: Input activations from the input hook point [batch, d_in]

        Returns:
            sae_out: Reconstructed activations for the output hook point
            [batch, d_out]
        """
        feature_acts = self.encode(x)
        return self.decode(feature_acts)

    def forward_with_activations(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both output and feature activations.

        Args:
            x: Input activations from the input hook point [batch, d_in]

        Returns:
            sae_out: Reconstructed activations for the output hook point
            [batch, d_out]
            feature_acts: Hidden activations [batch, d_sae]
        """
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)
        return sae_out, feature_acts

    @property
    def d_out(self) -> int:
        """Output dimension of the transcoder."""
        return self.cfg.d_out

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Transcoder":
        cfg = TranscoderConfig.from_dict(config_dict)
        return cls(cfg)


@dataclass
class SkipTranscoderConfig(TranscoderConfig):
    @classmethod
    def architecture(cls) -> str:
        """Return the architecture name for this config."""
        return "skip_transcoder"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SkipTranscoderConfig":
        """Create a SkipTranscoderConfig from a dictionary."""
        # Filter to only include valid dataclass fields
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)

        # Create the config instance
        res = cls(**filtered_config_dict)

        # Handle metadata if present
        if "metadata" in config_dict:
            res.metadata = SAEMetadata(**config_dict["metadata"])

        return res


class SkipTranscoder(Transcoder):
    """
    A transcoder with a learnable skip connection.

    Implements: f(x) = W_dec @ relu(W_enc @ x + b_enc) + W_skip @ x + b_dec
    where W_skip is initialized to zeros.
    """

    cfg: SkipTranscoderConfig  # type: ignore[assignment]
    W_skip: nn.Parameter

    def __init__(self, cfg: SkipTranscoderConfig):
        super().__init__(cfg)
        self.cfg = cfg

        # Initialize skip connection matrix
        # Shape: [d_out, d_in] to map from input to output dimension
        self.W_skip = nn.Parameter(torch.zeros(self.cfg.d_out, self.cfg.d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for skip transcoder.

        Args:
            x: Input activations from the input hook point [batch, d_in]

        Returns:
            sae_out: Reconstructed activations for the output hook point
            [batch, d_out]
        """
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        # Add skip connection: W_skip @ x
        # x has shape [batch, d_in], W_skip has shape [d_out, d_in]
        skip_out = x @ self.W_skip.T.to(x.device)
        return sae_out + skip_out

    def forward_with_activations(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both output and feature activations.

        Args:
            x: Input activations from the input hook point [batch, d_in]

        Returns:
            sae_out: Reconstructed activations for the output hook point
            [batch, d_out]
            feature_acts: Hidden activations [batch, d_sae]
        """
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        # Add skip connection: W_skip @ x
        # x has shape [batch, d_in], W_skip has shape [d_out, d_in]
        skip_out = x @ self.W_skip.T.to(x.device)
        sae_out = sae_out + skip_out

        return sae_out, feature_acts

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SkipTranscoder":
        cfg = SkipTranscoderConfig.from_dict(config_dict)
        return cls(cfg)


# JumpReLU Transcoder Classes
@dataclass
class JumpReLUTranscoderConfig(TranscoderConfig):
    """Configuration for JumpReLU transcoder."""

    @classmethod
    def architecture(cls) -> str:
        """Return the architecture name for this config."""
        return "jumprelu_transcoder"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "JumpReLUTranscoderConfig":
        """Create a JumpReLUTranscoderConfig from a dictionary."""
        # Filter to only include valid dataclass fields
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)

        # Create the config instance
        res = cls(**filtered_config_dict)

        # Handle metadata if present
        if "metadata" in config_dict:
            res.metadata = SAEMetadata(**config_dict["metadata"])

        return res


class JumpReLUTranscoder(Transcoder):
    """
    A transcoder with JumpReLU activation function.

    JumpReLU applies a threshold to activations: if pre-activation <=
    threshold, the unit is zeroed out; otherwise, it follows the base
    activation function.
    """

    cfg: JumpReLUTranscoderConfig  # type: ignore[assignment]
    threshold: nn.Parameter

    def __init__(self, cfg: JumpReLUTranscoderConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def initialize_weights(self):
        """Initialize transcoder weights including threshold parameter."""
        super().initialize_weights()

        # Initialize threshold parameter for JumpReLU
        self.threshold = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode using JumpReLU activation.

        Applies base activation function (ReLU) then masks based on threshold.
        """
        # Preprocess the SAE input
        sae_in = self.process_sae_in(x)

        # Compute pre-activation values
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        # Apply base activation function (ReLU)
        feature_acts = self.activation_fn(hidden_pre)

        # Apply JumpReLU threshold
        # During training, use detached threshold to prevent gradient flow
        threshold = self.threshold.detach() if self.training else self.threshold
        jump_relu_mask = (hidden_pre > threshold).to(self.dtype)

        # Apply mask and hook
        return self.hook_sae_acts_post(feature_acts * jump_relu_mask)

    def fold_W_dec_norm(self) -> None:
        """
        Fold the decoder weight norm into the threshold parameter.

        This is important for JumpReLU as the threshold needs to be scaled
        along with the decoder weights.
        """
        # Get the decoder weight norms before normalizing
        with torch.no_grad():
            W_dec_norms = self.W_dec.norm(dim=1)

        # Fold the decoder norms as in the parent class
        super().fold_W_dec_norm()

        # Scale the threshold by the decoder weight norms
        with torch.no_grad():
            self.threshold.data = self.threshold.data * W_dec_norms

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "JumpReLUTranscoder":
        cfg = JumpReLUTranscoderConfig.from_dict(config_dict)
        return cls(cfg)


@dataclass
class JumpReLUSkipTranscoderConfig(JumpReLUTranscoderConfig):
    """Configuration for JumpReLU transcoder."""

    @classmethod
    def architecture(cls) -> str:
        """Return the architecture name for this config."""
        return "jumprelu_skip_transcoder"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "JumpReLUSkipTranscoderConfig":
        """Create a JumpReLUSkipTranscoderConfig from a dictionary."""
        # Filter to only include valid dataclass fields
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)

        # Create the config instance
        res = cls(**filtered_config_dict)

        # Handle metadata if present
        if "metadata" in config_dict:
            res.metadata = SAEMetadata(**config_dict["metadata"])

        return res


class JumpReLUSkipTranscoder(JumpReLUTranscoder, SkipTranscoder):
    """
    A transcoder with a learnable skip connection and JumpReLU activation function.
    """

    cfg: JumpReLUSkipTranscoderConfig  # type: ignore[assignment]

    def __init__(self, cfg: JumpReLUSkipTranscoderConfig):
        super().__init__(cfg)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "JumpReLUSkipTranscoder":
        cfg = JumpReLUSkipTranscoderConfig.from_dict(config_dict)
        return cls(cfg)
