from typing import Any

import torch
from torch import nn
from transformer_lens.hook_points import HookPoint

from sae_lens.saes.sae import SAE, _disable_hooks


class _SAEWrapper(nn.Module):  # pyright: ignore[reportUnusedClass]
    """Wrapper for SAE/Transcoder that handles error term and hook coordination.

    For SAEs (input_hook == output_hook), _captured_input stays None and we use
    the forward argument directly. For transcoders, _captured_input is set at
    the input hook via capture_input().

    Implementation Note:
        The SAE is stored in __dict__ directly rather than as a registered submodule.
        This is intentional: PyTorch's module registration would add a ".sae." prefix
        to all hook names in the cache (e.g., "blocks.0.hook_mlp_out.sae.hook_sae_input"
        instead of "blocks.0.hook_mlp_out.hook_sae_input"). By storing in __dict__ and
        copying hooks directly to the wrapper, we preserve the expected cache paths
        for backwards compatibility.

        The wrapper creates its own hook_sae_error and hook_sae_output HookPoints
        rather than copying from the SAE, because the wrapper handles error term
        computation and needs to call these hooks with the correct values.
    """

    def __init__(
        self,
        sae: SAE[Any],
        use_error_term: bool = False,
        exclude_special_tokens: bool | list[int] = False,
    ):
        super().__init__()
        # Store SAE in __dict__ to avoid registering as submodule. This keeps cache
        # paths clean by avoiding a ".sae." prefix on hook names. See class docstring.
        self.__dict__["_sae"] = sae
        # Copy SAE's hooks directly to wrapper so they appear at the right path
        # EXCEPT for hook_sae_error and hook_sae_output which the wrapper manages
        for name, hook in sae.hook_dict.items():
            if name not in ("hook_sae_error", "hook_sae_output"):
                setattr(self, name, hook)
        # Create new hooks for error and output that the wrapper manages
        self.hook_sae_error = HookPoint()
        self.hook_sae_output = HookPoint()
        self.use_error_term = use_error_term
        self.exclude_special_tokens = exclude_special_tokens
        self._captured_input: torch.Tensor | None = None
        self._token_mask: torch.Tensor | None = None

    @property
    def sae(self) -> SAE[Any]:
        return self.__dict__["_sae"]

    def capture_input(self, x: torch.Tensor) -> None:
        """Capture input at input hook (for transcoders).

        Note: We don't clone the tensor here - the input should not be modified
        in-place between capture and use, and avoiding clone preserves memory.
        """
        self._captured_input = x

    def set_token_mask(self, mask: torch.Tensor | None) -> None:
        """Set a boolean mask (batch, seq) where True = bypass SAE at that position."""
        self._token_mask = mask

    def forward(self, original_output: torch.Tensor) -> torch.Tensor:
        """Run SAE/transcoder at output hook location."""
        # For SAE: use original_output as input (same hook for input/output)
        # For transcoder: use captured input from earlier hook
        sae_input = (
            self._captured_input
            if self._captured_input is not None
            else original_output
        )

        mask = self._token_mask
        if mask is not None:
            for _ in range(original_output.dim() - mask.dim()):
                mask = mask.unsqueeze(-1)

        try:
            # Call encode and decode directly so we can control hook_sae_output
            feature_acts = self.sae.encode(sae_input)
            sae_out = self.sae.decode(feature_acts)

            if self.use_error_term:
                with torch.no_grad():
                    # Recompute without hooks to get true error term
                    # This ensures interventions on features don't get masked by error
                    with _disable_hooks(self.sae):
                        feature_acts_clean = self.sae.encode(sae_input)
                        sae_out_clean = self.sae.decode(feature_acts_clean)
                    sae_error = original_output - sae_out_clean
                    # Excluded positions bypass the SAE entirely, so their error is 0
                    if mask is not None:
                        sae_error = torch.where(
                            mask, torch.zeros_like(sae_error), sae_error
                        )
                    sae_error = self.hook_sae_error(sae_error)
                sae_out = sae_out + sae_error

            # Restore original activations at excluded token positions before firing
            # hook_sae_output, so the cache reflects what actually flows forward.
            if mask is not None:
                sae_out = torch.where(mask, original_output, sae_out)

            return self.hook_sae_output(sae_out)
        finally:
            self._captured_input = None


def get_deep_attr(obj: Any, path: str):
    """Helper function to get a nested attribute from a object.
    In practice used to access HookedTransformer HookPoints (eg model.blocks[0].attn.hook_z)

    Args:
        obj: Any object. In practice, this is a HookedTransformer (or subclass)
        path: str. The path to the attribute you want to access. (eg "blocks.0.attn.hook_z")

    returns:
        Any. The attribute at the end of the path
    """
    parts = path.split(".")
    # Navigate to the last component in the path
    for part in parts:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def set_deep_attr(obj: Any, path: str, value: Any):
    """Helper function to change the value of a nested attribute from a object.
    In practice used to swap HookedTransformer HookPoints (eg model.blocks[0].attn.hook_z) with HookedSAEs and vice versa

    Args:
        obj: Any object. In practice, this is a HookedTransformer (or subclass)
        path: str. The path to the attribute you want to access. (eg "blocks.0.attn.hook_z")
        value: Any. The value you want to set the attribute to (eg a HookedSAE object)
    """
    parts = path.split(".")
    # Navigate to the last component in the path
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    # Set the value on the final attribute
    setattr(obj, parts[-1], value)
