from contextlib import contextmanager
from typing import Any, Callable

import torch
from torch import nn
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.hook_points import HookPoint  # Hooking utilities
from transformer_lens.HookedTransformer import HookedTransformer

from sae_lens import logger
from sae_lens.saes.sae import SAE

SingleLoss = torch.Tensor  # Type alias for a single element tensor
LossPerToken = torch.Tensor
Loss = SingleLoss | LossPerToken


class _SAEWrapper(nn.Module):
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
    """

    def __init__(self, sae: SAE[Any], use_error_term: bool = False):
        super().__init__()
        # Store SAE in __dict__ to avoid registering as submodule. This keeps cache
        # paths clean by avoiding a ".sae." prefix on hook names. See class docstring.
        self.__dict__["_sae"] = sae
        # Copy SAE's hooks directly to wrapper so they appear at the right path
        for name, hook in sae.hook_dict.items():
            setattr(self, name, hook)
        self.use_error_term = use_error_term
        self._captured_input: torch.Tensor | None = None

    @property
    def sae(self) -> SAE[Any]:
        return self.__dict__["_sae"]

    def capture_input(self, x: torch.Tensor) -> None:
        """Capture input at input hook (for transcoders).

        Note: We don't clone the tensor here - the input should not be modified
        in-place between capture and use, and avoiding clone preserves memory.
        """
        self._captured_input = x

    def forward(self, original_output: torch.Tensor) -> torch.Tensor:
        """Run SAE/transcoder at output hook location."""
        # For SAE: use original_output as input (same hook for input/output)
        # For transcoder: use captured input from earlier hook
        sae_input = (
            self._captured_input
            if self._captured_input is not None
            else original_output
        )

        # Temporarily disable SAE's internal use_error_term - we handle it here
        sae_use_error_term = self.sae.use_error_term
        self.sae.use_error_term = False
        try:
            sae_out = self.sae(sae_input)

            if self.use_error_term:
                error = original_output - sae_out.detach()
                sae_out = sae_out + error

            return sae_out
        finally:
            self.sae.use_error_term = sae_use_error_term
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


def add_hook_in_to_mlp(mlp: CanBeUsedAsMLP):
    # Temporary hack to add a `mlp.hook_in` hook to mimic what's in circuit-tracer
    mlp.hook_in = HookPoint()
    original_forward = mlp.forward
    mlp.forward = lambda x: original_forward(mlp.hook_in(x))  # type: ignore


class HookedSAETransformer(HookedTransformer):
    def __init__(
        self,
        *model_args: Any,
        **model_kwargs: Any,
    ):
        """Model initialization. Just HookedTransformer init, but adds a dictionary to keep track of attached SAEs.

        Note that if you want to load the model from pretrained weights, you should use
        :meth:`from_pretrained` instead.

        Args:
            *model_args: Positional arguments for HookedTransformer initialization
            **model_kwargs: Keyword arguments for HookedTransformer initialization
        """
        super().__init__(*model_args, **model_kwargs)

        for block in self.blocks:
            add_hook_in_to_mlp(block.mlp)  # type: ignore
        self.setup()

        self._acts_to_saes: dict[str, _SAEWrapper] = {}
        # Track output hooks used by transcoders for cleanup
        self._transcoder_output_hooks: dict[str, str] = {}

    @property
    def acts_to_saes(self) -> dict[str, SAE[Any]]:
        """Returns a dict mapping hook names to attached SAEs."""
        return {name: wrapper.sae for name, wrapper in self._acts_to_saes.items()}

    def add_sae(self, sae: SAE[Any], use_error_term: bool | None = None):
        """Attaches an SAE or Transcoder to the model.

        WARNING: This SAE will be permanently attached until you remove it with
        reset_saes. This function will also overwrite any existing SAE attached
        to the same hook point.

        Args:
            sae: The SAE or Transcoder to attach to the model.
            use_error_term: If True, computes error term so output matches what the
                model would have produced without the SAE. This works for both SAEs
                (where input==output hook) and transcoders (where they differ).
                Defaults to None (uses SAE's existing setting).
        """
        input_hook = sae.cfg.metadata.hook_name
        output_hook = sae.cfg.metadata.hook_name_out or input_hook

        if (input_hook not in self._acts_to_saes) and (
            input_hook not in self.hook_dict
        ):
            logger.warning(
                f"No hook found for {input_hook}. Skipping. Check model.hook_dict for available hooks."
            )
            return

        # Check if output hook exists (either as hook_dict entry or already has SAE attached)
        output_hook_exists = (
            output_hook in self.hook_dict
            or output_hook in self._acts_to_saes
            or any(v == output_hook for v in self._transcoder_output_hooks.values())
        )
        if not output_hook_exists:
            logger.warning(f"No hook found for output {output_hook}. Skipping.")
            return

        # Always use wrapper - it handles both SAEs and transcoders uniformly
        # If use_error_term not specified, respect SAE's existing setting
        effective_use_error_term = (
            use_error_term if use_error_term is not None else sae.use_error_term
        )
        wrapper = _SAEWrapper(sae, use_error_term=effective_use_error_term)

        # For transcoders (input != output), capture input at input hook
        if input_hook != output_hook:
            input_hook_point = get_deep_attr(self, input_hook)
            if isinstance(input_hook_point, HookPoint):
                input_hook_point.add_hook(
                    lambda tensor, hook: (wrapper.capture_input(tensor), tensor)[1],  # noqa: ARG005
                    dir="fwd",
                    is_permanent=True,
                )
            self._transcoder_output_hooks[input_hook] = output_hook

        # Store wrapper in _acts_to_saes and at output hook
        self._acts_to_saes[input_hook] = wrapper
        set_deep_attr(self, output_hook, wrapper)
        self.setup()

    def _reset_sae(
        self, act_name: str, prev_wrapper: _SAEWrapper | None = None
    ) -> None:
        """Resets an SAE that was attached to the model.

        By default will remove the SAE from that hook_point.
        If prev_wrapper is provided, will restore that wrapper's SAE with its settings.

        Args:
            act_name: The hook_name of the SAE to reset.
            prev_wrapper: The previous wrapper to restore. If None, will just
                remove the SAE from this hook point. Defaults to None.
        """
        if act_name not in self._acts_to_saes:
            logger.warning(
                f"No SAE is attached to {act_name}. There's nothing to reset."
            )
            return

        # Determine output hook location (different from input for transcoders)
        output_hook = self._transcoder_output_hooks.pop(act_name, act_name)

        # For transcoders, clear permanent hooks from input hook point
        if output_hook != act_name:
            input_hook_point = get_deep_attr(self, act_name)
            if isinstance(input_hook_point, HookPoint):
                input_hook_point.remove_hooks(dir="fwd", including_permanent=True)

        # Reset output hook location
        set_deep_attr(self, output_hook, HookPoint())
        del self._acts_to_saes[act_name]

        if prev_wrapper is not None:
            # Rebuild hook_dict before adding new SAE
            self.setup()
            self.add_sae(prev_wrapper.sae, use_error_term=prev_wrapper.use_error_term)

    def reset_saes(
        self,
        act_names: str | list[str] | None = None,
    ) -> None:
        """Reset the SAEs attached to the model.

        If act_names are provided will just reset SAEs attached to those hooks.
        Otherwise will reset all SAEs attached to the model.

        Args:
            act_names: The act_names of the SAEs to reset. If None, will reset
                all SAEs attached to the model. Defaults to None.
        """
        if isinstance(act_names, str):
            act_names = [act_names]
        elif act_names is None:
            act_names = list(self._acts_to_saes.keys())

        for act_name in act_names:
            self._reset_sae(act_name)

        self.setup()

    def run_with_saes(
        self,
        *model_args: Any,
        saes: SAE[Any] | list[SAE[Any]] = [],
        reset_saes_end: bool = True,
        use_error_term: bool | None = None,
        **model_kwargs: Any,
    ) -> None | torch.Tensor | Loss | tuple[torch.Tensor, Loss]:
        """Wrapper around HookedTransformer forward pass.

        Runs the model with the given SAEs attached for one forward pass, then removes them. By default, will reset all SAEs to original state after.

        Args:
            *model_args: Positional arguments for the model forward pass
            saes: (SAE | list[SAE]) The SAEs to be attached for this forward pass
            reset_saes_end (bool): If True, all SAEs added during this run are removed at the end, and previously attached SAEs are restored to their original state. Default is True.
            use_error_term: (bool | None) If provided, will set the use_error_term attribute of all SAEs attached during this run to this value. Defaults to None.
            **model_kwargs: Keyword arguments for the model forward pass
        """
        with self.saes(
            saes=saes, reset_saes_end=reset_saes_end, use_error_term=use_error_term
        ):
            return self(*model_args, **model_kwargs)

    def run_with_cache_with_saes(
        self,
        *model_args: Any,
        saes: SAE[Any] | list[SAE[Any]] = [],
        reset_saes_end: bool = True,
        use_error_term: bool | None = None,
        return_cache_object: bool = True,
        remove_batch_dim: bool = False,
        **kwargs: Any,
    ) -> tuple[
        None | torch.Tensor | Loss | tuple[torch.Tensor, Loss],
        ActivationCache | dict[str, torch.Tensor],
    ]:
        """Wrapper around 'run_with_cache' in HookedTransformer.

        Attaches given SAEs before running the model with cache and then removes them.
        By default, will reset all SAEs to original state after.

        Args:
            *model_args: Positional arguments for the model forward pass
            saes: (SAE | list[SAE]) The SAEs to be attached for this forward pass
            reset_saes_end: (bool) If True, all SAEs added during this run are removed at the end, and previously attached SAEs are restored to their original state. Default is True.
            use_error_term: (bool | None) If provided, will set the use_error_term attribute of all SAEs attached during this run to this value. Determines whether the SAE returns input or reconstruction. Defaults to None.
            return_cache_object: (bool) if True, this will return an ActivationCache object, with a bunch of
                useful HookedTransformer specific methods, otherwise it will return a dictionary of
                activations as in HookedRootModule.
            remove_batch_dim: (bool) Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
            **kwargs: Keyword arguments for the model forward pass
        """
        with self.saes(
            saes=saes, reset_saes_end=reset_saes_end, use_error_term=use_error_term
        ):
            return self.run_with_cache(  # type: ignore
                *model_args,
                return_cache_object=return_cache_object,  # type: ignore
                remove_batch_dim=remove_batch_dim,
                **kwargs,
            )

    def run_with_hooks_with_saes(
        self,
        *model_args: Any,
        saes: SAE[Any] | list[SAE[Any]] = [],
        reset_saes_end: bool = True,
        fwd_hooks: list[tuple[str | Callable, Callable]] = [],  # type: ignore
        bwd_hooks: list[tuple[str | Callable, Callable]] = [],  # type: ignore
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        **model_kwargs: Any,
    ):
        """Wrapper around 'run_with_hooks' in HookedTransformer.

        Attaches the given SAEs to the model before running the model with hooks and then removes them.
        By default, will reset all SAEs to original state after.

        Args:
            *model_args: Positional arguments for the model forward pass
            saes: (SAE | list[SAE]) The SAEs to be attached for this forward pass
            reset_saes_end: (bool) If True, all SAEs added during this run are removed at the end, and previously attached SAEs are restored to their original state. (default: True)
            fwd_hooks: (list[tuple[str | Callable, Callable]]) List of forward hooks to apply
            bwd_hooks: (list[tuple[str | Callable, Callable]]) List of backward hooks to apply
            reset_hooks_end: (bool) Whether to reset the hooks at the end of the forward pass (default: True)
            clear_contexts: (bool) Whether to clear the contexts at the end of the forward pass (default: False)
            **model_kwargs: Keyword arguments for the model forward pass
        """
        with self.saes(saes=saes, reset_saes_end=reset_saes_end):
            return self.run_with_hooks(
                *model_args,
                fwd_hooks=fwd_hooks,
                bwd_hooks=bwd_hooks,
                reset_hooks_end=reset_hooks_end,
                clear_contexts=clear_contexts,
                **model_kwargs,
            )

    @contextmanager
    def saes(
        self,
        saes: SAE[Any] | list[SAE[Any]] = [],
        reset_saes_end: bool = True,
        use_error_term: bool | None = None,
    ):
        """A context manager for adding temporary SAEs to the model.

        See HookedTransformer.hooks for a similar context manager for hooks.
        By default will keep track of previously attached SAEs, and restore
        them when the context manager exits.

        Args:
            saes: SAEs to be attached.
            reset_saes_end: If True, removes all SAEs added by this context
                manager when the context manager exits, returning previously
                attached SAEs to their original state.
            use_error_term: If provided, will set the use_error_term attribute
                of all SAEs attached during this run to this value.
        """
        saes_to_restore: list[tuple[str, _SAEWrapper | None]] = []
        if isinstance(saes, SAE):
            saes = [saes]
        try:
            for sae in saes:
                act_name = sae.cfg.metadata.hook_name
                prev_wrapper = self._acts_to_saes.get(act_name, None)
                saes_to_restore.append((act_name, prev_wrapper))
                self.add_sae(sae, use_error_term=use_error_term)
            yield self
        finally:
            if reset_saes_end:
                for act_name, prev_wrapper in saes_to_restore:
                    self._reset_sae(act_name, prev_wrapper)
                self.setup()
