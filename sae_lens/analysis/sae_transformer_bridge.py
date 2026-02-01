from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge

from sae_lens import logger
from sae_lens.analysis.hooked_sae_transformer import (
    _SAEWrapper,
    get_deep_attr,
    set_deep_attr,
)
from sae_lens.saes.sae import SAE

SingleLoss = torch.Tensor  # Type alias for a single element tensor
LossPerToken = torch.Tensor
Loss = SingleLoss | LossPerToken


class SAETransformerBridge(TransformerBridge):  # type: ignore[misc,no-untyped-call]
    """A TransformerBridge subclass that supports attaching SAEs.

    .. warning::
        This class is in **beta**. The API may change in future versions.

    This class provides the same SAE attachment functionality as HookedSAETransformer,
    but for transformer-lens v3's TransformerBridge instead of HookedTransformer.

    TransformerBridge is a lightweight wrapper around HuggingFace models that provides
    hook points without the overhead of HookedTransformer's weight processing. This is
    useful for models not natively supported by HookedTransformer, such as Gemma 3.
    """

    _acts_to_saes: dict[str, _SAEWrapper]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._acts_to_saes = {}
        # Track output hooks used by transcoders for cleanup
        self._transcoder_output_hooks: dict[str, str] = {}

    @property
    def acts_to_saes(self) -> dict[str, SAE[Any]]:
        """Returns a dict mapping hook names to attached SAEs."""
        return {name: wrapper.sae for name, wrapper in self._acts_to_saes.items()}

    @classmethod
    def boot_transformers(  # type: ignore[override]
        cls,
        model_name: str,
        **kwargs: Any,
    ) -> "SAETransformerBridge":
        """Factory method to boot a model and return SAETransformerBridge instance.

        Args:
            model_name: The name of the model to load (e.g., "gpt2", "gemma-2-2b")
            **kwargs: Additional arguments passed to TransformerBridge.boot_transformers

        Returns:
            SAETransformerBridge instance with the loaded model
        """
        # Boot parent TransformerBridge
        bridge = TransformerBridge.boot_transformers(model_name, **kwargs)
        # Convert to our class
        # NOTE: this is super hacky and scary, but I don't know how else to achieve this given TLens' internal code
        bridge.__class__ = cls
        bridge._acts_to_saes = {}  # type: ignore[attr-defined]
        bridge._transcoder_output_hooks = {}  # type: ignore[attr-defined]
        return bridge  # type: ignore[return-value]

    def _resolve_hook_name(self, hook_name: str) -> str:
        """Resolve alias to actual hook name.

        TransformerBridge supports hook aliases like 'blocks.0.hook_mlp_out'
        that map to actual paths like 'blocks.0.mlp.hook_out'.
        """
        # Combine static and dynamic aliases
        aliases: dict[str, Any] = {
            **self.hook_aliases,
            **self._collect_hook_aliases_from_registry(),
        }
        resolved = aliases.get(hook_name, hook_name)
        # aliases values are always strings, but type checker doesn't know this
        return resolved if isinstance(resolved, str) else hook_name

    def add_sae(self, sae: SAE[Any], use_error_term: bool | None = None) -> None:
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
        input_hook_alias = sae.cfg.metadata.hook_name
        output_hook_alias = sae.cfg.metadata.hook_name_out or input_hook_alias
        input_hook_actual = self._resolve_hook_name(input_hook_alias)
        output_hook_actual = self._resolve_hook_name(output_hook_alias)

        # Check if hooks exist
        if (input_hook_alias not in self._acts_to_saes) and (
            input_hook_actual not in self._hook_registry
        ):
            logger.warning(
                f"No hook found for {input_hook_alias}. Skipping. "
                f"Check model._hook_registry for available hooks."
            )
            return

        # Check if output hook exists (either as registry entry or already has SAE attached)
        output_hook_exists = (
            output_hook_actual in self._hook_registry
            or input_hook_alias in self._acts_to_saes
            or any(
                v == output_hook_actual for v in self._transcoder_output_hooks.values()
            )
        )
        if not output_hook_exists:
            logger.warning(f"No hook found for output {output_hook_alias}. Skipping.")
            return

        # Always use wrapper - it handles both SAEs and transcoders uniformly
        # If use_error_term not specified, respect SAE's existing setting
        effective_use_error_term = (
            use_error_term if use_error_term is not None else sae.use_error_term
        )
        wrapper = _SAEWrapper(sae, use_error_term=effective_use_error_term)

        # For transcoders (input != output), capture input at input hook
        if input_hook_alias != output_hook_alias:
            input_hook_point = get_deep_attr(self, input_hook_actual)
            if isinstance(input_hook_point, HookPoint):
                input_hook_point.add_hook(
                    lambda tensor, hook: (wrapper.capture_input(tensor), tensor)[1],  # noqa: ARG005
                    dir="fwd",
                    is_permanent=True,
                )
            self._transcoder_output_hooks[input_hook_alias] = output_hook_actual

        # Store wrapper in _acts_to_saes and at output hook
        set_deep_attr(self, output_hook_actual, wrapper)
        self._hook_registry[output_hook_actual] = wrapper  # type: ignore[assignment]
        self._acts_to_saes[input_hook_alias] = wrapper

    def _reset_sae(
        self, act_name: str, prev_wrapper: _SAEWrapper | None = None
    ) -> None:
        """Resets an SAE that was attached to the model.

        By default will remove the SAE from that hook_point.
        If prev_wrapper is provided, will restore that wrapper's SAE with its settings.

        Args:
            act_name: The hook_name of the SAE to reset
            prev_wrapper: The previous wrapper to restore. If None, will just
                remove the SAE from this hook point. Defaults to None.
        """
        if act_name not in self._acts_to_saes:
            logger.warning(
                f"No SAE is attached to {act_name}. There's nothing to reset."
            )
            return

        actual_name = self._resolve_hook_name(act_name)

        # Determine output hook location (different from input for transcoders)
        output_hook = self._transcoder_output_hooks.pop(act_name, actual_name)

        # For transcoders, clear permanent hooks from input hook point
        if output_hook != actual_name:
            input_hook_point = get_deep_attr(self, actual_name)
            if isinstance(input_hook_point, HookPoint):
                input_hook_point.remove_hooks(dir="fwd", including_permanent=True)

        # Reset output hook location
        new_hook = HookPoint()
        new_hook.name = output_hook
        set_deep_attr(self, output_hook, new_hook)
        self._hook_registry[output_hook] = new_hook
        del self._acts_to_saes[act_name]

        if prev_wrapper is not None:
            self.add_sae(prev_wrapper.sae, use_error_term=prev_wrapper.use_error_term)

    def reset_saes(
        self,
        act_names: str | list[str] | None = None,
    ) -> None:
        """Reset the SAEs attached to the model.

        If act_names are provided will just reset SAEs attached to those hooks.
        Otherwise will reset all SAEs attached to the model.

        Args:
            act_names: The act_names of the SAEs to reset. If None, will reset all
                SAEs attached to the model. Defaults to None.
        """
        if isinstance(act_names, str):
            act_names = [act_names]
        elif act_names is None:
            act_names = list(self._acts_to_saes.keys())

        for act_name in act_names:
            self._reset_sae(act_name)

    def run_with_saes(
        self,
        *model_args: Any,
        saes: SAE[Any] | list[SAE[Any]] = [],
        reset_saes_end: bool = True,
        use_error_term: bool | None = None,
        **model_kwargs: Any,
    ) -> torch.Tensor | Loss | tuple[torch.Tensor, Loss] | None:
        """Wrapper around forward pass.

        Runs the model with the given SAEs attached for one forward pass, then
        removes them. By default, will reset all SAEs to original state after.

        Args:
            *model_args: Positional arguments for the model forward pass
            saes: The SAEs to be attached for this forward pass
            reset_saes_end: If True, all SAEs added during this run are removed
                at the end, and previously attached SAEs are restored to their
                original state. Default is True.
            use_error_term: If provided, will set the use_error_term attribute
                of all SAEs attached during this run to this value. Defaults to None.
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
        torch.Tensor | Loss | tuple[torch.Tensor, Loss] | None,
        ActivationCache | dict[str, torch.Tensor],
    ]:
        """Wrapper around 'run_with_cache'.

        Attaches given SAEs before running the model with cache and then removes them.
        By default, will reset all SAEs to original state after.

        Args:
            *model_args: Positional arguments for the model forward pass
            saes: The SAEs to be attached for this forward pass
            reset_saes_end: If True, all SAEs added during this run are removed
                at the end, and previously attached SAEs are restored to their
                original state. Default is True.
            use_error_term: If provided, will set the use_error_term attribute
                of all SAEs attached during this run to this value. Defaults to None.
            return_cache_object: If True, returns an ActivationCache object with
                useful methods, otherwise returns a dictionary of activations.
            remove_batch_dim: Whether to remove the batch dimension
                (only works for batch_size==1). Defaults to False.
            **kwargs: Keyword arguments for the model forward pass
        """
        with self.saes(
            saes=saes, reset_saes_end=reset_saes_end, use_error_term=use_error_term
        ):
            return self.run_with_cache(
                *model_args,
                return_cache_object=return_cache_object,  # type: ignore[arg-type]
                remove_batch_dim=remove_batch_dim,
                **kwargs,
            )  # type: ignore[return-value]

    def run_with_hooks_with_saes(
        self,
        *model_args: Any,
        saes: SAE[Any] | list[SAE[Any]] = [],
        reset_saes_end: bool = True,
        fwd_hooks: list[tuple[str | Callable[..., Any], Callable[..., Any]]] = [],
        bwd_hooks: list[tuple[str | Callable[..., Any], Callable[..., Any]]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        **model_kwargs: Any,
    ) -> Any:
        """Wrapper around 'run_with_hooks'.

        Attaches the given SAEs to the model before running the model with hooks
        and then removes them. By default, will reset all SAEs to original state after.

        Args:
            *model_args: Positional arguments for the model forward pass
            saes: The SAEs to be attached for this forward pass
            reset_saes_end: If True, all SAEs added during this run are removed
                at the end, and previously attached SAEs are restored to their
                original state. Default is True.
            fwd_hooks: List of forward hooks to apply
            bwd_hooks: List of backward hooks to apply
            reset_hooks_end: Whether to reset the hooks at the end of the forward
                pass. Default is True.
            clear_contexts: Whether to clear the contexts at the end of the forward
                pass. Default is False.
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
    ):  # type: ignore[no-untyped-def]
        """A context manager for adding temporary SAEs to the model.

        By default will keep track of previously attached SAEs, and restore them
        when the context manager exits.

        Args:
            saes: SAEs to be attached.
            reset_saes_end: If True, removes all SAEs added by this context manager
                when the context manager exits, returning previously attached SAEs
                to their original state.
            use_error_term: If provided, will set the use_error_term attribute of
                all SAEs attached during this run to this value. Defaults to None.
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

    @property
    def hook_dict(self) -> dict[str, HookPoint]:
        """Return combined hook registry including SAE internal hooks.

        When SAEs are attached, they replace HookPoint entries in the registry.
        This property returns both the base hooks and any internal hooks from
        attached SAEs (like hook_sae_acts_post, hook_sae_input, etc.) with
        their full path names.
        """
        hooks: dict[str, HookPoint] = {}

        for name, hook_or_sae in self._hook_registry.items():
            if isinstance(hook_or_sae, _SAEWrapper):
                # Include SAE's internal hooks with full path names
                for sae_hook_name, sae_hook in hook_or_sae.sae.hook_dict.items():
                    full_name = f"{name}.{sae_hook_name}"
                    hooks[full_name] = sae_hook
            else:
                hooks[name] = hook_or_sae

        return hooks
