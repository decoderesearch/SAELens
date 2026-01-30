from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge

from sae_lens import logger
from sae_lens.analysis.hooked_sae_transformer import set_deep_attr
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

    acts_to_saes: dict[str, SAE[Any]]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.acts_to_saes = {}

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
        bridge.acts_to_saes = {}  # type: ignore[attr-defined]
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
        """Attaches an SAE to the model.

        WARNING: This SAE will be permanently attached until you remove it with
        reset_saes. This function will also overwrite any existing SAE attached
        to the same hook point.

        Args:
            sae: The SAE to attach to the model
            use_error_term: If provided, will set the use_error_term attribute of
                the SAE to this value. Determines whether the SAE returns input
                or reconstruction. Defaults to None.
        """
        alias_name = sae.cfg.metadata.hook_name
        actual_name = self._resolve_hook_name(alias_name)

        # Check if hook exists (either as alias or actual name)
        if (alias_name not in self.acts_to_saes) and (
            actual_name not in self._hook_registry
        ):
            logger.warning(
                f"No hook found for {alias_name}. Skipping. "
                f"Check model._hook_registry for available hooks."
            )
            return

        if use_error_term is not None:
            if not hasattr(sae, "_original_use_error_term"):
                sae._original_use_error_term = sae.use_error_term  # type: ignore[attr-defined]
            sae.use_error_term = use_error_term

        # Replace hook and update registry
        set_deep_attr(self, actual_name, sae)
        self._hook_registry[actual_name] = sae  # type: ignore[assignment]
        self.acts_to_saes[alias_name] = sae

    def _reset_sae(self, act_name: str, prev_sae: SAE[Any] | None = None) -> None:
        """Resets an SAE that was attached to the model.

        By default will remove the SAE from that hook_point.
        If prev_sae is provided, will replace the current SAE with the provided one.
        This is mainly used to restore previously attached SAEs after temporarily
        running with different SAEs (e.g., with run_with_saes).

        Args:
            act_name: The hook_name of the SAE to reset
            prev_sae: The SAE to replace the current one with. If None, will just
                remove the SAE from this hook point. Defaults to None.
        """
        if act_name not in self.acts_to_saes:
            logger.warning(
                f"No SAE is attached to {act_name}. There's nothing to reset."
            )
            return

        actual_name = self._resolve_hook_name(act_name)
        current_sae = self.acts_to_saes[act_name]

        if hasattr(current_sae, "_original_use_error_term"):
            current_sae.use_error_term = current_sae._original_use_error_term  # type: ignore[attr-defined]
            delattr(current_sae, "_original_use_error_term")

        if prev_sae is not None:
            set_deep_attr(self, actual_name, prev_sae)
            self._hook_registry[actual_name] = prev_sae  # type: ignore[assignment]
            self.acts_to_saes[act_name] = prev_sae
        else:
            new_hook = HookPoint()
            new_hook.name = actual_name
            set_deep_attr(self, actual_name, new_hook)
            self._hook_registry[actual_name] = new_hook
            del self.acts_to_saes[act_name]

    def reset_saes(
        self,
        act_names: str | list[str] | None = None,
        prev_saes: list[SAE[Any] | None] | None = None,
    ) -> None:
        """Reset the SAEs attached to the model.

        If act_names are provided will just reset SAEs attached to those hooks.
        Otherwise will reset all SAEs attached to the model.
        Optionally can provide a list of prev_saes to reset to. This is mainly
        used to restore previously attached SAEs after temporarily running with
        different SAEs (e.g., with run_with_saes).

        Args:
            act_names: The act_names of the SAEs to reset. If None, will reset all
                SAEs attached to the model. Defaults to None.
            prev_saes: List of SAEs to replace the current ones with. If None, will
                just remove the SAEs. Defaults to None.
        """
        if isinstance(act_names, str):
            act_names = [act_names]
        elif act_names is None:
            act_names = list(self.acts_to_saes.keys())

        if prev_saes:
            if len(act_names) != len(prev_saes):
                raise ValueError("act_names and prev_saes must have the same length")
        else:
            prev_saes = [None] * len(act_names)  # type: ignore[assignment]

        for act_name, prev_sae in zip(act_names, prev_saes):  # type: ignore[arg-type]
            self._reset_sae(act_name, prev_sae)

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
        act_names_to_reset: list[str] = []
        prev_saes: list[SAE[Any] | None] = []
        if isinstance(saes, SAE):
            saes = [saes]
        try:
            for sae in saes:
                act_names_to_reset.append(sae.cfg.metadata.hook_name)
                prev_sae = self.acts_to_saes.get(sae.cfg.metadata.hook_name, None)
                prev_saes.append(prev_sae)
                self.add_sae(sae, use_error_term=use_error_term)
            yield self
        finally:
            if reset_saes_end:
                self.reset_saes(act_names_to_reset, prev_saes)

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
            if isinstance(hook_or_sae, SAE):
                # Include SAE's internal hooks with full path names
                for sae_hook_name, sae_hook in hook_or_sae.hook_dict.items():
                    full_name = f"{name}.{sae_hook_name}"
                    hooks[full_name] = sae_hook
            else:
                hooks[name] = hook_or_sae

        return hooks
