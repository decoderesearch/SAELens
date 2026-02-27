from typing import Any, Callable, Literal, cast

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import Loss, Output
from transformer_lens.utils import (
    USE_DEFAULT_VALUE,
    get_tokens_with_bos_removed,
    lm_cross_entropy_loss,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from sae_lens import logger


def load_model(
    model_class_name: str,
    model_name: str,
    device: str | torch.device | None = None,
    model_from_pretrained_kwargs: dict[str, Any] | None = None,
) -> HookedRootModule:
    model_from_pretrained_kwargs = model_from_pretrained_kwargs or {}

    if "n_devices" in model_from_pretrained_kwargs:
        n_devices = model_from_pretrained_kwargs["n_devices"]
        if n_devices > 1:
            logger.info("MODEL LOADING:")
            logger.info("Setting model device to cuda for d_devices")
            logger.info(f"Will use cuda:0 to cuda:{n_devices-1}")
            device = "cuda"
            logger.info("-------------")

    if model_class_name == "HookedTransformer":
        return HookedTransformer.from_pretrained_no_processing(
            model_name=model_name, device=device, **model_from_pretrained_kwargs
        )
    if model_class_name == "HookedMamba":
        try:
            from mamba_lens import HookedMamba
        except ImportError:  # pragma: no cover
            raise ValueError(
                "mamba-lens must be installed to work with mamba models. This can be added with `pip install sae-lens[mamba]`"
            )
        # HookedMamba has incorrect typing information, so we need to cast the type here
        return cast(
            HookedRootModule,
            HookedMamba.from_pretrained(
                model_name, device=cast(Any, device), **model_from_pretrained_kwargs
            ),
        )
    if model_class_name == "AutoModelForCausalLM":
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_from_pretrained_kwargs
        ).to(device)  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return HookedProxyLM(hf_model, tokenizer)

    # pragma: no cover
    raise ValueError(f"Unknown model class: {model_class_name}")


class HookedProxyLM(HookedRootModule):
    """
    A HookedRootModule that wraps a Huggingface AutoModelForCausalLM.
    """

    tokenizer: PreTrainedTokenizerBase
    model: torch.nn.Module

    def __init__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.setup()

    # copied and modified from base HookedRootModule
    def setup(self):
        self.mod_dict = {}
        self.named_modules_dict = {}
        self.hook_dict: dict[str, HookPoint] = {}
        for name, module in self.model.named_modules():
            if name == "":
                continue

            hook_point = HookPoint()
            hook_point.name = name  # type: ignore

            module.register_forward_hook(get_hook_fn(hook_point))

            self.hook_dict[name] = hook_point
            self.mod_dict[name] = hook_point
            self.named_modules_dict[name] = module

    def run_with_cache(self, *args: Any, **kwargs: Any):  # type: ignore
        if "names_filter" in kwargs:
            # hacky way to make sure that the names_filter is passed to our forward method
            kwargs["_names_filter"] = kwargs["names_filter"]
        return super().run_with_cache(*args, **kwargs)

    def forward(
        self,
        tokens: torch.Tensor,
        return_type: Literal["both", "logits"] = "logits",
        loss_per_token: bool = False,
        stop_at_layer: int | None = None,
        _names_filter: list[str] | None = None,
        **kwargs: Any,
    ) -> Output | Loss | None:
        # This is just what's needed for evals, not everything that HookedTransformer has
        if return_type not in (
            "both",
            "logits",
        ):
            raise NotImplementedError(
                "Only return_type supported is 'both' or 'logits' to match what's in evals.py and ActivationsStore"
            )

        stop_hooks = []
        if stop_at_layer is not None and _names_filter is not None:
            if return_type != "logits":
                raise NotImplementedError(
                    "stop_at_layer is not supported for return_type='both'"
                )
            stop_manager = StopManager(_names_filter)

            for hook_name in _names_filter:
                module = self.named_modules_dict[hook_name]
                stop_fn = stop_manager.get_stop_hook_fn(hook_name)
                stop_hooks.append(module.register_forward_hook(stop_fn))
        try:
            output = self.model(tokens)
            logits = _extract_logits_from_output(output)
        except StopForward:
            # If we stop early, we don't care about the return output
            return None  # type: ignore
        finally:
            for stop_hook in stop_hooks:
                stop_hook.remove()

        if return_type == "logits":
            return logits

        if logits is None:
            raise ValueError(
                "return_type='both' requires logits, but the model returned logits=None."
            )

        if tokens.device != logits.device:
            tokens = tokens.to(logits.device)
        loss = lm_cross_entropy_loss(logits, tokens, per_token=loss_per_token)
        return Output(logits, loss)

    def to_tokens(
        self,
        input: str | list[str],
        prepend_bos: bool | None = USE_DEFAULT_VALUE,
        padding_side: Literal["left", "right"] | None = USE_DEFAULT_VALUE,
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> torch.Tensor:
        # Hackily modified version of HookedTransformer.to_tokens to work with ActivationsStore
        # Assumes that prepend_bos is always False, move_to_device is always False, and truncate is always False
        # copied from HookedTransformer.to_tokens

        if prepend_bos is not False:
            raise ValueError(
                "Only works with prepend_bos=False, to match ActivationsStore usage"
            )

        if padding_side is not None:
            raise ValueError(
                "Only works with padding_side=None, to match ActivationsStore usage"
            )

        if truncate is not False:
            raise ValueError(
                "Only works with truncate=False, to match ActivationsStore usage"
            )

        if move_to_device is not False:
            raise ValueError(
                "Only works with move_to_device=False, to match ActivationsStore usage"
            )

        tokens = self.tokenizer(
            input,
            return_tensors="pt",
            truncation=False,
            max_length=None,
        )["input_ids"]

        # We don't want to prepend bos but the tokenizer does it automatically, so we remove it manually
        if hasattr(self.tokenizer, "add_bos_token") and self.tokenizer.add_bos_token:  # type: ignore
            tokens = get_tokens_with_bos_removed(self.tokenizer, tokens)  # type: ignore
        return tokens  # type: ignore


def _extract_logits_from_output(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if hasattr(output, "logits"):
        return getattr(output, "logits")
    raise ValueError(
        f"Unsupported model output type for logits extraction: {type(output)}"
    )


def get_hook_fn(hook_point: HookPoint):
    def hook_fn(module: Any, input: Any, output: Any) -> Any:  # noqa: ARG001
        if isinstance(output, torch.Tensor):
            return hook_point(output)
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            return (hook_point(output[0]), *output[1:])
        # if this isn't a tensor, just skip the hook entirely as this will break otherwise
        return output

    return hook_fn


class StopForward(Exception):
    pass


class StopManager:
    def __init__(self, hook_names: list[str]):
        self.hook_names = hook_names
        self.total_hook_names = len(set(hook_names))
        self.called_hook_names = set()

    def get_stop_hook_fn(self, hook_name: str) -> Callable[[Any, Any, Any], Any]:
        def stop_hook_fn(module: Any, input: Any, output: Any) -> Any:  # noqa: ARG001
            self.called_hook_names.add(hook_name)
            if len(self.called_hook_names) == self.total_hook_names:
                raise StopForward()
            return output

        return stop_hook_fn
