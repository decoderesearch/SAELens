import copy
from collections.abc import Sequence
from typing import Any, Literal, TypedDict, cast

import pytest
import torch
from transformer_lens import HookedTransformer

from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAEConfig
from sae_lens.saes.gated_sae import GatedSAEConfig, GatedTrainingSAEConfig
from sae_lens.saes.jumprelu_sae import JumpReLUSAEConfig, JumpReLUTrainingSAEConfig
from sae_lens.saes.matching_pursuit_sae import (
    MatchingPursuitSAEConfig,
    MatchingPursuitTrainingSAEConfig,
)
from sae_lens.saes.matryoshka_batchtopk_sae import MatryoshkaBatchTopKTrainingSAEConfig
from sae_lens.saes.sae import T_TRAINING_SAE_CONFIG, SAEConfig, TrainingSAEConfig
from sae_lens.saes.standard_sae import StandardSAEConfig, StandardTrainingSAEConfig
from sae_lens.saes.temporal_sae import TemporalSAEConfig
from sae_lens.saes.topk_sae import TopKSAEConfig, TopKTrainingSAEConfig

TINYSTORIES_MODEL = "tiny-stories-1M"
NEEL_NANDA_C4_10K_DATASET = "NeelNanda/c4-10k"

ALL_ARCHITECTURES = [
    "standard",
    "gated",
    "jumprelu",
    "topk",
    "temporal",
    "matching_pursuit",
]
ALL_FOLDABLE_ARCHITECTURES = [
    "standard",
    "gated",
    "jumprelu",
]  # Architectures with fold W_dec to unit norm implementation
ALL_TRAINING_ARCHITECTURES = [
    "standard",
    "gated",
    "jumprelu",
    "topk",
    "batchtopk",
    "matryoshka_batchtopk",
    "matching_pursuit",
]


def to_sparse(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a dense tensor to sparse COO format."""
    return tensor.to_sparse_coo()


def to_dense(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to dense format if sparse."""
    return tensor.to_dense() if tensor.is_sparse else tensor


# This TypedDict should match the fields directly in LanguageModelSAERunnerConfig
class LanguageModelSAERunnerConfigDict(TypedDict, total=False):
    sae: TrainingSAEConfig  # The nested SAE config object
    model_name: str
    model_class_name: str
    hook_name: str
    hook_head_index: int | None
    dataset_path: str
    dataset_trust_remote_code: bool
    streaming: bool
    is_dataset_tokenized: bool
    context_size: int
    use_cached_activations: bool
    cached_activations_path: str | None
    from_pretrained_path: str | None
    n_batches_in_buffer: int
    training_tokens: int
    store_batch_size_prompts: int
    normalize_activations: str
    seqpos_slice: tuple[int | None, ...] | Sequence[int | None]
    disable_concat_sequences: bool
    sequence_separator_token: int | Literal["bos", "eos", "sep"] | None
    device: str
    act_store_device: str
    seed: int
    dtype: str
    prepend_bos: bool
    autocast: bool
    autocast_lm: bool
    compile_llm: bool
    llm_compilation_mode: str | None
    compile_sae: bool
    sae_compilation_mode: str | None
    train_batch_size_tokens: int
    adam_beta1: float
    adam_beta2: float
    lr: float
    lr_scheduler_name: str
    lr_warm_up_steps: int
    lr_end: float | None
    lr_decay_steps: int
    n_restart_cycles: int
    dead_feature_window: int
    feature_sampling_window: int
    dead_feature_threshold: float
    n_eval_batches: int
    eval_batch_size_prompts: int | None
    logger: LoggingConfig
    n_checkpoints: int
    checkpoint_path: str | None
    save_final_checkpoint: bool
    output_path: str | None
    verbose: bool
    model_kwargs: dict[str, Any]
    model_from_pretrained_kwargs: dict[str, Any] | None
    sae_lens_version: str
    sae_lens_training_version: str
    exclude_special_tokens: bool | list[int]


# Base TrainingSAEConfig fields + all architecture specific fields
class TrainingSAEConfigDict(TypedDict, total=False):
    d_in: int
    d_sae: int
    dtype: str
    device: str
    l1_coefficient: float
    lp_norm: float
    normalize_activations: str
    apply_b_dec_to_input: bool
    l1_warm_up_steps: int
    decoder_init_norm: float | None
    # Fields specific to some architectures
    jumprelu_init_threshold: float
    jumprelu_bandwidth: float
    k: int  # For TopK
    use_sparse_activations: bool  # For TopK
    l0_coefficient: float  # For JumpReLU
    l0_warm_up_steps: int
    pre_act_loss_coefficient: float | None  # For JumpReLU
    topk_threshold_lr: float  # For BatchTopK
    jumprelu_sparsity_loss_mode: Literal["step", "tanh"]  # For JumpReLU
    jumprelu_tanh_scale: float  # For JumpReLU
    rescale_acts_by_decoder_norm: bool  # For TopK
    matryoshka_widths: list[int]  # For MatryoshkaBatchTopK
    residual_threshold: float  # for MatchingPursuitSAE
    max_iterations: int | None  # for MatchingPursuitSAE
    stop_on_duplicate_support: bool  # for MatchingPursuitSAE


class SAEConfigDict(TypedDict, total=False):
    d_in: int
    d_sae: int
    dtype: str
    device: str
    normalize_activations: str
    apply_b_dec_to_input: bool
    k: int  # For TopK
    residual_threshold: float  # for matching pursuit
    max_iterations: int | None  # for MatchingPursuitSAE
    stop_on_duplicate_support: bool  # for MatchingPursuitSAE


# Helper to create the base runner config (reused by specific builders)
def _get_default_runner_config() -> LanguageModelSAERunnerConfigDict:
    # Ensure logger is always present in the default dict
    return {
        "model_name": TINYSTORIES_MODEL,
        "model_class_name": "HookedTransformer",
        "hook_name": "blocks.0.hook_mlp_out",
        "hook_head_index": None,
        "dataset_path": NEEL_NANDA_C4_10K_DATASET,
        "streaming": False,
        "dataset_trust_remote_code": True,
        "is_dataset_tokenized": False,
        "context_size": 6,
        "use_cached_activations": False,
        "cached_activations_path": None,
        "from_pretrained_path": None,
        "n_batches_in_buffer": 2,
        "training_tokens": 1_000_000,
        "store_batch_size_prompts": 4,
        "seqpos_slice": (None,),
        "disable_concat_sequences": False,
        "sequence_separator_token": "bos",
        "device": "cpu",
        "act_store_device": "cpu",
        "seed": 24,
        "dtype": "float32",
        "prepend_bos": True,
        "autocast": False,
        "autocast_lm": False,
        "compile_llm": False,
        "llm_compilation_mode": None,
        "compile_sae": False,
        "sae_compilation_mode": None,
        "train_batch_size_tokens": 4,
        "adam_beta1": 0.0,
        "adam_beta2": 0.999,
        "lr": 2e-4,
        "lr_scheduler_name": "constant",
        "lr_warm_up_steps": 0,
        "lr_end": None,
        "lr_decay_steps": 0,
        "n_restart_cycles": 1,
        "dead_feature_window": 1000,
        "dead_feature_threshold": 1e-7,
        "n_eval_batches": 10,
        "eval_batch_size_prompts": None,
        "logger": LoggingConfig(
            log_to_wandb=False,
            wandb_project="test_project",
            wandb_entity="test_entity",
            wandb_log_frequency=10,
            eval_every_n_wandb_logs=100,
        ),
        "n_checkpoints": 0,
        "checkpoint_path": None,
        "save_final_checkpoint": False,
        "output_path": None,
        "verbose": True,
        "model_kwargs": {},
        "model_from_pretrained_kwargs": None,
        "sae_lens_version": "test_version",
        "sae_lens_training_version": "test_version",
        "exclude_special_tokens": False,
    }


# Helper to separate kwargs and build final config
def _build_runner_config(
    SAEConfigClass: type[T_TRAINING_SAE_CONFIG],
    default_sae_config: dict[str, Any],
    **kwargs: Any,
) -> LanguageModelSAERunnerConfig[T_TRAINING_SAE_CONFIG]:
    default_runner_config = _get_default_runner_config()

    runner_overrides: LanguageModelSAERunnerConfigDict = {}
    sae_overrides: dict[str, Any] = {}  # Use generic dict for overrides
    sae_config_keys = default_sae_config.keys()
    runner_config_keys = LanguageModelSAERunnerConfigDict.__annotations__.keys()

    for key, value in kwargs.items():
        if key in sae_config_keys:
            sae_overrides[key] = value
        elif key in runner_config_keys:
            runner_overrides[key] = value  # type: ignore
        # else: Ignore unknown keys

    final_runner_config_dict = {**default_runner_config, **runner_overrides}
    final_sae_config_dict = {**default_sae_config, **sae_overrides}

    # Handle logger override (needs special handling as it's an object)
    default_logger = default_runner_config.get("logger")
    if "logger" in runner_overrides and isinstance(default_logger, LoggingConfig):
        logger_override = runner_overrides["logger"]
        if isinstance(logger_override, dict):
            logger_obj = default_logger
            for log_key, log_val in logger_override.items():
                if hasattr(logger_obj, log_key):
                    setattr(logger_obj, log_key, log_val)
            final_runner_config_dict["logger"] = logger_obj
        elif isinstance(logger_override, LoggingConfig):
            final_runner_config_dict["logger"] = logger_override  # type: ignore

    # Create the SAE config object
    sae_config_object = SAEConfigClass(**final_sae_config_dict)

    # Add SAE object to runner dict
    final_runner_config_dict["sae"] = sae_config_object

    # Instantiate the final runner config
    final_config = LanguageModelSAERunnerConfig(**final_runner_config_dict)  # type: ignore

    # Apply checkpoint path override *after* initialization
    if "checkpoint_path" in kwargs:
        final_config.checkpoint_path = kwargs["checkpoint_path"]

    return final_config


def _update_sae_metadata(runner_cfg: LanguageModelSAERunnerConfig[Any]):
    runner_cfg.sae.metadata.hook_name = runner_cfg.hook_name
    runner_cfg.sae.metadata.hook_head_index = runner_cfg.hook_head_index
    runner_cfg.sae.metadata.model_name = runner_cfg.model_name
    runner_cfg.sae.metadata.model_class_name = runner_cfg.model_class_name
    runner_cfg.sae.metadata.dataset_path = runner_cfg.dataset_path
    runner_cfg.sae.metadata.prepend_bos = runner_cfg.prepend_bos


# --- Standard SAE Builder ---
def build_runner_cfg(
    **kwargs: Any,
) -> LanguageModelSAERunnerConfig[StandardTrainingSAEConfig]:
    """Helper to create a mock instance for Standard SAE."""
    default_sae_config: TrainingSAEConfigDict = {
        "d_in": 64,
        "d_sae": 256,
        "dtype": "float32",
        "device": "cpu",
        "decoder_init_norm": 0.1,
        "l1_coefficient": 2e-3,
        "lp_norm": 1.0,
        "normalize_activations": "none",
        "apply_b_dec_to_input": False,
        "l1_warm_up_steps": 0,
    }
    runner_cfg = _build_runner_config(
        StandardTrainingSAEConfig,
        cast(dict[str, Any], default_sae_config),
        **kwargs,
    )
    _update_sae_metadata(runner_cfg)
    return runner_cfg


def build_sae_training_cfg(**kwargs: Any) -> StandardTrainingSAEConfig:
    return build_runner_cfg(**kwargs).sae  # type: ignore


def build_sae_cfg(**kwargs: Any) -> StandardSAEConfig:
    default_sae_config: SAEConfigDict = {
        "d_in": 64,
        "d_sae": 256,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
    }
    return StandardSAEConfig(**{**default_sae_config, **kwargs})  # type: ignore


# --- JumpReLU SAE Builder ---
def build_jumprelu_runner_cfg(
    **kwargs: Any,
) -> LanguageModelSAERunnerConfig[JumpReLUTrainingSAEConfig]:
    """Helper to create a mock instance for JumpReLU SAE."""
    default_sae_config: TrainingSAEConfigDict = {
        "d_in": 64,
        "d_sae": 256,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
        "apply_b_dec_to_input": False,
        "decoder_init_norm": 0.1,
        "jumprelu_sparsity_loss_mode": "step",
        "jumprelu_tanh_scale": 4.0,
        "jumprelu_init_threshold": 0.001,
        "jumprelu_bandwidth": 0.001,
        "l0_coefficient": 0.3,
        "l0_warm_up_steps": 0,
        "pre_act_loss_coefficient": None,
    }
    runner_cfg = _build_runner_config(
        JumpReLUTrainingSAEConfig,
        cast(dict[str, Any], default_sae_config),
        **kwargs,
    )
    _update_sae_metadata(runner_cfg)
    return runner_cfg


def build_jumprelu_sae_cfg(**kwargs: Any) -> JumpReLUSAEConfig:
    default_sae_config: SAEConfigDict = {
        "d_in": 64,
        "d_sae": 256,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
    }
    return JumpReLUSAEConfig(**{**default_sae_config, **kwargs})  # type: ignore


def build_jumprelu_sae_training_cfg(**kwargs: Any) -> JumpReLUTrainingSAEConfig:
    return build_jumprelu_runner_cfg(**kwargs).sae  # type: ignore


# --- Gated SAE Builder ---
def build_gated_runner_cfg(
    **kwargs: Any,
) -> LanguageModelSAERunnerConfig[GatedTrainingSAEConfig]:
    """Helper to create a mock instance for Gated SAE."""
    default_sae_config: TrainingSAEConfigDict = {
        "d_in": 64,
        "d_sae": 256,
        "dtype": "float32",
        "device": "cpu",
        "decoder_init_norm": 0.1,
        "l1_coefficient": 1.0,
        "normalize_activations": "none",
        "apply_b_dec_to_input": False,
        "l1_warm_up_steps": 0,
    }
    runner_cfg = _build_runner_config(
        GatedTrainingSAEConfig,
        cast(dict[str, Any], default_sae_config),
        **kwargs,
    )
    _update_sae_metadata(runner_cfg)
    return runner_cfg


def build_gated_sae_cfg(**kwargs: Any) -> GatedSAEConfig:
    default_sae_config: SAEConfigDict = {
        "d_in": 64,
        "d_sae": 256,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
    }
    return GatedSAEConfig(**{**default_sae_config, **kwargs})  # type: ignore


def build_gated_sae_training_cfg(**kwargs: Any) -> GatedTrainingSAEConfig:
    return build_gated_runner_cfg(**kwargs).sae  # type: ignore


# --- TopK SAE Builder ---
def build_topk_runner_cfg(
    **kwargs: Any,
) -> LanguageModelSAERunnerConfig[TopKTrainingSAEConfig]:
    """Helper to create a mock instance for TopK SAE."""
    default_sae_config: TrainingSAEConfigDict = {
        "d_in": 64,
        "d_sae": 256,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
        "decoder_init_norm": 0.1,
        "apply_b_dec_to_input": False,
        "k": 10,
        "rescale_acts_by_decoder_norm": True,
    }
    # Ensure activation_fn_kwargs has k if k is overridden
    temp_sae_overrides = {
        k: v for k, v in kwargs.items() if k in TrainingSAEConfigDict.__annotations__
    }
    temp_sae_config = {**default_sae_config, **temp_sae_overrides}
    # Update the default config *before* passing it to _build_runner_config
    final_default_sae_config = cast(dict[str, Any], temp_sae_config)

    runner_cfg = _build_runner_config(
        TopKTrainingSAEConfig,
        final_default_sae_config,
        **kwargs,
    )
    _update_sae_metadata(runner_cfg)
    return runner_cfg


def build_topk_sae_cfg(**kwargs: Any) -> TopKSAEConfig:
    default_sae_config: SAEConfigDict = {
        "k": 100,
        "d_in": 64,
        "d_sae": 256,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
    }
    return TopKSAEConfig(**{**default_sae_config, **kwargs})  # type: ignore


def build_topk_sae_training_cfg(**kwargs: Any) -> TopKTrainingSAEConfig:
    return build_topk_runner_cfg(**kwargs).sae  # type: ignore


# --- Matching Pursuit SAE Builder ---


def build_matching_pursuit_runner_cfg(
    **kwargs: Any,
) -> LanguageModelSAERunnerConfig[MatchingPursuitTrainingSAEConfig]:
    """Helper to create a mock instance for Matching Pursuit SAE."""
    default_sae_config: TrainingSAEConfigDict = {
        "d_in": 64,
        "d_sae": 256,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
        "decoder_init_norm": 0.1,
        "apply_b_dec_to_input": True,
        "residual_threshold": 0.1,
    }
    # Ensure activation_fn_kwargs has k if k is overridden
    temp_sae_overrides = {
        k: v for k, v in kwargs.items() if k in TrainingSAEConfigDict.__annotations__
    }
    temp_sae_config = {**default_sae_config, **temp_sae_overrides}
    # Update the default config *before* passing it to _build_runner_config
    final_default_sae_config = cast(dict[str, Any], temp_sae_config)

    runner_cfg = _build_runner_config(
        MatchingPursuitTrainingSAEConfig,
        final_default_sae_config,
        **kwargs,
    )
    _update_sae_metadata(runner_cfg)
    return runner_cfg


def build_matching_pursuit_sae_cfg(**kwargs: Any) -> MatchingPursuitSAEConfig:
    default_sae_config: SAEConfigDict = {
        "residual_threshold": 0.1,
        "d_in": 64,
        "d_sae": 256,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
    }
    return MatchingPursuitSAEConfig(**{**default_sae_config, **kwargs})  # type: ignore


def build_matching_pursuit_sae_training_cfg(
    **kwargs: Any,
) -> MatchingPursuitTrainingSAEConfig:
    return build_matching_pursuit_runner_cfg(**kwargs).sae  # type: ignore


# --- Temporal SAE Builder ---
def build_temporal_sae_cfg(**kwargs: Any) -> TemporalSAEConfig:
    default_sae_config: dict[str, Any] = {
        "d_in": 64,
        "d_sae": 512,
        "dtype": "bfloat16",
        "device": "cpu",
        "normalize_activations": "constant_scalar_rescale",
        "activation_normalization_factor": 0.2,
        "n_heads": 8,
        "n_attn_layers": 1,
        "bottleneck_factor": 2,
        "sae_diff_type": "topk",
        "kval_topk": 32,
        "tied_weights": True,
    }
    return TemporalSAEConfig(**{**default_sae_config, **kwargs})  # type: ignore


# --- BatchTopK SAE Builder ---
def build_batchtopk_runner_cfg(
    **kwargs: Any,
) -> LanguageModelSAERunnerConfig[BatchTopKTrainingSAEConfig]:
    """Helper to create a mock instance for BatchTopK SAE."""
    default_sae_config: TrainingSAEConfigDict = {
        "d_in": 64,
        "d_sae": 256,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
        "decoder_init_norm": 0.1,
        "apply_b_dec_to_input": False,
        "k": 10,
        "topk_threshold_lr": 0.02,
    }
    # Ensure activation_fn_kwargs has k if k is overridden
    temp_sae_overrides = {
        k: v for k, v in kwargs.items() if k in TrainingSAEConfigDict.__annotations__
    }
    temp_sae_config = {**default_sae_config, **temp_sae_overrides}
    # Update the default config *before* passing it to _build_runner_config
    final_default_sae_config = cast(dict[str, Any], temp_sae_config)

    runner_cfg = _build_runner_config(
        BatchTopKTrainingSAEConfig,
        final_default_sae_config,
        **kwargs,
    )
    _update_sae_metadata(runner_cfg)
    return runner_cfg


def build_batchtopk_sae_training_cfg(**kwargs: Any) -> BatchTopKTrainingSAEConfig:
    return build_batchtopk_runner_cfg(**kwargs).sae  # type: ignore


# --- Matryoshka BatchTopK SAE Builder ---
def build_matryoshka_batchtopk_runner_cfg(
    **kwargs: Any,
) -> LanguageModelSAERunnerConfig[MatryoshkaBatchTopKTrainingSAEConfig]:
    """Helper to create a mock instance for Matryoshka BatchTopK SAE."""
    default_sae_config: TrainingSAEConfigDict = {
        "matryoshka_widths": [10, kwargs.get("d_sae", 20)],
        "d_in": 64,
        "d_sae": 256,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
        "decoder_init_norm": 0.1,
        "apply_b_dec_to_input": False,
        "k": 10,
        "topk_threshold_lr": 0.02,
    }
    # Ensure activation_fn_kwargs has k if k is overridden
    temp_sae_overrides = {
        k: v for k, v in kwargs.items() if k in TrainingSAEConfigDict.__annotations__
    }
    temp_sae_config = {**default_sae_config, **temp_sae_overrides}
    # Update the default config *before* passing it to _build_runner_config
    final_default_sae_config = cast(dict[str, Any], temp_sae_config)

    runner_cfg = _build_runner_config(
        MatryoshkaBatchTopKTrainingSAEConfig,
        final_default_sae_config,
        **kwargs,
    )
    _update_sae_metadata(runner_cfg)
    return runner_cfg


def build_matryoshka_batchtopk_sae_training_cfg(
    **kwargs: Any,
) -> MatryoshkaBatchTopKTrainingSAEConfig:
    return build_matryoshka_batchtopk_runner_cfg(**kwargs).sae  # type: ignore


MODEL_CACHE: dict[str, HookedTransformer] = {}


def load_model_cached(model_name: str) -> HookedTransformer:
    """
    helper to avoid unnecessarily loading the same model multiple times.
    NOTE: if the model gets modified in tests this will not work.
    """
    if model_name not in MODEL_CACHE:
        MODEL_CACHE[model_name] = HookedTransformer.from_pretrained(
            model_name, device="cpu"
        )
    # we copy here to prevent sharing state across tests
    return copy.deepcopy(MODEL_CACHE[model_name])


def build_sae_cfg_for_arch(architecture: str, **kwargs: Any) -> SAEConfig:
    builder = SAE_CONFIG_BUILDERS[architecture]
    return builder(**kwargs)


def build_sae_training_cfg_for_arch(
    architecture: str, **kwargs: Any
) -> TrainingSAEConfig:
    builder = SAE_TRAINING_CONFIG_BUILDERS[architecture]
    return builder(**kwargs)


def build_runner_cfg_for_arch(
    architecture: str, **kwargs: Any
) -> LanguageModelSAERunnerConfig[Any]:
    builder = SAE_RUNNER_CONFIG_BUILDERS[architecture]
    return builder(**kwargs)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    allow_subclasses: bool = True,
    atol: float | None = 1e-8,
    rtol: float | None = 1e-5,
    equal_nan: bool = False,
    check_device: bool = True,
    check_dtype: bool = True,
    check_layout: bool = True,
    check_stride: bool = False,
    msg: str | None = None,
) -> None:
    """
    torch.testing.assert_close() with torch.allclose() defaults (atol=1e-8, rtol=1e-5).

    Pass a message string to customize the error header instead of writing lambda functions.
    """
    final_msg = msg and (lambda error_msg: f"{msg}\n\n{error_msg}")

    torch.testing.assert_close(
        actual,
        expected=expected,
        allow_subclasses=allow_subclasses,
        atol=atol,
        rtol=rtol,
        equal_nan=equal_nan,
        check_device=check_device,
        check_dtype=check_dtype,
        check_layout=check_layout,
        check_stride=check_stride,
        msg=final_msg,
    )


def assert_not_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    allow_subclasses: bool = True,
    atol: float | None = 1e-8,
    rtol: float | None = 1e-5,
    equal_nan: bool = False,
    check_device: bool = True,
    check_dtype: bool = True,
    check_layout: bool = True,
    check_stride: bool = False,
    msg: str | None = None,
) -> None:
    """
    Assert that two tensors are NOT close to each other.
    """
    with pytest.raises(AssertionError):
        assert_close(
            actual,
            expected=expected,
            allow_subclasses=allow_subclasses,
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
            check_device=check_device,
            check_dtype=check_dtype,
            check_layout=check_layout,
            check_stride=check_stride,
            msg=msg,
        )


def random_params(model: torch.nn.Module) -> None:
    """
    Fill the parameters of a model with random values.
    """
    for param in model.parameters():
        param.data = torch.rand_like(param)
    for buffer in model.buffers():
        buffer.data = torch.rand_like(buffer)


SAE_TRAINING_CONFIG_BUILDERS = {
    "standard": build_sae_training_cfg,
    "gated": build_gated_sae_training_cfg,
    "jumprelu": build_jumprelu_sae_training_cfg,
    "topk": build_topk_sae_training_cfg,
    "batchtopk": build_batchtopk_sae_training_cfg,
    "matryoshka_batchtopk": build_matryoshka_batchtopk_sae_training_cfg,
    "matching_pursuit": build_matching_pursuit_sae_training_cfg,
}

SAE_CONFIG_BUILDERS = {
    "standard": build_sae_cfg,
    "gated": build_gated_sae_cfg,
    "jumprelu": build_jumprelu_sae_cfg,
    "topk": build_topk_sae_cfg,
    "temporal": build_temporal_sae_cfg,
    "matching_pursuit": build_matching_pursuit_sae_cfg,
}

SAE_RUNNER_CONFIG_BUILDERS = {
    "standard": build_runner_cfg,
    "gated": build_gated_runner_cfg,
    "jumprelu": build_jumprelu_runner_cfg,
    "topk": build_topk_runner_cfg,
    "batchtopk": build_batchtopk_runner_cfg,
    "matryoshka_batchtopk": build_matryoshka_batchtopk_runner_cfg,
    "matching_pursuit": build_matching_pursuit_runner_cfg,
}
