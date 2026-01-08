# ruff: noqa: E402
__version__ = "6.27.3"

import logging

logger = logging.getLogger(__name__)

from sae_lens.saes import (
    SAE,
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
    GatedSAE,
    GatedSAEConfig,
    GatedTrainingSAE,
    GatedTrainingSAEConfig,
    JumpReLUSAE,
    JumpReLUSAEConfig,
    JumpReLUSkipTranscoder,
    JumpReLUSkipTranscoderConfig,
    JumpReLUTrainingSAE,
    JumpReLUTrainingSAEConfig,
    JumpReLUTranscoder,
    JumpReLUTranscoderConfig,
    MatchingPursuitSAE,
    MatchingPursuitSAEConfig,
    MatchingPursuitTrainingSAE,
    MatchingPursuitTrainingSAEConfig,
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
    SAEConfig,
    SkipTranscoder,
    SkipTranscoderConfig,
    StandardSAE,
    StandardSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
    TemporalSAE,
    TemporalSAEConfig,
    TopKSAE,
    TopKSAEConfig,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
    TrainingSAE,
    TrainingSAEConfig,
    Transcoder,
    TranscoderConfig,
)

from .analysis.hooked_sae_transformer import HookedSAETransformer
from .cache_activations_runner import CacheActivationsRunner
from .config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
    LoggingConfig,
    PretokenizeRunnerConfig,
)
from .evals import run_evals
from .llm_sae_training_runner import LanguageModelSAETrainingRunner, SAETrainingRunner
from .loading.pretrained_sae_loaders import (
    PretrainedSaeDiskLoader,
    PretrainedSaeHuggingfaceLoader,
)
from .pretokenize_runner import PretokenizeRunner, pretokenize_runner
from .registry import register_sae_class, register_sae_training_class
from .training.activations_store import ActivationsStore
from .training.sae_trainer import SAETrainer
from .training.upload_saes_to_huggingface import upload_saes_to_huggingface

__all__ = [
    "SAE",
    "SAEConfig",
    "TrainingSAE",
    "TrainingSAEConfig",
    "HookedSAETransformer",
    "ActivationsStore",
    "LanguageModelSAERunnerConfig",
    "LanguageModelSAETrainingRunner",
    "CacheActivationsRunnerConfig",
    "CacheActivationsRunner",
    "PretokenizeRunnerConfig",
    "PretokenizeRunner",
    "pretokenize_runner",
    "run_evals",
    "upload_saes_to_huggingface",
    "PretrainedSaeHuggingfaceLoader",
    "PretrainedSaeDiskLoader",
    "register_sae_class",
    "register_sae_training_class",
    "StandardSAE",
    "StandardSAEConfig",
    "StandardTrainingSAE",
    "StandardTrainingSAEConfig",
    "GatedSAE",
    "GatedSAEConfig",
    "GatedTrainingSAE",
    "GatedTrainingSAEConfig",
    "TopKSAE",
    "TopKSAEConfig",
    "TopKTrainingSAE",
    "TopKTrainingSAEConfig",
    "JumpReLUSAE",
    "JumpReLUSAEConfig",
    "JumpReLUTrainingSAE",
    "JumpReLUTrainingSAEConfig",
    "SAETrainingRunner",
    "SAETrainer",
    "LoggingConfig",
    "BatchTopKTrainingSAE",
    "BatchTopKTrainingSAEConfig",
    "Transcoder",
    "TranscoderConfig",
    "SkipTranscoder",
    "SkipTranscoderConfig",
    "JumpReLUTranscoder",
    "JumpReLUTranscoderConfig",
    "JumpReLUSkipTranscoder",
    "JumpReLUSkipTranscoderConfig",
    "MatryoshkaBatchTopKTrainingSAE",
    "MatryoshkaBatchTopKTrainingSAEConfig",
    "TemporalSAE",
    "TemporalSAEConfig",
    "MatchingPursuitSAE",
    "MatchingPursuitTrainingSAE",
    "MatchingPursuitSAEConfig",
    "MatchingPursuitTrainingSAEConfig",
]


register_sae_class("standard", StandardSAE, StandardSAEConfig)
register_sae_training_class("standard", StandardTrainingSAE, StandardTrainingSAEConfig)
register_sae_class("gated", GatedSAE, GatedSAEConfig)
register_sae_training_class("gated", GatedTrainingSAE, GatedTrainingSAEConfig)
register_sae_class("topk", TopKSAE, TopKSAEConfig)
register_sae_training_class("topk", TopKTrainingSAE, TopKTrainingSAEConfig)
register_sae_class("jumprelu", JumpReLUSAE, JumpReLUSAEConfig)
register_sae_training_class("jumprelu", JumpReLUTrainingSAE, JumpReLUTrainingSAEConfig)
register_sae_training_class(
    "batchtopk", BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
)
register_sae_training_class(
    "matryoshka_batchtopk",
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)
register_sae_class("transcoder", Transcoder, TranscoderConfig)
register_sae_class("skip_transcoder", SkipTranscoder, SkipTranscoderConfig)
register_sae_class("jumprelu_transcoder", JumpReLUTranscoder, JumpReLUTranscoderConfig)
register_sae_class(
    "jumprelu_skip_transcoder", JumpReLUSkipTranscoder, JumpReLUSkipTranscoderConfig
)
register_sae_class("temporal", TemporalSAE, TemporalSAEConfig)
register_sae_class("matching_pursuit", MatchingPursuitSAE, MatchingPursuitSAEConfig)
register_sae_training_class(
    "matching_pursuit", MatchingPursuitTrainingSAE, MatchingPursuitTrainingSAEConfig
)
