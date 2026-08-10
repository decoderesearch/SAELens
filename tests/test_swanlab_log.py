from __future__ import annotations

import os
from pathlib import Path

import pytest

from sae_lens.config import LoggingConfig
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.wandb_compat import BACKEND
from tests.helpers import (
    NEEL_NANDA_C4_10K_DATASET,
    TINYSTORIES_MODEL,
    build_runner_cfg_for_arch,
)


def run_minimal_standard_sae_training(output_dir: Path) -> None:
    cfg = build_runner_cfg_for_arch(
        architecture="standard",
        d_in=64,
        d_sae=128,
        training_tokens=64,
        store_batch_size_prompts=2,
        train_batch_size_tokens=4,
        context_size=10,
        n_batches_in_buffer=2,
        n_eval_batches=1,
        dataset_path=NEEL_NANDA_C4_10K_DATASET,
        hook_name="blocks.0.hook_resid_post",
        model_name=TINYSTORIES_MODEL,
        n_checkpoints=0,
        save_final_checkpoint=False,
        output_path=str(output_dir),
        logger=LoggingConfig(
            log_to_wandb=True,
            wandb_project=os.getenv("WANDB_PROJECT", "swanlab-minimal"),
            wandb_entity=os.getenv("WANDB_ENTITY"),
            wandb_log_frequency=1,
            eval_every_n_wandb_logs=1,
        ),
    )

    LanguageModelSAETrainingRunner(cfg).run()


def test_swanlab_logging_runs(tmp_path: Path) -> None:
    if BACKEND != "swanlab":
        pytest.skip("swanlab backend not active; set SAE_LENS_LOGGING_BACKEND=swanlab")
    os.environ.setdefault("SAE_LENS_LOGGING_BACKEND", "swanlab")
    os.environ.setdefault("SWANLAB_MODE", "offline")
    os.environ.setdefault("WANDB_MODE", "offline")
    run_minimal_standard_sae_training(tmp_path / "swanlab_output")


def main() -> None:
    os.environ.setdefault("SAE_LENS_LOGGING_BACKEND", "swanlab")
    os.environ.setdefault("SWANLAB_MODE", "offline")
    os.environ.setdefault("WANDB_MODE", "offline")
    output_dir = Path(os.getenv("SAE_OUTPUT_DIR", "swanlab_minimal_output"))
    run_minimal_standard_sae_training(output_dir)


if __name__ == "__main__":
    main()
