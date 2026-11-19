import os
import sys

import torch

from sae_lens.saes.standard_sae import StandardTrainingSAEConfig

sys.path.append("..")

from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

total_training_steps = 2000
batch_size = 4096
total_training_tokens = total_training_steps * batch_size
print(f"Total Training Tokens: {total_training_tokens}")

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training steps.
print(f"lr_decay_steps: {lr_decay_steps}")
l1_warmup_steps = total_training_steps // 20  # 5% of training steps.
print(f"l1_warmup_steps: {l1_warmup_steps}")
log_to_wandb = True
l1_coefficient = 5

for block in [11, 0]:
    for autocast_lm in [False, True]:
        cfg = LanguageModelSAERunnerConfig(
            sae=StandardTrainingSAEConfig(
                d_in=768,
                d_sae=768 * 64,
                l1_coefficient=l1_coefficient,
                l1_warm_up_steps=l1_warmup_steps,
                apply_b_dec_to_input=False,
                normalize_activations="expected_average_only_in",
            ),
            # Pick a tiny model to make this easier.
            model_name="gpt2",
            ## MLP ##
            hook_name=f"blocks.{block}.hook_mlp_out",
            dataset_path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
            streaming=True,
            context_size=512,
            is_dataset_tokenized=True,
            prepend_bos=True,
            # Dataset / Activation Store
            use_cached_activations=False,
            training_tokens=total_training_tokens,
            train_batch_size_tokens=4096,
            # Loss Function
            ## Reconstruction Coefficient.
            # Learning Rate
            lr_scheduler_name="constant",  # we set this independently of warmup and decay steps.
            lr_warm_up_steps=lr_warm_up_steps,
            lr_decay_steps=lr_warm_up_steps,
            # Optimizer
            lr=1e-4,
            ## adam optimizer has no weight decay by default so worry about this.
            adam_beta1=0.9,
            adam_beta2=0.999,
            # Unsure if this is enough
            n_sequences_in_buffer=64,
            store_batch_size_prompts=16,
            # Feature Store
            feature_sampling_window=1000,
            dead_feature_window=1000,
            dead_feature_threshold=1e-4,
            # WANDB
            logger=LoggingConfig(
                log_to_wandb=log_to_wandb,
                wandb_project="gpt-2-sweep-20may24-check-autocast-2",
                wandb_log_frequency=50,
                eval_every_n_wandb_logs=10,
            ),
            # Misc
            device=device,
            seed=42,
            n_checkpoints=0,
            checkpoint_path="checkpoints",
            dtype="float32",
            eval_batch_size_prompts=2,
            n_eval_batches=40,
            autocast=True,
            autocast_lm=autocast_lm,
            compile_llm=True,
            compile_sae=True,
        )

        LanguageModelSAETrainingRunner(cfg).run()

        print("=" * 50)
