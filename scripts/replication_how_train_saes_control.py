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

total_training_steps = 200_000
batch_size = 4096
total_training_tokens = total_training_steps * batch_size
print(f"Total Training Tokens: {total_training_tokens}")

# change these configs
model_name = "gelu-1l"
dataset_path = "NeelNanda/c4-tokenized-2b"
new_cached_activations_path = (
    f"./cached_activations/{model_name}/{dataset_path}/{total_training_steps}"
)

lr_warm_up_steps = total_training_steps // 20
lr_decay_steps = total_training_steps // 5  # 20% of training steps.
print(f"lr_decay_steps: {lr_decay_steps}")
l1_warm_up_steps = 0  # total_training_steps // 20  # 5% of training steps.
print(f"l1_warm_up_steps: {l1_warm_up_steps}")
log_to_wandb = True

for l1_coefficient in [2, 5, 10]:
    cfg = LanguageModelSAERunnerConfig(
        sae=StandardTrainingSAEConfig(
            d_in=512,
            d_sae=512 * 64,
            l1_coefficient=l1_coefficient,
            l1_warm_up_steps=l1_warm_up_steps,
            apply_b_dec_to_input=False,
        ),
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        ## MLP Layer 0 ##
        hook_name="blocks.0.hook_mlp_out",
        dataset_path="NeelNanda/c4-tokenized-2b",
        streaming=False,
        context_size=1024,
        is_dataset_tokenized=True,
        prepend_bos=True,
        # How big do we want our SAE to be?
        # Dataset / Activation Store
        # When we do a proper test
        # training_tokens= 820_000_000, # 200k steps * 4096 batch size ~ 820M tokens (doable overnight on an A100)
        # For now.
        use_cached_activations=True,
        cached_activations_path="/home/paperspace/shared_volumes/activations_volume_1/gelu-1l",
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size_tokens=4096,
        # Loss Function
        ## Reconstruction Coefficient.
        # Instead, they multiply the L1 loss contribution
        # Learning Rate
        lr_scheduler_name="constant",  # we set this independently of warmup and decay steps.
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_warm_up_steps,
        # Optimizer
        lr=3e-4,
        ## adam optimizer has no weight decay by default so worry about this.
        adam_beta1=0.9,
        adam_beta2=0.999,
        # Buffer details won't matter in we cache / shuffle our activations ahead of time.
        n_sequences_in_buffer=64,
        store_batch_size_prompts=16,
        # Feature Store
        feature_sampling_window=1000,
        dead_feature_window=1000,
        dead_feature_threshold=1e-4,
        # WANDB
        logger=LoggingConfig(
            log_to_wandb=log_to_wandb,  # always use wandb unless you are just testing code.
            wandb_project="how_we_train_SAEs_replication_1",
            wandb_log_frequency=50,
            eval_every_n_wandb_logs=10,
        ),
        # Misc
        device=device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype="float32",
    )

    # look at the next cell to see some instruction for what to do while this is running.
    sae = LanguageModelSAETrainingRunner(cfg).run()

    print("=" * 50)
