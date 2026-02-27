import os

import torch

from sae_lens import (
    SAE,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    upload_saes_to_huggingface,
)
from sae_lens.config import LoggingConfig
from sae_lens.saes.gated_sae import GatedTrainingSAEConfig

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


model_name = "othello-gpt"
model = HookedSAETransformer.from_pretrained(model_name)

dataset_path = "taufeeque/othellogpt"
context_size = 59

layer = 5
training_tokens = int(1e3)
train_batch_size_tokens = 2048
n_steps = int(training_tokens / train_batch_size_tokens)

assert model.cfg.d_mlp is not None, "Model must have d_mlp defined"

runner_cfg = LanguageModelSAERunnerConfig(
    sae=GatedTrainingSAEConfig(
        d_in=model.cfg.d_mlp,
        d_sae=model.cfg.d_mlp * 8,
        apply_b_dec_to_input=True,
        l1_coefficient=5,
        l1_warm_up_steps=int(0.2 * n_steps),
    ),
    #
    # Data generation
    model_name=model_name,
    hook_name=f"blocks.{layer}.mlp.hook_post",
    dataset_path=dataset_path,
    is_dataset_tokenized=True,
    prepend_bos=False,
    streaming=True,
    train_batch_size_tokens=train_batch_size_tokens,
    context_size=context_size,
    seqpos_slice=(5, -5),
    #
    # Activations store
    n_batches_in_buffer=32,
    store_batch_size_prompts=16,
    training_tokens=training_tokens,
    #
    # Training hyperparameters (standard)
    lr=2e-4,
    adam_beta1=0.9,
    adam_beta2=0.999,
    lr_scheduler_name="constant",
    lr_warm_up_steps=int(0.2 * n_steps),
    lr_decay_steps=int(0.2 * n_steps),
    #
    feature_sampling_window=1000,
    dead_feature_window=500,
    dead_feature_threshold=1e-5,
    #
    # Logging / evals
    logger=LoggingConfig(
        wandb_project=f"othello_gpt_sae_{layer=}",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=10,
    ),
    checkpoint_path="checkpoints",
    #
    # Misc.
    device=str(device),
    seed=42,
    n_checkpoints=5,
    dtype="float32",
)

# t.set_grad_enabled(True)
runner = LanguageModelSAETrainingRunner(runner_cfg)
sae = runner.run()

hf_repo_id = "callummcdougall/arena-demos-othellogpt"
sae_id = "blocks.5.mlp.hook_post-v1"

upload_saes_to_huggingface({sae_id: sae}, hf_repo_id=hf_repo_id)

othellogpt_sae = SAE.from_pretrained(
    release=hf_repo_id, sae_id=sae_id, device=str(device)
)
