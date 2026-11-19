import torch

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner

# os.environ["WANDB_MODE"] = "offline"  # turn this off if you want to see the output

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# total_training_steps = 20_000
total_training_steps = 500
batch_size = 4096
total_training_tokens = total_training_steps * batch_size
print(f"Total Training Tokens: {total_training_tokens}")

lr_warm_up_steps = 0
lr_decay_steps = 40_000
print(f"lr_decay_steps: {lr_decay_steps}")
l1_warmup_steps = 10_000
print(f"l1_warmup_steps: {l1_warmup_steps}")


BASE_CFG = dict(
    # Pick a tiny model to make this easier.
    model_name="gelu-1l",
    ## MLP Layer 0 ##
    hook_name="blocks.0.hook_mlp_out",
    d_in=512,
    dataset_path="NeelNanda/c4-tokenized-2b",
    context_size=256,
    is_dataset_tokenized=True,
    prepend_bos=True,  # I used to train GPT2 SAEs with a prepended-bos but no longer think we should do this.
    # How big do we want our SAE to be?
    expansion_factor=16,
    # Dataset / Activation Store
    # When we do a proper test
    # training_tokens= 820_000_000, # 200k steps * 4096 batch size ~ 820M tokens (doable overnight on an A100)
    # For now.
    training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
    train_batch_size_tokens=4096,
    # Loss Function
    ## Reconstruction Coefficient.
    ## Anthropic does not mention using an Lp norm other than L1.
    l1_coefficient=5,
    lp_norm=1.0,
    # Instead, they multiply the L1 loss contribution
    # from each feature of the activations by the decoder norm of the corresponding feature.
    scale_sparsity_penalty_by_decoder_norm=True,
    # Learning Rate
    lr_scheduler_name="constant",  # we set this independently of warmup and decay steps.
    l1_warm_up_steps=l1_warmup_steps,
    lr_warm_up_steps=lr_warm_up_steps,
    lr_decay_steps=lr_warm_up_steps,
    ## No ghost grad term.
    use_ghost_grads=False,
    # Initialization / Architecture
    apply_b_dec_to_input=False,
    # encoder bias zero's. (I'm not sure what it is by default now)
    normalize_sae_decoder=False,
    decoder_heuristic_init=True,
    init_encoder_as_decoder_transpose=True,
    # Optimizer
    lr=4e-5,
    ## adam optimizer has no weight decay by default so worry about this.
    adam_beta1=0.9,
    adam_beta2=0.999,
    # Buffer details won't matter in we cache / shuffle our activations ahead of time.
    n_sequences_in_buffer=16,
    store_batch_size_prompts=4,
    normalize_activations="constant_norm_rescale",
    n_eval_batches=3,
    eval_batch_size_prompts=4,
    # Feature Store
    feature_sampling_window=1000,
    dead_feature_window=1000,
    dead_feature_threshold=1e-4,
    # performance enhancement:
    compile_sae=False,
    # WANDB
    log_to_wandb=True,  # always use wandb unless you are just testing code.
    wandb_project="benchmark",
    wandb_log_frequency=100,
    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32",
)


# The way to run this with this command:
def test_sae_runner_multiple_devices():
    cfg_dict = BASE_CFG
    cfg_dict["model_name"] = "gemma-2b"
    cfg_dict["d_in"] = 2048
    cfg_dict["device"] = "cuda:3"
    cfg_dict["model_from_pretrained_kwargs"] = {  # type: ignore
        "n_devices": torch.cuda.device_count() - 1
    }
    cfg_dict["act_store_device"] = "cpu"
    cfg_dict["dtype"] = "torch.bfloat16"
    cfg_dict["eval_every_n_wandb_logs"] = 3
    cfg = LanguageModelSAERunnerConfig(**cfg_dict)  # type: ignore

    # look at the next cell to see some instruction for what to do while this is running.
    sae = LanguageModelSAETrainingRunner(cfg).run()

    assert sae is not None
    # know whether or not this works by looking at the dashboard!


# genuinely a bit faster!
def test_sae_runner_multiple_devices_sae_act_store_on_gpus():
    cfg_dict = BASE_CFG
    cfg_dict["model_name"] = "gemma-2b"
    cfg_dict["d_in"] = 2048
    cfg_dict["device"] = "cuda:2"
    cfg_dict["model_from_pretrained_kwargs"] = {  # type: ignore
        "n_devices": torch.cuda.device_count() - 2
    }
    cfg_dict["act_store_device"] = "cuda:3"
    cfg_dict["dtype"] = "torch.bfloat16"
    cfg_dict["eval_every_n_wandb_logs"] = 3
    cfg = LanguageModelSAERunnerConfig(**cfg_dict)  # type: ignore

    sae = LanguageModelSAETrainingRunner(cfg).run()

    assert sae is not None
    # know whether or not this works by looking at the dashboard!
