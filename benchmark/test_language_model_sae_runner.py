import torch

from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.saes.gated_sae import GatedTrainingSAEConfig
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig

# os.environ["WANDB_MODE"] = "offline"  # turn this off if you want to see the output


# The way to run this with this command:
# poetry run py.test benchmark/test_language_model_sae_runner.py --profile-svg -s
def test_language_model_sae_runner():
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

    cfg = LanguageModelSAERunnerConfig(
        sae=StandardTrainingSAEConfig(
            d_sae=512,
            d_in=512,
            dtype="float32",
            device=device,
            apply_b_dec_to_input=False,
            l1_coefficient=5,
            lp_norm=1.0,
            l1_warm_up_steps=l1_warmup_steps,
        ),
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        ## MLP Layer 0 ##
        hook_name="blocks.0.hook_mlp_out",
        dataset_path="NeelNanda/c4-tokenized-2b",
        context_size=256,
        is_dataset_tokenized=True,
        prepend_bos=True,  # I used to train GPT2 SAEs with a prepended-bos but no longer think we should do this.
        # How big do we want our SAE to be?
        # Dataset / Activation Store
        # When we do a proper test
        # training_tokens= 820_000_000, # 200k steps * 4096 batch size ~ 820M tokens (doable overnight on an A100)
        # For now.
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size_tokens=4096,
        # Loss Function
        ## Reconstruction Coefficient.
        # Learning Rate
        lr_scheduler_name="constant",  # we set this independently of warmup and decay steps.
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_warm_up_steps,
        ## No ghost grad term.
        # Optimizer
        lr=4e-5,
        ## adam optimizer has no weight decay by default so worry about this.
        adam_beta1=0.9,
        adam_beta2=0.999,
        # Buffer details won't matter in we cache / shuffle our activations ahead of time.
        n_batches_in_buffer=64,
        store_batch_size_prompts=16,
        # Feature Store
        feature_sampling_window=1000,
        dead_feature_window=1000,
        dead_feature_threshold=1e-4,
        # performance enhancement:
        compile_sae=False,
        # WANDB
        logger=LoggingConfig(
            wandb_project="benchmark",
            wandb_log_frequency=100,
        ),
        # Misc
        device=device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype="float32",
    )

    sae = LanguageModelSAETrainingRunner(cfg).run()

    assert sae is not None
    # know whether or not this works by looking at the dashboard!


def test_language_model_sae_runner_gated():
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

    cfg = LanguageModelSAERunnerConfig(
        sae=GatedTrainingSAEConfig(
            d_sae=512,
            d_in=512,
            dtype="float32",
            device=device,
            apply_b_dec_to_input=False,
            l1_coefficient=5,
            l1_warm_up_steps=l1_warmup_steps,
        ),
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        ## MLP Layer 0 ##
        hook_name="blocks.0.hook_mlp_out",
        dataset_path="NeelNanda/c4-tokenized-2b",
        context_size=256,
        is_dataset_tokenized=True,
        prepend_bos=True,  # I used to train GPT2 SAEs with a prepended-bos but no longer think we should do this.
        # When we do a proper test
        # training_tokens= 820_000_000, # 200k steps * 4096 batch size ~ 820M tokens (doable overnight on an A100)
        # For now.
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size_tokens=4096,
        # Loss Function
        ## Reconstruction Coefficient.
        # Learning Rate
        lr_scheduler_name="constant",  # we set this independently of warmup and decay steps.
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_warm_up_steps,
        ## No ghost grad term.
        # Optimizer
        lr=4e-5,
        ## adam optimizer has no weight decay by default so worry about this.
        adam_beta1=0.9,
        adam_beta2=0.999,
        # Buffer details won't matter in we cache / shuffle our activations ahead of time.
        n_batches_in_buffer=64,
        store_batch_size_prompts=16,
        # Feature Store
        feature_sampling_window=1000,
        dead_feature_window=1000,
        dead_feature_threshold=1e-4,
        # performance enhancement:
        compile_sae=False,
        # WANDB
        logger=LoggingConfig(
            wandb_project="benchmark",
            wandb_log_frequency=100,
        ),
        # Misc
        device=device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype="float32",
    )

    sae = LanguageModelSAETrainingRunner(cfg).run()

    assert sae is not None
    # know whether or not this works by looking at the dashboard!


def test_language_model_sae_runner_top_k():
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

    cfg = LanguageModelSAERunnerConfig(
        sae=TopKTrainingSAEConfig(
            k=32,
            d_sae=512,
            d_in=512,
            dtype="float32",
            device=device,
            apply_b_dec_to_input=True,
        ),
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        ## MLP Layer 0 ##
        hook_name="blocks.0.hook_mlp_out",
        dataset_path="NeelNanda/c4-tokenized-2b",
        context_size=256,
        is_dataset_tokenized=True,
        prepend_bos=True,  # I used to train GPT2 SAEs with a prepended-bos but no longer think we should do this.
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size_tokens=4096,
        # Learning Rate
        lr_scheduler_name="constant",  # we set this independently of warmup and decay steps.
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_warm_up_steps,
        # Optimizer
        lr=4e-5,
        ## adam optimizer has no weight decay by default so worry about this.
        adam_beta1=0.9,
        adam_beta2=0.999,
        # Buffer details won't matter in we cache / shuffle our activations ahead of time.
        n_batches_in_buffer=64,
        store_batch_size_prompts=16,
        # Feature Store
        feature_sampling_window=1000,
        dead_feature_window=1000,
        dead_feature_threshold=1e-4,
        # performance enhancement:
        compile_sae=False,
        # WANDB
        logger=LoggingConfig(
            wandb_project="benchmark",
            wandb_log_frequency=100,
        ),
        # Misc
        device=device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype="float32",
    )

    sae = LanguageModelSAETrainingRunner(cfg).run()

    assert sae is not None
    # know whether or not this works by looking at the dashboard!


def test_language_model_sae_runner_othellogpt():
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

    lr_warm_up_steps = 0
    lr_decay_steps = 40_000
    l1_warmup_steps = 10_000

    cfg = LanguageModelSAERunnerConfig(
        sae=StandardTrainingSAEConfig(
            d_sae=512,
            d_in=512,
            dtype="float32",
            device=device,
            apply_b_dec_to_input=True,
            l1_coefficient=0.001,
            lp_norm=1.0,
            l1_warm_up_steps=l1_warmup_steps,
            normalize_activations="expected_average_only_in",
        ),
        # Data Generating Function (Model + Training Distibuion)
        model_name="othello-gpt",  # othello-gpt model
        hook_name="blocks.6.hook_resid_pre",  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
        dataset_path="taufeeque/othellogpt",  # this is a tokenized language dataset on Huggingface for OthelloGPT games.
        is_dataset_tokenized=True,
        streaming=True,  # we could pre-download the token dataset if it was small.
        # Training Parameters
        lr=0.00003,  # lower the better, we'll go fairly high to speed up the tutorial.
        adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
        adam_beta2=0.999,
        lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
        lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
        lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
        train_batch_size_tokens=batch_size,
        context_size=59,  # will control the length of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
        seqpos_slice=(5, -5),
        # Activation Store Parameters
        n_batches_in_buffer=32,  # controls how many activations we store / shuffle.
        training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
        store_batch_size_prompts=32,
        # Resampling protocol
        feature_sampling_window=500,  # this controls our reporting of feature sparsity stats
        dead_feature_window=1000000,  # would effect resampling or ghost grads if we were using it.
        dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
        # WANDB
        logger=LoggingConfig(
            log_to_wandb=False,  # always use wandb unless you are just testing code.
            wandb_project="benchmark",
            wandb_log_frequency=100,
        ),
        # Misc
        device=device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype="torch.float32",
    )

    sae = LanguageModelSAETrainingRunner(cfg).run()

    assert sae is not None
    # know whether or not this works by looking at the dashboard!
