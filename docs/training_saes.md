# Training Sparse Autoencoders

Methods development for training SAEs is rapidly evolving, so these docs may change frequently. For all available training options, see the [LanguageModelSAERunnerConfig][sae_lens.LanguageModelSAERunnerConfig] and the architecture-specific configuration classes it uses (e.g., [StandardTrainingSAEConfig][sae_lens.StandardTrainingSAEConfig], [GatedTrainingSAEConfig][sae_lens.GatedTrainingSAEConfig], [JumpReLUTrainingSAEConfig][sae_lens.JumpReLUTrainingSAEConfig], and [TopKTrainingSAEConfig][sae_lens.TopKTrainingSAEConfig]).

However, we are attempting to maintain this [tutorial](https://github.com/decoderesearch/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/decoderesearch/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb).

We encourage readers to join the [Open Source Mechanistic Interpretability Slack](https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-375zalm04-GFd5tdBU1yLKlu_T_JSqZQ) for support!

## Basic training setup

Training a SAE is done using the [LanguageModelSAETrainingRunner][sae_lens.LanguageModelSAETrainingRunner] class. This class is configured using a [LanguageModelSAERunnerConfig][sae_lens.LanguageModelSAERunnerConfig]. The `LanguageModelSAERunnerConfig` holds parameters for the overall training run (like model, dataset, and learning rate), and it contains an `sae` field. This `sae` field should be an instance of an architecture-specific SAE configuration dataclass (e.g., `StandardTrainingSAEConfig` for standard SAEs, `TopKTrainingSAEConfig` for TopK SAEs, etc.), which holds parameters specific to the SAE's structure and sparsity mechanisms.

When using the command-line interface (CLI), you typically specify an `--architecture` argument (e.g., `"batchtopk"`, `"jumprelu"`, `"standard"`, `"topk"`), and the runner constructs the appropriate nested SAE configuration. When instantiating `LanguageModelSAERunnerConfig` programmatically, you should directly provide the configured SAE object to the `sae` field. The CLI can be run using `python -m sae_lens.llm_sae_training_runner`.

Some of the core config options available in `LanguageModelSAERunnerConfig` are:

- `model_name`: The base model name to train a SAE on (e.g., `"gpt2-small"`, `"tiny-stories-1L-21M"`). This must correspond to a [model from TransformerLens](https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html) or a Hugging Face `AutoModelForCausalLM` if `model_class_name` is set accordingly.
- `hook_name`: This is a TransformerLens hook in the model where our SAE will be trained from (e.g., `"blocks.0.hook_mlp_out"`). More info on hooks can be found [here](https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points).
- `dataset_path`: The path to a dataset on Huggingface for training (e.g., `"apollo-research/roneneldan-TinyStories-tokenizer-gpt2"`).
- `training_tokens`: The total number of tokens from the dataset to use for training the SAE.
- `train_batch_size_tokens`: The batch size used for training the SAE, measured in tokens. Adjust this to keep the GPU saturated.
- `model_from_pretrained_kwargs`: A dictionary of keyword arguments to pass to `HookedTransformer.from_pretrained` when loading the model. It's often best to set `"center_writing_weights": False`.
- `lr`: The learning rate for the optimizer.
- `context_size`: The sequence length of prompts fed to the model to generate activations.

Core options typically configured within the architecture-specific `sae` object (e.g., `cfg.sae = StandardTrainingSAEConfig(...)`):

- `d_in`: The input dimensionality of the SAE. This must match the size of the activations at `hook_name`.
- `d_sae`: The SAE's hidden layer dimensionality .
- Sparsity control parameters: These vary by architecture:
  - For Standard SAEs: `l1_coefficient` (controls L1 penalty), `lp_norm` (e.g., 1.0 for L1, 0.7 for L0.7), `l1_warm_up_steps`.
  - For Gated SAEs: `l1_coefficient` (controls L1-like penalty on gate activations), `l1_warm_up_steps`.
  - For JumpReLU SAEs: `l0_coefficient` (controls L0-like penalty), `l0_warm_up_steps`, `jumprelu_init_threshold`, `jumprelu_bandwidth`.
  - For TopK and BatchTopK SAEs: `k` (the number of features to keep active). Sparsity is enforced structurally.
- `normalize_activations`: Strategy for normalizing activations before they enter the SAE (e.g., `"expected_average_only_in"`).

A sample training run from the [tutorial](https://github.com/decoderesearch/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb) is shown below. Note how SAE-specific parameters are nested within the `sae` field:

```python
import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
)

# Define total training steps and batch size
total_training_steps = 30_000
batch_size = 4096
total_training_tokens = total_training_steps * batch_size

# Learning rate and L1 warmup schedules
lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distribution)
    model_name="tiny-stories-1L-21M",
    hook_name="blocks.0.hook_mlp_out",
    dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
    is_dataset_tokenized=True,
    streaming=True,

    # SAE Parameters are in the nested 'sae' config
    sae=StandardTrainingSAEConfig(
        d_in=1024, # Matches hook_mlp_out for tiny-stories-1L-21M
        d_sae=16 * 1024,
        apply_b_dec_to_input=True,
        normalize_activations="expected_average_only_in",
        l1_coefficient=5,
        l1_warm_up_steps=l1_warm_up_steps,
    ),

    # Training Parameters
    lr=5e-5,
    lr_warm_up_steps=lr_warm_up_steps,
    lr_decay_steps=lr_decay_steps,
    train_batch_size_tokens=batch_size,

    # Activation Store Parameters
    context_size=256,
    n_batches_in_buffer=64,
    training_tokens=total_training_tokens,
    store_batch_size_prompts=16,

    # WANDB
    logger=LoggingConfig(
        log_to_wandb=True,
        wandb_project="sae_lens_tutorial",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
    ),

    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32"
)
sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()
```

As you can see, the training setup provides a large number of options to explore. The full list of options can be found by inspecting the `LanguageModelSAERunnerConfig` class and the specific SAE configuration class you intend to use (e.g., `StandardTrainingSAEConfig`, `TopKTrainingSAEConfig`, etc.).

### Training BatchTopk SAEs

<!-- prettier-ignore-start -->
!!! tip "SOTA architecture"
    BatchTopK is a state-of-the-art SAE architecture and is easy to train.
<!-- prettier-ignore-end -->

The [BatchTopK](https://arxiv.org/abs/2412.06410) architecture is a more modern version of the TopK architecture, which fixes the mean L0 across a training batch rather than fixing the L0 for every sample in the batch. To train a BatchTopK SAE, provide a `BatchTopKTrainingSAEConfig` instance to the `sae` field. The primary parameter for TopK SAEs is `k`, the number of features to keep active. If not set, `k` defaults to 100 in `BatchTopKTrainingSAEConfig`. Like the TopK architecture, the BatchTopK architecture does not use an `l1_coefficient` or `lp_norm` for sparsity, as sparsity is structurally enforced.

Also worth noting is that `BatchTopK` SAEs will save as `JumpReLU` SAEs for use at inference. This is to avoid needing to provide a batch of inputs at inference time, allowing the SAE to work consistently on any batch size after training is complete.

```python
from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner, BatchTopKTrainingSAEConfig

cfg = LanguageModelSAERunnerConfig( # Full config would be defined here
    # ... other LanguageModelSAERunnerConfig parameters ...
    sae=BatchTopKTrainingSAEConfig(
        k=100, # Set the number of active features
        d_in=1024, # Must match your hook point
        d_sae=16 * 1024,
        # ... other common SAE parameters from SAEConfig if needed ...
    ),
    # ...
)
sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()
```

### Training JumpReLU SAEs

<!-- prettier-ignore-start -->
!!! tip "SOTA architecture"
    JumpReLU is a state-of-the-art SAE architecture, potentially even superior to BatchTopK. However, JumpReLU may be trickier to train. JumpReLU is used for the [Gemma Scope SAEs](https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/) and by [Anthropic](https://transformer-circuits.pub/2025/january-update/index.html).
<!-- prettier-ignore-end -->

[JumpReLU SAEs](https://arxiv.org/abs/2407.14435) are a state-of-the-art SAE architecture. To train one, provide a `JumpReLUTrainingSAEConfig` to the `sae` field. JumpReLU SAEs use a sparsity penalty controlled by the `l0_coefficient` parameter. The `JumpReLUTrainingSAEConfig` also has parameters `jumprelu_bandwidth` and `jumprelu_init_threshold` which affect the learning of the thresholds.

We support both the original JumpReLU sparsity loss and the more modern [tanh sparsity loss](https://transformer-circuits.pub/2025/january-update/index.html) variant from Anthropic. To use the tanh sparsity loss, set `jumprelu_sparsity_loss_mode="tanh"`. The tanh sparsity loss variant is a bit easier to train, but has more hyperparameters. We recommend using the tanh with `normalize_activations="expected_average_only_in"` to match Anthropic's setup. We also recommend enabling the pre-act loss by setting `pre_act_loss_coefficient` to match Anthropic's setup. An example of this is below:

```python
from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner, JumpReLUTrainingSAEConfig

cfg = LanguageModelSAERunnerConfig( # Full config would be defined here
    # ... other LanguageModelSAERunnerConfig parameters ...
    sae=JumpReLUTrainingSAEConfig(
        l0_coefficient=5.0, # Sparsity penalty coefficient
        jumprelu_sparsity_loss_mode="tanh",
        jumprelu_tanh_scale=4.0, # default value
        jumprelu_bandwidth=2.0,
        jumprelu_init_threshold=0.1,
        pre_act_loss_coefficient=3e-6,
        # Anthropic's settings assume normalized activations
        normalize_activations="expected_average_only_in",
        # Anthropic recommends using the full training steps for the warm-up
        l0_warm_up_steps=total_training_steps,
        d_in=1024, # must match your hook point
        d_sae=16 * 1024,
        # ... other common SAE parameters from SAEConfig ...
    ),
    # Anthropic recommends decaying the LR for the final 20% of training
    lr_decay_steps=total_training_steps // 5,
    # ...
)
sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()
```

If you'd like to use the original JumpReLU sparsity loss from DeepMind, set `jumprelu_sparsity_loss_mode="step"`. This requires a bit more tuning to work compared with the Anthropic tanh variant. We find this setup requires training on a large number of tokens to work well, ideally 2 billion or more. If you don't see L0 decreasing with this setup by the end of training, try increasing the `jumprelu_bandwidth` and possibly also the `jumprelu_init_threshold`.

```python
from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner, JumpReLUTrainingSAEConfig

cfg = LanguageModelSAERunnerConfig( # Full config would be defined here
    # ... other LanguageModelSAERunnerConfig parameters ...
    sae=JumpReLUTrainingSAEConfig(
        l0_coefficient=5.0, # Sparsity penalty coefficient
        jumprelu_bandwidth=0.01,
        jumprelu_init_threshold=0.01,
        d_in=1024, # must match your hook point
        d_sae=16 * 1024,
        normalize_activations="expected_average_only_in",
        # ... other common SAE parameters from SAEConfig ...
    ),
    # ...
)
sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()
```

### Training Standard L1 SAEs

<!-- prettier-ignore-start -->
!!! warning "Warning: legacy architecture"
    Standard L1 SAEs are not considered state-of-the-art, but they are the classic SAE architecture. They are a good starting point for understanding SAEs and are easy to train. For better performance, try BatchTopK or JumpReLU.
<!-- prettier-ignore-end -->

The classic SAE architecture is the Standard L1 SAE, which uses a L1 loss term with ReLU activation. To train a Standard L1 SAE, provide a `StandardTrainingSAEConfig` instance to the `sae` field. The Standard L1 SAE uses the `l1_coefficient` parameter to control the sparsity of the SAE.

```python

cfg = LanguageModelSAERunnerConfig( # Full config would be defined here
    # ... other LanguageModelSAERunnerConfig parameters ...
    sae=StandardTrainingSAEConfig(
        l1_coefficient=5.0, # Sparsity penalty coefficient
        d_in=1024, # Must match your hook point
        d_sae=16 * 1024,
        # ... other common SAE parameters from SAEConfig if needed ...
    ),
    # ...
)
sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()
```

### Training MatryoshkaBatchTopk SAEs

<!-- prettier-ignore-start -->
!!! tip "SOTA architecture"
    MatryoshkaBatchTopK is a state-of-the-art SAE architecture, but requires some understanding to train well.
<!-- prettier-ignore-end -->

The [MatryoshkaBatchTopK](https://arxiv.org/abs/2503.17547) architecture is an advanced version of BatchTopK that uses a series of nested reconstruction losses at different widths (called "matryoshka levels") during training. This approach helps avoid [feature absorption](https://arxiv.org/abs/2409.14507) and encourages higher-frequency features to be learned at smaller latent indices (e.g. latent 20 will be a higher frequency, more general latent, than latent 20,000). While Matryoshka SAEs can achieve better performance than standard SAE architectures, they require tuning additional hyperparameters and take longer to train due to requiring multiple forward passes per training step. The key parameter is `matryoshka_widths`, which specifies the widths of each nested level (e.g., `[4096, 8192, 16384]`). These widths should be strictly increasing, and the final width should match `d_sae`. While Matryoshka SAEs can achieve the best performance of any known architecture on many metrics, they are more susceptible to [feature hedging](https://arxiv.org/abs/2505.11756) than standard SAEs, and this gets worse the narrower the inner level of the Matryoshka SAEs. We thus recommend being cautious about using very narrow inner levels. Like BatchTopK SAEs, MatryoshkaBatchTopK SAEs save as JumpReLU SAEs for use at inference.

```python
from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner, MatryoshkaBatchTopKTrainingSAEConfig

cfg = LanguageModelSAERunnerConfig( # Full config would be defined here
    # ... other LanguageModelSAERunnerConfig parameters ...
    sae=MatryoshkaBatchTopKTrainingSAEConfig(
        k=100, # Set the number of active features
        d_in=1024, # Must match your hook point
        d_sae=16 * 1024,
        matryoshka_widths=[4 * 1024, 8 * 1024, 16 * 1024], # matryoshka widths
        # ... other common SAE parameters from SAEConfig if needed ...
    ),
    # ...
)
sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()
```

### Training Topk SAEs

<!-- prettier-ignore-start -->
!!! warning "Warning: legacy architecture"
    TopK SAEs are no longer considered state-of-the-art, and are not recommended for most use cases. We recommend using BatchTopK or JumpReLU SAEs for best performance.
<!-- prettier-ignore-end -->

A popular alternative architecture is the [TopK](https://arxiv.org/abs/2406.04093) architecture, which fixes the L0 of the SAE using a TopK activation function. To train a TopK SAE programmatically, you provide a `TopKTrainingSAEConfig` instance to the `sae` field. The primary parameter for TopK SAEs is `k`, the number of features to keep active. If not set, `k` defaults to 100 in `TopKTrainingSAEConfig`. The TopK architecture does not use an `l1_coefficient` or `lp_norm` for sparsity, as sparsity is structurally enforced.

```python
from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner, TopKTrainingSAEConfig

cfg = LanguageModelSAERunnerConfig( # Full config would be defined here
    # ... other LanguageModelSAERunnerConfig parameters ...
    sae=TopKTrainingSAEConfig(
        k=100, # Set the number of active features
        d_in=1024, # Must match your hook point
        d_sae=16 * 1024,
        # ... other common SAE parameters from SAEConfig if needed ...
    ),
    # ...
)
sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()
```

### Training Gated SAEs

<!-- prettier-ignore-start -->
!!! warning "Warning: legacy architecture"
    Gated SAEs are no longer considered state-of-the-art, and are not recommended for most use cases. We recommend using BatchTopK or JumpReLU SAEs for best performance.
<!-- prettier-ignore-end -->

[Gated SAEs](https://arxiv.org/abs/2404.16014) are another architecture option. To train a Gated SAE, provide a `GatedTrainingSAEConfig` to the `sae` field. Gated SAEs use the `l1_coefficient` parameter to control the sparsity of the SAE, similar to standard SAEs.

```python
from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner, GatedTrainingSAEConfig

cfg = LanguageModelSAERunnerConfig( # Full config would be defined here
    # ... other LanguageModelSAERunnerConfig parameters ...
    sae=GatedTrainingSAEConfig(
        l1_coefficient=5.0, # Sparsity penalty coefficient
        d_in=1024, # Must match your hook point
        d_sae=16 * 1024,
        # ... other common SAE parameters from SAEConfig ...
    ),
    # ...
)
sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()
```

## CLI Runner

The SAE training runner can also be run from the command line via the `sae_lens.sae_training_runner` module. This can be useful for quickly testing different hyperparameters or running training on a remote server. The command line interface is shown below. All options to the CLI are the same as the [LanguageModelSAERunnerConfig][sae_lens.LanguageModelSAERunnerConfig] with a `--` prefix. E.g., `--model_name` is the same as `model_name` in the config.

```bash
python -m sae_lens.sae_training_runner --help
```

## Logging to Weights and Biases

For any real training run, you should be logging to Weights and Biases (WandB). This will allow you to track your training progress and compare different runs. To enable WandB, set `log_to_wandb=True`. The `wandb_project` parameter in the config controls the project name in WandB. You can also control the logging frequency with `wandb_log_frequency` and `eval_every_n_wandb_logs`.

A number of helpful metrics are logged to WandB, including the sparsity of the SAE, the mean squared error (MSE) of the SAE, dead features, and explained variance. These metrics can be used to monitor the training progress and adjust the training parameters. Below is a screenshot from one training run.

![screenshot](dashboard_screenshot.png)

## Best practices for real SAEs

It may sound daunting to train a real SAE but nothing could be further from the truth! You can typically train a decent SAE for a real LLM on a single A100 GPU in a matter of hours.

SAE Training best practices are still rapidly evolving, so the default settings in SAELens may not be optimal for real SAEs. Fortunately, it's easy to see what any SAE trained using SAELens used for its training configuration and just copy its values as a starting point! If there's a SAE on Huggingface trained using SAELens, you can see all the training settings used by looking at the `cfg.json` file in the SAE's repo. For instance, here's the [cfg.json](https://huggingface.co/jbloom/Gemma-2b-Residual-Stream-SAEs/blob/main/gemma_2b_blocks.12.hook_resid_post_16384/cfg.json) for a Gemma 2B standard SAE trained by Joseph Bloom. You can also get the config in SAELens as the second return value from `SAE.from_pretrained_with_cfg_and_sparsity()`. For instance, the same config mentioned above can be accessed as `cfg_dict = SAE.from_pretrained_with_cfg_and_sparsity("jbloom/Gemma-2b-Residual-Stream-SAEs", "gemma_2b_blocks.12.hook_resid_post_16384")[1]`. You can browse all SAEs uploaded to Huggingface via SAELens to get some inspiration with the [SAELens library tag](https://huggingface.co/models?library=saelens).

Some general performance tips:

- If your GPU supports it (most modern nvidia-GPUs do), setting `autocast=True` and `autocast_lm=True` in the config will dramatically speed up training.
- We find that often SAEs struggle to train well with `dtype="bfloat16"`. We aren't sure why this is, but make sure to compare the SAE quality if you change the dtype.
- You can try turning on `compile_sae=True` and `compile_llm=True`in the config to see if it makes training faster. Your mileage may vary though, compilation can be finicky.

## Checkpoints

Checkpoints allow you to save a snapshot of the SAE and sparsitity statistics during training. To enable checkpointing, set `n_checkpoints` to a value larger than 0. If WandB logging is enabled, checkpoints will be uploaded as WandB artifacts. To save checkpoints locally, the `checkpoint_path` parameter can be set to a local directory. You can also set `save_final_checkpoint=True` to save a final checkpoint after training is finished.

To resume training from a saved checkpoint, set `resume_from_checkpoint` to the path of the checkpoint when creating a `LanguageModelSAETrainingRunner`, or set `--resume_from_checkpoint` when running the CLI.

```python

cfg = LanguageModelSAERunnerConfig(
    # ... other LanguageModelSAERunnerConfig parameters ...
    resume_from_checkpoint="path/to/checkpoint"
)
runner = LanguageModelSAETrainingRunner(cfg)
runner.run()

```

## Optimizers and Schedulers

The SAE training runner uses the Adam optimizer with a constant learning rate by default. The optimizer betas can be controlled with the settings `adam_beta1` and `adam_beta2`.

The learning rate scheduler can be controlled with the `lr_scheduler_name` parameter. The available schedulers are: `constant` (default), `consineannealing`, and `cosineannealingwarmrestarts`. All schedulers can be used with linear warmup and linear decay, set via `lr_warm_up_steps` and `lr_decay_steps`.

To avoid dead features, it's often helpful to slowly increase the L1 penalty. This can be done by setting `l1_warm_up_steps` to a value larger than 0. This will linearly increase the L1 penalty over the first `l1_warm_up_steps` training steps.

## Training on Huggingface Models

While TransformerLens is the recommended way to use SAELens, it is also possible to use any Huggingface AutoModelForCausalLM as the model. This is useful if you want to use a model that is not supported by TransformerLens, or if you cannot use TransformerLens due to memory or performance reasons. To use a Huggingface AutoModelForCausalLM, you can specify `model_class_name = 'AutoModelForCausalLM'` in the SAE config. Your hook points will then need to correspond to the named parameters of the Huggingface model rather than the typical TransformerLens hook points. For instance, if you were using GPT2 from Huggingface, you would use `hook_name = 'transformer.h.1'` rather than `hook_name = 'blocks.1.hook_resid_post'`. Otherwise everything should work the same as with TransformerLens models.

## Datasets, streaming, and context size

SAELens works with datasets hosted on Huggingface. However, these datsets are often very large and take a long time and a lot of disk space to download. To speed this up, you can set `streaming=True` in the config. This will stream the dataset from Huggingface during training, which will allow training to start immediately and save disk space.

The `context_size` parameter controls the length of the prompts fed to the model. Larger context sizes will result in better SAE performance, but will also slow down training. Each training batch will be tokens of size `train_batch_size_tokens x context_size`.

It's also possible to use pre-tokenized datasets to speed up training, since tokenization can be a bottleneck. To use a pre-tokenized dataset on Huggingface, update the `dataset_path` parameter and set `is_dataset_tokenized=True` in the config.

## Pretokenizing datasets

We also provider a runner, [PretokenizeRunner][sae_lens.PretokenizeRunner], which can be used to pre-tokenize a dataset and upload it to Huggingface. See [PretokenizeRunnerConfig][sae_lens.PretokenizeRunnerConfig] for all available options. We also provide a [pretokenizing datasets tutorial](https://github.com/decoderesearch/SAELens/blob/main/tutorials/pretokenizing_datasets.ipynb) with more details.

A sample run from the tutorial for GPT2 and the NeelNanda/c4-10k dataset is shown below.

```python
from sae_lens import PretokenizeRunner, PretokenizeRunnerConfig

cfg = PretokenizeRunnerConfig(
    tokenizer_name="gpt2",
    dataset_path="NeelNanda/c4-10k", # this is just a tiny test dataset
    shuffle=True,
    num_proc=4, # increase this number depending on how many CPUs you have

    # tweak these settings depending on the model
    context_size=128,
    begin_batch_token="bos",
    begin_sequence_token=None,
    sequence_separator_token="eos",

    # uncomment to upload to huggingface
    # hf_repo_id="your-username/c4-10k-tokenized-gpt2"

    # uncomment to save the dataset locally
    # save_path="./c4-10k-tokenized-gpt2"
)

dataset = PretokenizeRunner(cfg).run()
```

## List of Pretokenized datasets

Below is a list of pre-tokenized datasets that can be used with SAELens. If you have a dataset you would like to add to this list, please open a PR!

| Huggingface ID                                                                                                                                                                                 | Tokenizer               | Source Dataset                                                                             | context size | Created with SAELens                                                                                                                           |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------ | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| [chanind/openwebtext-gemma](https://huggingface.co/datasets/chanind/openwebtext-gemma)                                                                                                         | gemma                   | [Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext)           | 8192         | [Yes](https://huggingface.co/datasets/chanind/openwebtext-gemma/blob/main/sae_lens.json)                                                       |
| [chanind/openwebtext-llama3](https://huggingface.co/datasets/chanind/openwebtext-llama3)                                                                                                       | llama3                  | [Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext)           | 8192         | [Yes](https://huggingface.co/datasets/chanind/openwebtext-llama3/blob/main/sae_lens.json)                                                      |
| [apollo-research/Skylion007-openwebtext-tokenizer-EleutherAI-gpt-neox-20b](https://huggingface.co/datasets/apollo-research/Skylion007-openwebtext-tokenizer-EleutherAI-gpt-neox-20b)           | EleutherAI/gpt-neox-20b | [Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext)           | 2048         | [No](https://huggingface.co/datasets/apollo-research/Skylion007-openwebtext-tokenizer-EleutherAI-gpt-neox-20b/blob/main/upload_script.py)      |
| [apollo-research/monology-pile-uncopyrighted-tokenizer-EleutherAI-gpt-neox-20b](https://huggingface.co/datasets/apollo-research/monology-pile-uncopyrighted-tokenizer-EleutherAI-gpt-neox-20b) | EleutherAI/gpt-neox-20b | [monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted) | 2048         | [No](https://huggingface.co/datasets/apollo-research/monology-pile-uncopyrighted-tokenizer-EleutherAI-gpt-neox-20b/blob/main/upload_script.py) |
| [apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2](https://huggingface.co/datasets/apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2)                                       | gpt2                    | [monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted) | 1024         | [No](https://huggingface.co/datasets/apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2/blob/main/upload_script.py)                    |
| [apollo-research/Skylion007-openwebtext-tokenizer-gpt2](https://huggingface.co/datasets/apollo-research/Skylion007-openwebtext-tokenizer-gpt2)                                                 | gpt2                    | [Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext)           | 1024         | [No](https://huggingface.co/datasets/apollo-research/Skylion007-openwebtext-tokenizer-gpt2/blob/main/upload_script.py)                         |
| [GulkoA/TinyStories-tokenized-Llama-3.2](https://huggingface.co/datasets/GulkoA/TinyStories-tokenized-Llama-3.2)                                                                               | llama3.2                | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)           | 128          | [Yes](https://huggingface.co/datasets/GulkoA/TinyStories-tokenized-Llama-3.2/blob/main/sae_lens.json)                                          |

## Caching activations

The next step in improving performance beyond pre-tokenizing datasets is to cache model activations. This allows you to pre-calculate all the training activations for your SAE in advance so the model does not need to be run during training to generate activations. This allows rapid training of SAEs and is especially helpful for experimenting with training hyperparameters. However, pre-calculating activations can take a very large amount of disk space, so it may not always be possible.

SAELens provides a [CacheActivationsRunner][sae_lens.CacheActivationsRunner] class to help with pre-calculating activations. See [CacheActivationsRunnerConfig][sae_lens.CacheActivationsRunnerConfig] for all available options. This runner intentionally shares a lot of options with [LanguageModelSAERunnerConfig][sae_lens.LanguageModelSAERunnerConfig]. These options should be set identically when using the cached activations in training. The CacheActivationsRunner can be used as below:

```python
from sae_lens import CacheActivationsRunner, CacheActivationsRunnerConfig

cfg = CacheActivationsRunnerConfig(
    model_name="tiny-stories-1L-21M",
    hook_name="blocks.0.hook_mlp_out",
    dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
    # ...
    new_cached_activations_path="./tiny-stories-1L-21M-cache",
    hf_repo_id="your-username/tiny-stories-1L-21M-cache", # To push to hub
)

CacheActivationsRunner(cfg).run()
```

To use the cached activations during training, set `use_cached_activations=True` and `cached_activations_path` to match the `new_cached_activations_path` above option in training configuration.

## Uploading SAEs to Huggingface

Once you have a set of SAEs that you're happy with, your next step is to share them with the world! SAELens has a `upload_saes_to_huggingface()` function which makes this easy to do. We also provide a [uploading saes to huggingface tutorial](https://github.com/decoderesearch/SAELens/blob/main/tutorials/uploading_saes_to_huggingface.ipynb) with more details.

You'll just need to pass a dictionary of SAEs to upload along with the huggingface repo id to upload to. The dictionary keys will become the folders in the repo where each SAE will be located. It's best practice to use the hook point that the SAE was trained on as the key to make it clear to users where in the model to apply the SAE. The values of this dictionary can be either an SAE object, or a path to a saved SAE object on disk from the `sae.save_model()` method.

A sample is shown below:

```python
from sae_lens import upload_saes_to_huggingface

saes_dict = {
    "blocks.0.hook_resid_pre": layer_0_sae,
    "blocks.1.hook_resid_pre": layer_1_sae,
    # ...
}

upload_saes_to_huggingface(
    saes_dict,
    hf_repo_id="your-username/your-sae-repo",
)
```
