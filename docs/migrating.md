# Migrating to SAELens v6

SAELens v6 is a major update with a number of breaking changes that will require updating your code. The goal of this update was the following:

- Simplify SAE configs, removing legacy cruft and making it easy to train SAEs
- Separate SAE architectures into their own classes with separate options (e.g. TopK doesn't have the same options as JumpReLU)
- Make it easy to extend SAELens with your own custom SAEs
- Simplify loading and using SAEs
- Decouple the LLM-specific training code from SAE training, so it's possible to train an SAE on any type of model (e.g. vision models).

We apologize for any inconvenience this causes! These changes should put SAELens in a good place going forward though, making it possible to easily extend the library as new SAE architectures get released, and make it easier than ever to train and share customized SAEs. Expect lots of exciting updates building on this to come soon!

## Changes to SAE.from_pretrained()

We expect that most users will use SAELens to load SAEs rather than training SAEs, so we expect this section to be the most important for most users. Previously, `SAE.from_pretrained()` returned a tuple of SAE, a cfg dict, and a feature sparsity tensor. Now, `SAE.from_pretrained()` returns just the SAE to be consistent with how `from_pretrained()` functions in [Huggingface Transformers](https://huggingface.co/docs/transformers/en/index) and [TransformerLens](https://transformerlensorg.github.io/TransformerLens/).

Old code of the form `sae, cfg_dict, sparsity = SAE.from_pretrained(...)` should still continue to work, but will give a deprecation warning.

The old functionality also exists via `SAE.load_from_pretrained_with_cfg_and_sparsity()` - although we expect this will not be needed by most users.

## SAE Training config changes

The LLM SAE training runner config now follows a new nested structure, with SAE-specific options specified in a nested `sae` section, and logging options specified in a nested `logger` section. This looks like the following:

```python
from sae_lens import LanguageModelSAERunnerConfig, StandardTrainingSAEConfig, LoggingConfig

cfg = LanguageModelSAERunnerConfig(
    # SAE Parameters
    sae=StandardTrainingSAEConfig(
        d_in=1024,
        d_sae=16384,
        apply_b_dec_to_input=True,
        normalize_activations="expected_average_only_in",
        l1_coefficient=5,
    ),
    # Data Generating Function (Model + Training Distibuion)
    model_name="tiny-stories-1L-21M",
    hook_name="blocks.0.hook_mlp_out",
    dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
    is_dataset_tokenized=True,
    streaming=True,
    # Training Parameters
    lr=5e-5,
    train_batch_size_tokens=4096,
    context_size=512,
    n_sequences_in_buffer=64,
    training_tokens=100_000_000,
    store_batch_size_prompts=16,
    # WANDB
    logger=LoggingConfig(
        log_to_wandb=True,
        wandb_project="sae_lens_tutorial",
    ),
    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32",
)
```

There are corresponding config classes for `GatedTrainingSAEConfig`, `JumpReLUTrainingSAEConfig`, and `TopKTrainingSAEConfig` depending on the type of SAE you'd like to train.

## Removed legacy training options

We removed a number of legacy config options from the training config, as we found that having so many options was both confusing and daunting to new users of the library. SAE training best practices have changed rapidly, so we took this opportunity to remove rarely used and legacy options from the config.

The removed options include:

- expansion_factor (this was redundant given the `d_sae` option to set SAE width)
- hook_layer (this was redundant - we already know the layer from `hook_name`)
- ghost grads
- b_dec init options (b_dec is always init to zero now)
- decoder init options (we always use the heuristic init from [Anthropic's April 2024 update](https://transformer-circuits.pub/2024/april-update/index.html#training-saes))
- MSE loss normalization
- decoder normalization (we always scale L1 loss by decoder norm now, this is always the right thing to do)
- finetuning_tokens / finetuning_method
- noise_scale
- activation_fn / activation_fn_kwargs

## Config option renaming / changed defaults

- The JumpReLU L0 coefficient is now called `l0_coefficient`
- `k` is now set explicitly for TopK SAEs rather than being in `activation_fn_kwargs`
- Default JumpReLU bandwidth has been increased to 0.05 from 0.001 to make JumpReLU SAEs more responsive during training
- Default JumpReLU starting threshold has been increase to 0.01 from 0.001

## Other breaking changes

`SAETrainingRunner` has been renamed to `LanguageModelSAETrainingRunner`, although the `SAETrainingRunner` name will still keep working for now. This change was made to allow other types of SAETrainingRunners to be added in the future (e.g. for vision models).

Running SAE training from the CLI now requires running the script: `python -m sae_lens.llm_sae_training_runner` to reflect this change.

`SAE.cfg` now only contains config keys that are essential for running the SAE. Everything else, such as `prepend_bos`, has been moved to `SAE.cfg.metadata`.