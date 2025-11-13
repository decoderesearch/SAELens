# Creating Custom SAE Architectures

This guide explains how to create custom Sparse Autoencoder (SAE) architectures in SAELens. Often you'll want to modify the training procedure while keeping the same inference behavior, so we'll start with creating custom training SAEs first.

### Understanding the SAE Architecture System

SAELens uses a modular architecture system where each SAE variant is defined by:

1. **Configuration classes** that define hyperparameters and settings
2. **SAE classes** that implement the actual neural network logic
3. **Registration** that makes the architecture available to the training system

The system separates inference SAEs (designed to be deployed after training) from training SAEs (meant to be trained and then saved as inference SAEs).

### Training vs Inference SAEs

When an SAE is trained in SAELens, we create a `TrainingSAE` class. When the SAE is finished training, it is saved as an `SAE` for inference. For most SAE architectures, there is both a training and an inference SAE class available (e.g. there's is a `TopKSAE` class for inference, and a `TopKTrainingSAE` class for training). However, this does not always have to be the case. For instance, there is a `BatchTopKTrainingSAE` class for training batch TopK SAEs, but there is not corresponding inference class since Batch TopK SAEs are saved as `JumpReLU` SAEs for inference.

If you just want to modify the training procedure for an SAE, you can just create your own custom training SAE class, extending the `TrainingSAE` subclass you're interested in (or the base `TrainingSAE` class directly, depending on your needs).

### Base Classes

All SAE architectures inherit from these base classes:

- **`SAEConfig`**: Base configuration for inference SAEs
- **`SAE[T_SAE_CONFIG]`**: Base class for inference SAE implementations
- **`TrainingSAEConfig`**: Base configuration for training SAEs
- **`TrainingSAE[T_TRAINING_SAE_CONFIG]`**: Base class for training SAE implementations

## Creating a Custom Training SAE

Let's walk through creating a custom SAE architecture step by step. A classic SAE architecture that only modifies the training procedure is [Matryoshka SAEs](https://arxiv.org/pdf/2503.17547), which use a nested sparsity penalty during training. However, after training, Matryoshka SAEs are just loaded a standaed SAEs. We'll create a "TopKMatryoshka SAE" class that lets us nest a series of reconstruction losses of different widths during traning, but still saves as a standard TopK inference sae after training.

We'll start by creating the configuration class for our new SAE. Most important is to override the `architecture` method to return a unique name for our new SAE type. We also need to add any extra fields that our new SAE class will require. Here, we'll add a list of widths for the inner matryoshka levels:

```python
from dataclasses import dataclass
from typing_extensions import override
from sae_lens import TopKTrainingSAEConfig, TopKTrainingSAE

@dataclass
class TopKMatryoshkaTrainingSAEConfig(TopKTrainingSAEConfig):
    """Configuration for TopK Matryoshka SAE training."""

    inner_matryoshka_widths: list[int]

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk_matryoshka"
```

For instance, if our SAE has width 1024 and we want to use a matryoshka of width 128 and 256, we would set `inner_matryoshka_widths` to `[128, 256]`.

Next, we define the training SAE class. Here we'll extend the `TopKTrainingSAE` class and overwrite the `training_forward_pass` method to add the nested reconstruction losses. You can overwrite any method in the `TrainingSAE` class you want to modify the training procedure. The `training_forward_pass` method is the main method that gets called during training, and is a likely place to modify the training procedure. Other methods like `encode_with_hidden_pre` and `calculate_aux_loss` are also good places to modify the training procedure.

```python
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput

class TopKMatryoshkaTrainingSAE(TopKTrainingSAE):
    """
    TopK Matryoshka SAE for training.
    Uses a series of nested reconstruction losses during training.
    """

    cfg: TopKMatryoshkaTrainingSAEConfig # type: ignore[assignment]

    def __init__(self, cfg: TopKMatryoshkaTrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def def training_forward_pass(
        self,
        step_input: TrainStepInput,
    ) -> TrainStepOutput:
        """Forward pass during training."""
        base_output = super().training_forward_pass(step_input)
        hidden_pre = base_output.hidden_pre
        for width in self.cfg.inner_matryoshka_widths:
            inner_hidden_pre = hidden_pre[:, :width]
            inner_feat_acts = self.activation_fn(inner_hidden_pre)
            inner_reconstruction =  inner_feat_acts @ self.W_dec[:width] + self.b_dec
            inner_mse_loss = self.mse_loss_fn(inner_reconstruction, step_input.sae_in).sum(dim=-1).mean()
            base_output.losses[f"inner_mse_loss_{width}"] = inner_mse_loss
            base_output.loss = base_output.loss + inner_mse_loss
        return base_output
```

Some key things to note here:

- We use `super().training_forward_pass(step_input)` to get the base output from the parent class, which includes the default MSE loss and topk aux reconstruction loss.
- We use `self.activation_fn` to apply the topk activation function to the hidden pre-activations.
- We use `self.mse_loss_fn` to calculate the MSE loss between the inner reconstruction and the input.
- We use `base_output.loss` to add the inner MSE loss to the existing loss. The trainer will call `backwards` on this loss.
- We use add our inner losses to the `base_output.losses` dictionary. Everything in this dictionary will be logged to wandb so we can track these losses during training.

Finally, we register the training SAE class with SAELens so we can use it in the training runner.

```python
from sae_lens import register_sae_training_class

# Register the training SAE
register_sae_training_class(
    "topk_matryoshka",
    TopKMatryoshkaTrainingSAE,
    TopKMatryoshkaTrainingSAEConfig
)
```

Now you can use your custom training SAE in SAELens! You can train it using the standard training runner with our topk matryoshka training SAE config in the LLM training runner, like below:

```python
from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner

cfg = LanguageModelSAERunnerConfig(

    # SAE Parameters are in the nested 'sae' config
    sae=TopKMatryoshkaTrainingSAEConfig(
        k=20, # TopK parameter
        d_in=768, # Matches hook_resid_post for gpt2
        d_sae=16384,
        inner_matryoshka_widths=[1024, 4096], # inner matryoshka widths
    ),

    # Data Generating Function (Model + Training Distribution)
    model_name="gpt2",
    hook_name="blocks.3.hook_resid_post",
    dataset_path="monology/pile-uncopyrighted",
    streaming=True,

    # Training Parameters
    lr=3e-4,
    train_batch_size_tokens=4096,
    context_size=256,
    training_tokens=30_000 * 4096,

    # Misc
    device="cuda" if torch.cuda.is_available() else "cpu",
)

sae = LanguageModelSAETrainingRunner(cfg).run()

# optional, save the SAE to disk as an inference SAE
sae.save_inference_model("path/to/sae")
```

## Common Training Modifications

Often you'll want to modify just the training procedure while keeping the same basic architecture. Here are common patterns:

### Custom coefficients

If your training SAE has custom coefficients that you want to optionally warm up or decay during training, you can override the `get_coefficients` method to return a dictionary of coefficients.

```python
@override
def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
    """Define training coefficients including custom ones."""
    return {
        "l1": TrainCoefficientConfig(
            value=self.cfg.l1_coefficient,
            warm_up_steps=self.cfg.l1_warm_up_steps,
        ),
        "orthogonality": TrainCoefficientConfig(
            value=self.cfg.orthogonality_coefficient,
            warm_up_steps=0,
        ),
    }
```

The current value of these coefficients will be passed to the `training_forward_pass` method as the `step_input.coefficients` dictionary.

### Custom training_forward_pass

As we saw above, the `training_forward_pass` method is the main method that gets called during training, and is a likely place to modify the training procedure. This method is passed in a `TrainStepInput` object, which contains info about the current training step. It's signature is below:

```python
@dataclass
class TrainStepInput:
    """Input to a training step."""

    sae_in: torch.Tensor
    coefficients: dict[str, float]
    dead_neuron_mask: torch.Tensor | None
    n_training_steps: int
```

This method should return a `TrainStepOutput` object, which contains the output of the training step. It's signature is below:

```python
@dataclass
class TrainStepOutput:
    """Output from a training step."""

    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    hidden_pre: torch.Tensor
    loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    metrics: dict[str, torch.Tensor | float | int]
```

Of particular interest, the `loss` field is the total loss for the training step, and is what `.backwards()` is called on. The `losses` field is a dictionary of all the losses for the training step, and will be logged to wandb. The `metrics` field is a dictionary of any extra metrics to log during training to wandb.

Often, it's a good idea to call `super().training_forward_pass(step_input)` to get the base output from the parent class, which includes the default MSE loss and any aux reconstruction losses and then modify that output.

### Customizing the inference SAE export

If your training SAE needs to modify how it is saved for the inference version of the SAE, you can override the `process_state_dict_for_saving_inference` method of your training SAE. This method is called before the state dict is saved to disk, and is a good place to modify the state dict in-place. You may also need to override the `get_inference_sae_cfg_dict` method on the config class to customize the config for the inference version of the SAE.

A good example of this is the `BatchTopKTrainingSAE` class source code, which writes out a `JumpReLU` SAE for the inference version of the SAE.

## Creating a Custom Inference SAE (Optional)

If you need to create a separate inference version (e.g., if your SAE architecture is not a simple training modification of the training procedure of an existing SAE), you'll need to also create and register an inference SAE class.

Here, we'll demonstrate this process for a simple architecture we'll call a "GeluSAE", which uses `torch.nn.Gelu()` as the activation function. This is not actually a good idea, but it's a simple example to demonstrate the process.

First, we'll define the configuration class for our new SAE.

```python
from sae_lens.saes.sae import StandardSAEConfig

@dataclass
class GeluSAEConfig(StandardSAEConfig):
    """Configuration for Gelu SAE inference."""

    # Add any architecture-specific parameters here

    @override
    @classmethod
    def architecture(cls) -> str:
        return "gelu"
```

Next, we'll define our new GeluSAE class, extending the `StandardSAE` class and overwriting any methods necessary.

```python
import torch
import torch.nn as nn
from typing_extensions import override
from sae_lens.saes.sae import SAE

class GeluSAE(StandardSAE):
    """
    Gelu SAE for inference.
    Uses `torch.nn.Gelu()` as the activation function.
    """

    cfg: GeluSAEConfig # type: ignore[assignment]

    def __init__(self, cfg: GeluSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)


    @override
    def get_activation_fn(self) -> callable:
        return torch.nn.Gelu()
```

Finally, we register the inference SAE class with SAELens so we can use it in the inference runner.

```python
from sae_lens import register_sae_class

# Register the inference SAE
register_sae_class("gelu", GeluSAE, GeluSAEConfig)
```
