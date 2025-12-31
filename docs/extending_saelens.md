# Extending SAELens

This guide explains how to extend SAELens to work with new model types, custom activation sources, or non-standard training workflows. While the `LanguageModelSAETrainingRunner` provides a batteries-included experience for TransformerLens models, you may want more control over the training process—for example, to train SAEs on vision models, sequence classifiers, or custom architectures.

## Overview

SAELens separates the **SAE training logic** (handled by `SAETrainer`) from the **activation generation** (handled by data providers). This separation makes it straightforward to extend SAELens to new settings:

1. **`SAETrainer`**: The core training loop that handles optimization, logging, checkpointing, and evaluation
2. **`DataProvider`**: An iterator that yields batches of activations for training
3. **`Evaluator`**: An optional callable that runs custom evaluations during training
4. **`mixing_buffer`**: A utility for shuffling activations when your source produces them in sequence order

## Using SAETrainer Directly

The `SAETrainer` class is the core of SAELens training. You can use it directly when you want full control over the training process. Here's what you need:

```python
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.config import SAETrainerConfig, LoggingConfig
from sae_lens.saes.sae import TrainingSAE, TrainingSAEConfig
from sae_lens import StandardTrainingSAEConfig
from collections.abc import Iterator
import torch

# 1. Create a TrainingSAE with your desired architecture
sae_cfg = StandardTrainingSAEConfig(
    d_in=768,
    d_sae=768 * 8,
    l1_coefficient=5.0,
)
sae = TrainingSAE.from_dict(sae_cfg.to_dict())

# 2. Create a data provider (any iterator that yields activation tensors)
def my_activation_generator() -> Iterator[torch.Tensor]:
    while True:
        # Your logic to generate activations
        yield torch.randn(4096, 768)  # (batch_size, d_in)

data_provider = my_activation_generator()

# 3. Create the trainer config
trainer_cfg = SAETrainerConfig(
    total_training_samples=1_000_000,
    train_batch_size_samples=4096,
    device="cuda",
    lr=3e-4,
    lr_end=3e-5,
    lr_scheduler_name="constant",
    lr_warm_up_steps=1000,
    adam_beta1=0.9,
    adam_beta2=0.999,
    lr_decay_steps=0,
    n_restart_cycles=1,
    autocast=True,
    dead_feature_window=1000,
    feature_sampling_window=2000,
    n_checkpoints=0,
    checkpoint_path=None,
    save_final_checkpoint=False,
    logger=LoggingConfig(log_to_wandb=False),
)

# 4. Create and run the trainer
trainer = SAETrainer(
    cfg=trainer_cfg,
    sae=sae,
    data_provider=data_provider,
    evaluator=None,  # Optional: add custom evaluation
)

trained_sae = trainer.fit()

# Save the trained SAE
trained_sae.save_inference_model("path/to/sae")
```

### SAETrainerConfig

The `SAETrainerConfig` contains all training hyperparameters. Key fields include:

| Field                      | Description                                                                        |
| -------------------------- | ---------------------------------------------------------------------------------- |
| `total_training_samples`   | Total number of activation samples to train on                                     |
| `train_batch_size_samples` | Batch size (number of activation vectors per step)                                 |
| `device`                   | Device to train on (`"cuda"`, `"cpu"`, etc.)                                       |
| `lr`, `lr_end`             | Learning rate and final learning rate for schedulers                               |
| `lr_scheduler_name`        | Scheduler type: `"constant"`, `"cosineannealing"`, `"cosineannealingwarmrestarts"` |
| `lr_warm_up_steps`         | Number of warmup steps for learning rate                                           |
| `autocast`                 | Whether to use mixed precision training                                            |
| `dead_feature_window`      | Window for detecting dead features                                                 |
| `n_checkpoints`            | Number of checkpoints to save during training                                      |
| `logger`                   | `LoggingConfig` for W&B logging                                                    |

### DataProvider

A `DataProvider` is simply an iterator that yields `torch.Tensor` batches:

```python
from collections.abc import Iterator
import torch

DataProvider = Iterator[torch.Tensor]
```

Each tensor should have shape `(batch_size, d_in)` where `d_in` matches your SAE's input dimension. The trainer will call `next(data_provider)` to get each batch.

## Creating Custom Runners

For more complex scenarios, you may want to create a custom runner class that encapsulates your model, activation generation, and training logic. Here's a template:

```python
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any
import torch

from sae_lens.training.sae_trainer import SAETrainer, Evaluator
from sae_lens.config import SAETrainerConfig, LoggingConfig
from sae_lens.saes.sae import TrainingSAE, TrainingSAEConfig
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.types import DataProvider

@dataclass
class MyCustomRunnerConfig:
    """Configuration for your custom runner."""
    # Model settings
    model_name: str
    hook_layer: int

    # Training settings
    training_samples: int
    batch_size: int
    lr: float
    device: str = "cuda"

    # SAE settings
    d_in: int = 768
    d_sae: int = 768 * 8

    # Output
    output_path: str | None = None


class MyCustomRunner:
    """Custom runner for training SAEs on your model type."""

    def __init__(self, cfg: MyCustomRunnerConfig):
        self.cfg = cfg

        # Load your model
        self.model = self._load_model()

        # Create the SAE
        self.sae = self._create_sae()

    def _load_model(self) -> torch.nn.Module:
        """Load your model here."""
        # Example: load a custom model
        from transformers import AutoModel
        model = AutoModel.from_pretrained(self.cfg.model_name)
        return model.to(self.cfg.device)

    def _create_sae(self) -> TrainingSAE[Any]:
        """Create the SAE to train."""
        from sae_lens import StandardTrainingSAEConfig
        sae_cfg = StandardTrainingSAEConfig(
            d_in=self.cfg.d_in,
            d_sae=self.cfg.d_sae,
        )
        sae = TrainingSAE.from_dict(sae_cfg.to_dict())
        return sae.to(self.cfg.device)

    def _get_activations(self, inputs: Any) -> torch.Tensor:
        """Extract activations from your model."""
        activations = []

        def hook_fn(module: Any, input: Any, output: Any) -> None:
            activations.append(output.detach())

        # Register hook on desired layer
        target_layer = self._get_target_layer()
        handle = target_layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            self.model(inputs)

        handle.remove()
        return activations[0]

    def _get_target_layer(self) -> torch.nn.Module:
        """Get the layer to hook for activations."""
        # Implement based on your model architecture
        raise NotImplementedError

    def _create_data_provider(self) -> DataProvider:
        """Create an iterator that yields activation batches."""
        def activation_generator() -> Iterator[torch.Tensor]:
            # Your data loading logic here
            while True:
                # Get a batch of inputs for your model
                inputs = self._get_next_batch()

                # Extract activations
                activations = self._get_activations(inputs)

                # Flatten to (batch_size * seq_len, d_in) if needed
                if activations.dim() == 3:
                    activations = activations.view(-1, activations.size(-1))

                yield activations

        return activation_generator()

    def _get_next_batch(self) -> Any:
        """Get the next batch of inputs for your model."""
        raise NotImplementedError

    def _create_evaluator(self) -> Evaluator[TrainingSAE[Any]] | None:
        """Optionally create an evaluator for periodic evaluation."""
        return None

    def run(self) -> TrainingSAE[Any]:
        """Run training and return the trained SAE."""
        trainer_cfg = SAETrainerConfig(
            total_training_samples=self.cfg.training_samples,
            train_batch_size_samples=self.cfg.batch_size,
            device=self.cfg.device,
            lr=self.cfg.lr,
            lr_end=self.cfg.lr / 10,
            lr_scheduler_name="constant",
            lr_warm_up_steps=1000,
            adam_beta1=0.9,
            adam_beta2=0.999,
            lr_decay_steps=0,
            n_restart_cycles=1,
            autocast=True,
            dead_feature_window=1000,
            feature_sampling_window=2000,
            n_checkpoints=0,
            checkpoint_path=None,
            save_final_checkpoint=False,
            logger=LoggingConfig(log_to_wandb=True),
        )

        trainer = SAETrainer(
            cfg=trainer_cfg,
            sae=self.sae,
            data_provider=self._create_data_provider(),
            evaluator=self._create_evaluator(),
        )

        trained_sae = trainer.fit()

        # Save the trained SAE
        if self.cfg.output_path is not None:
            trained_sae.save_inference_model(self.cfg.output_path)

        return trained_sae
```

## Example: GPT2 Sequence Classifier

Here's a complete example showing how to train an SAE on a `GPT2ForSequenceClassification` model. This demonstrates training on a non-standard model type that isn't directly supported by the built-in `LanguageModelSAETrainingRunner`:

```python
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

from sae_lens.training.sae_trainer import SAETrainer, Evaluator
from sae_lens.training.mixing_buffer import mixing_buffer
from sae_lens.config import SAETrainerConfig, LoggingConfig
from sae_lens.saes.sae import TrainingSAE
from sae_lens import StandardTrainingSAEConfig
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.types import DataProvider


@dataclass
class ClassifierSAERunnerConfig:
    """Config for training SAEs on sequence classifiers."""
    model_name: str = "gpt2"
    dataset_path: str = "imdb"
    hook_layer: int = 6  # Which transformer block to hook

    # Training
    training_tokens: int = 1_000_000
    batch_size_prompts: int = 8
    train_batch_size: int = 4096
    context_size: int = 256
    lr: float = 3e-4
    device: str = "cuda"

    # SAE
    expansion_factor: int = 8
    l1_coefficient: float = 5.0

    # Buffer for activation shuffling
    n_batches_in_buffer: int = 16

    # Output
    output_path: str | None = None


class ClassifierSAERunner:
    """Runner for training SAEs on GPT2ForSequenceClassification."""

    def __init__(self, cfg: ClassifierSAERunnerConfig):
        self.cfg = cfg

        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2ForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=2,  # Binary classification
        ).to(cfg.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

        # Get hidden size from model config
        self.d_in = self.model.config.hidden_size

        # Load dataset
        self.dataset = load_dataset(cfg.dataset_path, split="train")

        # Create SAE
        sae_cfg = StandardTrainingSAEConfig(
            d_in=self.d_in,
            d_sae=self.d_in * cfg.expansion_factor,
            l1_coefficient=cfg.l1_coefficient,
        )
        self.sae = TrainingSAE.from_dict(sae_cfg.to_dict()).to(cfg.device)

    def _get_activations(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract activations from the target layer."""
        activations: list[torch.Tensor] = []

        def hook_fn(module: Any, input: Any, output: Any) -> None:
            # output is (hidden_states, ...) tuple
            hidden = output[0] if isinstance(output, tuple) else output
            activations.append(hidden.detach())

        # Hook into the transformer block's output
        target_block = self.model.transformer.h[self.cfg.hook_layer]
        handle = target_block.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            handle.remove()

        return activations[0]  # (batch, seq_len, hidden_size)

    def _iterate_raw_activations(self) -> Iterator[torch.Tensor]:
        """Iterate over batches of activations from the model."""
        dataloader = DataLoader(
            self.dataset,  # type: ignore
            batch_size=self.cfg.batch_size_prompts,
            shuffle=True,
        )

        while True:
            for batch in dataloader:
                # Tokenize
                encoded = self.tokenizer(
                    batch["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=self.cfg.context_size,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.cfg.device)

                # Get activations: (batch, seq_len, hidden_size)
                activations = self._get_activations(input_ids)

                # Flatten to (batch * seq_len, hidden_size)
                flat_activations = activations.view(-1, self.d_in)

                yield flat_activations

    def _create_data_provider(self) -> DataProvider:
        """Create a data provider with activation shuffling."""
        buffer_size = (
            self.cfg.n_batches_in_buffer
            * self.cfg.batch_size_prompts
            * self.cfg.context_size
        )

        return mixing_buffer(
            buffer_size=buffer_size,
            batch_size=self.cfg.train_batch_size,
            activations_loader=self._iterate_raw_activations(),
        )

    def _create_evaluator(
        self,
    ) -> Evaluator[TrainingSAE[Any]] | None:
        """Create an optional evaluator."""

        def simple_evaluator(
            sae: TrainingSAE[Any],
            data_provider: DataProvider,
            activation_scaler: ActivationScaler,
        ) -> dict[str, Any]:
            """Simple evaluation: compute reconstruction error on a batch."""
            sae.eval()
            batch = next(data_provider).to(sae.device)

            with torch.no_grad():
                sae_out = sae.decode(sae.encode(batch))
                mse = (batch - sae_out).pow(2).mean().item()

            sae.train()
            return {"eval/mse": mse}

        return simple_evaluator

    def run(self) -> TrainingSAE[Any]:
        """Run training."""
        trainer_cfg = SAETrainerConfig(
            total_training_samples=self.cfg.training_tokens,
            train_batch_size_samples=self.cfg.train_batch_size,
            device=self.cfg.device,
            lr=self.cfg.lr,
            lr_end=self.cfg.lr / 10,
            lr_scheduler_name="constant",
            lr_warm_up_steps=1000,
            adam_beta1=0.9,
            adam_beta2=0.999,
            lr_decay_steps=0,
            n_restart_cycles=1,
            autocast=True,
            dead_feature_window=1000,
            feature_sampling_window=2000,
            n_checkpoints=0,
            checkpoint_path=None,
            save_final_checkpoint=False,
            logger=LoggingConfig(
                log_to_wandb=True,
                wandb_project="classifier-sae",
            ),
        )

        trainer = SAETrainer(
            cfg=trainer_cfg,
            sae=self.sae,
            data_provider=self._create_data_provider(),
            evaluator=self._create_evaluator(),
        )

        trained_sae = trainer.fit()

        # Save the trained SAE
        if self.cfg.output_path is not None:
            trained_sae.save_inference_model(self.cfg.output_path)

        return trained_sae


# Usage
if __name__ == "__main__":
    cfg = ClassifierSAERunnerConfig(
        model_name="gpt2",
        dataset_path="imdb",
        hook_layer=6,
        training_tokens=500_000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_path="classifier_sae",
    )

    runner = ClassifierSAERunner(cfg)
    runner.run()
```

## Activation Shuffling with mixing_buffer

When collecting activations sequentially (e.g., processing documents one at a time), consecutive activations are highly correlated. This can hurt training. The `mixing_buffer` utility helps by shuffling activations:

```python
from sae_lens.training.mixing_buffer import mixing_buffer
from collections.abc import Iterator
import torch

def my_sequential_activations() -> Iterator[torch.Tensor]:
    """Yields activations in document order (correlated)."""
    while True:
        # Process a document and yield its activations
        yield torch.randn(1024, 768)  # (tokens_in_doc, d_in)

# Wrap with mixing_buffer to shuffle
shuffled_provider = mixing_buffer(
    buffer_size=100_000,      # Total activations to buffer
    batch_size=4096,          # Output batch size
    activations_loader=my_sequential_activations(),
)

# Now batches are shuffled!
for batch in shuffled_provider:
    print(batch.shape)  # (4096, 768)
    break
```

The mixing buffer:

1. Accumulates activations until reaching `buffer_size`
2. Randomly shuffles the buffer
3. Yields half as batches while keeping the other half
4. Refills with new activations and repeats

This ensures each batch contains activations from many different contexts rather than consecutive tokens from the same document.

## Custom Evaluators

Evaluators are called periodically during training to compute metrics. They receive the SAE, data provider, and activation scaler:

```python
from typing import Any
from sae_lens.saes.sae import TrainingSAE
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.types import DataProvider

def my_evaluator(
    sae: TrainingSAE[Any],
    data_provider: DataProvider,
    activation_scaler: ActivationScaler,
) -> dict[str, Any]:
    """Custom evaluation function."""
    sae.eval()

    # Collect some evaluation batches
    eval_batches = [next(data_provider) for _ in range(10)]

    metrics: dict[str, float] = {}

    with torch.no_grad():
        for batch in eval_batches:
            batch = batch.to(sae.device)
            scaled_batch = activation_scaler(batch)

            # Forward pass
            encoded = sae.encode(scaled_batch)
            decoded = sae.decode(encoded)

            # Compute metrics
            mse = (scaled_batch - decoded).pow(2).mean()
            l0 = (encoded != 0).float().sum(dim=-1).mean()

            metrics["mse"] = metrics.get("mse", 0) + mse.item()
            metrics["l0"] = metrics.get("l0", 0) + l0.item()

    # Average
    n_batches = len(eval_batches)
    metrics = {f"eval/{k}": v / n_batches for k, v in metrics.items()}

    sae.train()
    return metrics
```

The returned dictionary is logged to W&B if logging is enabled.

## Example: Vision Model SAE

Here's a sketch for training SAEs on vision model activations:

```python
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any
import torch
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader

from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.mixing_buffer import mixing_buffer
from sae_lens.config import SAETrainerConfig, LoggingConfig
from sae_lens.saes.sae import TrainingSAE
from sae_lens import StandardTrainingSAEConfig


@dataclass
class VisionSAEConfig:
    model_name: str = "vit_base_patch16_224"
    layer_name: str = "blocks.6"  # Target layer
    dataset_path: str = "/path/to/imagenet"
    batch_size: int = 32
    training_tokens: int = 1_000_000
    device: str = "cuda"
    output_path: str | None = None


class VisionSAERunner:
    def __init__(self, cfg: VisionSAEConfig):
        self.cfg = cfg

        # Load vision model (using timm for example)
        import timm
        self.model = timm.create_model(
            cfg.model_name,
            pretrained=True,
        ).to(cfg.device)
        self.model.eval()

        # Get the hidden dimension
        # This depends on your model - ViT base has d=768
        self.d_in = 768

        # Create SAE
        sae_cfg = StandardTrainingSAEConfig(
            d_in=self.d_in,
            d_sae=self.d_in * 8,
        )
        self.sae = TrainingSAE.from_dict(sae_cfg.to_dict()).to(cfg.device)

        # Setup data loading
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.dataset = ImageNet(cfg.dataset_path, split="train", transform=transform)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def _get_layer(self, name: str) -> torch.nn.Module:
        """Get a layer by name like 'blocks.6'."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _iterate_activations(self) -> Iterator[torch.Tensor]:
        """Extract activations from vision model."""
        while True:
            for images, _ in self.dataloader:
                images = images.to(self.cfg.device)
                activations: list[torch.Tensor] = []

                def hook_fn(m: Any, i: Any, o: Any) -> None:
                    activations.append(o.detach())

                layer = self._get_layer(self.cfg.layer_name)
                handle = layer.register_forward_hook(hook_fn)

                try:
                    with torch.no_grad():
                        self.model(images)
                finally:
                    handle.remove()

                # ViT outputs: (batch, num_patches + 1, hidden_dim)
                # Flatten patches: (batch * num_patches, hidden_dim)
                act = activations[0]
                flat_act = act.view(-1, act.size(-1))
                yield flat_act

    def run(self) -> TrainingSAE[Any]:
        # Use mixing buffer for shuffling patch activations
        data_provider = mixing_buffer(
            buffer_size=50_000,
            batch_size=4096,
            activations_loader=self._iterate_activations(),
        )

        trainer_cfg = SAETrainerConfig(
            total_training_samples=self.cfg.training_tokens,
            train_batch_size_samples=4096,
            device=self.cfg.device,
            lr=3e-4,
            lr_end=3e-5,
            lr_scheduler_name="constant",
            lr_warm_up_steps=1000,
            adam_beta1=0.9,
            adam_beta2=0.999,
            lr_decay_steps=0,
            n_restart_cycles=1,
            autocast=True,
            dead_feature_window=1000,
            feature_sampling_window=2000,
            n_checkpoints=0,
            checkpoint_path=None,
            save_final_checkpoint=False,
            logger=LoggingConfig(log_to_wandb=True),
        )

        trainer = SAETrainer(
            cfg=trainer_cfg,
            sae=self.sae,
            data_provider=data_provider,
        )

        trained_sae = trainer.fit()

        # Save the trained SAE
        if self.cfg.output_path is not None:
            trained_sae.save_inference_model(self.cfg.output_path)

        return trained_sae
```

## Tips for Custom Runners

1. **Use `mixing_buffer`** for any sequential data source to improve training stability

2. **Match batch sizes carefully**: The `train_batch_size_samples` in `SAETrainerConfig` should match the batch size yielded by your data provider

3. **Handle activation scaling**: If using `normalize_activations="expected_average_only_in"` in your SAE config, the trainer will automatically estimate and apply scaling

4. **Monitor dead features**: The trainer tracks feature activation frequency and logs dead feature counts to W&B

5. **For custom SAE architectures**: See the [Custom SAEs](custom_saes.md) guide for creating new SAE types

## Summary

| Component            | Purpose                                   | Required?   |
| -------------------- | ----------------------------------------- | ----------- |
| `SAETrainer`         | Core training loop                        | Yes         |
| `SAETrainerConfig`   | Training hyperparameters                  | Yes         |
| `TrainingSAE`        | The SAE to train                          | Yes         |
| `DataProvider`       | Iterator yielding `(batch, d_in)` tensors | Yes         |
| `Evaluator`          | Periodic evaluation callback              | No          |
| `mixing_buffer`      | Shuffle activations                       | Recommended |
| `save_checkpoint_fn` | Custom checkpoint logic                   | No          |

For most custom scenarios, you'll create a runner class that:

1. Loads your model
2. Creates a `TrainingSAE` with your desired architecture
3. Implements activation extraction with hooks
4. Optionally uses `mixing_buffer` to shuffle activations
5. Creates an `SAETrainer` and calls `.fit()` to train the SAE
6. Saves the trained inference SAE
