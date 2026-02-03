# Training SAEs on Synthetic Data

Training SAEs on synthetic data allows you to work with a known ground truth, enabling precise evaluation of how well your SAE recovers the true underlying features. This is useful for:

- **Controlled experiments**: Test SAE architectures and hyperparameters with known feature structures
- **Fast iteration**: Train on CPU in under a minute with small models
- **Algorithm development**: Benchmark new training methods against ground truth

For a hands-on walkthrough, see the [tutorial notebook](https://github.com/decoderesearch/SAELens/blob/main/tutorials/training_saes_on_synthetic_data.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/decoderesearch/SAELens/blob/main/tutorials/training_saes_on_synthetic_data.ipynb).

<!-- prettier-ignore-start -->
!!! info "Beta feature"
    The synthetic data utilities should be considered in beta, and their API and functionality may change over the next few months. If this is a concern, we recommend pinning your SAELens version to avoid breaking changes.
<!-- prettier-ignore-end -->

## Core Concepts

### Feature Dictionary

A [FeatureDictionary][sae_lens.synthetic.FeatureDictionary] maps sparse feature activations to dense hidden activations. It stores a matrix of feature vectors and computes `hidden = features @ feature_vectors + bias`.

```python
from sae_lens.synthetic import FeatureDictionary, orthogonal_initializer

# Create dictionary with 16 features in 32-dimensional space
feature_dict = FeatureDictionary(
    num_features=16,
    hidden_dim=32,
    initializer=orthogonal_initializer(),  # Makes features orthogonal
)
```

Use `orthogonal_initializer()` to create features that don't overlap, making it easier to evaluate SAE performance.

### Activation Generator

An [ActivationGenerator][sae_lens.synthetic.ActivationGenerator] samples sparse feature activations with controlled firing probabilities.

```python
from sae_lens.synthetic import ActivationGenerator
import torch

firing_probs = torch.ones(16) * 0.25  # Each feature fires 25% of the time

activation_gen = ActivationGenerator(
    num_features=16,
    firing_probabilities=firing_probs,
)

# Sample a batch of sparse feature activations
feature_activations = activation_gen.sample(batch_size=1024)
```

## Basic Training Workflow

Use [train_toy_sae][sae_lens.synthetic.train_toy_sae] to train an SAE on synthetic data:

```python
from sae_lens.synthetic import (
    FeatureDictionary,
    ActivationGenerator,
    train_toy_sae,
)
from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig
import torch

# 1. Create feature dictionary and activation generator
feature_dict = FeatureDictionary(num_features=16, hidden_dim=32)
activation_gen = ActivationGenerator(
    num_features=16,
    firing_probabilities=torch.ones(16) * 0.25,
)

# 2. Configure SAE
cfg = StandardTrainingSAEConfig(
    d_in=feature_dict.hidden_dim,
    d_sae=feature_dict.num_features,
    l1_coefficient=5e-2,
)
sae = StandardTrainingSAE(cfg)

# 3. Train
train_toy_sae(sae, feature_dict, activation_gen)
```

## Evaluation

Use [eval_sae_on_synthetic_data][sae_lens.synthetic.eval_sae_on_synthetic_data] to measure how well the SAE recovers the true features:

```python
from sae_lens.synthetic import eval_sae_on_synthetic_data

result = eval_sae_on_synthetic_data(sae, feature_dict, activation_gen)
print(f"MCC: {result.mcc:.3f}")  # Mean Correlation Coefficient
print(f"L0: {result.sae_l0:.1f}")  # Average active latents
print(f"Dead latents: {result.dead_latents}")
print(f"Shrinkage: {result.shrinkage:.3f}")
```

### Metrics

- **MCC (Mean Correlation Coefficient)**: Measures alignment between SAE decoder weights and true feature vectors. Uses the Hungarian algorithm to find the optimal one-to-one matching, then computes mean absolute cosine similarity. Range [0, 1] where 1 = perfect recovery.
- **L0**: Average number of active SAE latents per sample. Compare to `true_l0` to check if sparsity matches.
- **Dead latents**: Number of SAE latents that never activate. High values indicate capacity issues.
- **Shrinkage**: Ratio of SAE output norm to input norm. Values below 1.0 indicate the SAE is shrinking reconstructions.

### Visualization

Use [plot_sae_feature_similarity][sae_lens.synthetic.plot_sae_feature_similarity] to visualize how SAE features align with ground truth:

```python
from sae_lens.synthetic import plot_sae_feature_similarity

plot_sae_feature_similarity(sae, feature_dict, reorder_sae_latents=True)
```

This creates a heatmap showing cosine similarity between each SAE latent and each true feature.

## Realistic Data Properties

### Firing Probability Distributions

Real neural network features follow power-law distributions where few features fire frequently and most fire rarely. Use [zipfian_firing_probabilities][sae_lens.synthetic.zipfian_firing_probabilities]:

```python
from sae_lens.synthetic import zipfian_firing_probabilities

# Power-law distribution: some features common, most rare
firing_probs = zipfian_firing_probabilities(
    num_features=16,
    exponent=1.0,
    max_prob=0.5,
    min_prob=0.01,
)
```

Other options:
- [linear_firing_probabilities][sae_lens.synthetic.linear_firing_probabilities]: Linearly decreasing from max to min
- [random_firing_probabilities][sae_lens.synthetic.random_firing_probabilities]: Uniform random within bounds

### Feature Correlations

Features in real networks often co-occur or anti-occur. Add correlations with [generate_random_correlation_matrix][sae_lens.synthetic.generate_random_correlation_matrix]:

```python
from sae_lens.synthetic import generate_random_correlation_matrix

correlation_matrix = generate_random_correlation_matrix(
    num_features=16,
    uncorrelated_ratio=0.3,        # 30% of pairs have no correlation
    positive_ratio=0.7,            # 70% of correlations are positive
    min_correlation_strength=0.3,
    max_correlation_strength=0.8,
)

activation_gen = ActivationGenerator(
    num_features=16,
    firing_probabilities=firing_probs,
    correlation_matrix=correlation_matrix,
)
```

### Hierarchical Features

Model parent-child feature relationships where children can only fire when parents are active. Use [HierarchyNode][sae_lens.synthetic.HierarchyNode]:

```python
from sae_lens.synthetic import HierarchyNode, hierarchy_modifier

# Feature 0 is parent of features 1 and 2
# Feature 1 is parent of feature 3
hierarchy = HierarchyNode.from_dict({
    0: {
        1: {3: {}},
        2: {},
    }
})

modifier = hierarchy_modifier(hierarchy)

activation_gen = ActivationGenerator(
    num_features=4,
    firing_probabilities=torch.ones(4) * 0.5,
    modify_activations=modifier,
)
```

With hierarchies, you may observe **feature absorption**: when a child always fires with its parent, the SAE learns to encode both in a single latent.

## Advanced Topics

### Superposition

Create superposition by having more features than hidden dimensions:

```python
# 32 features in 16-dimensional space = 2x superposition
feature_dict = FeatureDictionary(num_features=32, hidden_dim=16)
```

With superposition, features must share directions, making recovery harder. The `orthogonal_initializer()` can only make features approximately orthogonal when `num_features > hidden_dim`.

### Custom Activation Modifiers

Create custom modifiers to implement arbitrary activation transformations. A modifier is a function `(activations: torch.Tensor) -> torch.Tensor`:

```python
from sae_lens.synthetic import ActivationsModifier

def my_modifier(activations: torch.Tensor) -> torch.Tensor:
    # Example: zero out feature 0 when feature 1 is active
    result = activations.clone()
    mask = activations[:, 1] > 0
    result[mask, 0] = 0
    return result

activation_gen = ActivationGenerator(
    num_features=16,
    firing_probabilities=firing_probs,
    modify_activations=my_modifier,
)
```

Pass a list of modifiers to apply them in sequence.


## Large-Scale Training with SyntheticModel

For training SAEs on larger synthetic datasets with features like checkpointing, wandb logging, and HuggingFace integration, use [SyntheticModel][sae_lens.synthetic.SyntheticModel] and [SyntheticSAERunner][sae_lens.synthetic.SyntheticSAERunner].

### SyntheticModel

[SyntheticModel][sae_lens.synthetic.SyntheticModel] combines all synthetic data components into a single, configurable model:

```python
from sae_lens.synthetic import SyntheticModel, SyntheticModelConfig

cfg = SyntheticModelConfig(
    num_features=10_000,
    hidden_dim=512,
)

model = SyntheticModel(cfg)

# Generate training data
hidden_activations = model.sample(batch_size=1024)

# Or get both hidden activations and ground-truth features
hidden_acts, feature_acts = model.sample_with_features(batch_size=1024)
```

### SyntheticModelConfig

[SyntheticModelConfig][sae_lens.synthetic.SyntheticModelConfig] provides declarative configuration for all model properties:

```python
from sae_lens.synthetic import (
    SyntheticModelConfig,
    ZipfianFiringProbabilityConfig,
    HierarchyConfig,
    OrthogonalizationConfig,
    LowRankCorrelationConfig,
    LinearMagnitudeConfig,
)

cfg = SyntheticModelConfig(
    num_features=10_000,
    hidden_dim=512,

    # Firing probability distribution
    firing_probability=ZipfianFiringProbabilityConfig(
        exponent=1.0,
        max_prob=0.3,
        min_prob=0.01,
    ),

    # Hierarchical feature structure
    hierarchy=HierarchyConfig(
        total_root_nodes=100,
        branching_factor=10,
        max_depth=2,
        mutually_exclusive_portion=0.3,
    ),

    # Feature orthogonalization
    orthogonalization=OrthogonalizationConfig(
        num_steps=200,
        lr=0.01,
    ),

    # Feature correlations
    correlation=LowRankCorrelationConfig(
        rank=32,
        correlation_scale=0.1,
    ),

    # Per-feature magnitude variation
    mean_firing_magnitudes=LinearMagnitudeConfig(start=0.5, end=2.0),
    std_firing_magnitudes=0.1,

    # Reproducibility
    seed=42,
)

model = SyntheticModel(cfg, device="cuda")
```

### Automatic Hierarchy Generation

Use [HierarchyConfig][sae_lens.synthetic.HierarchyConfig] to automatically generate hierarchical feature structures:

```python
from sae_lens.synthetic import HierarchyConfig

hierarchy_cfg = HierarchyConfig(
    total_root_nodes=100,           # Number of root features
    branching_factor=10,            # Children per parent (or tuple for range)
    max_depth=2,                    # Maximum tree depth
    mutually_exclusive_portion=0.3, # Fraction of parents with ME children
    mutually_exclusive_min_depth=0, # Minimum depth for ME
    compensate_probabilities=False,  # Adjust probs for hierarchy effects
)
```

With `compensate_probabilities=True`, firing probabilities are scaled up to compensate for the reduction caused by hierarchy constraints (children only fire when parents fire). This setting likely only makes sense when using a Zipfian firing probability distribution.

### Per-Feature Magnitude Distributions

Configure how firing magnitudes vary across features:

```python
from sae_lens.synthetic import (
    ConstantMagnitudeConfig,
    LinearMagnitudeConfig,
    ExponentialMagnitudeConfig,
    FoldedNormalMagnitudeConfig,
)

# All features have magnitude 1.0
constant = ConstantMagnitudeConfig(value=1.0)

# Linear interpolation from 0.5 to 2.0 across features
linear = LinearMagnitudeConfig(start=0.5, end=2.0)

# Exponential interpolation
exponential = ExponentialMagnitudeConfig(start=0.1, end=10.0)

# Random magnitudes from folded normal distribution
random = FoldedNormalMagnitudeConfig(mean=1.0, std=0.3)
```

### Training with SyntheticSAERunner

[SyntheticSAERunner][sae_lens.synthetic.SyntheticSAERunner] provides full training infrastructure:

```python
from sae_lens.synthetic import SyntheticSAERunner, SyntheticSAERunnerConfig
from sae_lens import StandardTrainingSAEConfig

runner_cfg = SyntheticSAERunnerConfig(
    synthetic_model=SyntheticModelConfig(
        num_features=10_000,
        hidden_dim=512,
    ),

    sae=StandardTrainingSAEConfig(
        d_in=512,  # Must match hidden_dim
        d_sae=16_000,
        l1_coefficient=5e-3,
    ),

    # Training parameters
    training_samples=100_000_000,
    batch_size=4096,
    lr=3e-4,

    # Checkpointing
    n_checkpoints=5,
    checkpoint_path="checkpoints",
    output_path="output",

    # Evaluation
    eval_frequency=1000,  # Evaluate MCC every N steps
    eval_samples=100_000,
)

runner = SyntheticSAERunner(runner_cfg)
result = runner.run()

print(f"Final MCC: {result.final_eval.mcc:.3f}")
```

### Saving and Loading Models

Save and load synthetic models for reproducibility:

```python
# Save to disk
model.save("./my_synthetic_model")

# Load from disk
model = SyntheticModel.load_from_disk("./my_synthetic_model")

# Smart loading from various sources
model = SyntheticModel.load_from_source(cfg)  # From config
model = SyntheticModel.load_from_source("./path")  # From disk
model = SyntheticModel.load_from_source("username/repo")  # From HuggingFace
```

### HuggingFace Integration

Share synthetic models via HuggingFace Hub:

```python
from sae_lens.synthetic import (
    SyntheticModel,
    upload_synthetic_model_to_huggingface,
)

# Upload a model
upload_synthetic_model_to_huggingface(
    model=model,  # Or path to saved model
    hf_repo_id="username/my-synthetic-model",
)

# Load from HuggingFace
model = SyntheticModel.from_pretrained("username/my-synthetic-model")

# Load from a subfolder in a repo
model = SyntheticModel.from_pretrained(
    "username/repo",
    model_path="models/large",
)
```

### Using Pretrained Synthetic Models with Runner

Load a pretrained synthetic model for training:

```python
runner_cfg = SyntheticSAERunnerConfig(
    # Load from HuggingFace
    synthetic_model="username/my-synthetic-model",

    # Or from disk
    # synthetic_model="./path/to/model",

    sae=StandardTrainingSAEConfig(
        d_in=512,
        d_sae=16_000,
        l1_coefficient=5e-3,
    ),
    training_samples=100_000_000,
)
```