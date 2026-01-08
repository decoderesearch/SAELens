# Training SAEs on Synthetic Data

Training SAEs on synthetic data allows you to work with a known ground truth, enabling precise evaluation of how well your SAE recovers the true underlying features. This is useful for:

- **Controlled experiments**: Test SAE architectures and hyperparameters with known feature structures
- **Fast iteration**: Train on CPU in under a minute with small models
- **Algorithm development**: Benchmark new training methods against ground truth

For a hands-on walkthrough, see the [tutorial notebook](https://github.com/decoderesearch/SAELens/blob/main/tutorials/training_saes_on_synthetic_data.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/decoderesearch/SAELens/blob/main/tutorials/training_saes_on_synthetic_data.ipynb).

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
