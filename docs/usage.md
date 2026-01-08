# Usage Guide

This guide covers how to use SAEs for inference and analysis. For training SAEs, see [Training SAEs](training_saes.md).

## Loading SAEs

### From Pretrained (Hugging Face)

Load SAEs from the SAELens registry or any Hugging Face repository with the `saelens` tag.

```python
from sae_lens import SAE

# Load from SAELens registry
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)

# Load from any Hugging Face repo with saelens tag
sae = SAE.from_pretrained(
    release="your-username/your-sae-repo",
    sae_id="path/to/sae",
    device="cuda"
)
```

See [Pretrained SAEs](pretrained_saes/index.md) for a full list of available SAEs.

### From Disk

Load SAEs that you've trained yourself or downloaded manually.

```python
from sae_lens import SAE

sae = SAE.load_from_disk(
    path="/path/to/your/sae",
    device="cuda"
)
```

### Loading with Config and Sparsity

If you need the configuration dictionary and sparsity information alongside the SAE:

```python
from sae_lens import SAE

sae, cfg_dict, log_sparsities = SAE.from_pretrained_with_cfg_and_sparsity(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)
```

## Running SAEs Directly

The SAE class provides three main methods for inference: `encode()`, `decode()`, and `forward()`.

### Encode

Convert activations to sparse feature representations.

```python
import torch
from sae_lens import SAE

sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)

# activations shape: (batch, seq_len, d_model)
activations = torch.randn(1, 128, 768, device="cuda")

# feature_acts shape: (batch, seq_len, d_sae)
feature_acts = sae.encode(activations)

# Check which features are active
active_features = (feature_acts > 0).sum(dim=-1)
print(f"Average L0: {active_features.float().mean().item()}")
```

### Decode

Convert sparse feature representations back to activation space.

```python
# Reconstruct activations from features
reconstructed = sae.decode(feature_acts)

# Compute reconstruction error
mse = (activations - reconstructed).pow(2).mean()
print(f"Reconstruction MSE: {mse.item()}")
```

### Forward

Run the full SAE pipeline (encode + decode) in one call.

```python
# Equivalent to sae.decode(sae.encode(activations))
reconstructed = sae.forward(activations)

# Or simply call the SAE directly
reconstructed = sae(activations)
```

## Using HookedSAETransformer

HookedSAETransformer extends TransformerLens's HookedTransformer to seamlessly integrate SAEs into the model's forward pass.

### Setup

```python
from sae_lens import SAE, HookedSAETransformer

# Load model
model = HookedSAETransformer.from_pretrained("gpt2-small", device="cuda")

# Load SAE
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)
```

### Run with SAEs (Temporary)

Run a forward pass with SAEs attached temporarily. SAEs are removed after the forward pass.

```python
tokens = model.to_tokens("Hello, world!")

# Run with SAE - SAE is removed after this call
logits = model.run_with_saes(tokens, saes=[sae])
```

### Run with Cache and SAEs

Cache activations including SAE feature activations.

```python
logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])

# Access SAE feature activations
sae_acts = cache["blocks.8.hook_resid_pre.hook_sae_acts_post"]
print(f"SAE activations shape: {sae_acts.shape}")
```

### Run with Hooks and SAEs

Intervene on SAE activations during the forward pass.

```python
from functools import partial

def ablate_feature(sae_acts, hook, feature_id):
    sae_acts[:, :, feature_id] = 0.0
    return sae_acts

# Ablate feature 1000 during forward pass
logits = model.run_with_hooks_with_saes(
    tokens,
    saes=[sae],
    fwd_hooks=[
        ("blocks.8.hook_resid_pre.hook_sae_acts_post",
         partial(ablate_feature, feature_id=1000))
    ]
)
```

### Add SAEs (Persistent)

Permanently attach SAEs to the model until explicitly removed.

```python
# Add SAE permanently
model.add_sae(sae)

# Now standard forward passes include the SAE
logits = model(tokens)
logits, cache = model.run_with_cache(tokens)

# Remove all attached SAEs
model.reset_saes()

# Or remove specific SAEs
model.reset_saes(act_names=["blocks.8.hook_resid_pre"])
```

### Using Error Terms

Include error terms to preserve original model behavior while accessing SAE features.

```python
sae.use_error_term = True
model.add_sae(sae)

# Output is now: SAE(x) + error_term = x (original activation)
logits = model(tokens)

# You can intervene on the error term
logits = model.run_with_hooks(
    tokens,
    fwd_hooks=[
        ("blocks.8.hook_resid_pre.hook_sae_error",
         lambda act, hook: torch.zeros_like(act))
    ]
)
```

## Using SAEs Without TransformerLens

SAEs from SAELens are standard PyTorch modules and can be used with any model or framework. The key is extracting activations from your model and passing them to the SAE's `encode()`, `decode()`, or `forward()` methods.

### Pure PyTorch with Hugging Face Transformers

Use standard PyTorch hooks to extract activations from Hugging Face models.

```python
import torch
from transformers import GPT2Model, GPT2Tokenizer
from sae_lens import SAE

# Load Hugging Face model
hf_model = GPT2Model.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
hf_model.eval()

# Load SAE (trained on GPT-2 residual stream at layer 8)
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cpu"
)

# Storage for activations
activations = {}

def hook_fn(module, input, output):
    # GPT-2 transformer blocks output a tuple; hidden states are first
    hidden_states = output[0] if isinstance(output, tuple) else output
    activations["layer_8"] = hidden_states.detach()

# Register hook on layer 8 (before layer 9's processing = "resid_pre" at layer 8)
handle = hf_model.h[8].register_forward_hook(hook_fn)

# Run forward pass
inputs = tokenizer("Hello, world!", return_tensors="pt")
with torch.no_grad():
    hf_model(**inputs)

# Remove hook
handle.remove()

# Use SAE on extracted activations
layer_8_acts = activations["layer_8"]
feature_acts = sae.encode(layer_8_acts)
reconstructed = sae.decode(feature_acts)

print(f"Input shape: {layer_8_acts.shape}")
print(f"Feature activations shape: {feature_acts.shape}")
print(f"Active features per token: {(feature_acts > 0).sum(dim=-1)}")
```

### Full Example: Analyzing Features with Hugging Face

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sae_lens import SAE

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# Load SAE
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device=device
)

def get_sae_features(text, layer=8):
    """Extract SAE features for a given text."""
    activations = {}

    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        activations["hidden"] = hidden_states.detach()

    handle = model.transformer.h[layer].register_forward_hook(hook_fn)

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)

    handle.remove()

    feature_acts = sae.encode(activations["hidden"])
    return feature_acts, inputs["input_ids"]

# Analyze a prompt
text = "The capital of France is"
features, tokens = get_sae_features(text)

# Find top active features at the last token
last_token_features = features[0, -1, :]
top_features = torch.topk(last_token_features, k=10)

print(f"Top 10 active features at last token:")
for idx, (feat_idx, value) in enumerate(zip(top_features.indices, top_features.values)):
    print(f"  Feature {feat_idx.item()}: {value.item():.4f}")
```

### Using SAEs with nnsight

[nnsight](https://nnsight.net/) provides a clean interface for model interventions. SAEs integrate naturally with nnsight's tracing API.

```python
import torch
from nnsight import LanguageModel
from sae_lens import SAE

# Load model with nnsight
model = LanguageModel("openai-community/gpt2", device_map="auto")

# Load SAE
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)

prompt = "The Eiffel Tower is located in"

# Extract activations and compute SAE features
with model.trace(prompt) as tracer:
    # Access hidden states at layer 8
    hidden_states = model.transformer.h[8].output[0]

    # Save the hidden states
    hidden_states_saved = hidden_states.save()

# Get SAE features outside the trace
with torch.no_grad():
    features = sae.encode(hidden_states_saved.value)

print(f"Feature activations shape: {features.shape}")
print(f"Average L0: {(features > 0).sum(dim=-1).float().mean().item():.1f}")
```

### Intervening on SAE Features with nnsight

```python
import torch
from nnsight import LanguageModel
from sae_lens import SAE

model = LanguageModel("openai-community/gpt2", device_map="auto")

sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)

prompt = "The Eiffel Tower is located in"

def ablate_top_features(hidden_states, sae, k=10):
    """Ablate the top-k active features and return modified activations."""
    features = sae.encode(hidden_states)

    # Find and ablate top-k features at each position
    for pos in range(features.shape[1]):
        top_k = torch.topk(features[0, pos], k=k)
        features[0, pos, top_k.indices] = 0.0

    # Reconstruct with ablated features
    return sae.decode(features)

# Run with intervention
with model.trace(prompt) as tracer:
    # Get hidden states
    hidden_states = model.transformer.h[8].output[0]

    # Modify using SAE
    modified = ablate_top_features(hidden_states, sae, k=10)

    # Replace the hidden states
    model.transformer.h[8].output[0][:] = modified

    # Get output logits
    logits = model.lm_head.output.save()

print(f"Output shape: {logits.value.shape}")
```

### Steering with SAE Features using nnsight

```python
import torch
from nnsight import LanguageModel
from sae_lens import SAE

model = LanguageModel("openai-community/gpt2", device_map="auto")

sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)

prompt = "I think this movie is"

# Identify a feature to amplify (you would find this through analysis)
feature_idx = 1000  # Example feature index
steering_strength = 5.0

def steer_with_feature(hidden_states, sae, feature_idx, strength):
    """Add a steering vector based on an SAE feature direction."""
    # Get the decoder direction for this feature
    steering_vector = sae.W_dec[feature_idx]

    # Add to hidden states (broadcast across sequence)
    return hidden_states + strength * steering_vector

with model.trace(prompt) as tracer:
    hidden_states = model.transformer.h[8].output[0]

    # Apply steering
    steered = steer_with_feature(hidden_states, sae, feature_idx, steering_strength)
    model.transformer.h[8].output[0][:] = steered

    logits = model.lm_head.output.save()

# Decode the most likely next tokens
next_token_logits = logits.value[0, -1, :]
top_tokens = torch.topk(next_token_logits, k=5)
tokenizer = model.tokenizer
print("Top predicted tokens:")
for token_id, logit in zip(top_tokens.indices, top_tokens.values):
    print(f"  {tokenizer.decode([token_id])!r}: {logit.item():.2f}")
```

## Key SAE Attributes

After loading an SAE, you can access useful configuration and metadata:

```python
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)

# Model dimensions
print(f"Input dimension (d_in): {sae.cfg.d_in}")
print(f"SAE dimension (d_sae): {sae.cfg.d_sae}")
print(f"Expansion factor: {sae.cfg.d_sae / sae.cfg.d_in}")

# Metadata about the SAE
print(f"Hook name: {sae.cfg.metadata.hook_name}")
print(f"Model name: {sae.cfg.metadata.model_name}")
print(f"Context size: {sae.cfg.metadata.context_size}")

# Weights
print(f"Encoder weights shape: {sae.W_enc.shape}")  # (d_in, d_sae)
print(f"Decoder weights shape: {sae.W_dec.shape}")  # (d_sae, d_in)
print(f"Decoder bias shape: {sae.b_dec.shape}")     # (d_in,)
```
