<!-- prettier-ignore-start -->
!!! tip "SAELens v6"
    SAELens 6.0.0 is live with changes to SAE training and loading. Check out the [migration guide →](migrating)
<!-- prettier-ignore-end -->

<img width="1308" height="532" alt="saes_pic" src="https://github.com/user-attachments/assets/2a5d752f-b261-4ee4-ad5d-ebf282321371" />

# SAELens

[![PyPI](https://img.shields.io/pypi/v/sae-lens?color=blue)](https://pypi.org/project/sae-lens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![build](https://github.com/decoderesearch/SAELens/actions/workflows/build.yml/badge.svg)](https://github.com/decoderesearch/SAELens/actions/workflows/build.yml)
[![Deploy Docs](https://github.com/decoderesearch/SAELens/actions/workflows/deploy_docs.yml/badge.svg)](https://github.com/decoderesearch/SAELens/actions/workflows/deploy_docs.yml)
[![codecov](https://codecov.io/gh/decoderesearch/SAELens/graph/badge.svg?token=N83NGH8CGE)](https://codecov.io/gh/decoderesearch/SAELens)

The SAELens training codebase exists to help researchers:

- Train sparse autoencoders.
- Analyse sparse autoencoders and neural network internals.
- Generate insights which make it easier to create safe and aligned AI systems.

**Please note these docs are in beta. We intend to make them cleaner and more comprehensive over time.**

## Quick Start

### Installation

```
pip install sae-lens
```

### Loading Sparse Autoencoders from Huggingface

To load a pretrained sparse autoencoder, you can use `SAE.from_pretrained()` as below:

```python
from sae_lens import SAE

sae = SAE.from_pretrained(
    release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = "blocks.8.hook_resid_pre", # won't always be a hook point
    device = "cuda"
)
```

You can see other importable SAEs on [this page](https://decoderesearch.github.io/SAELens/sae_table/).

Any SAE on Huggingface that's trained using SAELens can also be loaded using `SAE.from_pretrained()`. In this case, `release` is the name of the Huggingface repo, and `sae_id` is the path to the SAE in the repo. You can see a list of SAEs listed on Huggingface with the [saelens tag](https://huggingface.co/models?library=saelens).

### Loading Sparse Autoencoders from Disk

To load a pretrained sparse autoencoder from disk that you've trained yourself, you can use `SAE.load_from_disk()` as below.

```python
from sae_lens import SAE

sae = SAE.load_from_disk("/path/to/your/sae", device="cuda")
```

### Importing SAEs from other libraries

You can import an SAE created with another library by writing a custom `PretrainedSaeHuggingfaceLoader` or `PretrainedSaeDiskLoader` for use with `SAE.from_pretrained()` or `SAE.load_from_disk()`, respectively. See the [pretrained_sae_loaders.py](https://github.com/decoderesearch/SAELens/blob/main/sae_lens/loading/pretrained_sae_loaders.py) file for more details, or ask on the [Open Source Mechanistic Interpretability Slack](https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-375zalm04-GFd5tdBU1yLKlu_T_JSqZQ). If you write a good custom loader for another library, please consider contributing it back to SAELens!

### Background and further Readings

We highly recommend this [tutorial](https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab).

For recent progress in SAEs, we recommend the LessWrong forum's [Sparse Autoencoder tag](https://www.lesswrong.com/tag/sparse-autoencoders-saes)

## Tutorials

I wrote a tutorial to show users how to do some basic exploration of their SAE:

- Loading and Analysing Pre-Trained Sparse Autoencoders [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/decoderesearch/SAELens/blob/main/tutorials/basic_loading_and_analysing.ipynb)
- Understanding SAE Features with the Logit Lens [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/decoderesearch/SAELens/blob/main/tutorials/logits_lens_with_features.ipynb)
- Training a Sparse Autoencoder [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/decoderesearch/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb)

## Example WandB Dashboard

WandB Dashboards provide lots of useful insights while training SAEs. Here's a screenshot from one training run.

![screenshot](dashboard_screenshot.png)

## Citation

```
@misc{bloom2024saetrainingcodebase,
   title = {SAELens},
   author = {Bloom, Joseph and Tigges, Curt and Duong, Anthony and Chanin, David},
   year = {2024},
   howpublished = {\url{https://github.com/decoderesearch/SAELens}},
}}
```
