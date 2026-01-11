from time import perf_counter

import torch

from sae_lens.synthetic import (
    ActivationGenerator,
    FeatureDictionary,
    HierarchyNode,
    generate_random_low_rank_correlation_matrix,
    hierarchy_modifier,
    orthogonal_initializer,
    zipfian_firing_probabilities,
)


def create_hierarchical_synthetic_setup(
    num_features: int,
    hidden_dim: int = 512,
    branching_factor: int = 4,
    mutual_exclusion: bool = True,
    device: str = "cpu",
    correlation_rank: int = 1000,
    correlation_scale: float = 0.1,
):
    """
    Create a FeatureDictionary and ActivationGenerator with hierarchical features.

    Args:
        num_features: Approximate number of features (actual may vary due to tree structure)
        hidden_dim: Dimension of the hidden activations
        branching_factor: Number of children per node in the hierarchy
        mutual_exclusion: Whether siblings in the hierarchy are mutually exclusive
        device: Device to use
        correlation_rank: Rank of the low-rank correlation matrix
        correlation_scale: Scale for correlation matrix generation

    Returns:
        Tuple of (FeatureDictionary, ActivationGenerator, actual_num_features)
    """
    depth = 0
    while True:
        total = (branching_factor ** (depth + 1) - 1) // (branching_factor - 1)
        if total >= num_features:
            break
        depth += 1

    def build_tree(current_depth: int, start_idx: int) -> tuple[HierarchyNode, int]:
        if current_depth == depth:
            return HierarchyNode(feature_index=start_idx), start_idx + 1

        children = []
        next_idx = start_idx + 1
        for _ in range(branching_factor):
            child, next_idx = build_tree(current_depth + 1, next_idx)
            children.append(child)

        return HierarchyNode(
            feature_index=start_idx,
            children=children,
            mutually_exclusive_children=mutual_exclusion and len(children) >= 2,
        ), next_idx

    root, actual_num_features = build_tree(0, 0)
    modifier = hierarchy_modifier([root])

    firing_probabilities = zipfian_firing_probabilities(
        actual_num_features, min_prob=0.05
    )

    lr_correlation_matrix = generate_random_low_rank_correlation_matrix(
        num_features=actual_num_features,
        rank=correlation_rank,
        correlation_scale=correlation_scale,
        device=device,
    )

    generator = ActivationGenerator(
        num_features=actual_num_features,
        firing_probabilities=firing_probabilities,
        std_firing_magnitudes=0.2,
        mean_firing_magnitudes=1.0,
        modify_activations=modifier,
        correlation_matrix=lr_correlation_matrix,
        device=device,
    )

    feature_dict = FeatureDictionary(
        num_features=actual_num_features,
        hidden_dim=hidden_dim,
        bias=False,
        device=device,
        initializer=orthogonal_initializer(show_progress=False, num_steps=50),
    )

    return feature_dict, generator, actual_num_features


# Run with: poetry run pytest benchmark/test_synthetic_hierarchy.py -v -s
def test_benchmark_hierarchical_synthetic_pipeline():
    num_features = 10_000
    hidden_dim = 1024
    batch_size = 2048
    num_iterations = 5

    torch.set_grad_enabled(True)
    print(
        f"\nBenchmarking: {num_features:,} features, hidden_dim={hidden_dim}, batch_size={batch_size}"
    )

    setup_start = perf_counter()
    feature_dict, generator, actual_features = create_hierarchical_synthetic_setup(
        num_features=num_features,
        hidden_dim=hidden_dim,
        branching_factor=4,
        mutual_exclusion=True,
        device="cpu",
        correlation_rank=1000,
        correlation_scale=0.1,
    )
    setup_duration = perf_counter() - setup_start
    print(f"Setup time: {setup_duration:.4f}s")
    print(f"Actual features: {actual_features:,}")

    assert actual_features >= num_features

    # Warmup
    samples = generator.sample(batch_size)
    hidden = feature_dict(samples)

    # Benchmark sample generation
    gen_start = perf_counter()
    for _ in range(num_iterations):
        samples = generator.sample(batch_size)
    gen_duration = perf_counter() - gen_start
    print(f"Sample generation time per call: {gen_duration / num_iterations:.4f}s")

    # Benchmark full pipeline
    full_start = perf_counter()
    for _ in range(num_iterations):
        samples = generator.sample(batch_size)
        hidden = feature_dict(samples)
    full_duration = perf_counter() - full_start
    print(f"Full pipeline time per call: {full_duration / num_iterations:.4f}s")

    # Verify outputs
    assert samples.shape == (batch_size, actual_features)
    assert hidden.shape == (batch_size, hidden_dim)
    assert torch.all(samples >= 0)

    mean_active = (samples > 0).float().sum(dim=1).mean().item()
    print(f"Mean features active per sample: {mean_active:.1f}")

    assert mean_active > 0
    assert mean_active < actual_features
