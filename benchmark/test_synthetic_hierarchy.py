from collections import deque
from time import perf_counter

import torch

from sae_lens.synthetic import (
    ActivationGenerator,
    FeatureDictionary,
    HierarchyNode,
    generate_random_low_rank_correlation_matrix,
    hierarchy_modifier,
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
    use_sparse_tensors: bool = False,
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
    # Build tree structure BFS: branching_factor roots, each with branching_factor children
    roots_count = min(branching_factor, num_features)
    parent_to_children: dict[int, list[int]] = {}

    queue: deque[int] = deque(range(roots_count))
    child_idx = roots_count

    while queue and child_idx < num_features:
        parent_idx = queue.popleft()
        children_indices = []
        for _ in range(branching_factor):
            if child_idx >= num_features:
                break
            children_indices.append(child_idx)
            queue.append(child_idx)
            child_idx += 1
        if children_indices:
            parent_to_children[parent_idx] = children_indices

    # Build nodes bottom-up so children exist when parent is created
    nodes: dict[int, HierarchyNode] = {}
    for idx in range(num_features - 1, -1, -1):
        if idx in parent_to_children:
            children = [nodes[c] for c in parent_to_children[idx]]
            nodes[idx] = HierarchyNode(
                feature_index=idx,
                children=children,
                mutually_exclusive_children=mutual_exclusion and len(children) >= 2,
            )
        else:
            nodes[idx] = HierarchyNode(feature_index=idx)

    roots = [nodes[i] for i in range(roots_count)]
    actual_num_features = num_features
    modifier_start_time = perf_counter()
    modifier = hierarchy_modifier(roots)
    modifier_end_time = perf_counter()
    print(f"Modifier time: {modifier_end_time - modifier_start_time:.4f}s")

    firing_probabilities = zipfian_firing_probabilities(
        actual_num_features, min_prob=0.05
    )

    correlation_start_time = perf_counter()
    lr_correlation_matrix = generate_random_low_rank_correlation_matrix(
        num_features=actual_num_features,
        rank=correlation_rank,
        correlation_scale=correlation_scale,
        device=device,
    )
    correlation_end_time = perf_counter()
    print(f"Correlation time: {correlation_end_time - correlation_start_time:.4f}s")

    generator = ActivationGenerator(
        num_features=actual_num_features,
        firing_probabilities=firing_probabilities,
        std_firing_magnitudes=0.2,
        mean_firing_magnitudes=1.0,
        modify_activations=modifier,
        correlation_matrix=lr_correlation_matrix,
        device=device,
        use_sparse_tensors=use_sparse_tensors,
    )

    generator_end_time = perf_counter()
    print(f"Generator time: {generator_end_time - correlation_end_time:.4f}s")

    feature_dict = FeatureDictionary(
        num_features=actual_num_features,
        hidden_dim=hidden_dim,
        bias=False,
        device=device,
        # initializer=orthogonal_initializer(show_progress=True, num_steps=10),
        initializer=None,
    )

    feature_dict_end_time = perf_counter()
    print(f"Feature dictionary time: {feature_dict_end_time - generator_end_time:.4f}s")

    return feature_dict, generator, actual_num_features


# Run with: poetry run pytest benchmark/test_synthetic_hierarchy.py -v -s
def test_benchmark_hierarchical_synthetic_pipeline():
    num_features = 1_000_000
    hidden_dim = 1024
    batch_size = 2500
    # we want to see how long 1M samples takes to generate
    num_iterations = 1_000_000 // batch_size

    torch.set_grad_enabled(True)
    print(
        f"\nBenchmarking: {num_features:,} features, hidden_dim={hidden_dim}, batch_size={batch_size}"
    )

    setup_start = perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dict, generator, actual_features = create_hierarchical_synthetic_setup(
        num_features=num_features,
        hidden_dim=hidden_dim,
        branching_factor=1000,
        mutual_exclusion=True,
        device=device,
        correlation_rank=100,
        correlation_scale=0.1,
        use_sparse_tensors=True,
    )
    setup_duration = perf_counter() - setup_start
    print(f"Setup time: {setup_duration:.4f}s")
    print(f"Actual features: {actual_features:,}")

    assert actual_features >= num_features

    # Warmup
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        samples = generator.sample(batch_size)
        hidden = feature_dict(samples)
        # hidden = torch.zeros(batch_size, hidden_dim, device=device)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark sample generation
    gen_start = perf_counter()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        for _ in range(num_iterations):
            samples = generator.sample(batch_size)
    if device == "cuda":
        torch.cuda.synchronize()

    gen_duration = perf_counter() - gen_start
    print(f"Sample generation time per call: {gen_duration / num_iterations:.4f}s")

    # Benchmark full pipeline
    full_start = perf_counter()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        for _ in range(num_iterations):
            samples = generator.sample(batch_size)
            hidden = feature_dict(samples)
            # hidden = torch.zeros(batch_size, hidden_dim, device=device) # feature_dict(samples)
    if device == "cuda":
        torch.cuda.synchronize()

    full_duration = perf_counter() - full_start
    print(
        f"Full pipeline for {num_iterations * batch_size} samples time: {full_duration:.4f}s"
    )
    print(f"Full pipeline time per call: {full_duration / num_iterations:.4f}s")

    # Verify outputs
    assert samples.shape == (batch_size, actual_features)
    assert hidden.shape == (batch_size, hidden_dim)
    assert torch.all(samples >= 0)

    mean_active = (samples > 0).float().sum(dim=1).mean().item()
    print(f"Mean features active per sample: {mean_active:.1f}")

    assert mean_active > 0
    assert mean_active < actual_features
