import pytest
import torch

from sae_lens.toy_model import TreeFeatureGenerator


def test_TreeFeatureGenerator_simple_construction():
    root = TreeFeatureGenerator(active_prob=0.8)
    assert root.n_features == 1
    assert root.index == 0


def test_TreeFeatureGenerator_with_children():
    child1 = TreeFeatureGenerator(active_prob=0.5)
    child2 = TreeFeatureGenerator(active_prob=0.3)
    root = TreeFeatureGenerator(active_prob=0.8, children=[child1, child2])

    assert root.n_features == 3
    assert root.index == 0
    assert child1.index == 1
    assert child2.index == 2


def test_TreeFeatureGenerator_sample_shape():
    child = TreeFeatureGenerator(active_prob=0.5)
    root = TreeFeatureGenerator(active_prob=1.0, children=[child])

    samples = root.sample(batch_size=100)
    assert samples.shape == (100, 2)


def test_TreeFeatureGenerator_sample_respects_root_probability():
    root = TreeFeatureGenerator(active_prob=0.3)
    samples = root.sample(batch_size=10_000)

    actual_prob = samples[:, 0].mean().item()
    assert actual_prob == pytest.approx(0.3, abs=0.03)


def test_TreeFeatureGenerator_children_only_fire_when_parent_fires():
    child = TreeFeatureGenerator(active_prob=0.5)
    root = TreeFeatureGenerator(active_prob=0.5, children=[child])

    samples = root.sample(batch_size=10_000)
    parent_inactive = samples[:, 0] == 0

    # Child should never fire when parent is inactive
    assert torch.all(samples[parent_inactive, 1] == 0)


def test_TreeFeatureGenerator_child_conditional_probability():
    child = TreeFeatureGenerator(active_prob=0.4)
    root = TreeFeatureGenerator(active_prob=0.8, children=[child])

    samples = root.sample(batch_size=10_000)
    parent_active = samples[:, 0] > 0
    child_acts_given_parent = samples[parent_active, 1]

    # Child fires with prob 0.4 when parent is active
    actual_prob = (child_acts_given_parent > 0).float().mean().item()
    assert actual_prob == pytest.approx(0.4, abs=0.05)


def test_TreeFeatureGenerator_mutually_exclusive_children():
    child1 = TreeFeatureGenerator(active_prob=0.5)
    child2 = TreeFeatureGenerator(active_prob=0.5)
    root = TreeFeatureGenerator(
        active_prob=1.0,
        children=[child1, child2],
        mutually_exclusive_children=True,
    )

    samples = root.sample(batch_size=1000)

    # Both children should never fire simultaneously
    both_fire = (samples[:, 1] > 0) & (samples[:, 2] > 0)
    assert torch.sum(both_fire) == 0

    # But one should always fire (when parent is active)
    either_fires = (samples[:, 1] > 0) | (samples[:, 2] > 0)
    assert torch.all(either_fires)


def test_TreeFeatureGenerator_non_readout_node():
    child = TreeFeatureGenerator(active_prob=0.5)
    root = TreeFeatureGenerator(active_prob=0.8, children=[child], is_read_out=False)

    assert root.n_features == 1  # Only child is read out
    assert root.index is None
    assert child.index == 0


def test_TreeFeatureGenerator_magnitude_mean():
    root = TreeFeatureGenerator(active_prob=1.0, magnitude_mean=2.0, magnitude_std=0.0)
    samples = root.sample(batch_size=100)

    # All samples should have magnitude exactly 2.0
    assert torch.all(samples[:, 0] == 2.0)


def test_TreeFeatureGenerator_magnitude_std():
    root = TreeFeatureGenerator(active_prob=1.0, magnitude_mean=2.0, magnitude_std=0.5)
    samples = root.sample(batch_size=10_000)

    active_magnitudes = samples[samples[:, 0] > 0, 0]
    assert active_magnitudes.mean().item() == pytest.approx(2.0, abs=0.05)
    assert active_magnitudes.std().item() == pytest.approx(0.5, abs=0.05)


def test_TreeFeatureGenerator_from_dict():
    tree_dict = {
        "active_prob": 0.8,
        "magnitude_mean": 1.5,
        "children": [
            {"active_prob": 0.5},
            {"active_prob": 0.3, "magnitude_std": 0.2},
        ],
    }

    tree = TreeFeatureGenerator.from_dict(tree_dict)
    assert tree.n_features == 3
    assert tree.active_prob == 0.8
    assert tree.magnitude_mean == 1.5
    assert len(tree.children) == 2
    assert tree.children[0].active_prob == 0.5
    assert tree.children[1].magnitude_std == 0.2


def test_TreeFeatureGenerator_from_dict_mutually_exclusive():
    tree_dict = {
        "active_prob": 1.0,
        "mutually_exclusive_children": True,
        "children": [
            {"active_prob": 0.5},
            {"active_prob": 0.5},
        ],
    }

    tree = TreeFeatureGenerator.from_dict(tree_dict)
    samples = tree.sample(batch_size=100)

    # Children are mutually exclusive
    both_fire = (samples[:, 1] > 0) & (samples[:, 2] > 0)
    assert torch.sum(both_fire) == 0


def test_TreeFeatureGenerator_deep_hierarchy():
    grandchild = TreeFeatureGenerator(active_prob=0.5)
    child = TreeFeatureGenerator(active_prob=0.7, children=[grandchild])
    root = TreeFeatureGenerator(active_prob=0.8, children=[child])

    samples = root.sample(batch_size=10_000)

    # Grandchild can only fire when both parent and grandparent are active
    # Expected prob: 0.8 * 0.7 * 0.5 = 0.28
    expected_prob = 0.8 * 0.7 * 0.5
    actual_prob = (samples[:, 2] > 0).float().mean().item()
    assert actual_prob == pytest.approx(expected_prob, abs=0.03)

    # When parent is inactive, child and grandchild must be inactive
    parent_inactive = samples[:, 0] == 0
    assert torch.all(samples[parent_inactive, 1] == 0)
    assert torch.all(samples[parent_inactive, 2] == 0)


def test_TreeFeatureGenerator_device():
    root = TreeFeatureGenerator(active_prob=0.5)

    # CPU by default
    samples_cpu = root.sample(batch_size=10)
    assert samples_cpu.device.type == "cpu"

    # Explicit device
    samples_cpu = root.sample(batch_size=10, device=torch.device("cpu"))
    assert samples_cpu.device.type == "cpu"


def test_TreeFeatureGenerator_repr():
    child = TreeFeatureGenerator(active_prob=0.5, feature_id="child")
    root = TreeFeatureGenerator(
        active_prob=0.8,
        children=[child],
        mutually_exclusive_children=False,
        feature_id="root",
    )

    repr_str = repr(root)
    assert "0.8" in repr_str
    assert "root" in repr_str
    assert "0.5" in repr_str
    assert "child" in repr_str
