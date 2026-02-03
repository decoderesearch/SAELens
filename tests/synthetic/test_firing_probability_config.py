import pytest
import torch

from sae_lens.synthetic import (
    ConstantFiringProbabilityConfig,
    FiringProbabilityConfig,
    LinearFiringProbabilityConfig,
    RandomFiringProbabilityConfig,
    ZipfianFiringProbabilityConfig,
    get_firing_probability_class,
    register_firing_probability,
)


def test_registry_contains_builtins():
    assert get_firing_probability_class("zipfian") == ZipfianFiringProbabilityConfig
    assert get_firing_probability_class("linear") == LinearFiringProbabilityConfig
    assert get_firing_probability_class("random") == RandomFiringProbabilityConfig
    assert get_firing_probability_class("constant") == ConstantFiringProbabilityConfig


def test_get_firing_probability_class_returns_correct_class():
    cfg_class = get_firing_probability_class("zipfian")
    assert cfg_class == ZipfianFiringProbabilityConfig


def test_get_firing_probability_class_raises_for_unknown():
    with pytest.raises(ValueError, match="Unknown name"):
        get_firing_probability_class("nonexistent")


def test_zipfian_config_generates_correct_shape():
    cfg = ZipfianFiringProbabilityConfig(exponent=1.0, max_prob=0.3, min_prob=0.01)
    probs = cfg.generate(100)
    assert probs.shape == (100,)
    assert probs.min() >= cfg.min_prob - 1e-6
    assert probs.max() <= cfg.max_prob + 1e-6


def test_linear_config_generates_correct_shape():
    cfg = LinearFiringProbabilityConfig(max_prob=0.3, min_prob=0.01)
    probs = cfg.generate(50)
    assert probs.shape == (50,)
    assert probs[0] == pytest.approx(cfg.max_prob, abs=1e-6)
    assert probs[-1] == pytest.approx(cfg.min_prob, abs=1e-6)


def test_random_config_generates_correct_range():
    cfg = RandomFiringProbabilityConfig(max_prob=0.5, min_prob=0.1)
    probs = cfg.generate(1000, seed=42)
    assert probs.shape == (1000,)
    assert probs.min() >= cfg.min_prob - 1e-6
    assert probs.max() <= cfg.max_prob + 1e-6


def test_constant_config_generates_uniform_values():
    cfg = ConstantFiringProbabilityConfig(probability=0.15)
    probs = cfg.generate(100)
    assert probs.shape == (100,)
    assert torch.allclose(probs, torch.full((100,), 0.15))


def test_config_to_dict_includes_generator_name():
    cfg = ZipfianFiringProbabilityConfig(exponent=2.0, max_prob=0.5, min_prob=0.05)
    d = cfg.to_dict()
    assert d["generator_name"] == "zipfian"
    assert d["exponent"] == 2.0
    assert d["max_prob"] == 0.5
    assert d["min_prob"] == 0.05


def test_config_from_dict_roundtrip():
    original = ZipfianFiringProbabilityConfig(exponent=1.5, max_prob=0.4, min_prob=0.02)
    d = original.to_dict()
    restored = FiringProbabilityConfig.from_dict(d)
    assert isinstance(restored, ZipfianFiringProbabilityConfig)
    assert restored.exponent == original.exponent
    assert restored.max_prob == original.max_prob
    assert restored.min_prob == original.min_prob


def test_config_from_dict_without_generator_name_raises():
    with pytest.raises(ValueError, match="generator_name required"):
        FiringProbabilityConfig.from_dict({"exponent": 1.0})


def test_register_duplicate_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_firing_probability(
            "zipfian",
            ZipfianFiringProbabilityConfig,
        )
