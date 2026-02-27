import math

import pytest
import torch

from sae_lens.synthetic import (
    ConstantMagnitudeConfig,
    ExponentialMagnitudeConfig,
    FoldedNormalMagnitudeConfig,
    LinearMagnitudeConfig,
    MagnitudeConfig,
    generate_magnitudes,
    get_magnitude_class,
    register_magnitude,
)


class TestConstantMagnitudeConfig:
    def test_config_creates_with_defaults(self):
        cfg = ConstantMagnitudeConfig()
        assert cfg.value == 1.0

    def test_config_custom_value(self):
        cfg = ConstantMagnitudeConfig(value=2.5)
        assert cfg.value == 2.5

    def test_config_generates_uniform_tensor(self):
        cfg = ConstantMagnitudeConfig(value=3.0)
        result = cfg.generate(100)
        assert result.shape == (100,)
        assert torch.allclose(result, torch.full((100,), 3.0))

    def test_config_to_dict(self):
        cfg = ConstantMagnitudeConfig(value=2.5)
        d = cfg.to_dict()
        assert d == {"value": 2.5, "generator_name": "constant"}

    def test_config_from_dict_roundtrip(self):
        original = ConstantMagnitudeConfig(value=1.5)
        d = original.to_dict()
        restored = MagnitudeConfig.from_dict(d)
        assert isinstance(restored, ConstantMagnitudeConfig)
        assert restored.value == original.value


class TestLinearMagnitudeConfig:
    def test_config_creates(self):
        cfg = LinearMagnitudeConfig(start=10.0, end=1.0)
        assert cfg.start == 10.0
        assert cfg.end == 1.0

    def test_config_requires_positive_values(self):
        with pytest.raises(ValueError, match="must be positive"):
            LinearMagnitudeConfig(start=-1.0, end=0.1)
        with pytest.raises(ValueError, match="must be positive"):
            LinearMagnitudeConfig(start=1.0, end=-0.1)
        with pytest.raises(ValueError, match="must be positive"):
            LinearMagnitudeConfig(start=0.0, end=0.1)

    def test_config_generates_linear_interpolation(self):
        cfg = LinearMagnitudeConfig(start=10.0, end=1.0)
        result = cfg.generate(10)
        assert result.shape == (10,)
        assert result[0] == pytest.approx(10.0)
        assert result[-1] == pytest.approx(1.0)
        expected = torch.linspace(10.0, 1.0, 10)
        assert torch.allclose(result, expected)

    def test_single_feature_returns_start(self):
        cfg = LinearMagnitudeConfig(start=5.0, end=1.0)
        result = cfg.generate(1)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(5.0)

    def test_ascending_values(self):
        cfg = LinearMagnitudeConfig(start=0.1, end=10.0)
        result = cfg.generate(10)
        assert result[0] == pytest.approx(0.1)
        assert result[-1] == pytest.approx(10.0)
        assert torch.all(result[1:] > result[:-1])

    def test_config_to_dict_from_dict_roundtrip(self):
        original = LinearMagnitudeConfig(start=2.0, end=0.5)
        d = original.to_dict()
        restored = MagnitudeConfig.from_dict(d)
        assert isinstance(restored, LinearMagnitudeConfig)
        assert restored.start == original.start
        assert restored.end == original.end


class TestExponentialMagnitudeConfig:
    def test_config_creates(self):
        cfg = ExponentialMagnitudeConfig(start=10.0, end=1.0)
        assert cfg.start == 10.0
        assert cfg.end == 1.0

    def test_config_requires_positive_values(self):
        with pytest.raises(ValueError, match="must be positive for exponential"):
            ExponentialMagnitudeConfig(start=-1.0, end=0.1)
        with pytest.raises(ValueError, match="must be positive for exponential"):
            ExponentialMagnitudeConfig(start=1.0, end=-0.1)
        with pytest.raises(ValueError, match="must be positive for exponential"):
            ExponentialMagnitudeConfig(start=0.0, end=0.1)

    def test_config_generates_exponential_interpolation(self):
        cfg = ExponentialMagnitudeConfig(start=10.0, end=1.0)
        result = cfg.generate(5)
        assert result.shape == (5,)
        assert result[0] == pytest.approx(10.0)
        assert result[-1] == pytest.approx(1.0)
        # middle value (i=2, n=5) should be sqrt(10) = 10^0.5 ≈ 3.162
        expected_middle = 10.0 * (1.0 / 10.0) ** (2.0 / 4.0)
        assert result[2] == pytest.approx(expected_middle)

    def test_single_feature_returns_start(self):
        cfg = ExponentialMagnitudeConfig(start=5.0, end=1.0)
        result = cfg.generate(1)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(5.0)

    def test_ascending_values(self):
        cfg = ExponentialMagnitudeConfig(start=0.1, end=10.0)
        result = cfg.generate(10)
        assert result[0] == pytest.approx(0.1)
        assert result[-1] == pytest.approx(10.0)
        assert torch.all(result[1:] > result[:-1])

    def test_config_to_dict_from_dict_roundtrip(self):
        original = ExponentialMagnitudeConfig(start=2.0, end=0.5)
        d = original.to_dict()
        restored = MagnitudeConfig.from_dict(d)
        assert isinstance(restored, ExponentialMagnitudeConfig)
        assert restored.start == original.start
        assert restored.end == original.end


class TestFoldedNormalMagnitudeConfig:
    def test_config_creates_with_defaults(self):
        cfg = FoldedNormalMagnitudeConfig()
        assert cfg.mean == 0.0
        assert cfg.std == 0.1
        assert cfg.min_value is None
        assert cfg.max_value is None

    def test_config_custom_values(self):
        cfg = FoldedNormalMagnitudeConfig(
            mean=1.0, std=0.5, min_value=0.1, max_value=2.0
        )
        assert cfg.mean == 1.0
        assert cfg.std == 0.5
        assert cfg.min_value == 0.1
        assert cfg.max_value == 2.0

    def test_config_requires_positive_std(self):
        with pytest.raises(ValueError, match="std must be positive"):
            FoldedNormalMagnitudeConfig(std=0.0)
        with pytest.raises(ValueError, match="std must be positive"):
            FoldedNormalMagnitudeConfig(std=-1.0)

    def test_config_requires_non_negative_min_value(self):
        with pytest.raises(ValueError, match="min_value must be non-negative"):
            FoldedNormalMagnitudeConfig(std=1.0, min_value=-0.1)

    def test_config_requires_positive_max_value(self):
        with pytest.raises(ValueError, match="max_value must be positive"):
            FoldedNormalMagnitudeConfig(std=1.0, max_value=0.0)
        with pytest.raises(ValueError, match="max_value must be positive"):
            FoldedNormalMagnitudeConfig(std=1.0, max_value=-1.0)

    def test_config_requires_min_less_than_max(self):
        with pytest.raises(ValueError, match="min_value must be <= max_value"):
            FoldedNormalMagnitudeConfig(std=1.0, min_value=2.0, max_value=1.0)

    def test_config_generates_all_non_negative_values(self):
        cfg = FoldedNormalMagnitudeConfig(std=1.0)
        result = cfg.generate(100_000)
        assert result.shape == (100_000,)
        assert torch.all(result >= 0)

    def test_config_generates_all_non_negative_values_with_negative_mean(self):
        cfg = FoldedNormalMagnitudeConfig(mean=-2.0, std=1.0)
        result = cfg.generate(100_000)
        assert torch.all(result >= 0)

    def test_config_with_zero_mean_produces_half_normal_mean(self):
        std = 2.0
        cfg = FoldedNormalMagnitudeConfig(mean=0.0, std=std)
        result = cfg.generate(500_000)
        expected_mean = std * math.sqrt(2 / math.pi)
        actual_mean = result.mean().item()
        assert actual_mean == pytest.approx(expected_mean, rel=0.01)

    def test_config_with_zero_mean_produces_half_normal_std(self):
        std = 2.0
        cfg = FoldedNormalMagnitudeConfig(mean=0.0, std=std)
        result = cfg.generate(500_000)
        expected_std = std * math.sqrt(1 - 2 / math.pi)
        actual_std = result.std().item()
        assert actual_std == pytest.approx(expected_std, rel=0.01)

    def test_config_with_large_positive_mean_approaches_normal(self):
        mean = 10.0
        std = 1.0
        cfg = FoldedNormalMagnitudeConfig(mean=mean, std=std)
        result = cfg.generate(500_000)
        assert result.mean().item() == pytest.approx(mean, rel=0.01)
        assert result.std().item() == pytest.approx(std, rel=0.01)

    def test_config_clamps_to_max_value(self):
        cfg = FoldedNormalMagnitudeConfig(std=1.0, max_value=0.5)
        result = cfg.generate(100_000)
        assert torch.all(result <= 0.5)

    def test_config_clamps_to_min_value(self):
        cfg = FoldedNormalMagnitudeConfig(std=1.0, min_value=0.5)
        result = cfg.generate(100_000)
        assert torch.all(result >= 0.5)

    def test_config_clamps_to_both_min_and_max(self):
        cfg = FoldedNormalMagnitudeConfig(std=1.0, min_value=0.2, max_value=0.8)
        result = cfg.generate(100_000)
        assert torch.all(result >= 0.2)
        assert torch.all(result <= 0.8)

    def test_config_to_dict_from_dict_roundtrip(self):
        original = FoldedNormalMagnitudeConfig(
            mean=1.0, std=0.5, min_value=0.1, max_value=2.0
        )
        d = original.to_dict()
        restored = MagnitudeConfig.from_dict(d)
        assert isinstance(restored, FoldedNormalMagnitudeConfig)
        assert restored.mean == original.mean
        assert restored.std == original.std
        assert restored.min_value == original.min_value
        assert restored.max_value == original.max_value


class TestGenerateMagnitudes:
    def test_constant_float_returns_uniform_tensor(self):
        result = generate_magnitudes(100, 2.5)
        assert result.shape == (100,)
        assert torch.allclose(result, torch.full((100,), 2.5))

    def test_constant_int_returns_uniform_tensor(self):
        result = generate_magnitudes(50, 3)
        assert result.shape == (50,)
        assert torch.allclose(result, torch.full((50,), 3.0))

    def test_with_constant_config(self):
        cfg = ConstantMagnitudeConfig(value=5.0)
        result = generate_magnitudes(10, cfg)
        assert result.shape == (10,)
        assert torch.allclose(result, torch.full((10,), 5.0))

    def test_with_linear_config(self):
        cfg = LinearMagnitudeConfig(start=10.0, end=1.0)
        result = generate_magnitudes(10, cfg)
        assert result.shape == (10,)
        assert result[0] == pytest.approx(10.0)
        assert result[-1] == pytest.approx(1.0)

    def test_with_exponential_config(self):
        cfg = ExponentialMagnitudeConfig(start=10.0, end=0.1)
        result = generate_magnitudes(10, cfg)
        assert result.shape == (10,)
        assert result[0] == pytest.approx(10.0)
        assert result[-1] == pytest.approx(0.1)

    def test_with_folded_normal_config(self):
        cfg = FoldedNormalMagnitudeConfig(std=1.0)
        result = generate_magnitudes(10_000, cfg)
        assert result.shape == (10_000,)
        assert torch.all(result >= 0)

    def test_returns_float32_tensor(self):
        result = generate_magnitudes(10, 1.0)
        assert result.dtype == torch.float32

        cfg = LinearMagnitudeConfig(start=1.0, end=0.1)
        result = generate_magnitudes(10, cfg)
        assert result.dtype == torch.float32


class TestMagnitudeRegistry:
    def test_registry_contains_builtins(self):
        assert get_magnitude_class("constant") == ConstantMagnitudeConfig
        assert get_magnitude_class("linear") == LinearMagnitudeConfig
        assert get_magnitude_class("exponential") == ExponentialMagnitudeConfig
        assert get_magnitude_class("folded_normal") == FoldedNormalMagnitudeConfig

    def test_get_magnitude_class_raises_for_unknown(self):
        with pytest.raises(ValueError, match="Unknown name"):
            get_magnitude_class("nonexistent")

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            register_magnitude("constant", ConstantMagnitudeConfig)

    def test_config_from_dict_requires_generator_name(self):
        with pytest.raises(ValueError, match="generator_name required"):
            MagnitudeConfig.from_dict({"value": 1.0})
