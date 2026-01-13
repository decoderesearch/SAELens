import pytest
import torch

from sae_lens.synthetic import (
    ConstantMagnitudeConfig,
    ConstantMagnitudeGenerator,
    ExponentialMagnitudeConfig,
    ExponentialMagnitudeGenerator,
    LinearMagnitudeConfig,
    LinearMagnitudeGenerator,
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

    def test_generator_produces_uniform_tensor(self):
        cfg = ConstantMagnitudeConfig(value=3.0)
        gen = ConstantMagnitudeGenerator(cfg)
        result = gen.generate(100)
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

    def test_generator_produces_linear_interpolation(self):
        cfg = LinearMagnitudeConfig(start=10.0, end=1.0)
        gen = LinearMagnitudeGenerator(cfg)
        result = gen.generate(10)
        assert result.shape == (10,)
        assert result[0] == pytest.approx(10.0)
        assert result[-1] == pytest.approx(1.0)
        expected = torch.linspace(10.0, 1.0, 10)
        assert torch.allclose(result, expected)

    def test_single_feature_returns_start(self):
        cfg = LinearMagnitudeConfig(start=5.0, end=1.0)
        gen = LinearMagnitudeGenerator(cfg)
        result = gen.generate(1)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(5.0)

    def test_ascending_values(self):
        cfg = LinearMagnitudeConfig(start=0.1, end=10.0)
        gen = LinearMagnitudeGenerator(cfg)
        result = gen.generate(10)
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

    def test_generator_produces_exponential_interpolation(self):
        cfg = ExponentialMagnitudeConfig(start=10.0, end=1.0)
        gen = ExponentialMagnitudeGenerator(cfg)
        result = gen.generate(5)
        assert result.shape == (5,)
        assert result[0] == pytest.approx(10.0)
        assert result[-1] == pytest.approx(1.0)
        # middle value (i=2, n=5) should be sqrt(10) = 10^0.5 ≈ 3.162
        expected_middle = 10.0 * (1.0 / 10.0) ** (2.0 / 4.0)
        assert result[2] == pytest.approx(expected_middle)

    def test_single_feature_returns_start(self):
        cfg = ExponentialMagnitudeConfig(start=5.0, end=1.0)
        gen = ExponentialMagnitudeGenerator(cfg)
        result = gen.generate(1)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(5.0)

    def test_ascending_values(self):
        cfg = ExponentialMagnitudeConfig(start=0.1, end=10.0)
        gen = ExponentialMagnitudeGenerator(cfg)
        result = gen.generate(10)
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

    def test_returns_float32_tensor(self):
        result = generate_magnitudes(10, 1.0)
        assert result.dtype == torch.float32

        cfg = LinearMagnitudeConfig(start=1.0, end=0.1)
        result = generate_magnitudes(10, cfg)
        assert result.dtype == torch.float32


class TestMagnitudeRegistry:
    def test_registry_contains_builtins(self):
        assert get_magnitude_class("constant")[0] == ConstantMagnitudeConfig
        assert get_magnitude_class("linear")[0] == LinearMagnitudeConfig
        assert get_magnitude_class("exponential")[0] == ExponentialMagnitudeConfig

    def test_get_magnitude_class_raises_for_unknown(self):
        with pytest.raises(ValueError, match="Unknown name"):
            get_magnitude_class("nonexistent")

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            register_magnitude(
                "constant", ConstantMagnitudeConfig, ConstantMagnitudeGenerator
            )

    def test_config_from_dict_requires_generator_name(self):
        with pytest.raises(ValueError, match="generator_name required"):
            MagnitudeConfig.from_dict({"value": 1.0})
