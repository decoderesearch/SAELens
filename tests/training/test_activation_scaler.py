from typing import Generator

import numpy as np
import pytest
import torch
from transformer_lens import HookedTransformer

from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.activations_store import ActivationsStore
from tests.helpers import assert_close, build_runner_cfg


def test_ActivationScaler_scale_without_scaling_factor():
    scaler = ActivationScaler()
    acts = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = scaler.scale(acts)
    assert torch.equal(result, acts)


def test_ActivationScaler_scale_with_scaling_factor():
    scaler = ActivationScaler(scaling_factor=2.0)
    acts = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    result = scaler.scale(acts)
    assert_close(result, expected)


def test_ActivationScaler_unscale_without_scaling_factor():
    scaler = ActivationScaler()
    acts = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = scaler.unscale(acts)
    assert torch.equal(result, acts)


def test_ActivationScaler_unscale_with_scaling_factor():
    scaler = ActivationScaler(scaling_factor=2.0)
    acts = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = scaler.unscale(acts)
    assert_close(result, expected)


def test_ActivationScaler_call_method():
    scaler = ActivationScaler(scaling_factor=3.0)
    acts = torch.tensor([[1.0, 2.0]])
    expected = torch.tensor([[3.0, 6.0]])
    result = scaler(acts)
    assert_close(result, expected)


def test_ActivationScaler_scale_unscale_roundtrip():
    scaler = ActivationScaler(scaling_factor=1.5)
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scaled = scaler.scale(original)
    unscaled = scaler.unscale(scaled)
    assert_close(unscaled, original)


def test_ActivationScaler_calculate_mean_norm():
    scaler = ActivationScaler()

    def data_provider() -> Generator[torch.Tensor, None, None]:
        # Generate 5 batches of data with known norms
        batches = [
            torch.tensor([[3.0, 4.0]]),  # norm = 5.0
            torch.tensor([[6.0, 8.0]]),  # norm = 10.0
            torch.tensor([[0.0, 0.0]]),  # norm = 0.0
            torch.tensor([[1.0, 0.0]]),  # norm = 1.0
            torch.tensor([[0.0, 2.0]]),  # norm = 2.0
        ]
        yield from batches

    mean_norm = scaler._calculate_mean_norm(
        data_provider(), n_batches_for_norm_estimate=5
    )
    expected_mean = (5.0 + 10.0 + 0.0 + 1.0 + 2.0) / 5
    assert mean_norm == pytest.approx(expected_mean, abs=1e-6)


def test_ActivationScaler_estimate_scaling_factor():
    scaler = ActivationScaler()
    d_in = 64

    def data_provider():
        # Generate batches with consistent norm
        while True:
            yield torch.ones(10, d_in) * 2.0  # Each vector has norm sqrt(64) * 2 = 16

    scaler.estimate_scaling_factor(
        d_in=d_in, data_provider=data_provider(), n_batches_for_norm_estimate=10
    )

    # Expected scaling factor: sqrt(64) / 16 = 8 / 16 = 0.5
    expected_scaling_factor = 0.5
    assert scaler.scaling_factor is not None
    assert scaler.scaling_factor == pytest.approx(expected_scaling_factor, abs=1e-6)


def test_ActivationScaler_estimate_scaling_factor_updates_scaler():
    scaler = ActivationScaler()
    assert scaler.scaling_factor is None

    def data_provider():
        while True:
            yield torch.ones(5, 4)  # norm = 2.0

    scaler.estimate_scaling_factor(
        d_in=4, data_provider=data_provider(), n_batches_for_norm_estimate=3
    )

    assert scaler.scaling_factor is not None
    assert scaler.scaling_factor > 0


@pytest.mark.parametrize("scaling_factor", [0.5, 1.0, 2.0, 10.0])
def test_ActivationScaler_scale_unscale_with_different_factors(scaling_factor: float):
    scaler = ActivationScaler(scaling_factor=scaling_factor)
    acts = torch.randn(32, 64)

    scaled = scaler.scale(acts)
    unscaled = scaler.unscale(scaled)

    assert_close(unscaled, acts, rtol=1e-6)


def test_ActivationScaler_scale_with_zero_tensor():
    scaler = ActivationScaler(scaling_factor=5.0)
    zero_tensor = torch.zeros(10, 20)

    scaled = scaler.scale(zero_tensor)
    unscaled = scaler.unscale(scaled)

    assert torch.equal(scaled, zero_tensor)
    assert torch.equal(unscaled, zero_tensor)


def test_ActivationScaler_scale_preserves_tensor_shape():
    scaler = ActivationScaler(scaling_factor=2.0)

    for shape in [(1, 1), (10, 64), (32, 128, 256)]:
        acts = torch.randn(*shape)
        scaled = scaler.scale(acts)
        unscaled = scaler.unscale(scaled)

        assert scaled.shape == acts.shape
        assert unscaled.shape == acts.shape


def test_ActivationScaler_scale_with_negative_values():
    scaler = ActivationScaler(scaling_factor=3.0)
    acts = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
    expected_scaled = torch.tensor([[-3.0, 6.0], [-9.0, 12.0]])

    scaled = scaler.scale(acts)
    assert_close(scaled, expected_scaled)

    unscaled = scaler.unscale(scaled)
    assert_close(unscaled, acts)


def test_ActivationScaler_estimates_norm_scaling_factor_from_activations_store(
    ts_model: HookedTransformer,
):
    # --- first, test initialisation ---

    # config if you want to benchmark this:
    #
    # cfg.context_size = 1024
    # cfg.n_batches_in_buffer = 64
    # cfg.store_batch_size_prompts = 16

    cfg = build_runner_cfg(
        d_in=64,
        streaming=False,
        context_size=1024,
        n_batches_in_buffer=64,
        store_batch_size_prompts=16,
    )

    store = ActivationsStore.from_config(ts_model, cfg)

    scaler = ActivationScaler()
    assert scaler.scaling_factor is None
    scaler.estimate_scaling_factor(cfg.sae.d_in, store, n_batches_for_norm_estimate=10)
    assert isinstance(scaler.scaling_factor, float)

    # Collect multiple batches to get a more stable norm estimate
    batches = [store.get_filtered_llm_batch() for _ in range(10)]
    combined = torch.cat(batches, dim=0)
    scaled_norm = combined.norm(dim=-1).mean() * scaler.scaling_factor
    assert scaled_norm == pytest.approx(np.sqrt(store.d_in), abs=5)
