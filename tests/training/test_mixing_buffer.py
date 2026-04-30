from collections.abc import Iterator

import pytest
import torch

from sae_lens.training.mixing_buffer import (
    mixing_buffer,
    multi_hook_concat_split_iter,
)
from tests.helpers import assert_close, assert_not_close


def test_mixing_buffer_yields_batches_of_correct_size_despite_loader_size_fluctuations():
    # Create a simple activations loader that yields 2 batches
    batch_size = 4
    buffer_size = 16
    d_in = 8

    # total number of activations is 16 + 3 + 16 - 1 = 34
    # so we should get 34 // 4 = 8 batches
    activations = [
        torch.randn(buffer_size + 3, d_in),
        torch.randn(buffer_size - 1, d_in),
    ]

    # Get batches from mixing buffer
    batches = list(
        mixing_buffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            activations_loader=iter(activations),
        )
    )

    assert len(batches) == 8
    for batch in batches:
        assert batch.shape == (batch_size, d_in)


def test_mixing_buffer_mixes_activations():
    buffer_size = 100
    batch_size = 50
    activations = [torch.arange(30), torch.arange(30, 60), torch.arange(60, 120)]

    buffer = mixing_buffer(
        buffer_size=buffer_size,
        batch_size=batch_size,
        activations_loader=iter(activations),
    )

    batch = next(buffer)
    assert batch.shape == (50,)
    assert_not_close(batch, torch.arange(50))
    assert len(torch.unique(batch)) == len(batch)  # All elements are unique


def test_mixing_buffer_empty_loader():
    buffer = mixing_buffer(buffer_size=16, batch_size=4, activations_loader=iter([]))

    # Should not yield any batches
    assert not list(buffer)


def test_mixing_buffer_error_on_small_buffer():
    # Test when buffer size is smaller than batch size
    batch_size = 8
    buffer_size = 4  # Too small

    activations = [torch.randn(batch_size, 4)]

    with pytest.raises(ValueError):
        buffer = mixing_buffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            activations_loader=iter(activations),
        )
        next(buffer)


def test_mixing_buffer_maintains_dtype():
    # Test that dtype is preserved
    batch_size = 4
    buffer_size = 16
    dtype = torch.float64

    activations = [
        torch.randn(batch_size, 8, dtype=dtype),
        torch.randn(batch_size, 8, dtype=dtype),
    ]

    batches = list(
        mixing_buffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            activations_loader=iter(activations),
        )
    )

    for batch in batches:
        assert batch.dtype == dtype


@pytest.mark.parametrize("mix_fraction", [-0.1, 1.5])
def test_mixing_buffer_invalid_mix_fraction_raises(mix_fraction: float):
    activations = [torch.randn(16, 8)]
    with pytest.raises(ValueError, match="mix_fraction must be in"):
        buffer = mixing_buffer(
            buffer_size=16,
            batch_size=4,
            activations_loader=iter(activations),
            mix_fraction=mix_fraction,
        )
        next(buffer)


def test_mixing_buffer_mix_fraction_preserves_total_batches():
    buffer_size = 100
    batch_size = 10
    activations_low = [torch.randn(buffer_size, 8), torch.randn(buffer_size, 8)]
    activations_high = [torch.randn(buffer_size, 8), torch.randn(buffer_size, 8)]

    batches_low_mix = list(
        mixing_buffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            activations_loader=iter(activations_low),
            mix_fraction=0.2,
        )
    )

    batches_high_mix = list(
        mixing_buffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            activations_loader=iter(activations_high),
            mix_fraction=0.7,
        )
    )

    # Both should yield same total batches (200 activations / 10 = 20 batches)
    assert len(batches_low_mix) == len(batches_high_mix) == 20


def test_mixing_buffer_zero_mix_fraction_no_shuffle():
    buffer_size = 100
    batch_size = 10
    activations = [torch.arange(100)]

    batches = list(
        mixing_buffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            activations_loader=iter(activations),
            mix_fraction=0,
        )
    )

    # With mix_fraction=0, order should be preserved (no shuffle)
    all_values = torch.cat(batches)
    assert torch.equal(all_values, torch.arange(100))


def test_mixing_buffer_mix_fraction_matches_observed_mix_fraction():
    target_mix_frac = 0.7
    buffer_size = 10_000

    def input_activations():
        for i in range(100_000):
            yield torch.tensor([i])

    buffer = mixing_buffer(
        buffer_size=buffer_size,
        batch_size=1000,
        activations_loader=input_activations(),
        mix_fraction=target_mix_frac,
    )

    observed_mix_fractions = []

    for i, batch in enumerate(buffer):
        max_act = batch.max().item()
        if i > 10:
            # it should refill after depleting (1 - target_mix_frac) of the buffer
            old_indices = max_act - (1 - target_mix_frac) * buffer_size
            mix_frac = (batch < old_indices).sum() / len(batch)
            observed_mix_fractions.append(mix_frac)

    mean_mix_fraction = sum(observed_mix_fractions) / len(observed_mix_fractions)
    assert mean_mix_fraction == pytest.approx(target_mix_frac, abs=0.03)


def test_multi_hook_concat_split_iter_passthrough_no_shuffle():
    # mix_fraction=0 → no shuffling; output dict should match input order, per-hook.
    h_a, h_b = "a", "b"
    n_per = 16
    a_in = torch.arange(n_per * 4, dtype=torch.float32).reshape(n_per, 4)
    b_in = torch.arange(n_per * 7, dtype=torch.float32).reshape(n_per, 7) + 1000
    inputs = [{h_a: a_in, h_b: b_in}]

    out_batches = list(
        multi_hook_concat_split_iter(
            buffer_size=16,
            batch_size=4,
            activations_loader=iter(inputs),
            hook_names=[h_a, h_b],
            mix_fraction=0.0,
        )
    )
    assert len(out_batches) == 4
    assert_close(torch.cat([b[h_a] for b in out_batches]), a_in)
    assert_close(torch.cat([b[h_b] for b in out_batches]), b_in)


def test_multi_hook_concat_split_iter_alignment_under_shuffle():
    # Encode the row index in both tensors so we can recover the permutation.
    # Row i has hook_a[i] = (i, i, ...) and hook_b[i] = (i, i, i, i).
    # If the two hooks were shuffled with different permutations, the row index
    # encoded in hook_a wouldn't match the row index encoded in hook_b.
    h_a, h_b = "a", "b"
    n = 32
    a_in = torch.arange(n, dtype=torch.float32).unsqueeze(1).expand(n, 3).contiguous()
    b_in = torch.arange(n, dtype=torch.float32).unsqueeze(1).expand(n, 5).contiguous()

    out_batches = list(
        multi_hook_concat_split_iter(
            buffer_size=32,
            batch_size=4,
            activations_loader=iter([{h_a: a_in, h_b: b_in}]),
            hook_names=[h_a, h_b],
            mix_fraction=0.5,
        )
    )

    for batch in out_batches:
        # Every row in hook_a encodes its original row index; same for hook_b.
        a_rows = batch[h_a][:, 0]
        b_rows = batch[h_b][:, 0]
        assert_close(a_rows, b_rows)
        # And every column within a single hook's row should encode the same index.
        assert_close(batch[h_a], a_rows.unsqueeze(1).expand_as(batch[h_a]))
        assert_close(batch[h_b], b_rows.unsqueeze(1).expand_as(batch[h_b]))


def test_multi_hook_concat_split_iter_handles_different_d_ins():
    h_a, h_b, h_c = "a", "b", "c"
    n = 12
    a_in = torch.randn(n, 2)
    b_in = torch.randn(n, 5)
    c_in = torch.randn(n, 3)

    out_batches = list(
        multi_hook_concat_split_iter(
            buffer_size=12,
            batch_size=4,
            activations_loader=iter([{h_a: a_in, h_b: b_in, h_c: c_in}]),
            hook_names=[h_a, h_b, h_c],
            mix_fraction=0.0,
        )
    )
    assert len(out_batches) == 3
    for batch in out_batches:
        assert batch[h_a].shape == (4, 2)
        assert batch[h_b].shape == (4, 5)
        assert batch[h_c].shape == (4, 3)
    assert_close(torch.cat([b[h_a] for b in out_batches]), a_in)
    assert_close(torch.cat([b[h_b] for b in out_batches]), b_in)
    assert_close(torch.cat([b[h_c] for b in out_batches]), c_in)


def test_multi_hook_concat_split_iter_errors_on_missing_hook():
    h_a, h_b = "a", "b"
    inputs = [{h_a: torch.randn(8, 4)}]  # missing h_b
    with pytest.raises(ValueError, match="hooks"):
        list(
            multi_hook_concat_split_iter(
                buffer_size=8,
                batch_size=4,
                activations_loader=iter(inputs),
                hook_names=[h_a, h_b],
                mix_fraction=0.0,
            )
        )


def test_multi_hook_concat_split_iter_empty_loader():
    out = list(
        multi_hook_concat_split_iter(
            buffer_size=8,
            batch_size=4,
            activations_loader=iter([]),
            hook_names=["a", "b"],
            mix_fraction=0.0,
        )
    )
    assert out == []


@pytest.mark.parametrize("mix_fraction", [0.9, 1.0])
def test_mixing_buffer_bounded_storage_with_high_mix_fraction(mix_fraction: float):
    """
    Regression test: with high mix_fraction, the storage buffer should not grow
    unbounded. It should stay bounded around buffer_size, not grow to many times
    buffer_size.
    """
    buffer_size = 100
    batch_size = 10
    num_iterations = 20

    batches_yielded = [0]
    max_implied_buffer = [0]

    def tracking_loader() -> Iterator[torch.Tensor]:
        for i in range(num_iterations):
            # Calculate implied buffer size before this iteration
            total_input = i * buffer_size
            total_yielded = batches_yielded[0] * batch_size
            implied_buffer = total_input - total_yielded
            max_implied_buffer[0] = max(max_implied_buffer[0], implied_buffer)
            yield torch.randn(buffer_size, 8)

    for _ in mixing_buffer(
        buffer_size=buffer_size,
        batch_size=batch_size,
        activations_loader=tracking_loader(),
        mix_fraction=mix_fraction,
    ):
        batches_yielded[0] += 1

    # Buffer should stay bounded around buffer_size, allow some tolerance for
    # mix_fraction portion plus one incoming batch
    max_acceptable = buffer_size * 2
    assert max_implied_buffer[0] <= max_acceptable, (
        f"Buffer grew to {max_implied_buffer[0]}, exceeding {max_acceptable}. "
        f"With mix_fraction={mix_fraction}, buffer should stay bounded around buffer_size."
    )
