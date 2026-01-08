from collections.abc import Iterator

import torch


@torch.no_grad()
def mixing_buffer(
    buffer_size: int,
    batch_size: int,
    activations_loader: Iterator[torch.Tensor],
    mix_fraction: float = 0.5,
) -> Iterator[torch.Tensor]:
    """
    A generator that maintains a mix of old and new activations for better training.
    It keeps a portion of activations and mixes them with new ones to create batches.

    Args:
        buffer_size: Total size of the buffer
        batch_size: Size of batches to return
        activations_loader: Iterator providing new activations
        mix_fraction: Fraction of buffer to keep for mixing (default 0.5).
                      Higher values mean more temporal mixing but slower throughput.
                      If 0, no shuffling occurs (passthrough mode).

    Yields:
        Batches of activations of shape (batch_size, *activation_dims)
    """

    if buffer_size < batch_size:
        raise ValueError("Buffer size must be greater than or equal to batch size")
    if not 0 <= mix_fraction <= 1:
        raise ValueError("mix_fraction must be in [0, 1]")

    storage_buffer: torch.Tensor | None = None

    for new_activations in activations_loader:
        storage_buffer = (
            new_activations
            if storage_buffer is None
            else torch.cat([storage_buffer, new_activations], dim=0)
        )

        if storage_buffer.shape[0] >= buffer_size:
            if mix_fraction > 0:
                storage_buffer = storage_buffer[torch.randperm(storage_buffer.shape[0])]

            # Keep a fixed amount for mixing, serve the rest
            keep_for_mixing = int(buffer_size * mix_fraction)
            num_to_serve = storage_buffer.shape[0] - keep_for_mixing
            num_serving_batches = max(1, num_to_serve // batch_size)
            serving_cutoff = num_serving_batches * batch_size
            serving_buffer = storage_buffer[:serving_cutoff]
            storage_buffer = storage_buffer[serving_cutoff:]

            # Yield batches from the serving_buffer
            for batch_idx in range(num_serving_batches):
                yield serving_buffer[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]

    # If there are any remaining activations, yield them
    if storage_buffer is not None:
        remaining_batches = storage_buffer.shape[0] // batch_size
        for i in range(remaining_batches):
            yield storage_buffer[i * batch_size : (i + 1) * batch_size]
