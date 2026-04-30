from collections.abc import Iterator

import torch


@torch.no_grad()
def multi_hook_concat_split_iter(
    buffer_size: int,
    batch_size: int,
    activations_loader: Iterator[dict[str, torch.Tensor]],
    hook_names: list[str],
    mix_fraction: float = 0.5,
) -> Iterator[dict[str, torch.Tensor]]:
    """
    Multi-hook variant of `mixing_buffer`.

    The producer yields `dict[hook_name, (n_tokens, d_in_h)]` per LLM forward
    pass. Tensors at different hooks may have different feature dimensions
    (e.g. attn vs resid, or transcoder input vs output).

    To get a single shared shuffle across all hooks (preserving token
    alignment), we concatenate along the feature dim into a packed
    `(n_tokens, sum_d_in)` tensor, run it through the unmodified `mixing_buffer`,
    and split the yielded batches back into a per-hook dict on the way out.
    The single `torch.randperm(n_tokens)` inside `mixing_buffer` is therefore
    automatically applied identically to every hook's slice.
    """
    iterator = iter(activations_loader)
    try:
        first = next(iterator)
    except StopIteration:
        return

    missing = [h for h in hook_names if h not in first]
    if missing:
        raise ValueError(
            f"producer did not yield activations for hooks {missing}; got keys {list(first.keys())}"
        )

    boundaries = [0]
    for h in hook_names:
        boundaries.append(boundaries[-1] + first[h].shape[1])

    def packed() -> Iterator[torch.Tensor]:
        yield torch.cat([first[h] for h in hook_names], dim=1)
        for d in iterator:
            yield torch.cat([d[h] for h in hook_names], dim=1)

    for batch in mixing_buffer(buffer_size, batch_size, packed(), mix_fraction):
        yield {
            h: batch[:, boundaries[i] : boundaries[i + 1]]
            for i, h in enumerate(hook_names)
        }


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
