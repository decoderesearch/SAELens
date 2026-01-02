from collections.abc import Generator, Iterator

import torch


def _add_tokens_to_batch(
    batch: torch.Tensor | None,
    tokens: torch.Tensor,
    offset: int,
    context_size: int,
    is_start_of_sequence: bool,
    begin_batch_token_id: int | None = None,
    begin_sequence_token_id: int | None = None,
    sequence_separator_token_id: int | None = None,
) -> tuple[torch.Tensor, int]:
    prefix_toks = []
    first_token = tokens[offset]
    # prepend the start of sequence token if needed
    if is_start_of_sequence and begin_sequence_token_id is not None:
        begin_sequence_token_id_tensor = torch.tensor(
            [begin_sequence_token_id], dtype=torch.long, device=tokens.device
        )
        if first_token != begin_sequence_token_id_tensor:
            prefix_toks.insert(0, begin_sequence_token_id_tensor)
            first_token = begin_sequence_token_id_tensor
    # We're at the start of a new batch
    if batch is None:
        # add the BOS token to the start if needed
        if begin_batch_token_id is not None:
            begin_batch_token_id_tensor = torch.tensor(
                [begin_batch_token_id], dtype=torch.long, device=tokens.device
            )
            if first_token != begin_batch_token_id_tensor:
                prefix_toks.insert(0, begin_batch_token_id_tensor)
                first_token = begin_batch_token_id_tensor
        tokens_needed = max(context_size - len(prefix_toks), 0)
        tokens_part = tokens[offset : offset + tokens_needed]
        batch = torch.cat([*prefix_toks[:context_size], tokens_part])
        return batch, offset + tokens_needed
    # if we're concatting batches, add the separator token as needed
    if sequence_separator_token_id is not None:
        sequence_separator_token_id_tensor = torch.tensor(
            [sequence_separator_token_id], dtype=torch.long, device=tokens.device
        )
        if first_token != sequence_separator_token_id_tensor:
            prefix_toks.insert(0, sequence_separator_token_id_tensor)
            first_token = sequence_separator_token_id_tensor
    tokens_needed = max(context_size - batch.shape[0] - len(prefix_toks), 0)
    prefix_toks_needed = max(context_size - batch.shape[0], 0)
    batch = torch.concat(
        [
            batch,
            *prefix_toks[:prefix_toks_needed],
            tokens[offset : offset + tokens_needed],
        ]
    )
    return batch, offset + tokens_needed


@torch.no_grad()
def concat_and_batch_sequences(
    tokens_iterator: Iterator[torch.Tensor],
    context_size: int,
    begin_batch_token_id: int | None = None,
    begin_sequence_token_id: int | None = None,
    sequence_separator_token_id: int | None = None,
    disable_concat_sequences: bool = False,
) -> Generator[torch.Tensor, None, None]:
    """
    Generator to concat token sequences together from the tokens_interator, yielding
    sequences of size `context_size`. Batching across the batch dimension is handled by the caller.

    Args:
        tokens_iterator: An iterator which returns a 1D tensors of tokens
        context_size: Each batch will have this many tokens
        begin_batch_token_id: If provided, this token will be at position 0 of each batch
        begin_sequence_token_id: If provided, this token will be the first token of each sequence
        sequence_separator_token_id: If provided, this token will be inserted between concatenated sequences
        disable_concat_sequences: If True, disable concatenating sequences and ignore sequences shorter than context_size (including BOS token if present)
        max_batches: If not provided, the iterator will be run to completion.
    """
    if disable_concat_sequences:
        if begin_batch_token_id and not begin_sequence_token_id:
            begin_sequence_token_id = begin_batch_token_id
        for sequence in tokens_iterator:
            if (
                begin_sequence_token_id is not None
                and len(sequence) >= context_size - 1
                and sequence[0] != begin_sequence_token_id
            ):
                begin_sequence_token_id_tensor = torch.tensor(
                    [begin_sequence_token_id],
                    dtype=torch.long,
                    device=sequence.device,
                )
                sequence = torch.cat(
                    [begin_sequence_token_id_tensor, sequence[: context_size - 1]]
                )
            if len(sequence) >= context_size:
                yield sequence[:context_size]
        return

    batch: torch.Tensor | None = None
    for tokens in tokens_iterator:
        if len(tokens.shape) != 1:
            raise ValueError(f"tokens.shape should be 1D but was {tokens.shape}")
        offset = 0
        total_toks = tokens.shape[0]
        is_start_of_sequence = True
        while total_toks - offset > 0:
            batch, offset = _add_tokens_to_batch(
                batch=batch,
                tokens=tokens,
                offset=offset,
                context_size=context_size,
                is_start_of_sequence=is_start_of_sequence,
                begin_batch_token_id=begin_batch_token_id,
                begin_sequence_token_id=begin_sequence_token_id,
                sequence_separator_token_id=sequence_separator_token_id,
            )
            is_start_of_sequence = False
            if batch.shape[0] == context_size:
                yield batch
                batch = None
