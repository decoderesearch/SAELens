from itertools import islice

import pytest
import torch

from sae_lens.tokenization_and_batching import (
    _add_tokens_to_batch,
    concat_and_batch_sequences,
)


def test_add_tokens_to_batch_can_start_a_new_batch():
    tokens = torch.arange(10)
    new_batch, new_offset = _add_tokens_to_batch(
        batch=None, tokens=tokens, offset=1, context_size=5, is_start_of_sequence=True
    )
    assert torch.all(new_batch == tokens[1:6])
    assert new_offset == 6


def test_add_tokens_to_batch_adds_bos_if_new_batch():
    tokens = torch.arange(10)
    new_batch, new_offset = _add_tokens_to_batch(
        batch=None,
        tokens=tokens,
        offset=0,
        context_size=5,
        is_start_of_sequence=True,
        begin_batch_token_id=999,
        sequence_separator_token_id=998,
    )
    remaining_tokens = tokens[new_offset:]
    assert new_batch.tolist() == [999] + tokens[:4].tolist()
    assert torch.all(remaining_tokens == tokens[4:])


def test_add_tokens_respects_token_offset_when_adding_bos_if_new_batch():
    tokens = torch.arange(10)
    new_batch, new_offset = _add_tokens_to_batch(
        batch=None,
        tokens=tokens,
        offset=2,
        context_size=5,
        is_start_of_sequence=True,
        begin_batch_token_id=999,
        sequence_separator_token_id=998,
    )
    assert new_batch.tolist() == [999] + tokens[2:6].tolist()
    assert new_offset == 6


def test_add_tokens_to_batch_does_not_adds_bos_if_the_bos_is_already_there():
    tokens = torch.tensor([999, 1, 2, 3])
    new_batch, new_offset = _add_tokens_to_batch(
        batch=None,
        tokens=tokens,
        offset=0,
        context_size=5,
        is_start_of_sequence=True,
        begin_batch_token_id=999,
    )
    remaining_tokens = tokens[new_offset:]
    assert new_batch.tolist() == tokens.tolist()
    assert len(remaining_tokens) == 0


def test_add_tokens_to_batch_uses_all_tokens_if_less_than_context_size():
    tokens = torch.arange(3)
    new_batch, new_offset = _add_tokens_to_batch(
        batch=None,
        tokens=tokens,
        offset=0,
        context_size=5,
        is_start_of_sequence=False,
    )
    remaining_tokens = tokens[new_offset:]
    assert torch.all(new_batch == tokens)
    assert remaining_tokens.shape == (0,)


def test_add_tokens_to_batch_can_append_both_the_bos_and_start_of_sequence_token():
    tokens = torch.arange(10)
    new_batch, new_offset = _add_tokens_to_batch(
        batch=None,
        tokens=tokens,
        offset=0,
        context_size=5,
        is_start_of_sequence=True,
        begin_batch_token_id=999,
        begin_sequence_token_id=998,
    )
    remaining_tokens = tokens[new_offset:]
    assert new_batch.tolist() == [999, 998] + tokens[:3].tolist()
    assert torch.all(remaining_tokens == tokens[3:])


def test_add_tokens_to_batch_appends_to_the_existing_batch():
    batch = torch.arange(4)
    tokens = torch.arange(10)
    new_batch, new_offset = _add_tokens_to_batch(
        batch=batch,
        tokens=tokens,
        offset=0,
        context_size=5,
        is_start_of_sequence=True,
    )
    remaining_tokens = tokens[new_offset:]
    assert new_batch.tolist() == batch.tolist() + tokens[:1].tolist()
    assert torch.all(remaining_tokens == tokens[1:])


def test_add_tokens_to_batch_can_separate_sequences():
    batch = torch.arange(3)
    tokens = torch.arange(10)
    new_batch, new_offset = _add_tokens_to_batch(
        batch=batch,
        tokens=tokens,
        offset=0,
        context_size=5,
        is_start_of_sequence=True,
        sequence_separator_token_id=997,
        begin_batch_token_id=999,
    )
    remaining_tokens = tokens[new_offset:]
    assert new_batch.tolist() == batch.tolist() + [997] + tokens[:1].tolist()
    assert torch.all(remaining_tokens == tokens[1:])


def test_add_tokens_to_batch_can_both_separate_sequences_and_add_seq_start_token():
    batch = torch.arange(2)
    tokens = torch.arange(10)
    new_batch, new_offset = _add_tokens_to_batch(
        batch=batch,
        tokens=tokens,
        offset=0,
        context_size=5,
        is_start_of_sequence=True,
        sequence_separator_token_id=997,
        begin_sequence_token_id=998,
        begin_batch_token_id=999,
    )
    remaining_tokens = tokens[new_offset:]
    assert new_batch.tolist() == batch.tolist() + [997, 998] + tokens[:1].tolist()
    assert torch.all(remaining_tokens == tokens[1:])


@pytest.mark.parametrize("offset", [0, 1, 3])
def test_add_tokens_to_batch_wont_return_more_remaining_tokens_than_the_original(
    offset: int,
):
    batch = torch.arange(4)
    tokens = torch.arange(10)
    new_batch, new_offset = _add_tokens_to_batch(
        batch=batch,
        tokens=tokens,
        offset=offset,
        context_size=5,
        is_start_of_sequence=True,
        sequence_separator_token_id=997,
        begin_sequence_token_id=998,
        begin_batch_token_id=999,
    )
    assert new_batch.tolist() == batch.tolist() + [997]
    assert new_offset == offset


def test_add_tokens_to_batch_collapses_separate_sequences_and_add_seq_start_token_if_identical():
    batch = torch.arange(2)
    tokens = torch.arange(10)
    new_batch, new_offset = _add_tokens_to_batch(
        batch=batch,
        tokens=tokens,
        offset=0,
        context_size=5,
        is_start_of_sequence=True,
        sequence_separator_token_id=998,
        begin_sequence_token_id=998,
        begin_batch_token_id=999,
    )
    remaining_tokens = tokens[new_offset:]
    assert new_batch.tolist() == batch.tolist() + [998] + tokens[:2].tolist()
    assert torch.all(remaining_tokens == tokens[2:])


def test_add_tokens_to_batch_skips_add_seq_start_token_if_not_start_of_seq():
    batch = torch.arange(2)
    tokens = torch.arange(10)
    new_batch, new_offset = _add_tokens_to_batch(
        batch=batch,
        tokens=tokens,
        offset=0,
        context_size=5,
        is_start_of_sequence=False,
        sequence_separator_token_id=997,
        begin_sequence_token_id=998,
    )
    remaining_tokens = tokens[new_offset:]
    assert new_batch.tolist() == batch.tolist() + [997] + tokens[:2].tolist()
    assert torch.all(remaining_tokens == tokens[2:])


def test_concat_and_batch_sequences_generates_context_size_sequences():
    all_toks = torch.arange(20)
    seqs = [all_toks[:3], all_toks[3:10], all_toks[10:17], all_toks[17:]]
    batches_list = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(seqs),
            context_size=5,
        )
    )
    batches = torch.stack(batches_list)
    assert batches.shape == (4, 5)
    assert torch.all(batches == all_toks.reshape(4, 5))


def test_concat_and_batch_sequences_drops_the_final_batch_if_too_small():
    all_toks = torch.arange(19)
    seqs = [all_toks[:3], all_toks[3:10], all_toks[10:17], all_toks[17:]]
    batches_list = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(seqs),
            context_size=5,
        )
    )
    batches = torch.stack(batches_list)
    assert batches.shape == (3, 5)
    assert torch.all(batches == all_toks[:15].reshape(3, 5))


def test_concat_and_batch_sequences_can_ensure_everything_starts_with_bos():
    all_toks = torch.arange(19)
    seqs = [all_toks[:3], all_toks[3:10], all_toks[10:17], all_toks[17:]]
    batches_list = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(seqs),
            context_size=5,
            begin_batch_token_id=999,
        )
    )
    batches = torch.stack(batches_list)
    expected = [
        [999, 0, 1, 2, 3],
        [999, 4, 5, 6, 7],
        [999, 8, 9, 10, 11],
        [999, 12, 13, 14, 15],
    ]
    assert batches.tolist() == expected


def test_concat_and_batch_sequences_can_ensure_everything_starts_with_bos_with_concat_sequences_disabled():
    all_toks = torch.arange(19)
    seqs = [all_toks[:3], all_toks[3:10], all_toks[10:17], all_toks[17:]]
    batches_list = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(seqs),
            context_size=5,
            begin_batch_token_id=999,
            disable_concat_sequences=True,
        )
    )
    batches = torch.stack(batches_list)
    expected = [
        [999, 3, 4, 5, 6],
        [999, 10, 11, 12, 13],
    ]
    assert batches.tolist() == expected


def test_concat_and_batch_sequences_with_concat_sequences_disabled():
    all_toks = torch.arange(19)
    seqs = [all_toks[:3], all_toks[3:10], all_toks[10:17], all_toks[17:]]
    batches_list = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(seqs),
            context_size=5,
            disable_concat_sequences=True,
        )
    )
    batches = torch.stack(batches_list)
    expected = [
        [3, 4, 5, 6, 7],
        [10, 11, 12, 13, 14],
    ]
    assert batches.tolist() == expected


def test_concat_and_batch_sequences_can_ensure_each_seq_starts_with_a_token():
    all_toks = torch.arange(19)
    seqs = [all_toks[:3], all_toks[3:10], all_toks[10:17], all_toks[17:]]
    batches_list = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(seqs),
            context_size=5,
            begin_sequence_token_id=998,
        )
    )
    batches = torch.stack(batches_list)
    expected = [
        [998, 0, 1, 2, 998],
        [3, 4, 5, 6, 7],
        [8, 9, 998, 10, 11],
        [12, 13, 14, 15, 16],
    ]
    assert batches.tolist() == expected


def test_concat_and_batch_sequences_can_ensure_everything_starts_with_seq_start_token_with_concat_sequences_disabled():
    all_toks = torch.arange(19)
    seqs = [all_toks[:3], all_toks[3:10], all_toks[10:17], all_toks[17:]]
    batches_list = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(seqs),
            context_size=5,
            begin_sequence_token_id=999,
            disable_concat_sequences=True,
        )
    )
    batches = torch.stack(batches_list)
    expected = [
        [999, 3, 4, 5, 6],
        [999, 10, 11, 12, 13],
    ]
    assert batches.tolist() == expected


def test_concat_and_batch_sequences_can_ensure_everything_starts_with_matching_begin_batch_token_and_seq_start_tokens_with_concat_sequences_disabled():
    all_toks = torch.arange(19)
    seqs = [all_toks[:3], all_toks[3:10], all_toks[10:17], all_toks[17:]]
    batches_list = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(seqs),
            context_size=5,
            begin_batch_token_id=999,
            begin_sequence_token_id=999,
            disable_concat_sequences=True,
        )
    )
    batches = torch.stack(batches_list)
    expected = [
        [999, 3, 4, 5, 6],
        [999, 10, 11, 12, 13],
    ]
    assert batches.tolist() == expected


def test_concat_and_batch_sequences_prioritizes_seq_start_token_over_begin_batch_token_with_concat_sequences_disabled():
    all_toks = torch.arange(19)
    seqs = [all_toks[:3], all_toks[3:10], all_toks[10:17], all_toks[17:]]
    batches_list = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(seqs),
            context_size=5,
            begin_batch_token_id=999,
            begin_sequence_token_id=888,
            disable_concat_sequences=True,
        )
    )
    batches = torch.stack(batches_list)
    expected = [
        [888, 3, 4, 5, 6],
        [888, 10, 11, 12, 13],
    ]
    assert batches.tolist() == expected


def test_concat_and_batch_sequences_can_ensure_each_seq_is_separated_with_a_token():
    all_toks = torch.arange(19)
    seqs = [all_toks[:3], all_toks[3:10], all_toks[10:17], all_toks[17:]]
    batches_list = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(seqs),
            context_size=5,
            sequence_separator_token_id=997,
        )
    )
    batches = torch.stack(batches_list)
    expected = [
        [0, 1, 2, 997, 3],
        [4, 5, 6, 7, 8],
        [9, 997, 10, 11, 12],
        [13, 14, 15, 16, 997],
    ]
    assert batches.tolist() == expected


def test_concat_and_batch_sequences_can_use_all_token_types():
    all_toks = torch.arange(19)
    seqs = [all_toks[:3], all_toks[3:8], all_toks[8:11], all_toks[11:17], all_toks[17:]]
    batches_list = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(seqs),
            context_size=5,
            sequence_separator_token_id=997,
            begin_sequence_token_id=998,
            begin_batch_token_id=999,
        )
    )
    batches = torch.stack(batches_list)
    expected = [
        [999, 998, 0, 1, 2],
        [999, 998, 3, 4, 5],
        [999, 6, 7, 997, 998],
        [999, 8, 9, 10, 997],
        [999, 11, 12, 13, 14],
        [999, 15, 16, 997, 998],
    ]
    assert batches.tolist() == expected


def test_concat_and_batch_collapses_identical_special_tokens():
    all_toks = torch.arange(19)
    seqs = [all_toks[:3], all_toks[3:8], all_toks[8:11], all_toks[11:17], all_toks[17:]]
    batches_list = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(seqs),
            context_size=5,
            sequence_separator_token_id=999,
            begin_sequence_token_id=999,
            begin_batch_token_id=999,
        )
    )
    batches = torch.stack(batches_list)
    expected = [
        [999, 0, 1, 2, 999],
        [999, 3, 4, 5, 6],
        [999, 7, 999, 8, 9],
        [999, 10, 999, 11, 12],
        [999, 13, 14, 15, 16],
    ]
    assert batches.tolist() == expected


def test_concat_and_batch_sequences_works_with_extremely_long_samples():
    seq = torch.arange(10_000_000)
    batches_list = list(
        islice(
            concat_and_batch_sequences(
                tokens_iterator=iter([seq]),
                context_size=5,
                begin_sequence_token_id=999,
                begin_batch_token_id=999,
            ),
            50_000,  # cut off after 50k batchs so test still runs fast
        )
    )
    assert len(batches_list) == 50_000
    for batch in batches_list:
        assert batch.shape == (5,)
        assert batch[0] == 999


def test_concat_and_batch_sequences_disable_concat_sequences():
    all_toks = torch.arange(20)
    seqs = [all_toks[:3], all_toks[3:10], all_toks[10:17], all_toks[17:]]
    batches_list = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(seqs), context_size=5, disable_concat_sequences=True
        )
    )
    batches = torch.stack(batches_list)
    expected = [
        [3, 4, 5, 6, 7],
        [10, 11, 12, 13, 14],
    ]
    assert batches.tolist() == expected


def test_concat_and_batch_sequences_handles_empty_sequences_with_disable_concat():
    seqs = [
        torch.tensor([]),  # empty sequence
        torch.arange(10),  # long enough sequence
        torch.tensor([]),  # another empty sequence
        torch.arange(5, 15),  # another long enough sequence
    ]
    batches_list = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(seqs),
            context_size=5,
            begin_batch_token_id=999,
            disable_concat_sequences=True,
        )
    )
    batches = torch.stack(batches_list)
    expected = [
        [999, 0, 1, 2, 3],
        [999, 5, 6, 7, 8],
    ]
    assert batches.tolist() == expected


def test_concat_and_batch_sequences_handles_empty_sequences_with_begin_sequence_token():
    seqs = [
        torch.tensor([]),  # empty sequence
        torch.arange(10),  # long enough sequence
    ]
    batches_list = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(seqs),
            context_size=5,
            begin_sequence_token_id=999,
            disable_concat_sequences=True,
        )
    )
    batches = torch.stack(batches_list)
    expected = [
        [999, 0, 1, 2, 3],
    ]
    assert batches.tolist() == expected
