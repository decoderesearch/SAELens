import threading
import time
from collections.abc import Iterator

import pytest

from sae_lens.training.prefetch import PrefetchingIterator


def test_prefetching_iterator_yields_same_items_in_same_order():
    source = iter([1, 2, 3, 4, 5])
    prefetcher = PrefetchingIterator(source, prefetch=2)
    assert list(prefetcher) == [1, 2, 3, 4, 5]


def test_prefetching_iterator_propagates_source_exception():
    class _Boom(RuntimeError):
        pass

    def bad_source() -> Iterator[int]:
        yield 1
        yield 2
        raise _Boom("source failed")

    prefetcher = PrefetchingIterator(bad_source(), prefetch=1)
    assert next(prefetcher) == 1
    assert next(prefetcher) == 2
    with pytest.raises(_Boom, match="source failed"):
        next(prefetcher)


def test_prefetching_iterator_rejects_invalid_prefetch():
    with pytest.raises(ValueError, match="prefetch must be >= 1"):
        PrefetchingIterator(iter([1, 2]), prefetch=0)


def test_prefetching_iterator_runs_producer_concurrently():
    # Source sleeps before yielding each item; if the prefetcher works the
    # producer fills the queue while the consumer is "busy", so total wall time
    # is dominated by the consumer's work, not consumer + source serially.
    n = 4
    source_delay = 0.05
    consumer_delay = 0.05

    def slow_source() -> Iterator[int]:
        for i in range(n):
            time.sleep(source_delay)
            yield i

    prefetcher = PrefetchingIterator(slow_source(), prefetch=n)
    start = time.monotonic()
    out = []
    for item in prefetcher:
        time.sleep(consumer_delay)
        out.append(item)
    elapsed = time.monotonic() - start

    assert out == list(range(n))
    serial = n * (source_delay + consumer_delay)
    overlap = n * max(source_delay, consumer_delay) + source_delay + consumer_delay
    # We should be much closer to overlap-time than serial-time.
    assert elapsed < (serial + overlap) / 2


def test_prefetching_iterator_paused_blocks_producer():
    # Long-running source we only consume when paused is released.
    progress = []

    def source() -> Iterator[int]:
        for i in range(10):
            progress.append(i)
            yield i

    prefetcher = PrefetchingIterator(source(), prefetch=1)

    # Drain a couple items so the producer is actively running.
    assert next(prefetcher) == 0
    assert next(prefetcher) == 1

    with prefetcher.paused():
        # Snapshot once we're inside `paused`. The producer can have at most
        # one in-flight item (the one buffered in the queue) since we hold the
        # lock and prefetch=1.
        time.sleep(0.05)
        snapshot = len(progress)
        time.sleep(0.05)
        # Producer should not have advanced while we hold the lock.
        assert len(progress) == snapshot

    # After releasing, producer resumes and we can keep consuming.
    assert next(prefetcher) == 2


def test_prefetching_iterator_paused_lets_caller_use_source():
    # If the caller wants to consume from the source while paused, they can.
    items = list(range(5))
    source = iter(items)
    prefetcher = PrefetchingIterator(source, prefetch=1)
    # Wait for prefetcher to grab the first item.
    first = next(prefetcher)
    assert first == 0

    with prefetcher.paused():
        # Inside the lock the producer can't be in next(source); we may safely
        # call next(source) ourselves without ValueError("generator already
        # executing").
        assert next(source) in items[1:]


def test_prefetching_iterator_thread_is_daemon():
    prefetcher = PrefetchingIterator(iter([1]), prefetch=1)
    # Implementation detail check: the prefetch thread must be daemon so it
    # doesn't keep the process alive past training.
    threads = [t for t in threading.enumerate() if t is prefetcher._thread]
    assert threads
    assert threads[0].daemon is True
