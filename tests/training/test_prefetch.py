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


def test_prefetching_iterator_raises_stop_iteration_repeatedly_after_exhaustion():
    # Iterator protocol: once StopIteration is raised, every subsequent call
    # must also raise it (and not deadlock waiting on an empty queue).
    prefetcher = PrefetchingIterator(iter([1, 2]), prefetch=2)
    assert next(prefetcher) == 1
    assert next(prefetcher) == 2
    with pytest.raises(StopIteration):
        next(prefetcher)
    with pytest.raises(StopIteration):
        next(prefetcher)


def test_prefetching_iterator_rejects_invalid_prefetch():
    with pytest.raises(ValueError, match="prefetch must be >= 1"):
        PrefetchingIterator(iter([1, 2]), prefetch=0)


def test_prefetching_iterator_runs_producer_concurrently():
    # Source sleeps before yielding each item; if the prefetcher works the
    # producer fills the queue while the consumer is "busy", so wall time
    # should be meaningfully less than the serial sum.
    n = 5
    delay = 0.1

    def slow_source() -> Iterator[int]:
        for i in range(n):
            time.sleep(delay)
            yield i

    prefetcher = PrefetchingIterator(slow_source(), prefetch=n)
    start = time.monotonic()
    out = []
    for item in prefetcher:
        time.sleep(delay)
        out.append(item)
    elapsed = time.monotonic() - start

    assert out == list(range(n))
    serial = n * 2 * delay
    # Generous slack for loaded CI: just verify we're measurably faster than
    # serial. Perfect overlap would be ~(n + 1) * delay = 0.6s; serial = 1.0s.
    assert elapsed < serial * 0.85


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


def test_prefetching_iterator_paused_prevents_concurrent_source_access():
    # Wrap the source in a guard that explicitly fails on overlapping next()
    # calls — this is what raises if the producer thread and the caller race
    # on the same generator (Python's "generator already executing" error).
    class _GuardedIter(Iterator[int]):
        def __init__(self) -> None:
            self._n = 0
            self._busy = threading.Lock()

        def __iter__(self) -> "Iterator[int]":
            return self

        def __next__(self) -> int:
            if not self._busy.acquire(blocking=False):
                raise RuntimeError("concurrent next() detected")
            try:
                # Sleep so the producer thread spends real wall time inside
                # next(); if the lock isn't honored, our own next() call
                # overlaps and trips the guard.
                time.sleep(0.02)
                self._n += 1
                return self._n
            finally:
                self._busy.release()

    source = _GuardedIter()
    prefetcher = PrefetchingIterator(source, prefetch=1)
    # Let the producer get into a steady-state loop.
    assert next(prefetcher) == 1

    for _ in range(10):
        with prefetcher.paused():
            # Without paused() this would race the producer and the guard
            # would raise RuntimeError.
            next(source)


def test_prefetching_iterator_thread_is_daemon():
    prefetcher = PrefetchingIterator(iter([1]), prefetch=1)
    # Implementation detail check: the prefetch thread must be daemon so it
    # doesn't keep the process alive past training.
    threads = [t for t in threading.enumerate() if t is prefetcher._thread]
    assert threads
    assert threads[0].daemon is True
