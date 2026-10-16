import gc
import threading
import time
import weakref
from collections import deque
from collections.abc import Iterator
from typing import Any

import pytest

from sae_lens.training.prefetch import PrefetchingIterator


def _prefetch_queue_size(prefetcher: PrefetchingIterator[Any]) -> int:
    queue_size = getattr(prefetcher._queue, "qsize", None)
    if queue_size is not None:
        return queue_size()
    return len(prefetcher._queue)


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
    assert prefetcher._exception is None


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


def test_prefetching_iterator_close_stops_early_and_discards_buffered_items():
    source_started = threading.Event()
    release_second = threading.Event()

    class _ObservedCondition(threading.Condition):
        def __init__(self) -> None:
            super().__init__()
            self.waiting_on_full_queue = threading.Event()

        def wait(self, timeout: float | None = None) -> bool:
            self.waiting_on_full_queue.set()
            return super().wait(timeout)

    class _CountingIterator(Iterator[int]):
        def __init__(self) -> None:
            self._next_calls = 0

        def __iter__(self) -> "_CountingIterator":
            return self

        def __next__(self) -> int:
            self._next_calls += 1
            if self._next_calls == 2:
                source_started.set()
                release_second.wait()
            return self._next_calls

    prefetcher = PrefetchingIterator(_CountingIterator(), prefetch=1)
    try:
        assert source_started.wait(timeout=1)
        assert _prefetch_queue_size(prefetcher) == 1
        observed_condition = _ObservedCondition()
        prefetcher._queue_condition = observed_condition
        release_second.set()
        assert observed_condition.waiting_on_full_queue.wait(timeout=1)

        assert prefetcher.close()
        assert prefetcher.close()
        assert not prefetcher._thread.is_alive()
        assert not prefetcher._queue
        with pytest.raises(StopIteration):
            next(prefetcher)
    finally:
        release_second.set()
        try:
            prefetcher.close(timeout=0)
        finally:
            prefetcher._thread.join(timeout=1)


def test_prefetching_iterator_close_after_natural_exhaustion_is_idempotent():
    prefetcher = PrefetchingIterator(iter([1, 2]), prefetch=1)
    assert list(prefetcher) == [1, 2]

    assert prefetcher.close()
    assert prefetcher.close()
    with pytest.raises(StopIteration):
        next(prefetcher)


def test_prefetching_iterator_close_times_out_for_blocked_source_then_exits():
    source_entered = threading.Event()
    release_source = threading.Event()

    class _BlockedIterator(Iterator[int]):
        def __iter__(self) -> "_BlockedIterator":
            return self

        def __next__(self) -> int:
            source_entered.set()
            release_source.wait()
            return 1

    prefetcher = PrefetchingIterator(_BlockedIterator(), prefetch=1)
    try:
        assert source_entered.wait(timeout=1)

        assert not prefetcher.close(timeout=0.01)
        assert prefetcher._thread.is_alive()

        release_source.set()
        prefetcher._thread.join(timeout=1)
        assert not prefetcher._thread.is_alive()
        assert prefetcher.close()
        with pytest.raises(StopIteration):
            next(prefetcher)
    finally:
        release_source.set()
        try:
            prefetcher.close(timeout=0)
        finally:
            prefetcher._thread.join(timeout=1)


def test_prefetching_iterator_close_while_paused_prevents_another_source_step():
    pause_entered = threading.Event()
    release_pause = threading.Event()
    producer_waiting_for_lock = threading.Event()

    class _ObservedLock:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._acquisitions = 0

        def __enter__(self) -> "_ObservedLock":
            self._acquisitions += 1
            if self._acquisitions == 2:
                producer_waiting_for_lock.set()
            self._lock.acquire()
            return self

        def __exit__(self, *_: object) -> None:
            self._lock.release()

    class _Source(Iterator[int]):
        def __init__(self) -> None:
            self.next_calls = 0

        def __iter__(self) -> "_Source":
            return self

        def __next__(self) -> int:
            self.next_calls += 1
            return self.next_calls

    source = _Source()
    prefetcher: Any = PrefetchingIterator.__new__(PrefetchingIterator)
    prefetcher._queue = deque()
    prefetcher._queue_maxsize = 1
    prefetcher._queue_condition = threading.Condition()
    prefetcher._lock = _ObservedLock()
    prefetcher._closed = threading.Event()
    prefetcher._exception = None
    prefetcher._done = False

    def hold_pause() -> None:
        with prefetcher.paused():
            pause_entered.set()
            release_pause.wait()

    pause_thread = threading.Thread(target=hold_pause)
    pause_thread.start()
    producer_thread: threading.Thread | None = None
    try:
        assert pause_entered.wait(timeout=1)

        producer_thread = threading.Thread(
            target=prefetcher._run, args=(source,), daemon=True
        )
        prefetcher._thread = producer_thread
        producer_thread.start()
        assert producer_waiting_for_lock.wait(timeout=1)
        assert not prefetcher.close(timeout=0.01)

        release_pause.set()
        pause_thread.join(timeout=1)
        producer_thread.join(timeout=1)
        assert not pause_thread.is_alive()
        assert not producer_thread.is_alive()
        assert source.next_calls == 0
    finally:
        release_pause.set()
        try:
            if producer_thread is not None:
                prefetcher.close(timeout=0)
        finally:
            pause_thread.join(timeout=1)
            if producer_thread is not None:
                producer_thread.join(timeout=1)


def test_prefetching_iterator_close_releases_buffered_item_while_source_is_blocked():
    source_entered = threading.Event()
    release_source = threading.Event()

    class _Batch:
        pass

    class _BlockedAfterFirst(Iterator[_Batch]):
        def __init__(self) -> None:
            self.calls = 0
            self.batch_ref: weakref.ReferenceType[_Batch] | None = None

        def __iter__(self) -> "_BlockedAfterFirst":
            return self

        def __next__(self) -> _Batch:
            self.calls += 1
            if self.calls == 1:
                batch = _Batch()
                self.batch_ref = weakref.ref(batch)
                return batch
            source_entered.set()
            release_source.wait()
            return _Batch()

    source = _BlockedAfterFirst()
    prefetcher = PrefetchingIterator(source, prefetch=1)
    try:
        assert source_entered.wait(timeout=1)
        assert source.batch_ref is not None
        assert source.batch_ref() is not None

        assert not prefetcher.close(timeout=0.01)
        gc.collect()
        assert source.batch_ref() is None

        release_source.set()
        prefetcher._thread.join(timeout=1)
        assert not prefetcher._thread.is_alive()
    finally:
        release_source.set()
        try:
            prefetcher.close(timeout=0)
        finally:
            prefetcher._thread.join(timeout=1)


def test_prefetching_iterator_close_releases_pending_exception_traceback():
    class _Payload:
        pass

    payload_ref: weakref.ReferenceType[_Payload] | None = None

    class _FailingIterator(Iterator[int]):
        def __init__(self) -> None:
            self.calls = 0

        def __iter__(self) -> "_FailingIterator":
            return self

        def __next__(self) -> int:
            nonlocal payload_ref
            self.calls += 1
            if self.calls == 1:
                return 1
            payload = _Payload()
            payload_ref = weakref.ref(payload)
            raise RuntimeError("source failed")

    prefetcher = PrefetchingIterator(_FailingIterator(), prefetch=1)
    try:
        with prefetcher._queue_condition:
            assert prefetcher._queue_condition.wait_for(
                lambda: prefetcher._exception is not None, timeout=1
            )
        assert payload_ref is not None
        assert payload_ref() is not None

        assert prefetcher.close()
        gc.collect()
        assert payload_ref() is None
    finally:
        prefetcher.close(timeout=0)


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_prefetching_iterator_close_rejects_invalid_timeout(timeout: float):
    prefetcher = PrefetchingIterator(iter(()))
    with pytest.raises(ValueError, match="timeout must be finite and >= 0"):
        prefetcher.close(timeout=timeout)
    assert prefetcher.close()


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
    try:
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
    finally:
        prefetcher.close()


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
    try:
        # Let the producer get into a steady-state loop.
        assert next(prefetcher) == 1

        for _ in range(10):
            with prefetcher.paused():
                # Without paused() this would race the producer and the guard
                # would raise RuntimeError.
                next(source)
    finally:
        prefetcher.close()


def test_prefetching_iterator_thread_is_daemon():
    prefetcher = PrefetchingIterator(iter([1]), prefetch=1)
    # Implementation detail check: the prefetch thread must be daemon so it
    # doesn't keep the process alive past training.
    threads = [t for t in threading.enumerate() if t is prefetcher._thread]
    assert threads
    assert threads[0].daemon is True
