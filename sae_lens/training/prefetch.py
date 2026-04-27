import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Generic, TypeVar

T = TypeVar("T")
_SENTINEL: object = object()


class PrefetchingIterator(Iterator[T], Generic[T]):
    """Wrap an iterator with a background thread that prefills a bounded queue.

    Decouples the source's pipeline from the consumer so they can run in
    parallel. In SAE training this lets the LLM forward (on the LLM's device)
    overlap with the SAE training step (on the SAE's device): while the SAE
    trains step ``t``, the producer thread is already running the LLM to
    generate batch ``t + 1``.

    The producer is a daemon thread and dies with the process. The queue is
    bounded by ``prefetch`` so the producer naturally back-pressures when the
    consumer falls behind.

    ``paused()`` is a context manager that blocks the producer thread for the
    duration of the ``with``-block, so callers can use the underlying source
    directly (e.g. for eval) without racing the producer thread on shared
    generator state.
    """

    def __init__(self, source: Iterator[T], prefetch: int = 4):
        if prefetch < 1:
            raise ValueError("prefetch must be >= 1")
        self._queue: queue.Queue[object] = queue.Queue(maxsize=prefetch)
        self._lock = threading.Lock()
        self._exception: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()

    def _run(self, source: Iterator[T]) -> None:
        try:
            while True:
                with self._lock:
                    try:
                        item = next(source)
                    except StopIteration:
                        break
                self._queue.put(item)
        except BaseException as e:  # noqa: BLE001
            self._exception = e
        finally:
            self._queue.put(_SENTINEL)

    def __iter__(self) -> "PrefetchingIterator[T]":
        return self

    def __next__(self) -> T:
        item = self._queue.get()
        if item is _SENTINEL:
            if self._exception is not None:
                raise self._exception
            raise StopIteration
        return item  # type: ignore[return-value]

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Block the background thread for the duration of the ``with``-block.

        Acquires an internal lock that the producer holds while calling
        ``next(source)``. Use this when you need to use the underlying source
        from another thread (e.g. for eval) without racing the prefetcher.
        """
        with self._lock:
            yield
