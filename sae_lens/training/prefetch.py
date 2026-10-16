import math
import threading
from collections import deque
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

    ``paused()`` is a context manager that pauses the producer for the duration
    of the ``with``-block by holding the lock that gates calls to
    ``next(source)``. Callers can then drive the source themselves (e.g. eval)
    without racing the producer on shared generator state. Note that the lock
    is held *across* a ``next(source)`` call, so acquiring it can stall for up
    to one full source step (e.g. one LLM forward pass) at unlucky moments.
    Also note the queue may already contain up to ``prefetch`` items that the
    producer enqueued before ``paused()`` acquired the lock.

    Call ``close()`` when the consumer stops before exhausting the iterator.
    It discards queued items and asks the producer to stop. A source already
    inside ``next()`` cannot be interrupted, so ``close()`` bounds its join
    and reports whether the producer exited. It never closes the source from
    another thread.
    """

    def __init__(self, source: Iterator[T], prefetch: int = 4):
        if prefetch < 1:
            raise ValueError("prefetch must be >= 1")
        self._queue: deque[object] = deque()
        self._queue_maxsize = prefetch
        self._queue_condition = threading.Condition()
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._exception: BaseException | None = None
        self._done = False
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()

    def _run(self, source: Iterator[T]) -> None:
        try:
            while not self._closed.is_set():
                with self._lock:
                    if self._closed.is_set():
                        break
                    try:
                        item = next(source)
                    except StopIteration:
                        break
                if not self._put_if_open(item):
                    return
                del item
        # BaseException (not Exception) so KeyboardInterrupt etc. don't silently
        # kill the producer without surfacing to the consumer.
        except BaseException as e:  # noqa: BLE001
            with self._queue_condition:
                if not self._closed.is_set():
                    self._exception = e
                    self._queue_condition.notify_all()
        finally:
            self._put_if_open(_SENTINEL)

    def _put_if_open(self, item: object) -> bool:
        with self._queue_condition:
            while not self._closed.is_set():
                if len(self._queue) < self._queue_maxsize:
                    self._queue.append(item)
                    self._queue_condition.notify_all()
                    return True
                self._queue_condition.wait()
        return False

    def __iter__(self) -> "PrefetchingIterator[T]":
        return self

    def __next__(self) -> T:
        # Iterator protocol: once StopIteration was raised, every subsequent
        # call must also raise it. Without this guard we'd block forever on
        # an empty queue with no producer alive. Source-side exceptions are
        # raised once on the first SENTINEL, matching Python generator
        # semantics (subsequent calls then raise StopIteration).
        if self._done or self._closed.is_set():
            self._done = True
            raise StopIteration
        with self._queue_condition:
            while not self._queue and not self._closed.is_set():
                self._queue_condition.wait()
            if self._closed.is_set():
                self._done = True
                raise StopIteration
            item = self._queue.popleft()
            self._queue_condition.notify()
            if item is _SENTINEL:
                self._done = True
                exception = self._exception
                self._exception = None
                if exception is not None:
                    raise exception
                raise StopIteration
            return item  # type: ignore[return-value]

    def close(self, timeout: float = 1.0) -> bool:
        """Stop prefetching and discard buffered items.

        Returns whether the producer exited before ``timeout``. The source's
        current ``next()`` call cannot be interrupted; if it is still running,
        the producer exits after that call returns. The source itself is never
        closed from this thread.
        """
        if timeout < 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be finite and >= 0")

        with self._queue_condition:
            self._closed.set()
            self._queue.clear()
            self._exception = None
            self._queue_condition.notify_all()

        if self._thread is not threading.current_thread():
            self._thread.join(timeout)
        return not self._thread.is_alive()

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Block the background thread for the duration of the ``with``-block.

        Acquires the lock that the producer holds while calling
        ``next(source)``, so callers can drive ``source`` from another thread
        (e.g. for eval) without racing. Acquiring the lock can stall for up to
        one ``next(source)`` step. Don't pull from this iterator inside the
        ``with``-block: the paused producer can't refill the queue, so once it's
        empty ``next()`` blocks forever.
        """
        with self._lock:
            yield
