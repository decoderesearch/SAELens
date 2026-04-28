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

    ``paused()`` is a context manager that pauses the producer for the duration
    of the ``with``-block by holding the lock that gates calls to
    ``next(source)``. Callers can then drive the source themselves (e.g. eval)
    without racing the producer on shared generator state. Note that the lock
    is held *across* a ``next(source)`` call, so acquiring it can stall for up
    to one full source step (e.g. one LLM forward pass) at unlucky moments.
    Also note the queue may already contain up to ``prefetch`` items that the
    producer enqueued before ``paused()`` acquired the lock.

    There is no explicit shutdown API. The producer is daemonized so it dies
    with the process; if the consumer stops pulling early (e.g. exits the
    training loop without exhausting the iterator), the thread will block on
    ``queue.put`` until the process exits. Holding the buffered tensors keeps
    GPU memory pinned, which matters in notebook/multi-run contexts.
    """

    def __init__(self, source: Iterator[T], prefetch: int = 4):
        if prefetch < 1:
            raise ValueError("prefetch must be >= 1")
        self._queue: queue.Queue[object] = queue.Queue(maxsize=prefetch)
        self._lock = threading.Lock()
        self._exception: BaseException | None = None
        self._done = False
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
        # BaseException (not Exception) so KeyboardInterrupt etc. don't silently
        # kill the producer without surfacing to the consumer.
        except BaseException as e:  # noqa: BLE001
            self._exception = e
        finally:
            self._queue.put(_SENTINEL)

    def __iter__(self) -> "PrefetchingIterator[T]":
        return self

    def __next__(self) -> T:
        # Iterator protocol: once StopIteration was raised, every subsequent
        # call must also raise it. Without this guard we'd block forever on
        # an empty queue with no producer alive. Source-side exceptions are
        # raised once on the first SENTINEL, matching Python generator
        # semantics (subsequent calls then raise StopIteration).
        if self._done:
            raise StopIteration
        item = self._queue.get()
        if item is _SENTINEL:
            self._done = True
            if self._exception is not None:
                raise self._exception
            raise StopIteration
        return item  # type: ignore[return-value]

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Block the background thread for the duration of the ``with``-block.

        Acquires the lock that the producer holds while calling
        ``next(source)``, so callers can drive ``source`` from another thread
        (e.g. for eval) without racing. Acquiring the lock can stall for up to
        one ``next(source)`` step.
        """
        with self._lock:
            yield
