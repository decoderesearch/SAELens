from typing import Any


class InterruptedException(Exception):
    pass


def interrupt_callback(sig_num: Any, stack_frame: Any) -> None:  # noqa: ARG001
    raise InterruptedException()
