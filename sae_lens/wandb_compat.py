from __future__ import annotations

import os
import uuid
from typing import Any


def _load_backend() -> tuple[Any, str]:
    backend = os.getenv("SAE_LENS_LOGGING_BACKEND", "auto").lower()
    if backend not in {"auto", "wandb", "swanlab"}:
        backend = "auto"

    if backend in {"auto", "swanlab"}:
        try:
            import swanlab as backend_module  # type: ignore

            return backend_module, "swanlab"
        except Exception:
            if backend == "swanlab":
                raise

    import wandb as backend_module  # type: ignore

    return backend_module, "wandb"


wandb, BACKEND = _load_backend()


def generate_id() -> str:
    util = getattr(wandb, "util", None)
    if util is not None:
        generator = getattr(util, "generate_id", None)
        if callable(generator):
            try:
                return generator()
            except Exception:
                pass
    return uuid.uuid4().hex
