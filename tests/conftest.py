import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from sae_lens.saes.sae import SAE
from sae_lens.saes.standard_sae import StandardSAEConfig
from tests.helpers import TINYSTORIES_MODEL, load_model_cached

torch.set_grad_enabled(True)

# sparsify's triton implementation breaks in CI, so just disable it
os.environ["SPARSIFY_DISABLE_TRITON"] = "1"


@pytest.fixture(autouse=True)
def set_grad_enabled():
    # somehow grad is getting disabled before some tests
    torch.set_grad_enabled(True)


@pytest.fixture(autouse=True)
def reproducibility():
    """Apply various mechanisms to try to prevent nondeterminism in test runs."""
    # I have not in general attempted to verify that the below are necessary
    # for reproducibility, only that they are likely to help and unlikely to
    # hurt.
    # https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    seed = 0x1234_5678_9ABC_DEF0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # Python native RNG; docs don't give any limitations on seed range
    random.seed(seed)
    # this is a "legacy" method that operates on a global RandomState
    # sounds like the argument must be in [0, 2**32)
    np.random.seed(seed & 0xFFFF_FFFF)


@pytest.fixture
def ts_model():
    return load_model_cached(TINYSTORIES_MODEL)


# we started running out of space in CI, try cleaing up tmp paths after each test
@pytest.fixture(autouse=True)
def cleanup_tmp_path(tmp_path: Path):
    yield  # This line allows the test to run and use tmp_path
    # After the test is done, clean up the directory
    for item in tmp_path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def get_disk_usage_mb() -> tuple[float, float]:
    """Get used and available disk space in MB for root filesystem."""
    try:
        result = subprocess.run(
            ["df", "--block-size=1M", "/"], capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 4:
                used_mb = float(parts[2].replace("M", ""))
                avail_mb = float(parts[3].replace("M", ""))
                return used_mb, avail_mb
    except Exception:
        pass
    return 0.0, 0.0


@pytest.fixture(autouse=True)
def print_disk_space_after_test_in_CI(
    request: pytest.FixtureRequest, capfd: pytest.CaptureFixture[Any]
) -> Any:
    # Track disk space before test (only in CI)
    is_ci = bool(os.getenv("CI"))
    if is_ci:
        used_before, avail_before = get_disk_usage_mb()
    else:
        used_before, avail_before = 0.0, 0.0

    yield

    # Print disk space change after test (only in CI to avoid spam in local dev)
    if is_ci:
        test_name = request.node.name
        used_after, avail_after = get_disk_usage_mb()

        used_diff = used_after - used_before
        avail_diff = (
            avail_before - avail_after
        )  # Positive means less available (used more)

        # Only print if there's a notable change (>= 5 MB)
        if abs(used_diff) >= 5 or abs(avail_diff) >= 5:
            with capfd.disabled():
                print(f"\n=== Disk change after {test_name} ===", flush=True)  # noqa: T201
                print(  # noqa: T201
                    f"Used: {used_before:.0f}M -> {used_after:.0f}M ({used_diff:+.0f}M)",
                    flush=True,
                )
                print(  # noqa: T201
                    f"Available: {avail_before:.0f}M -> {avail_after:.0f}M ({-avail_diff:+.0f}M)",
                    flush=True,
                )


@pytest.fixture
def gpt2_res_jb_l4_sae() -> SAE[StandardSAEConfig]:
    return SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.4.hook_resid_pre",
        device="cpu",
    )
