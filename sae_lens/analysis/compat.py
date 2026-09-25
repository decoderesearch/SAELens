import importlib.metadata
import importlib.util

from packaging.version import parse as parse_version


def get_transformer_lens_version() -> tuple[int, int, int]:
    """Get transformer-lens version as (major, minor, patch)."""
    version_str = importlib.metadata.version("transformer-lens")
    version = parse_version(version_str)
    return (version.major, version.minor, version.micro)


def has_transformer_bridge() -> bool:
    """Check if TransformerBridge is available (v3+)."""
    # accommodate when transformer-lens is installed from source and version is set to 0.0.0
    if parse_version(importlib.metadata.version("transformer-lens")) == parse_version(
        "0.0.0"
    ):
        return True
    major, _, _ = get_transformer_lens_version()
    return major >= 3


def has_hooked_transformer() -> bool:
    """Check if HookedTransformer is available (removed in v4)."""
    return importlib.util.find_spec("transformer_lens.HookedTransformer") is not None
