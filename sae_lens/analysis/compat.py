import importlib.metadata


def get_transformer_lens_version() -> tuple[int, int, int]:
    """Get transformer-lens version as (major, minor, patch)."""
    version = importlib.metadata.version("transformer-lens")
    # Handle "3.0.0b1" -> (3, 0, 0)
    clean = version.split("a")[0].split("b")[0].split("rc")[0]
    parts = clean.split(".")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def has_transformer_bridge() -> bool:
    """Check if TransformerBridge is available (v3+)."""
    major, _, _ = get_transformer_lens_version()
    return major >= 3
