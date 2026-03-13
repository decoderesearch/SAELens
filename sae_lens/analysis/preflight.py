"""Lightweight preflight checks for SAE analysis workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from sae_lens.saes.sae import SAE


def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, Mapping):
        return dict(metadata)
    to_dict = getattr(metadata, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    items = getattr(metadata, "items", None)
    if callable(items):
        return {str(key): value for key, value in items()}
    if hasattr(metadata, "__dict__"):
        return {
            str(key): value
            for key, value in vars(metadata).items()
            if not str(key).startswith("_")
        }
    raise TypeError(f"Unsupported metadata type: {type(metadata)!r}")


def _extract_model_name(model: Any) -> str | None:
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        for key in ("model_name", "original_architecture", "architecture"):
            value = getattr(cfg, key, None)
            if value:
                return str(value)
    for key in ("name", "model_name"):
        value = getattr(model, key, None)
        if value:
            return str(value)
    return None


def _resolve_hook_name(model: Any, hook_name: str | None) -> str | None:
    if hook_name is None:
        return None
    resolver = getattr(model, "_resolve_hook_name", None)
    if callable(resolver):
        try:
            resolved = resolver(hook_name)
        except Exception:
            return hook_name
        if isinstance(resolved, str):
            return resolved
    return hook_name


def _hook_names(model: Any) -> set[str] | None:
    hook_dict = getattr(model, "hook_dict", None)
    if hook_dict is None:
        return None
    if isinstance(hook_dict, Mapping):
        return {str(key) for key in hook_dict.keys()}
    keys = getattr(hook_dict, "keys", None)
    if callable(keys):
        return {str(key) for key in keys()}
    return None


def _tracked_keys(
    expected_metadata: Mapping[str, Any] | None,
    metadata_keys: Sequence[str] | None,
) -> list[str] | None:
    if metadata_keys is not None:
        return [str(key) for key in metadata_keys]
    if expected_metadata is not None:
        return [str(key) for key in expected_metadata.keys()]
    return None


def _subset_metadata_comparison(
    observed_metadata: Mapping[str, Any],
    *,
    expected_metadata: Mapping[str, Any] | None,
    metadata_keys: Sequence[str] | None,
) -> dict[str, Any]:
    tracked = _tracked_keys(expected_metadata, metadata_keys)
    if tracked is None:
        tracked = sorted(str(key) for key in observed_metadata.keys())

    mismatches: list[dict[str, Any]] = []
    missing_required: list[str] = []
    missing_expected: list[str] = []

    expected_dict = dict(expected_metadata or {})
    for key in tracked:
        observed_present = key in observed_metadata
        expected_present = key in expected_dict
        if expected_present and not observed_present:
            missing_expected.append(key)
            continue
        if expected_present and observed_metadata.get(key) != expected_dict[key]:
            mismatches.append(
                {
                    "key": key,
                    "expected": expected_dict[key],
                    "observed": observed_metadata.get(key),
                }
            )
        if not expected_present and not observed_present:
            missing_required.append(key)

    return {
        "tracked_keys": tracked,
        "mismatches": mismatches,
        "missing_expected": missing_expected,
        "missing_required": missing_required,
    }


def check_sae_metadata(
    sae: SAE[Any],
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    metadata_keys: Sequence[str] | None = None,
    required_keys: Sequence[str] = ("hook_name",),
) -> dict[str, Any]:
    """Check whether an SAE exposes the metadata needed for analysis."""
    observed_metadata = _metadata_to_dict(sae.cfg.metadata)
    comparison = _subset_metadata_comparison(
        observed_metadata,
        expected_metadata=expected_metadata,
        metadata_keys=metadata_keys,
    )
    missing_required = [
        str(key)
        for key in required_keys
        if observed_metadata.get(str(key)) in (None, "", ())
    ]
    issues = (
        len(comparison["mismatches"])
        + len(comparison["missing_expected"])
        + len(comparison["missing_required"])
        + len(missing_required)
    )
    status = "pass" if issues == 0 else ("fail" if missing_required else "warn")
    return {
        "status": status,
        "summary": {
            "required_keys_ok": len(missing_required) == 0,
            "metadata_match": len(comparison["mismatches"]) == 0
            and len(comparison["missing_expected"]) == 0,
        },
        "details": {
            "required_keys": [str(key) for key in required_keys],
            "observed_metadata": observed_metadata,
            **comparison,
            "missing_required": missing_required,
        },
    }


def check_sae_hook_compatibility(
    sae: SAE[Any],
    model: Any,
    *,
    internal_hook_suffix: str = "hook_sae_acts_post",
) -> dict[str, Any]:
    """Check whether an SAE's hook metadata is compatible with a model."""
    metadata = _metadata_to_dict(sae.cfg.metadata)
    base_hook_name = metadata.get("hook_name_out") or metadata.get("hook_name")
    resolved_hook_name = _resolve_hook_name(model, base_hook_name)
    available_hooks = _hook_names(model)
    model_name = _extract_model_name(model)
    metadata_model_name = metadata.get("model_name")

    resolved_internal_hook = None
    get_sae_hook_name = getattr(model, "get_sae_hook_name", None)
    if callable(get_sae_hook_name):
        try:
            resolved_internal_hook = get_sae_hook_name(sae, internal=internal_hook_suffix)
        except Exception:
            resolved_internal_hook = None

    if available_hooks is None:
        base_hook_present = None
        resolved_hook_present = None
        internal_hook_present = None
        status = "warn"
    else:
        base_hook_present = (
            base_hook_name in available_hooks if isinstance(base_hook_name, str) else False
        )
        resolved_hook_present = (
            resolved_hook_name in available_hooks
            if isinstance(resolved_hook_name, str)
            else False
        )
        internal_hook_present = (
            resolved_internal_hook in available_hooks
            if isinstance(resolved_internal_hook, str)
            else None
        )
        status = "pass" if (base_hook_present or resolved_hook_present) else "fail"

    if (
        status == "pass"
        and metadata_model_name is not None
        and model_name is not None
        and metadata_model_name != model_name
    ):
        status = "warn"

    return {
        "status": status,
        "summary": {
            "base_hook_present": base_hook_present,
            "resolved_hook_present": resolved_hook_present,
            "alias_changed": bool(
                isinstance(base_hook_name, str)
                and isinstance(resolved_hook_name, str)
                and base_hook_name != resolved_hook_name
            ),
            "model_name_matches": (
                None
                if metadata_model_name is None or model_name is None
                else metadata_model_name == model_name
            ),
        },
        "details": {
            "base_hook_name": base_hook_name,
            "resolved_hook_name": resolved_hook_name,
            "resolved_internal_hook": resolved_internal_hook,
            "internal_hook_present": internal_hook_present,
            "available_hook_count": None if available_hooks is None else len(available_hooks),
            "metadata_model_name": metadata_model_name,
            "model_name": model_name,
        },
    }


def _flatten_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().reshape(-1)


def check_sae_reconstruction(
    sae: SAE[Any],
    activations: torch.Tensor,
) -> dict[str, Any]:
    """Check encode/decode reconstruction quality on a batch of activations."""
    feature_acts = sae.encode(activations)
    reconstructed = sae.decode(feature_acts)

    original_flat = _flatten_tensor(activations)
    reconstructed_flat = _flatten_tensor(reconstructed)
    if original_flat.shape != reconstructed_flat.shape:
        raise ValueError(
            "Reconstructed activations must match the original activation shape"
        )

    diff = original_flat - reconstructed_flat
    original_norm = torch.linalg.vector_norm(original_flat)
    reconstructed_norm = torch.linalg.vector_norm(reconstructed_flat)

    cosine_similarity = None
    if float(original_norm) > 0.0 and float(reconstructed_norm) > 0.0:
        cosine_similarity = float(
            torch.dot(original_flat, reconstructed_flat)
            / (original_norm * reconstructed_norm)
        )

    variance = torch.var(original_flat, unbiased=False)
    explained_variance = None
    if float(variance) > 0.0:
        explained_variance = float(
            1.0 - (torch.var(diff, unbiased=False) / variance)
        )

    status = "pass"
    if cosine_similarity is not None and cosine_similarity < 0.8:
        status = "fail"
    elif cosine_similarity is None or cosine_similarity < 0.9:
        status = "warn"

    return {
        "status": status,
        "summary": {
            "feature_shape": list(feature_acts.shape),
        },
        "metrics": {
            "cosine_similarity": cosine_similarity,
            "explained_variance": explained_variance,
            "mean_squared_error": float(torch.mean(diff.pow(2))),
            "max_absolute_error": float(torch.max(torch.abs(diff))),
            "feature_nonzero_fraction": float(torch.mean((feature_acts != 0).float())),
        },
    }


def run_sae_preflight(
    sae: SAE[Any],
    *,
    model: Any | None = None,
    activations: torch.Tensor | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
    metadata_keys: Sequence[str] | None = None,
    required_keys: Sequence[str] = ("hook_name",),
) -> dict[str, Any]:
    """Run a compact preflight suite for a candidate SAE analysis setup."""
    reports = {
        "metadata": check_sae_metadata(
            sae,
            expected_metadata=expected_metadata,
            metadata_keys=metadata_keys,
            required_keys=required_keys,
        )
    }
    if model is not None:
        reports["hook_compatibility"] = check_sae_hook_compatibility(sae, model)
    if activations is not None:
        reports["reconstruction"] = check_sae_reconstruction(sae, activations)

    status_priority = {"fail": 3, "warn": 2, "pass": 1}
    overall_status = max(
        (report["status"] for report in reports.values()),
        key=lambda status: status_priority[status],
    )
    return {
        "status": overall_status,
        "summary": {
            "check_count": len(reports),
            "status_counts": {
                "pass": sum(report["status"] == "pass" for report in reports.values()),
                "warn": sum(report["status"] == "warn" for report in reports.values()),
                "fail": sum(report["status"] == "fail" for report in reports.values()),
            },
        },
        "details": reports,
    }


__all__ = [
    "check_sae_hook_compatibility",
    "check_sae_metadata",
    "check_sae_reconstruction",
    "run_sae_preflight",
]
