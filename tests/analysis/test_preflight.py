from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from sae_lens.analysis.preflight import (
    check_sae_hook_compatibility,
    check_sae_metadata,
    check_sae_reconstruction,
    run_sae_preflight,
)
from sae_lens.saes.sae import SAE, SAEMetadata


@dataclass
class DummyConfig:
    metadata: SAEMetadata


class DummySAE:
    def __init__(self, metadata: SAEMetadata):
        self.cfg = DummyConfig(metadata=metadata)

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        return activations * 2.0

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts / 2.0


class DummyModel:
    def __init__(
        self,
        hooks: list[str],
        *,
        model_name: str = "tiny-stories-1M",
        resolved: dict[str, str] | None = None,
    ):
        self.hook_dict = {hook_name: object() for hook_name in hooks}
        self.cfg = type("Cfg", (), {"model_name": model_name})()
        self._resolved = dict(resolved or {})

    def _resolve_hook_name(self, hook_name: str) -> str:
        return self._resolved.get(hook_name, hook_name)

    def get_sae_hook_name(
        self, sae: DummySAE, internal: str = "hook_sae_acts_post"
    ) -> str:
        metadata = sae.cfg.metadata
        base_hook = metadata.hook_name_out or metadata.hook_name
        if base_hook is None:
            raise ValueError("hook_name is required")
        return f"{self._resolve_hook_name(base_hook)}.{internal}"


def make_sae(*, hook_name: str | None = "blocks.0.hook_resid_pre") -> DummySAE:
    return DummySAE(
        SAEMetadata(
            hook_name=hook_name,
            model_name="tiny-stories-1M",
            hook_head_index=None,
            prepend_bos=True,
        )
    )


def as_sae(sae: DummySAE) -> SAE[Any]:
    return cast(SAE[Any], sae)


def test_check_sae_metadata_passes_for_expected_subset() -> None:
    sae = make_sae()
    report = check_sae_metadata(
        as_sae(sae),
        expected_metadata={"hook_name": "blocks.0.hook_resid_pre"},
    )
    assert report["status"] == "pass"
    assert report["summary"]["metadata_match"] is True


def test_check_sae_metadata_warns_for_expected_mismatch() -> None:
    sae = make_sae()
    report = check_sae_metadata(
        as_sae(sae),
        expected_metadata={"model_name": "other-model"},
    )
    assert report["status"] == "warn"
    assert len(report["details"]["mismatches"]) == 1


def test_check_sae_metadata_fails_when_hook_name_missing() -> None:
    sae = make_sae(hook_name=None)
    report = check_sae_metadata(as_sae(sae))
    assert report["status"] == "fail"
    assert report["details"]["missing_required"] == ["hook_name"]


def test_check_sae_hook_compatibility_passes_with_alias_resolution() -> None:
    sae = make_sae(hook_name="blocks.0.hook_mlp_out")
    model = DummyModel(
        ["blocks.0.mlp.hook_out"],
        resolved={"blocks.0.hook_mlp_out": "blocks.0.mlp.hook_out"},
    )
    report = check_sae_hook_compatibility(as_sae(sae), model)
    assert report["status"] == "pass"
    assert report["summary"]["resolved_hook_present"] is True
    assert report["summary"]["alias_changed"] is True


def test_check_sae_hook_compatibility_fails_when_hook_absent() -> None:
    sae = make_sae(hook_name="blocks.0.hook_resid_pre")
    model = DummyModel(["blocks.1.hook_resid_pre"])
    report = check_sae_hook_compatibility(as_sae(sae), model)
    assert report["status"] == "fail"
    assert report["summary"]["base_hook_present"] is False


def test_check_sae_reconstruction_reports_metrics() -> None:
    sae = make_sae()
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    report = check_sae_reconstruction(as_sae(sae), activations)
    assert report["status"] == "pass"
    assert report["summary"]["feature_shape"] == [2, 2]
    assert report["metrics"]["feature_nonzero_fraction"] == 1.0


def test_run_sae_preflight_aggregates_subchecks() -> None:
    sae = make_sae()
    model = DummyModel(["blocks.0.hook_resid_pre"])
    activations = torch.tensor([[1.0, 0.0], [0.5, 0.25]])
    report = run_sae_preflight(
        # Cast the dummy test double to the public SAE type to exercise the
        # preflight helpers without constructing a full real SAE instance.
        as_sae(sae),
        model=model,
        activations=activations,
        expected_metadata={"hook_name": "blocks.0.hook_resid_pre"},
    )
    assert report["status"] == "pass"
    assert report["summary"]["check_count"] == 3
