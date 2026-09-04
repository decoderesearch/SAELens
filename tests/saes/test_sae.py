import copy
import pickle
import tracemalloc
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from sae_lens import __version__
from sae_lens.loading.pretrained_sae_loaders import sae_lens_disk_loader
from sae_lens.registry import get_sae_class, get_sae_training_class
from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    SAEMetadata,
    TrainingSAE,
    TrainingSAEConfig,
)
from sae_lens.saes.standard_sae import StandardTrainingSAE
from sae_lens.training.activation_scaler import ActivationScaler
from tests.helpers import (
    ALL_ARCHITECTURES,
    ALL_FOLDABLE_ARCHITECTURES,
    ALL_TRAINING_ARCHITECTURES,
    assert_close,
    build_sae_cfg_for_arch,
    build_sae_training_cfg,
    build_sae_training_cfg_for_arch,
    correlated_activations,
    random_params,
)


@pytest.mark.parametrize("architecture", ALL_TRAINING_ARCHITECTURES)
def test_TrainingSAEConfig_to_and_from_dict_all_architectures(architecture: str):
    cfg = build_sae_training_cfg_for_arch(architecture=architecture)
    reloaded_cfg = TrainingSAEConfig.from_dict(cfg.to_dict())
    assert reloaded_cfg.to_dict() == cfg.to_dict()
    assert reloaded_cfg.__class__ == cfg.__class__
    assert reloaded_cfg.__class__ == get_sae_training_class(architecture)[1]
    assert cfg.to_dict()["metadata"]["training_architecture"] == architecture


def test_TrainingSAEConfig_serializes_training_architecture_details():
    cfg = build_sae_training_cfg_for_arch(
        architecture="matryoshka_batchtopk",
        d_sae=128,
        k=8,
        rescale_acts_by_decoder_norm=False,
        topk_threshold_lr=0.03,
        matryoshka_widths=[32, 64, 128],
        use_matryoshka_aux_loss=True,
    )

    cfg_dict = cfg.to_dict()
    metadata = cfg_dict["metadata"]

    assert metadata["training_architecture"] == "matryoshka_batchtopk"
    assert metadata["training_architecture_details"] == {
        "decoder_init_norm": 0.1,
        "aux_loss_coefficient": 1.0,
        "use_sparse_activations": False,
        "k": 8,
        "rescale_acts_by_decoder_norm": False,
        "topk_threshold_lr": 0.03,
        "matryoshka_widths": [32, 64, 128],
        "use_matryoshka_aux_loss": True,
    }


@pytest.mark.parametrize("architecture", ALL_TRAINING_ARCHITECTURES)
def test_SAEConfig_to_and_from_dict_all_architectures(architecture: str):
    cfg_dict = build_sae_training_cfg_for_arch(
        architecture
    ).get_inference_sae_cfg_dict()
    reloaded_cfg = SAEConfig.from_dict(cfg_dict)
    if architecture in {"batchtopk", "matryoshka_batchtopk"}:
        assert reloaded_cfg.architecture() == "jumprelu"
    else:
        assert reloaded_cfg.architecture() == architecture
    assert reloaded_cfg.to_dict() == cfg_dict
    assert reloaded_cfg.__class__ == get_sae_class(reloaded_cfg.architecture())[1]


def test_SAEMetadata_initialization_empty():
    metadata = SAEMetadata()
    expected_dict = {
        "sae_lens_version": __version__,
        "sae_lens_training_version": __version__,
    }
    assert metadata.to_dict() == expected_dict


def test_SAEMetadata_initialization_with_kwargs():
    metadata = SAEMetadata(
        model_name="test_model", hook_name="test_hook", custom_field="value"
    )
    assert metadata.model_name == "test_model"
    assert metadata.hook_name == "test_hook"
    assert metadata.custom_field == "value"


def test_SAEMetadata_missing_attributes_return_none():
    metadata = SAEMetadata()
    assert metadata.nonexistent_field is None
    assert metadata.another_missing_field is None


def test_SAEMetadata_dictionary_style_access():
    metadata = SAEMetadata()
    metadata["key1"] = "value1"
    metadata["key2"] = 42

    assert metadata["key1"] == "value1"
    assert metadata["key2"] == 42
    assert metadata["nonexistent"] is None


def test_SAEMetadata_contains_operator():
    metadata = SAEMetadata(existing_field="value")

    assert "existing_field" in metadata
    assert "nonexistent_field" not in metadata

    metadata["new_field"] = "new_value"
    assert "new_field" in metadata


def test_SAEMetadata_get_method():
    metadata = SAEMetadata(existing_field="value")

    assert metadata.get("existing_field") == "value"
    assert metadata.get("nonexistent_field") is None
    assert metadata.get("nonexistent_field", "default") == "default"


def test_SAEMetadata_keys_values_items():
    metadata = SAEMetadata(field1="value1", field2="value2")

    expected_keys = {
        "field1",
        "field2",
        "sae_lens_version",
        "sae_lens_training_version",
    }
    expected_values = {"value1", "value2", __version__, __version__}
    expected_items = {
        ("field1", "value1"),
        ("field2", "value2"),
        ("sae_lens_version", __version__),
        ("sae_lens_training_version", __version__),
    }

    assert set(metadata.keys()) == expected_keys
    assert set(metadata.values()) == expected_values
    assert set(metadata.items()) == expected_items


def test_SAEMetadata_to_dict_from_dict_round_trip():
    original_data = {"model_name": "test", "hook_name": "hook", "custom": 123}
    metadata = SAEMetadata(**original_data)

    dict_repr = metadata.to_dict()
    restored_metadata = SAEMetadata.from_dict(dict_repr)

    # The restored metadata should have version fields added
    expected_data = {
        **original_data,
        "sae_lens_version": __version__,
        "sae_lens_training_version": __version__,
    }

    assert restored_metadata.to_dict() == expected_data
    assert restored_metadata.model_name == "test"
    assert restored_metadata.hook_name == "hook"
    assert restored_metadata.custom == 123


def test_SAEMetadata_equality():
    metadata1 = SAEMetadata(field1="value1", field2="value2")
    metadata2 = SAEMetadata(field1="value1", field2="value2")
    metadata3 = SAEMetadata(field1="value1", field2="different")

    assert metadata1 == metadata2
    assert metadata1 != metadata3
    assert metadata1 != "not_metadata"


def test_SAEMetadata_deep_copy():
    metadata = SAEMetadata(field1="value1", nested={"key": "value"})
    copied_metadata = copy.deepcopy(metadata)

    assert copied_metadata == metadata
    assert copied_metadata is not metadata
    assert copied_metadata.nested is not metadata.nested


def test_SAEMetadata_pickling():
    metadata = SAEMetadata(field1="value1", field2=42)
    pickled = pickle.dumps(metadata)
    unpickled = pickle.loads(pickled)

    assert unpickled == metadata
    assert unpickled.field1 == "value1"
    assert unpickled.field2 == 42


def test_SAEMetadata_version_defaults():
    metadata = SAEMetadata()
    assert metadata.sae_lens_version == __version__
    assert metadata.sae_lens_training_version == __version__


def test_SAEMetadata_version_override():
    metadata = SAEMetadata(sae_lens_version="custom_version")
    assert metadata.sae_lens_version == "custom_version"


def test_SAEMetadata_repr():
    metadata = SAEMetadata(field1="value1")
    repr_str = repr(metadata)
    assert "SAEMetadata" in repr_str
    assert "field1" in repr_str
    assert "value1" in repr_str


def test_SAEMetadata_dynamic_attribute_setting():
    metadata = SAEMetadata()
    metadata.new_field = "new_value"
    metadata.another_field = 123

    assert metadata.new_field == "new_value"
    assert metadata.another_field == 123
    assert "new_field" in metadata
    assert "another_field" in metadata


def test_SAE_from_pretrained_deprecated_usage_as_tuple():
    sae = SAE.from_pretrained("gpt2-small-hook-z-kk", "blocks.2.hook_z", device="cpu")

    # Test tuple unpacking with deprecation warning
    with pytest.warns(DeprecationWarning, match="Unpacking SAE objects is deprecated"):
        tup_sae, tup_cfg_dict, tup_sparsity = sae
    assert tup_sae is sae
    assert tup_cfg_dict == sae.cfg.to_dict()
    assert tup_sparsity is None

    # Test indexing with deprecation warnings
    with pytest.warns(DeprecationWarning, match="Indexing SAE objects is deprecated"):
        assert sae[0] is sae
    with pytest.warns(DeprecationWarning, match="Indexing SAE objects is deprecated"):
        assert sae[1] == sae.cfg.to_dict()
    with pytest.warns(DeprecationWarning, match="Indexing SAE objects is deprecated"):
        assert sae[2] is None

    # Test len with deprecation warning
    with pytest.warns(
        DeprecationWarning, match="Getting length of SAE objects is deprecated"
    ):
        assert len(sae) == 3


@pytest.mark.parametrize("architecture", ALL_TRAINING_ARCHITECTURES)
def test_TrainingSAE_fold_activation_norm_scaling_factor_all_architectures(
    architecture: str,
):
    cfg = build_sae_training_cfg_for_arch(architecture)
    sae = TrainingSAE.from_dict(cfg.to_dict())
    random_params(sae)

    inputs = torch.randn(100, cfg.d_in)

    original_outputs = sae(inputs)
    original_features = sae.encode(inputs)

    sae.fold_activation_norm_scaling_factor(2.0)

    folded_outputs = 2.0 * sae(inputs / 2.0)
    folded_features = sae.encode(inputs / 2.0)

    assert_close(folded_outputs, original_outputs)
    if architecture in {"topk", "batchtopk", "matryoshka_batchtopk"}:
        # Due to how rescale_acts_by_decoder_norm works in TopKSAEs, it's equivalent to
        # folding the W_dec norm after folding the activation norm scaling factor.
        # this is fine, since we just care about the ouputs being the same.
        assert_close(folded_features, original_features / 2.0)
    else:
        assert_close(folded_features, original_features)


def _estimate_whitening_scaler(d_in: int) -> tuple[ActivationScaler, torch.Tensor]:
    provider = correlated_activations(d_in, batch_size=256)
    scaler = ActivationScaler()
    scaler.estimate_whitening(
        d_in=d_in, data_provider=provider, n_batches_for_norm_estimate=4
    )
    return scaler, next(provider)


def _fold_scaler_whitening(sae: SAE[Any], scaler: ActivationScaler) -> None:
    assert scaler.whitening is not None
    sae.fold_activation_whitening(
        mean=scaler.whitening.mean,
        whitening_matrix=scaler.whitening.matrix,
        unwhitening_matrix=scaler.whitening.inverse_matrix,
    )


@pytest.mark.parametrize("apply_b_dec_to_input", [True, False])
@pytest.mark.parametrize("architecture", ALL_TRAINING_ARCHITECTURES)
def test_TrainingSAE_fold_activation_whitening_all_architectures(
    architecture: str, apply_b_dec_to_input: bool
):
    # float64 so that a threshold or topk decision cannot flip from rounding alone
    cfg = build_sae_training_cfg_for_arch(
        architecture,
        normalize_activations="covariance_whitening",
        apply_b_dec_to_input=apply_b_dec_to_input,
        dtype="float64",
    )
    sae = get_sae_training_class(architecture)[0](cfg)
    random_params(sae)
    scaler, inputs = _estimate_whitening_scaler(cfg.d_in)
    inputs = inputs.to(torch.float64)

    if architecture == "matching_pursuit":
        with pytest.raises(NotImplementedError):
            _fold_scaler_whitening(sae, scaler)
        return

    whitened_features = sae.encode(scaler.scale(inputs))
    whitened_outputs = scaler.unscale(sae(scaler.scale(inputs)))

    _fold_scaler_whitening(sae, scaler)

    assert sae.cfg.normalize_activations == "none"
    assert_close(sae.encode(inputs), whitened_features)
    assert_close(sae(inputs), whitened_outputs)


@pytest.mark.parametrize("apply_b_dec_to_input", [True, False])
@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
@torch.no_grad()
def test_SAE_fold_activation_whitening_all_architectures(
    architecture: str, apply_b_dec_to_input: bool
):
    cfg = build_sae_cfg_for_arch(
        architecture, apply_b_dec_to_input=apply_b_dec_to_input, dtype="float64"
    )
    sae = get_sae_class(architecture)[0](cfg)
    random_params(sae)
    scaler, inputs = _estimate_whitening_scaler(cfg.d_in)
    inputs = inputs.to(torch.float64)

    if architecture in {"temporal", "matching_pursuit"}:
        with pytest.raises(NotImplementedError):
            _fold_scaler_whitening(sae, scaler)
        return

    whitened_features = sae.encode(scaler.scale(inputs))
    whitened_outputs = scaler.unscale(sae(scaler.scale(inputs)))

    _fold_scaler_whitening(sae, scaler)

    assert sae.cfg.normalize_activations == "none"
    assert_close(sae.encode(inputs), whitened_features)
    assert_close(sae(inputs), whitened_outputs)


def test_TrainingSAE_fold_activation_whitening_float32_matches_within_1e_5():
    cfg = build_sae_training_cfg(normalize_activations="covariance_whitening")
    sae = StandardTrainingSAE(cfg)
    random_params(sae)
    scaler, inputs = _estimate_whitening_scaler(cfg.d_in)

    whitened_outputs = scaler.unscale(sae(scaler.scale(inputs)))
    _fold_scaler_whitening(sae, scaler)

    assert sae.W_enc.dtype == torch.float32
    # float32 rounding scales with the largest outputs, not with each element
    output_scale = whitened_outputs.abs().max().item()
    assert_close(sae(inputs), whitened_outputs, atol=1e-5 * output_scale, rtol=1e-5)


@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
@torch.no_grad()
def test_sae_fold_w_dec_norm_all_architectures(architecture: str):
    cfg = build_sae_cfg_for_arch(architecture)
    sae = SAE.from_dict(cfg.to_dict())

    # make sure all parameters are not 0s
    for param in sae.parameters():
        param.data = torch.rand_like(param)

    assert sae.W_dec.norm(dim=-1).mean().item() != pytest.approx(1.0, abs=1e-6)
    sae2 = deepcopy(sae)

    # If this is a topk SAE, assert this throws a NotImplementedError
    if architecture not in ALL_FOLDABLE_ARCHITECTURES:
        with pytest.raises(NotImplementedError):
            sae2.fold_W_dec_norm()
        return

    sae2.fold_W_dec_norm()

    # fold_W_dec_norm should normalize W_dec to have unit norm.
    assert sae2.W_dec.norm(dim=-1).mean().item() == pytest.approx(1.0, abs=1e-6)

    # we expect activations of features to differ by W_dec norm weights.
    activations = torch.randn(10, 4, cfg.d_in, device=cfg.device)
    feature_activations_1 = sae.encode(activations)
    feature_activations_2 = sae2.encode(activations)

    assert_close(
        feature_activations_1.nonzero(),
        feature_activations_2.nonzero(),
    )

    expected_feature_activations_2 = feature_activations_1 * sae.W_dec.norm(dim=-1)
    assert_close(feature_activations_2, expected_feature_activations_2, atol=1e-5)

    sae_out_1 = sae.decode(feature_activations_1)
    sae_out_2 = sae2.decode(feature_activations_2)

    # but actual outputs should be the same
    assert_close(sae_out_1, sae_out_2, atol=1e-5)


@pytest.mark.parametrize("architecture", ALL_FOLDABLE_ARCHITECTURES)
@torch.no_grad()
def test_sae_fold_w_dec_norm_with_d_sae_of_1(architecture: str):
    cfg = build_sae_cfg_for_arch(architecture, d_sae=1)
    sae = SAE.from_dict(cfg.to_dict())
    random_params(sae)

    activations = 10.0 * torch.ones(8, cfg.d_in, device=cfg.device)
    bias_shapes_before = {
        name: param.shape
        for name, param in sae.named_parameters()
        if name in {"b_enc", "b_gate", "b_mag", "log_threshold"}
    }
    sae_out_before = sae(activations)

    sae.fold_W_dec_norm()

    # W_dec should be normalized to unit norm.
    assert sae.W_dec.norm(dim=-1).item() == pytest.approx(1.0, abs=1e-6)
    # Validate shape of the single-element bias params.
    assert bias_shapes_before
    for name, param in sae.named_parameters():
        if name in bias_shapes_before:
            assert param.shape == bias_shapes_before[name]
    # Folding preserves the SAE function.
    assert_close(sae(activations), sae_out_before, atol=1e-4)


@pytest.mark.parametrize("architecture", ALL_TRAINING_ARCHITECTURES)
@torch.no_grad()
def test_training_sae_fold_w_dec_norm_all_architectures(architecture: str):
    cfg = build_sae_training_cfg_for_arch(architecture)
    sae = TrainingSAE.from_dict(cfg.to_dict())

    # make sure all parameters are not 0s
    for param in sae.parameters():
        param.data = torch.rand_like(param)

    assert sae.W_dec.norm(dim=-1).mean().item() != pytest.approx(1.0, abs=1e-6)
    sae2 = deepcopy(sae)

    if architecture in {"matching_pursuit"}:
        with pytest.raises(NotImplementedError):
            sae2.fold_W_dec_norm()
        return

    sae2.fold_W_dec_norm()

    # fold_W_dec_norm should normalize W_dec to have unit norm.
    assert sae2.W_dec.norm(dim=-1).mean().item() == pytest.approx(1.0, abs=1e-6)

    # we expect activations of features to differ by W_dec norm weights.
    activations = torch.randn(10, 4, cfg.d_in, device=cfg.device)
    feature_activations_1 = sae.encode(activations)
    feature_activations_2 = sae2.encode(activations)

    assert_close(
        feature_activations_1.nonzero(),
        feature_activations_2.nonzero(),
    )

    if architecture in {"topk", "batchtopk", "matryoshka_batchtopk"}:
        # Due to how rescale_acts_by_decoder_norm works in TopKSAEs, it's like the
        # SAE has the norm folded in throughout the entire training process.
        assert_close(feature_activations_2, feature_activations_1, atol=1e-4, rtol=1e-4)
    else:
        expected_feature_activations_2 = feature_activations_1 * sae.W_dec.norm(dim=-1)
        assert_close(
            feature_activations_2, expected_feature_activations_2, atol=1e-4, rtol=1e-4
        )

    sae_out_1 = sae.decode(feature_activations_1)
    sae_out_2 = sae2.decode(feature_activations_2)

    # but actual outputs should be the same
    assert_close(sae_out_1, sae_out_2)


@pytest.mark.parametrize("architecture", ALL_FOLDABLE_ARCHITECTURES)
@torch.no_grad()
def test_sae_fold_norm_scaling_factor_all_architectures(architecture: str):
    cfg = build_sae_cfg_for_arch(architecture)
    norm_scaling_factor = 3.0

    sae = SAE.from_dict(cfg.to_dict())
    # make sure all parameters are not 0s
    for param in sae.parameters():
        param.data = torch.rand_like(param)

    sae2 = deepcopy(sae)
    sae2.fold_activation_norm_scaling_factor(norm_scaling_factor)

    assert sae2.cfg.normalize_activations == "none"

    assert_close(sae2.W_enc.data, sae.W_enc.data * norm_scaling_factor)

    # we expect activations of features to differ by W_dec norm weights.
    # assume activations are already scaled
    activations = torch.randn(10, 4, cfg.d_in, device=cfg.device)
    # we divide to get the unscale activations
    unscaled_activations = activations / norm_scaling_factor

    feature_activations_1 = sae.encode(activations)
    if feature_activations_1.is_sparse:
        feature_activations_1 = feature_activations_1.to_dense()
    # with the scaling folded in, the unscaled activations should produce the same
    # result.
    feature_activations_2 = sae2.encode(unscaled_activations)
    if feature_activations_2.is_sparse:
        feature_activations_2 = feature_activations_2.to_dense()

    assert_close(
        feature_activations_1.nonzero(),
        feature_activations_2.nonzero(),
    )

    assert_close(feature_activations_2, feature_activations_1, atol=1e-5)

    sae_out_1 = sae.decode(feature_activations_1)
    sae_out_2 = norm_scaling_factor * sae2.decode(feature_activations_2)

    # but actual outputs should be the same
    assert_close(sae_out_1, sae_out_2, atol=1e-5)


@pytest.mark.parametrize("architecture", ALL_FOLDABLE_ARCHITECTURES)
@torch.no_grad()
def test_fold_W_dec_norm_does_not_produce_nan_with_zero_norm_decoder(
    architecture: str,
):
    """
    Regression test for https://github.com/decoderesearch/SAELens/issues/588

    When decoder weights have zero norm (dead latents), the division in
    fold_W_dec_norm should not produce NaN values. This is handled by
    clamping the norm to a minimum of 1e-8.
    """
    cfg = build_sae_cfg_for_arch(architecture)
    sae = SAE.from_dict(cfg.to_dict())

    # Initialize parameters with random values
    for param in sae.parameters():
        param.data = torch.rand_like(param)

    # Set some decoder rows to zero to simulate dead latents
    num_zero_rows = min(5, sae.W_dec.shape[0])
    sae.W_dec.data[:num_zero_rows] = 0.0

    # Verify that we actually have zero-norm rows
    norms_before = sae.W_dec.norm(dim=-1)
    assert (norms_before[:num_zero_rows] == 0).all()

    # TopK SAEs with rescale_acts_by_decoder_norm=False raise NotImplementedError
    if architecture == "topk" and not getattr(
        sae.cfg, "rescale_acts_by_decoder_norm", False
    ):
        with pytest.raises(NotImplementedError):
            sae.fold_W_dec_norm()
        return

    # Call fold_W_dec_norm - this should not produce NaN values
    sae.fold_W_dec_norm()

    # Verify no NaN or Inf values in any parameters
    for name, param in sae.named_parameters():
        assert not torch.isnan(
            param
        ).any(), f"NaN found in {name} after fold_W_dec_norm"
        assert not torch.isinf(
            param
        ).any(), f"Inf found in {name} after fold_W_dec_norm"


@pytest.mark.parametrize("architecture", ALL_TRAINING_ARCHITECTURES)
@torch.no_grad()
def test_training_fold_W_dec_norm_does_not_produce_nan_with_zero_norm_decoder(
    architecture: str,
):
    """
    Regression test for https://github.com/decoderesearch/SAELens/issues/588

    When decoder weights have zero norm (dead latents), the division in
    fold_W_dec_norm should not produce NaN values for TrainingSAE classes.
    """
    cfg = build_sae_training_cfg_for_arch(architecture)
    sae = TrainingSAE.from_dict(cfg.to_dict())

    # Initialize parameters with random values
    for param in sae.parameters():
        param.data = torch.rand_like(param)

    # Set some decoder rows to zero to simulate dead latents
    num_zero_rows = min(5, sae.W_dec.shape[0])
    sae.W_dec.data[:num_zero_rows] = 0.0

    # Verify that we actually have zero-norm rows
    norms_before = sae.W_dec.norm(dim=-1)
    assert (norms_before[:num_zero_rows] == 0).all()

    # Call fold_W_dec_norm - this should not produce NaN values

    if architecture in {"matching_pursuit"}:
        with pytest.raises(NotImplementedError):
            sae.fold_W_dec_norm()
        return

    sae.fold_W_dec_norm()

    # Verify no NaN or Inf values in any parameters
    for name, param in sae.named_parameters():
        assert not torch.isnan(
            param
        ).any(), f"NaN found in {name} after fold_W_dec_norm"
        assert not torch.isinf(
            param
        ).any(), f"Inf found in {name} after fold_W_dec_norm"


@pytest.mark.parametrize("architecture", ALL_TRAINING_ARCHITECTURES)
def test_TrainingSAE_save_and_load_from_checkpoint_all_architectures(
    architecture: str,
    tmp_path: Path,
):
    cfg = build_sae_training_cfg_for_arch(architecture)
    sae = TrainingSAE.from_dict(cfg.to_dict())
    random_params(sae)

    sae.save_model(tmp_path)
    loaded_sae = TrainingSAE.from_dict(cfg.to_dict())
    loaded_sae.load_weights_from_checkpoint(tmp_path)

    for param1, param2 in zip(sae.parameters(), loaded_sae.parameters()):
        assert_close(param1, param2, atol=1e-6, rtol=1e-4)


def test_SAE_load_from_disk_uses_meta_device_optimization(tmp_path: Path):
    cfg = build_sae_training_cfg_for_arch("standard", d_in=512, d_sae=2048)
    sae = TrainingSAE.from_dict(cfg.to_dict())
    random_params(sae)
    sae.save_model(tmp_path)

    # Patch load_state_dict to verify meta device is used before loading
    original_load_state_dict = SAE.load_state_dict
    meta_device_verified = False

    def patched_load_state_dict(
        self: SAE[Any], state_dict: dict[str, Any], assign: bool = False
    ):
        nonlocal meta_device_verified
        # Verify that parameters are on meta device before loading
        for _, param in self.named_parameters():
            if param.device.type == "meta":
                meta_device_verified = True
                break
        return original_load_state_dict(self, state_dict, assign=assign)

    with patch.object(SAE, "load_state_dict", patched_load_state_dict):
        loaded_sae = SAE.load_from_disk(tmp_path, device="cpu")

    assert meta_device_verified, "SAE should use meta device before load_state_dict"
    assert loaded_sae.W_enc.device.type == "cpu"


def test_SAE_load_from_disk_does_not_peak_at_double_memory(tmp_path: Path):
    d_in = 512
    d_sae = 4096
    cfg = build_sae_training_cfg_for_arch("standard", d_in=d_in, d_sae=d_sae)
    sae = TrainingSAE.from_dict(cfg.to_dict())
    random_params(sae)
    sae.save_model(tmp_path)

    # Calculate expected memory for a single SAE's parameters (approximate)
    # W_enc: d_in x d_sae, W_dec: d_sae x d_in, b_enc: d_sae, b_dec: d_in
    expected_param_bytes = (d_in * d_sae + d_sae * d_in + d_sae + d_in) * 4  # float32
    # Allow some overhead for Python objects, but should be well under 2x
    max_allowed_bytes = expected_param_bytes * 1.5

    tracemalloc.start()
    tracemalloc.reset_peak()

    loaded_sae = SAE.load_from_disk(tmp_path, device="cpu")

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Verify peak memory is reasonable (not 2x the parameter size)
    assert peak_memory < max_allowed_bytes, (
        f"Peak memory {peak_memory / 1e6:.2f}MB exceeds allowed "
        f"{max_allowed_bytes / 1e6:.2f}MB (1.5x param size). "
        f"Expected param size: {expected_param_bytes / 1e6:.2f}MB"
    )
    assert loaded_sae is not None


@pytest.mark.parametrize("architecture", ["standard", "gated", "topk", "jumprelu"])
def test_SAE_load_from_disk_memory_efficient_all_architectures(
    architecture: str, tmp_path: Path
):
    cfg = build_sae_training_cfg_for_arch(architecture, d_in=256, d_sae=1024)
    sae = TrainingSAE.from_dict(cfg.to_dict())
    random_params(sae)
    sae.save_model(tmp_path)

    # Verify meta device optimization is used
    original_load_state_dict = SAE.load_state_dict
    used_assign_true = False

    def patched_load_state_dict(
        self: SAE[Any], state_dict: dict[str, Any], assign: bool = False
    ):
        nonlocal used_assign_true
        if assign:
            used_assign_true = True
        return original_load_state_dict(self, state_dict, assign=assign)

    with patch.object(SAE, "load_state_dict", patched_load_state_dict):
        loaded_sae = SAE.load_from_disk(tmp_path, device="cpu")

    assert used_assign_true, "load_state_dict should be called with assign=True"
    assert loaded_sae is not None


def test_SAE_from_pretrained_uses_meta_device_optimization(tmp_path: Path):
    cfg = build_sae_training_cfg_for_arch("standard", d_in=256, d_sae=512)
    sae = TrainingSAE.from_dict(cfg.to_dict())
    random_params(sae)
    sae.save_model(tmp_path)

    cfg_dict, state_dict = sae_lens_disk_loader(tmp_path, device="cpu")

    def mock_loader(
        repo_id: str,  # noqa: ARG001
        folder_name: str,  # noqa: ARG001
        device: str = "cpu",  # noqa: ARG001
        force_download: bool = False,  # noqa: ARG001
        cfg_overrides: dict[str, Any] | None = None,  # noqa: ARG001
    ):
        return cfg_dict, state_dict, None

    # Verify meta device optimization is used
    original_load_state_dict = SAE.load_state_dict
    meta_device_verified = False
    used_assign_true = False

    def patched_load_state_dict(
        self: SAE[Any], state_dict: dict[str, Any], assign: bool = False
    ):
        nonlocal meta_device_verified, used_assign_true
        for _, param in self.named_parameters():
            if param.device.type == "meta":
                meta_device_verified = True
                break
        if assign:
            used_assign_true = True
        return original_load_state_dict(self, state_dict, assign=assign)

    # Mock the pretrained SAEs directory to include our mock release
    mock_sae_info = type(
        "MockSAEInfo", (), {"conversion_func": None, "saes_map": {"mock-sae-id": {}}}
    )()

    with (
        patch.object(SAE, "load_state_dict", patched_load_state_dict),
        patch(
            "sae_lens.saes.sae.NAMED_PRETRAINED_SAE_LOADERS",
            {"sae_lens": mock_loader},
        ),
        patch(
            "sae_lens.saes.sae.get_pretrained_saes_directory",
            return_value={"mock-release": mock_sae_info},
        ),
        patch(
            "sae_lens.saes.sae.get_conversion_loader_name",
            return_value="sae_lens",
        ),
        patch(
            "sae_lens.saes.sae.get_repo_id_and_folder_name",
            return_value=("mock/repo", "mock_folder"),
        ),
        patch(
            "sae_lens.saes.sae.get_config_overrides",
            return_value={},
        ),
        patch(
            "sae_lens.saes.sae.get_norm_scaling_factor",
            return_value=None,
        ),
    ):
        loaded_sae = SAE.from_pretrained(
            "mock-release", "mock-sae-id", device="cpu", dtype="float32"
        )

    assert meta_device_verified, "SAE should use meta device before load_state_dict"
    assert used_assign_true, "load_state_dict should be called with assign=True"
    assert loaded_sae.W_enc.device.type == "cpu"


def test_from_pretrained_folds_norm_scaling_factor(tmp_path: Path):
    cfg = build_sae_training_cfg_for_arch("standard", d_in=256, d_sae=512)
    sae = TrainingSAE.from_dict(cfg.to_dict())
    random_params(sae)
    sae.save_model(tmp_path)

    cfg_dict, state_dict = sae_lens_disk_loader(tmp_path, device="cpu")
    cfg_dict["normalize_activations"] = "expected_average_only_in"

    # Save original weights for comparison
    original_W_enc = state_dict["W_enc"].clone()
    original_W_dec = state_dict["W_dec"].clone()
    original_b_dec = state_dict["b_dec"].clone()

    scaling_factor = 2.0

    def mock_loader(
        repo_id: str,  # noqa: ARG001
        folder_name: str,  # noqa: ARG001
        device: str = "cpu",  # noqa: ARG001
        force_download: bool = False,  # noqa: ARG001
        cfg_overrides: dict[str, Any] | None = None,  # noqa: ARG001
    ):
        return cfg_dict, state_dict, None

    mock_sae_info = type(
        "MockSAEInfo", (), {"conversion_func": None, "saes_map": {"mock-sae-id": {}}}
    )()

    with (
        patch(
            "sae_lens.saes.sae.NAMED_PRETRAINED_SAE_LOADERS",
            {"sae_lens": mock_loader},
        ),
        patch(
            "sae_lens.saes.sae.get_pretrained_saes_directory",
            return_value={"mock-release": mock_sae_info},
        ),
        patch(
            "sae_lens.saes.sae.get_conversion_loader_name",
            return_value="sae_lens",
        ),
        patch(
            "sae_lens.saes.sae.get_repo_id_and_folder_name",
            return_value=("mock/repo", "mock_folder"),
        ),
        patch(
            "sae_lens.saes.sae.get_config_overrides",
            return_value={},
        ),
        patch(
            "sae_lens.saes.sae.get_norm_scaling_factor",
            return_value=scaling_factor,
        ),
    ):
        loaded_sae = SAE.from_pretrained(
            "mock-release", "mock-sae-id", device="cpu", dtype="float32"
        )

    assert_close(loaded_sae.W_enc, original_W_enc * scaling_factor)
    assert_close(loaded_sae.W_dec, original_W_dec / scaling_factor)
    assert_close(loaded_sae.b_dec, original_b_dec / scaling_factor)
    assert loaded_sae.cfg.normalize_activations == "none"
