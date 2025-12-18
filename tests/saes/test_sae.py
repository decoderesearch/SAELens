import copy
import pickle
from pathlib import Path

import pytest
import torch

from sae_lens import __version__
from sae_lens.registry import get_sae_class, get_sae_training_class
from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    SAEMetadata,
    TrainingSAE,
    TrainingSAEConfig,
)
from tests.helpers import (
    ALL_TRAINING_ARCHITECTURES,
    assert_close,
    build_sae_training_cfg_for_arch,
    random_params,
)


@pytest.mark.parametrize("architecture", ALL_TRAINING_ARCHITECTURES)
def test_TrainingSAEConfig_to_and_from_dict_all_architectures(architecture: str):
    cfg = build_sae_training_cfg_for_arch(architecture=architecture)
    reloaded_cfg = TrainingSAEConfig.from_dict(cfg.to_dict())
    assert reloaded_cfg.to_dict() == cfg.to_dict()
    assert reloaded_cfg.__class__ == cfg.__class__
    assert reloaded_cfg.__class__ == get_sae_training_class(architecture)[1]


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
