from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from huggingface_hub import hf_hub_download as real_hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import save_file
from sparsify import SparseCoder, SparseCoderConfig

from sae_lens import StandardSAE, StandardSAEConfig
from sae_lens.loading.pretrained_sae_loaders import (
    _infer_gemma_3_raw_cfg_dict,
    dictionary_learning_sae_huggingface_loader_1,
    gemma_2_transcoder_huggingface_loader,
    gemma_3_sae_huggingface_loader,
    get_deepseek_r1_config_from_hf,
    get_dictionary_learning_config_1_from_hf,
    get_gemma_2_transcoder_config_from_hf,
    get_gemma_3_config_from_hf,
    get_goodfire_config_from_hf,
    get_goodfire_huggingface_loader,
    get_llama_scope_config_from_hf,
    get_llama_scope_r1_distill_config_from_hf,
    get_mntss_clt_layer_config_from_hf,
    get_mwhanna_transcoder_config_from_hf,
    load_sae_config_from_huggingface,
    mntss_clt_layer_huggingface_loader,
    read_sae_components_from_disk,
    sparsify_disk_loader,
    sparsify_huggingface_loader,
)
from sae_lens.saes.sae import SAE
from tests.helpers import assert_close, random_params


def test_load_sae_config_from_huggingface():
    cfg_dict = load_sae_config_from_huggingface(
        "gpt2-small-res-jb",
        sae_id="blocks.0.hook_resid_pre",
    )

    expected_cfg_dict = {
        "d_in": 768,
        "device": "cpu",
        "dtype": "torch.float32",
        "d_sae": 24576,
        "apply_b_dec_to_input": True,
        "normalize_activations": "none",
        "reshape_activations": "none",
        "metadata": {
            "model_name": "gpt2-small",
            "hook_name": "blocks.0.hook_resid_pre",
            "hook_head_index": None,
            "context_size": 128,
            "model_from_pretrained_kwargs": {"center_writing_weights": True},
            "neuronpedia_id": "gpt2-small/0-res-jb",
            "prepend_bos": True,
            "dataset_path": "Skylion007/openwebtext",
            "sae_lens_training_version": None,
        },
        "architecture": "standard",
    }

    assert cfg_dict == expected_cfg_dict


def test_load_sae_config_from_huggingface_connor_rob_hook_z():
    cfg_dict = load_sae_config_from_huggingface(
        "gpt2-small-hook-z-kk",
        sae_id="blocks.0.hook_z",
    )

    expected_cfg_dict = {
        "d_in": 768,
        "d_sae": 24576,
        "dtype": "float32",
        "device": "cpu",
        "apply_b_dec_to_input": True,
        "normalize_activations": "none",
        "reshape_activations": "hook_z",
        "metadata": {
            "model_name": "gpt2-small",
            "hook_name": "blocks.0.attn.hook_z",
            "hook_head_index": None,
            "prepend_bos": True,
            "dataset_path": "Skylion007/openwebtext",
            "context_size": 128,
            "neuronpedia_id": "gpt2-small/0-att-kk",
            "sae_lens_training_version": None,
        },
        "architecture": "standard",
    }

    assert cfg_dict == expected_cfg_dict


def test_load_old_topk_saes_from_huggingface():
    cfg_dict = load_sae_config_from_huggingface(
        "gpt2-small-resid-post-v5-32k",
        sae_id="blocks.11.hook_resid_post",
    )

    expected_cfg_dict = {
        "d_in": 768,
        "d_sae": 32768,
        "dtype": "torch.float32",
        "device": "cpu",
        "apply_b_dec_to_input": True,
        "normalize_activations": "layer_norm",
        "reshape_activations": "none",
        "metadata": {
            "model_name": "gpt2-small",
            "hook_name": "blocks.11.hook_resid_post",
            "hook_head_index": None,
            "sae_lens_training_version": None,
            "prepend_bos": False,
            "dataset_path": "Skylion007/openwebtext",
            "context_size": 64,
            "neuronpedia_id": "gpt2-small/11-res_post_32k-oai",
        },
        "architecture": "topk",
        "k": 32,
    }

    assert cfg_dict == expected_cfg_dict


def test_load_sae_config_from_huggingface_gemma_2():
    cfg_dict = load_sae_config_from_huggingface(
        "gemma-scope-2b-pt-res",
        sae_id="embedding/width_4k/average_l0_6",
    )

    expected_cfg_dict = {
        "d_in": 2304,
        "d_sae": 4096,
        "dtype": "float32",
        "apply_b_dec_to_input": False,
        "normalize_activations": "none",
        "reshape_activations": "none",
        "device": "cpu",
        "metadata": {
            "model_name": "gemma-2-2b",
            "hook_name": "hook_embed",
            "hook_head_index": None,
            "prepend_bos": True,
            "dataset_path": "monology/pile-uncopyrighted",
            "context_size": 1024,
            "neuronpedia_id": None,
            "sae_lens_training_version": None,
        },
        "architecture": "jumprelu",
    }

    assert cfg_dict == expected_cfg_dict


@pytest.mark.parametrize(
    (
        "folder_name",
        "architecture",
        "hooks",
        "d_sae",
        "d_in",
        "d_out",
    ),
    [
        (
            "resid_post_all/layer_10_width_262k_l0_small",
            "jumprelu",
            {
                "hook_name": "blocks.10.hook_resid_post",
                "hf_hook_name": "model.layers.10.output",
            },
            262144,
            1152,
            None,
        ),
        (
            "transcoder_all/layer_10_width_262k_l0_small_affine",
            "jumprelu_skip_transcoder",
            {
                "hook_name": "blocks.10.ln2.hook_normalized",
                "hook_name_out": "blocks.10.hook_mlp_out",
                "hf_hook_name": "model.layers.10.pre_feedforward_layernorm.output",
                "hf_hook_name_out": "model.layers.10.post_feedforward_layernorm.output",
            },
            262144,
            1152,
            1152,
        ),
        (
            "attn_out_all/layer_11_width_16k_l0_small",
            "jumprelu",
            {
                "hook_name": "blocks.11.hook_attn_out",
                "hf_hook_name": "model.layers.11.self_attn.o_proj.input",
            },
            16384,
            1024,
            None,
        ),
    ],
)
def test_get_gemma_3_config_from_hf(
    folder_name: str,
    architecture: str,
    hooks: dict[str, str],
    d_sae: int,
    d_in: int,
    d_out: int | None,
):
    cfg_dict = get_gemma_3_config_from_hf(
        "google/gemma-scope-2-1b-pt", folder_name, "cpu"
    )

    expected_cfg_dict = {
        "architecture": architecture,
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "model_name": "google/gemma-3-1b-pt",
        "hook_head_index": None,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "apply_b_dec_to_input": False,
        "normalize_activations": None,
        "device": "cpu",
        **hooks,
    }
    if d_out is not None:
        expected_cfg_dict["d_out"] = d_out
    assert cfg_dict == expected_cfg_dict


@pytest.mark.parametrize(
    ("repo_id", "folder_name", "expected_cfg"),
    [
        (
            "google/gemma-scope-2-1b-pt",
            "resid_post_all/layer_10_width_262k_l0_small",
            {
                "architecture": "jump_relu",
                "model_name": "google/gemma-3-1b-pt",
                "hf_hook_point_in": "model.layers.10.output",
            },
        ),
        (
            "google/gemma-scope-2-4b-it",
            "transcoder_all/layer_5_width_16k_l0_big_affine",
            {
                "architecture": "jump_relu",
                "model_name": "google/gemma-3-4b-it",
                "hf_hook_point_in": "model.layers.5.pre_feedforward_layernorm.output",
                "hf_hook_point_out": "model.layers.5.post_feedforward_layernorm.output",
            },
        ),
        (
            "google/gemma-scope-2-12b-pt",
            "attn_out_all/layer_11_width_16k_l0_small",
            {
                "architecture": "jump_relu",
                "model_name": "google/gemma-3-12b-pt",
                "hf_hook_point_in": "model.layers.11.self_attn.o_proj.input",
            },
        ),
        (
            "google/gemma-scope-2-27b-it",
            "mlp_out/layer_17_width_16k_l0_big",
            {
                "architecture": "jump_relu",
                "model_name": "google/gemma-3-27b-it",
                "hf_hook_point_in": "model.layers.17.post_feedforward_layernorm.output",
            },
        ),
        (
            "google/gemma-scope-2-270m-pt",
            "clt/layer_3_width_16k_l0_medium",
            {
                "architecture": "jump_relu",
                "model_name": "google/gemma-3-270m-pt",
                "hf_hook_point_in": "model.layers.3.pre_feedforward_layernorm.output",
                "hf_hook_point_out": "model.layers.3.post_feedforward_layernorm.output",
            },
        ),
    ],
)
def test_infer_gemma_3_raw_cfg_dict(
    repo_id: str,
    folder_name: str,
    expected_cfg: dict[str, str],
):
    cfg = _infer_gemma_3_raw_cfg_dict(repo_id, folder_name)
    assert cfg == expected_cfg


def test_infer_gemma_3_raw_cfg_dict_invalid_folder_name():
    with pytest.raises(
        ValueError, match="Could not extract layer number from folder_name"
    ):
        _infer_gemma_3_raw_cfg_dict("google/gemma-scope-2-1b-pt", "invalid_folder_name")


def test_infer_gemma_3_raw_cfg_dict_unknown_hook_type():
    with pytest.raises(ValueError, match="Could not infer hook type from folder_name"):
        _infer_gemma_3_raw_cfg_dict(
            "google/gemma-scope-2-1b-pt", "unknown_hook/layer_5_width_16k"
        )


def test_gemma_3_sae_huggingface_loader_with_mocked_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo_id = "google/gemma-scope-2-1b-pt"
    folder_name = "resid_post_all/layer_10_width_262k_l0_small"
    device = "cpu"

    # These dimensions match the real SAE
    d_in = 1152
    d_sae = 262144

    # Create mock weights matching the expected safetensors format
    W_enc = torch.randn(d_in, d_sae, dtype=torch.float32)
    W_dec = torch.randn(d_sae, d_in, dtype=torch.float32)
    b_enc = torch.randn(d_sae, dtype=torch.float32)
    b_dec = torch.randn(d_in, dtype=torch.float32)
    threshold = torch.randn(d_sae, dtype=torch.float32)

    raw_state_dict = {
        "w_enc": W_enc,
        "w_dec": W_dec,
        "b_enc": b_enc,
        "b_dec": b_dec,
        "threshold": threshold,
    }

    # Save mock weights as safetensors file
    safetensors_path = tmp_path / "params.safetensors"
    save_file(raw_state_dict, safetensors_path)

    def mock_hf_hub_download(
        repo_id: str,  # noqa: ARG001
        filename: str,
        subfolder: str | None = None,  # noqa: ARG001
        force_download: bool = False,  # noqa: ARG001
        revision: str | None = None,  # noqa: ARG001
    ) -> str:
        # config.json doesn't exist for this SAE, causing fallback to _infer_gemma_3_raw_cfg_dict
        if filename.endswith("config.json"):
            raise EntryNotFoundError("config.json not found")
        return str(safetensors_path)

    def mock_get_safetensors_tensor_shapes(
        repo_id: str,  # noqa: ARG001
        filename: str,  # noqa: ARG001
    ) -> dict[str, list[int]]:
        return {
            "w_enc": [d_in, d_sae],
            "w_dec": [d_sae, d_in],
            "b_enc": [d_sae],
            "b_dec": [d_in],
            "threshold": [d_sae],
        }

    monkeypatch.setattr(
        "sae_lens.loading.pretrained_sae_loaders.hf_hub_download", mock_hf_hub_download
    )
    monkeypatch.setattr(
        "sae_lens.loading.pretrained_sae_loaders.get_safetensors_tensor_shapes",
        mock_get_safetensors_tensor_shapes,
    )

    cfg_dict, state_dict, log_sparsity = gemma_3_sae_huggingface_loader(
        repo_id=repo_id,
        folder_name=folder_name,
        device=device,
        force_download=False,
        cfg_overrides=None,
    )

    # Verify config matches expected values from real SAE
    expected_cfg = {
        "architecture": "jumprelu",
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "model_name": "google/gemma-3-1b-pt",
        "hook_name": "blocks.10.hook_resid_post",
        "hook_head_index": None,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "apply_b_dec_to_input": False,
        "normalize_activations": None,
        "hf_hook_name": "model.layers.10.output",
        "device": "cpu",
    }
    assert cfg_dict == expected_cfg
    assert log_sparsity is None

    # Verify state dict structure and shapes
    assert set(state_dict.keys()) == {"W_enc", "W_dec", "b_enc", "b_dec", "threshold"}
    assert state_dict["W_enc"].shape == (d_in, d_sae)
    assert state_dict["W_dec"].shape == (d_sae, d_in)
    assert state_dict["b_enc"].shape == (d_sae,)
    assert state_dict["b_dec"].shape == (d_in,)
    assert state_dict["threshold"].shape == (d_sae,)

    # Verify values match input
    torch.testing.assert_close(state_dict["W_enc"], W_enc)
    torch.testing.assert_close(state_dict["W_dec"], W_dec)
    torch.testing.assert_close(state_dict["b_enc"], b_enc)
    torch.testing.assert_close(state_dict["b_dec"], b_dec)
    torch.testing.assert_close(state_dict["threshold"], threshold)


def test_load_sae_config_from_huggingface_gemma_2_hook_z_saes():
    cfg_dict = load_sae_config_from_huggingface(
        "gemma-scope-2b-pt-att",
        sae_id="layer_0/width_16k/average_l0_104",
    )

    expected_cfg_dict = {
        "d_in": 2048,
        "d_sae": 16384,
        "dtype": "float32",
        "apply_b_dec_to_input": False,
        "normalize_activations": "none",
        "reshape_activations": "hook_z",
        "device": "cpu",
        "metadata": {
            "model_name": "gemma-2-2b",
            "hook_name": "blocks.0.attn.hook_z",
            "hook_head_index": None,
            "prepend_bos": True,
            "dataset_path": "monology/pile-uncopyrighted",
            "context_size": 1024,
            "neuronpedia_id": None,
            "sae_lens_training_version": None,
        },
        "architecture": "jumprelu",
    }

    assert cfg_dict == expected_cfg_dict


def test_load_sae_config_from_huggingface_dictionary_learning_1():
    cfg_dict = load_sae_config_from_huggingface(
        "sae_bench_gemma-2-2b_topk_width-2pow16_date-1109",
        sae_id="blocks.12.hook_resid_post__trainer_0",
    )

    expected_cfg_dict = {
        "d_in": 2304,
        "d_sae": 65536,
        "dtype": "float32",
        "device": "cpu",
        "apply_b_dec_to_input": True,
        "normalize_activations": "none",
        "reshape_activations": "none",
        "metadata": {
            "model_name": "gemma-2-2b",
            "hook_name": "blocks.12.hook_resid_post",
            "hook_head_index": None,
            "prepend_bos": True,
            "dataset_path": "monology/pile-uncopyrighted",
            "context_size": 128,
            "neuronpedia_id": "gemma-2-2b/12-sae_bench-topk-res-65k__trainer_0_step_final",
            "sae_lens_training_version": None,
        },
        "architecture": "topk",
        "k": 20,
    }

    assert cfg_dict == expected_cfg_dict


def test_load_sae_config_from_huggingface_matches_from_pretrained():
    from_pretrained_cfg_dict = SAE.from_pretrained_with_cfg_and_sparsity(
        "gpt2-small-res-jb",
        sae_id="blocks.0.hook_resid_pre",
        device="cpu",
    )[1]
    direct_sae_cfg = load_sae_config_from_huggingface(
        "gpt2-small-res-jb",
        sae_id="blocks.0.hook_resid_pre",
        device="cpu",
    )
    # Normalize dtype format - both "float32" and "torch.float32" are equivalent
    if direct_sae_cfg.get("dtype") == "torch.float32":
        direct_sae_cfg["dtype"] = "float32"
    assert direct_sae_cfg == from_pretrained_cfg_dict


def test_get_gemma_2_transcoder_config_from_hf():
    cfg = get_gemma_2_transcoder_config_from_hf(
        repo_id="google/gemma-scope-2b-pt-transcoders",
        folder_name="layer_3/width_16k/average_l0_54",
        device="cpu",
    )

    expected_cfg = {
        "architecture": "jumprelu_transcoder",
        "d_in": 2304,
        "d_out": 2304,
        "d_sae": 16384,
        "dtype": "float32",
        "device": "cpu",
        "activation_fn": "relu",
        "normalize_activations": "none",
        "model_name": "gemma-2-2b",
        "hook_name": "blocks.3.ln2.hook_normalized",
        "hook_name_out": "blocks.3.hook_mlp_out",
        "hook_head_index": None,
        "hook_head_index_out": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "apply_b_dec_to_input": False,
    }

    assert cfg == expected_cfg


def test_gemma_2_transcoder_huggingface_loader_with_mocked_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo_id = "google/gemma-scope-2b-pt-transcoders"
    folder_name = "layer_3/width_16k/average_l0_54"
    device = "cpu"

    # These dimensions match the real config from get_gemma_2_transcoder_config_from_hf
    d_in = 2304
    d_sae = 16384

    # Create mock weights matching the expected npz format
    W_enc = np.random.randn(d_in, d_sae).astype(np.float32)
    W_dec = np.random.randn(d_sae, d_in).astype(np.float32)
    b_enc = np.random.randn(d_sae).astype(np.float32)
    b_dec = np.random.randn(d_in).astype(np.float32)
    threshold = np.random.randn(d_sae).astype(np.float32)

    # Save mock weights as npz file
    npz_file_path = tmp_path / "params.npz"
    np.savez(
        npz_file_path,
        w_enc=W_enc,
        w_dec=W_dec,
        b_enc=b_enc,
        b_dec=b_dec,
        threshold=threshold,
    )

    def mock_hf_hub_download(
        repo_id: str,  # noqa: ARG001
        filename: str,  # noqa: ARG001
        force_download: bool = False,  # noqa: ARG001
        revision: str | None = None,  # noqa: ARG001
    ) -> str:
        return str(npz_file_path)

    monkeypatch.setattr(
        "sae_lens.loading.pretrained_sae_loaders.hf_hub_download", mock_hf_hub_download
    )

    cfg_dict, state_dict, log_sparsity = gemma_2_transcoder_huggingface_loader(
        repo_id=repo_id,
        folder_name=folder_name,
        device=device,
        force_download=False,
        cfg_overrides=None,
    )

    # Verify config matches the real config from get_gemma_2_transcoder_config_from_hf
    expected_cfg = {
        "architecture": "jumprelu_transcoder",
        "d_in": 2304,
        "d_out": 2304,
        "d_sae": 16384,
        "dtype": "float32",
        "device": "cpu",
        "activation_fn": "relu",
        "normalize_activations": "none",
        "model_name": "gemma-2-2b",
        "hook_name": "blocks.3.ln2.hook_normalized",
        "hook_name_out": "blocks.3.hook_mlp_out",
        "hook_head_index": None,
        "hook_head_index_out": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "apply_b_dec_to_input": False,
    }
    assert cfg_dict == expected_cfg
    assert log_sparsity is None

    # Verify state dict structure and shapes
    assert set(state_dict.keys()) == {"W_enc", "W_dec", "b_enc", "b_dec", "threshold"}
    assert state_dict["W_enc"].shape == (d_in, d_sae)
    assert state_dict["W_dec"].shape == (d_sae, d_in)
    assert state_dict["b_enc"].shape == (d_sae,)
    assert state_dict["b_dec"].shape == (d_in,)
    assert state_dict["threshold"].shape == (d_sae,)

    # Verify values match input (converted to torch tensors)
    torch.testing.assert_close(
        state_dict["W_enc"], torch.tensor(W_enc, dtype=torch.float32)
    )
    torch.testing.assert_close(
        state_dict["W_dec"], torch.tensor(W_dec, dtype=torch.float32)
    )
    torch.testing.assert_close(
        state_dict["b_enc"], torch.tensor(b_enc, dtype=torch.float32)
    )
    torch.testing.assert_close(
        state_dict["b_dec"], torch.tensor(b_dec, dtype=torch.float32)
    )
    torch.testing.assert_close(
        state_dict["threshold"], torch.tensor(threshold, dtype=torch.float32)
    )


def test_get_mntss_clt_layer_config_from_hf():
    cfg = get_mntss_clt_layer_config_from_hf(
        repo_id="mntss/clt-gemma-2-2b-426k",
        folder_name="0",
        device="cpu",
    )
    expected_cfg = {
        "architecture": "transcoder",
        "d_in": 2304,
        "d_out": 2304,
        "d_sae": 16384,
        "dtype": "float32",
        "device": "cpu",
        "activation_fn": "relu",
        "normalize_activations": "none",
        "model_name": "google/gemma-2-2b",
        "hook_name": "blocks.0.hook_resid_mid",
        "hook_name_out": "blocks.0.hook_mlp_out",
        "apply_b_dec_to_input": False,
        "model_from_pretrained_kwargs": {"fold_ln": False},
    }

    assert cfg == expected_cfg


def test_get_mwhanna_transcoder_config_from_hf():
    cfg = get_mwhanna_transcoder_config_from_hf(
        repo_id="mwhanna/qwen3-4b-transcoders",
        folder_name="layer_10.safetensors",
        device="cpu",
    )

    expected_cfg = {
        "architecture": "transcoder",
        "d_in": 2560,
        "d_out": 2560,
        "d_sae": 163840,
        "dtype": "float32",
        "device": "cpu",
        "activation_fn": "relu",
        "normalize_activations": "none",
        "model_name": "Qwen/Qwen3-4B",
        "hook_name": "blocks.10.mlp.hook_in",
        "hook_name_out": "blocks.10.hook_mlp_out",
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 8192,
        "model_from_pretrained_kwargs": {"fold_ln": False},
        "apply_b_dec_to_input": False,
    }

    assert cfg == expected_cfg


def test_get_mwhanna_transcoder_config_8b_from_hf():
    cfg = get_mwhanna_transcoder_config_from_hf(
        repo_id="mwhanna/qwen3-8b-transcoders",
        folder_name="layer_10.safetensors",
        device="cpu",
    )

    expected_cfg = {
        "architecture": "transcoder",
        "d_in": 4096,
        "d_out": 4096,
        "d_sae": 163840,
        "dtype": "float32",
        "device": "cpu",
        "activation_fn": "relu",
        "normalize_activations": "none",
        "model_name": "Qwen/Qwen3-8B",
        "hook_name": "blocks.10.mlp.hook_in",
        "hook_name_out": "blocks.10.hook_mlp_out",
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 8192,
        "model_from_pretrained_kwargs": {"fold_ln": False},
        "apply_b_dec_to_input": False,
    }

    assert cfg == expected_cfg


def test_get_mwhanna_transcoder_config_14b_from_hf():
    cfg = get_mwhanna_transcoder_config_from_hf(
        repo_id="mwhanna/qwen3-14b-transcoders",
        folder_name="layer_10.safetensors",
        device="cpu",
    )

    expected_cfg = {
        "architecture": "transcoder",
        "d_in": 5120,
        "d_out": 5120,
        "d_sae": 163840,
        "dtype": "float32",
        "device": "cpu",
        "activation_fn": "relu",
        "normalize_activations": "none",
        "model_name": "Qwen/Qwen3-14B",
        "hook_name": "blocks.10.mlp.hook_in",
        "hook_name_out": "blocks.10.hook_mlp_out",
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 8192,
        "model_from_pretrained_kwargs": {"fold_ln": False},
        "apply_b_dec_to_input": False,
    }

    assert cfg == expected_cfg


def test_load_sae_config_from_huggingface_gemma_2_transcoder():
    cfg = load_sae_config_from_huggingface(
        release="gemma-scope-2b-pt-transcoders",
        sae_id="layer_3/width_16k/average_l0_54",
        device="cpu",
    )

    expected_cfg = {
        "d_in": 2304,
        "d_out": 2304,
        "d_sae": 16384,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
        "apply_b_dec_to_input": False,
        "reshape_activations": "none",
        "metadata": {
            "model_name": "gemma-2-2b",
            "hook_name": "blocks.3.ln2.hook_normalized",
            "hook_name_out": "blocks.3.hook_mlp_out",
            "hook_head_index": None,
            "hook_head_index_out": None,
            "prepend_bos": True,
            "dataset_path": "monology/pile-uncopyrighted",
            "context_size": 1024,
            "neuronpedia_id": "gemma-2-2b/3-gemmascope-transcoder-16k",
            "sae_lens_training_version": None,
        },
        "architecture": "jumprelu_transcoder",
    }

    assert cfg == expected_cfg


def test_load_sae_config_from_huggingface_mwhanna_transcoder():
    cfg = load_sae_config_from_huggingface(
        release="mwhanna-qwen3-4b-transcoders",
        sae_id="layer_10",
        device="cpu",
    )

    expected_cfg = {
        "d_in": 2560,
        "d_out": 2560,
        "d_sae": 163840,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
        "apply_b_dec_to_input": False,
        "reshape_activations": "none",
        "metadata": {
            "model_name": "Qwen/Qwen3-4B",
            "hook_name": "blocks.10.mlp.hook_in",
            "hook_name_out": "blocks.10.hook_mlp_out",
            "dataset_path": "monology/pile-uncopyrighted",
            "context_size": 8192,
            "model_from_pretrained_kwargs": {"fold_ln": False},
            "neuronpedia_id": "qwen3-4b/10-transcoder-hp",
            "prepend_bos": True,
            "sae_lens_training_version": None,
        },
        "architecture": "transcoder",
    }

    assert cfg == expected_cfg


def test_get_deepseek_r1_config_from_hf():
    """Test that the DeepSeek R1 config is generated correctly."""
    cfg = get_deepseek_r1_config_from_hf(
        repo_id="some/repo",
        folder_name="DeepSeek-R1-Distill-Llama-8B-SAE-l19.pt",
        device="cpu",
    )

    expected_cfg = {
        "architecture": "standard",
        "d_in": 4096,  # LLaMA 8B hidden size
        "d_sae": 4096 * 16,  # Expansion factor 16
        "dtype": "bfloat16",
        "context_size": 1024,
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "hook_name": "blocks.19.hook_resid_post",
        "hook_head_index": None,
        "prepend_bos": True,
        "dataset_path": "lmsys/lmsys-chat-1m",
        "dataset_trust_remote_code": True,
        "sae_lens_training_version": None,
        "activation_fn": "relu",
        "normalize_activations": "none",
        "device": "cpu",
        "apply_b_dec_to_input": False,
        "finetuning_scaling_factor": False,
    }

    assert cfg == expected_cfg


def test_get_deepseek_r1_config_with_invalid_layer():
    """Test that get_deepseek_r1_config raises ValueError with invalid layer in filename."""
    with pytest.raises(
        ValueError, match="Could not find layer number in filename: invalid_filename.pt"
    ):
        get_deepseek_r1_config_from_hf(
            repo_id="some/repo", folder_name="invalid_filename.pt", device="cpu"
        )


def test_get_llama_scope_config_from_hf():
    cfg = get_llama_scope_config_from_hf(
        repo_id="fnlp/Llama3_1-8B-Base-LXA-32x",
        folder_name="Llama3_1-8B-Base-L0A-32x",
        device="cpu",
        force_download=False,
        cfg_overrides=None,
    )

    expected_cfg = {
        "architecture": "jumprelu",
        "d_in": 4096,
        "d_sae": 4096 * 32,
        "dtype": "bfloat16",
        "device": "cpu",
        "model_name": "meta-llama/Llama-3.1-8B",
        "hook_name": "blocks.0.hook_attn_out",
        "jump_relu_threshold": 1.0616438356164384,
        "hook_head_index": None,
        "activation_fn": "relu",
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "cerebras/SlimPajama-627B",
        "context_size": 1024,
        "dataset_trust_remote_code": True,
        "apply_b_dec_to_input": False,
        "normalize_activations": "expected_average_only_in",
    }

    assert cfg == expected_cfg


def test_get_goodfire_config_from_hf():
    cfg = get_goodfire_config_from_hf(
        repo_id="Goodfire/Llama-3.3-70B-Instruct-SAE-l50",
        folder_name="Llama-3.3-70B-Instruct-SAE-l50.pt",
        device="cpu",
    )
    expected_cfg = {
        "architecture": "standard",
        "d_in": 8192,
        "d_sae": 65536,
        "model_name": "meta-llama/Llama-3.3-70B-Instruct",
        "hook_name": "blocks.50.hook_resid_post",
        "hook_head_index": None,
        "dataset_path": "lmsys/lmsys-chat-1m",
        "apply_b_dec_to_input": False,
        "device": "cpu",
    }
    assert cfg == expected_cfg


def test_get_goodfire_llama_8b_config_from_hf():
    cfg = get_goodfire_config_from_hf(
        repo_id="Goodfire/Llama-3.1-8B-Instruct-SAE-l19",
        folder_name="Llama-3.1-8B-Instruct-SAE-l19.pth",
        device="cpu",
    )
    expected_cfg = {
        "architecture": "standard",
        "d_in": 4096,
        "d_sae": 65536,
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "hook_name": "blocks.19.hook_resid_post",
        "hook_head_index": None,
        "dataset_path": "lmsys/lmsys-chat-1m",
        "apply_b_dec_to_input": False,
        "device": "cpu",
    }
    assert cfg == expected_cfg


def test_get_goodfire_config_from_hf_errors_on_unsupported_sae():
    with pytest.raises(
        ValueError,
        match="Unsupported Goodfire SAE: wrong/repo",
    ):
        get_goodfire_config_from_hf(
            repo_id="wrong/repo",
            folder_name="Llama-3.3-70B-Instruct-SAE-l50.pt",
            device="cpu",
        )
    with pytest.raises(
        ValueError,
        match="Unsupported Goodfire SAE: Goodfire/Llama-3.3-70B-Instruct-SAE-l50/wrong_filename.pt",
    ):
        get_goodfire_config_from_hf(
            repo_id="Goodfire/Llama-3.3-70B-Instruct-SAE-l50",
            folder_name="wrong_filename.pt",
            device="cpu",
        )


def test_our_sae_matches_goodfires_implementation():
    # from https://colab.research.google.com/drive/1IBMQtJqy8JiRk1Q48jDEgTISmtxhlCRL
    class GoodfireSAE(torch.nn.Module):
        def __init__(
            self,
            d_in: int,
            d_hidden: int,
            device: torch.device,
            dtype: torch.dtype = torch.float32,
        ):
            super().__init__()
            self.d_in = d_in
            self.d_hidden = d_hidden
            self.device = device
            self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
            self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
            self.dtype = dtype
            self.to(self.device, self.dtype)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """Encode a batch of data using a linear, followed by a ReLU."""
            return torch.nn.functional.relu(self.encoder_linear(x))

        def decode(self, x: torch.Tensor) -> torch.Tensor:
            """Decode a batch of data using a linear."""
            return self.decoder_linear(x)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """SAE forward pass. Returns the reconstruction and the encoded features."""
            f = self.encode(x)
            return self.decode(f), f

    cfg_dict = load_sae_config_from_huggingface(
        release="goodfire-llama-3.3-70b-instruct",
        sae_id="layer_50",
        device="cpu",
    )
    cfg_dict["d_in"] = 128
    cfg_dict["d_sae"] = 256
    cfg_dict["dtype"] = "float32"

    assert cfg_dict["architecture"] == "standard"
    cfg = StandardSAEConfig.from_dict(cfg_dict)

    # make a SAE base on the Goodfire config, but smaller since the real SAE class is huge
    sae = StandardSAE(cfg)
    random_params(sae)

    sae_state_dict = sae.state_dict()
    goodfire_state_dict = {
        "encoder_linear.weight": sae_state_dict["W_enc"].T,
        "encoder_linear.bias": sae_state_dict["b_enc"],
        "decoder_linear.weight": sae_state_dict["W_dec"].T,
        "decoder_linear.bias": sae_state_dict["b_dec"],
    }

    goodfire_sae = GoodfireSAE(d_in=128, d_hidden=256, device=torch.device("cpu"))
    goodfire_sae.load_state_dict(goodfire_state_dict)

    test_input = torch.randn(10, 128)

    output = sae(test_input)
    features = sae.encode(test_input)
    goodfire_output, goodfire_features = goodfire_sae(test_input)

    assert_close(output, goodfire_output, rtol=1e-4, atol=1e-4)
    assert_close(features, goodfire_features, rtol=1e-4, atol=1e-4)


def test_get_goodfire_huggingface_loader_with_mocked_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo_id = "Goodfire/Llama-3.3-70B-Instruct-SAE-l50"
    folder_name = "Llama-3.3-70B-Instruct-SAE-l50.pt"
    device = "cpu"

    d_in = 128
    d_sae = 256

    encoder_weight = torch.randn(d_sae, d_in)
    decoder_weight = torch.randn(d_in, d_sae)
    encoder_bias = torch.randn(d_sae)
    decoder_bias = torch.randn(d_in)

    raw_state_dict = {
        "encoder_linear.weight": encoder_weight,
        "decoder_linear.weight": decoder_weight,
        "encoder_linear.bias": encoder_bias,
        "decoder_linear.bias": decoder_bias,
    }

    sae_file_path = tmp_path / folder_name
    torch.save(raw_state_dict, sae_file_path)

    def mock_get_goodfire_config_from_hf(
        repo_id: str,  # noqa: ARG001
        folder_name: str,  # noqa: ARG001
        device: str,
        force_download: bool = False,  # noqa: ARG001
        cfg_overrides: dict[str, Any] | None = None,  # noqa: ARG001
    ) -> dict[str, Any]:
        return {
            "architecture": "standard",
            "d_in": d_in,
            "d_sae": d_sae,
            "model_name": "meta-llama/Llama-3.3-70B-Instruct",
            "hook_name": "blocks.50.hook_resid_post",
            "hook_head_index": None,
            "dataset_path": "lmsys/lmsys-chat-1m",
            "apply_b_dec_to_input": False,
            "device": device,
        }

    def mock_hf_hub_download(
        repo_id: str,  # noqa: ARG001
        filename: str,  # noqa: ARG001
        force_download: bool = False,  # noqa: ARG001
    ) -> str:
        return str(sae_file_path)

    monkeypatch.setattr(
        "sae_lens.loading.pretrained_sae_loaders.get_goodfire_config_from_hf",
        mock_get_goodfire_config_from_hf,
    )
    monkeypatch.setattr(
        "sae_lens.loading.pretrained_sae_loaders.hf_hub_download", mock_hf_hub_download
    )

    cfg_dict, state_dict, log_sparsity = get_goodfire_huggingface_loader(
        repo_id=repo_id,
        folder_name=folder_name,
        device=device,
        force_download=False,
        cfg_overrides=None,
    )

    expected_cfg = {
        "architecture": "standard",
        "d_in": d_in,
        "d_sae": d_sae,
        "model_name": "meta-llama/Llama-3.3-70B-Instruct",
        "hook_name": "blocks.50.hook_resid_post",
        "hook_head_index": None,
        "dataset_path": "lmsys/lmsys-chat-1m",
        "apply_b_dec_to_input": False,
        "device": device,
    }

    assert cfg_dict == expected_cfg
    assert log_sparsity is None

    assert set(state_dict.keys()) == {"W_enc", "W_dec", "b_enc", "b_dec"}
    torch.testing.assert_close(state_dict["W_enc"], encoder_weight.T)
    torch.testing.assert_close(state_dict["W_dec"], decoder_weight.T)
    torch.testing.assert_close(state_dict["b_enc"], encoder_bias)
    torch.testing.assert_close(state_dict["b_dec"], decoder_bias)

    assert state_dict["W_enc"].shape == (d_in, d_sae)
    assert state_dict["W_dec"].shape == (d_sae, d_in)
    assert state_dict["b_enc"].shape == (d_sae,)
    assert state_dict["b_dec"].shape == (d_in,)


def test_get_llama_scope_r1_distill_config_from_hf():
    """Test that the Llama Scope R1 Distill config is generated correctly."""
    cfg = get_llama_scope_r1_distill_config_from_hf(
        repo_id="fnlp/Llama-Scope-R1-Distill",
        folder_name="800M-Slimpajama-0-OpenR1-Math-220k/L5R",
        device="cpu",
        force_download=False,
        cfg_overrides=None,
    )

    expected_cfg = {
        "architecture": "jumprelu",
        "d_in": 4096,  # LLaMA 8B hidden size
        "d_sae": 4096 * 8,  # Expansion factor
        "dtype": "float32",
        "device": "cpu",
        "model_name": "meta-llama/Llama-3.1-8B",
        "hook_name": "blocks.5.hook_resid_post",
        "hook_head_index": None,
        "activation_fn": "relu",
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "cerebras/SlimPajama-627B",
        "context_size": 1024,
        "dataset_trust_remote_code": True,
        "apply_b_dec_to_input": False,
        "normalize_activations": "expected_average_only_in",
    }

    assert cfg == expected_cfg


def test_get_llama_scope_r1_distill_config_with_overrides():
    """Test that config overrides work correctly for Llama Scope R1 Distill."""
    cfg_overrides = {
        "device": "cuda",
        "dtype": "float16",
        "d_sae": 8192,
    }

    cfg = get_llama_scope_r1_distill_config_from_hf(
        repo_id="fnlp/Llama-Scope-R1-Distill",
        folder_name="400M-Slimpajama-400M-OpenR1-Math-220k/L10R",
        device="cuda",
        cfg_overrides=cfg_overrides,
    )

    assert cfg["device"] == "cuda"
    assert cfg["dtype"] == "float16"
    assert cfg["d_sae"] == 8192


def test_sparsify_huggingface_loader():
    repo = "EleutherAI/sae-pythia-70m-32k"
    hookpoint = "layers.1"
    # Need to hackily load the SAE in float32 since sparsify doesn't handle dtypes correctly
    sparsify_sae = SparseCoder.load_from_hub(repo, device="cpu", hookpoint=hookpoint)

    cfg_dict, state_dict, _ = sparsify_huggingface_loader(
        "EleutherAI/sae-pythia-70m-32k", folder_name="layers.1"
    )

    assert cfg_dict["d_in"] == sparsify_sae.d_in
    assert cfg_dict["d_sae"] == sparsify_sae.num_latents
    assert cfg_dict["activation_fn_str"] == sparsify_sae.cfg.activation
    assert cfg_dict["activation_fn_kwargs"]["k"] == sparsify_sae.cfg.k

    torch.testing.assert_close(
        state_dict["W_enc"], sparsify_sae.encoder.weight.data.T, check_dtype=False
    )
    torch.testing.assert_close(
        state_dict["b_enc"], sparsify_sae.encoder.bias.data, check_dtype=False
    )
    # sparsify_sae.W_dec is Optional in the type stubs, so first assert it's present
    assert sparsify_sae.W_dec is not None
    torch.testing.assert_close(
        state_dict["W_dec"], sparsify_sae.W_dec.detach().T, check_dtype=False
    )
    torch.testing.assert_close(
        state_dict["b_dec"], sparsify_sae.b_dec.data, check_dtype=False
    )


def test_sparsify_disk_loader(tmp_path: Path):
    d_in = 5
    cfg = SparseCoderConfig(
        expansion_factor=3,
        num_latents=d_in * 3,
        k=2,
        normalize_decoder=False,
    )
    sparsify_sae = SparseCoder(d_in, cfg=cfg, dtype=torch.bfloat16)
    path = tmp_path / "layers.0"
    sparsify_sae.save_to_disk(path)

    cfg_dict, state_dict = sparsify_disk_loader(path)

    assert cfg_dict["d_in"] == sparsify_sae.d_in
    assert cfg_dict["d_sae"] == sparsify_sae.num_latents
    assert cfg_dict["activation_fn_str"] == sparsify_sae.cfg.activation
    assert cfg_dict["activation_fn_kwargs"]["k"] == sparsify_sae.cfg.k

    torch.testing.assert_close(state_dict["W_enc"], sparsify_sae.encoder.weight.data.T)
    torch.testing.assert_close(state_dict["b_enc"], sparsify_sae.encoder.bias.data)
    # sparsify_sae.W_dec is Optional in the type stubs, so first assert it's present
    assert sparsify_sae.W_dec is not None
    torch.testing.assert_close(state_dict["W_dec"], sparsify_sae.W_dec.detach().T)
    torch.testing.assert_close(state_dict["b_dec"], sparsify_sae.b_dec.data)


def test_get_dictionary_learning_config_1_from_hf_andy():
    cfg_dict = get_dictionary_learning_config_1_from_hf(
        "andyrdt/saes-llama-3.1-8b-instruct",
        "resid_post_layer_3/trainer_1",
        device="cpu",
        force_download=False,
        cfg_overrides=None,
    )
    assert cfg_dict == {
        "architecture": "jumprelu",  # BatchTopKSAE uses threshold-based activation
        "d_in": 4096,
        "d_sae": 131072,
        "dtype": "float32",
        "device": "cpu",
        "model_name": "Llama-3.1-8B-Instruct",
        "hook_name": "blocks.3.hook_resid_post",
        "hook_head_index": None,
        "activation_fn": "relu",
        "activation_fn_kwargs": {},
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "normalize_activations": "none",
        "neuronpedia_id": None,
        "dataset_trust_remote_code": True,
    }


@pytest.mark.skip(
    reason="This takes too long since the files are large. Also redundant-ish with the test below."
)
def test_dictionary_learning_sae_huggingface_loader_1_andy():
    cfg_dict, state_dict, _ = dictionary_learning_sae_huggingface_loader_1(
        "andyrdt/saes-llama-3.1-8b-instruct",
        "resid_post_layer_3/trainer_1",
        device="cpu",
        force_download=False,
        cfg_overrides=None,
    )
    # BatchTopKSAE includes a threshold parameter for JumpReLU-style activation
    assert state_dict.keys() == {"W_enc", "W_dec", "b_dec", "b_enc", "threshold"}
    assert cfg_dict == {
        "architecture": "jumprelu",  # BatchTopKSAE uses threshold-based activation
        "d_in": 4096,
        "d_sae": 131072,
        "dtype": "float32",
        "device": "cpu",
        "model_name": "Llama-3.1-8B-Instruct",
        "hook_name": "blocks.3.hook_resid_post",
        "hook_head_index": None,
        "activation_fn": "relu",
        "activation_fn_kwargs": {},
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "normalize_activations": "none",
        "neuronpedia_id": None,
        "dataset_trust_remote_code": True,
    }
    assert state_dict["W_enc"].shape == (4096, 131072)
    assert state_dict["W_dec"].shape == (131072, 4096)
    assert state_dict["b_dec"].shape == (4096,)
    assert state_dict["b_enc"].shape == (131072,)
    assert state_dict["threshold"].shape == (131072,)


def test_dictionary_learning_sae_huggingface_loader_1():
    cfg_dict, state_dict, sparsity = dictionary_learning_sae_huggingface_loader_1(
        "canrager/lm_sae",
        "pythia70m_sweep_gated_ctx128_0730/resid_post_layer_3/trainer_0",
        device="cpu",
        force_download=False,
        cfg_overrides=None,
    )
    assert sparsity is None
    assert state_dict.keys() == {"W_enc", "W_dec", "b_dec", "b_mag", "b_gate", "r_mag"}
    assert cfg_dict == {
        "architecture": "gated",
        "d_in": 512,
        "d_sae": 4096,
        "dtype": "float32",
        "device": "cpu",
        "model_name": "pythia-70m-deduped",
        "hook_name": "blocks.3.hook_resid_post",
        "hook_head_index": None,
        "activation_fn": "relu",
        "activation_fn_kwargs": {},
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 128,
        "normalize_activations": "none",
        "neuronpedia_id": None,
        "dataset_trust_remote_code": True,
    }
    assert state_dict["W_enc"].shape == (512, 4096)
    assert state_dict["W_dec"].shape == (4096, 512)
    assert state_dict["b_dec"].shape == (512,)
    assert state_dict["b_mag"].shape == (4096,)
    assert state_dict["b_gate"].shape == (4096,)
    assert state_dict["r_mag"].shape == (4096,)


def test_read_sae_components_from_disk(tmp_path: Path):
    d_in = 256
    d_sae = 512
    device = "cpu"
    dtype = torch.float32

    # Create dummy SAE components
    W_enc = torch.randn(d_in, d_sae, dtype=dtype)
    W_dec = torch.randn(d_sae, d_in, dtype=dtype)
    b_enc = torch.randn(d_sae, dtype=dtype)
    b_dec = torch.randn(d_in, dtype=dtype)

    # Create state dict
    state_dict = {
        "W_enc": W_enc,
        "W_dec": W_dec,
        "b_enc": b_enc,
        "b_dec": b_dec,
    }

    # Save to disk
    weights_path = tmp_path / "sae_weights.safetensors"
    save_file(state_dict, weights_path)

    # Create config dict
    cfg_dict = {
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "device": device,
        "finetuning_scaling_factor": False,
    }

    # Read back from disk
    loaded_cfg_dict, loaded_state_dict = read_sae_components_from_disk(
        cfg_dict=cfg_dict,
        weight_path=weights_path,
        device=device,
    )

    # Check that config dict is returned unchanged (except for finetuning_scaling_factor)
    assert loaded_cfg_dict["d_in"] == d_in
    assert loaded_cfg_dict["d_sae"] == d_sae
    assert loaded_cfg_dict["dtype"] == "float32"
    assert loaded_cfg_dict["device"] == device
    assert loaded_cfg_dict["finetuning_scaling_factor"] is False

    # Check that all tensors are loaded correctly
    assert loaded_state_dict.keys() == {"W_enc", "W_dec", "b_enc", "b_dec"}
    torch.testing.assert_close(loaded_state_dict["W_enc"], W_enc)
    torch.testing.assert_close(loaded_state_dict["W_dec"], W_dec)
    torch.testing.assert_close(loaded_state_dict["b_enc"], b_enc)
    torch.testing.assert_close(loaded_state_dict["b_dec"], b_dec)

    # Check tensor shapes
    assert loaded_state_dict["W_enc"].shape == (d_in, d_sae)
    assert loaded_state_dict["W_dec"].shape == (d_sae, d_in)
    assert loaded_state_dict["b_enc"].shape == (d_sae,)
    assert loaded_state_dict["b_dec"].shape == (d_in,)

    # Check tensor dtypes
    assert loaded_state_dict["W_enc"].dtype == dtype
    assert loaded_state_dict["W_dec"].dtype == dtype
    assert loaded_state_dict["b_enc"].dtype == dtype
    assert loaded_state_dict["b_dec"].dtype == dtype


def test_read_sae_components_from_disk_with_scaling_factor(tmp_path: Path):
    d_in = 128
    d_sae = 256
    device = "cpu"
    dtype = torch.float32

    # Create dummy SAE components with scaling factor
    W_enc = torch.randn(d_in, d_sae, dtype=dtype)
    W_dec = torch.randn(d_sae, d_in, dtype=dtype)
    b_enc = torch.randn(d_sae, dtype=dtype)
    b_dec = torch.randn(d_in, dtype=dtype)
    scaling_factor = torch.tensor([1.5, 2.0, 0.8], dtype=dtype)

    # Create state dict with scaling factor
    state_dict = {
        "W_enc": W_enc,
        "W_dec": W_dec,
        "b_enc": b_enc,
        "b_dec": b_dec,
        "scaling_factor": scaling_factor,
    }

    # Save to disk
    weights_path = tmp_path / "sae_weights.safetensors"
    save_file(state_dict, weights_path)

    # Create config dict with finetuning_scaling_factor enabled
    cfg_dict = {
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "device": device,
        "finetuning_scaling_factor": True,
    }

    # Read back from disk
    loaded_cfg_dict, loaded_state_dict = read_sae_components_from_disk(
        cfg_dict=cfg_dict,
        weight_path=weights_path,
        device=device,
    )

    # Check that scaling factor is renamed to finetuning_scaling_factor
    assert "scaling_factor" not in loaded_state_dict
    assert "finetuning_scaling_factor" in loaded_state_dict
    torch.testing.assert_close(
        loaded_state_dict["finetuning_scaling_factor"], scaling_factor
    )

    # Check that config dict is returned correctly
    assert loaded_cfg_dict["d_in"] == d_in
    assert loaded_cfg_dict["d_sae"] == d_sae
    assert loaded_cfg_dict["dtype"] == "float32"
    assert loaded_cfg_dict["device"] == device
    assert loaded_cfg_dict["finetuning_scaling_factor"] is True

    # Check that other tensors are still there
    assert loaded_state_dict.keys() == {
        "W_enc",
        "W_dec",
        "b_enc",
        "b_dec",
        "finetuning_scaling_factor",
    }
    torch.testing.assert_close(loaded_state_dict["W_enc"], W_enc)
    torch.testing.assert_close(loaded_state_dict["W_dec"], W_dec)
    torch.testing.assert_close(loaded_state_dict["b_enc"], b_enc)
    torch.testing.assert_close(loaded_state_dict["b_dec"], b_dec)


def test_read_sae_components_from_disk_with_ones_scaling_factor(tmp_path: Path):
    d_in = 64
    d_sae = 128
    device = "cpu"
    dtype = torch.float32

    # Create dummy SAE components with scaling factor of all ones
    W_enc = torch.randn(d_in, d_sae, dtype=dtype)
    W_dec = torch.randn(d_sae, d_in, dtype=dtype)
    b_enc = torch.randn(d_sae, dtype=dtype)
    b_dec = torch.randn(d_in, dtype=dtype)
    scaling_factor = torch.ones(3, dtype=dtype)

    # Create state dict with scaling factor
    state_dict = {
        "W_enc": W_enc,
        "W_dec": W_dec,
        "b_enc": b_enc,
        "b_dec": b_dec,
        "scaling_factor": scaling_factor,
    }

    # Save to disk
    weights_path = tmp_path / "sae_weights.safetensors"
    save_file(state_dict, weights_path)

    # Create config dict
    cfg_dict = {
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "device": device,
        "finetuning_scaling_factor": False,
    }

    # Read back from disk
    loaded_cfg_dict, loaded_state_dict = read_sae_components_from_disk(
        cfg_dict=cfg_dict,
        weight_path=weights_path,
        device=device,
    )

    # Check that scaling factor of all ones is removed
    assert "scaling_factor" not in loaded_state_dict
    assert "finetuning_scaling_factor" not in loaded_state_dict
    assert loaded_cfg_dict["finetuning_scaling_factor"] is False

    # Check that config dict is returned correctly
    assert loaded_cfg_dict["d_in"] == d_in
    assert loaded_cfg_dict["d_sae"] == d_sae
    assert loaded_cfg_dict["dtype"] == "float32"
    assert loaded_cfg_dict["device"] == device

    # Check that other tensors are still there
    assert loaded_state_dict.keys() == {"W_enc", "W_dec", "b_enc", "b_dec"}
    torch.testing.assert_close(loaded_state_dict["W_enc"], W_enc)
    torch.testing.assert_close(loaded_state_dict["W_dec"], W_dec)
    torch.testing.assert_close(loaded_state_dict["b_enc"], b_enc)
    torch.testing.assert_close(loaded_state_dict["b_dec"], b_dec)


def test_get_mntss_clt_layer_huggingface_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test the MNTSS CLT layer loader with mocked files."""
    # Test parameters
    repo_id = "test/mntss-clt-repo"
    folder_name = "5"  # layer number
    device = "cpu"

    # Create test dimensions
    d_in = 128
    d_sae = 512

    # Create fake config.yaml
    config_data = {
        "model_name": "test-model",
        "feature_input_hook": "mlp.hook_in",
        "feature_output_hook": "hook_mlp_out",
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Create the actual tensor data that will be inside the nested structure
    W_enc_tensor = torch.randn(d_sae, d_in)  # This will be transposed
    b_enc_tensor = torch.randn(d_sae)
    b_dec_tensor = torch.randn(d_in)
    W_dec_tensor = torch.randn(d_in, 10)  # Will be summed to (d_in,)

    # Create fake encoder file with placeholder tensors (we'll mock load_file)
    encoder_tensors = {
        "placeholder": torch.tensor(0.0),
    }
    encoder_path = tmp_path / f"W_enc_{folder_name}.safetensors"
    save_file(encoder_tensors, encoder_path)

    # Create fake decoder file with placeholder tensors
    decoder_tensors = {
        "placeholder": torch.tensor(0.0),
    }
    decoder_path = tmp_path / f"W_dec_{folder_name}.safetensors"
    save_file(decoder_tensors, decoder_path)

    # Mock hf_hub_download to return our temporary files
    def mock_hf_hub_download(
        repo_id_arg: str,  # noqa: ARG001
        filename: str,
        force_download: bool = False,  # noqa: ARG001
    ) -> str:
        if filename == "config.yaml":
            return str(config_path)
        if filename == f"W_enc_{folder_name}.safetensors":
            return str(encoder_path)
        if filename == f"W_dec_{folder_name}.safetensors":
            return str(decoder_path)
        raise ValueError(f"Unexpected filename: {filename}")

    # Mock load_safetensors_weights to return the expected nested structure
    def mock_load_safetensors_weights(
        path: str | Path,  # noqa: ARG001
        device: str = "cpu",  # noqa: ARG001
        dtype: torch.dtype | str | None = None,  # noqa: ARG001
    ) -> dict[str, torch.Tensor]:
        path_str = str(path)
        if f"W_enc_{folder_name}.safetensors" in path_str:
            return {
                f"W_enc_{folder_name}": W_enc_tensor,
                f"b_enc_{folder_name}": b_enc_tensor,
                f"b_dec_{folder_name}": b_dec_tensor,
            }
        if f"W_dec_{folder_name}.safetensors" in path_str:
            return {f"W_dec_{folder_name}": W_dec_tensor}
        raise ValueError(f"Unexpected file path: {path_str}")

    # Mock get_safetensors_tensor_shapes to return expected tensor shapes
    def mock_get_safetensors_tensor_shapes(
        repo_id_arg: str,  # noqa: ARG001
        filename: str,  # noqa: ARG001
    ) -> dict[str, list[int]]:
        return {
            f"b_dec_{folder_name}": [d_in],
            f"b_enc_{folder_name}": [d_sae],
            f"W_enc_{folder_name}": [d_sae, d_in],
        }

    # Apply the mocks
    monkeypatch.setattr(
        "sae_lens.loading.pretrained_sae_loaders.hf_hub_download", mock_hf_hub_download
    )
    monkeypatch.setattr(
        "sae_lens.loading.pretrained_sae_loaders.load_safetensors_weights",
        mock_load_safetensors_weights,
    )
    monkeypatch.setattr(
        "sae_lens.loading.pretrained_sae_loaders.get_safetensors_tensor_shapes",
        mock_get_safetensors_tensor_shapes,
    )

    # Call the function
    cfg_dict, state_dict, log_sparsity = mntss_clt_layer_huggingface_loader(
        repo_id=repo_id,
        folder_name=folder_name,
        device=device,
        force_download=False,
        cfg_overrides=None,
    )

    # Verify the config
    expected_cfg = {
        "architecture": "transcoder",
        "d_in": d_in,
        "d_out": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "device": device,
        "activation_fn": "relu",
        "normalize_activations": "none",
        "model_name": "test-model",
        "hook_name": f"blocks.{folder_name}.mlp.hook_in",
        "hook_name_out": f"blocks.{folder_name}.hook_mlp_out",
        "apply_b_dec_to_input": False,
        "model_from_pretrained_kwargs": {"fold_ln": False},
    }

    assert cfg_dict == expected_cfg

    # Verify the state dict structure
    assert set(state_dict.keys()) == {"W_enc", "b_enc", "b_dec", "W_dec"}

    # Verify tensor shapes
    assert state_dict["W_enc"].shape == (d_in, d_sae)  # Transposed from original
    assert state_dict["b_enc"].shape == (d_sae,)
    assert state_dict["b_dec"].shape == (d_in,)
    assert state_dict["W_dec"].shape == (d_in,)  # Summed from (d_in, 10)

    # Verify log_sparsity is None
    assert log_sparsity is None

    # Verify the tensors match expected transformations
    torch.testing.assert_close(state_dict["W_enc"], W_enc_tensor.T)
    torch.testing.assert_close(state_dict["b_enc"], b_enc_tensor)
    torch.testing.assert_close(state_dict["b_dec"], b_dec_tensor)
    torch.testing.assert_close(state_dict["W_dec"], W_dec_tensor.sum(dim=1))


def test_TemporalSAE_config_from_pretrained():
    cfg_dict = load_sae_config_from_huggingface(
        "temporal-sae-gemma-2-2b",
        sae_id="blocks.12.hook_resid_post",
    )

    expected_cfg = {
        "d_in": 2304,
        "d_sae": 9216,
        "n_heads": 4,
        "n_attn_layers": 1,
        "bottleneck_factor": 1,
        "sae_diff_type": "topk",
        "kval_topk": 192,
        "tied_weights": True,
        "dtype": "bfloat16",
        "device": "cpu",
        "normalize_activations": "constant_scalar_rescale",
        "apply_b_dec_to_input": True,
        "reshape_activations": "none",
        "activation_normalization_factor": 0.00666666667,
        "metadata": {
            "model_name": "gemma-2-2b",
            "hook_name": "blocks.12.hook_resid_post",
            "dataset_path": "monology/pile-uncopyrighted",
            "neuronpedia_id": "gemma-2-2b/12-temporal-res",
            "prepend_bos": True,
            "sae_lens_training_version": None,
        },
        "architecture": "temporal",
    }

    assert cfg_dict == expected_cfg


def test_load_sae_config_from_huggingface_dictionary_learning_matryoshka():
    cfg_dict = load_sae_config_from_huggingface(
        "saebench_gemma-2-2b_width-2pow12_date-0108",
        sae_id="blocks.12.hook_resid_post__trainer_0",
    )

    expected_cfg_dict = {
        "d_in": 2304,
        "d_sae": 4096,
        "dtype": "float32",
        "device": "cpu",
        "apply_b_dec_to_input": True,
        "normalize_activations": "none",
        "reshape_activations": "none",
        "metadata": {
            "model_name": "gemma-2-2b",
            "hook_name": "blocks.12.hook_resid_post",
            "hook_head_index": None,
            "prepend_bos": True,
            "dataset_path": "monology/pile-uncopyrighted",
            "context_size": 1024,
            "neuronpedia_id": "gemma-2-2b/12-sae_bench-matryoshka-res-4k__trainer_0_step_final",
            "sae_lens_training_version": None,
        },
        "architecture": "jumprelu",
    }

    assert cfg_dict == expected_cfg_dict


def test_dictionary_learning_sae_huggingface_loader_1_matryoshka(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo_id = "adamkarvonen/saebench_gemma-2-2b_width-2pow12_date-0108"
    folder_name = "MatryoshkaBatchTopK_gemma-2-2b__0108/resid_post_layer_12/trainer_0"
    device = "cpu"

    D_IN = 2304
    D_SAE = 4096

    W_enc = torch.randn(D_IN, D_SAE)
    W_dec = torch.randn(D_SAE, D_IN)
    b_enc = torch.randn(D_SAE)
    b_dec = torch.randn(D_IN)
    threshold_scalar = 0.5

    raw_state_dict = {
        "W_enc": W_enc,
        "W_dec": W_dec,
        "b_enc": b_enc,
        "b_dec": b_dec,
        "threshold": torch.tensor(threshold_scalar),
    }

    sae_file_path = tmp_path / "ae.pt"
    torch.save(raw_state_dict, sae_file_path)

    def mock_hf_hub_download(
        repo_id: str,
        filename: str,
        force_download: bool = False,
    ) -> str:
        if filename.endswith("ae.pt"):
            return str(sae_file_path)
        return real_hf_hub_download(
            repo_id=repo_id, filename=filename, force_download=force_download
        )

    monkeypatch.setattr(
        "sae_lens.loading.pretrained_sae_loaders.hf_hub_download", mock_hf_hub_download
    )

    cfg_dict, state_dict, sparsity = dictionary_learning_sae_huggingface_loader_1(
        repo_id,
        folder_name,
        device=device,
        force_download=False,
        cfg_overrides=None,
    )

    assert sparsity is None
    assert state_dict.keys() == {"W_enc", "W_dec", "b_dec", "b_enc", "threshold"}
    assert cfg_dict == {
        "architecture": "jumprelu",
        "d_in": D_IN,
        "d_sae": D_SAE,
        "dtype": "float32",
        "device": device,
        "model_name": "gemma-2-2b",
        "hook_name": "blocks.12.hook_resid_post",
        "hook_head_index": None,
        "activation_fn": "relu",
        "activation_fn_kwargs": {},
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "normalize_activations": "none",
        "neuronpedia_id": None,
        "dataset_trust_remote_code": True,
    }
    torch.testing.assert_close(state_dict["W_enc"], W_enc)
    torch.testing.assert_close(state_dict["W_dec"], W_dec)
    torch.testing.assert_close(state_dict["b_dec"], b_dec)
    torch.testing.assert_close(state_dict["b_enc"], b_enc)
    assert state_dict["threshold"].shape == (D_SAE,)
    assert torch.all(state_dict["threshold"] == threshold_scalar)


def test_from_pretrained_warns_when_using_registered_repo_id_directly(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that a warning is issued when loading using a repo_id that is in the pretrained_saes.yaml."""
    D_IN = 768
    D_SAE = 24576

    # Create mock config and state dict
    mock_cfg_dict = {
        "architecture": "standard",
        "d_in": D_IN,
        "d_sae": D_SAE,
        "dtype": "float32",
        "device": "cpu",
        "apply_b_dec_to_input": True,
        "normalize_activations": "none",
        "reshape_activations": "none",
        "metadata": {},
    }
    mock_state_dict = {
        "W_enc": torch.randn(D_IN, D_SAE),
        "W_dec": torch.randn(D_SAE, D_IN),
        "b_enc": torch.zeros(D_SAE),
        "b_dec": torch.zeros(D_IN),
    }

    def mock_sae_lens_loader(
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
        return mock_cfg_dict, mock_state_dict, None

    monkeypatch.setattr(
        "sae_lens.saes.sae.NAMED_PRETRAINED_SAE_LOADERS",
        {"sae_lens": mock_sae_lens_loader},
    )
    monkeypatch.setattr(
        "sae_lens.saes.sae.get_conversion_loader_name",
        lambda _release: "sae_lens",
    )

    # Using repo_id "jbloom/GPT2-Small-SAEs-Reformatted" directly should warn
    # because it's registered in pretrained_saes.yaml under "gpt2-small-res-jb"
    with pytest.warns(
        UserWarning,
        match=r"You are loading an SAE using the HuggingFace repo_id.*jbloom/GPT2-Small-SAEs-Reformatted.*directly",
    ):
        SAE.from_pretrained(
            release="jbloom/GPT2-Small-SAEs-Reformatted",
            sae_id="blocks.0.hook_resid_pre",
            device="cpu",
        )
