# Placeholder for standard SAE tests

import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformer_lens.hook_points import HookPoint

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.saes.sae import SAE, TrainingSAE, _disable_hooks
from sae_lens.saes.standard_sae import (
    StandardSAE,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
)
from tests.helpers import (
    ALL_ARCHITECTURES,
    ALL_FOLDABLE_ARCHITECTURES,
    ALL_TRAINING_ARCHITECTURES,
    assert_close,
    assert_not_close,
    build_runner_cfg,
    build_sae_cfg,
    build_sae_cfg_for_arch,
    build_sae_training_cfg,
    build_sae_training_cfg_for_arch,
)


# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "NeelNanda/c4-10k",
            "hook_name": "blocks.1.hook_resid_pre",
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "NeelNanda/c4-10k",
            "hook_name": "blocks.1.hook_resid_pre",
            "d_in": 64,
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
            "hook_name": "blocks.1.hook_resid_pre",
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "NeelNanda/c4-10k",
            "hook_name": "blocks.1.attn.hook_z",
            "d_in": 64,
        },
    ],
    ids=[
        "tiny-stories-1M-resid-pre",
        "tiny-stories-1M-resid-pre-L1-W-dec-Norm",
        "tiny-stories-1M-resid-pre-pretokenized",
        "tiny-stories-1M-attn-out",
    ],
)
def cfg(request: pytest.FixtureRequest):
    """
    Pytest fixture to create a mock instance of LanguageModelSAERunnerConfig.
    """
    params = request.param
    return build_runner_cfg(**params)


def test_sae_init(cfg: LanguageModelSAERunnerConfig[StandardTrainingSAEConfig]):
    sae = StandardSAE(cfg.sae)  # type: ignore

    assert isinstance(sae, SAE)

    assert sae.W_enc.shape == (cfg.sae.d_in, cfg.sae.d_sae)
    assert sae.W_dec.shape == (cfg.sae.d_sae, cfg.sae.d_in)
    assert sae.b_enc.shape == (cfg.sae.d_sae,)
    assert sae.b_dec.shape == (cfg.sae.d_in,)


def test_StandardSAE_to_device_string():
    cfg = build_sae_cfg(device="cpu")
    sae = StandardSAE(cfg)

    # Test moving to device as string
    sae_moved = sae.to("meta")
    assert sae_moved.device == torch.device("meta")
    assert sae_moved.cfg.device == "meta"
    assert sae_moved.W_enc.device == torch.device("meta")
    assert sae_moved.W_dec.device == torch.device("meta")
    assert sae_moved.b_enc.device == torch.device("meta")
    assert sae_moved.b_dec.device == torch.device("meta")


def test_StandardSAE_to_device_torch_device():
    cfg = build_sae_cfg(device="cpu")
    sae = StandardSAE(cfg)

    # Test moving to device as torch.device
    target_device = torch.device("meta")
    sae_moved = sae.to(target_device)
    assert sae_moved.device == target_device
    assert sae_moved.cfg.device == str(target_device)
    assert sae_moved.W_enc.device == target_device
    assert sae_moved.W_dec.device == target_device
    assert sae_moved.b_enc.device == target_device
    assert sae_moved.b_dec.device == target_device


def test_StandardSAE_to_dtype():
    cfg = build_sae_cfg(dtype="float32")
    sae = StandardSAE(cfg)

    # Test changing dtype
    sae_moved = sae.to(torch.float16)
    assert sae_moved.dtype == torch.float16
    assert sae_moved.cfg.dtype == "torch.float16"
    assert sae_moved.W_enc.dtype == torch.float16
    assert sae_moved.W_dec.dtype == torch.float16
    assert sae_moved.b_enc.dtype == torch.float16
    assert sae_moved.b_dec.dtype == torch.float16


def test_sae_to_device_and_dtype():
    cfg = build_sae_cfg(device="cpu", dtype="float32")
    sae = StandardSAE(cfg)

    # Test changing both device and dtype
    sae_moved = sae.to(device="meta", dtype=torch.float16)
    assert sae_moved.device == torch.device("meta")
    assert sae_moved.dtype == torch.float16
    assert sae_moved.cfg.device == "meta"
    assert sae_moved.cfg.dtype == "torch.float16"
    assert sae_moved.W_enc.device == torch.device("meta")
    assert sae_moved.W_enc.dtype == torch.float16
    assert sae_moved.W_dec.device == torch.device("meta")
    assert sae_moved.W_dec.dtype == torch.float16
    assert sae_moved.b_enc.device == torch.device("meta")
    assert sae_moved.b_enc.dtype == torch.float16
    assert sae_moved.b_dec.device == torch.device("meta")
    assert sae_moved.b_dec.dtype == torch.float16


def test_StandardSAE_to_tensor():
    cfg = build_sae_cfg(device="cpu", dtype="float32")
    sae = StandardSAE(cfg)

    # Test using tensor to specify device and dtype
    reference_tensor = torch.randn(10, 10, device="meta", dtype=torch.float16)
    sae_moved = sae.to(reference_tensor)
    assert sae_moved.device == reference_tensor.device
    assert sae_moved.dtype == reference_tensor.dtype
    assert sae_moved.cfg.device == str(reference_tensor.device)
    assert sae_moved.cfg.dtype == str(reference_tensor.dtype)
    assert sae_moved.W_enc.device == reference_tensor.device
    assert sae_moved.W_enc.dtype == reference_tensor.dtype
    assert sae_moved.W_dec.device == reference_tensor.device
    assert sae_moved.W_dec.dtype == reference_tensor.dtype


def test_StandardSAE_to_kwargs_only():
    cfg = build_sae_cfg(device="cpu", dtype="float32")
    sae = StandardSAE(cfg)

    # Test using only kwargs
    sae_moved = sae.to(device="meta", dtype=torch.float16, non_blocking=False)
    assert sae_moved.device == torch.device("meta")
    assert sae_moved.dtype == torch.float16
    assert sae_moved.cfg.device == "meta"
    assert sae_moved.cfg.dtype == "torch.float16"


def test_StandardSAE_to_positional_args():
    cfg = build_sae_cfg(device="cpu", dtype="float32")
    sae = StandardSAE(cfg)

    # Test using positional args (device, dtype)
    sae_moved = sae.to(torch.device("meta"), torch.float16)
    assert sae_moved.device == torch.device("meta")
    assert sae_moved.dtype == torch.float16
    assert sae_moved.cfg.device == "meta"
    assert sae_moved.cfg.dtype == "torch.float16"


def test_StandardSAE_to_device_only_positional():
    cfg = build_sae_cfg(device="cpu", dtype="float32")
    sae = StandardSAE(cfg)

    # Test using only device as positional arg
    sae_moved = sae.to("meta")
    assert sae_moved.device == torch.device("meta")
    assert sae_moved.cfg.device == "meta"
    # dtype should remain unchanged
    assert sae_moved.dtype == torch.float32
    assert sae_moved.cfg.dtype == "float32"


def test_StandardSAE_to_dtype_only_positional():
    cfg = build_sae_cfg(device="cpu", dtype="float32")
    sae = StandardSAE(cfg)

    # Test using only dtype as positional arg
    sae_moved = sae.to(torch.float16)
    assert sae_moved.dtype == torch.float16
    assert sae_moved.cfg.dtype == "torch.float16"
    # device should remain unchanged
    assert sae_moved.device == torch.device("cpu")
    assert sae_moved.cfg.device == "cpu"


def test_StandardSAE_to_no_args():
    cfg = build_sae_cfg(device="cpu", dtype="float32")
    sae = StandardSAE(cfg)

    # Test calling to() with no args should not change anything
    sae_moved = sae.to()
    assert sae_moved.device == torch.device("cpu")
    assert sae_moved.dtype == torch.float32
    assert sae_moved.cfg.device == "cpu"
    assert sae_moved.cfg.dtype == "float32"


def test_StandardSAE_to_returns_same_instance():
    cfg = build_sae_cfg(device="cpu", dtype="float32")
    sae = StandardSAE(cfg)

    # Test that to() returns the same instance (not a copy)
    sae_moved = sae.to("meta")
    assert sae_moved is sae


def test_StandardSAE_fold_w_dec_norm(
    cfg: LanguageModelSAERunnerConfig[StandardTrainingSAEConfig],
):
    sae = StandardSAE(cfg.sae)  # type: ignore
    sae.turn_off_forward_pass_hook_z_reshaping()  # hook z reshaping not needed here.
    # TODO: verify if we're initializing the SAE correctly by default since we now have unit normed W_dec
    # assert sae.W_dec.norm(dim=-1).mean().item() != pytest.approx(1.0, abs=1e-6)
    sae2 = deepcopy(sae)
    sae2.fold_W_dec_norm()

    W_dec_norms = sae.W_dec.norm(dim=-1).unsqueeze(1)
    assert_close(sae2.W_dec.data, sae.W_dec.data / W_dec_norms)
    assert_close(sae2.W_enc.data, sae.W_enc.data * W_dec_norms.T)
    assert_close(sae2.b_enc.data, sae.b_enc.data * W_dec_norms.squeeze())

    # fold_W_dec_norm should normalize W_dec to have unit norm.
    assert sae2.W_dec.norm(dim=-1).mean().item() == pytest.approx(1.0, abs=1e-6)

    # we expect activations of features to differ by W_dec norm weights.
    activations = torch.randn(10, 4, cfg.sae.d_in, device=cfg.device)
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


@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
@torch.no_grad()
def test_sae_fold_w_dec_norm_all_architectures(architecture: str):
    cfg = build_sae_cfg_for_arch(architecture)
    sae = SAE.from_dict(cfg.to_dict())
    sae.turn_off_forward_pass_hook_z_reshaping()  # hook z reshaping not needed here.

    # make sure all parameters are not 0s
    for param in sae.parameters():
        param.data = torch.rand_like(param)

    assert sae.W_dec.norm(dim=-1).mean().item() != pytest.approx(1.0, abs=1e-6)
    sae2 = deepcopy(sae)

    # If this is a topk SAE, assert this throws a NotImplementedError
    if architecture in {"topk", "temporal"}:
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


@pytest.mark.parametrize("architecture", ALL_TRAINING_ARCHITECTURES)
@torch.no_grad()
def test_training_sae_fold_w_dec_norm_all_architectures(architecture: str):
    cfg = build_sae_training_cfg_for_arch(architecture)
    sae = TrainingSAE.from_dict(cfg.to_dict())
    sae.turn_off_forward_pass_hook_z_reshaping()  # hook z reshaping not needed here.

    # make sure all parameters are not 0s
    for param in sae.parameters():
        param.data = torch.rand_like(param)

    assert sae.W_dec.norm(dim=-1).mean().item() != pytest.approx(1.0, abs=1e-6)
    sae2 = deepcopy(sae)

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


@torch.no_grad()
def test_StandardSAE_fold_norm_scaling_factor(
    cfg: LanguageModelSAERunnerConfig[StandardTrainingSAEConfig],
):
    norm_scaling_factor = 3.0

    sae = StandardSAE(cfg.sae)  # type: ignore
    # make sure b_dec and b_enc are not 0s
    sae.b_dec.data = torch.randn(cfg.sae.d_in, device=cfg.device)
    sae.b_enc.data = torch.randn(cfg.sae.d_sae, device=cfg.device)  # type: ignore
    sae.turn_off_forward_pass_hook_z_reshaping()  # hook z reshaping not needed here.

    sae2 = deepcopy(sae)
    sae2.fold_activation_norm_scaling_factor(norm_scaling_factor)

    assert sae2.cfg.normalize_activations == "none"

    assert_close(sae2.W_enc.data, sae.W_enc.data * norm_scaling_factor)

    # we expect activations of features to differ by W_dec norm weights.
    # assume activations are already scaled
    activations = torch.randn(10, 4, cfg.sae.d_in, device=cfg.device)
    # we divide to get the unscale activations
    unscaled_activations = activations / norm_scaling_factor

    feature_activations_1 = sae.encode(activations)
    # with the scaling folded in, the unscaled activations should produce the same
    # result.
    feature_activations_2 = sae2.encode(unscaled_activations)

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
    sae.turn_off_forward_pass_hook_z_reshaping()

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
    sae.turn_off_forward_pass_hook_z_reshaping()

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
    sae.fold_W_dec_norm()

    # Verify no NaN or Inf values in any parameters
    for name, param in sae.named_parameters():
        assert not torch.isnan(
            param
        ).any(), f"NaN found in {name} after fold_W_dec_norm"
        assert not torch.isinf(
            param
        ).any(), f"Inf found in {name} after fold_W_dec_norm"


def test_StandardSAE_save_and_load_from_pretrained(tmp_path: Path) -> None:
    cfg = build_sae_cfg()
    model_path = str(tmp_path)
    sae = SAE.from_dict(cfg.to_dict())
    sae_state_dict = sae.state_dict()
    sae.save_model(model_path)

    assert os.path.exists(model_path)

    sae_loaded = SAE.load_from_pretrained(model_path, device="cpu")

    sae_loaded_state_dict = sae_loaded.state_dict()

    # check state_dict matches the original
    for key in sae.state_dict():
        assert_close(
            sae_state_dict[key],
            sae_loaded_state_dict[key],
        )

    sae_in = torch.randn(10, cfg.d_in, device=cfg.device)
    sae_out_1 = sae(sae_in)
    sae_out_2 = sae_loaded(sae_in)
    assert_close(sae_out_1, sae_out_2)


def test_StandardSAE_get_name_returns_correct_name_from_cfg_vals() -> None:
    cfg = build_sae_cfg(d_sae=128)
    cfg.metadata.model_name = "test_model"
    cfg.metadata.hook_name = "test_hook_name"
    sae = SAE.from_dict(cfg.to_dict())
    assert sae.get_name() == "sae_test_model_test_hook_name_128"


def test_StandardSAE_move_between_devices() -> None:
    cfg = build_sae_cfg()
    sae = SAE.from_dict(cfg.to_dict())

    sae.to("meta")
    assert sae.device == torch.device("meta")
    assert sae.cfg.device == "meta"
    assert sae.W_enc.device == torch.device("meta")


def test_StandardSAE_disable_hooks_temporarily_stops_hooks_from_running():
    cfg = build_sae_cfg(d_in=2, d_sae=3)
    sae = SAE.from_dict(cfg.to_dict())
    sae_in = torch.randn(10, cfg.d_in)

    orig_out, orig_cache = sae.run_with_cache(sae_in)
    with _disable_hooks(sae):
        disabled_out, disabled_cache = sae.run_with_cache(sae_in)
    subseq_out, subseq_cache = sae.run_with_cache(sae_in)

    assert_close(orig_out, disabled_out)
    assert_close(orig_out, subseq_out)
    assert disabled_cache.keys() == set()
    for key in orig_cache:
        assert_close(orig_cache[key], subseq_cache[key])


@pytest.mark.parametrize("architecture", ["standard", "gated", "jumprelu"])
def test_sae_forward_pass_works_with_error_term_and_hooks(architecture: str):
    cfg = build_sae_cfg_for_arch(architecture=architecture, d_in=32, d_sae=64)
    sae = SAE.from_dict(cfg.to_dict())
    sae.use_error_term = True
    sae_in = torch.randn(10, cfg.d_in)
    original_out, original_cache = sae.run_with_cache(sae_in)

    def ablate_hooked_sae(acts: torch.Tensor, hook: HookPoint):  # noqa: ARG001
        acts[:, :] = 20
        return acts

    with sae.hooks(fwd_hooks=[("hook_sae_acts_post", ablate_hooked_sae)]):
        ablated_out, ablated_cache = sae.run_with_cache(sae_in)

    assert_not_close(original_out, ablated_out, rtol=1e-2)
    assert torch.all(ablated_cache["hook_sae_acts_post"] == 20)
    assert_close(
        original_cache["hook_sae_error"],
        ablated_cache["hook_sae_error"],
        rtol=1e-4,
    )


def test_SAE_from_pretrained_loads_from_hugginface_using_shorthand():
    sae, original_cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
        release="gpt2-small-res-jb",
        sae_id="blocks.0.hook_resid_pre",
        device="cpu",
    )

    assert (
        sae.cfg.metadata.neuronpedia_id == "gpt2-small/0-res-jb"
    )  # what we expect from the yml

    # it should match what we get when manually loading from hf
    repo_id = "jbloom/GPT2-Small-SAEs-Reformatted"
    hook_point = "blocks.0.hook_resid_pre"
    filename = f"{hook_point}/sae_weights.safetensors"
    weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
    state_dict = {}
    with safe_open(weight_path, framework="pt", device="cpu") as f:  # type: ignore
        for k in f.keys():  # noqa: SIM118
            state_dict[k] = f.get_tensor(k)

    assert isinstance(sae, SAE)
    assert sae.cfg.metadata.model_name == "gpt2-small"
    assert sae.cfg.metadata.hook_name == "blocks.0.hook_resid_pre"

    assert isinstance(original_cfg_dict, dict)

    assert isinstance(sparsity, torch.Tensor)
    assert sparsity.shape == (sae.cfg.d_sae,)
    assert sparsity.max() < 0.0

    for k in sae.state_dict():
        if k == "finetuning_scaling_factor":
            continue
        assert_close(sae.state_dict()[k], state_dict[k])


def test_SAE_from_pretrained_can_load_arbitrary_saes_from_huggingface():
    sae, original_cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
        release="jbloom/GPT2-Small-SAEs-Reformatted",
        sae_id="blocks.0.hook_resid_pre",
        device="cpu",
    )

    # it should match what we get when manually loading from hf
    repo_id = "jbloom/GPT2-Small-SAEs-Reformatted"
    hook_point = "blocks.0.hook_resid_pre"
    filename = f"{hook_point}/sae_weights.safetensors"
    weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
    state_dict = {}
    with safe_open(weight_path, framework="pt", device="cpu") as f:  # type: ignore
        for k in f.keys():  # noqa: SIM118
            state_dict[k] = f.get_tensor(k)

    assert isinstance(sae, SAE)
    assert sae.cfg.metadata.model_name == "gpt2-small"
    assert sae.cfg.metadata.hook_name == "blocks.0.hook_resid_pre"

    assert isinstance(original_cfg_dict, dict)

    assert isinstance(sparsity, torch.Tensor)
    assert sparsity.shape == (sae.cfg.d_sae,)
    assert sparsity.max() < 0.0

    for k in sae.state_dict():
        if k == "finetuning_scaling_factor":
            continue
        assert_close(sae.state_dict()[k], state_dict[k])


def test_SAE_from_pretrained_errors_for_invalid_releases():
    with pytest.raises(ValueError):
        SAE.from_pretrained(
            release="wrong",
            sae_id="blocks.0.hook_resid_pre",
            device="cpu",
        )


def test_SAE_from_pretrained_errors_for_invalid_sae_ids():
    with pytest.raises(ValueError):
        SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="wrong",
            device="cpu",
        )


def test_StandardTrainingSAE_initialization_standard():
    cfg = build_runner_cfg()

    sae = StandardTrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    assert sae.W_enc.shape == (cfg.sae.d_in, cfg.sae.d_sae)
    assert sae.W_dec.shape == (cfg.sae.d_sae, cfg.sae.d_in)
    assert sae.b_enc.shape == (cfg.sae.d_sae,)
    assert sae.b_dec.shape == (cfg.sae.d_in,)
    assert isinstance(sae.activation_fn, torch.nn.ReLU)
    assert sae.device == torch.device("cpu")
    assert sae.dtype == torch.float32

    # biases
    assert_close(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert_close(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)

    # check if the decoder weight norm is 0.1 by default
    assert_close(
        sae.W_dec.norm(dim=1),
        0.1 * torch.ones_like(sae.W_dec.norm(dim=1)),
        atol=1e-6,
    )

    #  Default currently should be tranpose initialization
    assert_close(sae.W_enc, sae.W_dec.T, atol=1e-6)


def test_StandardTrainingSAE_initialization_decoder_norm():
    cfg = build_runner_cfg(decoder_init_norm=0.7)

    sae = StandardTrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    assert_close(
        sae.W_dec.norm(dim=1),
        0.7 * torch.ones_like(sae.W_dec.norm(dim=1)),
        atol=1e-6,
    )

    # initialized weights of biases are 0
    assert_close(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert_close(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)


def test_StandardTrainingSAE_initialization_enc_dec_T_no_unit_norm():
    cfg = build_runner_cfg(
        init_encoder_as_decoder_transpose=True,
        normalize_sae_decoder=False,
    )

    sae = StandardTrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    assert_close(sae.W_dec, sae.W_enc.T, atol=1e-6)

    # initialized weights of biases are 0
    assert_close(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert_close(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)


def test_StandardSAE_constant_norm_rescale():
    cfg = build_sae_cfg(d_in=2, d_sae=3, normalize_activations="constant_norm_rescale")

    sae = StandardSAE(cfg)

    test_input = torch.randn(10, 2, device=cfg.device)

    scaled_input = sae.run_time_activation_norm_fn_in(test_input)
    expected_scaler = (cfg.d_in**0.5) / test_input.norm(dim=-1, keepdim=True)
    assert_close(scaled_input, test_input * expected_scaler, atol=1e-6)
    scaled_output = sae.run_time_activation_norm_fn_out(scaled_input)
    assert_close(scaled_output, test_input)


def test_StandardSAE_layer_norm():
    cfg = build_sae_cfg(d_in=2, d_sae=3, normalize_activations="layer_norm")

    sae = StandardSAE(cfg)

    test_input = torch.randn(10, 2, device=cfg.device)

    scaled_input = sae.run_time_activation_norm_fn_in(test_input)
    expected_mu = test_input.mean(dim=-1, keepdim=True)
    expected_std = test_input.std(dim=-1, keepdim=True)
    assert_close(sae.ln_mu, expected_mu, atol=1e-6)  # type: ignore
    assert_close(sae.ln_std, expected_std, atol=1e-6)  # type: ignore
    assert_close(
        scaled_input, (test_input - expected_mu) / (expected_std + 1e-5), atol=1e-6
    )
    scaled_output = sae.run_time_activation_norm_fn_out(scaled_input)
    assert_close(scaled_output, test_input, atol=1e-4)


def test_StandardSAE_none():
    cfg = build_sae_cfg(d_in=2, d_sae=3, normalize_activations="none")
    sae = StandardSAE(cfg)

    test_input = torch.randn(10, 2, device=cfg.device)

    scaled_input = sae.run_time_activation_norm_fn_in(test_input)
    assert_close(scaled_input, test_input)
    scaled_output = sae.run_time_activation_norm_fn_out(scaled_input)
    assert_close(scaled_output, test_input)


def test_StandardTrainingSAE_save_and_load_inference_sae(tmp_path: Path) -> None:
    # Create a training SAE with specific parameter values
    cfg = build_sae_training_cfg(device="cpu")
    training_sae = StandardTrainingSAE(cfg)

    # Set some known values for testing
    training_sae.W_enc.data = torch.randn_like(training_sae.W_enc.data)
    training_sae.W_dec.data = torch.randn_like(training_sae.W_dec.data)
    training_sae.b_enc.data = torch.randn_like(training_sae.b_enc.data)
    training_sae.b_dec.data = torch.randn_like(training_sae.b_dec.data)

    # Save original state for comparison
    original_W_enc = training_sae.W_enc.data.clone()
    original_W_dec = training_sae.W_dec.data.clone()
    original_b_enc = training_sae.b_enc.data.clone()
    original_b_dec = training_sae.b_dec.data.clone()

    # Save as inference model
    model_path = str(tmp_path)
    training_sae.save_inference_model(model_path)

    assert os.path.exists(model_path)

    # Load as inference SAE
    inference_sae = SAE.load_from_disk(model_path, device="cpu")

    # Should be loaded as StandardSAE
    assert isinstance(inference_sae, StandardSAE)

    # Check that all parameters match
    assert_close(inference_sae.W_enc, original_W_enc)
    assert_close(inference_sae.W_dec, original_W_dec)
    assert_close(inference_sae.b_enc, original_b_enc)
    assert_close(inference_sae.b_dec, original_b_dec)

    # Verify forward pass gives same results
    sae_in = torch.randn(10, cfg.d_in, device="cpu")

    # Get output from training SAE
    training_feature_acts, _ = training_sae.encode_with_hidden_pre(sae_in)
    training_sae_out = training_sae.decode(training_feature_acts)

    # Get output from inference SAE
    inference_feature_acts = inference_sae.encode(sae_in)
    inference_sae_out = inference_sae.decode(inference_feature_acts)

    # Should produce identical outputs
    assert_close(training_feature_acts, inference_feature_acts)
    assert_close(training_sae_out, inference_sae_out)

    # Test the full forward pass
    training_full_out = training_sae(sae_in)
    inference_full_out = inference_sae(sae_in)
    assert_close(training_full_out, inference_full_out)
