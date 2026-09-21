import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from transformer_lens.HookedTransformer import HookedRootModule

from sae_lens.evals import get_sparsity_and_variance_metrics
from sae_lens.registry import get_sae_class, get_sae_training_class
from sae_lens.saes.abstopk_sae import (
    AbsTopK,
    AbsTopKSAE,
    AbsTopKSAEConfig,
    AbsTopKTrainingSAE,
    AbsTopKTrainingSAEConfig,
    calculate_abstopk_aux_acts,
)
from sae_lens.saes.sae import SAE, SAEMetadata, TrainStepInput
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.activations_store import ActivationsStore
from tests.helpers import (
    assert_close,
    build_abstopk_sae_cfg,
    build_abstopk_sae_training_cfg,
    random_params,
)


def _identity_abstopk_sae(k: int, d: int = 6) -> AbsTopKSAE:
    cfg = AbsTopKSAEConfig(
        d_in=d,
        d_sae=d,
        k=k,
        apply_b_dec_to_input=False,
        metadata=SAEMetadata(hook_name="blocks.0.hook_resid_post"),
    )
    sae = AbsTopKSAE(cfg)
    with torch.no_grad():
        sae.W_enc.data = torch.eye(d)
        sae.W_dec.data = torch.eye(d)
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    return sae


def _identity_abstopk_training_sae(k: int, d: int = 6) -> AbsTopKTrainingSAE:
    cfg = AbsTopKTrainingSAEConfig(
        d_in=d,
        d_sae=d,
        k=k,
        apply_b_dec_to_input=False,
        metadata=SAEMetadata(hook_name="blocks.0.hook_resid_post"),
    )
    sae = AbsTopKTrainingSAE(cfg)
    with torch.no_grad():
        sae.W_enc.data = torch.eye(d)
        sae.W_dec.data = torch.eye(d)
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    return sae


def test_AbsTopK_selects_top_k_by_magnitude_and_preserves_sign():
    # -4.0 has larger magnitude than 0.5 and 3.0's magnitude ties nothing, so the
    # top 3 by |x| are 5.0, -4.0 and 3.0, all kept with their signs (no ReLU).
    x = torch.tensor([[5.0, -4.0, 0.5, 3.0, -0.1, 0.0]])
    assert AbsTopK(3)(x).equal(torch.tensor([[5.0, -4.0, 0.0, 3.0, 0.0, 0.0]]))

    # a large negative outranks a smaller positive: -10.0 survives, +1.0 doesn't
    x = torch.tensor([[-10.0, 1.0, -0.5, 0.25, -6.0, 2.0]])
    assert AbsTopK(2)(x).equal(torch.tensor([[-10.0, 0.0, 0.0, 0.0, -6.0, 0.0]]))


def test_AbsTopK_keeps_exactly_k_signed_values_on_random_inputs():
    x = torch.randn(500, 128)
    acts = AbsTopK(7)(x)
    nonzero = acts != 0
    # exactly k features survive per sample
    assert nonzero.sum(dim=-1).eq(7).all()
    # the surviving values are the original signed pre-activations
    assert torch.equal(acts[nonzero], x[nonzero])
    # with roughly balanced signs, a positive-only activation would drop about
    # half of the selected features, so most samples must keep at least one
    # negative value
    assert (acts < 0).any(dim=-1).float().mean() > 0.9


def test_AbsTopKSAE_encode_applies_magnitude_topk_without_relu():
    sae = _identity_abstopk_sae(k=3)
    # with identity weights, hidden_pre == x, so encode is exactly the AbsTopK
    # operator on the input; in both rows the top 3 by magnitude are two large
    # negatives plus one positive, all kept with their signs (no ReLU)
    x = torch.tensor(
        [[5.0, -4.0, 0.5, 3.0, -0.1, 0.0], [-10.0, 1.0, -0.5, 7.0, -6.0, 0.25]]
    )
    expected = torch.tensor(
        [[5.0, -4.0, 0.0, 3.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 7.0, -6.0, 0.0]]
    )
    assert sae.encode(x).equal(expected)


def test_AbsTopKSAE_identity_round_trip_reconstructs_input_exactly():
    # with k = d_sae every feature survives, so an identity-weight SAE must
    # reconstruct its input exactly (decode adds b_dec = 0)
    sae = _identity_abstopk_sae(k=6)
    x = torch.randn(20, 6)
    assert sae(x).equal(x)


def test_AbsTopKTrainingSAE_encode_with_hidden_pre_and_decode_round_trip():
    sae = _identity_abstopk_training_sae(k=6)
    x = torch.randn(20, 6)
    feature_acts, hidden_pre = sae.encode_with_hidden_pre(x)
    # decoder columns have unit norm, so rescale_acts_by_decoder_norm is a no-op
    # and hidden_pre is exactly x @ W_enc + b_enc
    assert hidden_pre.equal(x)
    assert feature_acts.equal(x)
    assert sae.decode(feature_acts).equal(x)


def test_AbsTopKTrainingSAE_training_step_gradients_flow_to_all_params():
    sae = AbsTopKTrainingSAE(build_abstopk_sae_training_cfg(k=5))
    random_params(sae)
    step_input = TrainStepInput(
        sae_in=torch.randn(32, sae.cfg.d_in),
        coefficients={},
        dead_neuron_mask=torch.rand(sae.cfg.d_sae) < 0.5,
        n_training_steps=0,
        is_logging_step=False,
    )
    output = sae.training_forward_pass(step_input)
    output.loss.backward()
    for param in [sae.W_enc, sae.W_dec, sae.b_enc]:
        assert param.grad is not None
        assert param.grad.abs().sum() > 0.0


def test_calculate_abstopk_aux_acts_selects_dead_latents_by_magnitude_with_sign():
    hidden_pre = torch.tensor([[5.0, -4.0, 0.5, 3.0, -0.1]])
    dead_neuron_mask = torch.ones(5, dtype=torch.bool)
    # top 2 by magnitude among dead latents are 5.0 and -4.0, both kept signed;
    # a value-based (TopK-style) selection would pick 5.0 and 3.0 instead
    assert calculate_abstopk_aux_acts(2, hidden_pre, dead_neuron_mask).equal(
        torch.tensor([[5.0, -4.0, 0.0, 0.0, 0.0]])
    )

    # living latents are excluded even if they have the largest magnitudes
    dead_neuron_mask = torch.tensor([False, True, True, True, True])
    assert calculate_abstopk_aux_acts(1, hidden_pre, dead_neuron_mask).equal(
        torch.tensor([[0.0, -4.0, 0.0, 0.0, 0.0]])
    )


def test_AbsTopKTrainingSAE_calculate_aux_loss_returns_zero_without_dead_neurons():
    sae = _identity_abstopk_training_sae(k=3)
    step_input = TrainStepInput(
        sae_in=torch.randn(4, 6),
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )
    x = torch.randn(4, 6)
    feature_acts, hidden_pre = sae.encode_with_hidden_pre(x)
    sae_out = sae.decode(feature_acts)
    losses = sae.calculate_aux_loss(step_input, feature_acts, hidden_pre, sae_out)
    assert set(losses.keys()) == {"auxiliary_reconstruction_loss"}
    assert losses["auxiliary_reconstruction_loss"].item() == pytest.approx(0.0)


class _StubActivationsStore:
    def __init__(self, batch_tokens: torch.Tensor):
        self.batch_tokens = batch_tokens

    def get_batch_tokens(self, batch_size_prompts: int) -> torch.Tensor:
        return self.batch_tokens[:batch_size_prompts]


class _StubModel:
    def __init__(self, hook_name: str, acts: torch.Tensor):
        self.hook_name = hook_name
        self.acts = acts

    def run_with_cache(
        self,
        batch_tokens: torch.Tensor,  # noqa: ARG002
        prepend_bos: bool,  # noqa: ARG002
        names_filter: list[str],  # noqa: ARG002
        stop_at_layer: int,  # noqa: ARG002
        **model_kwargs: Any,  # noqa: ARG002
    ) -> tuple[None, dict[str, torch.Tensor]]:
        return None, {self.hook_name: self.acts}


def test_evals_l0_and_feature_density_count_negative_active_features():
    # every activation is negative with distinct magnitudes, so all k selected
    # features fire below zero: an eval that counts acts > 0 would report l0 = 0
    # and empty feature densities, while the correct != 0 count reports l0 = k
    d = 6
    k = 3
    hook_name = "blocks.0.hook_resid_post"
    sae = _identity_abstopk_sae(k=k, d=d)
    per_token = -(torch.arange(d, dtype=torch.float32) + 1.0)
    acts = per_token.repeat(2, 4, 1)
    stub_store = _StubActivationsStore(torch.zeros(2, 4, dtype=torch.long))
    stub_model = _StubModel(hook_name, acts)

    metrics, feature_metrics = get_sparsity_and_variance_metrics(
        sae=sae,
        model=cast(HookedRootModule, stub_model),
        activation_store=cast(ActivationsStore, stub_store),
        activation_scaler=ActivationScaler(None),
        n_batches=1,
        compute_l2_norms=False,
        compute_sparsity_metrics=True,
        compute_variance_metrics=False,
        compute_featurewise_density_statistics=True,
        eval_batch_size_prompts=2,
        model_kwargs={},
    )

    assert metrics["l0"] == pytest.approx(k)
    # the same k features (largest magnitude, indices d-k..d-1) fire on every
    # token, each with density 1.0
    assert sum(feature_metrics["feature_density"]) == pytest.approx(k)


def test_AbsTopKSAE_save_and_load_from_pretrained(tmp_path: Path) -> None:
    cfg = build_abstopk_sae_cfg(k=30)
    model_path = str(tmp_path)
    sae = AbsTopKSAE(cfg)
    random_params(sae)

    sae_state_dict = sae.state_dict()
    sae.save_model(model_path)

    assert os.path.exists(model_path)

    sae_loaded = SAE.load_from_pretrained(model_path, device="cpu")
    assert isinstance(sae_loaded, AbsTopKSAE)

    sae_loaded_state_dict = sae_loaded.state_dict()

    for key in sae.state_dict():
        assert_close(
            sae_state_dict[key],
            sae_loaded_state_dict[key],
        )

    sae_in = torch.randn(10, cfg.d_in, device=cfg.device)
    assert_close(sae(sae_in), sae_loaded(sae_in))


@pytest.mark.parametrize("rescale_acts_by_decoder_norm", [True, False])
def test_AbsTopKTrainingSAE_save_and_load_inference_sae(
    rescale_acts_by_decoder_norm: bool,
    tmp_path: Path,
):
    cfg = build_abstopk_sae_training_cfg(
        device="cpu", k=30, rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm
    )
    training_sae = AbsTopKTrainingSAE(cfg)
    random_params(training_sae)

    original_W_dec = training_sae.W_dec.data.clone()
    original_b_dec = training_sae.b_dec.data.clone()

    model_path = str(tmp_path)
    training_sae.save_inference_model(model_path)

    assert os.path.exists(model_path)

    inference_sae = SAE.load_from_disk(model_path, device="cpu")
    assert isinstance(inference_sae, AbsTopKSAE)

    if rescale_acts_by_decoder_norm:
        assert_close(
            inference_sae.W_dec.norm(dim=-1),
            torch.ones_like(inference_sae.b_enc),
        )
        assert not torch.allclose(inference_sae.W_dec, original_W_dec)
    else:
        assert_close(inference_sae.W_dec, original_W_dec)
    assert_close(inference_sae.b_dec, original_b_dec)

    assert inference_sae.cfg.k == cfg.k

    sae_in = torch.randn(10, cfg.d_in, device="cpu")
    training_feature_acts, _ = training_sae.encode_with_hidden_pre(sae_in)
    training_sae_out = training_sae.decode(training_feature_acts)
    inference_feature_acts = inference_sae.encode(sae_in)
    inference_sae_out = inference_sae.decode(inference_feature_acts)

    assert_close(training_feature_acts, inference_feature_acts, rtol=1e-4, atol=1e-4)
    assert_close(training_sae_out, inference_sae_out, rtol=1e-4, atol=1e-4)
    assert_close(training_sae(sae_in), inference_sae(sae_in), rtol=1e-4, atol=1e-4)


def test_abstopk_is_registered_with_inference_and_training_classes():
    assert get_sae_class("abstopk") == (AbsTopKSAE, AbsTopKSAEConfig)
    assert get_sae_training_class("abstopk") == (
        AbsTopKTrainingSAE,
        AbsTopKTrainingSAEConfig,
    )
    assert AbsTopKSAEConfig.architecture() == "abstopk"
    assert AbsTopKTrainingSAEConfig.architecture() == "abstopk"
    # training configs export their inference counterpart
    assert (
        AbsTopKTrainingSAEConfig(d_in=2, d_sae=4).get_inference_config_class()
        is AbsTopKSAEConfig
    )
