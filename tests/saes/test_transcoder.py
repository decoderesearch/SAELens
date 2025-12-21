import copy
from typing import Any

import pytest
import torch

from sae_lens.saes.transcoder import (
    JumpReLUSkipTranscoder,
    JumpReLUSkipTranscoderConfig,
    JumpReLUTranscoder,
    JumpReLUTranscoderConfig,
    SkipTranscoder,
    SkipTranscoderConfig,
    Transcoder,
    TranscoderConfig,
)
from tests.helpers import assert_close

# pyright: reportArgumentType=false


# Helper functions to build configs
def build_transcoder_cfg(**kwargs: Any) -> TranscoderConfig:  # type: ignore
    """Helper to create a TranscoderConfig with defaults."""
    default_config = {
        "d_in": 64,
        "d_sae": 128,
        "d_out": 96,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
        "apply_b_dec_to_input": False,
        "metadata": {
            "hook_name_out": "blocks.0.hook_mlp_out",
        },
    }
    merged = {**default_config, **kwargs}
    # Cast types to ensure they match expected types
    if "d_in" in merged:
        merged["d_in"] = int(merged["d_in"])
    if "d_sae" in merged:
        merged["d_sae"] = int(merged["d_sae"])
    if "d_out" in merged:
        merged["d_out"] = int(merged["d_out"])
    if "hook_layer_out" in merged:
        merged["hook_layer_out"] = int(merged["hook_layer_out"])
    if "dtype" in merged:
        merged["dtype"] = str(merged["dtype"])
    if "device" in merged:
        merged["device"] = str(merged["device"])
    if "hook_name_out" in merged:
        merged["hook_name_out"] = str(merged["hook_name_out"])
    if "normalize_activations" in merged:
        merged["normalize_activations"] = str(merged["normalize_activations"])
    if "apply_b_dec_to_input" in merged:
        merged["apply_b_dec_to_input"] = bool(merged["apply_b_dec_to_input"])
    return TranscoderConfig.from_dict(merged)


def build_jumprelu_transcoder_cfg(**kwargs: Any) -> JumpReLUTranscoderConfig:
    """Helper to create a JumpReLUTranscoderConfig with defaults."""
    default_dict = build_transcoder_cfg(**kwargs).to_dict()
    # Remove fields that aren't part of the config initialization
    default_dict.pop("architecture", None)
    default_dict.pop("metadata", None)
    return JumpReLUTranscoderConfig(**default_dict)


def build_skip_transcoder_cfg(**kwargs: Any) -> SkipTranscoderConfig:
    """Helper to create a SkipTranscoderConfig with defaults."""
    default_dict = build_transcoder_cfg(**kwargs).to_dict()
    default_dict.pop("architecture", None)
    default_dict.pop("metadata", None)
    return SkipTranscoderConfig(**default_dict)


def build_jumprelu_skip_transcoder_cfg(**kwargs: Any) -> JumpReLUSkipTranscoderConfig:
    """Helper to create a JumpReLUSkipTranscoderConfig with defaults."""
    default_dict = build_transcoder_cfg(**kwargs).to_dict()
    default_dict.pop("architecture", None)
    default_dict.pop("metadata", None)
    return JumpReLUSkipTranscoderConfig(**default_dict)


class TestTranscoderConfig:
    """Test TranscoderConfig functionality."""

    def test_transcoder_config_initialization(self):
        cfg = build_transcoder_cfg()
        assert cfg.d_in == 64
        assert cfg.d_sae == 128
        assert cfg.d_out == 96
        assert cfg.metadata.hook_name_out == "blocks.0.hook_mlp_out"
        assert cfg.architecture() == "transcoder"

    def test_transcoder_config_to_dict(self):
        cfg = build_transcoder_cfg()
        cfg_dict = cfg.to_dict()

        # Check transcoder-specific fields are included
        assert "d_out" in cfg_dict
        assert "hook_name_out" in cfg_dict["metadata"]

        # Check values
        assert cfg_dict["d_out"] == 96
        assert cfg_dict["metadata"]["hook_name_out"] == "blocks.0.hook_mlp_out"

    def test_transcoder_config_from_dict(self):
        original_dict = {
            "d_in": 128,
            "d_sae": 256,
            "d_out": 192,
            "dtype": "float16",
            "device": "cuda",
            "metadata": {
                "hook_name_out": "test_out",
            },
            "architecture": "transcoder",
            "apply_b_dec_to_input": False,
        }

        cfg = TranscoderConfig.from_dict(original_dict)
        assert cfg.d_in == 128
        assert cfg.d_sae == 256
        assert cfg.d_out == 192
        assert cfg.metadata.hook_name_out == "test_out"


class TestTranscoder:
    """Test base Transcoder functionality."""

    def test_transcoder_initialization(self):
        cfg = build_transcoder_cfg()
        transcoder = Transcoder(cfg)

        # Check dimensions
        assert transcoder.W_enc.shape == (cfg.d_in, cfg.d_sae)
        assert transcoder.W_dec.shape == (cfg.d_sae, cfg.d_out)
        assert transcoder.b_enc.shape == (cfg.d_sae,)
        assert transcoder.b_dec.shape == (cfg.d_out,)

        # Check output hook info
        assert transcoder.d_out == cfg.d_out

    def test_transcoder_forward_pass(self):
        cfg = build_transcoder_cfg(d_in=32, d_sae=64, d_out=48)
        transcoder = Transcoder(cfg)

        batch_size = 10
        x = torch.randn(batch_size, cfg.d_in)

        output, features = transcoder.forward_with_activations(x)

        assert output.shape == (batch_size, cfg.d_out)
        assert features.shape == (batch_size, cfg.d_sae)

        # Check that features are sparse (ReLU should zero out some)
        assert (features == 0).any()

    def test_transcoder_encode_decode(self):
        cfg = build_transcoder_cfg()
        transcoder = Transcoder(cfg)

        batch_size = 5
        x = torch.randn(batch_size, cfg.d_in)

        # Test encode
        features = transcoder.encode(x)
        assert features.shape == (batch_size, cfg.d_sae)

        # Test decode
        output = transcoder.decode(features)
        assert output.shape == (batch_size, cfg.d_out)

    def test_transcoder_process_sae_in(self):
        """Test that process_sae_in doesn't apply b_dec (wrong dimension)."""
        cfg = build_transcoder_cfg()
        transcoder = Transcoder(cfg)

        # Set b_dec to non-zero values
        transcoder.b_dec.data = torch.ones_like(transcoder.b_dec) * 5.0

        x = torch.randn(10, cfg.d_in)
        processed = transcoder.process_sae_in(x)

        # Should be same as input (just dtype conversion and hooks)
        assert_close(processed, x.to(transcoder.dtype))

    def test_transcoder_from_dict(self):
        config_dict = {
            "d_in": 256,
            "d_sae": 512,
            "d_out": 384,
            "dtype": "float32",
            "device": "cpu",
            "hook_name_out": "test_out",
            "architecture": "transcoder",
            "apply_b_dec_to_input": False,
        }

        transcoder = Transcoder.from_dict(config_dict)
        assert isinstance(transcoder, Transcoder)
        assert transcoder.cfg.d_in == 256
        assert transcoder.cfg.d_out == 384

    def test_transcoder_to_device(self):
        cfg = build_transcoder_cfg(device="cpu")
        transcoder = Transcoder(cfg)

        # Move to meta device
        transcoder_moved = transcoder.to("meta")
        assert transcoder_moved.device == torch.device("meta")
        assert transcoder_moved.W_enc.device == torch.device("meta")
        assert transcoder_moved.W_dec.device == torch.device("meta")
        assert transcoder_moved.b_enc.device == torch.device("meta")
        assert transcoder_moved.b_dec.device == torch.device("meta")

    def test_transcoder_fold_w_dec_norm(self):
        cfg = build_transcoder_cfg()
        transcoder = Transcoder(cfg)

        # Set weights to non-unit norm
        # make sure all parameters are not 0s

        for param in transcoder.parameters():
            param.data = torch.rand_like(param)
        original_dec_norms = transcoder.W_dec.norm(dim=1).clone()

        transcoder2 = copy.deepcopy(transcoder)
        transcoder2.fold_W_dec_norm()

        # Check W_dec is normalized
        assert transcoder2.W_dec.norm(dim=1).mean().item() == pytest.approx(
            1.0, abs=1e-6
        )

        # Check that encode/decode produces same output
        x = torch.randn(10, cfg.d_in)
        out1, feat1 = transcoder.forward_with_activations(x)
        out2, feat2 = transcoder2.forward_with_activations(x)

        # Features should differ by decoder norm
        expected_feat2 = feat1 * original_dec_norms
        assert_close(feat2, expected_feat2, atol=1e-5)

        # But outputs should be the same
        assert_close(out1, out2, atol=5e-5)


class TestJumpReLUTranscoder:
    """Test JumpReLUTranscoder functionality."""

    def test_jumprelu_transcoder_initialization(self):
        cfg = build_jumprelu_transcoder_cfg()
        transcoder = JumpReLUTranscoder(cfg)

        # Check threshold parameter
        assert hasattr(transcoder, "threshold")
        assert transcoder.threshold.shape == (cfg.d_sae,)
        assert torch.all(transcoder.threshold == 0)

    def test_jumprelu_transcoder_encode(self):
        cfg = build_jumprelu_transcoder_cfg(d_in=8, d_sae=16)
        transcoder = JumpReLUTranscoder(cfg)

        # Set specific threshold values
        transcoder.threshold.data = torch.tensor([0.5] * 8 + [1.0] * 8)

        # Create input that will trigger different thresholds
        batch_size = 4
        x = torch.ones(batch_size, cfg.d_in) * 0.1

        # Set W_enc to identity-like for easier testing
        transcoder.W_enc.data = torch.eye(cfg.d_in, cfg.d_sae)[: cfg.d_in, : cfg.d_sae]
        transcoder.b_enc.data = torch.zeros_like(transcoder.b_enc)

        features = transcoder.encode(x)

        # Pre-activations should be ~0.1 for first 8 features
        # Only features with pre-activation > threshold should be active
        # Since 0.1 < 0.5, first 8 features should be zero
        assert torch.all(features[:, :8] == 0)

    def test_jumprelu_transcoder_fold_w_dec_norm(self):
        cfg = build_jumprelu_transcoder_cfg()
        transcoder = JumpReLUTranscoder(cfg)

        # Set non-zero threshold
        transcoder.threshold.data = torch.ones_like(transcoder.threshold) * 0.5

        # Set W_dec to non-unit norm
        transcoder.W_dec.data = torch.randn_like(transcoder.W_dec) * 2.0
        original_dec_norms = transcoder.W_dec.norm(dim=1).clone()
        original_threshold = transcoder.threshold.clone()

        transcoder.fold_W_dec_norm()

        # Check W_dec is normalized
        assert transcoder.W_dec.norm(dim=1).mean().item() == pytest.approx(
            1.0, abs=1e-6
        )

        # Check threshold is scaled by decoder norms
        expected_threshold = original_threshold * original_dec_norms
        assert_close(transcoder.threshold, expected_threshold)

    def test_jumprelu_transcoder_architecture(self):
        cfg = build_jumprelu_transcoder_cfg()
        assert cfg.architecture() == "jumprelu_transcoder"


class TestSkipTranscoder:
    """Test SkipTranscoder functionality."""

    def test_skip_transcoder_initialization(self):
        cfg = build_skip_transcoder_cfg()
        transcoder = SkipTranscoder(cfg)

        # Check W_skip parameter exists and has correct shape
        assert hasattr(transcoder, "W_skip")
        assert transcoder.W_skip.shape == (cfg.d_out, cfg.d_in)
        # W_skip should be initialized to zeros
        assert torch.all(transcoder.W_skip == 0)

    def test_skip_transcoder_forward_with_zero_skip(self):
        """With W_skip=0, output should match base Transcoder."""
        cfg = build_skip_transcoder_cfg(d_in=32, d_sae=64, d_out=48)
        skip_transcoder = SkipTranscoder(cfg)

        # Create base transcoder with same weights
        base_cfg = build_transcoder_cfg(d_in=32, d_sae=64, d_out=48)
        base_transcoder = Transcoder(base_cfg)
        base_transcoder.W_enc.data = skip_transcoder.W_enc.data.clone()
        base_transcoder.W_dec.data = skip_transcoder.W_dec.data.clone()
        base_transcoder.b_enc.data = skip_transcoder.b_enc.data.clone()
        base_transcoder.b_dec.data = skip_transcoder.b_dec.data.clone()

        x = torch.randn(10, cfg.d_in)
        skip_out, skip_feat = skip_transcoder.forward_with_activations(x)
        base_out, base_feat = base_transcoder.forward_with_activations(x)

        # With W_skip=0, outputs should be identical
        assert_close(skip_out, base_out)
        assert_close(skip_feat, base_feat)

    def test_skip_transcoder_forward_with_nonzero_skip(self):
        """Test that skip connection is properly applied."""
        cfg = build_skip_transcoder_cfg(d_in=32, d_sae=64, d_out=48)
        transcoder = SkipTranscoder(cfg)

        # Set W_skip to non-zero
        transcoder.W_skip.data = torch.randn_like(transcoder.W_skip)

        x = torch.randn(10, cfg.d_in)
        output, _ = transcoder.forward_with_activations(x)

        # Compute expected skip contribution
        skip_contribution = x @ transcoder.W_skip.T

        # Get output without skip (by temporarily zeroing W_skip)
        w_skip_backup = transcoder.W_skip.data.clone()
        transcoder.W_skip.data.zero_()
        output_no_skip, _ = transcoder.forward_with_activations(x)
        transcoder.W_skip.data = w_skip_backup

        # Output should be no-skip output + skip contribution
        assert_close(output, output_no_skip + skip_contribution)

    def test_skip_transcoder_architecture(self):
        cfg = build_skip_transcoder_cfg()
        assert cfg.architecture() == "skip_transcoder"

    def test_skip_transcoder_from_dict(self):
        config_dict = {
            "d_in": 256,
            "d_sae": 512,
            "d_out": 384,
            "dtype": "float32",
            "device": "cpu",
            "architecture": "skip_transcoder",
            "apply_b_dec_to_input": False,
        }

        transcoder = SkipTranscoder.from_dict(config_dict)
        assert isinstance(transcoder, SkipTranscoder)
        assert hasattr(transcoder, "W_skip")
        assert transcoder.cfg.d_in == 256
        assert transcoder.cfg.d_out == 384


class TestJumpReLUSkipTranscoder:
    """Test JumpReLUSkipTranscoder functionality (double inheritance)."""

    def test_jumprelu_skip_transcoder_initialization(self):
        """Test that both W_skip and threshold parameters are initialized."""
        cfg = build_jumprelu_skip_transcoder_cfg()
        transcoder = JumpReLUSkipTranscoder(cfg)

        # Check W_skip from SkipTranscoder
        assert hasattr(transcoder, "W_skip")
        assert transcoder.W_skip.shape == (cfg.d_out, cfg.d_in)
        assert torch.all(transcoder.W_skip == 0)

        # Check threshold from JumpReLUTranscoder
        assert hasattr(transcoder, "threshold")
        assert transcoder.threshold.shape == (cfg.d_sae,)
        assert torch.all(transcoder.threshold == 0)

        # Check base transcoder weights
        assert transcoder.W_enc.shape == (cfg.d_in, cfg.d_sae)
        assert transcoder.W_dec.shape == (cfg.d_sae, cfg.d_out)
        assert transcoder.b_enc.shape == (cfg.d_sae,)
        assert transcoder.b_dec.shape == (cfg.d_out,)

    def test_jumprelu_skip_transcoder_uses_jumprelu_activation(self):
        """Test that JumpReLU activation is used, not regular ReLU."""
        cfg = build_jumprelu_skip_transcoder_cfg(d_in=8, d_sae=16, d_out=12)
        transcoder = JumpReLUSkipTranscoder(cfg)

        # Set threshold high enough to zero out some activations
        transcoder.threshold.data = torch.ones(cfg.d_sae) * 0.5

        # Set W_enc to identity-like for predictable pre-activations
        transcoder.W_enc.data = torch.eye(cfg.d_in, cfg.d_sae)
        transcoder.b_enc.data = torch.zeros(cfg.d_sae)

        # Input with values < threshold for first 8 features
        x = torch.ones(4, cfg.d_in) * 0.3

        features = transcoder.encode(x)

        # Pre-activations should be ~0.3 which is < 0.5 threshold
        # So first 8 features should be zeroed by JumpReLU
        assert torch.all(features[:, :8] == 0)

    def test_jumprelu_skip_transcoder_uses_skip_connection(self):
        """Test that skip connection is applied in forward pass."""
        cfg = build_jumprelu_skip_transcoder_cfg(d_in=32, d_sae=64, d_out=48)
        transcoder = JumpReLUSkipTranscoder(cfg)

        # Set non-zero W_skip
        transcoder.W_skip.data = torch.randn_like(transcoder.W_skip)

        x = torch.randn(10, cfg.d_in)
        output, _ = transcoder.forward_with_activations(x)

        # Compute expected skip contribution
        skip_contribution = x @ transcoder.W_skip.T

        # Get output without skip
        w_skip_backup = transcoder.W_skip.data.clone()
        transcoder.W_skip.data.zero_()
        output_no_skip, _ = transcoder.forward_with_activations(x)
        transcoder.W_skip.data = w_skip_backup

        # Output should include skip contribution
        assert_close(output, output_no_skip + skip_contribution)

    def test_jumprelu_skip_transcoder_architecture(self):
        cfg = build_jumprelu_skip_transcoder_cfg()
        assert cfg.architecture() == "jumprelu_skip_transcoder"

    def test_jumprelu_skip_transcoder_fold_w_dec_norm(self):
        """Test that fold_W_dec_norm properly scales threshold (from JumpReLU)."""
        cfg = build_jumprelu_skip_transcoder_cfg()
        transcoder = JumpReLUSkipTranscoder(cfg)

        # Set non-zero threshold
        transcoder.threshold.data = torch.ones_like(transcoder.threshold) * 0.5

        # Set W_dec to non-unit norm
        transcoder.W_dec.data = torch.randn_like(transcoder.W_dec) * 2.0
        original_dec_norms = transcoder.W_dec.norm(dim=1).clone()
        original_threshold = transcoder.threshold.clone()

        transcoder.fold_W_dec_norm()

        # Check W_dec is normalized
        assert transcoder.W_dec.norm(dim=1).mean().item() == pytest.approx(
            1.0, abs=1e-6
        )

        # Check threshold is scaled by decoder norms
        expected_threshold = original_threshold * original_dec_norms
        assert_close(transcoder.threshold, expected_threshold)

    def test_jumprelu_skip_transcoder_from_dict(self):
        config_dict = {
            "d_in": 256,
            "d_sae": 512,
            "d_out": 384,
            "dtype": "float32",
            "device": "cpu",
            "architecture": "jumprelu_skip_transcoder",
            "apply_b_dec_to_input": False,
        }

        transcoder = JumpReLUSkipTranscoder.from_dict(config_dict)
        assert isinstance(transcoder, JumpReLUSkipTranscoder)
        assert hasattr(transcoder, "W_skip")
        assert hasattr(transcoder, "threshold")

    def test_jumprelu_skip_transcoder_mro(self):
        """Test that method resolution order is correct for diamond inheritance."""
        # JumpReLUSkipTranscoder should inherit:
        # - encode from JumpReLUTranscoder (with threshold)
        # - forward/forward_with_activations from SkipTranscoder (with skip)
        cfg = build_jumprelu_skip_transcoder_cfg()
        transcoder = JumpReLUSkipTranscoder(cfg)

        # Check MRO - JumpReLUTranscoder should come before SkipTranscoder
        mro = type(transcoder).__mro__
        jumprelu_idx = mro.index(JumpReLUTranscoder)
        skip_idx = mro.index(SkipTranscoder)
        assert (
            jumprelu_idx < skip_idx
        ), "JumpReLUTranscoder should be before SkipTranscoder in MRO"


class TestJumpReLUTranscoderVariants:
    """Parametrized tests for JumpReLUTranscoder and JumpReLUSkipTranscoder."""

    @pytest.fixture(params=["jumprelu", "jumprelu_skip"])
    def jumprelu_transcoder(
        self, request: pytest.FixtureRequest
    ) -> JumpReLUTranscoder | JumpReLUSkipTranscoder:
        """Create either JumpReLUTranscoder or JumpReLUSkipTranscoder."""
        if request.param == "jumprelu":
            cfg = build_jumprelu_transcoder_cfg(d_in=32, d_sae=64, d_out=48)
            return JumpReLUTranscoder(cfg)
        cfg = build_jumprelu_skip_transcoder_cfg(d_in=32, d_sae=64, d_out=48)
        return JumpReLUSkipTranscoder(cfg)

    def test_has_threshold_parameter(
        self, jumprelu_transcoder: JumpReLUTranscoder | JumpReLUSkipTranscoder
    ) -> None:
        """Both variants should have threshold parameter."""
        assert hasattr(jumprelu_transcoder, "threshold")
        assert jumprelu_transcoder.threshold.shape == (64,)

    def test_threshold_affects_encoding(
        self, jumprelu_transcoder: JumpReLUTranscoder | JumpReLUSkipTranscoder
    ) -> None:
        """Both variants should apply threshold in encoding."""
        jumprelu_transcoder.threshold.data = (
            torch.ones(64) * 1000
        )  # Very high threshold
        jumprelu_transcoder.b_enc.data.zero_()

        x = torch.randn(10, 32)
        features = jumprelu_transcoder.encode(x)

        # With very high threshold, all features should be zero
        assert torch.all(features == 0)

    def test_forward_output_shape(
        self, jumprelu_transcoder: JumpReLUTranscoder | JumpReLUSkipTranscoder
    ) -> None:
        """Both variants should produce correct output shapes."""
        x = torch.randn(10, 32)
        output, features = jumprelu_transcoder.forward_with_activations(x)

        assert output.shape == (10, 48)
        assert features.shape == (10, 64)

    def test_fold_w_dec_norm_scales_threshold(
        self, jumprelu_transcoder: JumpReLUTranscoder | JumpReLUSkipTranscoder
    ) -> None:
        """Both variants should scale threshold when folding decoder norms."""
        jumprelu_transcoder.threshold.data = torch.ones(64) * 0.5
        jumprelu_transcoder.W_dec.data = (
            torch.randn_like(jumprelu_transcoder.W_dec) * 2.0
        )
        original_dec_norms = jumprelu_transcoder.W_dec.norm(dim=1).clone()
        original_threshold = jumprelu_transcoder.threshold.clone()

        jumprelu_transcoder.fold_W_dec_norm()

        expected_threshold = original_threshold * original_dec_norms
        assert_close(jumprelu_transcoder.threshold, expected_threshold)


class TestSkipTranscoderVariants:
    """Parametrized tests for SkipTranscoder and JumpReLUSkipTranscoder."""

    @pytest.fixture(params=["skip", "jumprelu_skip"])
    def skip_transcoder(
        self, request: pytest.FixtureRequest
    ) -> SkipTranscoder | JumpReLUSkipTranscoder:
        """Create either SkipTranscoder or JumpReLUSkipTranscoder."""
        if request.param == "skip":
            cfg = build_skip_transcoder_cfg(d_in=32, d_sae=64, d_out=48)
            return SkipTranscoder(cfg)
        cfg = build_jumprelu_skip_transcoder_cfg(d_in=32, d_sae=64, d_out=48)
        return JumpReLUSkipTranscoder(cfg)

    def test_has_w_skip_parameter(
        self, skip_transcoder: SkipTranscoder | JumpReLUSkipTranscoder
    ) -> None:
        """Both variants should have W_skip parameter."""
        assert hasattr(skip_transcoder, "W_skip")
        assert skip_transcoder.W_skip.shape == (48, 32)

    def test_w_skip_initialized_to_zeros(
        self, skip_transcoder: SkipTranscoder | JumpReLUSkipTranscoder
    ) -> None:
        """Both variants should initialize W_skip to zeros."""
        assert torch.all(skip_transcoder.W_skip == 0)

    def test_skip_connection_applied(
        self, skip_transcoder: SkipTranscoder | JumpReLUSkipTranscoder
    ) -> None:
        """Both variants should apply skip connection in forward pass."""
        # Set non-zero W_skip
        skip_transcoder.W_skip.data = torch.randn_like(skip_transcoder.W_skip)

        x = torch.randn(10, 32)
        skip_contribution = x @ skip_transcoder.W_skip.T

        output, _ = skip_transcoder.forward_with_activations(x)

        # Get output without skip
        skip_transcoder.W_skip.data.zero_()
        output_no_skip, _ = skip_transcoder.forward_with_activations(x)

        # Re-add skip for comparison
        assert_close(output, output_no_skip + skip_contribution, atol=1e-5)

    def test_forward_output_shape(
        self, skip_transcoder: SkipTranscoder | JumpReLUSkipTranscoder
    ) -> None:
        """Both variants should produce correct output shapes."""
        x = torch.randn(10, 32)
        output, features = skip_transcoder.forward_with_activations(x)

        assert output.shape == (10, 48)
        assert features.shape == (10, 64)


class TestTranscoderIntegration:
    """Integration tests for transcoders."""

    @pytest.mark.parametrize(
        "d_in,d_sae,d_out",
        [
            (32, 64, 48),  # d_in < d_sae, d_out between
            (64, 32, 96),  # d_in > d_sae, d_out > d_in
            (48, 96, 48),  # d_in == d_out
            (128, 256, 64),  # all different, d_out < d_in
        ],
        ids=[
            "d_in_lt_d_sae_d_out_between",
            "d_in_gt_d_sae_d_out_gt_d_in",
            "d_in_eq_d_out",
            "all_different_d_out_lt_d_in",
        ],
    )
    def test_transcoder_different_dimensions(self, d_in: int, d_sae: int, d_out: int):
        """Test transcoder with various input/output dimension combinations."""
        cfg = build_transcoder_cfg(d_in=d_in, d_sae=d_sae, d_out=d_out)
        transcoder = Transcoder(cfg)

        x = torch.randn(10, d_in)
        output, features = transcoder.forward_with_activations(x)

        assert output.shape == (10, d_out)
        assert features.shape == (10, d_sae)

    def test_transcoder_state_dict_save_load(self):
        """Test saving and loading transcoder state."""
        cfg = build_transcoder_cfg()
        transcoder1 = Transcoder(cfg)

        # Modify weights
        transcoder1.W_enc.data = torch.randn_like(transcoder1.W_enc)
        transcoder1.W_dec.data = torch.randn_like(transcoder1.W_dec)

        # Save state
        state_dict = transcoder1.state_dict()

        # Create new transcoder and load state
        transcoder2 = Transcoder(cfg)
        transcoder2.load_state_dict(state_dict)

        # Check weights match
        assert_close(transcoder1.W_enc, transcoder2.W_enc)
        assert_close(transcoder1.W_dec, transcoder2.W_dec)
        assert_close(transcoder1.b_enc, transcoder2.b_enc)
        assert_close(transcoder1.b_dec, transcoder2.b_dec)

        # Check same output
        x = torch.randn(5, cfg.d_in)
        out1, feat1 = transcoder1.forward_with_activations(x)
        out2, feat2 = transcoder2.forward_with_activations(x)
        assert_close(out1, out2)
        assert_close(feat1, feat2)
