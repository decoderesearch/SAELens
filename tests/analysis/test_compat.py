from unittest.mock import patch

from sae_lens.analysis.compat import (
    get_transformer_lens_version,
    has_transformer_bridge,
)


def test_get_transformer_lens_version_parses_stable_version():
    with patch("importlib.metadata.version", return_value="2.7.1"):
        assert get_transformer_lens_version() == (2, 7, 1)


def test_get_transformer_lens_version_parses_beta_version():
    with patch("importlib.metadata.version", return_value="3.0.0b1"):
        assert get_transformer_lens_version() == (3, 0, 0)


def test_get_transformer_lens_version_parses_alpha_version():
    with patch("importlib.metadata.version", return_value="3.1.0a2"):
        assert get_transformer_lens_version() == (3, 1, 0)


def test_get_transformer_lens_version_parses_rc_version():
    with patch("importlib.metadata.version", return_value="3.0.0rc1"):
        assert get_transformer_lens_version() == (3, 0, 0)


def test_has_transformer_bridge_returns_true_for_v3():
    with patch("importlib.metadata.version", return_value="3.0.0b1"):
        assert has_transformer_bridge() is True


def test_has_transformer_bridge_returns_false_for_v2():
    with patch("importlib.metadata.version", return_value="2.7.1"):
        assert has_transformer_bridge() is False
