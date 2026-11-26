import json
import urllib.parse
from unittest.mock import MagicMock, patch

import pytest

from sae_lens.analysis.neuronpedia_integration import (
    NEURONPEDIA_DOMAIN,
    NanAndInfReplacer,
    NeuronpediaFeature,
    get_neuronpedia_feature,
    get_neuronpedia_quick_list,
    make_neuronpedia_list_with_features,
    open_neuronpedia_feature_dashboard,
)
from sae_lens.saes.sae import SAE
from sae_lens.saes.standard_sae import StandardSAEConfig


def test_get_neuronpedia_feature():
    result = get_neuronpedia_feature(
        feature=0, layer=0, model="gpt2-small", dataset="res-jb"
    )

    assert result["modelId"] == "gpt2-small"
    assert result["layer"] == "0-res-jb"
    assert result["index"] == 0


@patch("sae_lens.analysis.neuronpedia_integration.webbrowser.open")
def test_get_neuronpedia_quick_list(
    mock_open: MagicMock, gpt2_res_jb_l4_sae: SAE[StandardSAEConfig]
):
    features = [0, 1, 2, 3]
    name = "test_list"

    url = get_neuronpedia_quick_list(gpt2_res_jb_l4_sae, features, name=name)

    # Verify webbrowser.open was called with the returned URL
    mock_open.assert_called_once_with(url)

    # Verify URL structure
    assert url.startswith(f"{NEURONPEDIA_DOMAIN}/quick-list/")
    assert f"name={urllib.parse.quote(name)}" in url

    # Verify the features are properly encoded in the URL
    neuronpedia_id = gpt2_res_jb_l4_sae.cfg.metadata.neuronpedia_id
    assert neuronpedia_id is not None  # This should be true for the test fixture

    expected_features = [
        {
            "modelId": gpt2_res_jb_l4_sae.cfg.metadata.model_name,
            "layer": neuronpedia_id.split("/")[1],
            "index": str(feature),
        }
        for feature in features
    ]
    expected_features_encoded = urllib.parse.quote(json.dumps(expected_features))
    assert f"features={expected_features_encoded}" in url


@patch("sae_lens.analysis.neuronpedia_integration.webbrowser.open")
def test_open_neuronpedia_feature_dashboard(
    mock_open: MagicMock, gpt2_res_jb_l4_sae: SAE[StandardSAEConfig]
):
    index = 42

    open_neuronpedia_feature_dashboard(gpt2_res_jb_l4_sae, index)

    # Verify webbrowser.open was called with the correct URL
    neuronpedia_id = gpt2_res_jb_l4_sae.cfg.metadata.neuronpedia_id
    assert neuronpedia_id is not None  # This should be true for the test fixture

    expected_url = f"{NEURONPEDIA_DOMAIN}/{neuronpedia_id}/{index}"
    mock_open.assert_called_once_with(expected_url)


@patch("sae_lens.analysis.neuronpedia_integration.webbrowser.open")
@patch("sae_lens.analysis.neuronpedia_integration.logger.warning")
def test_open_neuronpedia_feature_dashboard_with_none_id(
    mock_warning: MagicMock,
    mock_open: MagicMock,
    gpt2_res_jb_l4_sae: SAE[StandardSAEConfig],
):
    # Temporarily set neuronpedia_id to None to test the warning path
    original_neuronpedia_id = gpt2_res_jb_l4_sae.cfg.metadata.neuronpedia_id
    gpt2_res_jb_l4_sae.cfg.metadata.neuronpedia_id = None

    try:
        index = 42
        open_neuronpedia_feature_dashboard(gpt2_res_jb_l4_sae, index)

        # Verify warning was logged
        mock_warning.assert_called_once_with(
            "SAE does not have a Neuronpedia ID. Either dashboards for this SAE do not exist (yet) on Neuronpedia, or the SAE was not loaded via the from_pretrained method"
        )

        # Verify webbrowser.open was not called
        mock_open.assert_not_called()
    finally:
        # Restore original neuronpedia_id
        gpt2_res_jb_l4_sae.cfg.metadata.neuronpedia_id = original_neuronpedia_id


@patch("sae_lens.analysis.neuronpedia_integration.requests.post")
@patch("sae_lens.analysis.neuronpedia_integration.webbrowser.open")
def test_make_neuronpedia_list_with_features_success(
    mock_webbrowser_open: MagicMock,
    mock_requests_post: MagicMock,
):
    # Mock successful response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "url": "https://neuronpedia.org/list/test-list-id",
        "listId": "test-list-id",
    }
    mock_requests_post.return_value = mock_response

    # Create test features
    features = [
        NeuronpediaFeature(
            modelId="gpt2-small",
            layer=0,
            dataset="res-jb",
            feature=10,
            description="Test feature description",
        ),
        NeuronpediaFeature(
            modelId="gpt2-small",
            layer=1,
            dataset="att-kk",
            feature=20,
            description="Another test feature",
        ),
    ]

    # Call the function
    result_url = make_neuronpedia_list_with_features(
        api_key="test_api_key",
        list_name="Test List",
        features=features,
        list_description="Test list description",
        open_browser=True,
    )

    # Verify the POST request was made with correct parameters
    expected_url = f"{NEURONPEDIA_DOMAIN}/api/list/new-with-features"
    expected_body = {
        "name": "Test List",
        "description": "Test list description",
        "features": [
            {
                "modelId": "gpt2-small",
                "layer": "0-res-jb",
                "index": 10,
                "description": "Test feature description",
            },
            {
                "modelId": "gpt2-small",
                "layer": "1-att-kk",
                "index": 20,
                "description": "Another test feature",
            },
        ],
    }
    expected_headers = {"x-api-key": "test_api_key"}

    mock_requests_post.assert_called_once_with(
        expected_url, json=expected_body, headers=expected_headers
    )

    # Verify webbrowser.open was called with the returned URL
    mock_webbrowser_open.assert_called_once_with(
        "https://neuronpedia.org/list/test-list-id"
    )

    # Verify the function returned the correct URL
    assert result_url == "https://neuronpedia.org/list/test-list-id"


@patch("sae_lens.analysis.neuronpedia_integration.requests.post")
@patch("sae_lens.analysis.neuronpedia_integration.webbrowser.open")
def test_make_neuronpedia_list_with_features_without_browser(
    mock_webbrowser_open: MagicMock,
    mock_requests_post: MagicMock,
):
    # Mock successful response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "url": "https://neuronpedia.org/list/test-list-id",
    }
    mock_requests_post.return_value = mock_response

    features = [
        NeuronpediaFeature(
            modelId="gpt2-small",
            layer=0,
            dataset="res-jb",
            feature=10,
        )
    ]

    # open_browser=False only skips opening the page; the list was still created
    # successfully, so its URL must be returned rather than raised as an error.
    result_url = make_neuronpedia_list_with_features(
        api_key="test_api_key",
        list_name="Test List",
        features=features,
        open_browser=False,
    )

    mock_webbrowser_open.assert_not_called()
    assert result_url == "https://neuronpedia.org/list/test-list-id"


@patch("sae_lens.analysis.neuronpedia_integration.requests.post")
@patch("sae_lens.analysis.neuronpedia_integration.webbrowser.open")
def test_make_neuronpedia_list_with_features_minimal_features(
    _mock_webbrowser_open: MagicMock,
    mock_requests_post: MagicMock,
):
    # Mock successful response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "url": "https://neuronpedia.org/list/test-list-id",
    }
    mock_requests_post.return_value = mock_response

    # Create feature without description
    features = [
        NeuronpediaFeature(
            modelId="gpt2-small",
            layer=0,
            dataset="res-jb",
            feature=10,
        )
    ]

    # Call the function with minimal parameters
    make_neuronpedia_list_with_features(
        api_key="test_api_key",
        list_name="Test List",
        features=features,
    )

    # Verify the POST request was made with correct parameters
    expected_body = {
        "name": "Test List",
        "description": None,
        "features": [
            {
                "modelId": "gpt2-small",
                "layer": "0-res-jb",
                "index": 10,
                "description": "",  # Default empty description
            }
        ],
    }

    mock_requests_post.assert_called_once_with(
        f"{NEURONPEDIA_DOMAIN}/api/list/new-with-features",
        json=expected_body,
        headers={"x-api-key": "test_api_key"},
    )


@patch("sae_lens.analysis.neuronpedia_integration.requests.post")
@patch("sae_lens.analysis.neuronpedia_integration.webbrowser.open")
def test_make_neuronpedia_list_with_features_error_response(
    _mock_webbrowser_open: MagicMock,
    mock_requests_post: MagicMock,
):
    # Mock error response (missing 'url' field)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "success": False,
        "message": "API key is invalid",
    }
    mock_requests_post.return_value = mock_response

    features = [
        NeuronpediaFeature(
            modelId="gpt2-small",
            layer=0,
            dataset="res-jb",
            feature=10,
        )
    ]

    # Call the function and expect an exception
    with pytest.raises(Exception) as exc_info:
        make_neuronpedia_list_with_features(
            api_key="invalid_api_key",
            list_name="Test List",
            features=features,
        )

    assert "Error in creating list: API key is invalid" in str(exc_info.value)


@patch("sae_lens.analysis.neuronpedia_integration.requests.post")
@patch("sae_lens.analysis.neuronpedia_integration.webbrowser.open")
def test_make_neuronpedia_list_with_features_error_response_no_message(
    _mock_webbrowser_open: MagicMock,
    mock_requests_post: MagicMock,
):
    # Mock error response (missing 'url' field and no 'message' field)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "success": False,
        # No 'message' field - this will cause a KeyError
    }
    mock_requests_post.return_value = mock_response

    features = [
        NeuronpediaFeature(
            modelId="gpt2-small",
            layer=0,
            dataset="res-jb",
            feature=10,
        )
    ]

    # Call the function and expect a KeyError (due to missing 'message' field)
    with pytest.raises(KeyError):
        make_neuronpedia_list_with_features(
            api_key="invalid_api_key",
            list_name="Test List",
            features=features,
        )


def test_NanAndInfReplacer():
    assert NanAndInfReplacer("NaN") == 0
    assert NanAndInfReplacer("Infinity") == 9999
    assert NanAndInfReplacer("-Infinity") == -9999
    assert NanAndInfReplacer("123") == 0
    assert NanAndInfReplacer("test") == 0
