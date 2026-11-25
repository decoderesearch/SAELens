import json
import urllib.parse
import webbrowser
from typing import Any

import requests
from dotenv import load_dotenv

from sae_lens import SAE, logger

NEURONPEDIA_DOMAIN = "https://neuronpedia.org"

# Constants for replacing NaNs and Infs in outputs
POSITIVE_INF_REPLACEMENT = 9999
NEGATIVE_INF_REPLACEMENT = -9999
NAN_REPLACEMENT = 0
OTHER_INVALID_REPLACEMENT = -99999

# Pick up OPENAI_API_KEY from environment variable
load_dotenv()


def NanAndInfReplacer(value: str):
    """Replace NaNs and Infs in outputs."""
    replacements = {
        "-Infinity": NEGATIVE_INF_REPLACEMENT,
        "Infinity": POSITIVE_INF_REPLACEMENT,
        "NaN": NAN_REPLACEMENT,
    }
    if value in replacements:
        replaced_value = replacements[value]
        return float(replaced_value)
    return NAN_REPLACEMENT


def open_neuronpedia_feature_dashboard(sae: SAE[Any], index: int):
    sae_id = sae.cfg.metadata.neuronpedia_id
    if sae_id is None:
        logger.warning(
            "SAE does not have a Neuronpedia ID. Either dashboards for this SAE do not exist (yet) on Neuronpedia, or the SAE was not loaded via the from_pretrained method"
        )
    else:
        url = f"{NEURONPEDIA_DOMAIN}/{sae_id}/{index}"
        webbrowser.open(url)


def get_neuronpedia_quick_list(
    sae: SAE[Any],
    features: list[int],
    name: str = "temporary_list",
):
    sae_id = sae.cfg.metadata.neuronpedia_id
    if sae_id is None:
        logger.warning(
            "SAE does not have a Neuronpedia ID. Either dashboards for this SAE do not exist (yet) on Neuronpedia, or the SAE was not loaded via the from_pretrained method"
        )
    assert sae_id is not None

    url = NEURONPEDIA_DOMAIN + "/quick-list/"
    name = urllib.parse.quote(name)
    url = url + "?name=" + name
    list_feature = [
        {
            "modelId": sae.cfg.metadata.model_name,
            "layer": sae_id.split("/")[1],
            "index": str(feature),
        }
        for feature in features
    ]
    url = url + "&features=" + urllib.parse.quote(json.dumps(list_feature))
    webbrowser.open(url)

    return url


def get_neuronpedia_feature(
    feature: int, layer: int, model: str = "gpt2-small", dataset: str = "res-jb"
) -> dict[str, Any]:
    """Fetch a feature from Neuronpedia API."""
    url = f"{NEURONPEDIA_DOMAIN}/api/feature/{model}/{layer}-{dataset}/{feature}"
    result = requests.get(url).json()
    result["index"] = int(result["index"])
    return result


class NeuronpediaActivation:
    """Represents an activation from Neuronpedia."""

    def __init__(self, id: str, tokens: list[str], act_values: list[float]):
        self.id = id
        self.tokens = tokens
        self.act_values = act_values


class NeuronpediaFeature:
    """Represents a feature from Neuronpedia."""

    def __init__(
        self,
        modelId: str,
        layer: int,
        dataset: str,
        feature: int,
        description: str = "",
        activations: list[NeuronpediaActivation] | None = None,
        autointerp_explanation: str = "",
        autointerp_explanation_score: float = 0.0,
    ):
        self.modelId = modelId
        self.layer = layer
        self.dataset = dataset
        self.feature = feature
        self.description = description
        self.activations = activations
        self.autointerp_explanation = autointerp_explanation
        self.autointerp_explanation_score = autointerp_explanation_score

    def has_activating_text(self) -> bool:
        """Check if the feature has activating text."""
        if self.activations is None:
            return False
        return any(max(activation.act_values) > 0 for activation in self.activations)


def make_neuronpedia_list_with_features(
    api_key: str,
    list_name: str,
    features: list[NeuronpediaFeature],
    list_description: str | None = None,
    open_browser: bool = True,
):
    url = NEURONPEDIA_DOMAIN + "/api/list/new-with-features"

    # make POST json request with body
    body = {
        "name": list_name,
        "description": list_description,
        "features": [
            {
                "modelId": feature.modelId,
                "layer": f"{feature.layer}-{feature.dataset}",
                "index": feature.feature,
                "description": feature.description,
            }
            for feature in features
        ],
    }
    response = requests.post(url, json=body, headers={"x-api-key": api_key})
    result = response.json()

    if "url" in result:
        if open_browser:
            webbrowser.open(result["url"])
        return result["url"]
    raise Exception("Error in creating list: " + result["message"])


def test_key(api_key: str):
    """Test the validity of the Neuronpedia API key."""
    url = f"{NEURONPEDIA_DOMAIN}/api/test"
    body = {"apiKey": api_key}
    response = requests.post(url, json=body)
    if response.status_code != 200:
        raise Exception("Neuronpedia API key is not valid.")
