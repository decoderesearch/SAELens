import os
import re
import string
from typing import TYPE_CHECKING

import nltk
import numpy as np
import pandas as pd
import plotly_express as px
import torch
from babe import UsNames

from sae_lens import logger

if TYPE_CHECKING:
    # HookedTransformer was removed in transformer-lens v4
    from transformer_lens import HookedTransformer


def get_enrichment_df(
    projections: torch.Tensor,
    features: list[int],
    gene_sets_selected: dict[str, set[int]],
):
    gene_sets_token_ids_padded = pad_gene_sets(gene_sets_selected)
    gene_sets_token_ids_tensor = torch.tensor(list(gene_sets_token_ids_padded.values()))
    enrichment_scores = calculate_batch_enrichment_scores(
        projections[features], gene_sets_token_ids_tensor
    )
    return pd.DataFrame(
        enrichment_scores.numpy(),
        index=gene_sets_selected.keys(),  # type: ignore
        columns=features,  # type: ignore
    )


def calculate_batch_enrichment_scores(scores: torch.Tensor, index_lists: torch.Tensor):
    """
    # features with large skew
    features_top_800_by_prediction_skew = W_U_stats_df_dec["skewness"].sort_values(ascending=False).head(12000).index
    gene_sets_index = ["starts_with_space", "starts_with_capi", "all_digits", "is_punctuation"]
    gene_sets_temp = {k:v for k,v in gene_sets_token_ids_padded.items() if k in gene_sets_index}

    gene_sets_token_ids_tensor = torch.tensor([value for value in gene_sets_temp.values()])
    gene_sets_token_ids_tensor.shape
    gene_sets = gene_sets_token_ids_tensor


    enrichment_scores = calculate_batch_enrichment_scores(dec_projection_onto_W_U[features_top_800_by_prediction_skew], gene_sets_token_ids_tensor)
    df_enrichment_scores = pd.DataFrame(enrichment_scores.numpy(), index=gene_sets_index, columns=features_top_800_by_prediction_skew)
    """
    n_sets, _ = index_lists.shape
    n_scores, vocab_size = scores.shape

    # Ensure scores and index_lists are on the same device
    scores = scores.to(index_lists.device)

    # Create a mask for valid indices (ignore padding)
    valid_mask = index_lists != -1  # Assuming -1 is used for padding

    # Initialize a mask for all scores
    score_mask = torch.zeros(
        n_sets, vocab_size, device=index_lists.device, dtype=torch.bool
    )

    # Set true for valid indices in score_mask
    for i in range(n_sets):
        score_mask[i, index_lists[i][valid_mask[i]]] = True

    # Sort scores along each row
    _, sorted_indices = scores.sort(dim=1, descending=True)

    # Create a mask to identify hits within the sorted indices
    hits = (
        score_mask.unsqueeze(1)
        .expand(-1, n_scores, -1)
        .gather(2, sorted_indices.unsqueeze(0).expand(n_sets, -1, -1))
    )

    # Calculate hit increment and miss decrement dynamically for each list
    list_sizes = valid_mask.sum(dim=1).float()  # Actual sizes of each list
    hit_increment = (1.0 / list_sizes).view(-1, 1, 1)  # Reshape for broadcasting
    miss_decrement = (1.0 / (vocab_size - list_sizes)).view(
        -1, 1, 1
    )  # Reshape for broadcasting

    # Ensure hit_increment and miss_decrement are broadcastable to the shape of hits
    # Apply hit increment or miss decrement based on hits
    running_sums = torch.where(hits, hit_increment, -miss_decrement).cumsum(dim=2)
    return running_sums.abs().max(dim=2).values


def manhattan_plot_enrichment_scores(
    df_enrichment_scores: pd.DataFrame, label_threshold: float = 1.0, top_n: int = 3
):
    tmp_df = df_enrichment_scores.apply(lambda x: -1 * np.log(1 - x))

    # wide to long format
    tmp_df = tmp_df.reset_index().melt(
        id_vars="index", var_name="Feature", value_name="Enrichment Score"
    )
    tmp_df.rename(columns={"index": "gene_set"}, inplace=True)
    tmp_df.reset_index(drop=True, inplace=True)

    fig = px.scatter(
        tmp_df,
        x="Feature",
        y="Enrichment Score",
        color=tmp_df.gene_set,
        facet_col=tmp_df.gene_set,
        labels={"index": "", "value": "Enrichment Score", "variable": "Token Set"},
        width=1400,
        height=500,
    )

    fig.update_traces(marker={"size": 3})

    #  only annotate the top n points in each gene set
    annotation_df = (
        tmp_df.groupby("gene_set")
        .apply(lambda x: x.nlargest(top_n, "Enrichment Score"))
        .reset_index(drop=True)
    )
    gene_set_to_subplot = {
        gene_set: i + 1 for i, gene_set in enumerate(tmp_df["gene_set"].unique())
    }

    # Annotate all points above the label_threshold
    for _, row in annotation_df.iterrows():
        if row["Enrichment Score"] > label_threshold:
            # Find the subplot index
            subplot_index = gene_set_to_subplot[row["gene_set"]]
            # Add annotation at the position of the point that exceeds the threshold
            fig.add_annotation(
                x=row["Feature"],
                y=row["Enrichment Score"],
                text=row["Feature"],  # Or any other text you want to display
                showarrow=False,
                arrowhead=1,
                xref=f"x{subplot_index}",  # Refer to the correct x-axis
                yref=f"y{subplot_index}",  # Refer to the correct y-axis
                ax=20,  # Adjusts the x position of the arrow (try changing this if needed)
                ay=-30,  # Adjusts the y position of the arrow (try changing this if needed)
                yshift=15,
            )

    # relabel facet cols to remove gene_set
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # hide legend
    fig.update_layout(showlegend=False)

    # increase font size
    fig.update_layout(font={"size": 16})

    return fig


def plot_top_k_feature_projections_by_token_and_category(
    gene_sets_selected: dict[str, set[int]],
    df_enrichment_scores: pd.DataFrame,
    category: str,
    model: "HookedTransformer",
    dec_projection_onto_W_U: torch.Tensor,
    k: int = 5,
    projection_onto: str = "W_U",
    features: list[int] | None = None,
    log_y: bool = True,
    histnorm: str | None = None,
):
    if not os.path.exists("es_plots"):
        os.makedirs("es_plots")

    enrichment_scores = df_enrichment_scores.filter(like=category, axis=0).T

    if features is None:
        features = (
            enrichment_scores.sort_values(category, ascending=False).head(k).index
        ).to_list()

    # scores = enrichment_scores[category][features]
    scores = enrichment_scores[category].loc[features]
    logger.debug(scores)
    tokens_list = [model.to_single_str_token(i) for i in list(range(model.cfg.d_vocab))]

    logger.debug(features)
    feature_logit_scores = pd.DataFrame(
        dec_projection_onto_W_U[features].numpy(),
        index=features,  # type: ignore
    ).T
    feature_logit_scores["token"] = tokens_list
    feature_logit_scores[category] = [
        i in gene_sets_selected[category] for i in list(range(model.cfg.d_vocab))
    ]

    # display(feature_)
    logger.debug(category)
    for feature, score in zip(features, scores):  # type: ignore
        logger.debug(feature)
        score = -1 * np.log(1 - score)  # convert to enrichment score
        fig = px.histogram(
            feature_logit_scores,
            x=feature,
            color=category,
            title=f"W_dec_{feature}, {projection_onto}, {category}: {score:2.2f}",
            barmode="overlay",
            histnorm=histnorm,
            log_y=log_y,
            hover_name="token",
            marginal="box",
            width=800,
            height=400,
            labels={f"{feature}": f"W_U W_dec[{feature}]"},
        )

        # increase the font size
        fig.update_layout(font={"size": 16})
        fig.show()
        fig.write_html(
            f"es_plots/{feature}_projection_onto_{projection_onto}_by_{category}.html"
        )


def pad_gene_sets(gene_sets_token_ids: dict[str, set[int]]) -> dict[str, list[int]]:
    for k, v in gene_sets_token_ids.items():
        gene_sets_token_ids[k] = list(v)  # type: ignore
    max_len = max([len(v) for v in gene_sets_token_ids.values()])

    # pad with -1's to max length
    return {
        key: value + [-1] * (max_len - len(value))  # type: ignore
        for key, value in gene_sets_token_ids.items()
    }


def get_baby_name_sets(vocab: dict[str, int], k: int = 300) -> dict[str, list[int]]:
    d = UsNames()
    baby_df = d.data
    boy_names = baby_df[baby_df.gender == "M"].name.value_counts().head(k).index
    girl_names = baby_df[baby_df.gender == "F"].name.value_counts().head(k).index

    # prepend spaces
    boy_names = [f"Ġ{name}" for name in boy_names]
    girl_names = [f"Ġ{name}" for name in girl_names]

    # get all the tokens in the tokenizer that are in each of thes
    names = {"boy_names": [], "girl_names": []}
    for token, id in vocab.items():
        if token in boy_names:
            names["boy_names"].append(id)
        elif token in girl_names:
            names["girl_names"].append(id)

    return names


def get_letter_gene_sets(vocab: dict[str, int]) -> dict[str, set[int]]:
    letters = string.ascii_lowercase
    gene_sets = {letter: set() for letter in letters}
    for token, id in vocab.items():
        clean_token = token.strip("Ġ")  # Remove leading 'Ġ'
        if (
            clean_token.isalpha() and clean_token[0].lower() in letters
        ):  # Check if the first character is in letters
            gene_sets[clean_token[0].lower()].add(id)

    return gene_sets


def generate_pos_sets(vocab: dict[str, int]) -> dict[str, set[int]]:
    # tagged_tokens = nltk.pos_tag([i.strip("Ġ") for i in list(vocab.keys())])
    # tagged_vocab = {word: tag for word, tag in tagged_tokens}
    pos_sets = {}
    for token, id in vocab.items():
        clean_token = token.strip("Ġ")  # Remove leading 'Ġ'
        tagged_token = nltk.pos_tag([clean_token])
        tag = tagged_token[0][1]
        if f"nltk_pos_{tag}" not in pos_sets:
            pos_sets[f"nltk_pos_{tag}"] = set()
        pos_sets[f"nltk_pos_{tag}"].add(id)

    return pos_sets


def get_gene_set_from_regex(vocab: dict[str, int], pattern: str) -> set[int]:
    gene_set = set()
    for token, id in vocab.items():
        if re.match(pattern, token):
            gene_set.add(id)
    return gene_set


def get_test_gene_sets(model: "HookedTransformer") -> dict[str, set[int]]:
    colors = [
        "red",
        "blue",
        "yellow",  # Primary colors
        "green",
        "orange",
        "purple",  # Secondary colors
        "pink",
        "teal",
        "lavender",
        "maroon",
        "olive",
        "navy",
        "grey",  # Tertiary and common colors
        "black",
        "white",
        "brown",  # Basics
    ]

    negative_words = [
        "terrible",
        "awful",
        "horrible",
        "dreadful",
        "abysmal",
        "wretched",
        "dire",
        "appalling",
        "horrific",
        "disastrous",
        "ghastly",
        "hideous",
        "gruesome",
        "vile",
        "foul",
        "atrocious",
        "heinous",
        "abhorrent",
        "detestable",
        "loathsome",
        "repulsive",
        "repugnant",
        "disgusting",
        "revolting",
        "noxious",
        "offensive",
        "nauseating",
        "sickening",
        "distasteful",
        "unpleasant",
        "obnoxious",
        "odious",
        "unsavory",
        "unpalatable",
        "grim",
        "gloomy",
        "deplorable",
        "depressing",
        "despicable",
        "miserable",
        "pathetic",
        "pitiful",
        "lamentable",
        "direful",
        "tragic",
        "woeful",
        "painful",
        "harsh",
        "bitter",
    ]

    positive_words = [
        "wonderful",
        "amazing",
        "fabulous",
        "excellent",
        "fantastic",
        "brilliant",
        "awesome",
        "spectacular",
        "marvelous",
        "incredible",
        "superb",
        "magical",
        "delightful",
        "charming",
        "beautiful",
        "astonishing",
        "impressive",
        "stunning",
        "breathtaking",
        "admirable",
        "lovely",
        "pleasing",
        "enchanting",
        "exquisite",
        "radiant",
        "splendid",
        "glorious",
        "divine",
        "sublime",
        "heavenly",
        "idyllic",
        "blissful",
        "serene",
        "tranquil",
        "peaceful",
        "joyful",
        "ecstatic",
        "jubilant",
        "elated",
        "uplifting",
        "inspiring",
        "revitalizing",
        "refreshing",
        "invigorating",
        "energizing",
        "thrilling",
        "captivating",
        "enthralling",
        "enlightening",
    ]

    emotions = [
        "anger",
        "fear",
        "joy",
        "sadness",  # Basic emotions
        "trust",
        "disgust",
        "anticipation",
        "surprise",  # Complex emotions
        "love",
        "hate",
        "envy",
        "compassion",
        "pride",
        "shame",
        "guilt",
        "hope",
        "despair",  # Complex emotions
    ]

    boys_names = [
        "Michael",
        "James",
        "John",
        "Robert",
        "David",
        "William",
        "Joseph",
        "Charles",
        "Thomas",
        "Christopher",
    ]

    girls_names = [
        "Mary",
        "Patricia",
        "Jennifer",
        "Linda",
        "Elizabeth",
        "Barbara",
        "Susan",
        "Jessica",
        "Sarah",
        "Karen",
    ]

    capital_cities = [
        "Washington, D.C.",
        "Ottawa",
        "London",
        "Paris",
        "Berlin",
        "Tokyo",
        "Moscow",
        "Beijing",
        "Canberra",
        "New Delhi",
    ]

    countries = [
        "United States",
        "Canada",
        "United Kingdom",
        "France",
        "Germany",
        "Japan",
        "Russia",
        "China",
        "Australia",
        "India",
    ]

    neuroscience_terms = [
        "Neuron",
        "Synapse",
        "Axon",
        "Dendrite",
        "Neuroplasticity",
        "Cerebral cortex",
        "Neurotransmitter",
        "Myelin sheath",
        "Action potential",
        "Grey matter",
        "White matter",
        "Neurogenesis",
        "Neurotransmission",
        "Neurodegeneration",
        "Neuroinflammation",
        "Neurodevelopment",
        "Neuroimaging",
        "Neuropharmacology",
        "Neurophysiology",
        "Neuropsychology",
    ]
    neuroscience_terms = [i.lower() for i in neuroscience_terms]

    economics_terms = [
        "Supply and Demand",
        "Elasticity",
        "Gross Domestic Product (GDP)",
        "Inflation",
        "Monetary policy",
        "Fiscal policy",
        "Marginal utility",
        "Opportunity cost",
        "Equilibrium price",
        "Market efficiency",
        "Monopoly",
        "Oligopoly",
        "Monopolistic competition",
        "Perfect competition",
        "Economic surplus",
        "Consumer surplus",
        "Producer surplus",
        "Deadweight loss",
        "Economic rent",
        "Externality",
    ]
    economics_terms = [i.lower() for i in economics_terms]

    spanish_words = [
        "hola",
        "amor",
        "feliz",
        "casa",
        "familia",
        "gracias",
        "libro",
        "mañana",
        "noche",
        "amigo",
    ]

    french_words = [
        "bonjour",
        "amour",
        "heureux",
        "maison",
        "famille",
        "merci",
        "livre",
        "matin",
        "nuit",
        "ami",
    ]

    jewish_last_names = [
        "Bloom",
        "Levine",
        "Goldstein",
        "Cohen",
        "Katz",
        "Kaplan",
        "Adler",
        "Stein",
        "Weiss",
        "Stern",
        "Cohen",
        "Levi",
        "Katz",
        "Kahan",
        "Weiss",
        "Gross",
        "Friedman",
        "Kramer",
        "Grossman",
        "Zimmerman",
    ]

    ologies = [
        "Biology",
        "Ecology",
        "Psychology",
        "Sociology",
        "Geology",
        "Meteorology",
        "Zoology",
        "Botany",
        "Anthropology",
        "Astrology",
        "Astronomy",
        "Theology",
        "Philology",
        "Pharmacology",
        "Pathology",
        "Oceanology",
        "Toxicology",
        "Volcanology",
        "Entomology",
        "Paleontology",
        "Neurology",
        "Ethnology",
        "Criminology",
        "Seismology",
        "Cytology",
    ]

    gene_sets = {
        "1910's": [str(i) for i in range(1910, 1920)],
        "1920's": [str(i) for i in range(1920, 1930)],
        "1930's": [str(i) for i in range(1930, 1940)],
        "1940's": [str(i) for i in range(1940, 1950)],
        "1950's": [str(i) for i in range(1950, 1960)],
        "1960's": [str(i) for i in range(1960, 1970)],
        "1970's": [str(i) for i in range(1970, 1980)],
        "1980's": [str(i) for i in range(1980, 1990)],
        "1990's": [str(i) for i in range(1990, 2000)],
        "2000's": [str(i) for i in range(2000, 2010)],
        "2010's": [str(i) for i in range(2010, 2020)],
        "colors": colors,
        "positive_words": positive_words,
        "negative_words": negative_words,
        "emotions": emotions,
        "boys_names": boys_names,
        "girls_names": girls_names,
        "spanish_words": spanish_words,
        "french_words": french_words,
        "neuroscience_terms": neuroscience_terms,
        "economics_terms": economics_terms,
        "capital_cities": capital_cities,
        "countries": countries,
        "jewish_last_names": jewish_last_names,
        "ologies": ologies,
    }

    def convert_tokens_to_ids(
        list_of_strings: list[str], model: "HookedTransformer"
    ) -> set[int]:
        token_ids = [
            model.tokenizer.encode(f" {word}", add_special_tokens=False)  # type: ignore
            for word in list_of_strings
        ]
        token_ids = [item for sublist in token_ids for item in sublist]
        return set(token_ids)

    return {
        key: convert_tokens_to_ids(value, model) for key, value in gene_sets.items()
    }  # type: ignore
