# type: ignore
# ruff: noqa: T201
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from textwrap import dedent

import pandas as pd
import yaml
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from sae_lens import SAEConfig
from sae_lens.loading.pretrained_sae_loaders import (
    load_sae_config_from_huggingface,
)

MAX_WORKERS = 4

INCLUDED_CFG = [
    "id",
    "neuronpedia",
    "hook_name",
    "d_sae",
]

CACHE_DIR = Path(
    os.environ.get("SAE_CONFIG_CACHE_DIR", "docs/.sae_config_cache")
).expanduser()
OUTPUT_DIR = Path("docs/pretrained_saes")

# Modal HTML that gets added to each page
MODAL_HTML = dedent(
    """
    <div id="codeModal" class="saetable-modal">
        <div class="saetable-modalContent">
            <span class="saetable-close" onclick="SaeTable.closeCode()">&times;</span>
            <pre><code id="codeContent" onclick="SaeTable.selectCode(this)"></code></pre>
            <button onclick="SaeTable.copyCode()" class="saetable-copyButton">Copy Code</button>
        </div>
    </div>
    """
)


def get_cached_config(release: str, sae_id: str) -> dict | None:
    """Load config from cache if it exists."""
    cache_file = CACHE_DIR / f"{release}" / f"{sae_id.replace('/', '_')}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def save_config_to_cache(release: str, sae_id: str, config: dict) -> None:
    """Save config to cache."""
    cache_dir = CACHE_DIR / f"{release}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{sae_id.replace('/', '_')}.json"
    with open(cache_file, "w") as f:
        json.dump(config, f, indent=2)


def load_sae_config_cached(release: str, sae_id: str) -> dict:
    """Load SAE config, using cache if available."""
    cached = get_cached_config(release, sae_id)
    if cached is not None:
        return cached
    config = load_sae_config_from_huggingface(release, sae_id=sae_id)
    save_config_to_cache(release, sae_id, config)
    return config


def model_to_filename(model: str) -> str:
    """Convert model name to a valid filename."""
    return model.replace("/", "_").replace(" ", "_").lower()


def model_to_display_name(model: str) -> str:
    """Convert model name to display name (lowercase, remove org prefix)."""
    if "/" in model:
        model = model.split("/")[-1]
    return model.lower()


def on_pre_build(config):  # noqa: ARG001
    print("Generating SAE table...")
    generate_sae_table()
    print("SAE table generation complete.")


# hacky fix until 1password fixes its chrome plugin:
# https://www.1password.community/discussions/developers/1password-chrome-extension-is-incorrectly-manipulating--blocks/165639
def on_page_content(html, page, config, files):  # noqa: ARG001
    """Strip 'language-*' classes from divs to prevent 1Password's Prism.js from
    overriding Pygments syntax highlighting.

    See: https://www.linkedin.com/posts/rinehart_if-youre-highlighting-code-in-html-pages-activity-7407447059852582912-4LaY/
    """
    soup = BeautifulSoup(html, "html.parser")

    for div in soup.find_all("div"):
        if div.get("class"):
            # Build a new class list without any "language-*" classes
            updated_classes = [
                cls for cls in div["class"] if not cls.startswith("language")
            ]
            if updated_classes:
                div["class"] = updated_classes
            else:
                del div["class"]

    return str(soup)


def style_hook_name(row: pd.Series) -> str:
    """Format hook_name column to show both TransformerLens and HuggingFace hooks."""
    hook_name = row.get("hook_name", "")
    hf_hook_name = row.get("hf_hook_name")

    tl_label = '<span class="saetable-hookLabel" data-tooltip="Hook point for TransformerLens">tl</span>'
    hf_label = '<span class="saetable-hookLabel" data-tooltip="Hook point for Hugging Face / NNsight">hf</span>'

    if hf_hook_name:
        return (
            f'<span class="saetable-hookRow">{hook_name} {tl_label}</span>'
            f'<span class="saetable-hookRow">{hf_hook_name} {hf_label}</span>'
        )
    return f'<span class="saetable-hookRow">{hook_name} {tl_label}</span>'


def generate_release_content(
    release: str, model_info: dict, executor: ThreadPoolExecutor
) -> str:
    """Generate markdown content for a single release."""
    content = ""
    repo_link = f"https://huggingface.co/{model_info['repo_id']}"
    content += f"### [{release}]({repo_link})\n\n"
    content += f"- **Huggingface Repo**: {model_info['repo_id']}\n"

    if "links" in model_info:
        content += "- **Additional Links**:\n"
        for link_type, url in model_info["links"].items():
            content += f"    - [{link_type.capitalize()}]({url})\n"

    content += "\n"

    # Load all configs in parallel
    sae_list = model_info["saes"]
    future_to_info = {
        executor.submit(load_sae_config_cached, release, info["id"]): info
        for info in sae_list
    }

    for future in tqdm(
        as_completed(future_to_info),
        total=len(future_to_info),
        desc=f"  {release}",
        leave=False,
    ):
        info = future_to_info[future]
        raw_cfg = future.result()
        cfg = SAEConfig.from_dict(raw_cfg).to_dict()

        if "neuronpedia" not in info:
            info["neuronpedia"] = None

        if "metadata" in cfg:
            info.update(cfg["metadata"])

        info.update(cfg)

    df = pd.DataFrame(model_info["saes"])

    def style_id(val):
        return f"<div>{val}</div><a class=\"saetable-loadSaeId\" onclick=\"SaeTable.showCode('{release}', '{val}')\">Load this SAE</a>"

    df["id"] = df["id"].apply(style_id)
    df["hook_name"] = df.apply(style_hook_name, axis=1)

    # Determine which columns to include
    columns = list(INCLUDED_CFG)
    has_neuronpedia = df["neuronpedia"].notna().any()
    if not has_neuronpedia:
        columns.remove("neuronpedia")

    df = df[columns]
    table = df.to_markdown(index=False)
    content += table + "\n\n"

    return content


def generate_model_page(
    model: str, releases: list[tuple[str, dict]], executor: ThreadPoolExecutor
) -> str:
    """Generate a full page for a model with all its releases."""
    display_name = model_to_display_name(model)
    content = f"# {display_name}\n\n"
    content += (
        f"This page lists all pretrained SAEs available for **{display_name}**.\n\n"
    )

    for release, model_info in sorted(releases, key=lambda x: x[0].lower()):
        content += generate_release_content(release, model_info, executor)

    content += MODAL_HTML
    return content


def generate_index_page(models_with_counts: list[tuple[str, int, int]]) -> str:
    """Generate the index page linking to all model pages."""
    content = "# Pretrained SAEs\n\n"
    content += "This is a list of SAEs importable from the SAELens package. Click on a model to see all available SAEs.\n\n"
    content += "*These pages contain the contents of `sae_lens/pretrained_saes.yaml` in Markdown*\n\n"

    content += "## Models\n\n"
    content += "| Model | Releases | Total SAEs |\n"
    content += "|-------|----------|------------|\n"

    for model, num_releases, total_saes in sorted(
        models_with_counts, key=lambda x: model_to_display_name(x[0])
    ):
        filename = model_to_filename(model)
        display_name = model_to_display_name(model)
        content += (
            f"| [{display_name}]({filename}.md) | {num_releases} | {total_saes} |\n"
        )

    content += "\n"
    return content


def generate_sae_table() -> None:
    # Read the YAML file
    yaml_path = Path("sae_lens/pretrained_saes.yaml")
    with open(yaml_path) as file:
        data = yaml.safe_load(file)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group releases by model
    model_releases: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for release, model_info in data.items():
        model = model_info["model"]
        model_releases[model].append((release, model_info))

    # Generate a page for each model
    models_with_counts: list[tuple[str, int, int]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for model, releases in tqdm(model_releases.items(), desc="Models"):
            filename = model_to_filename(model)
            page_content = generate_model_page(model, releases, executor)

            output_path = OUTPUT_DIR / f"{filename}.md"
            with open(output_path, "w") as f:
                f.write(page_content)

            total_saes = sum(len(model_info["saes"]) for _, model_info in releases)
            models_with_counts.append((model, len(releases), total_saes))

    # Generate index page
    index_content = generate_index_page(models_with_counts)
    index_path = OUTPUT_DIR / "index.md"
    with open(index_path, "w") as f:
        f.write(index_content)

    print(f"Generated {len(model_releases)} model pages in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_sae_table()
