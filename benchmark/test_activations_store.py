from time import perf_counter
from typing import Any, cast

from datasets import load_dataset
from transformer_lens import HookedTransformer

from sae_lens.config import PretokenizeRunnerConfig
from sae_lens.pretokenize_runner import pretokenize_dataset
from sae_lens.training.activations_store import ActivationsStore
from tests.helpers import build_runner_cfg


# The way to run this with this command:
# uv run pytest -k test_benchmark_activations_store_get_batch_tokens_pretokenized_vs_raw -ss
def test_benchmark_activations_store_get_batch_tokens_pretokenized_vs_raw():
    model = HookedTransformer.from_pretrained("gpt2")
    tokenizer = model.tokenizer
    assert tokenizer is not None

    cfg = build_runner_cfg(
        model_name="gpt2",
        dataset_path="NeelNanda/c4-10k",
        context_size=512,
    )

    dataset = load_dataset(cfg.dataset_path, split="train")
    pretokenize_cfg = PretokenizeRunnerConfig(
        tokenizer_name="gpt2",
        dataset_path=cfg.dataset_path,
        context_size=cfg.context_size,
        shuffle=False,
        num_proc=1,
        pretokenize_batch_size=None,
    )
    tokenized_dataset = pretokenize_dataset(
        cast(Any, dataset), tokenizer, pretokenize_cfg
    )

    text_dataset_store = ActivationsStore.from_config(
        model, cfg, override_dataset=dataset
    )
    pretokenized_dataset_store = ActivationsStore.from_config(
        model, cfg, override_dataset=tokenized_dataset
    )

    text_start_time = perf_counter()
    text_batch_toks = [text_dataset_store.get_batch_tokens(50) for _ in range(100)]
    text_duration = perf_counter() - text_start_time

    pretokenized_start_time = perf_counter()
    pretokenized_batch_toks = [
        pretokenized_dataset_store.get_batch_tokens(50) for _ in range(100)
    ]
    pretokenized_duration = perf_counter() - pretokenized_start_time

    print(f"get_batch_tokens() duration with text dataset: {text_duration}")
    print(
        f"get_batch_tokens() duration with pretokenized dataset: {pretokenized_duration}"
    )

    for text_toks, pretokenized_toks in zip(text_batch_toks, pretokenized_batch_toks):
        assert text_toks.tolist() == pretokenized_toks.tolist()
    assert pretokenized_duration < text_duration
