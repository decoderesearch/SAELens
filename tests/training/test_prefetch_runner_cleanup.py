import threading
from types import SimpleNamespace
from typing import Any

import pytest

import sae_lens.llm_sae_training_runner as llm_runner_module
import sae_lens.multi_sae_training_runner as multi_runner_module
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.multi_sae_training_runner import MultiSAETrainingRunner
from sae_lens.training.prefetch import PrefetchingIterator


class _Source:
    def __init__(self) -> None:
        self.calls = 0
        self.second_item_requested = threading.Event()
        self.close_calls = 0

    def __iter__(self) -> "_Source":
        return self

    def __next__(self) -> int:
        self.calls += 1
        if self.calls == 2:
            self.second_item_requested.set()
        return self.calls

    def close(self) -> None:
        self.close_calls += 1


def _make_config(prefetch: bool) -> SimpleNamespace:
    return SimpleNamespace(
        logger=SimpleNamespace(log_to_wandb=False),
        prefetch_llm_batches=prefetch,
        resume_from_checkpoint=None,
        output_path=None,
        hook_names_per_sae={},
        to_sae_trainer_config=lambda: object(),
    )


def _make_runner(runner_kind: str, source: _Source, prefetch: bool):
    runner_type = (
        LanguageModelSAETrainingRunner
        if runner_kind == "single"
        else MultiSAETrainingRunner
    )
    runner: Any = object.__new__(runner_type)
    runner.cfg = _make_config(prefetch)
    runner.evaluator = object()
    runner._set_sae_metadata = lambda: None

    if runner_kind == "single":
        runner.sae = object()
        runner.activations_store = source
        runner.save_checkpoint = lambda: None
        runner._compile_sae_if_needed = lambda: None
        runner.run_trainer_with_interruption_handling = lambda trainer: trainer.fit()
    else:

        class _Store:
            def get_multi_hook_data_loader(self) -> _Source:
                return source

        runner.activations_store = _Store()
        runner.saes = {}
        runner._save_runner_checkpoint_files = lambda: None
        runner._run_with_interruption_handling = lambda trainer: trainer.fit()

    return runner


@pytest.mark.parametrize("runner_kind", ["single", "multi"])
def test_runner_closes_owned_prefetcher_when_fit_raises(
    monkeypatch: pytest.MonkeyPatch, runner_kind: str
):
    source = _Source()
    runner = _make_runner(runner_kind, source, prefetch=True)
    captured_providers: list[Any] = []

    class _FailingTrainer:
        def __init__(self, **kwargs: object) -> None:
            captured_providers.append(kwargs["data_provider"])

        def fit(self) -> None:
            assert source.second_item_requested.wait(timeout=1)
            raise RuntimeError("training failed")

    if runner_kind == "single":
        monkeypatch.setattr(llm_runner_module, "SAETrainer", _FailingTrainer)
    else:
        monkeypatch.setattr(multi_runner_module, "MultiSAETrainer", _FailingTrainer)

    with pytest.raises(RuntimeError, match="training failed"):
        runner.run()

    assert len(captured_providers) == 1
    provider = captured_providers[0]
    assert isinstance(provider, PrefetchingIterator)
    assert provider._closed.is_set()
    assert not provider._thread.is_alive()
    assert not provider._queue


@pytest.mark.parametrize("runner_kind", ["single", "multi"])
def test_runner_closes_owned_prefetcher_after_successful_fit(
    monkeypatch: pytest.MonkeyPatch, runner_kind: str
):
    source = _Source()
    runner = _make_runner(runner_kind, source, prefetch=True)
    captured_providers: list[Any] = []
    expected_result: Any = object() if runner_kind == "single" else {"sae": object()}

    class _SuccessfulTrainer:
        def __init__(self, **kwargs: object) -> None:
            captured_providers.append(kwargs["data_provider"])

        def fit(self) -> Any:
            assert source.second_item_requested.wait(timeout=1)
            return expected_result

    if runner_kind == "single":
        monkeypatch.setattr(llm_runner_module, "SAETrainer", _SuccessfulTrainer)
    else:
        monkeypatch.setattr(multi_runner_module, "MultiSAETrainer", _SuccessfulTrainer)

    assert runner.run() is expected_result

    assert len(captured_providers) == 1
    provider = captured_providers[0]
    assert isinstance(provider, PrefetchingIterator)
    assert provider._closed.is_set()
    assert not provider._thread.is_alive()
    assert not provider._queue


@pytest.mark.parametrize("runner_kind", ["single", "multi"])
def test_runner_does_not_close_unowned_source(
    monkeypatch: pytest.MonkeyPatch, runner_kind: str
):
    source = _Source()
    runner = _make_runner(runner_kind, source, prefetch=False)
    captured_providers: list[Any] = []

    class _FailingTrainer:
        def __init__(self, **kwargs: object) -> None:
            captured_providers.append(kwargs["data_provider"])

        def fit(self) -> None:
            raise RuntimeError("training failed")

    if runner_kind == "single":
        monkeypatch.setattr(llm_runner_module, "SAETrainer", _FailingTrainer)
    else:
        monkeypatch.setattr(multi_runner_module, "MultiSAETrainer", _FailingTrainer)

    with pytest.raises(RuntimeError, match="training failed"):
        runner.run()

    assert captured_providers == [source]
    assert source.close_calls == 0
