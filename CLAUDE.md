# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test/Lint Commands

- Install dependencies: `poetry install`
- Run all tests: `poetry run pytest`
- Run single test: `poetry run pytest tests/path/to/test_file.py::test_function_name`
- Run with verbose output: `poetry run pytest -v`
- Linting/formatting: `poetry run ruff check .` and `poetry run ruff format .`
- Type checking: `poetry run pyright`
- Pre-commit hook: `poetry run pre-commit install`
- Run a Python file as a script: `poetry run python -m sae_lens.path.to.file`

## Guidelines

- Do not use `Example:` in docstrings.
- If you use a markdown list in docstrings, you must put a blank line before the start of the list.
- Each test file should mirror a file in the `sae_lens` package.
- When writing tests, focus on testing the core logic and meat of the code rather than just superficial things like tensor shapes.
- Make sure to have some tests with simple inputs that can only pass if the code is truly correct, rather than superficially correct.
- Do not relax assertion tolerances in tests unless absolutely necessary. Never relax tolerances to mask an underlying bug. Ask for input if you are unsure.
- Never set random seeds in tests. If you want to check something random that runs fast, generate large number of samples and check the statistics.
- For statistical tests, don't be afraid to use large number of samples to allow for tight bounds. Please make bounds as tight as possible too.
- Do not add doc comments to test functions. The test name should be self-explanatory.
- Never place imports inside of functions. Always import at the top of the file.
- Use parentheses for tensor shapes in docs and messages, e.g. (batch_size, num_features)
