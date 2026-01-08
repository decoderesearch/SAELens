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
