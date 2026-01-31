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
- We do not want tons of tests asserting trivial things. Focus on a few tests for functional correctness instead. A test should fail if the underlying feature being tested is wrong.
- Never set random seeds in tests. If you want to check something random that runs fast, generate large number of samples and check the statistics.
- For statistical tests, don't be afraid to use large number of samples to allow for tight bounds. Please make bounds as tight as possible too.
- Do not add doc comments to test functions. The test name should be self-explanatory.
- Never place imports inside of functions. Always import at the top of the file.
- Use parentheses for tensor shapes in docs and messages, e.g. (batch_size, num_features)
- Never disable type checking for an entire file, even in tests.

### Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.