format:
	uv run ruff format .
	uv run ruff check --fix-only .

check-format:
	uv run ruff check .
	uv run ruff format --check .

check-type:
	uv run pyright .

test:
	uv run pytest -v --cov=sae_lens/ --cov-report=term-missing --cov-branch tests

check-ci:
	make check-format
	make check-type
	make test

docstring-coverage:
	uv run docstr-coverage sae_lens --skip-file-doc

docs-serve:
	uv run mkdocs serve --livereload