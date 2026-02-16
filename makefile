format:
	poetry run ruff format .
	poetry run ruff check --fix-only .

check-format:
	poetry run ruff check .
	poetry run ruff format --check .

check-type:
	poetry run pyright .

test:
	poetry run pytest -v --cov=sae_lens/ --cov-report=term-missing --cov-branch tests

check-ci:
	make check-format
	make check-type
	make test

docstring-coverage:
	poetry run docstr-coverage sae_lens --skip-file-doc

docs-serve:
	poetry run mkdocs serve --livereload