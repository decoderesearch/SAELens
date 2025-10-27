# Contributing

Contributions are welcome! To get setup for development, follow the instructions below.

## Setup

Make sure you have [poetry](https://python-poetry.org/) installed, clone the repository, and install dependencies with:

```bash
git clone https://github.com/decoderesearch/SAELens.git # we recommend you make a fork for submitting PR's and clone that!
poetry lock # can take a while.
poetry install
make check-ci # validate the install
```

## Testing, Linting, and Formatting

This project uses [pytest](https://docs.pytest.org/en/stable/) for testing, [pyright](https://github.com/microsoft/pyright) for type-checking, and [Ruff](https://docs.astral.sh/ruff/) for formatting and linting.

If you add new code, it would be greatly appreciated if you could add tests in the `tests` directory. You can run the tests with:

```bash
make test
```

Before commiting, make sure you format the code with:

```bash
make format
```

Finally, run all CI checks locally with:

```bash
make check-ci
```

If these pass, you're good to go! Open a pull request with your changes.

## Documentation

This project uses [mkdocs](https://www.mkdocs.org/) for documentation. You can see the docs locally with:

```bash
make docs-serve
```

If you make changes to code which requires updating documentation, it would be greatly appreciated if you could update the docs as well.
