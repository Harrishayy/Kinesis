# Contributing

Thanks for your interest in Kinesis.

## Development setup

```bash
git clone https://github.com/Harrishayy/Kinesis.git
cd Kinesis
make setup
```

This creates a Python 3.11 virtualenv, installs the package in editable mode with dev extras, pulls submodules, and installs pre-commit hooks.

## Running tests and lint

```bash
make test     # pytest
make lint     # ruff check
```

Both must pass before opening a PR.

## Code style

- `ruff` is the source of truth for formatting and lint (config in `pyproject.toml`).
- Type hints on public functions.
- Tests for new env / trajectory logic.

## Reporting issues

Open an issue at https://github.com/Harrishayy/Kinesis/issues with a minimal reproduction and the relevant config / commit SHA.
