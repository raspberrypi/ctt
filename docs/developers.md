# Developers

## Linting and formatting

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for lint errors
python3 -m ruff check .

# Auto-fix lint errors
python3 -m ruff check --fix .

# Format code
python3 -m ruff format .

# Check formatting without modifying files
python3 -m ruff format --check .
```

These same checks run in CI. To catch problems before committing, enable the
[pre-commit](https://pre-commit.com/) hooks (they run ruff lint + format on each
commit, mirroring CI):

```bash
pip install -e ".[dev]"
pre-commit install            # one-time, per clone
pre-commit run --all-files    # optional: check the whole tree now
```

## Running tests

Install with the test extra and run pytest:

```bash
pip install -e ".[test]"
pytest -v
```

## Building a wheel package

```bash
pip install build
python3 -m build
```

This produces a `.whl` file in the `dist/` directory.
